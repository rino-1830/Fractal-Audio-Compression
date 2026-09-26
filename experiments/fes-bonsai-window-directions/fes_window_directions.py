import argparse, gc, itertools, json, math, os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

GROUP=128
FLIP_FRAC=0.00025

def windows_from_text(tok,text,seq_len,n,offset=0,step_extra=19):
    ids=tok(text,return_tensors="pt",add_special_tokens=False)["input_ids"][0]
    need=offset+n*(seq_len+step_extra)+seq_len
    if ids.numel()<need: ids=ids.repeat(math.ceil(need/ids.numel()))
    out=[]; pos=offset
    for _ in range(n):
        out.append(ids[pos:pos+seq_len].clone()); pos+=seq_len+step_extra
    return out

def load_data(tok,seq_len=24,n_cal=4,n_wiki=8,n_pile=8):
    from datasets import load_dataset
    val=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="validation")
    test=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
    pile=load_dataset("NeelNanda/pile-10k",split="train")
    vtext="\n\n".join(x["text"] for x in val if x["text"].strip())
    ttext="\n\n".join(x["text"] for x in test if x["text"].strip())
    ptext="\n\n".join(x["text"] for x in pile[:96]["text"] if x.strip())
    return (
      windows_from_text(tok,vtext,seq_len,n_cal,offset=0),
      windows_from_text(tok,ttext,seq_len,n_wiki,offset=713),
      windows_from_text(tok,ptext,seq_len,n_pile,offset=271),
    )

@torch.inference_mode()
def cache_logits(model,ws,vocab):
    return [model(input_ids=x.unsqueeze(0)).logits[0,:-1,:vocab].float().cpu().half() for x in ws]

@torch.inference_mode()
def metrics(model,ws,ref,vocab):
    kl=nll=rnll=0.; tokens=0
    for ids,rh in zip(ws,ref):
        r=rh.float(); q=model(input_ids=ids.unsqueeze(0)).logits[0,:-1,:vocab].float().cpu()
        rl=F.log_softmax(r,dim=-1); ql=F.log_softmax(q,dim=-1)
        kl+=(rl.exp()*(rl-ql)).sum().item()
        nll+=F.nll_loss(ql,ids[1:].cpu(),reduction="sum").item()
        rnll+=F.nll_loss(rl,ids[1:].cpu(),reduction="sum").item()
        tokens+=ids.numel()-1
    return {"kl_to_qwen":kl/tokens,"nll":nll/tokens,
            "ppl":math.exp(min(nll/tokens,20)),"qwen_ppl":math.exp(min(rnll/tokens,20))}

def target_modules(model,n):
    return [model.model.layers[i].mlp.up_proj for i in range(n)]

def single_window_grads(model,mods,cal,ref,vocab):
    for p in model.parameters(): p.requires_grad_(False)
    for m in mods: m.weight.requires_grad_(True)
    per_window=[]
    for wi,(ids,rh) in enumerate(zip(cal,ref)):
        model.zero_grad(set_to_none=True)
        q=model(input_ids=ids.unsqueeze(0)).logits[0,:-1,:vocab].float()
        r=rh.float().to(q.device)
        rl=F.log_softmax(r,dim=-1); ql=F.log_softmax(q,dim=-1)
        loss=(rl.exp()*(rl-ql)).sum(-1).mean()
        loss.backward()
        per_window.append([m.weight.grad.detach().float().cpu().clone() for m in mods])
        print("gradient_window",wi,"loss",float(loss.item()),flush=True)
    for m in mods: m.weight.grad=None; m.weight.requires_grad_(False)
    return per_window

def flip_candidate(wb,grad,frac):
    wb=wb.float().cpu(); grad=grad.float().cpu()
    rows,cols=wb.shape; assert cols%GROUP==0
    bg=wb.reshape(rows,cols//GROUP,GROUP); gg=grad.reshape_as(bg)
    sc=bg.abs().amax(-1,keepdim=True).expand_as(bg); cur=bg
    inf=torch.tensor(float("inf"))
    d0=torch.where(cur==0,inf,gg*(0-cur))
    dp=torch.where(cur>0,inf,gg*(sc-cur))
    dn=torch.where(cur<0,inf,gg*(-sc-cur))
    best_delta,best_alt=torch.stack((d0,dp,dn),0).min(0)
    fd=best_delta.reshape(-1); fa=best_alt.reshape(-1); fs=sc.reshape(-1)
    improving=torch.nonzero(fd<0,as_tuple=False).reshape(-1)
    order=improving[torch.argsort(fd[improving])] if improving.numel() else improving
    k=min(max(1,int(round(frac*wb.numel()))),order.numel())
    q=wb.half().reshape(-1).clone()
    if k:
        idx=order[:k]; alt=fa[idx]; scale=fs[idx].half()
        q[idx]=torch.where(alt==0,torch.zeros_like(scale),torch.where(alt==1,scale,-scale))
    return q.reshape_as(wb).contiguous(),{
      "flips":int(k),
      "estimated_delta":float(fd[order[:k]].sum().item()) if k else 0.0,
    }

def build_candidates(wb,per_window_layer_grads):
    names=["base"]; sets=[wb.half().contiguous()]; stats=[{"flips":0,"estimated_delta":0.0}]
    for i,g in enumerate(per_window_layer_grads):
        q,st=flip_candidate(wb,g,FLIP_FRAC)
        names.append(f"window{i}"); sets.append(q); stats.append(st)
    gavg=torch.stack(per_window_layer_grads).mean(0)
    q,st=flip_candidate(wb,gavg,FLIP_FRAC)
    names.append("aggregate"); sets.append(q); stats.append(st)
    # Add two contrastive directions. They deliberately emphasize disagreement
    # between calibration windows, providing output-error directions that a
    # global selector can combine across layers.
    if len(per_window_layer_grads)>=4:
        contrasts=[
          per_window_layer_grads[0]+per_window_layer_grads[1]-per_window_layer_grads[2]-per_window_layer_grads[3],
          per_window_layer_grads[0]-per_window_layer_grads[1]+per_window_layer_grads[2]-per_window_layer_grads[3],
        ]
        for ci,g in enumerate(contrasts):
            q,st=flip_candidate(wb,g,FLIP_FRAC)
            names.append(f"contrast{ci}"); sets.append(q); stats.append(st)
    return names,sets,stats

def apply(mods,sets,choices):
    for i,m in enumerate(mods): m.weight.data.copy_(sets[i][choices[i]].to(m.weight.dtype))

def eval_choice(model,mods,sets,choices,cal,wiki,pile,rc,rw,rp,vocab):
    apply(mods,sets,choices)
    return {"choices":list(choices),
            "cal":metrics(model,cal,rc,vocab),
            "wiki":metrics(model,wiki,rw,vocab),
            "pile":metrics(model,pile,rp,vocab)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--layers",type=int,default=3)
    ap.add_argument("--out",default="results-window-directions")
    args=ap.parse_args()
    torch.manual_seed(0); torch.set_num_threads(min(os.cpu_count() or 1,8))
    refn="Qwen/Qwen3-1.7B"; bonn="prism-ml/Ternary-Bonsai-1.7B-unpacked"
    tok=AutoTokenizer.from_pretrained(bonn); vocab=len(tok.get_vocab())
    cal,wiki,pile=load_data(tok)

    ref=AutoModelForCausalLM.from_pretrained(refn,dtype=torch.float32,low_cpu_mem_usage=True); ref.eval()
    rc=cache_logits(ref,cal,vocab); rw=cache_logits(ref,wiki,vocab); rp=cache_logits(ref,pile,vocab)
    del ref; gc.collect()

    model=AutoModelForCausalLM.from_pretrained(bonn,dtype=torch.float32,low_cpu_mem_usage=True); model.eval()
    mods=target_modules(model,args.layers)
    originals=[m.weight.detach().cpu().clone() for m in mods]
    pg=single_window_grads(model,mods,cal,rc,vocab)

    sets=[]; stats=[]; names=None
    for li,wb in enumerate(originals):
        layer_grads=[pg[wi][li] for wi in range(len(pg))]
        nm,cs,st=build_candidates(wb,layer_grads)
        if names is None: names=nm
        assert names==nm
        sets.append(cs); stats.append(st)
        print("layer",li,"candidates",list(zip(names,st)),flush=True)
    del pg; gc.collect()

    base=[0]*args.layers
    independent=[]
    for i in range(args.layers):
        vals=[]
        for j,name in enumerate(names):
            trial=list(base); trial[i]=j; apply(mods,sets,trial)
            v=metrics(model,cal,rc,vocab)["kl_to_qwen"]; vals.append(v)
            print("independent",i,name,v,flush=True)
        independent.append(min(range(len(vals)),key=lambda j:vals[j]))

    grid=[]
    for ch in itertools.product(range(len(names)),repeat=args.layers):
        apply(mods,sets,ch)
        grid.append((metrics(model,cal,rc,vocab)["kl_to_qwen"],ch))
    grid.sort(key=lambda x:x[0]); glob=list(grid[0][1])

    methods={
      "original":eval_choice(model,mods,sets,base,cal,wiki,pile,rc,rw,rp,vocab),
      "independent":eval_choice(model,mods,sets,independent,cal,wiki,pile,rc,rw,rp,vocab),
      "global":eval_choice(model,mods,sets,glob,cal,wiki,pile,rc,rw,rp,vocab),
    }
    g=methods["global"]; ind=methods["independent"]; b=methods["original"]
    summary={
      "candidate_names":names,"flip_fraction":FLIP_FRAC,
      "global_differs_from_independent":glob!=independent,
      "wiki_kl_global_vs_independent":g["wiki"]["kl_to_qwen"]/ind["wiki"]["kl_to_qwen"],
      "wiki_nll_global_vs_independent":g["wiki"]["nll"]/ind["wiki"]["nll"],
      "pile_kl_global_vs_independent":g["pile"]["kl_to_qwen"]/ind["pile"]["kl_to_qwen"],
      "pile_nll_global_vs_independent":g["pile"]["nll"]/ind["pile"]["nll"],
      "wiki_nll_global_vs_original":g["wiki"]["nll"]/b["wiki"]["nll"],
      "pile_nll_global_vs_original":g["pile"]["nll"]/b["pile"]["nll"],
    }
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps({"summary":summary,"stats":stats,"methods":methods,
      "top_grid":[{"cal_kl":v,"choices":list(ch),"names":[names[j] for j in ch]} for v,ch in grid[:20]]},indent=2))
    print("SUMMARY",json.dumps(summary,sort_keys=True),flush=True)

if __name__=="__main__": main()
