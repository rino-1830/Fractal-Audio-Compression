import argparse, gc, json, math, os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

GROUP=128
FRACTIONS=(0.0,0.0001,0.00025,0.0005,0.001)

def text_windows(tok,text,seq_len,n,offset=0,stride_extra=17):
    ids=tok(text,return_tensors="pt",add_special_tokens=False)["input_ids"][0]
    need=offset+n*(seq_len+stride_extra)+seq_len
    if ids.numel()<need: ids=ids.repeat(math.ceil(need/ids.numel()))
    out=[]; pos=offset
    for _ in range(n):
        out.append(ids[pos:pos+seq_len].clone()); pos+=seq_len+stride_extra
    return out

def load_data(tok,seq_len,cal_n,wiki_n,pile_n):
    from datasets import load_dataset
    val=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="validation")
    test=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
    pile=load_dataset("NeelNanda/pile-10k",split="train")
    vtext="\n\n".join(x["text"] for x in val if x["text"].strip())
    ttext="\n\n".join(x["text"] for x in test if x["text"].strip())
    ptext="\n\n".join(x for x in pile[:64]["text"] if x.strip())
    return (
      text_windows(tok,vtext,seq_len,cal_n,offset=0),
      text_windows(tok,ttext,seq_len,wiki_n,offset=997),
      text_windows(tok,ptext,seq_len,pile_n,offset=313),
    )

@torch.inference_mode()
def cache_logits(model,ws,vocab):
    return [model(input_ids=x.unsqueeze(0)).logits[0,:-1,:vocab].float().cpu().half() for x in ws]

@torch.inference_mode()
def metrics(model,ws,ref,vocab):
    kl=nll=rnll=0.0; agree=tokens=0
    for ids,rh in zip(ws,ref):
        r=rh.float()
        q=model(input_ids=ids.unsqueeze(0)).logits[0,:-1,:vocab].float().cpu()
        target=ids[1:].cpu()
        rl=F.log_softmax(r,dim=-1); ql=F.log_softmax(q,dim=-1)
        kl+=(rl.exp()*(rl-ql)).sum().item()
        nll+=F.nll_loss(ql,target,reduction="sum").item()
        rnll+=F.nll_loss(rl,target,reduction="sum").item()
        agree+=(r.argmax(-1)==q.argmax(-1)).sum().item()
        tokens+=target.numel()
    return {"kl_to_qwen":kl/tokens,"nll":nll/tokens,
            "ppl":math.exp(min(nll/tokens,20)),
            "qwen_ppl":math.exp(min(rnll/tokens,20)),
            "top1_agreement":agree/tokens}

def target_modules(model,n):
    return [(f"layer.{i}.mlp.up_proj",model.model.layers[i].mlp.up_proj) for i in range(n)]

def calibration_gradients(model,mods,cal,ref_cal,vocab):
    for p in model.parameters(): p.requires_grad_(False)
    for _,m in mods: m.weight.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    for ids,rh in zip(cal,ref_cal):
        q=model(input_ids=ids.unsqueeze(0)).logits[0,:-1,:vocab].float()
        r=rh.float().to(q.device)
        rl=F.log_softmax(r,dim=-1); ql=F.log_softmax(q,dim=-1)
        ((rl.exp()*(rl-ql)).sum(dim=-1).mean()/len(cal)).backward()
    gs=[m.weight.grad.detach().float().cpu().clone() for _,m in mods]
    for _,m in mods: m.weight.grad=None; m.weight.requires_grad_(False)
    return gs

def candidates(wb,wr,grad):
    wb=wb.float().cpu(); wr=wr.float().cpu(); grad=grad.float().cpu()
    rows,cols=wb.shape; assert cols%GROUP==0
    bg=wb.reshape(rows,cols//GROUP,GROUP); gg=grad.reshape_as(bg)
    scale=bg.abs().amax(-1,keepdim=True).expand_as(bg); cur=bg
    inf=torch.tensor(float("inf"))
    d0=torch.where(cur==0,inf,gg*(0-cur))
    dp=torch.where(cur>0,inf,gg*(scale-cur))
    dn=torch.where(cur<0,inf,gg*(-scale-cur))
    best_delta,best_alt=torch.stack((d0,dp,dn),0).min(0)
    fd=best_delta.reshape(-1); fa=best_alt.reshape(-1); fs=scale.reshape(-1)
    improving=torch.nonzero(fd<0,as_tuple=False).reshape(-1)
    order=improving[torch.argsort(fd[improving])] if improving.numel() else improving
    sets=[]; mses=[]; stats=[]; n=wb.numel()
    for frac in FRACTIONS:
        k=min(int(round(frac*n)),order.numel())
        q=wb.half().reshape(-1).clone()
        if k:
            idx=order[:k]; alt=fa[idx]; sc=fs[idx].half()
            q[idx]=torch.where(alt==0,torch.zeros_like(sc),torch.where(alt==1,sc,-sc))
        q=q.reshape_as(wb).contiguous()
        sets.append(q); mses.append(float(F.mse_loss(q.float(),wr).item()))
        stats.append({"fraction":frac,"flips":int(k)})
    return sets,mses,stats

def apply(mods,sets,choices):
    for i,(_,m) in enumerate(mods): m.weight.data.copy_(sets[i][choices[i]].to(m.weight.dtype))

def score(model,cal,ref_cal,vocab):
    return metrics(model,cal,ref_cal,vocab)["kl_to_qwen"]

def evaluate(model,mods,sets,choices,cal,wiki,pile,rc,rw,rp,vocab,mses):
    apply(mods,sets,choices)
    return {"choices":list(choices),"fractions":[FRACTIONS[j] for j in choices],
            "weight_mse_to_qwen":sum(mses[i][choices[i]] for i in range(len(choices)))/len(choices),
            "cal":metrics(model,cal,rc,vocab),
            "wikitext_test":metrics(model,wiki,rw,vocab),
            "pile10k":metrics(model,pile,rp,vocab)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--layers",type=int,default=6)
    ap.add_argument("--passes",type=int,default=2)
    ap.add_argument("--out",default="results-cross")
    args=ap.parse_args()
    torch.manual_seed(0); torch.set_num_threads(min(os.cpu_count() or 1,8))
    refn="Qwen/Qwen3-1.7B"; bonn="prism-ml/Ternary-Bonsai-1.7B-unpacked"
    tok=AutoTokenizer.from_pretrained(bonn); vocab=len(tok.get_vocab())
    cal,wiki,pile=load_data(tok,32,4,8,8)

    ref=AutoModelForCausalLM.from_pretrained(refn,dtype=torch.float32,low_cpu_mem_usage=True); ref.eval()
    rmods=target_modules(ref,args.layers)
    rweights=[m.weight.detach().cpu().half().clone() for _,m in rmods]
    rc=cache_logits(ref,cal,vocab); rw=cache_logits(ref,wiki,vocab); rp=cache_logits(ref,pile,vocab)
    del ref,rmods; gc.collect()

    model=AutoModelForCausalLM.from_pretrained(bonn,dtype=torch.float32,low_cpu_mem_usage=True); model.eval()
    mods=target_modules(model,args.layers); bweights=[m.weight.detach().cpu().clone() for _,m in mods]
    grads=calibration_gradients(model,mods,cal,rc,vocab)
    sets=[]; mses=[]; stats=[]
    for wb,wr,g in zip(bweights,rweights,grads):
        a,b,c=candidates(wb,wr,g); sets.append(a); mses.append(b); stats.append(c)
    del grads; gc.collect()

    base=[0]*args.layers
    local=[min(range(len(FRACTIONS)),key=lambda j:mses[i][j]) for i in range(args.layers)]

    independent=[]
    for i in range(args.layers):
        vals=[]
        for j in range(len(FRACTIONS)):
            trial=list(base); trial[i]=j; apply(mods,sets,trial)
            vals.append(score(model,cal,rc,vocab))
        independent.append(min(range(len(vals)),key=lambda j:vals[j]))
        print("independent",i,independent[-1],FRACTIONS[independent[-1]],min(vals),flush=True)

    coord=list(independent); history=[]
    for p in range(args.passes):
        changed=False
        for i in range(args.layers):
            vals=[]
            for j in range(len(FRACTIONS)):
                trial=list(coord); trial[i]=j; apply(mods,sets,trial)
                vals.append(score(model,cal,rc,vocab))
            best=min(range(len(vals)),key=lambda j:vals[j])
            if best!=coord[i]: changed=True
            coord[i]=best
            history.append({"pass":p,"layer":i,"choice":best,"fraction":FRACTIONS[best],"cal_kl":vals[best]})
            print("coordinate",p,i,best,FRACTIONS[best],vals[best],flush=True)
        if not changed: break

    methods={
      "original":evaluate(model,mods,sets,base,cal,wiki,pile,rc,rw,rp,vocab,mses),
      "local":evaluate(model,mods,sets,local,cal,wiki,pile,rc,rw,rp,vocab,mses),
      "independent":evaluate(model,mods,sets,independent,cal,wiki,pile,rc,rw,rp,vocab,mses),
      "coordinate":evaluate(model,mods,sets,coord,cal,wiki,pile,rc,rw,rp,vocab,mses),
    }
    def ratio(metric,dataset,a="coordinate",b="independent"):
        return methods[a][dataset][metric]/methods[b][dataset][metric]
    summary={
      "layers":args.layers,"fractions":FRACTIONS,
      "coordinate_differs_from_independent":coord!=independent,
      "wiki_kl_coord_vs_independent":ratio("kl_to_qwen","wikitext_test"),
      "wiki_nll_coord_vs_independent":ratio("nll","wikitext_test"),
      "pile_kl_coord_vs_independent":ratio("kl_to_qwen","pile10k"),
      "pile_nll_coord_vs_independent":ratio("nll","pile10k"),
      "wiki_nll_coord_vs_original":ratio("nll","wikitext_test","coordinate","original"),
      "pile_nll_coord_vs_original":ratio("nll","pile10k","coordinate","original"),
      "wiki_kl_coord_vs_original":ratio("kl_to_qwen","wikitext_test","coordinate","original"),
      "pile_kl_coord_vs_original":ratio("kl_to_qwen","pile10k","coordinate","original"),
    }
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps({"summary":summary,"stats":stats,"history":history,"methods":methods},indent=2))
    print("SUMMARY",json.dumps(summary,sort_keys=True),flush=True)

if __name__=="__main__": main()
