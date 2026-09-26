import argparse, gc, json, math, os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

GROUP=128
FRACTIONS=(0.0,0.0001,0.00025,0.0005,0.001)

def windows_from_text(tok,text,seq_len,n,offset=0,extra=23):
    ids=tok(text,return_tensors="pt",add_special_tokens=False)["input_ids"][0]
    need=offset+n*(seq_len+extra)+seq_len
    if ids.numel()<need: ids=ids.repeat(math.ceil(need/ids.numel()))
    out=[]; p=offset
    for _ in range(n):
        out.append(ids[p:p+seq_len].clone()); p+=seq_len+extra
    return out

def load_data(tok,seq_len=32,n_cal=4,n_wiki=8,n_pile=8):
    from datasets import load_dataset
    va=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="validation")
    te=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
    pi=load_dataset("NeelNanda/pile-10k",split="train")
    vat="\n\n".join(x["text"] for x in va if x["text"].strip())
    tet="\n\n".join(x["text"] for x in te if x["text"].strip())
    pit="\n\n".join(x for x in pi[:96]["text"] if x.strip())
    return (
      windows_from_text(tok,vat,seq_len,n_cal,0),
      windows_from_text(tok,tet,seq_len,n_wiki,911),
      windows_from_text(tok,pit,seq_len,n_pile,307),
    )

@torch.inference_mode()
def logits(model,ws,vocab):
    return [model(input_ids=x.unsqueeze(0)).logits[0,:-1,:vocab].float().cpu() for x in ws]

def half_list(xs):
    return [x.half() for x in xs]

@torch.inference_mode()
def metrics_from_logits(qs,refs,ws):
    kl=nll=rnll=0.; tokens=0
    for q,rh,ids in zip(qs,refs,ws):
        r=rh.float()
        rl=F.log_softmax(r,dim=-1); ql=F.log_softmax(q.float(),dim=-1)
        kl+=(rl.exp()*(rl-ql)).sum().item()
        nll+=F.nll_loss(ql,ids[1:].cpu(),reduction="sum").item()
        rnll+=F.nll_loss(rl,ids[1:].cpu(),reduction="sum").item()
        tokens+=ids.numel()-1
    return {"kl_to_qwen":kl/tokens,"nll":nll/tokens,
            "ppl":math.exp(min(nll/tokens,20)),"qwen_ppl":math.exp(min(rnll/tokens,20))}

@torch.inference_mode()
def metrics(model,ws,refs,vocab):
    return metrics_from_logits(logits(model,ws,vocab),refs,ws)

def target_modules(model,n):
    return [model.model.layers[i].mlp.up_proj for i in range(n)]

def aggregate_gradients(model,mods,cal,ref,vocab):
    for p in model.parameters(): p.requires_grad_(False)
    for m in mods: m.weight.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    for ids,rh in zip(cal,ref):
        q=model(input_ids=ids.unsqueeze(0)).logits[0,:-1,:vocab].float()
        r=rh.float().to(q.device)
        rl=F.log_softmax(r,dim=-1); ql=F.log_softmax(q,dim=-1)
        ((rl.exp()*(rl-ql)).sum(-1).mean()/len(cal)).backward()
    gs=[m.weight.grad.detach().float().cpu().clone() for m in mods]
    for m in mods: m.weight.grad=None; m.weight.requires_grad_(False)
    return gs

def candidates(wb,g):
    wb=wb.float().cpu(); g=g.float().cpu()
    rows,cols=wb.shape; assert cols%GROUP==0
    bg=wb.reshape(rows,cols//GROUP,GROUP); gg=g.reshape_as(bg)
    sc=bg.abs().amax(-1,keepdim=True).expand_as(bg); cur=bg; inf=torch.tensor(float("inf"))
    d0=torch.where(cur==0,inf,gg*(0-cur))
    dp=torch.where(cur>0,inf,gg*(sc-cur))
    dn=torch.where(cur<0,inf,gg*(-sc-cur))
    bd,ba=torch.stack((d0,dp,dn),0).min(0)
    fd=bd.reshape(-1); fa=ba.reshape(-1); fs=sc.reshape(-1)
    good=torch.nonzero(fd<0,as_tuple=False).reshape(-1)
    order=good[torch.argsort(fd[good])] if good.numel() else good
    sets=[]; stats=[]; n=wb.numel()
    for frac in FRACTIONS:
        k=min(int(round(frac*n)),order.numel())
        q=wb.half().reshape(-1).clone()
        if k:
            idx=order[:k]; alt=fa[idx]; scale=fs[idx].half()
            q[idx]=torch.where(alt==0,torch.zeros_like(scale),torch.where(alt==1,scale,-scale))
        sets.append(q.reshape_as(wb).contiguous())
        stats.append({"fraction":frac,"flips":int(k),
                      "estimated_first_order_delta":float(fd[order[:k]].sum().item()) if k else 0.0})
    return sets,stats

def restore(mods,orig):
    for m,w in zip(mods,orig): m.weight.data.copy_(w.to(m.weight.dtype))

def apply(mods,sets,ch):
    for i,m in enumerate(mods): m.weight.data.copy_(sets[i][ch[i]].to(m.weight.dtype))

def topk_probe(ref_logits,base_logits,k):
    idxs=[]; refvals=[]; basevals=[]; parts=[]
    for r,b in zip(ref_logits,base_logits):
        idx=torch.topk(r,k=k,dim=-1).indices
        rv=torch.gather(r,1,idx); bv=torch.gather(b,1,idx)
        rv=rv-rv.mean(-1,keepdim=True); bv=bv-bv.mean(-1,keepdim=True)
        idxs.append(idx); refvals.append(rv); basevals.append(bv)
        parts.append((bv-rv).reshape(-1))
    return idxs,basevals,torch.cat(parts)

def candidate_delta(q_logits,idxs,basevals):
    parts=[]
    for q,idx,bv in zip(q_logits,idxs,basevals):
        qv=torch.gather(q,1,idx); qv=qv-qv.mean(-1,keepdim=True)
        parts.append((qv-bv).reshape(-1))
    return torch.cat(parts)

def beam_search(base_error,delta_sets,width):
    beam=[(float(torch.dot(base_error,base_error)),base_error.clone(),tuple())]
    for li,layer in enumerate(delta_sets):
        nxt=[]
        for _,acc,ch in beam:
            for j,d in enumerate(layer):
                v=acc+d
                nxt.append((float(torch.dot(v,v)),v,ch+(j,)))
        nxt.sort(key=lambda x:x[0]); beam=nxt[:width]
        print("beam_layer",li,"best",beam[0][0],"choices",beam[0][2],flush=True)
    return beam

def evaluate(model,mods,sets,ch,cal,wiki,pile,rc,rw,rp,vocab):
    apply(mods,sets,ch)
    return {"choices":list(ch),"fractions":[FRACTIONS[j] for j in ch],
            "cal":metrics(model,cal,rc,vocab),
            "wiki":metrics(model,wiki,rw,vocab),
            "pile":metrics(model,pile,rp,vocab)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--layers",type=int,default=6)
    ap.add_argument("--beam",type=int,default=128)
    ap.add_argument("--rerank",type=int,default=12)
    ap.add_argument("--topk",type=int,default=32)
    ap.add_argument("--out",default="results-sketch")
    args=ap.parse_args()
    torch.manual_seed(0); torch.set_num_threads(min(os.cpu_count() or 1,8))
    refn="Qwen/Qwen3-1.7B"; bonn="prism-ml/Ternary-Bonsai-1.7B-unpacked"
    tok=AutoTokenizer.from_pretrained(bonn); vocab=len(tok.get_vocab())
    cal,wiki,pile=load_data(tok)

    ref=AutoModelForCausalLM.from_pretrained(refn,dtype=torch.float32,low_cpu_mem_usage=True); ref.eval()
    rc_full=logits(ref,cal,vocab); rw=half_list(logits(ref,wiki,vocab)); rp=half_list(logits(ref,pile,vocab))
    rc=half_list(rc_full)
    del ref; gc.collect()

    model=AutoModelForCausalLM.from_pretrained(bonn,dtype=torch.float32,low_cpu_mem_usage=True); model.eval()
    mods=target_modules(model,args.layers); orig=[m.weight.detach().cpu().clone() for m in mods]
    base_cal=logits(model,cal,vocab)
    base_metrics=metrics_from_logits(base_cal,rc,cal)
    idxs,basevals,base_error=topk_probe(rc_full,base_cal,args.topk)
    print("baseline_probe_energy",float(torch.dot(base_error,base_error)),flush=True)
    del rc_full; gc.collect()

    gs=aggregate_gradients(model,mods,cal,rc,vocab)
    sets=[]; stats=[]
    for wb,g in zip(orig,gs):
        cs,st=candidates(wb,g); sets.append(cs); stats.append(st)
    del gs; gc.collect()

    delta_sets=[]; independent=[]
    for i in range(args.layers):
        layer=[]; exact=[]
        for j in range(len(FRACTIONS)):
            restore(mods,orig)
            mods[i].weight.data.copy_(sets[i][j].to(mods[i].weight.dtype))
            q=logits(model,cal,vocab)
            d=candidate_delta(q,idxs,basevals); layer.append(d)
            exact.append(metrics_from_logits(q,rc,cal)["kl_to_qwen"])
            print("candidate",i,j,FRACTIONS[j],"cal_kl",exact[-1],"delta_energy",float(torch.dot(d,d)),flush=True)
        delta_sets.append(layer)
        independent.append(min(range(len(exact)),key=lambda j:exact[j]))
    restore(mods,orig)

    beam=beam_search(base_error,delta_sets,args.beam)
    rr=[]
    for sketch_score,_,ch in beam[:args.rerank]:
        apply(mods,sets,ch)
        k=metrics(model,cal,rc,vocab)["kl_to_qwen"]
        rr.append((k,sketch_score,ch))
        print("rerank",k,sketch_score,ch,flush=True)
    rr.sort(key=lambda x:x[0]); fes=list(rr[0][2])

    methods={
      "original":evaluate(model,mods,sets,[0]*args.layers,cal,wiki,pile,rc,rw,rp,vocab),
      "independent":evaluate(model,mods,sets,independent,cal,wiki,pile,rc,rw,rp,vocab),
      "fes_sketch":evaluate(model,mods,sets,fes,cal,wiki,pile,rc,rw,rp,vocab),
    }
    f=methods["fes_sketch"]; ind=methods["independent"]; b=methods["original"]
    summary={
      "layers":args.layers,"fractions":FRACTIONS,"topk":args.topk,"beam":args.beam,
      "fes_differs_from_independent":fes!=independent,
      "cal_kl_fes_vs_independent":f["cal"]["kl_to_qwen"]/ind["cal"]["kl_to_qwen"],
      "wiki_kl_fes_vs_independent":f["wiki"]["kl_to_qwen"]/ind["wiki"]["kl_to_qwen"],
      "wiki_nll_fes_vs_independent":f["wiki"]["nll"]/ind["wiki"]["nll"],
      "pile_kl_fes_vs_independent":f["pile"]["kl_to_qwen"]/ind["pile"]["kl_to_qwen"],
      "pile_nll_fes_vs_independent":f["pile"]["nll"]/ind["pile"]["nll"],
      "wiki_nll_fes_vs_original":f["wiki"]["nll"]/b["wiki"]["nll"],
      "pile_nll_fes_vs_original":f["pile"]["nll"]/b["pile"]["nll"],
      "original_cal_kl":base_metrics["kl_to_qwen"],
    }
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps({"summary":summary,"candidate_stats":stats,"methods":methods,
      "rerank":[{"cal_kl":k,"sketch_score":s,"choices":list(ch)} for k,s,ch in rr]},indent=2))
    print("SUMMARY",json.dumps(summary,sort_keys=True),flush=True)

if __name__=="__main__": main()
