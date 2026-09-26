import argparse, gc, json, math, os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

GROUP=128
FRACTIONS=(0.0,0.0001,0.00025,0.0005,0.001)

def windows_from_text(tok,text,seq_len,n,offset=0,extra=29):
    ids=tok(text,return_tensors="pt",add_special_tokens=False)["input_ids"][0]
    need=offset+n*(seq_len+extra)+seq_len
    if ids.numel()<need: ids=ids.repeat(math.ceil(need/ids.numel()))
    out=[]; p=offset
    for _ in range(n):
        out.append(ids[p:p+seq_len].clone()); p+=seq_len+extra
    return out

def load_data(tok,seq_len=32):
    from datasets import load_dataset
    va=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="validation")
    te=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
    pi=load_dataset("NeelNanda/pile-10k",split="train")
    vat="\n\n".join(x["text"] for x in va if x["text"].strip())
    tet="\n\n".join(x["text"] for x in te if x["text"].strip())
    pit="\n\n".join(x for x in pi[:128]["text"] if x.strip())
    # Search and validation are disjoint windows from WikiText validation.
    return (
      windows_from_text(tok,vat,seq_len,4,offset=0),
      windows_from_text(tok,vat,seq_len,4,offset=4096),
      windows_from_text(tok,tet,seq_len,8,offset=733),
      windows_from_text(tok,pit,seq_len,8,offset=337),
    )

@torch.inference_mode()
def logits(model,ws,vocab):
    return [model(input_ids=x.unsqueeze(0)).logits[0,:-1,:vocab].float().cpu() for x in ws]

@torch.inference_mode()
def metrics_from_logits(qs,refs,ws):
    kl=nll=rnll=0.; tokens=0
    for q,rh,ids in zip(qs,refs,ws):
        r=rh.float(); q=q.float()
        rl=F.log_softmax(r,dim=-1); ql=F.log_softmax(q,dim=-1)
        kl+=(rl.exp()*(rl-ql)).sum().item()
        nll+=F.nll_loss(ql,ids[1:].cpu(),reduction="sum").item()
        rnll+=F.nll_loss(rl,ids[1:].cpu(),reduction="sum").item()
        tokens+=ids.numel()-1
    return {"kl_to_qwen":kl/tokens,"nll":nll/tokens,
            "ppl":math.exp(min(nll/tokens,20)),"qwen_ppl":math.exp(min(rnll/tokens,20))}

@torch.inference_mode()
def metrics(model,ws,refs,vocab):
    return metrics_from_logits(logits(model,ws,vocab),refs,ws)

def half(xs): return [x.half() for x in xs]

def mods(model,n): return [model.model.layers[i].mlp.up_proj for i in range(n)]

def gradients(model,ms,search,ref,vocab):
    for p in model.parameters(): p.requires_grad_(False)
    for m in ms: m.weight.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    for ids,rh in zip(search,ref):
        q=model(input_ids=ids.unsqueeze(0)).logits[0,:-1,:vocab].float()
        r=rh.float().to(q.device)
        rl=F.log_softmax(r,dim=-1); ql=F.log_softmax(q,dim=-1)
        ((rl.exp()*(rl-ql)).sum(-1).mean()/len(search)).backward()
    gs=[m.weight.grad.detach().float().cpu().clone() for m in ms]
    for m in ms: m.weight.grad=None; m.weight.requires_grad_(False)
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
        stats.append({"fraction":frac,"flips":int(k)})
    return sets,stats

def restore(ms,orig):
    for m,w in zip(ms,orig): m.weight.data.copy_(w.to(m.weight.dtype))

def apply(ms,sets,ch):
    for i,m in enumerate(ms): m.weight.data.copy_(sets[i][ch[i]].to(m.weight.dtype))

def probe(refs,bases,k):
    idxs=[]; basevals=[]; err=[]
    for r,b in zip(refs,bases):
        idx=torch.topk(r,k=k,dim=-1).indices
        rv=torch.gather(r,1,idx); bv=torch.gather(b,1,idx)
        rv=rv-rv.mean(-1,keepdim=True); bv=bv-bv.mean(-1,keepdim=True)
        idxs.append(idx); basevals.append(bv); err.append((bv-rv).reshape(-1))
    return idxs,basevals,torch.cat(err)

def delta(qs,idxs,basevals):
    out=[]
    for q,idx,bv in zip(qs,idxs,basevals):
        qv=torch.gather(q,1,idx); qv=qv-qv.mean(-1,keepdim=True)
        out.append((qv-bv).reshape(-1))
    return torch.cat(out)

def beam(base_error,dsets,width):
    b=[(float(torch.dot(base_error,base_error)),base_error.clone(),tuple())]
    for li,layer in enumerate(dsets):
        nx=[]
        for _,acc,ch in b:
            for j,d in enumerate(layer):
                v=acc+d; nx.append((float(torch.dot(v,v)),v,ch+(j,)))
        nx.sort(key=lambda x:x[0]); b=nx[:width]
        print("beam",li,b[0][0],b[0][2],flush=True)
    return b

def evaluate(model,ms,sets,ch,search,valid,wiki,pile,rs,rv,rw,rp,vocab):
    apply(ms,sets,ch)
    return {"choices":list(ch),"fractions":[FRACTIONS[j] for j in ch],
      "search":metrics(model,search,rs,vocab),
      "validation":metrics(model,valid,rv,vocab),
      "wiki":metrics(model,wiki,rw,vocab),
      "pile":metrics(model,pile,rp,vocab)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--layers",type=int,default=6)
    ap.add_argument("--beam",type=int,default=512)
    ap.add_argument("--rerank",type=int,default=64)
    ap.add_argument("--topk",type=int,default=32)
    ap.add_argument("--out",default="results-valfes")
    args=ap.parse_args()
    torch.manual_seed(0); torch.set_num_threads(min(os.cpu_count() or 1,8))
    refn="Qwen/Qwen3-1.7B"; bonn="prism-ml/Ternary-Bonsai-1.7B-unpacked"
    tok=AutoTokenizer.from_pretrained(bonn); vocab=len(tok.get_vocab())
    search,valid,wiki,pile=load_data(tok)

    ref=AutoModelForCausalLM.from_pretrained(refn,dtype=torch.float32,low_cpu_mem_usage=True); ref.eval()
    rs_full=logits(ref,search,vocab)
    rs=half(rs_full); rv=half(logits(ref,valid,vocab)); rw=half(logits(ref,wiki,vocab)); rp=half(logits(ref,pile,vocab))
    del ref; gc.collect()

    model=AutoModelForCausalLM.from_pretrained(bonn,dtype=torch.float32,low_cpu_mem_usage=True); model.eval()
    ms=mods(model,args.layers); orig=[m.weight.detach().cpu().clone() for m in ms]
    base_search=logits(model,search,vocab)
    idxs,bvals,berr=probe(rs_full,base_search,args.topk)
    del rs_full; gc.collect()

    gs=gradients(model,ms,search,rs,vocab)
    sets=[]; stats=[]
    for w,g in zip(orig,gs):
        cs,st=candidates(w,g); sets.append(cs); stats.append(st)
    del gs; gc.collect()

    dsets=[]; independent=[]
    # Independent baseline is selected on validation, not on search.
    for i in range(args.layers):
        ld=[]; vals=[]
        for j in range(len(FRACTIONS)):
            restore(ms,orig); ms[i].weight.data.copy_(sets[i][j].to(ms[i].weight.dtype))
            qs=logits(model,search,vocab); ld.append(delta(qs,idxs,bvals))
            vals.append(metrics(model,valid,rv,vocab)["kl_to_qwen"])
            print("candidate",i,j,FRACTIONS[j],"valid_kl",vals[-1],flush=True)
        dsets.append(ld); independent.append(min(range(len(vals)),key=lambda j:vals[j]))
    restore(ms,orig)

    b=beam(berr,dsets,args.beam)
    rr=[]
    for sketch,_,ch in b[:args.rerank]:
        apply(ms,sets,ch)
        vk=metrics(model,valid,rv,vocab)["kl_to_qwen"]
        rr.append((vk,sketch,ch))
        print("validation_rerank",vk,sketch,ch,flush=True)
    rr.sort(key=lambda x:x[0]); fes=list(rr[0][2])

    methods={
      "original":evaluate(model,ms,sets,[0]*args.layers,search,valid,wiki,pile,rs,rv,rw,rp,vocab),
      "independent":evaluate(model,ms,sets,independent,search,valid,wiki,pile,rs,rv,rw,rp,vocab),
      "fes_validation":evaluate(model,ms,sets,fes,search,valid,wiki,pile,rs,rv,rw,rp,vocab),
    }
    f=methods["fes_validation"]; ind=methods["independent"]; o=methods["original"]
    summary={
      "layers":args.layers,"beam":args.beam,"rerank":args.rerank,"topk":args.topk,
      "fes_differs_from_independent":fes!=independent,
      "validation_kl_fes_vs_independent":f["validation"]["kl_to_qwen"]/ind["validation"]["kl_to_qwen"],
      "wiki_kl_fes_vs_independent":f["wiki"]["kl_to_qwen"]/ind["wiki"]["kl_to_qwen"],
      "wiki_nll_fes_vs_independent":f["wiki"]["nll"]/ind["wiki"]["nll"],
      "pile_kl_fes_vs_independent":f["pile"]["kl_to_qwen"]/ind["pile"]["kl_to_qwen"],
      "pile_nll_fes_vs_independent":f["pile"]["nll"]/ind["pile"]["nll"],
      "wiki_nll_fes_vs_original":f["wiki"]["nll"]/o["wiki"]["nll"],
      "pile_nll_fes_vs_original":f["pile"]["nll"]/o["pile"]["nll"],
    }
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps({"summary":summary,"candidate_stats":stats,"methods":methods,
      "rerank":[{"validation_kl":v,"sketch":s,"choices":list(ch)} for v,s,ch in rr]},indent=2))
    print("SUMMARY",json.dumps(summary,sort_keys=True),flush=True)

if __name__=="__main__": main()
