import gc,json,math,os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer

GROUP=128
FRACTIONS=(0.0,0.0001,0.00025,0.0005,0.001)
INDEPENDENT=(0,0,2,0,0,0)
FES=(1,0,2,0,0,0)
ORIGINAL=(0,0,0,0,0,0)

def windows(tok,text,seq,n,offset,extra=31):
    ids=tok(text,return_tensors="pt",add_special_tokens=False)["input_ids"][0]
    need=offset+n*(seq+extra)+seq
    if ids.numel()<need:ids=ids.repeat(math.ceil(need/ids.numel()))
    out=[];p=offset
    for _ in range(n):
        out.append(ids[p:p+seq].clone());p+=seq+extra
    return torch.stack(out)

def load_data(tok):
    from datasets import load_dataset
    va=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="validation")
    te=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
    pi=load_dataset("NeelNanda/pile-10k",split="train")
    vat="\n\n".join(x["text"] for x in va if x["text"].strip())
    tet="\n\n".join(x["text"] for x in te if x["text"].strip())
    pit="\n\n".join(x for x in pi[:256]["text"] if x.strip())
    return windows(tok,vat,32,4,0),windows(tok,tet,64,64,977),windows(tok,pit,64,64,421)

@torch.inference_mode()
def ref_logits(model,x,vocab,batch=4):
    return torch.cat([model(input_ids=x[i:i+batch]).logits[:,:-1,:vocab].float().cpu().half()
                      for i in range(0,len(x),batch)],0)

def mods(model):return [model.model.layers[i].mlp.up_proj for i in range(6)]

def gradient(model,ms,cal,ref,vocab):
    for p in model.parameters():p.requires_grad_(False)
    for m in ms:m.weight.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    for bi in range(len(cal)):
        ids=cal[bi:bi+1]
        q=model(input_ids=ids).logits[0,:-1,:vocab].float()
        r=ref[bi].float().to(q.device)
        rl=F.log_softmax(r,-1);ql=F.log_softmax(q,-1)
        ((rl.exp()*(rl-ql)).sum(-1).mean()/len(cal)).backward()
    gs=[m.weight.grad.detach().float().cpu().clone() for m in ms]
    for m in ms:m.weight.grad=None;m.weight.requires_grad_(False)
    return gs

def candidates(wb,g):
    wb=wb.float().cpu();g=g.float().cpu()
    rows,cols=wb.shape;bg=wb.reshape(rows,cols//GROUP,GROUP);gg=g.reshape_as(bg)
    sc=bg.abs().amax(-1,keepdim=True).expand_as(bg);cur=bg;inf=torch.tensor(float("inf"))
    d0=torch.where(cur==0,inf,gg*(0-cur));dp=torch.where(cur>0,inf,gg*(sc-cur));dn=torch.where(cur<0,inf,gg*(-sc-cur))
    bd,ba=torch.stack((d0,dp,dn),0).min(0)
    fd=bd.reshape(-1);fa=ba.reshape(-1);fs=sc.reshape(-1)
    good=torch.nonzero(fd<0,as_tuple=False).reshape(-1);order=good[torch.argsort(fd[good])]
    out=[];n=wb.numel()
    for frac in FRACTIONS:
        k=min(int(round(frac*n)),order.numel());q=wb.half().reshape(-1).clone()
        if k:
            idx=order[:k];alt=fa[idx];s=fs[idx].half()
            q[idx]=torch.where(alt==0,torch.zeros_like(s),torch.where(alt==1,s,-s))
        out.append(q.reshape_as(wb).contiguous())
    return out

def apply(ms,sets,ch):
    for i,m in enumerate(ms):m.weight.data.copy_(sets[i][ch[i]].to(m.weight.dtype))

@torch.inference_mode()
def evaluate(model,x,ref,vocab,batch=4):
    all_nll=[];all_kl=[]
    for i in range(0,len(x),batch):
        ids=x[i:i+batch]
        q=model(input_ids=ids).logits[:,:-1,:vocab].float().cpu()
        r=ref[i:i+batch].float()
        rl=F.log_softmax(r,-1);ql=F.log_softmax(q,-1)
        targets=ids[:,1:].cpu()
        tok_nll=F.nll_loss(ql.reshape(-1,vocab),targets.reshape(-1),reduction="none").reshape(len(ids),-1)
        tok_kl=(rl.exp()*(rl-ql)).sum(-1)
        all_nll.extend(tok_nll.mean(-1).tolist());all_kl.extend(tok_kl.mean(-1).tolist())
    t=torch.tensor(all_nll);k=torch.tensor(all_kl)
    return {"nll":t.mean().item(),"ppl":math.exp(min(t.mean().item(),20)),"kl_to_qwen":k.mean().item(),
            "window_nll":all_nll,"window_kl":all_kl}

def paired(a,b,key):
    d=torch.tensor(a[key])-torch.tensor(b[key]);mean=d.mean().item();se=d.std(unbiased=True).item()/math.sqrt(len(d))
    return {"mean":mean,"se":se,"ci95_low":mean-1.96*se,"ci95_high":mean+1.96*se,"n":len(d)}

def main():
    torch.manual_seed(0);torch.set_num_threads(min(os.cpu_count() or 1,8))
    refn="Qwen/Qwen3-1.7B";bonn="prism-ml/Ternary-Bonsai-1.7B-unpacked"
    tok=AutoTokenizer.from_pretrained(bonn);vocab=len(tok.get_vocab());cal,wiki,pile=load_data(tok)

    ref=AutoModelForCausalLM.from_pretrained(refn,dtype=torch.float32,low_cpu_mem_usage=True);ref.eval()
    rc=ref_logits(ref,cal,vocab);rw=ref_logits(ref,wiki,vocab);rp=ref_logits(ref,pile,vocab)
    del ref;gc.collect()

    model=AutoModelForCausalLM.from_pretrained(bonn,dtype=torch.float32,low_cpu_mem_usage=True);model.eval()
    ms=mods(model);orig=[m.weight.detach().cpu().clone() for m in ms]
    gs=gradient(model,ms,cal,rc,vocab);sets=[candidates(w,g) for w,g in zip(orig,gs)];del gs;gc.collect()

    res={}
    for name,ch in [("original",ORIGINAL),("independent",INDEPENDENT),("fes",FES)]:
        apply(ms,sets,ch)
        res[name]={"choices":list(ch),"wiki":evaluate(model,wiki,rw,vocab),"pile":evaluate(model,pile,rp,vocab)}
        print(name,res[name]["wiki"]["nll"],res[name]["pile"]["nll"],flush=True)
    paired_stats={
      "wiki_fes_minus_independent_nll":paired(res["fes"]["wiki"],res["independent"]["wiki"],"window_nll"),
      "pile_fes_minus_independent_nll":paired(res["fes"]["pile"],res["independent"]["pile"],"window_nll"),
      "wiki_fes_minus_original_nll":paired(res["fes"]["wiki"],res["original"]["wiki"],"window_nll"),
      "pile_fes_minus_original_nll":paired(res["fes"]["pile"],res["original"]["pile"],"window_nll"),
      "wiki_fes_minus_independent_kl":paired(res["fes"]["wiki"],res["independent"]["wiki"],"window_kl"),
      "pile_fes_minus_independent_kl":paired(res["fes"]["pile"],res["independent"]["pile"],"window_kl"),
    }
    summary={
      "wiki_tokens":int(wiki.numel()),"pile_tokens":int(pile.numel()),
      "wiki_kl_ratio_fes_ind":res["fes"]["wiki"]["kl_to_qwen"]/res["independent"]["wiki"]["kl_to_qwen"],
      "pile_kl_ratio_fes_ind":res["fes"]["pile"]["kl_to_qwen"]/res["independent"]["pile"]["kl_to_qwen"],
      "wiki_nll_ratio_fes_ind":res["fes"]["wiki"]["nll"]/res["independent"]["wiki"]["nll"],
      "pile_nll_ratio_fes_ind":res["fes"]["pile"]["nll"]/res["independent"]["pile"]["nll"],
      "wiki_nll_ratio_fes_orig":res["fes"]["wiki"]["nll"]/res["original"]["wiki"]["nll"],
      "pile_nll_ratio_fes_orig":res["fes"]["pile"]["nll"]/res["original"]["pile"]["nll"],
    }
    out=Path("results-bonsai-large");out.mkdir(exist_ok=True)
    (out/"results.json").write_text(json.dumps({"summary":summary,"paired":paired_stats,"methods":res},indent=2))
    print("SUMMARY",json.dumps(summary,sort_keys=True),flush=True)
    print("PAIRED",json.dumps(paired_stats,sort_keys=True),flush=True)

if __name__=="__main__":main()
