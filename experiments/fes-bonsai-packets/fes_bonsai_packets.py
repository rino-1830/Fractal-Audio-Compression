import argparse, gc, itertools, json, math, os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

GROUP=128
PACKET_FRAC=0.00025
N_PACKETS=5

def load_windows(tok,seq_len=32,n_cal=3,n_test=8):
    from datasets import load_dataset
    ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="validation")
    text="\n\n".join(x["text"] for x in ds if x["text"].strip())
    ids=tok(text,return_tensors="pt",add_special_tokens=False)["input_ids"][0]
    out=[]; pos=0
    for _ in range(n_cal+n_test):
        out.append(ids[pos:pos+seq_len].clone()); pos+=seq_len+29
    return out[:n_cal],out[n_cal:]

@torch.inference_mode()
def cache(model,ws,vocab):
    return [model(input_ids=x.unsqueeze(0)).logits[0,:-1,:vocab].float().cpu().half() for x in ws]

@torch.inference_mode()
def metrics(model,ws,ref,vocab):
    kl=nll=0.; tokens=0
    for ids,rh in zip(ws,ref):
        r=rh.float(); q=model(input_ids=ids.unsqueeze(0)).logits[0,:-1,:vocab].float().cpu()
        rl=F.log_softmax(r,dim=-1); ql=F.log_softmax(q,dim=-1)
        kl+=(rl.exp()*(rl-ql)).sum().item()
        nll+=F.nll_loss(ql,ids[1:].cpu(),reduction="sum").item(); tokens+=ids.numel()-1
    return {"kl":kl/tokens,"nll":nll/tokens}

def mods(model,n):
    return [model.model.layers[i].mlp.up_proj for i in range(n)]

def grads(model,ms,cal,ref,vocab):
    for p in model.parameters(): p.requires_grad_(False)
    for m in ms: m.weight.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    for ids,rh in zip(cal,ref):
        q=model(input_ids=ids.unsqueeze(0)).logits[0,:-1,:vocab].float()
        r=rh.float().to(q.device)
        rl=F.log_softmax(r,dim=-1); ql=F.log_softmax(q,dim=-1)
        ((rl.exp()*(rl-ql)).sum(-1).mean()/len(cal)).backward()
    out=[m.weight.grad.detach().float().cpu().clone() for m in ms]
    for m in ms: m.weight.grad=None; m.weight.requires_grad_(False)
    return out

def packet_candidates(wb,wr,g):
    wb=wb.float().cpu(); wr=wr.float().cpu(); g=g.float().cpu()
    rows,cols=wb.shape; bg=wb.reshape(rows,cols//GROUP,GROUP); gg=g.reshape_as(bg)
    sc=bg.abs().amax(-1,keepdim=True).expand_as(bg); cur=bg; inf=torch.tensor(float("inf"))
    d0=torch.where(cur==0,inf,gg*(0-cur))
    dp=torch.where(cur>0,inf,gg*(sc-cur))
    dn=torch.where(cur<0,inf,gg*(-sc-cur))
    bd,ba=torch.stack((d0,dp,dn),0).min(0)
    fd=bd.reshape(-1); fa=ba.reshape(-1); fs=sc.reshape(-1)
    improving=torch.nonzero(fd<0,as_tuple=False).reshape(-1)
    order=improving[torch.argsort(fd[improving])]
    k=max(1,int(round(PACKET_FRAC*wb.numel())))
    choices=[("base",torch.tensor([],dtype=torch.long))]
    for p in range(N_PACKETS):
        idx=order[p*k:(p+1)*k]
        choices.append((f"packet{p}",idx))
    # plus union of first two packets for a different magnitude/direction.
    choices.append(("packet01",order[:2*k]))
    sets=[]; mses=[]; stats=[]
    for name,idx in choices:
        q=wb.half().reshape(-1).clone()
        if idx.numel():
            alt=fa[idx]; scale=fs[idx].half()
            q[idx]=torch.where(alt==0,torch.zeros_like(scale),torch.where(alt==1,scale,-scale))
        q=q.reshape_as(wb).contiguous()
        sets.append(q); mses.append(float(F.mse_loss(q.float(),wr).item()))
        stats.append({"name":name,"flips":int(idx.numel()),"first_order_delta":float(fd[idx].sum().item()) if idx.numel() else 0.0})
    return sets,mses,stats

def apply(ms,sets,ch):
    for i,m in enumerate(ms): m.weight.data.copy_(sets[i][ch[i]].to(m.weight.dtype))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--layers",type=int,default=3); ap.add_argument("--out",default="results-packets"); args=ap.parse_args()
    torch.manual_seed(0); torch.set_num_threads(min(os.cpu_count() or 1,8))
    refn="Qwen/Qwen3-1.7B"; bonn="prism-ml/Ternary-Bonsai-1.7B-unpacked"
    tok=AutoTokenizer.from_pretrained(bonn); vocab=len(tok.get_vocab()); cal,test=load_windows(tok)

    refm=AutoModelForCausalLM.from_pretrained(refn,dtype=torch.float32,low_cpu_mem_usage=True); refm.eval()
    rms=mods(refm,args.layers); rws=[m.weight.detach().cpu().half().clone() for m in rms]
    rc=cache(refm,cal,vocab); rt=cache(refm,test,vocab); del refm,rms; gc.collect()

    model=AutoModelForCausalLM.from_pretrained(bonn,dtype=torch.float32,low_cpu_mem_usage=True); model.eval()
    ms=mods(model,args.layers); bws=[m.weight.detach().cpu().clone() for m in ms]
    gs=grads(model,ms,cal,rc,vocab)
    sets=[]; mse=[]; st=[]
    for wb,wr,g in zip(bws,rws,gs):
        a,b,c=packet_candidates(wb,wr,g); sets.append(a); mse.append(b); st.append(c)
    names=[x["name"] for x in st[0]]
    base=[0]*args.layers

    independent=[]
    for i in range(args.layers):
        vals=[]
        for j in range(len(names)):
            trial=list(base); trial[i]=j; apply(ms,sets,trial); vals.append(metrics(model,cal,rc,vocab)["kl"])
        independent.append(min(range(len(vals)),key=lambda j:vals[j]))

    grid=[]
    for ch in itertools.product(range(len(names)),repeat=args.layers):
        apply(ms,sets,ch); grid.append((metrics(model,cal,rc,vocab)["kl"],ch))
    grid.sort(key=lambda x:x[0]); glob=list(grid[0][1])

    def ev(ch):
        apply(ms,sets,ch)
        return {"choices":list(ch),"names":[names[j] for j in ch],
                "weight_mse":sum(mse[i][ch[i]] for i in range(args.layers))/args.layers,
                "cal":metrics(model,cal,rc,vocab),"test":metrics(model,test,rt,vocab)}
    methods={"original":ev(base),"independent":ev(independent),"global":ev(glob)}
    s={"candidate_names":names,"packet_fraction":PACKET_FRAC,
       "global_differs_from_independent":glob!=independent,
       "test_kl_global_vs_independent":methods["global"]["test"]["kl"]/methods["independent"]["test"]["kl"],
       "test_nll_global_vs_independent":methods["global"]["test"]["nll"]/methods["independent"]["test"]["nll"],
       "test_kl_global_vs_original":methods["global"]["test"]["kl"]/methods["original"]["test"]["kl"],
       "test_nll_global_vs_original":methods["global"]["test"]["nll"]/methods["original"]["test"]["nll"]}
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps({"summary":s,"stats":st,"methods":methods,"top_grid":[{"kl":v,"names":[names[j] for j in ch]} for v,ch in grid[:20]]},indent=2))
    print("SUMMARY",json.dumps(s,sort_keys=True),flush=True)

if __name__=="__main__": main()
