import json,math,os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer

GROUP=128
TFS=(0.35,0.55,0.75)
SMS=(0.92,1.0,1.08)
TOPK=32

def windows(tok,text,seq,n,offset=0,extra=23):
    ids=tok(text,return_tensors="pt",add_special_tokens=False)["input_ids"][0]
    out=[];p=offset
    for _ in range(n):
        out.append(ids[p:p+seq].clone());p+=seq+extra
    return torch.stack(out)

def data(tok):
    from datasets import load_dataset
    va=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="validation")
    te=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
    vat="\n\n".join(x["text"] for x in va if x["text"].strip())
    tet="\n\n".join(x["text"] for x in te if x["text"].strip())
    return windows(tok,vat,24,6,0),windows(tok,tet,24,8,811)

@torch.inference_mode()
def full_logits(m,x,v,b=2):
    return torch.cat([m(input_ids=x[i:i+b]).logits[:,:-1,:v].float().cpu()
                      for i in range(0,len(x),b)],0)

def metric(q,r):
    rl=F.log_softmax(r,-1);ql=F.log_softmax(q,-1)
    return float((rl.exp()*(rl-ql)).sum(-1).mean().item())

def candidates(w):
    w=w.float().cpu();rows,cols=w.shape;pad=(-cols)%GROUP
    wp=F.pad(w,(0,pad)) if pad else w
    g=wp.reshape(rows,-1,GROUP)
    ma=g.abs().mean(-1,keepdim=True).clamp_min(1e-12)
    out=[]
    for tf in TFS:
        sym=torch.sign(g)*(g.abs()>=tf*ma)
        den=(sym*sym).sum(-1,keepdim=True)
        base=torch.where(den>0,(g*sym).sum(-1,keepdim=True)/den.clamp_min(1),ma)
        for sm in SMS:
            q=(sym*(base*sm).half().float()).reshape(rows,-1)[:,:cols].contiguous()
            out.append({"q":q.half(),"mse":float(F.mse_loss(q.float(),w).item())})
    return out

def mods(m,n=3):return [m.model.layers[i].mlp.up_proj for i in range(n)]
def restore(ms,orig):
    for m,w in zip(ms,orig):m.weight.data.copy_(w.to(m.weight.dtype))
def apply(ms,sets,ch):
    for i,m in enumerate(ms):m.weight.data.copy_(sets[i][ch[i]]["q"].to(m.weight.dtype))

@torch.inference_mode()
def score(m,x,ref,v):return metric(full_logits(m,x,v),ref)

def sweep(m,ms,sets,ch,cal,rc,v):
    ch=list(ch)
    for i in range(len(ms)):
        vals=[]
        for j in range(len(sets[i])):
            t=list(ch);t[i]=j;apply(ms,sets,t)
            vals.append(score(m,cal,rc,v))
        ch[i]=min(range(len(vals)),key=lambda j:vals[j])
    return ch

@torch.inference_mode()
def probe_delta(m,test,base_top_idx,base_top_vals):
    q=full_logits(m,test,base_top_idx.shape[-1] if False else m.config.vocab_size)
    # Some Qwen heads are padded; base_top_idx always lies inside assigned vocab.
    vals=torch.gather(q,-1,base_top_idx)
    vals=vals-vals.mean(-1,keepdim=True)
    return (vals-base_top_vals).reshape(-1)

def decomposition(m,ms,orig,sets,ch,test,base_full,vocab):
    top=torch.topk(base_full,k=TOPK,dim=-1)
    idx=top.indices
    base_vals=top.values
    base_vals=base_vals-base_vals.mean(-1,keepdim=True)

    indiv=[]
    for i in range(len(ms)):
        restore(ms,orig)
        ms[i].weight.data.copy_(sets[i][ch[i]]["q"].to(ms[i].weight.dtype))
        q=full_logits(m,test,vocab)
        qv=torch.gather(q,-1,idx);qv=qv-qv.mean(-1,keepdim=True)
        indiv.append((qv-base_vals).reshape(-1))
    restore(ms,orig)

    apply(ms,sets,ch)
    q=full_logits(m,test,vocab)
    qv=torch.gather(q,-1,idx);qv=qv-qv.mean(-1,keepdim=True)
    total=(qv-base_vals).reshape(-1)
    restore(ms,orig)

    summed=torch.stack(indiv).sum(0)
    indiv_energy=sum(float(torch.dot(d,d).item()) for d in indiv)
    sum_energy=float(torch.dot(summed,summed).item())
    total_energy=float(torch.dot(total,total).item())
    resid=total-summed
    resid_energy=float(torch.dot(resid,resid).item())
    cross=sum_energy-indiv_energy
    return {
      "individual_energy_sum":indiv_energy,
      "linear_sum_energy":sum_energy,
      "actual_combined_energy":total_energy,
      "cancellation_ratio":indiv_energy/max(sum_energy,1e-30),
      "normalized_cross_term":cross/max(indiv_energy,1e-30),
      "nonlinearity_residual_ratio":resid_energy/max(total_energy,1e-30),
    }

def main():
    torch.manual_seed(0);torch.set_num_threads(min(os.cpu_count() or 1,8))
    name="Qwen/Qwen3-1.7B"
    tok=AutoTokenizer.from_pretrained(name);v=len(tok.get_vocab())
    m=AutoModelForCausalLM.from_pretrained(name,dtype=torch.float32,low_cpu_mem_usage=True);m.eval()
    cal,test=data(tok)
    rc=full_logits(m,cal,v).half()
    rt=full_logits(m,test,v).float()
    ms=mods(m,3);orig=[x.weight.detach().cpu().clone() for x in ms]
    sets=[candidates(w) for w in orig]
    local=[min(range(len(sets[i])),key=lambda j:sets[i][j]["mse"]) for i in range(3)]
    fes=sweep(m,ms,sets,local,cal,rc,v)
    fes=sweep(m,ms,sets,fes,cal,rc,v)

    methods={}
    for name_,ch in [("local",local),("fes",fes)]:
        apply(ms,sets,ch);q=full_logits(m,test,v);test_kl=metric(q,rt);restore(ms,orig)
        methods[name_]={"choices":ch,"test_kl":test_kl,
                        "decomposition":decomposition(m,ms,orig,sets,ch,test,rt,v)}
    summary={
      "fes_differs":fes!=local,
      "fes_test_kl_vs_local":methods["fes"]["test_kl"]/methods["local"]["test_kl"],
      "local_cancellation_ratio":methods["local"]["decomposition"]["cancellation_ratio"],
      "fes_cancellation_ratio":methods["fes"]["decomposition"]["cancellation_ratio"],
      "local_cross_term":methods["local"]["decomposition"]["normalized_cross_term"],
      "fes_cross_term":methods["fes"]["decomposition"]["normalized_cross_term"],
      "local_nonlinearity":methods["local"]["decomposition"]["nonlinearity_residual_ratio"],
      "fes_nonlinearity":methods["fes"]["decomposition"]["nonlinearity_residual_ratio"],
    }
    out=Path("results-interaction");out.mkdir(exist_ok=True)
    (out/"results.json").write_text(json.dumps({"summary":summary,"methods":methods},indent=2))
    print("SUMMARY",json.dumps(summary,sort_keys=True),flush=True)

if __name__=="__main__":main()
