import json,math,os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer

GROUP=128
TFS=(0.35,0.55,0.75)
SMS=(0.92,1.0,1.08)

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
    return windows(tok,vat,24,6,0),windows(tok,tet,24,4,811)

@torch.inference_mode()
def logits(m,x,v,b=1):
    return torch.cat([m(input_ids=x[i:i+b]).logits[:,:-1,:v].float().cpu()
                      for i in range(0,len(x),b)],0)

def kl(q,r):
    rl=F.log_softmax(r,-1);ql=F.log_softmax(q,-1)
    return float((rl.exp()*(rl-ql)).sum(-1).mean().item())

def candidates(w):
    w=w.float().cpu();rows,cols=w.shape;pad=(-cols)%GROUP
    wp=F.pad(w,(0,pad)) if pad else w
    g=wp.reshape(rows,-1,GROUP);ma=g.abs().mean(-1,keepdim=True).clamp_min(1e-12)
    out=[]
    for tf in TFS:
        sym=torch.sign(g)*(g.abs()>=tf*ma);den=(sym*sym).sum(-1,keepdim=True)
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
def score(m,x,r,v):return kl(logits(m,x,v),r)

def sweep(m,ms,sets,ch,cal,rc,v):
    ch=list(ch)
    for i in range(len(ms)):
        vals=[]
        for j in range(len(sets[i])):
            t=list(ch);t[i]=j;apply(ms,sets,t);vals.append(score(m,cal,rc,v))
        ch[i]=min(range(len(vals)),key=lambda j:vals[j])
    return ch

def fisher_inner(a,b,p):
    # Mean token-wise a^T (diag(p)-pp^T) b.
    pa=(p*a).sum(-1)
    pb=(p*b).sum(-1)
    return float(((p*a*b).sum(-1)-pa*pb).mean().item())

@torch.inference_mode()
def decomposition(m,ms,orig,sets,ch,test,base,v):
    p=F.softmax(base,-1)
    indiv=[]
    for i in range(len(ms)):
        restore(ms,orig)
        ms[i].weight.data.copy_(sets[i][ch[i]]["q"].to(ms[i].weight.dtype))
        indiv.append(logits(m,test,v)-base)
    restore(ms,orig)

    apply(ms,sets,ch);actual=logits(m,test,v)-base;restore(ms,orig)
    summed=torch.stack(indiv).sum(0)

    indiv_e=sum(fisher_inner(d,d,p) for d in indiv)
    sum_e=fisher_inner(summed,summed,p)
    actual_e=fisher_inner(actual,actual,p)
    resid=actual-summed
    resid_e=fisher_inner(resid,resid,p)
    pair=[]
    for i in range(len(indiv)):
        for j in range(i+1,len(indiv)):
            pair.append({"i":i,"j":j,"fisher_cross":2*fisher_inner(indiv[i],indiv[j],p)})
    return {
      "individual_energy_sum":indiv_e,
      "linear_sum_energy":sum_e,
      "actual_combined_energy":actual_e,
      "fisher_cancellation_ratio":indiv_e/max(sum_e,1e-30),
      "normalized_fisher_cross_term":(sum_e-indiv_e)/max(indiv_e,1e-30),
      "fisher_nonlinearity_residual_ratio":resid_e/max(actual_e,1e-30),
      "pairwise_cross_terms":pair,
    }

def main():
    torch.manual_seed(0);torch.set_num_threads(min(os.cpu_count() or 1,8))
    name="Qwen/Qwen3-1.7B"
    tok=AutoTokenizer.from_pretrained(name);v=len(tok.get_vocab())
    m=AutoModelForCausalLM.from_pretrained(name,dtype=torch.float32,low_cpu_mem_usage=True);m.eval()
    cal,test=data(tok);rc=logits(m,cal,v).half();rt=logits(m,test,v)
    ms=mods(m);orig=[x.weight.detach().cpu().clone() for x in ms];sets=[candidates(w) for w in orig]
    local=[min(range(len(sets[i])),key=lambda j:sets[i][j]["mse"]) for i in range(len(ms))]
    fes=sweep(m,ms,sets,local,cal,rc,v);fes=sweep(m,ms,sets,fes,cal,rc,v)

    methods={}
    for nm,ch in [("local",local),("fes",fes)]:
        apply(ms,sets,ch);q=logits(m,test,v);testkl=kl(q,rt);restore(ms,orig)
        methods[nm]={"choices":ch,"test_kl":testkl,
                     "decomposition":decomposition(m,ms,orig,sets,ch,test,rt,v)}
    s={
      "fes_test_kl_vs_local":methods["fes"]["test_kl"]/methods["local"]["test_kl"],
      "local_fisher_cancellation":methods["local"]["decomposition"]["fisher_cancellation_ratio"],
      "fes_fisher_cancellation":methods["fes"]["decomposition"]["fisher_cancellation_ratio"],
      "local_fisher_cross":methods["local"]["decomposition"]["normalized_fisher_cross_term"],
      "fes_fisher_cross":methods["fes"]["decomposition"]["normalized_fisher_cross_term"],
      "local_fisher_nonlinearity":methods["local"]["decomposition"]["fisher_nonlinearity_residual_ratio"],
      "fes_fisher_nonlinearity":methods["fes"]["decomposition"]["fisher_nonlinearity_residual_ratio"],
    }
    out=Path("results-fisher");out.mkdir(exist_ok=True)
    (out/"results.json").write_text(json.dumps({"summary":s,"methods":methods},indent=2))
    print("SUMMARY",json.dumps(s,sort_keys=True),flush=True)

if __name__=="__main__":main()
