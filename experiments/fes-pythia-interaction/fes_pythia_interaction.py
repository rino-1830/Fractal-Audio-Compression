import json,math,os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer

TFS=(0.35,0.50,0.65,0.80,0.95)
SMS=(0.90,1.00,1.10)
TOPK=32

def win(tok,text,seq,n,off=0,extra=23):
    ids=tok(text,return_tensors="pt",add_special_tokens=False)["input_ids"][0]
    out=[];p=off
    for _ in range(n):out.append(ids[p:p+seq].clone());p+=seq+extra
    return torch.stack(out)

def data(tok):
    from datasets import load_dataset
    va=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="validation")
    te=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
    vat="\n\n".join(x["text"] for x in va if x["text"].strip())
    tet="\n\n".join(x["text"] for x in te if x["text"].strip())
    return win(tok,vat,48,8),win(tok,tet,48,16,991)

@torch.inference_mode()
def logits(m,x,b=8):
    return torch.cat([m(input_ids=x[i:i+b]).logits[:,:-1].float().cpu()
                      for i in range(0,len(x),b)],0)

def kl(q,r):
    rl=F.log_softmax(r.float(),-1);ql=F.log_softmax(q.float(),-1)
    return float((rl.exp()*(rl-ql)).sum(-1).mean().item())

def cand(w):
    w=w.float().cpu();ma=w.abs().mean().item()+1e-12;o=[]
    for tf in TFS:
        s=torch.sign(w)*(w.abs()>=tf*ma)
        den=(s*s).sum().item();base=ma if den==0 else (w*s).sum().item()/den
        for sm in SMS:
            q=(base*sm)*s
            o.append({"q":q.contiguous(),"mse":F.mse_loss(q,w).item()})
    return o

def mods(m):return [l.mlp.dense_h_to_4h for l in m.gpt_neox.layers]
def restore(ms,orig):
    for m,w in zip(ms,orig):m.weight.data.copy_(w.to(m.weight.dtype))
def apply(ms,sets,ch):
    for i,m in enumerate(ms):m.weight.data.copy_(sets[i][ch[i]]["q"].to(m.weight.dtype))

@torch.inference_mode()
def score(m,x,r):return kl(logits(m,x),r)

def sweep(m,ms,sets,ch,cal,rc):
    ch=list(ch)
    for i in range(len(ms)):
        vals=[]
        for j in range(len(sets[i])):
            t=list(ch);t[i]=j;apply(ms,sets,t);vals.append(score(m,cal,rc))
        ch[i]=min(range(len(vals)),key=lambda j:vals[j])
    return ch

@torch.inference_mode()
def decomposition(m,ms,orig,sets,ch,test,base):
    top=torch.topk(base,k=TOPK,dim=-1)
    idx=top.indices
    bv=top.values;bv=bv-bv.mean(-1,keepdim=True)
    indiv=[]
    for i in range(len(ms)):
        restore(ms,orig)
        ms[i].weight.data.copy_(sets[i][ch[i]]["q"].to(ms[i].weight.dtype))
        q=logits(m,test)
        qv=torch.gather(q,-1,idx);qv=qv-qv.mean(-1,keepdim=True)
        indiv.append((qv-bv).reshape(-1))
    restore(ms,orig)
    apply(ms,sets,ch);q=logits(m,test);restore(ms,orig)
    qv=torch.gather(q,-1,idx);qv=qv-qv.mean(-1,keepdim=True)
    total=(qv-bv).reshape(-1)
    summed=torch.stack(indiv).sum(0)
    indiv_e=sum(float(torch.dot(d,d).item()) for d in indiv)
    sum_e=float(torch.dot(summed,summed).item())
    total_e=float(torch.dot(total,total).item())
    resid=total-summed
    resid_e=float(torch.dot(resid,resid).item())
    return {
      "individual_energy_sum":indiv_e,
      "linear_sum_energy":sum_e,
      "actual_combined_energy":total_e,
      "cancellation_ratio":indiv_e/max(sum_e,1e-30),
      "normalized_cross_term":(sum_e-indiv_e)/max(indiv_e,1e-30),
      "nonlinearity_residual_ratio":resid_e/max(total_e,1e-30),
    }

def main():
    torch.manual_seed(0);torch.set_num_threads(min(os.cpu_count() or 1,16))
    name="EleutherAI/pythia-70m"
    tok=AutoTokenizer.from_pretrained(name)
    m=AutoModelForCausalLM.from_pretrained(name,dtype=torch.float32,low_cpu_mem_usage=True);m.eval()
    cal,test=data(tok);rc=logits(m,cal).half();rt=logits(m,test).float()
    ms=mods(m);orig=[x.weight.detach().cpu().clone() for x in ms];sets=[cand(w) for w in orig]
    local=[min(range(len(sets[i])),key=lambda j:sets[i][j]["mse"]) for i in range(len(ms))]
    one=sweep(m,ms,sets,local,cal,rc);fes=sweep(m,ms,sets,one,cal,rc)
    methods={}
    for nm,ch in [("local",local),("fes",fes)]:
        apply(ms,sets,ch);q=logits(m,test);testkl=kl(q,rt);restore(ms,orig)
        methods[nm]={"choices":ch,"test_kl":testkl,"decomposition":decomposition(m,ms,orig,sets,ch,test,rt)}
    s={
      "fes_test_kl_vs_local":methods["fes"]["test_kl"]/methods["local"]["test_kl"],
      "local_cancellation_ratio":methods["local"]["decomposition"]["cancellation_ratio"],
      "fes_cancellation_ratio":methods["fes"]["decomposition"]["cancellation_ratio"],
      "local_cross_term":methods["local"]["decomposition"]["normalized_cross_term"],
      "fes_cross_term":methods["fes"]["decomposition"]["normalized_cross_term"],
      "local_nonlinearity":methods["local"]["decomposition"]["nonlinearity_residual_ratio"],
      "fes_nonlinearity":methods["fes"]["decomposition"]["nonlinearity_residual_ratio"],
    }
    out=Path("results-pythia-interaction");out.mkdir(exist_ok=True)
    (out/"results.json").write_text(json.dumps({"summary":s,"methods":methods},indent=2))
    print("SUMMARY",json.dumps(s,sort_keys=True),flush=True)

if __name__=="__main__":main()
