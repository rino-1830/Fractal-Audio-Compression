import json,math,os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer

TFS=(0.35,0.50,0.65,0.80,0.95);SMS=(0.90,1.00,1.10)
def win(tok,text,seq,n,off=0,extra=23):
    ids=tok(text,return_tensors="pt",add_special_tokens=False)["input_ids"][0];out=[];p=off
    for _ in range(n):out.append(ids[p:p+seq].clone());p+=seq+extra
    return torch.stack(out)
def data(tok):
    from datasets import load_dataset
    va=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="validation");te=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test");pi=load_dataset("NeelNanda/pile-10k",split="train")
    vat="\n\n".join(x["text"] for x in va if x["text"].strip());tet="\n\n".join(x["text"] for x in te if x["text"].strip());pit="\n\n".join(x for x in pi[:256]["text"] if x.strip())
    return win(tok,vat,48,8),win(tok,tet,48,32,991),win(tok,pit,48,32,311)
@torch.inference_mode()
def logits(m,x,b=8):return torch.cat([m(input_ids=x[i:i+b]).logits[:,:-1].float().cpu() for i in range(0,len(x),b)],0)
def metric(q,r,x):
    rl=F.log_softmax(r.float(),-1);ql=F.log_softmax(q.float(),-1);return {"kl":(rl.exp()*(rl-ql)).sum(-1).mean().item()}
def cand(w):
    w=w.float().cpu();ma=w.abs().mean().item()+1e-12;o=[]
    for tf in TFS:
        s=torch.sign(w)*(w.abs()>=tf*ma);den=(s*s).sum().item();base=ma if den==0 else (w*s).sum().item()/den
        for sm in SMS:
            q=(base*sm)*s;o.append({"q":q.contiguous(),"mse":F.mse_loss(q,w).item()})
    return o
def mods(m):return [l.mlp.dense_h_to_4h for l in m.gpt_neox.layers]
def apply(ms,sets,ch):
    for i,m in enumerate(ms):m.weight.data.copy_(sets[i][ch[i]]["q"].to(m.weight.dtype))
@torch.inference_mode()
def score(m,x,r):return metric(logits(m,x),r,x)["kl"]
def sweep(m,ms,sets,ch,cal,rc):
    ch=list(ch);ev=0
    for i in range(len(ms)):
        vals=[]
        for j in range(15):
            t=list(ch);t[i]=j;apply(ms,sets,t);vals.append(score(m,cal,rc));ev+=1
        ch[i]=min(range(15),key=lambda j:vals[j])
    return ch,ev
def evaluate(m,ms,sets,ch,wiki,rw,pile,rp):
    apply(ms,sets,ch);return {"choices":ch,"wiki":metric(logits(m,wiki),rw,wiki),"pile":metric(logits(m,pile),rp,pile)}
def main():
    torch.manual_seed(0);torch.set_num_threads(min(os.cpu_count() or 1,16));name="EleutherAI/pythia-70m"
    tok=AutoTokenizer.from_pretrained(name);m=AutoModelForCausalLM.from_pretrained(name,dtype=torch.float32,low_cpu_mem_usage=True);m.eval()
    cal,wiki,pile=data(tok);rc=logits(m,cal).half();rw=logits(m,wiki).half();rp=logits(m,pile).half()
    ms=mods(m);orig=[x.weight.detach().cpu().clone() for x in ms];sets=[cand(w) for w in orig]
    local=[min(range(15),key=lambda j:sets[i][j]["mse"]) for i in range(len(ms))]
    one,e1=sweep(m,ms,sets,local,cal,rc);two,e2=sweep(m,ms,sets,one,cal,rc);three,e3=sweep(m,ms,sets,two,cal,rc)
    meth={"local":evaluate(m,ms,sets,local,wiki,rw,pile,rp),"one":evaluate(m,ms,sets,one,wiki,rw,pile,rp),"two":evaluate(m,ms,sets,two,wiki,rw,pile,rp),"three":evaluate(m,ms,sets,three,wiki,rw,pile,rp)}
    b=meth["local"];s={"evals_per_sweep":e1,"change1":one!=local,"change2":two!=one,"change3":three!=two}
    for k in ("one","two","three"):
        s[k+"_wiki_vs_local"]=meth[k]["wiki"]["kl"]/b["wiki"]["kl"];s[k+"_pile_vs_local"]=meth[k]["pile"]["kl"]/b["pile"]["kl"]
    out=Path("results-localstart");out.mkdir(exist_ok=True);(out/"results.json").write_text(json.dumps({"summary":s,"methods":meth},indent=2));print("SUMMARY",json.dumps(s,sort_keys=True),flush=True)
if __name__=="__main__":main()
