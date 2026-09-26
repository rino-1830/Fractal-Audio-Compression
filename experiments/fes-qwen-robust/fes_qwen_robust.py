import argparse,json,math,os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer

GROUP=128
TFS=(0.35,0.55,0.75)
SMS=(0.92,1.0,1.08)

def windows(tok,text,seq,n,offset,extra=23):
    ids=tok(text,return_tensors="pt",add_special_tokens=False)["input_ids"][0]
    need=offset+n*(seq+extra)+seq
    if ids.numel()<need:ids=ids.repeat(math.ceil(need/ids.numel()))
    out=[];p=offset
    for _ in range(n):out.append(ids[p:p+seq].clone());p+=seq+extra
    return torch.stack(out)

def load(tok,offset):
    from datasets import load_dataset
    va=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="validation")
    te=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
    pi=load_dataset("NeelNanda/pile-10k",split="train")
    vat="\n\n".join(x["text"] for x in va if x["text"].strip())
    tet="\n\n".join(x["text"] for x in te if x["text"].strip())
    pit="\n\n".join(x for x in pi[:256]["text"] if x.strip())
    return windows(tok,vat,24,4,offset),windows(tok,tet,24,12,811),windows(tok,pit,24,12,307)

@torch.inference_mode()
def logits(m,x,v,b=2):return torch.cat([m(input_ids=x[i:i+b]).logits[:,:-1,:v].float().cpu() for i in range(0,len(x),b)],0)
def metric(q,r,x):
    rl=F.log_softmax(r.float(),-1);ql=F.log_softmax(q.float(),-1)
    kl=(rl.exp()*(rl-ql)).sum(-1).mean().item()
    tar=x[:,1:].reshape(-1);nll=F.nll_loss(ql.reshape(-1,ql.shape[-1]),tar).item()
    return {"kl":kl,"nll":nll}
def candidates(w):
    w=w.float().cpu();rows,cols=w.shape;pad=(-cols)%GROUP
    wp=F.pad(w,(0,pad)) if pad else w;g=wp.reshape(rows,-1,GROUP)
    ma=g.abs().mean(-1,keepdim=True).clamp_min(1e-12);o=[]
    for tf in TFS:
        sym=torch.sign(g)*(g.abs()>=tf*ma);den=(sym*sym).sum(-1,keepdim=True)
        base=torch.where(den>0,(g*sym).sum(-1,keepdim=True)/den.clamp_min(1),ma)
        for sm in SMS:
            q=(sym*(base*sm).half().float()).reshape(rows,-1)[:,:cols].contiguous()
            o.append({"q":q.half(),"mse":F.mse_loss(q.float(),w).item()})
    return o
def mods(m):return [m.model.layers[i].mlp.up_proj for i in range(3)]
def restore(ms,orig):
    for m,w in zip(ms,orig):m.weight.data.copy_(w.to(m.weight.dtype))
def apply(ms,sets,ch):
    for i,m in enumerate(ms):m.weight.data.copy_(sets[i][ch[i]]["q"].to(m.weight.dtype))
@torch.inference_mode()
def score(m,x,r,v):return metric(logits(m,x,v),r,x)["kl"]
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--offset",type=int,required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
    torch.manual_seed(0);torch.set_num_threads(min(os.cpu_count() or 1,8))
    name="Qwen/Qwen3-1.7B";tok=AutoTokenizer.from_pretrained(name);v=len(tok.get_vocab())
    m=AutoModelForCausalLM.from_pretrained(name,dtype=torch.float32,low_cpu_mem_usage=True);m.eval()
    cal,wiki,pile=load(tok,a.offset);rc=logits(m,cal,v).half();rw=logits(m,wiki,v).half();rp=logits(m,pile,v).half()
    ms=mods(m);orig=[x.weight.detach().cpu().clone() for x in ms];sets=[candidates(w) for w in orig];n=9
    local=[min(range(n),key=lambda j:sets[i][j]["mse"]) for i in range(3)]
    ind=[]
    for i in range(3):
        vals=[]
        for j in range(n):
            restore(ms,orig);ms[i].weight.data.copy_(sets[i][j]["q"].to(ms[i].weight.dtype));vals.append(score(m,cal,rc,v))
        ind.append(min(range(n),key=lambda j:vals[j]))
    restore(ms,orig)
    fes=list(ind)
    for p in range(2):
        changed=False
        for i in range(3):
            vals=[]
            for j in range(n):
                t=list(fes);t[i]=j;apply(ms,sets,t);vals.append(score(m,cal,rc,v))
            b=min(range(n),key=lambda j:vals[j]);changed|=(b!=fes[i]);fes[i]=b
        if not changed:break
    restore(ms,orig)
    def ev(ch):
        apply(ms,sets,ch);return {"choices":ch,"weight_mse":sum(sets[i][j]["mse"] for i,j in enumerate(ch))/3,
                                  "wiki":metric(logits(m,wiki,v),rw,wiki),"pile":metric(logits(m,pile,v),rp,pile)}
    d={"local":ev(local),"independent":ev(ind),"fes":ev(fes)}
    f=d["fes"];ii=d["independent"];l=d["local"]
    s={"offset":a.offset,"fes_differs":fes!=ind,
       "wiki_fes_ind":f["wiki"]["kl"]/ii["wiki"]["kl"],"pile_fes_ind":f["pile"]["kl"]/ii["pile"]["kl"],
       "wiki_fes_local":f["wiki"]["kl"]/l["wiki"]["kl"],"pile_fes_local":f["pile"]["kl"]/l["pile"]["kl"],
       "weight_mse_fes_local":f["weight_mse"]/l["weight_mse"]}
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/"results.json").write_text(json.dumps({"summary":s,"methods":d},indent=2))
    print("SUMMARY",json.dumps(s,sort_keys=True),flush=True)
if __name__=="__main__":main()
