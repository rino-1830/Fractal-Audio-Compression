import argparse,json,math,os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer

GROUP=128
CLIPS=(0.80,0.90,1.0)
SMS=(0.92,1.0,1.08)
BITS=2
QMAX=1

def make_windows(tok,text,seq,n,offset,extra=31):
    ids=tok(text,return_tensors="pt",add_special_tokens=False)["input_ids"][0]
    need=offset+n*(seq+extra)+seq
    if ids.numel()<need:ids=ids.repeat(math.ceil(need/ids.numel()))
    out=[];p=offset
    for _ in range(n):
        out.append(ids[p:p+seq].clone());p+=seq+extra
    return torch.stack(out)

def data(tok,offset):
    from datasets import load_dataset
    va=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="validation")
    te=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
    pi=load_dataset("NeelNanda/pile-10k",split="train")
    vat="\n\n".join(x["text"] for x in va if x["text"].strip())
    tet="\n\n".join(x["text"] for x in te if x["text"].strip())
    pit="\n\n".join(x for x in pi[:384]["text"] if x.strip())
    return (make_windows(tok,vat,48,8,offset),
            make_windows(tok,tet,48,48,offset+937),
            make_windows(tok,pit,48,48,offset+271))

@torch.inference_mode()
def logits(model,x,batch=8):
    return torch.cat([model(input_ids=x[i:i+batch]).logits[:,:-1].float().cpu()
                      for i in range(0,len(x),batch)],0)

def metric(q,r,x):
    rl=F.log_softmax(r.float(),-1);ql=F.log_softmax(q.float(),-1)
    kl=(rl.exp()*(rl-ql)).sum(-1)
    target=x[:,1:].reshape(-1)
    nll=F.nll_loss(ql.reshape(-1,ql.shape[-1]),target,reduction="none").reshape(x.shape[0],-1).mean(-1)
    return {"kl":float(kl.mean().item()),"nll":float(nll.mean().item()),
            "window_kl":kl.mean(-1).tolist(),"window_nll":nll.tolist()}

def candidates(w):
    w=w.float().cpu();rows,cols=w.shape;pad=(-cols)%GROUP
    wp=F.pad(w,(0,pad)) if pad else w;g=wp.reshape(rows,-1,GROUP)
    mx=g.abs().amax(-1,keepdim=True).clamp_min(1e-8);out=[]
    for clip in CLIPS:
        for sm in SMS:
            scale=(mx*clip/QMAX*sm).clamp_min(1e-8)
            q=torch.round(g/scale).clamp(-QMAX,QMAX)*scale
            q=q.reshape(rows,-1)[:,:cols].contiguous()
            out.append({"q":q,"mse":float(F.mse_loss(q,w).item())})
    return out

def mods(model):return [l.mlp.dense_h_to_4h for l in model.gpt_neox.layers]
def restore(ms,orig):
    for m,w in zip(ms,orig):m.weight.data.copy_(w.to(m.weight.dtype))
def apply(ms,sets,ch):
    for i,m in enumerate(ms):m.weight.data.copy_(sets[i][ch[i]]["q"].to(m.weight.dtype))
@torch.inference_mode()
def evm(model,x,ref):return metric(logits(model,x),ref,x)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--offset",type=int,required=True);ap.add_argument("--out",required=True);args=ap.parse_args()
    torch.manual_seed(0);torch.set_num_threads(min(os.cpu_count() or 1,16))
    name="EleutherAI/pythia-70m";tok=AutoTokenizer.from_pretrained(name)
    model=AutoModelForCausalLM.from_pretrained(name,dtype=torch.float32,low_cpu_mem_usage=True);model.eval()
    cal,wiki,pile=data(tok,args.offset);rc=logits(model,cal).half();rw=logits(model,wiki).half();rp=logits(model,pile).half()
    ms=mods(model);orig=[m.weight.detach().cpu().clone() for m in ms];sets=[candidates(w) for w in orig]
    ncan=len(sets[0]);local=[min(range(ncan),key=lambda j:sets[i][j]["mse"]) for i in range(len(ms))]
    independent=[]
    for i in range(len(ms)):
        vals=[]
        for j in range(ncan):
            restore(ms,orig);ms[i].weight.data.copy_(sets[i][j]["q"].to(ms[i].weight.dtype));vals.append(evm(model,cal,rc)["kl"])
        independent.append(min(range(ncan),key=lambda j:vals[j]))
    restore(ms,orig)
    coord=list(independent)
    for p in range(2):
        changed=False
        for i in range(len(ms)):
            vals=[]
            for j in range(ncan):
                trial=list(coord);trial[i]=j;apply(ms,sets,trial);vals.append(evm(model,cal,rc)["kl"])
            b=min(range(ncan),key=lambda j:vals[j]);changed|=(b!=coord[i]);coord[i]=b
        if not changed:break
    restore(ms,orig)
    def evaluate(ch):
        apply(ms,sets,ch);return {"choices":ch,"weight_mse":sum(sets[i][j]["mse"] for i,j in enumerate(ch))/len(ch),
                                  "wiki":evm(model,wiki,rw),"pile":evm(model,pile,rp)}
    m={"local":evaluate(local),"independent":evaluate(independent),"fes":evaluate(coord)}
    f=m["fes"];ind=m["independent"];loc=m["local"]
    s={"offset":args.offset,"wiki_tokens":int(wiki.numel()),"pile_tokens":int(pile.numel()),
       "fes_differs":coord!=independent,
       "wiki_fes_ind":f["wiki"]["kl"]/ind["wiki"]["kl"],"pile_fes_ind":f["pile"]["kl"]/ind["pile"]["kl"],
       "wiki_fes_local":f["wiki"]["kl"]/loc["wiki"]["kl"],"pile_fes_local":f["pile"]["kl"]/loc["pile"]["kl"],
       "weight_mse_fes_local":f["weight_mse"]/loc["weight_mse"]}
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);(out/"results.json").write_text(json.dumps({"summary":s,"methods":m},indent=2))
    print("SUMMARY",json.dumps(s,sort_keys=True),flush=True)
if __name__=="__main__":main()
