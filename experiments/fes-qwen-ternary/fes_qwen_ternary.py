import argparse,json,math,os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer

GROUP=128
TFS=(0.35,0.55,0.75)
SMS=(0.92,1.0,1.08)

def make_windows(tok,text,seq,n,offset=0,extra=23):
    ids=tok(text,return_tensors="pt",add_special_tokens=False)["input_ids"][0]
    need=offset+n*(seq+extra)+seq
    if ids.numel()<need:ids=ids.repeat(math.ceil(need/ids.numel()))
    out=[];p=offset
    for _ in range(n):
        out.append(ids[p:p+seq].clone());p+=seq+extra
    return torch.stack(out)

def data(tok):
    from datasets import load_dataset
    va=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="validation")
    te=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
    pi=load_dataset("NeelNanda/pile-10k",split="train")
    vat="\n\n".join(x["text"] for x in va if x["text"].strip())
    tet="\n\n".join(x["text"] for x in te if x["text"].strip())
    pit="\n\n".join(x for x in pi[:256]["text"] if x.strip())
    return make_windows(tok,vat,24,4,0),make_windows(tok,tet,24,12,811),make_windows(tok,pit,24,12,307)

@torch.inference_mode()
def logits(model,x,vocab,batch=2):
    return torch.cat([model(input_ids=x[i:i+batch]).logits[:,:-1,:vocab].float().cpu()
                      for i in range(0,len(x),batch)],0)

def metrics(q,r,x):
    rl=F.log_softmax(r.float(),-1);ql=F.log_softmax(q.float(),-1)
    kl=(rl.exp()*(rl-ql)).sum(-1).mean().item()
    target=x[:,1:].reshape(-1)
    nll=F.nll_loss(ql.reshape(-1,ql.shape[-1]),target,reduction="mean").item()
    return {"kl":float(kl),"nll":float(nll)}

def ternary_candidates(w):
    w=w.float().cpu();rows,cols=w.shape;pad=(-cols)%GROUP
    wp=F.pad(w,(0,pad)) if pad else w
    g=wp.reshape(rows,-1,GROUP)
    meanabs=g.abs().mean(-1,keepdim=True).clamp_min(1e-12)
    out=[]
    for tf in TFS:
        sym=torch.sign(g)*(g.abs()>=tf*meanabs)
        den=(sym*sym).sum(-1,keepdim=True)
        base=torch.where(den>0,(g*sym).sum(-1,keepdim=True)/den.clamp_min(1),meanabs)
        for sm in SMS:
            scale=(base*sm).half().float()
            q=(sym*scale).reshape(rows,-1)[:,:cols].contiguous()
            out.append({"q":q.half(),"mse":float(F.mse_loss(q.float(),w).item()),
                        "tf":tf,"sm":sm,
                        "zero_fraction":float((sym==0).float().mean().item())})
    return out

def target_modules(model,n):
    return [model.model.layers[i].mlp.up_proj for i in range(n)]
def restore(ms,orig):
    for m,w in zip(ms,orig):m.weight.data.copy_(w.to(m.weight.dtype))
def apply(ms,sets,ch):
    for i,m in enumerate(ms):m.weight.data.copy_(sets[i][ch[i]]["q"].to(m.weight.dtype))
@torch.inference_mode()
def evm(model,x,ref,vocab):return metrics(logits(model,x,vocab),ref,x)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--layers",type=int,default=6);ap.add_argument("--out",required=True);args=ap.parse_args()
    torch.manual_seed(0);torch.set_num_threads(min(os.cpu_count() or 1,8))
    name="Qwen/Qwen3-1.7B";tok=AutoTokenizer.from_pretrained(name);vocab=len(tok.get_vocab())
    model=AutoModelForCausalLM.from_pretrained(name,dtype=torch.float32,low_cpu_mem_usage=True);model.eval()
    cal,wiki,pile=data(tok)
    rc=logits(model,cal,vocab).half();rw=logits(model,wiki,vocab).half();rp=logits(model,pile,vocab).half()
    ms=target_modules(model,args.layers);orig=[m.weight.detach().cpu().clone() for m in ms]
    sets=[ternary_candidates(w) for w in orig];ncan=len(sets[0])
    local=[min(range(ncan),key=lambda j:sets[i][j]["mse"]) for i in range(args.layers)]

    independent=[]
    for i in range(args.layers):
        vals=[]
        for j in range(ncan):
            restore(ms,orig);ms[i].weight.data.copy_(sets[i][j]["q"].to(ms[i].weight.dtype))
            vals.append(evm(model,cal,rc,vocab)["kl"])
        independent.append(min(range(ncan),key=lambda j:vals[j]))
        print("independent",i,independent[-1],min(vals),flush=True)
    restore(ms,orig)

    coord=list(independent);hist=[]
    for p in range(2):
        changed=False
        for i in range(args.layers):
            vals=[]
            for j in range(ncan):
                trial=list(coord);trial[i]=j;apply(ms,sets,trial)
                vals.append(evm(model,cal,rc,vocab)["kl"])
            b=min(range(ncan),key=lambda j:vals[j]);changed|=(b!=coord[i]);coord[i]=b
            hist.append([p,i,b,vals[b]]);print("coord",p,i,b,vals[b],flush=True)
        if not changed:break
    restore(ms,orig)

    def evaluate(ch):
        apply(ms,sets,ch)
        return {"choices":ch,
                "params":[{"tf":sets[i][j]["tf"],"sm":sets[i][j]["sm"],"zero":sets[i][j]["zero_fraction"]} for i,j in enumerate(ch)],
                "weight_mse":sum(sets[i][j]["mse"] for i,j in enumerate(ch))/len(ch),
                "wiki":evm(model,wiki,rw,vocab),"pile":evm(model,pile,rp,vocab)}
    m={"local":evaluate(local),"independent":evaluate(independent),"fes":evaluate(coord)}
    f=m["fes"];ind=m["independent"];loc=m["local"]
    s={"model":name,"layers_quantized":args.layers,"group_size":GROUP,"ternary":True,
       "fes_differs":coord!=independent,
       "wiki_fes_ind":f["wiki"]["kl"]/ind["wiki"]["kl"],"pile_fes_ind":f["pile"]["kl"]/ind["pile"]["kl"],
       "wiki_fes_local":f["wiki"]["kl"]/loc["wiki"]["kl"],"pile_fes_local":f["pile"]["kl"]/loc["pile"]["kl"],
       "weight_mse_fes_local":f["weight_mse"]/loc["weight_mse"]}
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);(out/"results.json").write_text(json.dumps({"summary":s,"methods":m,"history":hist},indent=2))
    print("SUMMARY",json.dumps(s,sort_keys=True),flush=True)
if __name__=="__main__":main()
