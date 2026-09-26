import argparse,json,math,os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer

GROUP=128
TFS=(0.35,0.55,0.75)
SMS=(0.92,1.0,1.08)

def windows(tok,text,seq,n,offset,extra=29):
    ids=tok(text,return_tensors="pt",add_special_tokens=False)["input_ids"][0]
    need=offset+n*(seq+extra)+seq
    if ids.numel()<need: ids=ids.repeat(math.ceil(need/ids.numel()))
    out=[]; p=offset
    for _ in range(n):
        out.append(ids[p:p+seq].clone()); p+=seq+extra
    return torch.stack(out)

def load(tok):
    from datasets import load_dataset
    va=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="validation")
    te=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
    pi=load_dataset("NeelNanda/pile-10k",split="train")
    vat="\n\n".join(x["text"] for x in va if x["text"].strip())
    tet="\n\n".join(x["text"] for x in te if x["text"].strip())
    pit="\n\n".join(x for x in pi[:512]["text"] if x.strip())
    cal_w=windows(tok,vat,24,4,0)
    cal_p=windows(tok,pit,24,4,0)
    test_w=windows(tok,tet,24,12,911)
    test_p=windows(tok,pit,24,12,12000)
    return cal_w,cal_p,test_w,test_p

@torch.inference_mode()
def logits(model,x,vocab,batch=2):
    return torch.cat([model(input_ids=x[i:i+batch]).logits[:,:-1,:vocab].float().cpu()
                      for i in range(0,len(x),batch)],0)

def metric(q,r,x):
    rl=F.log_softmax(r.float(),-1); ql=F.log_softmax(q.float(),-1)
    kl=(rl.exp()*(rl-ql)).sum(-1).mean().item()
    tar=x[:,1:].reshape(-1)
    nll=F.nll_loss(ql.reshape(-1,ql.shape[-1]),tar).item()
    return {"kl":float(kl),"nll":float(nll)}

def candidates(w):
    w=w.float().cpu(); rows,cols=w.shape; pad=(-cols)%GROUP
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
            out.append({"q":q.half(),"mse":float(F.mse_loss(q.float(),w).item()),
                        "tf":tf,"sm":sm})
    return out

def mods(model,n):
    return [model.model.layers[i].mlp.up_proj for i in range(n)]

def restore(ms,orig):
    for m,w in zip(ms,orig): m.weight.data.copy_(w.to(m.weight.dtype))

def apply(ms,sets,ch):
    for i,m in enumerate(ms): m.weight.data.copy_(sets[i][ch[i]]["q"].to(m.weight.dtype))

@torch.inference_mode()
def evm(model,x,ref,vocab): return metric(logits(model,x,vocab),ref,x)

def robust_score(model,cal_w,ref_w,cal_p,ref_p,vocab,beta):
    kw=evm(model,cal_w,ref_w,vocab)["kl"]
    kp=evm(model,cal_p,ref_p,vocab)["kl"]
    mean=0.5*(kw+kp)
    worst=max(kw,kp)
    return mean+beta*worst,kw,kp

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--layers",type=int,default=6)
    ap.add_argument("--beta",type=float,default=0.5)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    torch.manual_seed(0); torch.set_num_threads(min(os.cpu_count() or 1,8))
    name="Qwen/Qwen3-1.7B"
    tok=AutoTokenizer.from_pretrained(name); vocab=len(tok.get_vocab())
    model=AutoModelForCausalLM.from_pretrained(name,dtype=torch.float32,low_cpu_mem_usage=True); model.eval()
    cal_w,cal_p,test_w,test_p=load(tok)
    rcw=logits(model,cal_w,vocab).half(); rcp=logits(model,cal_p,vocab).half()
    rtw=logits(model,test_w,vocab).half(); rtp=logits(model,test_p,vocab).half()
    ms=mods(model,a.layers); orig=[m.weight.detach().cpu().clone() for m in ms]
    sets=[candidates(w) for w in orig]; ncan=len(sets[0])

    local=[min(range(ncan),key=lambda j:sets[i][j]["mse"]) for i in range(a.layers)]

    independent=[]
    for i in range(a.layers):
        vals=[]
        for j in range(ncan):
            restore(ms,orig)
            ms[i].weight.data.copy_(sets[i][j]["q"].to(ms[i].weight.dtype))
            s,kw,kp=robust_score(model,cal_w,rcw,cal_p,rcp,vocab,a.beta)
            vals.append(s)
            print("independent",i,j,s,kw,kp,flush=True)
        independent.append(min(range(ncan),key=lambda j:vals[j]))
    restore(ms,orig)

    fes=list(independent); hist=[]
    for p in range(3):
        changed=False
        for i in range(a.layers):
            vals=[]
            for j in range(ncan):
                t=list(fes); t[i]=j; apply(ms,sets,t)
                vals.append(robust_score(model,cal_w,rcw,cal_p,rcp,vocab,a.beta))
            b=min(range(ncan),key=lambda j:vals[j][0])
            changed|=(b!=fes[i]); fes[i]=b
            hist.append({"pass":p,"layer":i,"choice":b,"score":vals[b][0],"wiki_cal":vals[b][1],"pile_cal":vals[b][2]})
            print("coord",p,i,b,*vals[b],flush=True)
        if not changed: break
    restore(ms,orig)

    def evaluate(ch):
        apply(ms,sets,ch)
        return {"choices":ch,
                "weight_mse":sum(sets[i][j]["mse"] for i,j in enumerate(ch))/len(ch),
                "wiki":evm(model,test_w,rtw,vocab),
                "pile":evm(model,test_p,rtp,vocab)}
    outm={"local":evaluate(local),"independent":evaluate(independent),"fes":evaluate(fes)}
    f=outm["fes"]; ind=outm["independent"]; loc=outm["local"]
    s={"beta":a.beta,"layers":a.layers,"fes_differs":fes!=independent,
       "wiki_fes_ind":f["wiki"]["kl"]/ind["wiki"]["kl"],
       "pile_fes_ind":f["pile"]["kl"]/ind["pile"]["kl"],
       "wiki_fes_local":f["wiki"]["kl"]/loc["wiki"]["kl"],
       "pile_fes_local":f["pile"]["kl"]/loc["pile"]["kl"],
       "wiki_nll_fes_local":f["wiki"]["nll"]/loc["wiki"]["nll"],
       "pile_nll_fes_local":f["pile"]["nll"]/loc["pile"]["nll"],
       "weight_mse_fes_local":f["weight_mse"]/loc["weight_mse"]}
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps({"summary":s,"methods":outm,"history":hist},indent=2))
    print("SUMMARY",json.dumps(s,sort_keys=True),flush=True)

if __name__=="__main__": main()
