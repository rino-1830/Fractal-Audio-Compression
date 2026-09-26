import argparse,json,math,os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer

TFS=(0.35,0.50,0.65,0.80,0.95)
SMS=(0.90,1.00,1.10)

def make_windows(tok,text,seq,n,offset=0,extra=23):
    ids=tok(text,return_tensors="pt",add_special_tokens=False)["input_ids"][0]
    need=offset+n*(seq+extra)+seq
    if ids.numel()<need: ids=ids.repeat(math.ceil(need/ids.numel()))
    out=[];p=offset
    for _ in range(n):
        out.append(ids[p:p+seq].clone());p+=seq+extra
    return torch.stack(out)

def data(tok,seq=48,ncal=12,ntest=48,npile=48):
    from datasets import load_dataset
    va=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="validation")
    te=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
    pi=load_dataset("NeelNanda/pile-10k",split="train")
    vat="\n\n".join(x["text"] for x in va if x["text"].strip())
    tet="\n\n".join(x["text"] for x in te if x["text"].strip())
    pit="\n\n".join(x for x in pi[:256]["text"] if x.strip())
    return make_windows(tok,vat,seq,ncal,0),make_windows(tok,tet,seq,ntest,733),make_windows(tok,pit,seq,npile,271)

@torch.inference_mode()
def logits(model,x,batch):
    return torch.cat([model(input_ids=x[i:i+batch]).logits[:,:-1].float().cpu()
                      for i in range(0,len(x),batch)],0)

def metric(q,r,x):
    rl=F.log_softmax(r.float(),-1);ql=F.log_softmax(q.float(),-1)
    kl=(rl.exp()*(rl-ql)).sum(-1)
    target=x[:,1:].reshape(-1)
    nll=F.nll_loss(ql.reshape(-1,ql.shape[-1]),target,reduction="none").reshape(x.shape[0],-1).mean(-1)
    return {"kl":float(kl.mean().item()),"nll":float(nll.mean().item()),
            "kl_window":kl.mean(-1).tolist(),"nll_window":nll.tolist()}

def candidates(w):
    w=w.float().cpu();ma=w.abs().mean().item()+1e-12;out=[]
    for tf in TFS:
        s=torch.sign(w)*(w.abs()>=tf*ma)
        den=(s*s).sum().item();base=ma if den==0 else (w*s).sum().item()/den
        for sm in SMS:
            q=(base*sm)*s
            out.append({"q":q.contiguous(),"mse":float(F.mse_loss(q,w).item()),"tf":tf,"sm":sm})
    return out

def mods(model): return [l.mlp.dense_h_to_4h for l in model.gpt_neox.layers]
def restore(ms,orig):
    for m,w in zip(ms,orig):m.weight.data.copy_(w.to(m.weight.dtype))
def apply(ms,sets,ch):
    for i,m in enumerate(ms):m.weight.data.copy_(sets[i][ch[i]]["q"].to(m.weight.dtype))

@torch.inference_mode()
def eval_model(model,x,ref,batch):
    return metric(logits(model,x,batch),ref,x)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True)
    ap.add_argument("--passes",type=int,default=2)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    torch.manual_seed(0);torch.set_num_threads(min(os.cpu_count() or 1,16))
    tok=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.float32,low_cpu_mem_usage=True);model.eval()
    nparams=sum(p.numel() for p in model.parameters())
    batch=8 if nparams<100_000_000 else 4
    cal,wiki,pile=data(tok)
    ref_cal=logits(model,cal,batch).half()
    ref_wiki=logits(model,wiki,batch).half()
    ref_pile=logits(model,pile,batch).half()
    ms=mods(model);orig=[m.weight.detach().cpu().clone() for m in ms];sets=[candidates(w) for w in orig]

    local=[min(range(15),key=lambda j:sets[i][j]["mse"]) for i in range(len(ms))]
    independent=[]
    for i in range(len(ms)):
        vals=[]
        for j in range(15):
            restore(ms,orig);ms[i].weight.data.copy_(sets[i][j]["q"].to(ms[i].weight.dtype))
            vals.append(eval_model(model,cal,ref_cal,batch)["kl"])
        independent.append(min(range(15),key=lambda j:vals[j]))
        print("independent",i,independent[-1],min(vals),flush=True)
    restore(ms,orig)

    coord=list(independent);history=[]
    for p in range(args.passes):
        changed=False
        for i in range(len(ms)):
            vals=[]
            for j in range(15):
                trial=list(coord);trial[i]=j;apply(ms,sets,trial)
                vals.append(eval_model(model,cal,ref_cal,batch)["kl"])
            b=min(range(15),key=lambda j:vals[j])
            changed|=(b!=coord[i]);coord[i]=b;history.append([p,i,b,vals[b]])
            print("coord",p,i,b,vals[b],flush=True)
        if not changed: break
    restore(ms,orig)

    def ev(ch):
        apply(ms,sets,ch)
        return {"choices":ch,"weight_mse":sum(sets[i][ch[i]]["mse"] for i in range(len(ch)))/len(ch),
                "cal":eval_model(model,cal,ref_cal,batch),
                "wiki":eval_model(model,wiki,ref_wiki,batch),
                "pile":eval_model(model,pile,ref_pile,batch)}
    methods={"local":ev(local),"independent":ev(independent),"fes":ev(coord)}
    f=methods["fes"];ind=methods["independent"];loc=methods["local"]
    s={"model":args.model,"parameters":nparams,"layers":len(ms),"cal_tokens":int(cal.numel()),
       "wiki_tokens":int(wiki.numel()),"pile_tokens":int(pile.numel()),
       "fes_differs_from_independent":coord!=independent,
       "wiki_kl_fes_vs_independent":f["wiki"]["kl"]/ind["wiki"]["kl"],
       "pile_kl_fes_vs_independent":f["pile"]["kl"]/ind["pile"]["kl"],
       "wiki_kl_fes_vs_local":f["wiki"]["kl"]/loc["wiki"]["kl"],
       "pile_kl_fes_vs_local":f["pile"]["kl"]/loc["pile"]["kl"],
       "weight_mse_fes_vs_local":f["weight_mse"]/loc["weight_mse"]}
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps({"summary":s,"methods":methods,"history":history},indent=2))
    print("SUMMARY",json.dumps(s,sort_keys=True),flush=True)

if __name__=="__main__":main()
