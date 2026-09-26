import argparse,json,math,os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer

GROUP=128
TFS=(0.35,0.55,0.75)
SMS=(0.92,1.0,1.08)

def windows(tok,text,seq,n,offset,extra=31):
    ids=tok(text,return_tensors="pt",add_special_tokens=False)["input_ids"][0]
    out=[];p=offset
    for _ in range(n):
        if p+seq>ids.numel(): p=0
        out.append(ids[p:p+seq].clone());p+=seq+extra
    return torch.stack(out)

def load(tok):
    from datasets import load_dataset
    va=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="validation")
    te=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
    pi=load_dataset("NeelNanda/pile-10k",split="train")
    vat="\n\n".join(x["text"] for x in va if x["text"].strip())
    tet="\n\n".join(x["text"] for x in te if x["text"].strip())
    pit="\n\n".join(x for x in pi[:768]["text"] if x.strip())
    shards=[
      ("wiki0",windows(tok,vat,24,2,0)),
      ("wiki4k",windows(tok,vat,24,2,4096)),
      ("pile0",windows(tok,pit,24,2,0)),
      ("pile4k",windows(tok,pit,24,2,4096)),
    ]
    tests={
      "wiki":windows(tok,tet,24,16,997),
      "pile":windows(tok,pit,24,16,16000),
    }
    return shards,tests

@torch.inference_mode()
def logits(m,x,v,b=2):
    return torch.cat([m(input_ids=x[i:i+b]).logits[:,:-1,:v].float().cpu()
                      for i in range(0,len(x),b)],0)

def metric(q,r,x):
    rl=F.log_softmax(r.float(),-1);ql=F.log_softmax(q.float(),-1)
    kl=(rl.exp()*(rl-ql)).sum(-1).mean().item()
    tar=x[:,1:].reshape(-1)
    nll=F.nll_loss(ql.reshape(-1,ql.shape[-1]),tar).item()
    return {"kl":float(kl),"nll":float(nll)}

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
            out.append({"q":q.half(),"mse":float(F.mse_loss(q.float(),w).item()),
                        "tf":tf,"sm":sm})
    return out

def mods(m,n):return [m.model.layers[i].mlp.up_proj for i in range(n)]
def apply(ms,sets,ch):
    for i,m in enumerate(ms):m.weight.data.copy_(sets[i][ch[i]]["q"].to(m.weight.dtype))

@torch.inference_mode()
def kl(m,x,ref,v):return metric(logits(m,x,v),ref,x)["kl"]

def evaluate_shards(m,shards,refs,v):
    return [kl(m,x,refs[name],v) for name,x in shards]

def objective(vals,base,tau,beta):
    rel=[v/max(b,1e-30)-1.0 for v,b in zip(vals,base)]
    worst=max(rel);mean=sum(rel)/len(rel)
    feasible=worst<=tau
    # Feasible solutions rank by robust relative regret. Infeasible ones get a
    # large barrier while retaining ordering for diagnostics.
    score=mean+beta*worst+(0 if feasible else 1000*(worst-tau))
    return score,rel,feasible

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tau",type=float,required=True)
    ap.add_argument("--beta",type=float,default=1.0)
    ap.add_argument("--layers",type=int,default=3)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    torch.manual_seed(0);torch.set_num_threads(min(os.cpu_count() or 1,8))
    name="Qwen/Qwen3-1.7B"
    tok=AutoTokenizer.from_pretrained(name);v=len(tok.get_vocab())
    m=AutoModelForCausalLM.from_pretrained(name,dtype=torch.float32,low_cpu_mem_usage=True);m.eval()
    shards,tests=load(tok)
    refs={n:logits(m,x,v).half() for n,x in shards}
    test_refs={n:logits(m,x,v).half() for n,x in tests.items()}

    ms=mods(m,a.layers);orig=[x.weight.detach().cpu().clone() for x in ms]
    sets=[candidates(w) for w in orig]
    local=[min(range(len(sets[i])),key=lambda j:sets[i][j]["mse"]) for i in range(a.layers)]
    apply(ms,sets,local)
    base_vals=evaluate_shards(m,shards,refs,v)
    print("local_shards",base_vals,flush=True)

    ch=list(local);history=[]
    for p in range(3):
        changed=False
        for i in range(a.layers):
            cand=[]
            for j in range(len(sets[i])):
                t=list(ch);t[i]=j;apply(ms,sets,t)
                vals=evaluate_shards(m,shards,refs,v)
                sc,rel,feas=objective(vals,base_vals,a.tau,a.beta)
                cand.append((sc,j,vals,rel,feas))
            cand.sort(key=lambda z:z[0])
            _,best,vals,rel,feas=cand[0]
            changed|=(best!=ch[i]);ch[i]=best
            history.append({"pass":p,"layer":i,"choice":best,
                            "score":cand[0][0],"vals":vals,"rel":rel,"feasible":feas})
            print("coord",p,i,best,cand[0][0],rel,feas,flush=True)
        if not changed:break

    def ev(choice):
        apply(ms,sets,choice)
        return {"choices":list(choice),
                "weight_mse":sum(sets[i][j]["mse"] for i,j in enumerate(choice))/len(choice),
                "wiki":metric(logits(m,tests["wiki"],v),test_refs["wiki"],tests["wiki"]),
                "pile":metric(logits(m,tests["pile"],v),test_refs["pile"],tests["pile"]),
                "shards":evaluate_shards(m,shards,refs,v)}
    methods={"local":ev(local),"fes":ev(ch)}
    l=methods["local"];f=methods["fes"]
    summary={"tau":a.tau,"beta":a.beta,"layers":a.layers,"fes_differs":ch!=local,
      "wiki_kl_fes_local":f["wiki"]["kl"]/l["wiki"]["kl"],
      "pile_kl_fes_local":f["pile"]["kl"]/l["pile"]["kl"],
      "wiki_nll_fes_local":f["wiki"]["nll"]/l["wiki"]["nll"],
      "pile_nll_fes_local":f["pile"]["nll"]/l["pile"]["nll"],
      "weight_mse_fes_local":f["weight_mse"]/l["weight_mse"],
      "max_cal_relative_regret":max(v/max(b,1e-30)-1 for v,b in zip(f["shards"],base_vals))}
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps({"summary":summary,"methods":methods,"history":history},indent=2))
    print("SUMMARY",json.dumps(summary,sort_keys=True),flush=True)

if __name__=="__main__":main()
