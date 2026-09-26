import json,math,os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer

TFS=(0.35,0.50,0.65,0.80,0.95)
SMS=(0.90,1.00,1.10)
LAYERS=(0,2,4)
CLUSTERS=12
TANGENT_RANK=8

def win(tok,text,seq,n,off=0,extra=19):
    ids=tok(text,return_tensors="pt",add_special_tokens=False)["input_ids"][0]
    out=[];p=off
    for _ in range(n):
        out.append(ids[p:p+seq].clone());p+=seq+extra
    return torch.stack(out)

def data(tok):
    from datasets import load_dataset
    va=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="validation")
    te=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
    vat="\n\n".join(x["text"] for x in va if x["text"].strip())
    tet="\n\n".join(x["text"] for x in te if x["text"].strip())
    return win(tok,vat,48,8),win(tok,tet,48,8,991)

@torch.inference_mode()
def forward(m,x,hidden=False,b=4):
    ls=[]; hs=None
    if hidden: hs=[[] for _ in range(len(m.gpt_neox.layers)+1)]
    for i in range(0,len(x),b):
        o=m(input_ids=x[i:i+b],output_hidden_states=hidden,use_cache=False)
        ls.append(o.logits[:,:-1].float().cpu())
        if hidden:
            for k,h in enumerate(o.hidden_states):
                hs[k].append(h.float().cpu())
    logits=torch.cat(ls,0)
    if hidden: return logits,[torch.cat(z,0) for z in hs]
    return logits

def kl(q,r):
    rl=F.log_softmax(r.float(),-1); ql=F.log_softmax(q.float(),-1)
    return float((rl.exp()*(rl-ql)).sum(-1).mean().item())

def candidates(w):
    w=w.float().cpu();ma=w.abs().mean().item()+1e-12;o=[]
    for tf in TFS:
        s=torch.sign(w)*(w.abs()>=tf*ma)
        den=(s*s).sum().item();base=ma if den==0 else (w*s).sum().item()/den
        for sm in SMS:
            q=(base*sm)*s
            o.append({"q":q.contiguous(),"weight_mse":float(F.mse_loss(q,w).item()),
                      "tf":tf,"sm":sm})
    return o

def simple_kmeans(x,k,iters=8):
    # deterministic far-spread-ish initialization
    idx=torch.linspace(0,x.shape[0]-1,k).long()
    c=x[idx].clone()
    for _ in range(iters):
        d=torch.cdist(x,c)
        a=d.argmin(1)
        nc=[]
        for j in range(k):
            pts=x[a==j]
            nc.append(pts.mean(0) if len(pts) else c[j])
        nc=torch.stack(nc)
        if torch.allclose(nc,c): break
        c=nc
    return c,a

def tangent_model(h):
    # h: [B,S,D]; cluster token states and estimate local tangent PCA from centered cluster.
    x=h.reshape(-1,h.shape[-1]).float()
    c,a=simple_kmeans(x,CLUSTERS)
    bases=[]
    for j in range(CLUSTERS):
        pts=x[a==j]
        if pts.shape[0]<TANGENT_RANK+2:
            # nearest points to centroid as fallback
            ids=torch.cdist(c[j:j+1],x).squeeze(0).topk(min(TANGENT_RANK+8,x.shape[0]),largest=False).indices
            pts=x[ids]
        z=pts-pts.mean(0,keepdim=True)
        # right singular vectors span local tangent
        _,_,vh=torch.linalg.svd(z,full_matrices=False)
        bases.append(vh[:min(TANGENT_RANK,vh.shape[0])].T.contiguous())
    return a,bases

def manifold_error(delta,assign,bases):
    x=delta.reshape(-1,delta.shape[-1]).float()
    normal=0.0;tangent=0.0;total=0.0;n=0
    for j,B in enumerate(bases):
        e=x[assign==j]
        if e.numel()==0: continue
        coeff=e@B
        te=(coeff*coeff).sum().item()
        tot=(e*e).sum().item()
        tangent+=te;normal+=max(tot-te,0.0);total+=tot;n+=e.shape[0]
    return {"activation_mse":total/max(n,1),
            "normal_energy":normal/max(n,1),
            "tangent_energy":tangent/max(n,1),
            "normal_fraction":normal/max(total,1e-30)}

def ranks(v):
    order=sorted(range(len(v)),key=lambda i:v[i])
    r=[0]*len(v)
    for rank,i in enumerate(order):r[i]=rank
    return r

def spearman(a,b):
    ra=ranks(a);rb=ranks(b);n=len(a)
    ma=sum(ra)/n;mb=sum(rb)/n
    num=sum((ra[i]-ma)*(rb[i]-mb) for i in range(n))
    da=math.sqrt(sum((x-ma)**2 for x in ra));db=math.sqrt(sum((x-mb)**2 for x in rb))
    return num/max(da*db,1e-30)

def main():
    torch.manual_seed(0);torch.set_num_threads(min(os.cpu_count() or 1,16))
    name="EleutherAI/pythia-70m"
    tok=AutoTokenizer.from_pretrained(name)
    m=AutoModelForCausalLM.from_pretrained(name,dtype=torch.float32,low_cpu_mem_usage=True);m.eval()
    cal,test=data(tok)
    fp_cal,fp_h=forward(m,cal,hidden=True)
    fp_test=forward(m,test)
    rows=[]

    for li in LAYERS:
        module=m.gpt_neox.layers[li].mlp.dense_h_to_4h
        orig=module.weight.detach().cpu().clone()
        cs=candidates(orig)
        # hidden_states[li+1] is post-block residual state
        assign,bases=tangent_model(fp_h[li+1])
        for ci,c in enumerate(cs):
            module.weight.data.copy_(c["q"].to(module.weight.dtype))
            qcal,qh=forward(m,cal,hidden=True)
            qtest=forward(m,test)
            me=manifold_error(qh[li+1]-fp_h[li+1],assign,bases)
            row={"layer":li,"candidate":ci,"tf":c["tf"],"sm":c["sm"],
                 "weight_mse":c["weight_mse"],
                 "cal_kl":kl(qcal,fp_cal),"test_kl":kl(qtest,fp_test),**me}
            rows.append(row)
            print("ROW",json.dumps(row,sort_keys=True),flush=True)
            module.weight.data.copy_(orig.to(module.weight.dtype))

    # rank-normalize within each layer by converting values to per-layer ranks,
    # then correlate pooled ranks across layers.
    metrics=["weight_mse","activation_mse","normal_energy","normal_fraction","tangent_energy","cal_kl"]
    pooled={k:[] for k in metrics}; y=[]
    for li in LAYERS:
        rr=[r for r in rows if r["layer"]==li]
        yrank=ranks([r["test_kl"] for r in rr]);y.extend(yrank)
        for k in metrics:
            pooled[k].extend(ranks([r[k] for r in rr]))
    corr={k:spearman(pooled[k],y) for k in metrics}
    summary={"model":name,"layers":LAYERS,"clusters":CLUSTERS,"tangent_rank":TANGENT_RANK,
             "spearman_to_test_kl":corr,
             "normal_beats_activation_mse":abs(corr["normal_energy"])>abs(corr["activation_mse"]),
             "normal_beats_weight_mse":abs(corr["normal_energy"])>abs(corr["weight_mse"])}
    out=Path("results-tes");out.mkdir(exist_ok=True)
    (out/"results.json").write_text(json.dumps({"summary":summary,"rows":rows},indent=2))
    print("SUMMARY",json.dumps(summary,sort_keys=True),flush=True)

if __name__=="__main__":main()
