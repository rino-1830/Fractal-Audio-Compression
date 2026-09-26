import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy.linalg import eigh
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID="prism-ml/Bonsai-1.7B-unpacked"
LAYER=int(os.environ["CASE_LAYER"])
FOLD=int(os.environ["CASE_FOLD"])
FRAC=0.50
NUM_MASKS=16
SELECT_N=6
BASIS_N=20
HOLD_N=16
FAIL_N=8
STABLE_N=8
K=8
RANKS=[2,4,8]
MAX_LENGTH=20
BASE_SKIP=2600
FOLD_STRIDE=60
GROUP_SIZE=128
OUT=Path("results-residual-rank-sweep")


def load_texts():
    ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
    xs=[]; eligible=0
    start=BASE_SKIP+FOLD*FOLD_STRIDE
    need=SELECT_N+BASIS_N+HOLD_N
    for row in ds:
        t=" ".join(row["text"].split())
        if len(t)<100 or t.startswith("="):
            continue
        if eligible<start:
            eligible+=1
            continue
        xs.append(t[:500])
        if len(xs)>=need:
            break
    if len(xs)<need:
        raise RuntimeError("not enough text")
    return xs[:SELECT_N],xs[SELECT_N:SELECT_N+BASIS_N],xs[SELECT_N+BASIS_N:]


def enc(tok,t):
    return tok(t,return_tensors="pt",truncation=True,max_length=MAX_LENGTH)


@torch.no_grad()
def losses(model,tok,texts):
    vals=[]
    for t in texts:
        x=enc(tok,t)
        vals.append(float(model(**x,labels=x["input_ids"]).loss))
    return np.asarray(vals,dtype=np.float64)


def mask_weight(w,code):
    x=w.detach().clone()
    g=torch.Generator(device="cpu")
    g.manual_seed(246813579+code*1000003+LAYER*1009)
    m=torch.rand(x.shape,generator=g)<FRAC
    x[m]=0
    return x


def metric(fp,vals):
    d=vals-fp
    return {
      "mean_delta_nll":float(d.mean()),
      "mean_positive_delta_nll":float(np.maximum(d,0).mean()),
      "max_delta_nll":float(d.max()),
      "fraction_worse":float(np.mean(d>0)),
      "per_sample_delta":d.tolist(),
    }


def key(m):
    return (m["mean_positive_delta_nll"],m["mean_delta_nll"],m["max_delta_nll"])


def grad(model,tok,target,text):
    model.zero_grad(set_to_none=True)
    x=enc(tok,text)
    loss=model(**x,labels=x["input_ids"]).loss
    g=torch.autograd.grad(loss,target,retain_graph=False,create_graph=False)[0]
    g=g.detach().float().reshape(-1)
    return g/g.norm().clamp_min(1e-12)


def orth(X):
    q,_=torch.linalg.qr(X,mode="reduced")
    return q


def failure_basis(G,nf,k):
    n=G.shape[0]
    kg=(G@G.T).double().cpu().numpy()
    fi=np.arange(nf); si=np.arange(nf,n)
    A=kg[:,fi]@kg[fi,:]
    B=kg[:,si]@kg[si,:]
    reg=max(float(np.trace(kg))/max(n,1),1e-8)*1e-2
    vals,vecs=eigh(A,B+reg*np.eye(n),check_finite=False)
    idx=np.argsort(vals)[::-1][:k]
    coeff=torch.from_numpy(vecs[:,idx]).to(dtype=G.dtype)
    return orth(G.T@coeff),vals[idx].tolist()


def low_rank(M,rank):
    q=min(rank+2,min(M.shape))
    U,S,V=torch.svd_lowrank(M,q=q,niter=2)
    return (U[:,:rank]*S[:rank])@V[:,:rank].T


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    sel_texts,basis_texts,hold_texts=load_texts()

    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    model=AutoModelForCausalLM.from_pretrained(
      MODEL_ID,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    target=model.model.layers[LAYER].mlp.down_proj.weight
    orig=target.detach().clone()
    target.requires_grad_(True)

    with torch.no_grad():
        target.copy_(orig)
    sel_fp=losses(model,tok,sel_texts)
    basis_fp=losses(model,tok,basis_texts)
    hold_fp=losses(model,tok,hold_texts)

    search=[]
    for code in range(NUM_MASKS):
        mw=mask_weight(orig,code)
        with torch.no_grad():
            target.copy_(mw)
        m=metric(sel_fp,losses(model,tok,sel_texts))
        search.append({"code":code,"selection":m})
        print("MASK",code,key(m),flush=True)
    best_code=min(search,key=lambda x:key(x["selection"]))["code"]
    selected=mask_weight(orig,best_code)

    with torch.no_grad():
        target.copy_(selected)
    basis_vals=losses(model,tok,basis_texts)
    base_hold=metric(hold_fp,losses(model,tok,hold_texts))
    bd=basis_vals-basis_fp
    fail_idx=np.argsort(bd)[-FAIL_N:][::-1]
    stable_idx=np.argsort(np.abs(bd))[:STABLE_N]

    with torch.no_grad():
        target.copy_(orig)
    G=torch.stack([grad(model,tok,target,basis_texts[int(i)]) for i in list(fail_idx)+list(stable_idx)])
    U,eigs=failure_basis(G,FAIL_N,K)

    residual=(selected.float()-orig.float()).reshape(-1)
    corr=(U@(U.T@residual)).reshape_as(orig)
    m,n=orig.shape
    index_bpw=float(np.ceil(np.log2(NUM_MASKS)))/orig.numel()
    base_bpw=(1.0-FRAC)+16.0/GROUP_SIZE+index_bpw

    results={}
    for rank in RANKS:
        lr=low_rank(corr,rank)
        corrected=selected.float()-lr
        with torch.no_grad():
            target.copy_(corrected.to(target.dtype))
        hm=metric(hold_fp,losses(model,tok,hold_texts))
        factor_bpw=16.0*rank*(m+n)/(m*n)
        results[str(rank)]={
          "holdout":hm,
          "factor_bpw":float(factor_bpw),
          "total_bpw":float(base_bpw+factor_bpw),
        }
        print("RANK",rank,hm["mean_positive_delta_nll"],hm["mean_delta_nll"],base_bpw+factor_bpw,flush=True)

    with torch.no_grad():
        target.copy_(orig)
    target.requires_grad_(False)

    payload={
      "model":MODEL_ID,"layer":LAYER,"fold":FOLD,"prune_fraction":FRAC,
      "best_code":best_code,"search":search,
      "failure_eigenvalues":eigs,
      "baseline_selected_holdout":base_hold,
      "base_bpw":float(base_bpw),
      "correction_energy_fraction":float(corr.pow(2).sum()/residual.pow(2).sum().clamp_min(1e-12)),
      "ranks":results,
      "settings":{
        "select_n":SELECT_N,"basis_n":BASIS_N,"holdout_n":HOLD_N,
        "fail_n":FAIL_N,"stable_n":STABLE_N,"basis_k":K,
      }
    }
    (OUT/f"l{LAYER}_f{FOLD}.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
