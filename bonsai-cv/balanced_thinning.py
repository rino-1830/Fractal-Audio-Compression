import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID="prism-ml/Bonsai-1.7B-unpacked"
LAYER=int(os.environ["CASE_LAYER"])
FOLD=int(os.environ["CASE_FOLD"])
G=128
KEEP=64
N=24
MAX_LENGTH=24
BASE_SKIP=1250
FOLD_STRIDE=35
OUT=Path("results-balanced-thinning")


def load_texts():
    ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
    xs=[]; eligible=0
    start=BASE_SKIP+FOLD*FOLD_STRIDE
    for row in ds:
        t=" ".join(row["text"].split())
        if len(t)<100 or t.startswith("="):
            continue
        if eligible<start:
            eligible+=1
            continue
        xs.append(t[:500])
        if len(xs)>=N:
            break
    if len(xs)<N:
        raise RuntimeError("not enough text")
    return xs


def enc(tok,t):
    return tok(t,return_tensors="pt",truncation=True,max_length=MAX_LENGTH)


@torch.no_grad()
def losses(model,tok,texts):
    out=[]
    for t in texts:
        x=enc(tok,t)
        out.append(float(model(**x,labels=x["input_ids"]).loss))
    return np.asarray(out,dtype=np.float64)


def metric(fp,v):
    d=v-fp
    return {
      "mean_delta_nll":float(d.mean()),
      "mean_positive_delta_nll":float(np.maximum(d,0).mean()),
      "max_delta_nll":float(d.max()),
      "fraction_worse":float(np.mean(d>0)),
      "per_sample_delta":d.tolist(),
    }


def bernoulli_sparse(w):
    gen=torch.Generator(device="cpu")
    gen.manual_seed(246813579+LAYER*1009)
    mask=torch.rand(w.shape,generator=gen)<0.5
    x=w.clone(); x[mask]=0
    return x


def balanced_sparse(w):
    flat=w.reshape(-1)
    assert flat.numel()%G==0
    ng=flat.numel()//G
    gen=torch.Generator(device="cpu")
    gen.manual_seed(864209753+LAYER*1009)
    # Exactly KEEP positions retained in every g128 block.
    scores=torch.rand((ng,G),generator=gen)
    idx=torch.topk(scores,k=KEEP,dim=1,largest=False).indices
    keep=torch.zeros((ng,G),dtype=torch.bool)
    keep.scatter_(1,idx,True)
    x=flat.reshape(ng,G).clone()
    x[~keep]=0
    return x.reshape_as(w)


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    texts=load_texts()
    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    model=AutoModelForCausalLM.from_pretrained(
      MODEL_ID,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True
    )
    model.eval()
    target=model.model.layers[LAYER].mlp.down_proj.weight
    orig=target.detach().clone()

    with torch.no_grad(): target.copy_(orig)
    fp=losses(model,tok,texts)

    bern=bernoulli_sparse(orig)
    bal=balanced_sparse(orig)
    a=math.sqrt(2.0)
    variants={
      "bernoulli_unscaled":bern,
      "bernoulli_sqrt":bern.float()*a,
      "balanced_unscaled":bal,
      "balanced_sqrt":bal.float()*a,
    }
    results={}
    for name,w in variants.items():
        with torch.no_grad(): target.copy_(w.to(target.dtype))
        results[name]=metric(fp,losses(model,tok,texts))
        print(name,results[name],flush=True)

    with torch.no_grad(): target.copy_(orig)
    payload={
      "model":MODEL_ID,"layer":LAYER,"fold":FOLD,
      "group_size":G,"keep_per_group":KEEP,"n":N,
      "results":results,"storage_bpw":KEEP/G+16.0/G,
    }
    (OUT/f"l{LAYER}_f{FOLD}.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
