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
FRAC=0.50
CODE=0
N=24
MAX_LENGTH=24
BASE_SKIP=600
FOLD_STRIDE=35
GROUP_SIZE=128
OUT=Path("results-scale-compensation")


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
    g=torch.Generator(device="cpu")
    g.manual_seed(246813579+CODE*1000003+LAYER*1009)
    mask=torch.rand(orig.shape,generator=g)<FRAC
    base=orig.clone()
    base[mask]=0

    with torch.no_grad(): target.copy_(orig)
    fp=losses(model,tok,texts)

    q=1.0-FRAC
    scales={
      "unscaled":1.0,
      "sqrt_comp":1.0/math.sqrt(q),
      "mean_comp":1.0/q,
    }
    results={}
    for name,a in scales.items():
        with torch.no_grad():
            target.copy_((base.float()*a).to(target.dtype))
        results[name]=metric(fp,losses(model,tok,texts))
        print(name,results[name],flush=True)

    with torch.no_grad(): target.copy_(orig)

    payload={
      "model":MODEL_ID,"layer":LAYER,"fold":FOLD,"prune_fraction":FRAC,
      "code":CODE,"n":N,"results":results,
      "storage_bpw":float((1.0-FRAC)+16.0/GROUP_SIZE),
    }
    (OUT/f"l{LAYER}_f{FOLD}.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
