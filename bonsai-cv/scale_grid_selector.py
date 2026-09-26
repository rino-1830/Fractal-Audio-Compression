import json
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
ALPHAS=[0.8,1.0,1.2,1.4,1.6,1.8,2.0]
CAL_N=4
HOLD_N=16
MAX_LENGTH=24
BASE_SKIP=850
FOLD_STRIDE=35
GROUP_SIZE=128
OUT=Path("results-scale-grid")


def load_texts():
    ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
    xs=[]; eligible=0
    start=BASE_SKIP+FOLD*FOLD_STRIDE
    need=CAL_N+HOLD_N
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
    return xs[:CAL_N],xs[CAL_N:]


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
      "per_sample_delta":d.tolist(),
    }


def key(m):
    return (m["mean_positive_delta_nll"],m["mean_delta_nll"],m["max_delta_nll"])


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    cal,hold=load_texts()
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
    sparse=orig.clone(); sparse[mask]=0

    with torch.no_grad(): target.copy_(orig)
    cal_fp=losses(model,tok,cal)
    hold_fp=losses(model,tok,hold)

    cal_rows=[]
    for a in ALPHAS:
        with torch.no_grad(): target.copy_((sparse.float()*a).to(target.dtype))
        m=metric(cal_fp,losses(model,tok,cal))
        cal_rows.append({"alpha":a,"metric":m})
        print("CAL",a,key(m),flush=True)
    best=min(cal_rows,key=lambda x:key(x["metric"]))["alpha"]

    compare=sorted(set([1.0,2**0.5,best]))
    hold_rows={}
    for a in compare:
        with torch.no_grad(): target.copy_((sparse.float()*a).to(target.dtype))
        hold_rows[str(a)]=metric(hold_fp,losses(model,tok,hold))
        print("HOLD",a,hold_rows[str(a)],flush=True)

    with torch.no_grad(): target.copy_(orig)

    # One FP16 alpha per layer is negligible.
    alpha_bpw=16.0/orig.numel()
    storage=(1.0-FRAC)+16.0/GROUP_SIZE+alpha_bpw

    payload={
      "model":MODEL_ID,"layer":LAYER,"fold":FOLD,"prune_fraction":FRAC,
      "alphas":ALPHAS,"calibration":cal_rows,"best_alpha":best,
      "holdout":hold_rows,"storage_bpw":float(storage),
      "summary":{
        "unscaled_positive":hold_rows["1.0"]["mean_positive_delta_nll"],
        "sqrt_positive":hold_rows[str(2**0.5)]["mean_positive_delta_nll"],
        "selected_positive":hold_rows[str(best)]["mean_positive_delta_nll"],
        "unscaled_mean":hold_rows["1.0"]["mean_delta_nll"],
        "sqrt_mean":hold_rows[str(2**0.5)]["mean_delta_nll"],
        "selected_mean":hold_rows[str(best)]["mean_delta_nll"],
      }
    }
    (OUT/f"l{LAYER}_f{FOLD}.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
