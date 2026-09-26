import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID="prism-ml/Bonsai-1.7B-unpacked"
CASE=os.environ["CASE_SET"]  # five or all
FOLD=int(os.environ["CASE_FOLD"])
FRAC=0.50
N=24
MAX_LENGTH=24
BASE_SKIP=950
FOLD_STRIDE=35
GROUP_SIZE=128
OUT=Path("results-scale-composition")


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

    layers=[0,7,14,21,27] if CASE=="five" else list(range(len(model.model.layers)))
    targets={l:model.model.layers[l].mlp.down_proj.weight for l in layers}
    originals={l:targets[l].detach().clone() for l in layers}

    masks={}
    sparse={}
    for l in layers:
        g=torch.Generator(device="cpu")
        g.manual_seed(246813579+l*1009)
        m=torch.rand(originals[l].shape,generator=g)<FRAC
        masks[l]=m
        w=originals[l].clone(); w[m]=0
        sparse[l]=w

    def restore():
        with torch.no_grad():
            for l in layers:
                targets[l].copy_(originals[l])

    def apply(alpha):
        with torch.no_grad():
            for l in layers:
                targets[l].copy_((sparse[l].float()*alpha).to(targets[l].dtype))

    restore()
    fp=losses(model,tok,texts)

    results={}
    for name,a in {
        "unscaled":1.0,
        "sqrt_comp":1.0/math.sqrt(1.0-FRAC),
        "mean_comp":1.0/(1.0-FRAC),
    }.items():
        apply(a)
        results[name]=metric(fp,losses(model,tok,texts))
        print(name,results[name],flush=True)

    restore()

    # Whole-model bit saving attributable to these matrices.
    tested_weights=sum(w.numel() for w in originals.values())
    total_weights=sum(p.numel() for p in model.parameters())
    tested_fraction=tested_weights/total_weights
    local_bpw=(1.0-FRAC)+16.0/GROUP_SIZE

    payload={
      "model":MODEL_ID,"case":CASE,"fold":FOLD,"layers":layers,
      "prune_fraction":FRAC,"n":N,"results":results,
      "tested_weights":tested_weights,"total_weights":total_weights,
      "tested_fraction":tested_fraction,
      "local_bpw":local_bpw,
      "global_bpw_saving_vs_1bit_sign_payload":FRAC*tested_fraction,
    }
    (OUT/f"{CASE}_f{FOLD}.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
