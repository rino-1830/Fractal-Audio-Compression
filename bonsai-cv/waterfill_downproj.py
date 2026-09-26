import json
import math
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID="prism-ml/Bonsai-1.7B-unpacked"
P_LEVELS=[0.25,0.50]
TARGETS=[0.25,0.35]
CAL_N=4
HOLD_N=16
MAX_LENGTH=24
GROUP_SIZE=128
OUT=Path("results-waterfill-downproj")


def collect(split,skip,n):
    ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split=split)
    xs=[]; k=0
    for row in ds:
        t=" ".join(row["text"].split())
        if len(t)<100 or t.startswith("="):
            continue
        if k<skip:
            k+=1; continue
        xs.append(t[:500])
        if len(xs)>=n:
            break
    if len(xs)<n:
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


def score_tensor(shape,layer):
    g=torch.Generator(device="cpu")
    g.manual_seed(246813579+layer*1009)
    return torch.rand(shape,generator=g)


def thinned(orig,score,p):
    if p<=0:
        return orig
    x=orig.clone()
    x[score<p]=0
    x=x.float()*(1.0/math.sqrt(1.0-p))
    return x.to(orig.dtype)


def greedy_allocate(sens,target):
    # Each layer has two 0.25-rate increments. Enforce first before second.
    layers=sorted(sens)
    steps=int(round(target*len(layers)/0.25))
    state={l:0 for l in layers}  # 0,1,2 increments
    trace=[]
    for _ in range(steps):
        options=[]
        for l in layers:
            s=state[l]
            if s>=2: continue
            d25=sens[l]["0.25"]
            d50=sens[l]["0.5"]
            if s==0:
                cost=max(d25,0.0)
            else:
                cost=max(d50-d25,0.0)
            options.append((cost,l,s+1))
        cost,l,new= min(options,key=lambda x:(x[0],x[1]))
        state[l]=new
        trace.append({"layer":l,"new_increment":new,"marginal_cost":cost})
    rates={l:0.25*state[l] for l in layers}
    return rates,trace


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    cal=collect("validation",300,CAL_N)
    hold=collect("test",500,HOLD_N)

    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    model=AutoModelForCausalLM.from_pretrained(
      MODEL_ID,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True
    )
    model.eval()

    layers=list(range(len(model.model.layers)))
    targets={l:model.model.layers[l].mlp.down_proj.weight for l in layers}
    originals={l:targets[l].detach().clone() for l in layers}

    with torch.no_grad():
        for l in layers: targets[l].copy_(originals[l])
    cal_fp=losses(model,tok,cal)
    hold_fp=losses(model,tok,hold)

    sensitivity={}
    for l in layers:
        score=score_tensor(originals[l].shape,l)
        sensitivity[l]={}
        for p in P_LEVELS:
            with torch.no_grad():
                targets[l].copy_(thinned(originals[l],score,p))
            m=metric(cal_fp,losses(model,tok,cal))
            sensitivity[l][str(p)]=m["mean_positive_delta_nll"]
            print("SENS",l,p,m["mean_positive_delta_nll"],m["mean_delta_nll"],flush=True)
            with torch.no_grad():
                targets[l].copy_(originals[l])
        del score

    total_weights=sum(p.numel() for p in model.parameters())
    tested_weights=sum(w.numel() for w in originals.values())
    tested_fraction=tested_weights/total_weights

    def restore():
        with torch.no_grad():
            for l in layers: targets[l].copy_(originals[l])

    def apply_rates(rates):
        restore()
        with torch.no_grad():
            for l,p in rates.items():
                if p<=0: continue
                sc=score_tensor(originals[l].shape,l)
                targets[l].copy_(thinned(originals[l],sc,p))
                del sc

    results={}
    for target in TARGETS:
        rates,trace=greedy_allocate(sensitivity,target)
        actual=float(np.mean(list(rates.values())))
        apply_rates(rates)
        water=metric(hold_fp,losses(model,tok,hold))

        uniform={l:target for l in layers}
        apply_rates(uniform)
        uni=metric(hold_fp,losses(model,tok,hold))

        local_bpw=1.0-actual+16.0/GROUP_SIZE
        results[str(target)]={
          "waterfill_rates":rates,
          "allocation_trace":trace,
          "actual_mean_prune":actual,
          "waterfill_holdout":water,
          "uniform_holdout":uni,
          "downproj_bpw":local_bpw,
          "global_bpw_saving_from_signs":actual*tested_fraction,
        }
        print("TARGET",target,"actual",actual,"W",water,"U",uni,flush=True)

    restore()
    payload={
      "model":MODEL_ID,"layers":layers,
      "calibration_split":"validation","holdout_split":"test",
      "calibration_n":CAL_N,"holdout_n":HOLD_N,
      "sensitivity":sensitivity,
      "downproj_weight_fraction":tested_fraction,
      "results":results,
    }
    (OUT/"result.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
