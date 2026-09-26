import json
import math
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID="prism-ml/Bonsai-1.7B-unpacked"
GROUP=128
MAX_LENGTH=24
N=24
OUT=Path("results-family-schedule")

SCHEDULES={
  "core":{
    "q_proj":0.0,"k_proj":0.5,"v_proj":0.0,"o_proj":0.0,
    "gate_proj":0.0,"up_proj":0.5,"down_proj":0.5,
  },
  "safe":{
    "q_proj":0.25,"k_proj":0.5,"v_proj":0.0,"o_proj":0.25,
    "gate_proj":0.25,"up_proj":0.5,"down_proj":0.5,
  },
  "aggressive":{
    "q_proj":0.25,"k_proj":0.5,"v_proj":0.25,"o_proj":0.25,
    "gate_proj":0.5,"up_proj":0.5,"down_proj":0.5,
  },
}


def collect():
    ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="train")
    xs=[]; eligible=0
    for row in ds:
        t=" ".join(row["text"].split())
        if len(t)<100 or t.startswith("="):
            continue
        if eligible<14000:
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
    vals=[]
    for t in texts:
        x=enc(tok,t)
        vals.append(float(model(**x,labels=x["input_ids"]).loss))
    return np.asarray(vals,dtype=np.float64)


def metric(fp,v):
    d=v-fp
    return {
      "mean_delta_nll":float(d.mean()),
      "mean_positive_delta_nll":float(np.maximum(d,0).mean()),
      "median_delta_nll":float(np.median(d)),
      "p90_delta_nll":float(np.quantile(d,0.9)),
      "max_delta_nll":float(d.max()),
      "fraction_worse":float(np.mean(d>0)),
      "per_sample_delta":d.tolist(),
    }


def get_weight(block,name):
    if name in {"q_proj","k_proj","v_proj","o_proj"}:
        return getattr(block.self_attn,name).weight
    return getattr(block.mlp,name).weight


def thinned(orig,layer,name,p):
    if p<=0:
        return orig
    names=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    g=torch.Generator(device="cpu")
    g.manual_seed(975310864 + layer*1009 + names.index(name)*100003)
    x=orig.detach().clone()
    mask=torch.rand(x.shape,generator=g)<p
    x[mask]=0
    x=x.float()/math.sqrt(1.0-p)
    return x.to(orig.dtype)


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    texts=collect()
    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    model=AutoModelForCausalLM.from_pretrained(
      MODEL_ID,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True
    )
    model.eval()

    refs={}
    originals={}
    for li,b in enumerate(model.model.layers):
        for name in SCHEDULES["core"]:
            w=get_weight(b,name)
            refs[(li,name)]=w
            originals[(li,name)]=w.detach().clone()

    with torch.no_grad():
        for k,w in refs.items(): w.copy_(originals[k])
    fp=losses(model,tok,texts)

    total_model_weights=sum(p.numel() for p in model.parameters())
    targeted_weights=sum(w.numel() for w in originals.values())

    def restore():
        with torch.no_grad():
            for k,w in refs.items(): w.copy_(originals[k])

    results={}
    for sname,sched in SCHEDULES.items():
        restore()
        removed=0.0
        storage_bits=0.0
        with torch.no_grad():
            for (li,name),w in refs.items():
                p=sched[name]
                n=originals[(li,name)].numel()
                removed += p*n
                # sign payload + existing fp16 group scale
                storage_bits += n*((1.0-p)+16.0/GROUP)
                if p>0:
                    w.copy_(thinned(originals[(li,name)],li,name,p))
        vals=losses(model,tok,texts)
        results[sname]={
            "metric":metric(fp,vals),
            "targeted_linear_bpw":float(storage_bits/targeted_weights),
            "targeted_sign_prune_fraction":float(removed/targeted_weights),
            "global_sign_bpw_saving":float(removed/total_model_weights),
            "targeted_weight_fraction_of_model":float(targeted_weights/total_model_weights),
            "schedule":sched,
        }
        print("RESULT",sname,
              results[sname]["metric"]["mean_positive_delta_nll"],
              results[sname]["metric"]["mean_delta_nll"],
              results[sname]["targeted_linear_bpw"],flush=True)

    restore()
    payload={
      "model":MODEL_ID,
      "evaluation":"WikiText-2 test eligible offset 3300",
      "n":N,
      "results":results,
    }
    (OUT/"result.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
