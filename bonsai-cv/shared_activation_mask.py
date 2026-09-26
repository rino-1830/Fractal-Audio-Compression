import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "prism-ml/Bonsai-1.7B-unpacked"
LAYER = int(os.environ["CASE_LAYER"])
MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
RATES = [0.25, 0.50]
GROUP = 128
CAL_N = 16
HOLD_N = 16
MAX_LENGTH = 24
OUT = Path("results-shared-activation-mask")


def collect(split, skip, n):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    xs=[]; eligible=0
    for row in ds:
        t=" ".join(row["text"].split())
        if len(t)<100 or t.startswith("="):
            continue
        if eligible<skip:
            eligible += 1
            continue
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


def get_module(block,name):
    if name in {"q_proj","k_proj","v_proj","o_proj"}:
        return getattr(block.self_attn,name)
    return getattr(block.mlp,name)


def collect_activation_energy(model,tok,texts,modules):
    sums={}
    counts={}
    handles=[]
    for name,mod in modules.items():
        sums[name]=torch.zeros(mod.weight.shape[1],dtype=torch.float64)
        counts[name]=0
        def make_hook(k):
            def hook(module,args):
                x=args[0].detach().float().reshape(-1,args[0].shape[-1])
                sums[k] += x.square().sum(dim=0).double().cpu()
                counts[k] += x.shape[0]
            return hook
        handles.append(mod.register_forward_pre_hook(make_hook(name)))
    try:
        with torch.no_grad():
            for t in texts:
                x=enc(tok,t)
                model(**x)
    finally:
        for h in handles: h.remove()
    return {k:sums[k]/max(counts[k],1) for k in sums}


def shared_mask_weight(orig, act2, rate, mode, layer, name):
    rows,cols=orig.shape
    if cols % GROUP:
        raise RuntimeError(f"{name}: cols {cols} not divisible by {GROUP}")
    blocks=cols//GROUP
    prune_n=int(round(rate*GROUP))
    keep_n=GROUP-prune_n

    if mode=="activation":
        a=act2.reshape(blocks,GROUP)
        # keep highest activation-energy positions in each input group.
        idx=torch.topk(a,k=keep_n,dim=-1,largest=True,sorted=False).indices
        keep=torch.zeros((blocks,GROUP),dtype=torch.bool)
        keep.scatter_(1,idx,True)
    elif mode=="random":
        keep=torch.zeros((blocks,GROUP),dtype=torch.bool)
        gen=torch.Generator(device="cpu")
        gen.manual_seed(60606060 + layer*1009 + MODULES.index(name)*100003 + int(rate*1000))
        for b in range(blocks):
            perm=torch.randperm(GROUP,generator=gen)
            keep[b,perm[:keep_n]]=True
    else:
        raise ValueError(mode)

    # Shared across all output rows: metadata is one bitmap per input group.
    x=orig.detach().float().reshape(rows,blocks,GROUP).clone()
    x[:,~keep] = 0
    x *= 1.0/math.sqrt(1.0-rate)
    return x.reshape_as(orig).to(orig.dtype), keep


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    cal=collect("train",18000 + LAYER*20,CAL_N)
    hold=collect("train",23000 + LAYER*20,HOLD_N)

    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    model=AutoModelForCausalLM.from_pretrained(
      MODEL_ID,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True
    )
    model.eval()

    block=model.model.layers[LAYER]
    mods={name:get_module(block,name) for name in MODULES}
    originals={name:mod.weight.detach().clone() for name,mod in mods.items()}

    with torch.no_grad():
        for name,mod in mods.items(): mod.weight.copy_(originals[name])
    hold_fp=losses(model,tok,hold)

    act2=collect_activation_energy(model,tok,cal,mods)

    rows=[]
    for name in MODULES:
        mod=mods[name]
        orig=originals[name]
        out_dim,in_dim=orig.shape
        for rate in RATES:
            entry={
              "layer":LAYER,"module":name,"rate":rate,
              "shape":[out_dim,in_dim],
              "raw_bitmap_metadata_bpw":float(1.0/out_dim),
              "sign_plus_scale_plus_rawmask_bpw":float((1.0-rate)+16.0/GROUP+1.0/out_dim),
            }
            for mode in ["random","activation"]:
                sparse,keep=shared_mask_weight(orig,act2[name],rate,mode,LAYER,name)
                with torch.no_grad(): mod.weight.copy_(sparse)
                m=metric(hold_fp,losses(model,tok,hold))
                entry[mode]=m
                print("ROW",LAYER,name,rate,mode,m["mean_positive_delta_nll"],m["mean_delta_nll"],flush=True)
                with torch.no_grad(): mod.weight.copy_(orig)

            # theoretical diagonal proxy improvement in removed activation energy.
            blocks=in_dim//GROUP
            a=act2[name].reshape(blocks,GROUP)
            prune_n=int(round(rate*GROUP))
            low=torch.topk(a,k=prune_n,dim=-1,largest=False).values.sum().item()
            total=a.sum().item()
            entry["activation_energy_pruned_fraction"]=float(low/max(total,1e-30))
            entry["uniform_expected_pruned_fraction"]=float(rate)
            rows.append(entry)

    with torch.no_grad():
        for name,mod in mods.items(): mod.weight.copy_(originals[name])

    aggregate={}
    for rate in RATES:
        subset=[r for r in rows if r["rate"]==rate]
        aggregate[str(rate)]={
          "random_mean_positive":float(np.mean([r["random"]["mean_positive_delta_nll"] for r in subset])),
          "activation_mean_positive":float(np.mean([r["activation"]["mean_positive_delta_nll"] for r in subset])),
          "random_mean_delta":float(np.mean([r["random"]["mean_delta_nll"] for r in subset])),
          "activation_mean_delta":float(np.mean([r["activation"]["mean_delta_nll"] for r in subset])),
        }

    payload={
      "model":MODEL_ID,"layer":LAYER,"group":GROUP,
      "calibration_n":CAL_N,"holdout_n":HOLD_N,
      "rows":rows,"aggregate":aggregate
    }
    (OUT/f"layer{LAYER}.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
