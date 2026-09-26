import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID="prism-ml/Bonsai-1.7B-unpacked"
LAYER=int(os.environ["CASE_LAYER"])
FOLD=int(os.environ["CASE_FOLD"])
FRAC=0.50
NUM_MASKS=16
ANCHORS=[0,5,10,15]
POOL_N=24
SELECT_N=6
HOLD_N=16
MAX_LENGTH=20
BASE_SKIP=1500
FOLD_STRIDE=50
GROUP_SIZE=128
OUT=Path("results-active-calibration")


def load_texts():
    ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
    xs=[]; eligible=0
    start=BASE_SKIP+FOLD*FOLD_STRIDE
    need=POOL_N+HOLD_N
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
    return xs[:POOL_N],xs[POOL_N:]


def enc(tok,t):
    return tok(t,return_tensors="pt",truncation=True,max_length=MAX_LENGTH)


@torch.no_grad()
def losses(model,tok,texts):
    out=[]
    for t in texts:
        x=enc(tok,t)
        out.append(float(model(**x,labels=x["input_ids"]).loss))
    return np.asarray(out,dtype=np.float64)


def mask_weight(w,code):
    x=w.detach().clone()
    g=torch.Generator(device="cpu")
    g.manual_seed(246813579+code*1000003+LAYER*1009)
    m=torch.rand(x.shape,generator=g)<FRAC
    x[m]=0
    return x


def metric_from_delta(d):
    d=np.asarray(d,dtype=np.float64)
    return {
      "mean_delta_nll":float(d.mean()),
      "mean_positive_delta_nll":float(np.maximum(d,0).mean()),
      "mean_abs_delta_nll":float(np.abs(d).mean()),
      "max_delta_nll":float(d.max()),
    }


def key(m):
    return (m["mean_positive_delta_nll"],m["mean_delta_nll"],m["max_delta_nll"])


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    pool,hold=load_texts()

    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    model=AutoModelForCausalLM.from_pretrained(
      MODEL_ID,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    target=model.model.layers[LAYER].mlp.down_proj.weight
    orig=target.detach().clone()

    with torch.no_grad():
        target.copy_(orig)
    pool_fp=losses(model,tok,pool)
    hold_fp=losses(model,tok,hold)

    # Cheap probe: a few anchor masks estimate which prompts expose compression differences.
    anchor_delta=[]
    for code in ANCHORS:
        with torch.no_grad():
            target.copy_(mask_weight(orig,code))
        anchor_delta.append(losses(model,tok,pool)-pool_fp)
    A=np.stack(anchor_delta,axis=0)

    # Diagnostic score rewards both variability across masks and consistent positive harm.
    susceptibility=np.std(A,axis=0)+np.mean(np.maximum(A,0),axis=0)
    active_idx=np.argsort(susceptibility)[-SELECT_N:][::-1].tolist()

    rng=random.Random(99173+LAYER*31+FOLD)
    remaining=list(range(POOL_N))
    rng.shuffle(remaining)
    random_idx=remaining[:SELECT_N]

    # Evaluate all mask codes only on the selected diagnostic prompts and random-control prompts.
    cal_active=[]
    cal_random=[]
    for code in range(NUM_MASKS):
        with torch.no_grad():
            target.copy_(mask_weight(orig,code))
        union_idx=sorted(set(active_idx+random_idx))
        vals=losses(model,tok,[pool[i] for i in union_idx])
        d=vals-pool_fp[union_idx]
        pos={idx:k for k,idx in enumerate(union_idx)}
        da=np.asarray([d[pos[i]] for i in active_idx])
        dr=np.asarray([d[pos[i]] for i in random_idx])
        cal_active.append({"code":code,"metric":metric_from_delta(da)})
        cal_random.append({"code":code,"metric":metric_from_delta(dr)})
        print("CAL",code,key(cal_active[-1]["metric"]),key(cal_random[-1]["metric"]),flush=True)

    active_best=min(cal_active,key=lambda x:key(x["metric"]))["code"]
    random_best=min(cal_random,key=lambda x:key(x["metric"]))["code"]

    # Holdout oracle included only for research diagnostics.
    hold_rows=[]
    for code in range(NUM_MASKS):
        with torch.no_grad():
            target.copy_(mask_weight(orig,code))
        d=losses(model,tok,hold)-hold_fp
        hold_rows.append({"code":code,"metric":metric_from_delta(d)})
    oracle=min(hold_rows,key=lambda x:key(x["metric"]))["code"]

    def hm(code):
        return next(x["metric"] for x in hold_rows if x["code"]==code)

    with torch.no_grad():
        target.copy_(orig)

    code_bits=float(np.ceil(np.log2(NUM_MASKS)))/orig.numel()
    storage=(1.0-FRAC)+16.0/GROUP_SIZE+code_bits

    payload={
      "model":MODEL_ID,"layer":LAYER,"fold":FOLD,"prune_fraction":FRAC,
      "anchor_codes":ANCHORS,"pool_n":POOL_N,"selected_n":SELECT_N,"holdout_n":HOLD_N,
      "susceptibility":susceptibility.tolist(),
      "active_indices":active_idx,"random_indices":random_idx,
      "active_best_code":active_best,"random_best_code":random_best,"oracle_code":oracle,
      "calibration_active":cal_active,"calibration_random":cal_random,
      "holdout_rows":hold_rows,
      "summary":{
        "code0_positive":hm(0)["mean_positive_delta_nll"],
        "active_positive":hm(active_best)["mean_positive_delta_nll"],
        "random_positive":hm(random_best)["mean_positive_delta_nll"],
        "oracle_positive":hm(oracle)["mean_positive_delta_nll"],
        "code0_mean":hm(0)["mean_delta_nll"],
        "active_mean":hm(active_best)["mean_delta_nll"],
        "random_mean":hm(random_best)["mean_delta_nll"],
        "oracle_mean":hm(oracle)["mean_delta_nll"],
      },
      "storage_bpw":float(storage),
      "prompt_eval_budget_deployment":{
        "anchor_pool":len(ANCHORS)*POOL_N,
        "candidate_selected":NUM_MASKS*SELECT_N,
        "total":len(ANCHORS)*POOL_N+NUM_MASKS*SELECT_N,
      }
    }
    (OUT/f"l{LAYER}_f{FOLD}.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload["summary"]|{
      "active_best_code":active_best,"random_best_code":random_best,"oracle_code":oracle,
      "active_indices":active_idx,"random_indices":random_idx,
    },indent=2))


if __name__=="__main__":
    main()
