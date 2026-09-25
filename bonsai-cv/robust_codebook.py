import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID="prism-ml/Bonsai-1.7B-unpacked"
LAYER=int(os.environ["CASE_LAYER"])
FRAC=0.50
NUM_MASKS=16
CAL_N=8
HOLD_N=16
MAX_LENGTH=20
SKIP_ELIGIBLE=1200
GROUP_SIZE=128
OUT=Path("results-robust-codebook")


def load_texts():
    ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
    xs=[]; skip=0
    for row in ds:
        t=" ".join(row["text"].split())
        if len(t)<100 or t.startswith("="): continue
        if skip<SKIP_ELIGIBLE:
            skip+=1; continue
        xs.append(t[:500])
        if len(xs)>=CAL_N+HOLD_N: break
    return xs[:CAL_N],xs[CAL_N:CAL_N+HOLD_N]


def enc(tok,t): return tok(t,return_tensors="pt",truncation=True,max_length=MAX_LENGTH)


@torch.no_grad()
def losses(model,tok,texts):
    out=[]
    for t in texts:
        x=enc(tok,t); out.append(float(model(**x,labels=x["input_ids"]).loss))
    return np.array(out,dtype=np.float64)


def mask_weight(w,code):
    x=w.detach().clone()
    gen=torch.Generator(device="cpu")
    gen.manual_seed(314159265 + code*1000003 + LAYER*1009 + int(FRAC*1000))
    m=torch.rand(x.shape,generator=gen)<FRAC
    x[m]=0
    return x


def metric(fp,v):
    d=v-fp
    pos=np.maximum(d,0)
    top=np.sort(pos)[-max(1,len(pos)//4):]
    return {
      "mean_delta_nll":float(d.mean()),
      "mean_positive_delta_nll":float(pos.mean()),
      "cvar25_positive":float(top.mean()),
      "p90_delta_nll":float(np.quantile(d,0.9)),
      "max_delta_nll":float(d.max()),
      "fraction_worse":float(np.mean(d>0)),
      "per_sample_delta":d.tolist(),
    }


def key(m):
    return (m["cvar25_positive"],m["mean_positive_delta_nll"],m["p90_delta_nll"],m["max_delta_nll"],m["mean_delta_nll"])


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    cal,hold=load_texts()
    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    target=model.model.layers[LAYER].mlp.down_proj.weight
    orig=target.detach().clone()

    with torch.no_grad(): target.copy_(orig)
    cal_fp=losses(model,tok,cal); hold_fp=losses(model,tok,hold)

    candidates=[]; best=None; bestk=None
    for code in range(NUM_MASKS):
        m=mask_weight(orig,code)
        with torch.no_grad(): target.copy_(m)
        cm=metric(cal_fp,losses(model,tok,cal))
        candidates.append({"code":code,"calibration":cm})
        k=key(cm)
        print("MASK",code,k,flush=True)
        if bestk is None or k<bestk: bestk=k;best=code

    holdout={}
    for code in sorted(set([0,best])):
        m=mask_weight(orig,code)
        with torch.no_grad(): target.copy_(m)
        holdout[str(code)]=metric(hold_fp,losses(model,tok,hold))

    with torch.no_grad(): target.copy_(orig)
    index_bpw=float(np.ceil(np.log2(NUM_MASKS)))/orig.numel()
    raw_bpw=(1-FRAC)+16/GROUP_SIZE+index_bpw
    payload={
      "model":MODEL_ID,"layer":LAYER,"prune_fraction":FRAC,
      "num_masks":NUM_MASKS,"calibration_n":CAL_N,"holdout_n":HOLD_N,
      "selected_code":best,"candidates":candidates,
      "fixed_code0_holdout":holdout["0"],"selected_holdout":holdout[str(best)],
      "storage_bpw_raw_scale":float(raw_bpw)
    }
    (OUT/f"l{LAYER}_p50.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps({
      "selected_code":best,
      "fixed_positive":holdout["0"]["mean_positive_delta_nll"],
      "selected_positive":holdout[str(best)]["mean_positive_delta_nll"],
      "fixed_cvar":holdout["0"]["cvar25_positive"],
      "selected_cvar":holdout[str(best)]["cvar25_positive"],
      "fixed_max":holdout["0"]["max_delta_nll"],
      "selected_max":holdout[str(best)]["max_delta_nll"],
    },indent=2))


if __name__=="__main__":
    main()
