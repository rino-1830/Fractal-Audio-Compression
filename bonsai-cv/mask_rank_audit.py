import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "prism-ml/Bonsai-1.7B-unpacked"
LAYER = int(os.environ.get("CASE_LAYER", "14"))
FRAC = 0.50
NUM_MASKS = 16
CAL_N = 8
HOLD_N = 16
MAX_LENGTH = 20
SKIP_ELIGIBLE = 360
OUT = Path("results-mask-rank-audit")


def load_texts():
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    xs=[]; skipped=0
    for row in ds:
        t=" ".join(row["text"].split())
        if len(t)<100 or t.startswith("="):
            continue
        if skipped < SKIP_ELIGIBLE:
            skipped += 1
            continue
        xs.append(t[:500])
        if len(xs) >= CAL_N + HOLD_N:
            break
    return xs[:CAL_N], xs[CAL_N:CAL_N+HOLD_N]


def encode(tok,text):
    return tok(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)


@torch.no_grad()
def losses(model,tok,texts):
    vals=[]
    for t in texts:
        x=encode(tok,t)
        vals.append(float(model(**x, labels=x["input_ids"]).loss))
    return np.asarray(vals, dtype=np.float64)


def masked_weight(w, code):
    x=w.detach().clone()
    g=torch.Generator(device="cpu")
    g.manual_seed(246813579 + code*1000003 + LAYER*1009)
    mask=torch.rand(x.shape, generator=g) < FRAC
    x[mask]=0
    return x


def metrics(fp, vals):
    d=vals-fp
    pos=np.maximum(d,0.0)
    k=max(1,int(np.ceil(len(pos)*0.25)))
    return {
        "mean_delta":float(d.mean()),
        "positive_harm":float(pos.mean()),
        "cvar25":float(np.sort(pos)[-k:].mean()),
        "max_delta":float(d.max()),
    }


def rank_of_selected(rows, selected, key):
    ordered=sorted(rows,key=lambda r:r["holdout"][key])
    return 1 + next(i for i,r in enumerate(ordered) if r["code"]==selected)


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    cal,hold=load_texts()
    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    target=model.model.layers[LAYER].mlp.down_proj.weight
    original=target.detach().clone()

    with torch.no_grad(): target.copy_(original)
    cal_fp=losses(model,tok,cal); hold_fp=losses(model,tok,hold)

    rows=[]
    for code in range(NUM_MASKS):
        mw=masked_weight(original,code)
        with torch.no_grad(): target.copy_(mw)
        cm=metrics(cal_fp,losses(model,tok,cal))
        hm=metrics(hold_fp,losses(model,tok,hold))
        rows.append({"code":code,"calibration":cm,"holdout":hm})
        print("CODE",code,cm["positive_harm"],hm["positive_harm"],flush=True)

    with torch.no_grad(): target.copy_(original)

    keys=["positive_harm","mean_delta","cvar25","max_delta"]
    correlations={}
    selected={}
    for key in keys:
        c=np.asarray([r["calibration"][key] for r in rows])
        h=np.asarray([r["holdout"][key] for r in rows])
        rho,p=spearmanr(c,h)
        best=min(rows,key=lambda r:r["calibration"][key])["code"]
        oracle=min(rows,key=lambda r:r["holdout"][key])["code"]
        correlations[key]={"spearman_rho":float(rho),"p_value":float(p)}
        selected[key]={
            "calibration_selected_code":best,
            "oracle_holdout_code":oracle,
            "selected_holdout_rank":rank_of_selected(rows,best,key),
            "selected_holdout_value":next(r["holdout"][key] for r in rows if r["code"]==best),
            "oracle_holdout_value":next(r["holdout"][key] for r in rows if r["code"]==oracle),
            "code0_holdout_value":rows[0]["holdout"][key],
        }

    payload={
        "model":MODEL_ID,"layer":LAYER,"prune_fraction":FRAC,
        "num_masks":NUM_MASKS,"calibration_n":CAL_N,"holdout_n":HOLD_N,
        "correlations":correlations,"selected":selected,"rows":rows
    }
    (OUT/f"l{LAYER}_p50.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps({"layer":LAYER,"correlations":correlations,"selected":selected},indent=2))


if __name__=="__main__":
    main()
