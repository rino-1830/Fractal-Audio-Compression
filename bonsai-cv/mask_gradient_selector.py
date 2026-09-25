import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "prism-ml/Bonsai-1.7B-unpacked"
LAYER = int(os.environ.get("CASE_LAYER", "14"))
FRAC = float(os.environ.get("CASE_FRAC", "0.25"))
NUM_MASKS = 16
CAL_N = 4
HOLD_N = 12
MAX_LENGTH = 20
SKIP_ELIGIBLE = 220
GROUP_SIZE = 128
OUT = Path("results-gradient-selector")


def load_texts():
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    xs, skipped = [], 0
    for row in ds:
        t = " ".join(row["text"].split())
        if len(t) < 100 or t.startswith("="):
            continue
        if skipped < SKIP_ELIGIBLE:
            skipped += 1
            continue
        xs.append(t[:500])
        if len(xs) >= CAL_N + HOLD_N:
            break
    return xs[:CAL_N], xs[CAL_N:CAL_N+HOLD_N]


def encode(tok, text):
    return tok(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)


@torch.no_grad()
def losses(model, tok, texts):
    vals=[]
    for text in texts:
        x=encode(tok,text)
        vals.append(float(model(**x,labels=x["input_ids"]).loss))
    return np.array(vals,dtype=np.float64)


def mask_weight(w, code):
    x=w.detach().clone()
    gen=torch.Generator(device="cpu")
    gen.manual_seed(123456789 + code*1000003 + LAYER*1009 + int(FRAC*1000))
    mask=torch.rand(x.shape,generator=gen) < FRAC
    x[mask]=0
    return x


def raw_grad(model,tok,target,text):
    model.zero_grad(set_to_none=True)
    x=encode(tok,text)
    loss=model(**x,labels=x["input_ids"]).loss
    g=torch.autograd.grad(loss,target,retain_graph=False,create_graph=False)[0]
    return g.detach().float().reshape(-1)


def metrics(fp,vals):
    d=vals-fp
    return {
        "mean_delta_nll":float(d.mean()),
        "mean_positive_delta_nll":float(np.maximum(d,0).mean()),
        "max_delta_nll":float(d.max()),
        "fraction_worse":float(np.mean(d>0)),
        "per_sample_delta":d.tolist(),
    }


def pred_metrics(pred):
    pred=np.asarray(pred,dtype=np.float64)
    return {
        "mean_pred_delta":float(pred.mean()),
        "mean_positive_pred_delta":float(np.maximum(pred,0).mean()),
        "max_pred_delta":float(pred.max()),
        "per_sample_pred_delta":pred.tolist(),
    }


def score_pred(p):
    return (p["mean_positive_pred_delta"],p["mean_pred_delta"],p["max_pred_delta"])


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    cal,hold=load_texts()

    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    model=AutoModelForCausalLM.from_pretrained(
        MODEL_ID,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True
    )
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    target=model.model.layers[LAYER].mlp.down_proj.weight
    target.requires_grad_(True)
    original=target.detach().clone()

    with torch.no_grad(): target.copy_(original)
    hold_fp=losses(model,tok,hold)

    grads=[]
    for i,text in enumerate(cal):
        print("GRAD",i,flush=True)
        grads.append(raw_grad(model,tok,target,text))
    G=torch.stack(grads,dim=0)

    candidates=[]
    best_code=None
    best_score=None
    for code in range(NUM_MASKS):
        masked=mask_weight(original,code)
        delta=(masked.float()-original.float()).reshape(-1)
        pred=(G @ delta).double().cpu().numpy()
        pm=pred_metrics(pred)
        s=score_pred(pm)
        candidates.append({"code":code,"prediction":pm})
        print("MASK",code,s,flush=True)
        if best_score is None or s < best_score:
            best_score=s
            best_code=code

    holdout={}
    for code in sorted(set([0,best_code])):
        masked=mask_weight(original,code)
        with torch.no_grad(): target.copy_(masked)
        vals=losses(model,tok,hold)
        holdout[str(code)]=metrics(hold_fp,vals)

    with torch.no_grad(): target.copy_(original)
    target.requires_grad_(False)

    index_bpw=float(np.ceil(np.log2(NUM_MASKS)))/original.numel()
    storage_bpw=(1.0-FRAC)+16.0/GROUP_SIZE+index_bpw

    payload={
        "model":MODEL_ID,
        "layer":LAYER,
        "prune_fraction":FRAC,
        "num_masks":NUM_MASKS,
        "selector":"first-order per-example loss gradient dot pruning residual",
        "gradient_selected_code":best_code,
        "storage_bpw_raw_fp16_scale":float(storage_bpw),
        "candidates":candidates,
        "fixed_code0_holdout":holdout["0"],
        "gradient_selected_holdout":holdout[str(best_code)],
    }
    stem=f"l{LAYER}_p{int(FRAC*100)}"
    (OUT/f"{stem}.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps({
        "selected_code":best_code,
        "fixed_positive_harm":holdout["0"]["mean_positive_delta_nll"],
        "selected_positive_harm":holdout[str(best_code)]["mean_positive_delta_nll"],
        "fixed_mean_delta":holdout["0"]["mean_delta_nll"],
        "selected_mean_delta":holdout[str(best_code)]["mean_delta_nll"],
        "storage_bpw":storage_bpw,
    },indent=2))


if __name__=="__main__":
    main()
