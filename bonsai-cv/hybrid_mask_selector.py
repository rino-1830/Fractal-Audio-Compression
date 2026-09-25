import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "prism-ml/Bonsai-1.7B-unpacked"
LAYER = int(os.environ.get("CASE_LAYER", "14"))
FRAC = float(os.environ.get("CASE_FRAC", "0.50"))
NUM_MASKS = 16
SHORTLIST = 4
GRAD_N = 4
REFINE_N = 4
HOLD_N = 12
MAX_LENGTH = 20
SKIP_ELIGIBLE = 420
GROUP_SIZE = 128
OUT = Path("results-hybrid-selector")


def load_texts():
    ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
    xs=[]; skipped=0
    need=GRAD_N+REFINE_N+HOLD_N
    for row in ds:
        t=" ".join(row["text"].split())
        if len(t)<100 or t.startswith("="): continue
        if skipped<SKIP_ELIGIBLE:
            skipped+=1; continue
        xs.append(t[:500])
        if len(xs)>=need: break
    return xs[:GRAD_N],xs[GRAD_N:GRAD_N+REFINE_N],xs[GRAD_N+REFINE_N:need]


def encode(tok,text):
    return tok(text,return_tensors="pt",truncation=True,max_length=MAX_LENGTH)


@torch.no_grad()
def losses(model,tok,texts):
    vals=[]
    for t in texts:
        x=encode(tok,t)
        vals.append(float(model(**x,labels=x["input_ids"]).loss))
    return np.array(vals,dtype=np.float64)


def mask_weight(w,code):
    x=w.detach().clone()
    gen=torch.Generator(device="cpu")
    gen.manual_seed(246813579 + code*1000003 + LAYER*1009 + int(FRAC*1000))
    m=torch.rand(x.shape,generator=gen)<FRAC
    x[m]=0
    return x


def raw_grad(model,tok,target,text):
    model.zero_grad(set_to_none=True)
    x=encode(tok,text)
    loss=model(**x,labels=x["input_ids"]).loss
    g=torch.autograd.grad(loss,target,retain_graph=False,create_graph=False)[0]
    return g.detach().float().reshape(-1)


def metric(fp,vals):
    d=vals-fp
    return {
        "mean_delta_nll":float(d.mean()),
        "mean_positive_delta_nll":float(np.maximum(d,0).mean()),
        "max_delta_nll":float(d.max()),
        "fraction_worse":float(np.mean(d>0)),
        "per_sample_delta":d.tolist(),
    }


def score(m):
    return (m["mean_positive_delta_nll"],m["mean_delta_nll"],m["max_delta_nll"])


def pred_metric(v):
    v=np.asarray(v,dtype=np.float64)
    return {
        "mean_delta_nll":float(v.mean()),
        "mean_positive_delta_nll":float(np.maximum(v,0).mean()),
        "max_delta_nll":float(v.max()),
    }


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    grad_texts,refine_texts,hold_texts=load_texts()

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
    refine_fp=losses(model,tok,refine_texts)
    hold_fp=losses(model,tok,hold_texts)

    G=torch.stack([raw_grad(model,tok,target,t) for t in grad_texts])

    predicted=[]
    for code in range(NUM_MASKS):
        masked=mask_weight(original,code)
        delta=(masked.float()-original.float()).reshape(-1)
        pm=pred_metric((G@delta).double().cpu().numpy())
        predicted.append({"code":code,"prediction":pm})
    pred_sorted=sorted(predicted,key=lambda x:score(x["prediction"]))
    shortlist=[x["code"] for x in pred_sorted[:SHORTLIST]]

    refined=[]
    best_code=None; best_score=None
    for code in shortlist:
        masked=mask_weight(original,code)
        with torch.no_grad(): target.copy_(masked)
        rm=metric(refine_fp,losses(model,tok,refine_texts))
        refined.append({"code":code,"refine":rm})
        s=score(rm)
        if best_score is None or s<best_score:
            best_score=s; best_code=code

    holdout={}
    for code in sorted(set([0,best_code])):
        masked=mask_weight(original,code)
        with torch.no_grad(): target.copy_(masked)
        holdout[str(code)]=metric(hold_fp,losses(model,tok,hold_texts))

    with torch.no_grad(): target.copy_(original)
    target.requires_grad_(False)

    index_bpw=float(np.ceil(np.log2(NUM_MASKS)))/original.numel()
    storage_bpw=(1.0-FRAC)+16.0/GROUP_SIZE+index_bpw

    payload={
        "model":MODEL_ID,"layer":LAYER,"prune_fraction":FRAC,
        "num_masks":NUM_MASKS,"shortlist":SHORTLIST,
        "gradient_shortlist":shortlist,"best_code":best_code,
        "predicted":predicted,"refined":refined,
        "fixed_code0_holdout":holdout["0"],
        "hybrid_selected_holdout":holdout[str(best_code)],
        "storage_bpw_raw_fp16_scale":float(storage_bpw),
    }
    stem=f"l{LAYER}_p{int(FRAC*100)}"
    (OUT/f"{stem}.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps({
        "shortlist":shortlist,"best_code":best_code,
        "fixed_positive_harm":holdout["0"]["mean_positive_delta_nll"],
        "selected_positive_harm":holdout[str(best_code)]["mean_positive_delta_nll"],
        "fixed_mean_delta":holdout["0"]["mean_delta_nll"],
        "selected_mean_delta":holdout[str(best_code)]["mean_delta_nll"],
        "storage_bpw":storage_bpw,
    },indent=2))


if __name__=="__main__":
    main()
