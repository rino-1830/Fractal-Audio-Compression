import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "prism-ml/Bonsai-1.7B-unpacked"
LAYER = int(os.environ["CASE_LAYER"])
FOLD = int(os.environ["CASE_FOLD"])
FRAC = 0.50
NUM_MASKS = 16
CAL_N = 8
HOLD_N = 16
MAX_LENGTH = 20
BASE_SKIP = 1700
FOLD_STRIDE = 40
GROUP_SIZE = 128
OUT = Path("results-rank-selector")


def load_texts():
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    xs=[]; eligible=0
    start=BASE_SKIP + FOLD*FOLD_STRIDE
    need=CAL_N+HOLD_N
    for row in ds:
        t=" ".join(row["text"].split())
        if len(t)<100 or t.startswith("="):
            continue
        if eligible<start:
            eligible += 1
            continue
        xs.append(t[:500])
        if len(xs)>=need:
            break
    if len(xs)<need:
        raise RuntimeError("not enough text")
    return xs[:CAL_N], xs[CAL_N:]


def encode(tok,text):
    return tok(text,return_tensors="pt",truncation=True,max_length=MAX_LENGTH)


@torch.no_grad()
def losses(model,tok,texts):
    vals=[]
    for t in texts:
        x=encode(tok,t)
        vals.append(float(model(**x,labels=x["input_ids"]).loss))
    return np.asarray(vals,dtype=np.float64)


def mask_weight(w,code):
    x=w.detach().clone()
    g=torch.Generator(device="cpu")
    g.manual_seed(246813579 + code*1000003 + LAYER*1009)
    m=torch.rand(x.shape,generator=g)<FRAC
    x[m]=0
    return x


def metrics(fp,vals):
    d=vals-fp
    return {
        "mean_delta_nll":float(d.mean()),
        "mean_positive_delta_nll":float(np.maximum(d,0).mean()),
        "mean_abs_delta_nll":float(np.abs(d).mean()),
        "max_delta_nll":float(d.max()),
        "per_sample_delta":d.tolist(),
    }


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    cal,hold=load_texts()

    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    model=AutoModelForCausalLM.from_pretrained(
        MODEL_ID,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    target=model.model.layers[LAYER].mlp.down_proj.weight
    original=target.detach().clone()

    with torch.no_grad():
        target.copy_(original)
    cal_fp=losses(model,tok,cal)
    hold_fp=losses(model,tok,hold)

    deltas=[]
    rows=[]
    for code in range(NUM_MASKS):
        with torch.no_grad():
            target.copy_(mask_weight(original,code))
        vals=losses(model,tok,cal)
        d=vals-cal_fp
        deltas.append(d)
        rows.append({"code":code,"calibration":metrics(cal_fp,vals)})
        print("CODE",code,rows[-1]["calibration"]["mean_positive_delta_nll"],flush=True)

    D=np.stack(deltas,axis=0)  # masks x prompts

    # Per-prompt robust ordinal score:
    # primary = positive harm, tie-break = absolute behavior change.
    prompt_ranks=np.empty_like(D,dtype=np.float64)
    for p in range(CAL_N):
        order=sorted(range(NUM_MASKS), key=lambda c:(max(D[c,p],0.0), abs(D[c,p]), D[c,p]))
        for rank,c in enumerate(order):
            prompt_ranks[c,p]=rank

    mean_rank=prompt_ranks.mean(axis=1)
    q75_rank=np.quantile(prompt_ranks,0.75,axis=1)
    worst_rank=prompt_ranks.max(axis=1)

    # Pairwise win rate across prompts.
    win=np.zeros(NUM_MASKS,dtype=np.float64)
    for a in range(NUM_MASKS):
        for b in range(NUM_MASKS):
            if a==b:
                continue
            wa=0
            for p in range(CAL_N):
                ka=(max(D[a,p],0.0),abs(D[a,p]),D[a,p])
                kb=(max(D[b,p],0.0),abs(D[b,p]),D[b,p])
                wa += 1 if ka < kb else 0
            win[a] += wa / CAL_N
    win /= (NUM_MASKS-1)

    # Select robust ordinal winner. High win rate first, then mean/q75 rank.
    selected=min(range(NUM_MASKS), key=lambda c:(-win[c], mean_rank[c], q75_rank[c], worst_rank[c]))

    # Standard absolute-risk baselines from same calibration.
    positive_best=min(range(NUM_MASKS), key=lambda c:(
        rows[c]["calibration"]["mean_positive_delta_nll"],
        rows[c]["calibration"]["mean_delta_nll"],
    ))
    abs_best=min(range(NUM_MASKS), key=lambda c:(
        rows[c]["calibration"]["mean_abs_delta_nll"],
        rows[c]["calibration"]["mean_positive_delta_nll"],
    ))

    compare=sorted(set([0,selected,positive_best,abs_best]))
    holdout={}
    for code in compare:
        with torch.no_grad():
            target.copy_(mask_weight(original,code))
        holdout[str(code)]=metrics(hold_fp,losses(model,tok,hold))

    with torch.no_grad():
        target.copy_(original)

    storage_bpw=(1.0-FRAC)+16.0/GROUP_SIZE+np.ceil(np.log2(NUM_MASKS))/original.numel()

    payload={
        "model":MODEL_ID,"layer":LAYER,"fold":FOLD,"prune_fraction":FRAC,
        "calibration_n":CAL_N,"holdout_n":HOLD_N,"num_masks":NUM_MASKS,
        "selected_code":selected,"positive_best_code":positive_best,"abs_best_code":abs_best,
        "mean_rank":mean_rank.tolist(),"q75_rank":q75_rank.tolist(),
        "worst_rank":worst_rank.tolist(),"pairwise_win_rate":win.tolist(),
        "rows":rows,"holdout":holdout,"storage_bpw":float(storage_bpw),
        "summary":{
            "code0_positive":holdout["0"]["mean_positive_delta_nll"],
            "rank_selected_positive":holdout[str(selected)]["mean_positive_delta_nll"],
            "positive_best_positive":holdout[str(positive_best)]["mean_positive_delta_nll"],
            "abs_best_positive":holdout[str(abs_best)]["mean_positive_delta_nll"],
        }
    }
    (OUT/f"l{LAYER}_f{FOLD}.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload["summary"] | {
        "selected_code":selected,"positive_best_code":positive_best,"abs_best_code":abs_best
    },indent=2))


if __name__=="__main__":
    main()
