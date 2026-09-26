import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "prism-ml/Bonsai-1.7B-unpacked"
LAYERS = [0, 7, 14, 21, 27]
FRAC = 0.50
NUM_MASKS = 16
SHORTLIST = 4
SWEEPS = 2
GRAD_N = 4
REFINE_N = 6
HOLD_N = 16
MAX_LENGTH = 20
BASE_SKIP = 1900
FOLD_STRIDE = 50
FOLD = int(os.environ.get("CASE_FOLD", "0"))
GROUP_SIZE = 128
OUT = Path("results-joint-coordinate")


def load_texts():
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    xs=[]; eligible=0
    start=BASE_SKIP + FOLD*FOLD_STRIDE
    need=GRAD_N+REFINE_N+HOLD_N
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
    return xs[:GRAD_N], xs[GRAD_N:GRAD_N+REFINE_N], xs[GRAD_N+REFINE_N:]


def encode(tok,text):
    return tok(text,return_tensors="pt",truncation=True,max_length=MAX_LENGTH)


@torch.no_grad()
def losses(model,tok,texts):
    vals=[]
    for t in texts:
        x=encode(tok,t)
        vals.append(float(model(**x,labels=x["input_ids"]).loss))
    return np.asarray(vals,dtype=np.float64)


def masked_weight(w,layer,code):
    x=w.detach().clone()
    g=torch.Generator(device="cpu")
    g.manual_seed(246813579 + code*1000003 + layer*1009)
    m=torch.rand(x.shape,generator=g)<FRAC
    x[m]=0
    return x


def metrics(fp,vals):
    d=vals-fp
    return {
        "mean_delta_nll":float(d.mean()),
        "mean_positive_delta_nll":float(np.maximum(d,0).mean()),
        "max_delta_nll":float(d.max()),
        "fraction_worse":float(np.mean(d>0)),
        "per_sample_delta":d.tolist(),
    }


def mkey(m):
    return (m["mean_positive_delta_nll"],m["mean_delta_nll"],m["max_delta_nll"])


def predkey(v):
    v=np.asarray(v,dtype=np.float64)
    return (float(np.maximum(v,0).mean()),float(v.mean()),float(v.max()))


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    grad_texts,refine_texts,hold_texts=load_texts()

    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    model=AutoModelForCausalLM.from_pretrained(
        MODEL_ID,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    targets={l:model.model.layers[l].mlp.down_proj.weight for l in LAYERS}
    originals={l:targets[l].detach().clone() for l in LAYERS}
    for t in targets.values():
        t.requires_grad_(True)

    with torch.no_grad():
        for l in LAYERS:
            targets[l].copy_(originals[l])
    refine_fp=losses(model,tok,refine_texts)
    hold_fp=losses(model,tok,hold_texts)

    target_list=[targets[l] for l in LAYERS]
    grads={l:[] for l in LAYERS}
    for i,text in enumerate(grad_texts):
        model.zero_grad(set_to_none=True)
        x=encode(tok,text)
        loss=model(**x,labels=x["input_ids"]).loss
        gs=torch.autograd.grad(loss,target_list,retain_graph=False,create_graph=False)
        for l,g in zip(LAYERS,gs):
            grads[l].append(g.detach().float().reshape(-1))
        print("GRAD",i,flush=True)
    G={l:torch.stack(grads[l]) for l in LAYERS}

    shortlists={}
    predictions={}
    for l in LAYERS:
        ps=[]
        for code in range(NUM_MASKS):
            mw=masked_weight(originals[l],l,code)
            delta=(mw.float()-originals[l].float()).reshape(-1)
            vec=(G[l]@delta).double().cpu().numpy()
            ps.append((predkey(vec),code,vec.tolist()))
        ps.sort(key=lambda x:x[0])
        shortlists[l]=[code for _,code,_ in ps[:SHORTLIST]]
        predictions[l]=[{"code":code,"key":list(k),"vec":vec} for k,code,vec in ps]
        print("SHORT",l,shortlists[l],flush=True)

    del G

    def restore():
        with torch.no_grad():
            for l in LAYERS:
                targets[l].copy_(originals[l])

    def apply_codes(codes):
        restore()
        with torch.no_grad():
            for l in LAYERS:
                targets[l].copy_(masked_weight(originals[l],l,int(codes[l])))

    def evaluate_codes(codes):
        apply_codes(codes)
        return metrics(refine_fp,losses(model,tok,refine_texts))

    # Initialize with best first-order code per layer.
    codes={l:shortlists[l][0] for l in LAYERS}
    history=[]
    current=evaluate_codes(codes)
    history.append({"stage":"init","codes":dict(codes),"refine":current})
    print("INIT",codes,mkey(current),flush=True)

    for sweep in range(SWEEPS):
        changed=False
        for l in LAYERS:
            best_code=codes[l]
            best_metric=current
            trials=[]
            for code in shortlists[l]:
                trial=dict(codes); trial[l]=code
                tm=evaluate_codes(trial)
                trials.append({"code":code,"metric":tm})
                if mkey(tm)<mkey(best_metric):
                    best_code=code
                    best_metric=tm
            if best_code!=codes[l]:
                changed=True
                codes[l]=best_code
                current=best_metric
            history.append({
                "stage":f"sweep{sweep}_layer{l}",
                "codes":dict(codes),
                "refine":current,
                "trials":trials,
            })
            print("STEP",sweep,l,codes,mkey(current),flush=True)
        if not changed:
            break

    selected_codes=dict(codes)

    # Baselines on untouched holdout.
    fixed0={l:0 for l in LAYERS}
    first_order={l:shortlists[l][0] for l in LAYERS}
    holdout={}
    for name,c in {
        "fixed0":fixed0,
        "first_order_independent":first_order,
        "coordinate_selected":selected_codes,
    }.items():
        apply_codes(c)
        holdout[name]=metrics(hold_fp,losses(model,tok,hold_texts))

    restore()
    for t in targets.values():
        t.requires_grad_(False)

    code_bits=len(LAYERS)*int(np.ceil(np.log2(NUM_MASKS)))
    nweights=sum(originals[l].numel() for l in LAYERS)
    storage_bpw=(1.0-FRAC)+16.0/GROUP_SIZE+code_bits/nweights

    payload={
        "model":MODEL_ID,"fold":FOLD,"layers":LAYERS,"prune_fraction":FRAC,
        "num_masks":NUM_MASKS,"shortlist":SHORTLIST,"sweeps":SWEEPS,
        "gradient_n":GRAD_N,"refine_n":REFINE_N,"holdout_n":HOLD_N,
        "shortlists":shortlists,"selected_codes":selected_codes,
        "history":history,"holdout":holdout,
        "storage_bpw":float(storage_bpw),
    }
    (OUT/f"fold{FOLD}.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps({
        "fold":FOLD,"shortlists":shortlists,"selected_codes":selected_codes,
        "fixed0":holdout["fixed0"],
        "first_order_independent":holdout["first_order_independent"],
        "coordinate_selected":holdout["coordinate_selected"],
        "storage_bpw":storage_bpw,
    },indent=2))


if __name__=="__main__":
    main()
