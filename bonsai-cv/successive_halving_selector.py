import json
import os
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
MAX_LENGTH=20
BASE_SKIP=1000
FOLD_STRIDE=80
STAGES=[(16,2),(8,2),(4,4),(2,8)]
HOLD_N=16
GROUP_SIZE=128
OUT=Path("results-successive-halving")


def load_pool():
    ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
    xs=[]; eligible=0
    start=BASE_SKIP+FOLD*FOLD_STRIDE
    need=sum(n for _,n in STAGES)+HOLD_N
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


def mask_weight(w,code):
    x=w.detach().clone()
    g=torch.Generator(device="cpu")
    g.manual_seed(246813579+code*1000003+LAYER*1009)
    m=torch.rand(x.shape,generator=g)<FRAC
    x[m]=0
    return x


def batch_metric(delta):
    d=np.asarray(delta,dtype=np.float64)
    return (
        float(np.maximum(d,0).mean()),
        float(d.mean()),
        float(np.abs(d).mean()),
        float(d.max()),
    )


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
    pool=load_pool()
    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    model=AutoModelForCausalLM.from_pretrained(
      MODEL_ID,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    target=model.model.layers[LAYER].mlp.down_proj.weight
    orig=target.detach().clone()

    # Partition independent batches.
    batches=[]
    pos=0
    for _,n in STAGES:
        batches.append(pool[pos:pos+n]); pos+=n
    hold=pool[pos:pos+HOLD_N]

    with torch.no_grad():
        target.copy_(orig)
    fp_batches=[losses(model,tok,b) for b in batches]
    hold_fp=losses(model,tok,hold)

    active=list(range(NUM_MASKS))
    history=[]
    for si,((expected,n),texts,fp) in enumerate(zip(STAGES,batches,fp_batches)):
        assert len(active)==expected
        scored=[]
        for code in active:
            with torch.no_grad():
                target.copy_(mask_weight(orig,code))
            vals=losses(model,tok,texts)
            d=vals-fp
            scored.append((batch_metric(d),code,d.tolist()))
        scored.sort(key=lambda x:x[0])
        keep=max(1,len(active)//2) if si<len(STAGES)-1 else 1
        next_active=[code for _,code,_ in scored[:keep]]
        history.append({
          "stage":si,"batch_n":n,"active":active,
          "ranking":[{"code":code,"metric":list(k),"delta":d} for k,code,d in scored],
          "kept":next_active,
        })
        print("STAGE",si,"active",active,"best",scored[0][1],scored[0][0],"keep",next_active,flush=True)
        active=next_active

    selected=active[0]

    compare=sorted(set([0,selected]))
    holdout={}
    for code in compare:
        with torch.no_grad():
            target.copy_(mask_weight(orig,code))
        holdout[str(code)]=metrics(hold_fp,losses(model,tok,hold))

    with torch.no_grad():
        target.copy_(orig)

    storage_bpw=(1-FRAC)+16/GROUP_SIZE+np.ceil(np.log2(NUM_MASKS))/orig.numel()
    total_cal_evals=sum(expected*n for expected,n in STAGES)

    payload={
      "model":MODEL_ID,"layer":LAYER,"fold":FOLD,"prune_fraction":FRAC,
      "stages":STAGES,"total_mask_prompt_evals":total_cal_evals,
      "selected_code":selected,"history":history,
      "fixed_code0_holdout":holdout["0"],
      "selected_holdout":holdout[str(selected)],
      "storage_bpw":float(storage_bpw),
    }
    (OUT/f"l{LAYER}_f{FOLD}.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps({
      "layer":LAYER,"fold":FOLD,"selected_code":selected,
      "total_cal_evals":total_cal_evals,
      "fixed_positive":holdout["0"]["mean_positive_delta_nll"],
      "selected_positive":holdout[str(selected)]["mean_positive_delta_nll"],
      "fixed_mean":holdout["0"]["mean_delta_nll"],
      "selected_mean":holdout[str(selected)]["mean_delta_nll"],
      "storage_bpw":storage_bpw,
    },indent=2))


if __name__=="__main__":
    main()
