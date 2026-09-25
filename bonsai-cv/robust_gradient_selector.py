import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID="prism-ml/Bonsai-1.7B-unpacked"
LAYER=int(os.environ.get("CASE_LAYER","14"))
FRAC=0.50
NUM_MASKS=16
CAL_N=16
HOLD_N=24
MAX_LENGTH=20
SKIP_ELIGIBLE=700
GROUP_SIZE=128
OUT=Path("results-robust-gradient-selector")


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


def enc(tok,t):
    return tok(t,return_tensors="pt",truncation=True,max_length=MAX_LENGTH)


@torch.no_grad()
def losses(model,tok,texts):
    a=[]
    for t in texts:
        x=enc(tok,t); a.append(float(model(**x,labels=x["input_ids"]).loss))
    return np.array(a,dtype=np.float64)


def masked(w,code):
    x=w.detach().clone()
    gen=torch.Generator(device="cpu")
    gen.manual_seed(11235813+code*1000003+LAYER*1009+int(FRAC*1000))
    m=torch.rand(x.shape,generator=gen)<FRAC
    x[m]=0
    return x


def grad(model,tok,target,text):
    model.zero_grad(set_to_none=True)
    x=enc(tok,text)
    loss=model(**x,labels=x["input_ids"]).loss
    return torch.autograd.grad(loss,target)[0].detach().float().reshape(-1)


def actual_metric(fp,v):
    d=v-fp
    return {
        "mean_delta_nll":float(d.mean()),
        "mean_positive_delta_nll":float(np.maximum(d,0).mean()),
        "p90_delta_nll":float(np.quantile(d,0.9)),
        "max_delta_nll":float(d.max()),
        "fraction_worse":float(np.mean(d>0)),
        "per_sample_delta":d.tolist(),
    }


def risk(pred):
    p=np.asarray(pred,dtype=np.float64)
    pos=np.maximum(p,0)
    top=np.sort(pos)[-max(1,len(pos)//4):]
    return {
        "cvar25_positive":float(top.mean()),
        "mean_positive":float(pos.mean()),
        "p90":float(np.quantile(p,0.9)),
        "max":float(p.max()),
        "mean":float(p.mean()),
        "per_sample":p.tolist(),
    }


def key(r):
    return (r["cvar25_positive"],r["mean_positive"],r["p90"],r["max"],r["mean"])


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    cal,hold=load_texts()
    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    target=model.model.layers[LAYER].mlp.down_proj.weight
    target.requires_grad_(True)
    orig=target.detach().clone()

    with torch.no_grad():target.copy_(orig)
    hold_fp=losses(model,tok,hold)

    gs=[]
    for i,t in enumerate(cal):
        print("GRAD",i,flush=True)
        gs.append(grad(model,tok,target,t))
    G=torch.stack(gs)

    candidates=[]
    best=None; bestk=None
    for code in range(NUM_MASKS):
        m=masked(orig,code)
        d=(m.float()-orig.float()).reshape(-1)
        r=risk((G@d).double().cpu().numpy())
        candidates.append({"code":code,"risk":r})
        k=key(r)
        print("MASK",code,k,flush=True)
        if bestk is None or k<bestk:
            bestk=k;best=code

    holdout={}
    for code in sorted(set([0,best])):
        m=masked(orig,code)
        with torch.no_grad():target.copy_(m)
        holdout[str(code)]=actual_metric(hold_fp,losses(model,tok,hold))

    with torch.no_grad():target.copy_(orig)
    target.requires_grad_(False)

    index_bpw=float(np.ceil(np.log2(NUM_MASKS)))/orig.numel()
    rawscale_bpw=(1-FRAC)+16/GROUP_SIZE+index_bpw
    payload={
      "layer":LAYER,"prune_fraction":FRAC,"calibration_n":CAL_N,"holdout_n":HOLD_N,
      "selected_code":best,"candidates":candidates,
      "fixed_code0_holdout":holdout["0"],"selected_holdout":holdout[str(best)],
      "storage_bpw_raw_fp16_scale":float(rawscale_bpw),
    }
    (OUT/f"l{LAYER}_p50.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps({
      "selected_code":best,
      "fixed_positive_harm":holdout["0"]["mean_positive_delta_nll"],
      "selected_positive_harm":holdout[str(best)]["mean_positive_delta_nll"],
      "fixed_p90":holdout["0"]["p90_delta_nll"],
      "selected_p90":holdout[str(best)]["p90_delta_nll"],
      "fixed_max":holdout["0"]["max_delta_nll"],
      "selected_max":holdout[str(best)]["max_delta_nll"],
    },indent=2))


if __name__=="__main__":
    main()
