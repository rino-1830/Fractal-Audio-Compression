import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "prism-ml/Bonsai-1.7B-unpacked"
LAYER = int(os.environ["CASE_LAYER"])
FRAC = 0.50
NUM_MASKS = 16
N_PROMPTS = 32
MAX_LENGTH = 20
SKIP_ELIGIBLE = 200
OUT = Path("results-sensitivity-spectrum")


def load_texts():
    ds = load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
    xs=[]; skipped=0
    for row in ds:
        t=" ".join(row["text"].split())
        if len(t)<100 or t.startswith("="):
            continue
        if skipped<SKIP_ELIGIBLE:
            skipped+=1
            continue
        xs.append(t[:500])
        if len(xs)>=N_PROMPTS:
            break
    if len(xs)<N_PROMPTS:
        raise RuntimeError("not enough text")
    return xs


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


def spectrum(A):
    # Remove row/column mean effects so spectrum measures interaction structure.
    A=np.asarray(A,dtype=np.float64)
    C=A-A.mean(axis=1,keepdims=True)-A.mean(axis=0,keepdims=True)+A.mean()
    u,s,vt=np.linalg.svd(C,full_matrices=False)
    power=s*s
    total=float(power.sum())
    ratio=(power/total) if total>0 else np.zeros_like(power)
    c=np.cumsum(ratio)
    p=ratio[ratio>0]
    effective=float(np.exp(-(p*np.log(p)).sum())) if len(p) else 0.0
    stable=float((s*s).sum()/(s[0]*s[0])) if len(s) and s[0]>0 else 0.0
    return {
        "singular_values":s.tolist(),
        "explained_variance_ratio":ratio.tolist(),
        "cumulative_explained_variance":c.tolist(),
        "top1":float(c[0]) if len(c)>0 else 0.0,
        "top2":float(c[min(1,len(c)-1)]) if len(c)>0 else 0.0,
        "top4":float(c[min(3,len(c)-1)]) if len(c)>0 else 0.0,
        "top8":float(c[min(7,len(c)-1)]) if len(c)>0 else 0.0,
        "effective_rank":effective,
        "stable_rank":stable,
        "right_singular_vectors":vt[:8].tolist(),
    }


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    texts=load_texts()
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
    fp=losses(model,tok,texts)

    D=[]
    for code in range(NUM_MASKS):
        with torch.no_grad():
            target.copy_(mask_weight(original,code))
        vals=losses(model,tok,texts)
        d=vals-fp
        D.append(d)
        print("MASK",code,float(d.mean()),float(np.maximum(d,0).mean()),flush=True)

    with torch.no_grad():
        target.copy_(original)

    D=np.stack(D,axis=0)
    P=np.maximum(D,0.0)
    raw_spec=spectrum(D)
    pos_spec=spectrum(P)

    # Prompt susceptibility: mean harm across random masks.
    susceptibility=P.mean(axis=0)
    payload={
        "model":MODEL_ID,"layer":LAYER,"prune_fraction":FRAC,
        "num_masks":NUM_MASKS,"num_prompts":N_PROMPTS,
        "delta_matrix":D.tolist(),
        "positive_harm_matrix":P.tolist(),
        "raw_centered_spectrum":raw_spec,
        "positive_centered_spectrum":pos_spec,
        "prompt_susceptibility":susceptibility.tolist(),
        "summary":{
            "raw_top1":raw_spec["top1"],
            "raw_top2":raw_spec["top2"],
            "raw_top4":raw_spec["top4"],
            "raw_effective_rank":raw_spec["effective_rank"],
            "positive_top1":pos_spec["top1"],
            "positive_top2":pos_spec["top2"],
            "positive_top4":pos_spec["top4"],
            "positive_effective_rank":pos_spec["effective_rank"],
        }
    }
    (OUT/f"l{LAYER}.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload["summary"],indent=2))


if __name__=="__main__":
    main()
