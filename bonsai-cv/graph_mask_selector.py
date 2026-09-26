import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID="prism-ml/Bonsai-1.7B-unpacked"
LAYERS=[0,7,14,21,27]
EDGES=[(0,7),(0,21),(21,27),(0,27),(7,21)]
FRAC=0.50
NUM_MASKS=16
SHORTLIST=4
TOPK=8
GRAD_N=4
FIT_N=6
GATE_N=6
HOLD_N=16
MAX_LENGTH=20
BASE_SKIP=1050
FOLD_STRIDE=45
FOLD=int(os.environ.get("CASE_FOLD","0"))
GROUP_SIZE=128
OUT=Path("results-graph-selector")


def load_texts():
    ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
    xs=[]; eligible=0
    start=BASE_SKIP+FOLD*FOLD_STRIDE
    need=GRAD_N+FIT_N+GATE_N+HOLD_N
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
    a=0
    grad=xs[a:a+GRAD_N]; a+=GRAD_N
    fit=xs[a:a+FIT_N]; a+=FIT_N
    gate=xs[a:a+GATE_N]; a+=GATE_N
    hold=xs[a:a+HOLD_N]
    return grad,fit,gate,hold


def enc(tok,t):
    return tok(t,return_tensors="pt",truncation=True,max_length=MAX_LENGTH)


@torch.no_grad()
def losses(model,tok,texts):
    out=[]
    for t in texts:
        x=enc(tok,t)
        out.append(float(model(**x,labels=x["input_ids"]).loss))
    return np.asarray(out,dtype=np.float64)


def mask_weight(w,layer,code):
    x=w.detach().clone()
    g=torch.Generator(device="cpu")
    g.manual_seed(246813579+code*1000003+layer*1009)
    m=torch.rand(x.shape,generator=g)<FRAC
    x[m]=0
    return x


def metric_vec(d):
    d=np.asarray(d,dtype=np.float64)
    return {
      "mean_delta_nll":float(d.mean()),
      "mean_positive_delta_nll":float(np.maximum(d,0).mean()),
      "max_delta_nll":float(d.max()),
      "fraction_worse":float(np.mean(d>0)),
      "per_sample_delta":d.tolist(),
    }


def mkey_from_vec(d):
    d=np.asarray(d,dtype=np.float64)
    return (float(np.maximum(d,0).mean()),float(d.mean()),float(d.max()))


def predkey(v):
    return mkey_from_vec(v)


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    grad_texts,fit_texts,gate_texts,hold_texts=load_texts()

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

    def restore():
        with torch.no_grad():
            for l in LAYERS:
                targets[l].copy_(originals[l])

    def apply_codes(codes):
        restore()
        with torch.no_grad():
            for l,c in codes.items():
                targets[l].copy_(mask_weight(originals[l],l,int(c)))

    restore()
    fit_fp=losses(model,tok,fit_texts)
    gate_fp=losses(model,tok,gate_texts)
    hold_fp=losses(model,tok,hold_texts)

    # First-order shortlist.
    target_list=[targets[l] for l in LAYERS]
    grads={l:[] for l in LAYERS}
    for i,text in enumerate(grad_texts):
        model.zero_grad(set_to_none=True)
        x=enc(tok,text)
        loss=model(**x,labels=x["input_ids"]).loss
        gs=torch.autograd.grad(loss,target_list,retain_graph=False,create_graph=False)
        for l,g in zip(LAYERS,gs):
            grads[l].append(g.detach().float().reshape(-1))
        print("GRAD",i,flush=True)
    G={l:torch.stack(grads[l]) for l in LAYERS}

    shortlists={}
    for l in LAYERS:
        rows=[]
        for code in range(NUM_MASKS):
            mw=mask_weight(originals[l],l,code)
            delta=(mw.float()-originals[l].float()).reshape(-1)
            v=(G[l]@delta).double().cpu().numpy()
            rows.append((predkey(v),code))
        rows.sort(key=lambda x:x[0])
        shortlists[l]=[c for _,c in rows[:SHORTLIST]]
        print("SHORT",l,shortlists[l],flush=True)
    del G

    # Exact single-layer response vectors on fit set.
    indiv={}
    for l in LAYERS:
        indiv[l]={}
        for c in shortlists[l]:
            apply_codes({l:c})
            v=losses(model,tok,fit_texts)-fit_fp
            indiv[l][c]=v
            print("IND",l,c,mkey_from_vec(v),flush=True)

    # Exact pairwise interaction tensors only on strong graph edges.
    inter={}
    for i,j in EDGES:
        inter[(i,j)]={}
        for ci in shortlists[i]:
            for cj in shortlists[j]:
                apply_codes({i:ci,j:cj})
                pair=losses(model,tok,fit_texts)-fit_fp
                interaction=pair-indiv[i][ci]-indiv[j][cj]
                inter[(i,j)][(ci,cj)]=interaction
        print("EDGE",i,j,"done",flush=True)

    # Enumerate 4^5 combinations using pairwise graphical surrogate.
    candidates=[]
    lists=[shortlists[l] for l in LAYERS]
    for combo in itertools.product(*lists):
        codes={l:c for l,c in zip(LAYERS,combo)}
        v=sum(indiv[l][codes[l]] for l in LAYERS)
        for i,j in EDGES:
            v=v+inter[(i,j)][(codes[i],codes[j])]
        candidates.append((mkey_from_vec(v),codes,v))
    candidates.sort(key=lambda x:x[0])
    top=candidates[:TOPK]

    # Exact gate evaluation of only top graphical-model candidates.
    gated=[]
    for rank,(pred,codes,pv) in enumerate(top):
        apply_codes(codes)
        gv=losses(model,tok,gate_texts)-gate_fp
        gm=metric_vec(gv)
        gated.append({"rank":rank,"codes":codes,"predicted_key":list(pred),"gate":gm})
        print("GATE",rank,codes,mkey_from_vec(gv),flush=True)

    gated.sort(key=lambda x:(
        x["gate"]["mean_positive_delta_nll"],
        x["gate"]["mean_delta_nll"],
        x["gate"]["max_delta_nll"],
    ))
    selected=gated[0]["codes"]

    fixed0={l:0 for l in LAYERS}
    first_order={l:shortlists[l][0] for l in LAYERS}
    holdout={}
    for name,codes in {
        "fixed0":fixed0,
        "first_order":first_order,
        "graph_selected":selected,
    }.items():
        apply_codes(codes)
        holdout[name]=metric_vec(losses(model,tok,hold_texts)-hold_fp)

    restore()
    for t in targets.values():
        t.requires_grad_(False)

    code_bits=len(LAYERS)*int(np.ceil(np.log2(NUM_MASKS)))
    nweights=sum(originals[l].numel() for l in LAYERS)
    storage=(1.0-FRAC)+16.0/GROUP_SIZE+code_bits/nweights

    payload={
      "model":MODEL_ID,"fold":FOLD,"layers":LAYERS,"edges":EDGES,
      "prune_fraction":FRAC,"num_masks":NUM_MASKS,"shortlist":SHORTLIST,
      "shortlists":shortlists,"topk":TOPK,
      "gated_candidates":gated,"selected_codes":selected,
      "holdout":holdout,"storage_bpw":float(storage),
      "settings":{"grad_n":GRAD_N,"fit_n":FIT_N,"gate_n":GATE_N,"holdout_n":HOLD_N},
    }
    (OUT/f"fold{FOLD}.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps({
      "fold":FOLD,"shortlists":shortlists,"selected":selected,
      "holdout":holdout,"storage_bpw":storage
    },indent=2))


if __name__=="__main__":
    main()
