import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import joint_coordinate_selector as jc

MODEL_ID=jc.MODEL_ID
LAYERS=jc.LAYERS
FRAC=jc.FRAC
NUM_MASKS=jc.NUM_MASKS
SHORTLIST=jc.SHORTLIST
SWEEPS=2
GRAD_N=4
REFINE_N=6
GATE_N=6
HOLD_N=16
MAX_LENGTH=20
BASE_SKIP=1150
FOLD_STRIDE=45
FOLD=int(os.environ.get("CASE_FOLD","0"))
GROUP_SIZE=128
OUT=Path("results-joint-coordinate-gate")


def load_texts():
    ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
    xs=[]; eligible=0
    start=BASE_SKIP+FOLD*FOLD_STRIDE
    need=GRAD_N+REFINE_N+GATE_N+HOLD_N
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
    refine=xs[a:a+REFINE_N]; a+=REFINE_N
    gate=xs[a:a+GATE_N]; a+=GATE_N
    hold=xs[a:a+HOLD_N]
    return grad,refine,gate,hold


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    grad_texts,refine_texts,gate_texts,hold_texts=load_texts()

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
    refine_fp=jc.losses(model,tok,refine_texts)
    gate_fp=jc.losses(model,tok,gate_texts)
    hold_fp=jc.losses(model,tok,hold_texts)

    target_list=[targets[l] for l in LAYERS]
    grads={l:[] for l in LAYERS}
    for i,text in enumerate(grad_texts):
        model.zero_grad(set_to_none=True)
        x=jc.encode(tok,text)
        loss=model(**x,labels=x["input_ids"]).loss
        gs=torch.autograd.grad(loss,target_list,retain_graph=False,create_graph=False)
        for l,g in zip(LAYERS,gs):
            grads[l].append(g.detach().float().reshape(-1))
        print("GRAD",i,flush=True)
    G={l:torch.stack(grads[l]) for l in LAYERS}

    shortlists={}
    for l in LAYERS:
        ps=[]
        for code in range(NUM_MASKS):
            mw=jc.masked_weight(originals[l],l,code)
            delta=(mw.float()-originals[l].float()).reshape(-1)
            vec=(G[l]@delta).double().cpu().numpy()
            ps.append((jc.predkey(vec),code))
        ps.sort(key=lambda x:x[0])
        shortlists[l]=[code for _,code in ps[:SHORTLIST]]
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
                targets[l].copy_(jc.masked_weight(originals[l],l,int(codes[l])))

    def eval_on(codes,fp,texts):
        apply_codes(codes)
        return jc.metrics(fp,jc.losses(model,tok,texts))

    codes={l:shortlists[l][0] for l in LAYERS}
    current=eval_on(codes,refine_fp,refine_texts)
    history=[{"stage":"init","codes":dict(codes),"refine":current}]

    for sweep in range(SWEEPS):
        changed=False
        for l in LAYERS:
            best_code=codes[l]
            best_metric=current
            trials=[]
            for code in shortlists[l]:
                trial=dict(codes); trial[l]=code
                tm=eval_on(trial,refine_fp,refine_texts)
                trials.append({"code":code,"metric":tm})
                if jc.mkey(tm)<jc.mkey(best_metric):
                    best_code=code; best_metric=tm
            if best_code!=codes[l]:
                changed=True; codes[l]=best_code; current=best_metric
            history.append({"stage":f"sweep{sweep}_layer{l}","codes":dict(codes),"refine":current,"trials":trials})
        if not changed:
            break

    candidate=dict(codes)
    fixed0={l:0 for l in LAYERS}
    first_order={l:shortlists[l][0] for l in LAYERS}

    gate_metrics={
        "fixed0":eval_on(fixed0,gate_fp,gate_texts),
        "first_order":eval_on(first_order,gate_fp,gate_texts),
        "coordinate":eval_on(candidate,gate_fp,gate_texts),
    }

    # Conservative gate: candidate must beat fixed0 on positive harm and not worsen signed mean.
    accept=(
        gate_metrics["coordinate"]["mean_positive_delta_nll"]
        < gate_metrics["fixed0"]["mean_positive_delta_nll"]
        and gate_metrics["coordinate"]["mean_delta_nll"]
        <= gate_metrics["fixed0"]["mean_delta_nll"]
    )
    selected=candidate if accept else fixed0

    holdout={
        "fixed0":eval_on(fixed0,hold_fp,hold_texts),
        "first_order":eval_on(first_order,hold_fp,hold_texts),
        "coordinate":eval_on(candidate,hold_fp,hold_texts),
        "gated_selected":eval_on(selected,hold_fp,hold_texts),
    }

    restore()
    for t in targets.values():
        t.requires_grad_(False)

    code_bits=len(LAYERS)*int(np.ceil(np.log2(NUM_MASKS)))
    nweights=sum(originals[l].numel() for l in LAYERS)
    storage=(1.0-FRAC)+16.0/GROUP_SIZE+code_bits/nweights

    payload={
        "model":MODEL_ID,"fold":FOLD,"layers":LAYERS,"prune_fraction":FRAC,
        "shortlists":shortlists,"candidate_codes":candidate,
        "gate_metrics":gate_metrics,"gate_accept":bool(accept),
        "selected_codes":selected,"holdout":holdout,
        "history":history,"storage_bpw":float(storage),
        "settings":{"grad_n":GRAD_N,"refine_n":REFINE_N,"gate_n":GATE_N,"holdout_n":HOLD_N},
    }
    (OUT/f"fold{FOLD}.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps({
        "fold":FOLD,"candidate":candidate,"gate_accept":accept,
        "gate":gate_metrics,"holdout":holdout,"storage_bpw":storage
    },indent=2))


if __name__=="__main__":
    main()
