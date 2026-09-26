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
BEAM = 256
REFINE_K = 8
GRAD_N = 4
REFINE_N = 4
HOLD_N = 16
MAX_LENGTH = 20
BASE_SKIP = 1400
FOLD_STRIDE = 40
FOLD = int(os.environ.get("CASE_FOLD", "0"))
GROUP_SIZE = 128
INDEPENDENT_CODES = {0: 4, 7: 3, 14: 6, 21: 1, 27: 3}
OUT = Path("results-joint-mask")


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
            eligible+=1
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


def score_vec(v):
    v=np.asarray(v,dtype=np.float64)
    return (float(np.maximum(v,0).mean()), float(v.mean()), float(v.max()))


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

    # Raw gradients for all five target matrices on the same gradient texts.
    grads={l:[] for l in LAYERS}
    target_list=[targets[l] for l in LAYERS]
    for i,text in enumerate(grad_texts):
        model.zero_grad(set_to_none=True)
        x=encode(tok,text)
        loss=model(**x,labels=x["input_ids"]).loss
        gs=torch.autograd.grad(loss,target_list,retain_graph=False,create_graph=False)
        for l,g in zip(LAYERS,gs):
            grads[l].append(g.detach().float().reshape(-1))
        print("GRAD",i,flush=True)

    G={l:torch.stack(grads[l]) for l in LAYERS}

    # First-order loss deltas for each layer/code on the gradient texts.
    pred={}
    for l in LAYERS:
        pred[l]={}
        for code in range(NUM_MASKS):
            mw=masked_weight(originals[l],l,code)
            delta=(mw.float()-originals[l].float()).reshape(-1)
            pred[l][code]=(G[l]@delta).double().cpu().numpy()
        del G[l]
        print("PRED_LAYER",l,flush=True)

    # Joint beam search in the additive first-order space.
    beam=[{"codes":{}, "vec":np.zeros(GRAD_N,dtype=np.float64)}]
    for l in LAYERS:
        expanded=[]
        for item in beam:
            for code in range(NUM_MASKS):
                vec=item["vec"]+pred[l][code]
                codes=dict(item["codes"]); codes[l]=code
                expanded.append({"codes":codes,"vec":vec})
        expanded.sort(key=lambda x:score_vec(x["vec"]))
        beam=expanded[:BEAM]
        print("BEAM",l,score_vec(beam[0]["vec"]),beam[0]["codes"],flush=True)

    candidates=beam[:REFINE_K]

    def restore():
        with torch.no_grad():
            for l in LAYERS:
                targets[l].copy_(originals[l])

    def apply_codes(codes):
        restore()
        with torch.no_grad():
            for l in LAYERS:
                targets[l].copy_(masked_weight(originals[l],l,int(codes[l])))

    refined=[]
    for idx,item in enumerate(candidates):
        apply_codes(item["codes"])
        m=metrics(refine_fp,losses(model,tok,refine_texts))
        refined.append({"codes":item["codes"],"prediction":score_vec(item["vec"]),"refine":m})
        print("REFINE",idx,item["codes"],m["mean_positive_delta_nll"],flush=True)

    refined.sort(key=lambda x:(
        x["refine"]["mean_positive_delta_nll"],
        x["refine"]["mean_delta_nll"],
        x["refine"]["max_delta_nll"],
    ))
    selected_codes=refined[0]["codes"]

    comparisons={
        "fixed0":{l:0 for l in LAYERS},
        "independent_previous":INDEPENDENT_CODES,
        "joint_selected":selected_codes,
    }
    holdout={}
    for name,codes in comparisons.items():
        apply_codes(codes)
        holdout[name]=metrics(hold_fp,losses(model,tok,hold_texts))

    restore()
    for t in targets.values():
        t.requires_grad_(False)

    code_bits=len(LAYERS)*int(np.ceil(np.log2(NUM_MASKS)))
    nweights=sum(originals[l].numel() for l in LAYERS)
    storage_bpw=(1.0-FRAC)+16.0/GROUP_SIZE+code_bits/nweights

    payload={
        "model":MODEL_ID,"fold":FOLD,"layers":LAYERS,
        "prune_fraction":FRAC,"num_masks":NUM_MASKS,
        "beam":BEAM,"refine_k":REFINE_K,
        "gradient_n":GRAD_N,"refine_n":REFINE_N,"holdout_n":HOLD_N,
        "selected_codes":selected_codes,
        "refined":refined,
        "holdout":holdout,
        "storage_bpw_across_tested_matrices":float(storage_bpw),
    }
    (OUT/f"fold{FOLD}.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps({
        "fold":FOLD,
        "selected_codes":selected_codes,
        "fixed0":holdout["fixed0"],
        "independent_previous":holdout["independent_previous"],
        "joint_selected":holdout["joint_selected"],
        "storage_bpw":storage_bpw,
    },indent=2))


if __name__=="__main__":
    main()
