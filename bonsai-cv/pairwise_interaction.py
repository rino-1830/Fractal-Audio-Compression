import json
from pathlib import Path
from itertools import combinations

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID="prism-ml/Bonsai-1.7B-unpacked"
LAYERS=[0,7,14,21,27]
SELECTED_CODES={0:4,7:3,14:6,21:1,27:3}
FRAC=0.50
N=16
MAX_LENGTH=20
SKIP_ELIGIBLE=500
OUT=Path("results-pairwise-interaction")


def load_texts():
    ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="validation")
    xs=[]; skipped=0
    for row in ds:
        t=" ".join(row["text"].split())
        if len(t)<100 or t.startswith("="):
            continue
        if skipped<SKIP_ELIGIBLE:
            skipped+=1
            continue
        xs.append(t[:500])
        if len(xs)>=N:
            break
    if len(xs)<N:
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


def mask_weight(w,layer,code):
    x=w.detach().clone()
    g=torch.Generator(device="cpu")
    g.manual_seed(246813579+code*1000003+layer*1009)
    m=torch.rand(x.shape,generator=g)<FRAC
    x[m]=0
    return x


def vec_stats(v):
    v=np.asarray(v,dtype=np.float64)
    return {
      "mean":float(v.mean()),
      "rms":float(np.sqrt(np.mean(v*v))),
      "mean_abs":float(np.abs(v).mean()),
      "max_abs":float(np.abs(v).max()),
      "positive_mean":float(np.maximum(v,0).mean()),
      "vector":v.tolist(),
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

    targets={l:model.model.layers[l].mlp.down_proj.weight for l in LAYERS}
    originals={l:targets[l].detach().clone() for l in LAYERS}

    def restore():
        with torch.no_grad():
            for l in LAYERS:
                targets[l].copy_(originals[l])

    restore()
    fp=losses(model,tok,texts)

    def eval_config(codes):
        restore()
        with torch.no_grad():
            for l,code in codes.items():
                targets[l].copy_(mask_weight(originals[l],l,code))
        return losses(model,tok,texts)-fp

    families={
      "fixed0":{l:0 for l in LAYERS},
      "selected":SELECTED_CODES,
    }
    payload={"model":MODEL_ID,"layers":LAYERS,"n":N,"families":{}}

    for fname,codes in families.items():
        individual={}
        for l in LAYERS:
            individual[l]=eval_config({l:codes[l]})
            print(fname,"IND",l,float(individual[l].mean()),flush=True)

        pairs={}
        rms_matrix=np.zeros((len(LAYERS),len(LAYERS)),dtype=np.float64)
        for i,j in combinations(LAYERS,2):
            pair=eval_config({i:codes[i],j:codes[j]})
            inter=pair-individual[i]-individual[j]
            key=f"{i}-{j}"
            pairs[key]={
              "pair_delta":vec_stats(pair),
              "interaction":vec_stats(inter),
              "additive_prediction":vec_stats(individual[i]+individual[j]),
            }
            a=LAYERS.index(i); b=LAYERS.index(j)
            rms=pairs[key]["interaction"]["rms"]
            rms_matrix[a,b]=rms_matrix[b,a]=rms
            print(fname,"PAIR",i,j,"rms",rms,"mean",pairs[key]["interaction"]["mean"],flush=True)

        all5=eval_config(codes)
        additive=sum(individual.values())
        full_inter=all5-additive
        payload["families"][fname]={
          "codes":codes,
          "individual":{str(l):vec_stats(individual[l]) for l in LAYERS},
          "pairs":pairs,
          "interaction_rms_matrix":rms_matrix.tolist(),
          "all5_delta":vec_stats(all5),
          "all5_additive_prediction":vec_stats(additive),
          "all5_interaction":vec_stats(full_inter),
        }

    restore()
    (OUT/"result.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    for fname in families:
        x=payload["families"][fname]
        print(json.dumps({
          "family":fname,
          "all5_rms":x["all5_delta"]["rms"],
          "interaction_rms":x["all5_interaction"]["rms"],
          "interaction_mean":x["all5_interaction"]["mean"],
          "largest_pairs":sorted(
            [(k,v["interaction"]["rms"]) for k,v in x["pairs"].items()],
            key=lambda z:z[1],reverse=True
          )[:5],
        },indent=2))


if __name__=="__main__":
    main()
