import json, math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

MODEL_ID="prism-ml/Bonsai-1.7B-unpacked"
MODULES=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
OUT=Path("results-crosslayer-sign")


def h2(p):
    if p<=0 or p>=1: return 0.0
    return -p*math.log2(p)-(1-p)*math.log2(1-p)


def getw(block,name):
    if name in {"q_proj","k_proj","v_proj","o_proj"}:
        return getattr(block.self_attn,name).weight
    return getattr(block.mlp,name).weight


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    model=AutoModelForCausalLM.from_pretrained(
        MODEL_ID,torch_dtype=torch.float16,low_cpu_mem_usage=True
    )
    model.eval()

    rows=[]
    for name in MODULES:
        prev=None
        for li,b in enumerate(model.model.layers):
            bits=(getw(b,name).detach().cpu().numpy()>0)
            if prev is not None:
                xor=np.logical_xor(prev,bits)
                p=float(xor.mean())
                rows.append({
                    "module":name,"layer_a":li-1,"layer_b":li,
                    "xor_fraction":p,
                    "delta_entropy_bpw":h2(p),
                    "agreement":1.0-p,
                    "weights":int(bits.size),
                })
            prev=bits

    agg={}
    for name in MODULES:
        xs=[r for r in rows if r["module"]==name]
        total=sum(r["weights"] for r in xs)
        agg[name]={
            "weighted_xor_fraction":sum(r["xor_fraction"]*r["weights"] for r in xs)/total,
            "weighted_delta_entropy_bpw":sum(r["delta_entropy_bpw"]*r["weights"] for r in xs)/total,
            "min_delta_entropy_bpw":min(r["delta_entropy_bpw"] for r in xs),
            "max_agreement":max(r["agreement"] for r in xs),
        }

    total=sum(r["weights"] for r in rows)
    overall={
        "weighted_delta_entropy_bpw":sum(r["delta_entropy_bpw"]*r["weights"] for r in rows)/total,
        "weighted_xor_fraction":sum(r["xor_fraction"]*r["weights"] for r in rows)/total,
    }
    payload={"model":MODEL_ID,"aggregate":agg,"overall":overall,"rows":rows}
    (OUT/"result.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps({"overall":overall,"aggregate":agg},indent=2))


if __name__=="__main__":
    main()
