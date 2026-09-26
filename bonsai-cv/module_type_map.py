import json, math
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID="prism-ml/Bonsai-1.7B-unpacked"
LAYERS=[0,14,27]
MODULES=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
RATES=[0.25,0.50]
CAL_N=4
HOLD_N=8
MAX_LENGTH=24
GROUP_SIZE=128
OUT=Path("results-module-type-map")

def collect(split,skip,n):
    ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split=split)
    xs=[]; k=0
    for row in ds:
        t=" ".join(row["text"].split())
        if len(t)<100 or t.startswith("="): continue
        if k<skip: k+=1; continue
        xs.append(t[:500])
        if len(xs)>=n: break
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

def metric(fp,v):
    d=v-fp
    return {
      "mean_delta_nll":float(d.mean()),
      "mean_positive_delta_nll":float(np.maximum(d,0).mean()),
      "max_delta_nll":float(d.max()),
      "fraction_worse":float(np.mean(d>0)),
    }

def get_weight(model,layer,name):
    b=model.model.layers[layer]
    if name in {"q_proj","k_proj","v_proj","o_proj"}:
        return getattr(b.self_attn,name).weight
    return getattr(b.mlp,name).weight

def masked(orig,layer,name,p):
    g=torch.Generator(device="cpu")
    seed=975310864 + layer*1009 + MODULES.index(name)*100003
    g.manual_seed(seed)
    x=orig.clone()
    m=torch.rand(x.shape,generator=g)<p
    x[m]=0
    x=x.float()/math.sqrt(1.0-p)
    return x.to(orig.dtype)

def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    cal=collect("validation",1500,CAL_N)
    hold=collect("test",2200,HOLD_N)
    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    model=AutoModelForCausalLM.from_pretrained(
      MODEL_ID,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True
    )
    model.eval()
    cal_fp=losses(model,tok,cal); hold_fp=losses(model,tok,hold)

    rows=[]
    for layer in LAYERS:
      for name in MODULES:
        target=get_weight(model,layer,name)
        orig=target.detach().clone()
        entry={"layer":layer,"module":name,"shape":list(orig.shape),"weights":orig.numel(),"rates":{}}
        for p in RATES:
            with torch.no_grad(): target.copy_(masked(orig,layer,name,p))
            cm=metric(cal_fp,losses(model,tok,cal))
            hm=metric(hold_fp,losses(model,tok,hold))
            entry["rates"][str(p)]={"calibration":cm,"holdout":hm,"local_bpw":1.0-p+16.0/GROUP_SIZE}
            print("ROW",layer,name,p,cm["mean_positive_delta_nll"],hm["mean_positive_delta_nll"],flush=True)
            with torch.no_grad(): target.copy_(orig)
        rows.append(entry)

    aggregate={}
    for name in MODULES:
        aggregate[name]={}
        subset=[r for r in rows if r["module"]==name]
        for p in RATES:
            k=str(p)
            aggregate[name][k]={
                "mean_calibration_positive":float(np.mean([r["rates"][k]["calibration"]["mean_positive_delta_nll"] for r in subset])),
                "mean_holdout_positive":float(np.mean([r["rates"][k]["holdout"]["mean_positive_delta_nll"] for r in subset])),
                "mean_holdout_delta":float(np.mean([r["rates"][k]["holdout"]["mean_delta_nll"] for r in subset])),
            }

    payload={"model":MODEL_ID,"layers":LAYERS,"modules":MODULES,"rates":RATES,"rows":rows,"aggregate":aggregate}
    (OUT/"result.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")

if __name__=="__main__":
    main()
