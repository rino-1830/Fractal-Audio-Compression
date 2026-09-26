import json, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import waterfill_downproj as wf

CAL_N=4
TARGETS=[0.15,0.20,0.25,0.30,0.35]
UNCERTAINTY_WEIGHT=0.75
OUT=Path("results-robust-waterfill")

def greedy(sens,target):
    layers=sorted(sens)
    steps=int(round(target*len(layers)/0.25))
    state={l:0 for l in layers}
    trace=[]
    for _ in range(steps):
        opts=[]
        for l in layers:
            s=state[l]
            if s>=2: continue
            a=sens[l]["0.25"]
            b=sens[l]["0.5"]
            c=max(a,0.0) if s==0 else max(b-a,0.0)
            opts.append((c,l,s+1))
        c,l,n=min(opts,key=lambda x:(x[0],x[1]))
        state[l]=n
        trace.append({"layer":l,"increment":n,"robust_marginal_cost":c})
    return {l:0.25*state[l] for l in layers},trace

def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    cal_a=wf.collect("validation",700,CAL_N)
    cal_b=wf.collect("validation",1100,CAL_N)
    hold=wf.collect("test",1800,20)

    tok=AutoTokenizer.from_pretrained(wf.MODEL_ID)
    model=AutoModelForCausalLM.from_pretrained(
        wf.MODEL_ID,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True
    )
    model.eval()
    layers=list(range(len(model.model.layers)))
    targets={l:model.model.layers[l].mlp.down_proj.weight for l in layers}
    originals={l:targets[l].detach().clone() for l in layers}

    with torch.no_grad():
        for l in layers: targets[l].copy_(originals[l])
    fp_a=wf.losses(model,tok,cal_a)
    fp_b=wf.losses(model,tok,cal_b)
    fp_h=wf.losses(model,tok,hold)

    raw={}
    robust={}
    for l in layers:
        sc=wf.score_tensor(originals[l].shape,l)
        raw[l]={}; robust[l]={}
        for p in wf.P_LEVELS:
            with torch.no_grad(): targets[l].copy_(wf.thinned(originals[l],sc,p))
            ma=wf.metric(fp_a,wf.losses(model,tok,cal_a))
            mb=wf.metric(fp_b,wf.losses(model,tok,cal_b))
            a=ma["mean_positive_delta_nll"]; b=mb["mean_positive_delta_nll"]
            mean=(a+b)/2.0
            uncertainty=abs(a-b)/2.0
            risk=mean+UNCERTAINTY_WEIGHT*uncertainty
            raw[l][str(p)]={"a":a,"b":b,"mean":mean,"uncertainty":uncertainty,"risk":risk}
            robust[l][str(p)]=risk
            print("SENS",l,p,a,b,risk,flush=True)
            with torch.no_grad(): targets[l].copy_(originals[l])
        del sc

    def restore():
        with torch.no_grad():
            for l in layers: targets[l].copy_(originals[l])

    def apply(rates):
        restore()
        with torch.no_grad():
            for l,p in rates.items():
                if p<=0: continue
                sc=wf.score_tensor(originals[l].shape,l)
                targets[l].copy_(wf.thinned(originals[l],sc,p))
                del sc

    total_weights=sum(p.numel() for p in model.parameters())
    down_weights=sum(x.numel() for x in originals.values())
    frac=down_weights/total_weights
    results={}
    for target in TARGETS:
        rates,trace=greedy(robust,target)
        actual=float(np.mean(list(rates.values())))
        apply(rates)
        wm=wf.metric(fp_h,wf.losses(model,tok,hold))
        apply({l:target for l in layers})
        um=wf.metric(fp_h,wf.losses(model,tok,hold))
        results[str(target)]={
            "rates":rates,"trace":trace,"actual_mean_prune":actual,
            "robust_waterfill":wm,"uniform":um,
            "downproj_bpw":1.0-actual+16.0/wf.GROUP_SIZE,
            "global_bpw_saving_from_signs":actual*frac,
        }
        print("RESULT",target,wm["mean_positive_delta_nll"],um["mean_positive_delta_nll"],flush=True)

    restore()
    payload={
        "model":wf.MODEL_ID,"uncertainty_weight":UNCERTAINTY_WEIGHT,
        "calibration_n_each":CAL_N,"holdout_n":len(hold),
        "downproj_weight_fraction":frac,"raw_sensitivity":raw,"results":results,
    }
    (OUT/"result.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")

if __name__=="__main__":
    main()
