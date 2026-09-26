import json
import math
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "prism-ml/Bonsai-1.7B-unpacked"
MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
CAPS = {
    "q_proj": 0.1875,
    "k_proj": 0.25,
    "v_proj": 0.125,
    "o_proj": 0.1875,
    "gate_proj": 0.25,
    "up_proj": 0.25,
    "down_proj": 0.25,
}
STEP = 0.03125
TARGETS = [0.02, 0.04, 0.06, 0.08]
CAL_N = 8
HOLD_N = 20
MAX_LENGTH = 20
GROUP_SIZE = 128
OUT = Path("results-kfac-all-linear-lowrate")


def collect(split, skip, n):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    xs=[]; eligible=0
    for row in ds:
        t=" ".join(row["text"].split())
        if len(t)<100 or t.startswith("="):
            continue
        if eligible<skip:
            eligible += 1
            continue
        xs.append(t[:500])
        if len(xs)>=n:
            break
    if len(xs)<n:
        raise RuntimeError("not enough text")
    return xs


def enc(tok,text):
    return tok(text,return_tensors="pt",truncation=True,max_length=MAX_LENGTH)


@torch.no_grad()
def losses(model,tok,texts):
    vals=[]
    for t in texts:
        x=enc(tok,t)
        vals.append(float(model(**x,labels=x["input_ids"]).loss))
    return np.asarray(vals,dtype=np.float64)


def metric(fp,vals):
    d=vals-fp
    return {
        "mean_delta_nll":float(d.mean()),
        "mean_positive_delta_nll":float(np.maximum(d,0).mean()),
        "median_delta_nll":float(np.median(d)),
        "p90_delta_nll":float(np.quantile(d,0.9)),
        "max_delta_nll":float(d.max()),
        "fraction_worse":float(np.mean(d>0)),
        "per_sample_delta":d.tolist(),
    }


def get_module(block,name):
    if name in {"q_proj","k_proj","v_proj","o_proj"}:
        return getattr(block.self_attn,name)
    return getattr(block.mlp,name)


def perturb_factor(p):
    return 2.0*(1.0-math.sqrt(1.0-p))


def thinned(orig,layer,name,p):
    if p<=0:
        return orig
    g=torch.Generator(device="cpu")
    g.manual_seed(135791357 + layer*1009 + MODULES.index(name)*100003)
    x=orig.detach().clone()
    mask=torch.rand(x.shape,generator=g)<p
    x[mask]=0
    x=x.float()/math.sqrt(1.0-p)
    return x.to(orig.dtype)


def levels_for(name):
    n=int(round(CAPS[name]/STEP))
    return [i*STEP for i in range(n+1)]


def greedy(scores,target):
    keys=sorted(scores,key=lambda k:(k[0],MODULES.index(k[1])))
    total=sum(scores[k]["weights"] for k in keys)
    target_removed=target*total
    state={k:0 for k in keys}
    removed=0.0
    trace=[]
    while removed+1e-9<target_removed:
        opts=[]
        for k in keys:
            lv=levels_for(k[1])
            idx=state[k]
            if idx>=len(lv)-1:
                continue
            p0,p1=lv[idx],lv[idx+1]
            w=scores[k]["weights"]
            saved=(p1-p0)*w
            c0=perturb_factor(p0)*scores[k]["kfac_energy"]
            c1=perturb_factor(p1)*scores[k]["kfac_energy"]
            marginal=max(c1-c0,0.0)
            opts.append((marginal/max(saved,1.0),marginal,-saved,k,idx+1))
        if not opts:
            break
        _,marginal,neg_saved,k,new_idx=min(opts)
        lv=levels_for(k[1])
        p0,p1=lv[state[k]],lv[new_idx]
        saved=(p1-p0)*scores[k]["weights"]
        state[k]=new_idx
        removed += saved
        trace.append({
            "layer":k[0],"module":k[1],"new_rate":p1,
            "marginal_cost":marginal,"bits_saved":saved,
        })
    rates={k:levels_for(k[1])[state[k]] for k in keys}
    actual=sum(rates[k]*scores[k]["weights"] for k in keys)/total
    return rates,actual,trace


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    cal=collect("validation",1200,CAL_N)
    hold=collect("train",27000,HOLD_N)

    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    model=AutoModelForCausalLM.from_pretrained(
        MODEL_ID,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    targets={}
    for li,b in enumerate(model.model.layers):
        for name in MODULES:
            targets[(li,name)]=get_module(b,name)

    accum={}
    handles=[]

    def emb_hook(module,inp,out):
        out.requires_grad_(True)
        return out
    handles.append(model.model.embed_tokens.register_forward_hook(emb_hook))

    for key,module in targets.items():
        accum[key]={
            "a_sum":torch.zeros(module.weight.shape[1],dtype=torch.float64),
            "g_sum":torch.zeros(module.weight.shape[0],dtype=torch.float64),
            "a_count":0,"g_count":0,
        }
        def make_hook(k):
            def hook(mod,inp,out):
                x=inp[0].detach().float().reshape(-1,inp[0].shape[-1])
                a=accum[k]
                a["a_sum"] += x.square().sum(dim=0).double().cpu()
                a["a_count"] += x.shape[0]
                if out.requires_grad:
                    def gh(grad):
                        g=grad.detach().float().reshape(-1,grad.shape[-1])
                        a["g_sum"] += g.square().sum(dim=0).double().cpu()
                        a["g_count"] += g.shape[0]
                    out.register_hook(gh)
            return hook
        handles.append(module.register_forward_hook(make_hook(key)))

    for i,text in enumerate(cal):
        x=enc(tok,text)
        out=model(**x,labels=x["input_ids"])
        out.loss.backward()
        print("CAL",i,float(out.loss),flush=True)

    for h in handles:
        h.remove()

    scores={}
    for key,module in targets.items():
        a=accum[key]
        A=(a["a_sum"]/max(a["a_count"],1)).float()
        G=(a["g_sum"]/max(a["g_count"],1)).float()
        W2=module.weight.detach().float().square()
        energy=float(torch.dot(G,torch.mv(W2,A)))
        scores[key]={
            "kfac_energy":energy,
            "energy_per_weight":energy/module.weight.numel(),
            "weights":int(module.weight.numel()),
            "cap":CAPS[key[1]],
        }
        print("SCORE",key[0],key[1],scores[key]["energy_per_weight"],flush=True)

    del accum
    originals={k:m.weight.detach().clone() for k,m in targets.items()}

    with torch.no_grad():
        for k,m in targets.items():
            m.weight.copy_(originals[k])
    fp=losses(model,tok,hold)

    def restore():
        with torch.no_grad():
            for k,m in targets.items():
                m.weight.copy_(originals[k])

    def apply(rates):
        restore()
        with torch.no_grad():
            for k,p in rates.items():
                if p<=0: continue
                li,name=k
                targets[k].weight.copy_(thinned(originals[k],li,name,p))

    total_model_weights=sum(p.numel() for p in model.parameters())
    linear_weights=sum(w.numel() for w in originals.values())
    linear_fraction=linear_weights/total_model_weights

    results={}
    for target in TARGETS:
        rates,actual,trace=greedy(scores,target)
        apply(rates)
        km=metric(fp,losses(model,tok,hold))

        # Type-aware uniform baseline: same scalar target clipped to each module cap.
        urates={k:min(target,CAPS[k[1]]) for k in scores}
        uactual=sum(urates[k]*scores[k]["weights"] for k in scores)/linear_weights
        apply(urates)
        um=metric(fp,losses(model,tok,hold))

        hist={}
        for name in MODULES:
            vals=[rates[k] for k in rates if k[1]==name]
            hist[name]={str(x):vals.count(x) for x in sorted(set(vals))}

        results[str(target)]={
            "actual_weighted_prune":actual,
            "kfac":km,
            "type_capped_uniform_actual":uactual,
            "type_capped_uniform":um,
            "global_sign_bpw_saving":actual*linear_fraction,
            "linear_fraction_of_model":linear_fraction,
            "rate_histogram_by_module":hist,
            "trace":trace,
        }
        print("RESULT",target,actual,
              km["mean_positive_delta_nll"],um["mean_positive_delta_nll"],
              actual*linear_fraction,flush=True)

    restore()
    payload={
        "model":MODEL_ID,"modules":MODULES,"caps":CAPS,"step":STEP,
        "targets":TARGETS,"calibration_n":CAL_N,"holdout_n":HOLD_N,
        "linear_weight_fraction":linear_fraction,
        "scores":{f"{k[0]}:{k[1]}":v for k,v in scores.items()},
        "results":results,
    }
    (OUT/"result.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
