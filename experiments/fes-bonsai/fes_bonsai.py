import argparse, gc, json, math, os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

FACTORS=(0.96,0.98,1.00,1.02,1.04)

def windows(tok, seq_len=24, n_cal=1, n_test=2):
    from datasets import load_dataset
    ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="validation")
    text="\n\n".join(x["text"] for x in ds if x["text"].strip())
    ids=tok(text,return_tensors="pt",add_special_tokens=False)["input_ids"][0]
    out=[]; pos=0
    for _ in range(n_cal+n_test):
        out.append(ids[pos:pos+seq_len].clone()); pos+=seq_len+17
    return out[:n_cal],out[n_cal:]

@torch.inference_mode()
def cache_logits(model, ws):
    return [model(input_ids=x.unsqueeze(0)).logits[0,:-1].float().cpu().half() for x in ws]

@torch.inference_mode()
def metrics(model, ws, ref):
    kl=nll=ref_nll=0.0; agree=tokens=0
    for ids,rh in zip(ws,ref):
        r=rh.float()
        q=model(input_ids=ids.unsqueeze(0)).logits[0,:-1].float().cpu()
        target=ids[1:].cpu()
        rl=F.log_softmax(r,dim=-1); ql=F.log_softmax(q,dim=-1)
        kl+=(rl.exp()*(rl-ql)).sum().item()
        nll+=F.nll_loss(ql,target,reduction="sum").item()
        ref_nll+=F.nll_loss(rl,target,reduction="sum").item()
        agree+=(r.argmax(-1)==q.argmax(-1)).sum().item()
        tokens+=target.numel()
    return {
      "kl_to_qwen":kl/tokens,
      "nll":nll/tokens,
      "ppl":math.exp(min(nll/tokens,20)),
      "qwen_ppl":math.exp(min(ref_nll/tokens,20)),
      "top1_agreement":agree/tokens,
    }

def target_modules(model, n):
    return [(f"layer.{i}.mlp.up_proj",model.model.layers[i].mlp.up_proj) for i in range(n)]

def restore(mods,orig):
    for (_,m),w in zip(mods,orig): m.weight.data.copy_(w.to(m.weight.dtype))

def apply(mods,orig,choices):
    for i,(_,m) in enumerate(mods):
        m.weight.data.copy_((orig[i]*FACTORS[choices[i]]).to(m.weight.dtype))

def sampled_ternary_group_cv(w, group=128, max_groups=512):
    x=w.float().cpu()
    last=x.shape[-1]
    usable=(last//group)*group
    x=x[...,:usable].reshape(-1,group)
    if x.shape[0]>max_groups:
        idx=torch.linspace(0,x.shape[0]-1,max_groups).long()
        x=x[idx]
    cvs=[]; zero_fracs=[]
    for g in x:
        nz=g.abs()[g!=0]
        zero_fracs.append((g==0).float().mean().item())
        if nz.numel()>1:
            mean=nz.mean().item()
            if mean>0: cvs.append(nz.std(unbiased=False).item()/mean)
    return {
      "sampled_nonzero_abs_cv_mean":float(sum(cvs)/max(len(cvs),1)),
      "sampled_zero_fraction_mean":float(sum(zero_fracs)/max(len(zero_fracs),1)),
      "sampled_groups":int(x.shape[0]),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--layers",type=int,default=2)
    ap.add_argument("--out",default="results-bonsai")
    args=ap.parse_args()
    torch.manual_seed(0); torch.set_num_threads(min(os.cpu_count() or 1,8))
    ref_name="Qwen/Qwen3-1.7B"
    bonsai_name="prism-ml/Ternary-Bonsai-1.7B-unpacked"

    tok=AutoTokenizer.from_pretrained(bonsai_name)
    cal,test=windows(tok)

    print("loading reference",flush=True)
    ref_model=AutoModelForCausalLM.from_pretrained(ref_name,dtype=torch.float32,low_cpu_mem_usage=True)
    ref_model.eval()
    ref_mods=target_modules(ref_model,args.layers)
    ref_weights=[m.weight.detach().cpu().half().clone() for _,m in ref_mods]
    ref_cal=cache_logits(ref_model,cal); ref_test=cache_logits(ref_model,test)
    del ref_model,ref_mods; gc.collect()

    print("loading bonsai",flush=True)
    model=AutoModelForCausalLM.from_pretrained(bonsai_name,dtype=torch.float32,low_cpu_mem_usage=True)
    model.eval()
    mods=target_modules(model,args.layers)
    orig=[m.weight.detach().cpu().clone() for _,m in mods]

    conformity=[sampled_ternary_group_cv(w) for w in orig]
    print("conformity",json.dumps(conformity),flush=True)

    local=[]
    local_tables=[]
    for i,w in enumerate(orig):
        rw=ref_weights[i].float()
        vals=[]
        for f in FACTORS:
            vals.append(float(F.mse_loss(w*f,rw).item()))
        local.append(int(min(range(len(vals)),key=lambda j:vals[j])))
        local_tables.append(vals)
        print(f"local layer={i} choice={local[-1]} factor={FACTORS[local[-1]]} mse={vals[local[-1]]:.8g}",flush=True)

    independent=[]
    for i in range(args.layers):
        vals=[]
        for j,f in enumerate(FACTORS):
            restore(mods,orig)
            mods[i][1].weight.data.copy_((orig[i]*f).to(mods[i][1].weight.dtype))
            v=metrics(model,cal,ref_cal)["kl_to_qwen"]; vals.append(v)
            print(f"independent layer={i} factor={f} cal_kl={v:.8g}",flush=True)
        independent.append(int(min(range(len(vals)),key=lambda j:vals[j])))
    restore(mods,orig)

    choices=list(independent)
    for i in range(args.layers):
        vals=[]
        for j,f in enumerate(FACTORS):
            trial=list(choices); trial[i]=j
            apply(mods,orig,trial)
            v=metrics(model,cal,ref_cal)["kl_to_qwen"]; vals.append(v)
            print(f"coordinate layer={i} factor={f} cal_kl={v:.8g}",flush=True)
        choices[i]=int(min(range(len(vals)),key=lambda j:vals[j]))
    restore(mods,orig)

    methods={
      "original_bonsai":[FACTORS.index(1.0)]*args.layers,
      "local_weight_mse":local,
      "independent_functional":independent,
      "fes_coordinate":choices,
    }
    results={}
    for name,ch in methods.items():
        apply(mods,orig,ch)
        results[name]={
          "choices":ch,
          "factors":[FACTORS[j] for j in ch],
          "cal":metrics(model,cal,ref_cal),
          "test":metrics(model,test,ref_test),
          "weight_mse_to_qwen":float(sum(local_tables[i][ch[i]] for i in range(args.layers))/args.layers),
        }
        restore(mods,orig)

    s={
      "reference":ref_name,
      "bonsai":bonsai_name,
      "layers_scaled":args.layers,
      "factors":FACTORS,
      "fes_vs_original_test_kl_ratio":results["fes_coordinate"]["test"]["kl_to_qwen"]/results["original_bonsai"]["test"]["kl_to_qwen"],
      "fes_vs_independent_test_kl_ratio":results["fes_coordinate"]["test"]["kl_to_qwen"]/results["independent_functional"]["test"]["kl_to_qwen"],
      "fes_vs_local_test_kl_ratio":results["fes_coordinate"]["test"]["kl_to_qwen"]/results["local_weight_mse"]["test"]["kl_to_qwen"],
      "same_ternary_codes":True,
      "storage_width_unchanged":True,
    }
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps({"summary":s,"conformity":conformity,"methods":results},indent=2))
    print("SUMMARY",json.dumps(s,sort_keys=True),flush=True)
    for k,v in results.items(): print(k,json.dumps(v),flush=True)

if __name__=="__main__": main()
