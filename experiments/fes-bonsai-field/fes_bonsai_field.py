import argparse, gc, itertools, json, math, os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

GROUP=128
CANDIDATE_NAMES=("base","half_ls","full_ls","positive_delta","negative_delta","large_delta","small_delta")

def load_windows(tok, seq_len=24, n_cal=1, n_test=3, offset=0):
    from datasets import load_dataset
    ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="validation")
    text="\n\n".join(x["text"] for x in ds if x["text"].strip())
    ids=tok(text,return_tensors="pt",add_special_tokens=False)["input_ids"][0]
    out=[]; pos=offset
    for _ in range(n_cal+n_test):
        out.append(ids[pos:pos+seq_len].clone()); pos+=seq_len+17
    return out[:n_cal],out[n_cal:]

@torch.inference_mode()
def cache_logits(model,ws,vocab):
    return [model(input_ids=x.unsqueeze(0)).logits[0,:-1,:vocab].float().cpu().half() for x in ws]

@torch.inference_mode()
def metrics(model,ws,ref,vocab):
    kl=nll=ref_nll=0.0; agree=tokens=0
    for ids,rh in zip(ws,ref):
        r=rh.float()
        q=model(input_ids=ids.unsqueeze(0)).logits[0,:-1,:vocab].float().cpu()
        target=ids[1:].cpu()
        rl=F.log_softmax(r,dim=-1); ql=F.log_softmax(q,dim=-1)
        kl+=(rl.exp()*(rl-ql)).sum().item()
        nll+=F.nll_loss(ql,target,reduction="sum").item()
        ref_nll+=F.nll_loss(rl,target,reduction="sum").item()
        agree+=(r.argmax(-1)==q.argmax(-1)).sum().item()
        tokens+=target.numel()
    return {"kl_to_qwen":kl/tokens,"nll":nll/tokens,
            "ppl":math.exp(min(nll/tokens,20)),
            "qwen_ppl":math.exp(min(ref_nll/tokens,20)),
            "top1_agreement":agree/tokens}

def modules(model,n):
    return [(f"layer.{i}.mlp.up_proj",model.model.layers[i].mlp.up_proj) for i in range(n)]

def make_candidates(wb,wr):
    wb=wb.float().cpu(); wr=wr.float().cpu()
    rows,cols=wb.shape; assert cols%GROUP==0
    bg=wb.reshape(rows,cols//GROUP,GROUP)
    rg=wr.reshape(rows,cols//GROUP,GROUP)
    sym=torch.sign(bg)
    s0=bg.abs().amax(-1,keepdim=True)
    nnz=(sym*sym).sum(-1,keepdim=True)
    sls=torch.where(nnz>0,(rg*sym).sum(-1,keepdim=True)/nnz.clamp_min(1),s0).clamp_min(0)
    d=sls-s0
    med=d.abs().median()
    masks={
      "base":torch.zeros_like(d),
      "half_ls":torch.full_like(d,0.5),
      "full_ls":torch.ones_like(d),
      "positive_delta":(d>0).float(),
      "negative_delta":(d<0).float(),
      "large_delta":(d.abs()>=med).float(),
      "small_delta":(d.abs()<med).float(),
    }
    cs=[]; ms=[]
    for name in CANDIDATE_NAMES:
        s=(s0+masks[name]*d).clamp_min(0).half().float()
        q=(sym*s).reshape_as(wb).half().contiguous()
        cs.append(q); ms.append(float(F.mse_loss(q.float(),wr).item()))
    # exact ternary sanity: within every group, all nonzero abs values equal.
    nzmin=torch.where(bg!=0,bg.abs(),torch.tensor(float("inf"))).amin(-1)
    nzmax=bg.abs().amax(-1)
    spread=torch.where((bg!=0).any(-1),(nzmax-nzmin).abs(),torch.zeros_like(nzmax)).max().item()
    return cs,ms,{
      "max_nonzero_group_spread":float(spread),
      "delta_abs_mean":float(d.abs().mean().item()),
      "delta_positive_fraction":float((d>0).float().mean().item()),
      "delta_negative_fraction":float((d<0).float().mean().item()),
    }

def apply(mods,sets,choices):
    for i,(_,m) in enumerate(mods):
        m.weight.data.copy_(sets[i][choices[i]].to(m.weight.dtype))

def eval_choice(model,mods,sets,choices,cal,test,ref_cal,ref_test,vocab,mses):
    apply(mods,sets,choices)
    return {"choices":list(choices),
            "names":[CANDIDATE_NAMES[j] for j in choices],
            "weight_mse_to_qwen":float(sum(mses[i][choices[i]] for i in range(len(choices)))/len(choices)),
            "cal":metrics(model,cal,ref_cal,vocab),
            "test":metrics(model,test,ref_test,vocab)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--layers",type=int,default=3)
    ap.add_argument("--offset",type=int,default=0)
    ap.add_argument("--out",default="results-bonsai-field")
    args=ap.parse_args()
    torch.manual_seed(0); torch.set_num_threads(min(os.cpu_count() or 1,8))
    ref_name="Qwen/Qwen3-1.7B"; bon_name="prism-ml/Ternary-Bonsai-1.7B-unpacked"

    tok=AutoTokenizer.from_pretrained(bon_name); vocab=len(tok.get_vocab())
    cal,test=load_windows(tok,offset=args.offset)

    ref=AutoModelForCausalLM.from_pretrained(ref_name,dtype=torch.float32,low_cpu_mem_usage=True); ref.eval()
    rmods=modules(ref,args.layers)
    rweights=[m.weight.detach().cpu().half().clone() for _,m in rmods]
    ref_cal=cache_logits(ref,cal,vocab); ref_test=cache_logits(ref,test,vocab)
    del ref,rmods; gc.collect()

    model=AutoModelForCausalLM.from_pretrained(bon_name,dtype=torch.float32,low_cpu_mem_usage=True); model.eval()
    mods=modules(model,args.layers)
    bweights=[m.weight.detach().cpu().clone() for _,m in mods]
    sets=[]; mses=[]; sanity=[]
    for i,(wb,wr) in enumerate(zip(bweights,rweights)):
        cs,ms,st=make_candidates(wb,wr); sets.append(cs); mses.append(ms); sanity.append(st)
        print("layer",i,"mse",dict(zip(CANDIDATE_NAMES,ms)),"sanity",st,flush=True)

    base=[0]*args.layers
    local=[min(range(len(CANDIDATE_NAMES)),key=lambda j:mses[i][j]) for i in range(args.layers)]

    independent=[]
    for i in range(args.layers):
        vals=[]
        for j,name in enumerate(CANDIDATE_NAMES):
            trial=list(base); trial[i]=j
            apply(mods,sets,trial)
            v=metrics(model,cal,ref_cal,vocab)["kl_to_qwen"]; vals.append(v)
            print(f"independent layer={i} candidate={name} cal_kl={v:.8g}",flush=True)
        independent.append(min(range(len(vals)),key=lambda j:vals[j]))

    grid=[]
    for choices in itertools.product(range(len(CANDIDATE_NAMES)),repeat=args.layers):
        apply(mods,sets,choices)
        v=metrics(model,cal,ref_cal,vocab)["kl_to_qwen"]
        grid.append((v,choices))
        print("grid",choices,f"{v:.8g}",flush=True)
    grid.sort(key=lambda x:x[0])
    global_choice=list(grid[0][1])

    methods={
      "original_bonsai":eval_choice(model,mods,sets,base,cal,test,ref_cal,ref_test,vocab,mses),
      "local_weight_mse":eval_choice(model,mods,sets,local,cal,test,ref_cal,ref_test,vocab,mses),
      "independent_functional":eval_choice(model,mods,sets,independent,cal,test,ref_cal,ref_test,vocab,mses),
      "fes_exact_global":eval_choice(model,mods,sets,global_choice,cal,test,ref_cal,ref_test,vocab,mses),
    }
    g=methods["fes_exact_global"]; ind=methods["independent_functional"]; loc=methods["local_weight_mse"]; base_m=methods["original_bonsai"]
    summary={
      "offset":args.offset,"layers":args.layers,"candidate_names":CANDIDATE_NAMES,
      "same_ternary_codes":True,"existing_fp16_scales_only":True,"extra_storage_bits":0,
      "global_differs_from_independent":global_choice!=independent,
      "heldout_interaction_gain":g["test"]["kl_to_qwen"]<ind["test"]["kl_to_qwen"],
      "global_vs_independent_test_kl_ratio":g["test"]["kl_to_qwen"]/ind["test"]["kl_to_qwen"],
      "global_vs_original_test_kl_ratio":g["test"]["kl_to_qwen"]/base_m["test"]["kl_to_qwen"],
      "global_vs_local_test_kl_ratio":g["test"]["kl_to_qwen"]/loc["test"]["kl_to_qwen"],
      "global_weight_mse_ratio_vs_local":g["weight_mse_to_qwen"]/loc["weight_mse_to_qwen"],
    }
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps({"summary":summary,"sanity":sanity,"methods":methods,
      "top_grid":[{"cal_kl":v,"choices":list(c),"names":[CANDIDATE_NAMES[j] for j in c]} for v,c in grid[:20]]},indent=2))
    print("SUMMARY",json.dumps(summary,sort_keys=True),flush=True)
    for k,v in methods.items(): print(k,json.dumps(v),flush=True)

if __name__=="__main__": main()
