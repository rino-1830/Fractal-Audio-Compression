import argparse, gc, itertools, json, math, os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ALPHAS=(0.0,0.25,0.5,0.75,1.0)
GROUP=128

def load_windows(tok, seq_len=24, n_cal=1, n_test=3, offset=0):
    from datasets import load_dataset
    ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="validation")
    text="\n\n".join(x["text"] for x in ds if x["text"].strip())
    ids=tok(text,return_tensors="pt",add_special_tokens=False)["input_ids"][0]
    out=[]; pos=offset
    for _ in range(n_cal+n_test):
        out.append(ids[pos:pos+seq_len].clone()); pos += seq_len+17
    return out[:n_cal],out[n_cal:]

@torch.inference_mode()
def cache_logits(model, ws, common_vocab):
    return [model(input_ids=x.unsqueeze(0)).logits[0,:-1,:common_vocab].float().cpu().half() for x in ws]

@torch.inference_mode()
def kl_metrics(model, ws, ref, common_vocab):
    kl=nll=ref_nll=0.0; agree=tokens=0
    for ids,rh in zip(ws,ref):
        r=rh.float()
        q=model(input_ids=ids.unsqueeze(0)).logits[0,:-1,:common_vocab].float().cpu()
        target=ids[1:].cpu()
        rl=F.log_softmax(r,dim=-1); ql=F.log_softmax(q,dim=-1)
        kl += (rl.exp()*(rl-ql)).sum().item()
        nll += F.nll_loss(ql,target,reduction="sum").item()
        ref_nll += F.nll_loss(rl,target,reduction="sum").item()
        agree += (r.argmax(-1)==q.argmax(-1)).sum().item()
        tokens += target.numel()
    return {
      "kl_to_qwen":kl/tokens,
      "nll":nll/tokens,
      "ppl":math.exp(min(nll/tokens,20)),
      "qwen_ppl":math.exp(min(ref_nll/tokens,20)),
      "top1_agreement":agree/tokens,
    }

def modules(model,n):
    return [(f"layer.{i}.mlp.up_proj",model.model.layers[i].mlp.up_proj) for i in range(n)]

def fixed_code_candidates(w_bonsai, w_ref):
    # Bonsai Q2_0 groups along the last dimension in blocks of 128.
    wb=w_bonsai.float().cpu()
    wr=w_ref.float().cpu()
    rows,cols=wb.shape
    assert cols % GROUP == 0
    bg=wb.reshape(rows,cols//GROUP,GROUP)
    rg=wr.reshape(rows,cols//GROUP,GROUP)

    symbols=torch.sign(bg)
    # Existing Bonsai group scale. In a genuine ternary group all non-zero abs
    # magnitudes are identical, so max(abs(.)) recovers the scale exactly.
    s0=bg.abs().amax(dim=-1,keepdim=True)
    nnz=(symbols*symbols).sum(dim=-1,keepdim=True)
    # Least-squares scale to the Qwen reference with ternary codes held fixed.
    s1=torch.where(nnz>0,(rg*symbols).sum(dim=-1,keepdim=True)/nnz.clamp_min(1),s0)
    s1=s1.clamp_min(0)

    # Simulate packed representation: scales are stored as FP16.
    candidates=[]; mses=[]
    for a in ALPHAS:
        s=((1-a)*s0+a*s1).half().float()
        q=(symbols*s).reshape_as(wb).half().contiguous()
        candidates.append(q)
        mses.append(float(F.mse_loss(q.float(),wr).item()))

    nz=bg[bg!=0].abs()
    # Exact group-format sanity check sampled globally.
    conformity=float((bg.abs().amax(-1)-torch.where(
        (bg!=0).any(-1),
        torch.where(bg!=0,bg.abs(),torch.tensor(float("inf"))).amin(-1),
        bg.abs().amax(-1)
    )).abs().max().item())
    return candidates,mses,{
      "max_group_nonzero_magnitude_spread":conformity,
      "base_scale_mean":float(s0.mean().item()),
      "reference_ls_scale_mean":float(s1.mean().item()),
    }

def restore(mods,orig):
    for (_,m),w in zip(mods,orig): m.weight.data.copy_(w.to(m.weight.dtype))

def apply(mods,candidates,choices):
    for i,(_,m) in enumerate(mods):
        m.weight.data.copy_(candidates[i][choices[i]].to(m.weight.dtype))

def evaluate(model,mods,candidates,choices,cal,test,ref_cal,ref_test,common_vocab,mses):
    apply(mods,candidates,choices)
    out={
      "choices":list(choices),
      "alphas":[ALPHAS[j] for j in choices],
      "weight_mse_to_qwen":float(sum(mses[i][choices[i]] for i in range(len(choices)))/len(choices)),
      "cal":kl_metrics(model,cal,ref_cal,common_vocab),
      "test":kl_metrics(model,test,ref_test,common_vocab),
    }
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--layers",type=int,default=3)
    ap.add_argument("--offset",type=int,default=0)
    ap.add_argument("--out",default="results-bonsai-groups")
    args=ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(min(os.cpu_count() or 1,8))
    ref_name="Qwen/Qwen3-1.7B"
    bonsai_name="prism-ml/Ternary-Bonsai-1.7B-unpacked"

    tok=AutoTokenizer.from_pretrained(bonsai_name)
    common_vocab=len(tok.get_vocab())
    cal,test=load_windows(tok,offset=args.offset)

    print("load reference",flush=True)
    ref_model=AutoModelForCausalLM.from_pretrained(ref_name,dtype=torch.float32,low_cpu_mem_usage=True)
    ref_model.eval()
    ref_mods=modules(ref_model,args.layers)
    ref_weights=[m.weight.detach().cpu().half().clone() for _,m in ref_mods]
    ref_cal=cache_logits(ref_model,cal,common_vocab)
    ref_test=cache_logits(ref_model,test,common_vocab)
    del ref_model,ref_mods
    gc.collect()

    print("load bonsai",flush=True)
    model=AutoModelForCausalLM.from_pretrained(bonsai_name,dtype=torch.float32,low_cpu_mem_usage=True)
    model.eval()
    mods=modules(model,args.layers)
    originals=[m.weight.detach().cpu().clone() for _,m in mods]

    candidate_sets=[]; mse_sets=[]; sanity=[]
    for i,(wb,wr) in enumerate(zip(originals,ref_weights)):
        cs,ms,st=fixed_code_candidates(wb,wr)
        candidate_sets.append(cs); mse_sets.append(ms); sanity.append(st)
        print("layer",i,"mse",ms,"sanity",st,flush=True)

    # Local weight-optimal alpha per layer.
    local=[min(range(len(ALPHAS)),key=lambda j:mse_sets[i][j]) for i in range(args.layers)]

    # Independent functional optimization: vary one layer from original Bonsai.
    baseline=[0]*args.layers
    independent=[]
    for i in range(args.layers):
        vals=[]
        for j in range(len(ALPHAS)):
            trial=list(baseline); trial[i]=j
            apply(mods,candidate_sets,trial)
            v=kl_metrics(model,cal,ref_cal,common_vocab)["kl_to_qwen"]
            vals.append(v)
            print(f"independent layer={i} alpha={ALPHAS[j]} cal_kl={v:.8g}",flush=True)
        independent.append(min(range(len(ALPHAS)),key=lambda j:vals[j]))

    # Exact Cartesian joint search. This is feasible for 3 layers x 5 choices
    # and avoids attributing a coordinate-descent artifact to FES.
    grid=[]
    for choices in itertools.product(range(len(ALPHAS)),repeat=args.layers):
        apply(mods,candidate_sets,choices)
        v=kl_metrics(model,cal,ref_cal,common_vocab)["kl_to_qwen"]
        grid.append((v,choices))
        print("grid",choices,f"{v:.8g}",flush=True)
    grid.sort(key=lambda x:x[0])
    global_choices=list(grid[0][1])

    methods={
      "original_bonsai":evaluate(model,mods,candidate_sets,baseline,cal,test,ref_cal,ref_test,common_vocab,mse_sets),
      "local_weight_mse":evaluate(model,mods,candidate_sets,local,cal,test,ref_cal,ref_test,common_vocab,mse_sets),
      "independent_functional":evaluate(model,mods,candidate_sets,independent,cal,test,ref_cal,ref_test,common_vocab,mse_sets),
      "fes_exact_global":evaluate(model,mods,candidate_sets,global_choices,cal,test,ref_cal,ref_test,common_vocab,mse_sets),
    }
    g=methods["fes_exact_global"]; ind=methods["independent_functional"]; loc=methods["local_weight_mse"]; orig=methods["original_bonsai"]
    summary={
      "reference":ref_name,
      "bonsai":bonsai_name,
      "layers":args.layers,
      "offset":args.offset,
      "group_size":GROUP,
      "alphas":ALPHAS,
      "same_ternary_codes":True,
      "existing_fp16_scales_only":True,
      "extra_storage_bits":0,
      "global_vs_independent_test_kl_ratio":g["test"]["kl_to_qwen"]/ind["test"]["kl_to_qwen"],
      "global_vs_local_test_kl_ratio":g["test"]["kl_to_qwen"]/loc["test"]["kl_to_qwen"],
      "global_vs_original_test_kl_ratio":g["test"]["kl_to_qwen"]/orig["test"]["kl_to_qwen"],
      "global_differs_from_independent":global_choices!=independent,
      "heldout_interaction_gain":g["test"]["kl_to_qwen"]<ind["test"]["kl_to_qwen"],
      "global_weight_mse_ratio_vs_local":g["weight_mse_to_qwen"]/loc["weight_mse_to_qwen"],
    }

    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps({
      "summary":summary,"sanity":sanity,"methods":methods,
      "top_grid":[{"cal_kl":v,"choices":list(c),"alphas":[ALPHAS[j] for j in c]} for v,c in grid[:10]]
    },indent=2))
    print("SUMMARY",json.dumps(summary,sort_keys=True),flush=True)
    for k,v in methods.items(): print(k,json.dumps(v),flush=True)

if __name__=="__main__":
    main()
