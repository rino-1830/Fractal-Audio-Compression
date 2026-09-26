import argparse, gc, itertools, json, math, os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

GROUP=128
FRACTIONS=(0.0,0.00025,0.0005,0.001,0.002,0.005,0.01)

def load_windows(tok, seq_len=16, n_cal=1, n_test=3, offset=0):
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
    kl=nll=rnll=0.0; agree=tokens=0
    for ids,rh in zip(ws,ref):
        r=rh.float()
        q=model(input_ids=ids.unsqueeze(0)).logits[0,:-1,:vocab].float().cpu()
        target=ids[1:].cpu()
        rl=F.log_softmax(r,dim=-1); ql=F.log_softmax(q,dim=-1)
        kl+=(rl.exp()*(rl-ql)).sum().item()
        nll+=F.nll_loss(ql,target,reduction="sum").item()
        rnll+=F.nll_loss(rl,target,reduction="sum").item()
        agree+=(r.argmax(-1)==q.argmax(-1)).sum().item()
        tokens+=target.numel()
    return {"kl_to_qwen":kl/tokens,"nll":nll/tokens,
            "ppl":math.exp(min(nll/tokens,20)),
            "qwen_ppl":math.exp(min(rnll/tokens,20)),
            "top1_agreement":agree/tokens}

def target_modules(model,n):
    return [(f"layer.{i}.mlp.up_proj",model.model.layers[i].mlp.up_proj) for i in range(n)]

def calibration_gradients(model,mods,cal_ids,ref_logits,vocab):
    for p in model.parameters():
        p.requires_grad_(False)
    for _,m in mods:
        m.weight.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    q=model(input_ids=cal_ids.unsqueeze(0)).logits[0,:-1,:vocab].float()
    r=ref_logits.float().to(q.device)
    rl=F.log_softmax(r,dim=-1)
    ql=F.log_softmax(q,dim=-1)
    loss=(rl.exp()*(rl-ql)).sum(dim=-1).mean()
    print("gradient_loss",float(loss.item()),flush=True)
    loss.backward()
    grads=[m.weight.grad.detach().float().cpu().clone() for _,m in mods]
    for _,m in mods:
        m.weight.grad=None
        m.weight.requires_grad_(False)
    return grads

def gradient_flip_candidates(wb,wr,grad):
    wb=wb.float().cpu(); wr=wr.float().cpu(); grad=grad.float().cpu()
    rows,cols=wb.shape
    assert cols%GROUP==0
    bg=wb.reshape(rows,cols//GROUP,GROUP)
    gg=grad.reshape_as(bg)
    scale=bg.abs().amax(-1,keepdim=True).expand_as(bg)
    cur=bg

    # Candidate alternative values with the current group scale held fixed.
    vzero=torch.zeros_like(cur)
    vpos=scale
    vneg=-scale
    inf=torch.tensor(float("inf"))
    d0=gg*(vzero-cur); d0=torch.where(cur==0,inf,d0)
    dp=gg*(vpos-cur);  dp=torch.where(cur>0,inf,dp)
    dn=gg*(vneg-cur);  dn=torch.where(cur<0,inf,dn)
    stack=torch.stack((d0,dp,dn),dim=0)
    best_delta,best_alt=stack.min(dim=0)
    flat_delta=best_delta.reshape(-1)
    flat_alt=best_alt.reshape(-1)
    flat_scale=scale.reshape(-1)

    improving=torch.nonzero(flat_delta<0,as_tuple=False).reshape(-1)
    if improving.numel()==0:
        improving=torch.tensor([],dtype=torch.long)
        order=improving
    else:
        vals=-flat_delta[improving]
        order=improving[torch.argsort(vals,descending=True)]

    candidates=[]; mses=[]; stats=[]
    n=wb.numel()
    for frac in FRACTIONS:
        k=min(int(round(frac*n)),order.numel())
        q=wb.half().reshape(-1).clone()
        if k>0:
            idx=order[:k]
            alt=flat_alt[idx]
            sc=flat_scale[idx].half()
            new=torch.where(alt==0,torch.zeros_like(sc),torch.where(alt==1,sc,-sc))
            q[idx]=new
        q=q.reshape_as(wb).contiguous()
        candidates.append(q)
        mses.append(float(F.mse_loss(q.float(),wr).item()))
        est=float(flat_delta[order[:k]].sum().item()) if k>0 else 0.0
        stats.append({"fraction":frac,"flips":int(k),"estimated_first_order_delta":est})
    spread=(torch.where(bg!=0,bg.abs(),torch.tensor(float("inf"))).amin(-1)-bg.abs().amax(-1))
    spread=torch.where((bg!=0).any(-1),spread.abs(),torch.zeros_like(spread)).max().item()
    return candidates,mses,stats,{"max_group_nonzero_spread":float(spread),"improving_alternatives":int(improving.numel())}

def apply(mods,sets,choices):
    for i,(_,m) in enumerate(mods):
        m.weight.data.copy_(sets[i][choices[i]].to(m.weight.dtype))

def eval_choice(model,mods,sets,choices,cal,test,ref_cal,ref_test,vocab,mses):
    apply(mods,sets,choices)
    return {"choices":list(choices),"fractions":[FRACTIONS[j] for j in choices],
            "weight_mse_to_qwen":float(sum(mses[i][choices[i]] for i in range(len(choices)))/len(choices)),
            "cal":metrics(model,cal,ref_cal,vocab),
            "test":metrics(model,test,ref_test,vocab)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--layers",type=int,default=3)
    ap.add_argument("--offset",type=int,default=0)
    ap.add_argument("--out",default="results-bonsai-grad")
    args=ap.parse_args()
    torch.manual_seed(0); torch.set_num_threads(min(os.cpu_count() or 1,8))
    ref_name="Qwen/Qwen3-1.7B"; bon_name="prism-ml/Ternary-Bonsai-1.7B-unpacked"

    tok=AutoTokenizer.from_pretrained(bon_name); vocab=len(tok.get_vocab())
    cal,test=load_windows(tok,offset=args.offset)

    print("load reference",flush=True)
    ref=AutoModelForCausalLM.from_pretrained(ref_name,dtype=torch.float32,low_cpu_mem_usage=True); ref.eval()
    rmods=target_modules(ref,args.layers)
    rweights=[m.weight.detach().cpu().half().clone() for _,m in rmods]
    ref_cal=cache_logits(ref,cal,vocab); ref_test=cache_logits(ref,test,vocab)
    del ref,rmods; gc.collect()

    print("load bonsai",flush=True)
    model=AutoModelForCausalLM.from_pretrained(bon_name,dtype=torch.float32,low_cpu_mem_usage=True); model.eval()
    mods=target_modules(model,args.layers)
    bweights=[m.weight.detach().cpu().clone() for _,m in mods]

    grads=calibration_gradients(model,mods,cal[0],ref_cal[0],vocab)
    sets=[]; mses=[]; candidate_stats=[]; sanity=[]
    for i,(wb,wr,g) in enumerate(zip(bweights,rweights,grads)):
        cs,mm,st,sa=gradient_flip_candidates(wb,wr,g)
        sets.append(cs); mses.append(mm); candidate_stats.append(st); sanity.append(sa)
        print("layer",i,"mse",mm,"candidate_stats",st,"sanity",sa,flush=True)
    del grads; gc.collect()

    base=[0]*args.layers
    local=[min(range(len(FRACTIONS)),key=lambda j:mses[i][j]) for i in range(args.layers)]

    independent=[]
    for i in range(args.layers):
        vals=[]
        for j,frac in enumerate(FRACTIONS):
            trial=list(base); trial[i]=j
            apply(mods,sets,trial)
            v=metrics(model,cal,ref_cal,vocab)["kl_to_qwen"]; vals.append(v)
            print(f"independent layer={i} fraction={frac} cal_kl={v:.8g}",flush=True)
        independent.append(min(range(len(vals)),key=lambda j:vals[j]))

    grid=[]
    for choices in itertools.product(range(len(FRACTIONS)),repeat=args.layers):
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
    g=methods["fes_exact_global"]; ind=methods["independent_functional"]; loc=methods["local_weight_mse"]; b=methods["original_bonsai"]
    summary={
      "offset":args.offset,"layers":args.layers,"fractions":FRACTIONS,
      "same_q2_0_width":True,"same_group_scales":True,"extra_storage_bits":0,
      "global_differs_from_independent":global_choice!=independent,
      "heldout_interaction_gain":g["test"]["kl_to_qwen"]<ind["test"]["kl_to_qwen"],
      "global_vs_independent_test_kl_ratio":g["test"]["kl_to_qwen"]/ind["test"]["kl_to_qwen"],
      "global_vs_original_test_kl_ratio":g["test"]["kl_to_qwen"]/b["test"]["kl_to_qwen"],
      "global_vs_local_test_kl_ratio":g["test"]["kl_to_qwen"]/loc["test"]["kl_to_qwen"],
      "global_weight_mse_ratio_vs_local":g["weight_mse_to_qwen"]/loc["weight_mse_to_qwen"],
    }
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps({"summary":summary,"sanity":sanity,
      "candidate_stats":candidate_stats,"methods":methods,
      "top_grid":[{"cal_kl":v,"choices":list(c),"fractions":[FRACTIONS[j] for j in c]} for v,c in grid[:20]]},indent=2))
    print("SUMMARY",json.dumps(summary,sort_keys=True),flush=True)
    for k,v in methods.items(): print(k,json.dumps(v),flush=True)

if __name__=="__main__": main()
