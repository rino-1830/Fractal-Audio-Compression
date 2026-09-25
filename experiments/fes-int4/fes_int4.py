import argparse
import importlib.util
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_base():
    path = Path("experiments/fes-transformer/fes_transformer.py")
    spec = importlib.util.spec_from_file_location("fes_base", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_all_linear_modules(model):
    modules = []
    for i, layer in enumerate(model.gpt_neox.layers):
        modules += [
            (f"layer.{i}.attn.qkv", layer.attention.query_key_value),
            (f"layer.{i}.attn.out", layer.attention.dense),
            (f"layer.{i}.mlp.up", layer.mlp.dense_h_to_4h),
            (f"layer.{i}.mlp.down", layer.mlp.dense_4h_to_h),
        ]
    return modules


def int4_candidates(w):
    factors = (0.60, 0.75, 0.85, 0.95, 1.00)
    out = []
    row_max = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    for factor in factors:
        clip = row_max * factor
        scale = clip / 7.0
        qint = torch.round(torch.clamp(w, -clip, clip) / scale).clamp(-7, 7)
        q = qint * scale
        out.append({
            "q": q.float().contiguous(),
            "weight_mse": F.mse_loss(q, w).item(),
            "clip_factor": factor,
        })
    return out


def split_windows(tok, split, n, seq_len=64, offset=0):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(x["text"] for x in ds if x["text"].strip())
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    stride = seq_len + 23
    pos = offset
    out = []
    for _ in range(n):
        if pos + seq_len > ids.numel():
            pos = pos % max(1, ids.numel() - seq_len)
        out.append(ids[pos:pos+seq_len].clone())
        pos += stride
    return out


def weight_mse(candidate_sets, choices):
    return sum(candidate_sets[i][c]["weight_mse"] for i,c in enumerate(choices)) / len(choices)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cal-windows", type=int, default=2)
    ap.add_argument("--test-windows", type=int, default=16)
    args=ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(min(os.cpu_count() or 1, 16))
    base=load_base()

    tok=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    modules=get_all_linear_modules(model)
    originals=[m.weight.detach().float().cpu().clone() for _,m in modules]
    candidate_sets=[int4_candidates(w) for w in originals]

    cal_w=split_windows(tok,"validation",args.cal_windows,64,317)
    test_w=split_windows(tok,"test",args.test_windows,64,811)
    fp_cal,fp_test,fp_top=base.cache_fp(model,cal_w,test_w,16)

    local=[min(range(len(cs)), key=lambda j:cs[j]["weight_mse"]) for cs in candidate_sets]

    independent=[]
    for i,cs in enumerate(candidate_sets):
        scores=[]
        for j,c in enumerate(cs):
            base.restore(modules, originals)
            modules[i][1].weight.data.copy_(c["q"].to(modules[i][1].weight.dtype))
            _,score=base.functional_delta(model,cal_w,fp_top)
            scores.append(score)
        independent.append(min(range(len(cs)),key=lambda j:scores[j]))
        print(f"independent matrix={i} name={modules[i][0]} choice={independent[-1]}",flush=True)
    base.restore(modules,originals)

    global_choices=list(independent)
    history=[]
    for i,cs in enumerate(candidate_sets):
        best_j=global_choices[i]
        best_kl=None
        for j in range(len(cs)):
            trial=list(global_choices)
            trial[i]=j
            base.apply_choices(modules,candidate_sets,trial)
            kl=base.metrics(model,cal_w,fp_cal)["kl_to_fp"]
            if best_kl is None or kl<best_kl:
                best_kl=kl
                best_j=j
        global_choices[i]=best_j
        history.append({"matrix":i,"name":modules[i][0],"choice":best_j,"cal_kl":best_kl})
        print(f"global matrix={i} name={modules[i][0]} choice={best_j} cal_kl={best_kl:.8g}",flush=True)
    base.restore(modules,originals)

    def evaluate(choices):
        base.apply_choices(modules,candidate_sets,choices)
        cal=base.metrics(model,cal_w,fp_cal)
        test=base.metrics(model,test_w,fp_test)
        mse=weight_mse(candidate_sets,choices)
        base.restore(modules,originals)
        return {
            "weight_mse":mse,
            **{f"cal_{k}":v for k,v in cal.items()},
            **{f"test_{k}":v for k,v in test.items()},
            "choices":list(map(int,choices)),
        }

    methods={
        "local_weight_mse":evaluate(local),
        "independent_functional":evaluate(independent),
        "global_coordinate":evaluate(global_choices),
    }
    a=methods["local_weight_mse"]
    b=methods["independent_functional"]
    g=methods["global_coordinate"]

    summary={
        "model":args.model,
        "quantization":"per-output-channel symmetric int4, 5 clipping candidates",
        "matrices_quantized":len(modules),
        "matrix_names":[n for n,_ in modules],
        "cal_tokens":args.cal_windows*63,
        "test_tokens":args.test_windows*63,
        "global_test_kl_ratio_vs_independent":g["test_kl_to_fp"]/max(b["test_kl_to_fp"],1e-30),
        "global_test_kl_ratio_vs_local":g["test_kl_to_fp"]/max(a["test_kl_to_fp"],1e-30),
        "global_weight_mse_ratio_vs_independent":g["weight_mse"]/max(b["weight_mse"],1e-30),
        "global_weight_mse_ratio_vs_local":g["weight_mse"]/max(a["weight_mse"],1e-30),
        "interaction_gain":bool(g["test_kl_to_fp"]<b["test_kl_to_fp"]),
        "higher_error_better_function_vs_independent":bool(
            g["test_kl_to_fp"]<b["test_kl_to_fp"] and g["weight_mse"]>b["weight_mse"]
        ),
    }

    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps({"summary":summary,"methods":methods,"history":history},indent=2))
    lines=[
        "# FES with full-block INT4 PTQ",
        "",
        f"Model: {args.model}",
        f"Quantized matrices: {len(modules)} (attention QKV/out + MLP up/down in every block)",
        f"Calibration tokens: {summary['cal_tokens']}",
        f"Test tokens: {summary['test_tokens']} on WikiText test split",
        "",
        f"Local test KL: {a['test_kl_to_fp']:.8g}",
        f"Independent test KL: {b['test_kl_to_fp']:.8g}",
        f"Global test KL: {g['test_kl_to_fp']:.8g}",
        f"Global/independent KL ratio: {summary['global_test_kl_ratio_vs_independent']:.6f}",
        f"Global/local KL ratio: {summary['global_test_kl_ratio_vs_local']:.6f}",
        f"Global/independent weight-MSE ratio: {summary['global_weight_mse_ratio_vs_independent']:.6f}",
        f"Interaction gain: {summary['interaction_gain']}",
        f"Higher-error-better-function vs independent: {summary['higher_error_better_function_vs_independent']}",
    ]
    (out/"summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__=="__main__":
    main()
