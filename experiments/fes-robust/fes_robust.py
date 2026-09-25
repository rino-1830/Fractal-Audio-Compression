import argparse
import importlib.util
import json
import math
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_base():
    path = Path("experiments/fes-transformer/fes_transformer.py")
    spec = importlib.util.spec_from_file_location("fes_base", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def windows_from_split(tok, split, seq_len, n_windows, offset_tokens=0):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(x["text"] for x in ds if x["text"].strip())
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    stride = seq_len + 17
    pos = offset_tokens
    out = []
    for _ in range(n_windows):
        if pos + seq_len > ids.numel():
            pos = max(0, (pos % max(1, ids.numel() - seq_len)))
        out.append(ids[pos:pos + seq_len].clone())
        pos += stride
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cal-windows", type=int, default=4)
    ap.add_argument("--test-windows", type=int, default=32)
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(min(os.cpu_count() or 1, 16))
    base = load_base()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    modules = base.get_target_modules(model)
    originals = [m.weight.detach().float().cpu().clone() for _, m in modules]
    candidate_sets = [base.ternary_candidates(w) for w in originals]

    cal_w = windows_from_split(tok, "validation", 64, args.cal_windows, 211)
    test_w = windows_from_split(tok, "test", 64, args.test_windows, 503)
    fp_cal, fp_test, fp_top = base.cache_fp(model, cal_w, test_w, 32)

    local = [
        min(range(len(cs)), key=lambda j: cs[j]["weight_mse"])
        for cs in candidate_sets
    ]

    independent = []
    for i, cs in enumerate(candidate_sets):
        scores = []
        for j, c in enumerate(cs):
            base.restore(modules, originals)
            modules[i][1].weight.data.copy_(c["q"].to(modules[i][1].weight.dtype))
            _, score = base.functional_delta(model, cal_w, fp_top)
            scores.append(score)
        independent.append(min(range(len(cs)), key=lambda j: scores[j]))
        print(f"independent layer={i} choice={independent[-1]}", flush=True)
    base.restore(modules, originals)

    choices = list(independent)
    for i, cs in enumerate(candidate_sets):
        best_choice = choices[i]
        best_kl = None
        for j in range(len(cs)):
            trial = list(choices)
            trial[i] = j
            base.apply_choices(modules, candidate_sets, trial)
            kl = base.metrics(model, cal_w, fp_cal)["kl_to_fp"]
            if best_kl is None or kl < best_kl:
                best_kl = kl
                best_choice = j
        choices[i] = best_choice
        print(f"global layer={i} choice={best_choice} cal_kl={best_kl:.8g}", flush=True)
    base.restore(modules, originals)

    methods = {
        "local_weight_mse": base.evaluate(
            model, modules, originals, candidate_sets, local,
            cal_w, test_w, fp_cal, fp_test
        ),
        "independent_functional": base.evaluate(
            model, modules, originals, candidate_sets, independent,
            cal_w, test_w, fp_cal, fp_test
        ),
        "global_coordinate": base.evaluate(
            model, modules, originals, candidate_sets, choices,
            cal_w, test_w, fp_cal, fp_test
        ),
    }

    a = methods["local_weight_mse"]
    b = methods["independent_functional"]
    g = methods["global_coordinate"]
    summary = {
        "model": args.model,
        "calibration_split": "wikitext-2-raw-v1/validation",
        "test_split": "wikitext-2-raw-v1/test",
        "cal_tokens": args.cal_windows * 63,
        "test_tokens": args.test_windows * 63,
        "layers": len(modules),
        "independent_choices": independent,
        "global_choices": choices,
        "global_test_kl_ratio_vs_independent": g["test_kl_to_fp"] / max(b["test_kl_to_fp"], 1e-30),
        "global_test_kl_ratio_vs_local": g["test_kl_to_fp"] / max(a["test_kl_to_fp"], 1e-30),
        "global_weight_mse_ratio_vs_independent": g["weight_mse"] / max(b["weight_mse"], 1e-30),
        "global_weight_mse_ratio_vs_local": g["weight_mse"] / max(a["weight_mse"], 1e-30),
        "cross_split_interaction_gain": bool(g["test_kl_to_fp"] < b["test_kl_to_fp"]),
        "higher_error_better_function_vs_independent": bool(
            g["test_kl_to_fp"] < b["test_kl_to_fp"] and g["weight_mse"] > b["weight_mse"]
        ),
        "higher_error_better_function_vs_local": bool(
            g["test_kl_to_fp"] < a["test_kl_to_fp"] and g["weight_mse"] > a["weight_mse"]
        ),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({
        "summary": summary,
        "methods": methods,
    }, indent=2))

    lines = [
        "# FES cross-split robustness",
        "",
        f"Model: {args.model}",
        f"Calibration: validation ({summary['cal_tokens']} tokens)",
        f"Test: test split ({summary['test_tokens']} tokens)",
        f"Independent choices: {independent}",
        f"Global choices: {choices}",
        "",
        f"Local test KL: {a['test_kl_to_fp']:.8g}",
        f"Independent test KL: {b['test_kl_to_fp']:.8g}",
        f"Global test KL: {g['test_kl_to_fp']:.8g}",
        f"Global/independent KL ratio: {summary['global_test_kl_ratio_vs_independent']:.6f}",
        f"Global/local KL ratio: {summary['global_test_kl_ratio_vs_local']:.6f}",
        f"Global/independent weight-MSE ratio: {summary['global_weight_mse_ratio_vs_independent']:.6f}",
        f"Cross-split interaction gain: {summary['cross_split_interaction_gain']}",
        f"Higher-error-better-function vs independent: {summary['higher_error_better_function_vs_independent']}",
        f"Higher-error-better-function vs local: {summary['higher_error_better_function_vs_local']}",
    ]
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
