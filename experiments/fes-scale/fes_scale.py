import argparse
import importlib.util
import json
import math
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_base():
    path = Path("experiments/fes-transformer/fes_transformer.py")
    spec = importlib.util.spec_from_file_location("fes_base", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cal-windows", type=int, default=3)
    ap.add_argument("--test-windows", type=int, default=6)
    ap.add_argument("--passes", type=int, default=1)
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
    cal_w, test_w, source = base.load_windows(tok, 64, args.cal_windows, args.test_windows)
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
            print(
                f"model={args.model} layer={i} candidate={j} topk_kl={score:.8g}",
                flush=True,
            )
        independent.append(min(range(len(cs)), key=lambda j: scores[j]))
    base.restore(modules, originals)

    choices = list(independent)
    history = []
    for pass_idx in range(args.passes):
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
            history.append({
                "pass": pass_idx,
                "layer": i,
                "choice": best_choice,
                "cal_kl": best_kl,
            })
            print(
                f"model={args.model} pass={pass_idx} layer={i} choice={best_choice} cal_kl={best_kl:.8g}",
                flush=True,
            )
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
        "data_source": source,
        "layers": len(modules),
        "independent_choices": independent,
        "global_choices": choices,
        "global_test_kl_ratio_vs_independent": (
            g["test_kl_to_fp"] / max(b["test_kl_to_fp"], 1e-30)
        ),
        "global_test_kl_ratio_vs_local": (
            g["test_kl_to_fp"] / max(a["test_kl_to_fp"], 1e-30)
        ),
        "global_weight_mse_ratio_vs_independent": (
            g["weight_mse"] / max(b["weight_mse"], 1e-30)
        ),
        "global_weight_mse_ratio_vs_local": (
            g["weight_mse"] / max(a["weight_mse"], 1e-30)
        ),
        "heldout_interaction_gain": bool(
            g["test_kl_to_fp"] < b["test_kl_to_fp"]
        ),
        "higher_error_better_function": bool(
            g["test_kl_to_fp"] < b["test_kl_to_fp"]
            and g["weight_mse"] > b["weight_mse"]
        ),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({
        "summary": summary,
        "methods": methods,
        "history": history,
    }, indent=2))

    lines = [
        "# FES scaling experiment",
        "",
        f"Model: {args.model}",
        f"Layers: {len(modules)}",
        f"Independent choices: {independent}",
        f"Global choices: {choices}",
        "",
        f"Local test KL: {a['test_kl_to_fp']:.8g}",
        f"Independent test KL: {b['test_kl_to_fp']:.8g}",
        f"Global test KL: {g['test_kl_to_fp']:.8g}",
        f"Global/independent KL ratio: {summary['global_test_kl_ratio_vs_independent']:.6f}",
        f"Global/local KL ratio: {summary['global_test_kl_ratio_vs_local']:.6f}",
        f"Global/independent weight-MSE ratio: {summary['global_weight_mse_ratio_vs_independent']:.6f}",
        f"Global/local weight-MSE ratio: {summary['global_weight_mse_ratio_vs_local']:.6f}",
        f"Held-out interaction gain: {summary['heldout_interaction_gain']}",
        f"Higher-error-better-function: {summary['higher_error_better_function']}",
    ]
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
