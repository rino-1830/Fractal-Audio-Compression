import argparse
import importlib.util
import json
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
    ap.add_argument("--model", default="EleutherAI/pythia-14m")
    ap.add_argument("--out", default="results-coordinate")
    ap.add_argument("--passes", type=int, default=2)
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
    cal_w, test_w, source = base.load_windows(tok, 64, 4, 8)
    fp_cal, fp_test, _ = base.cache_fp(model, cal_w, test_w, 32)

    independent = [7, 11, 9, 6, 7, 7]
    start_metrics = base.evaluate(
        model, modules, originals, candidate_sets, independent,
        cal_w, test_w, fp_cal, fp_test
    )
    choices = list(independent)
    history = []

    for pass_idx in range(args.passes):
        changed = False
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
            if best_choice != choices[i]:
                changed = True
            choices[i] = best_choice
            history.append({
                "pass": pass_idx,
                "layer": i,
                "choice": best_choice,
                "cal_kl": best_kl,
            })
            print(
                f"pass={pass_idx} layer={i} choice={best_choice} cal_kl={best_kl:.8g}",
                flush=True,
            )
        if not changed:
            break

    base.restore(modules, originals)
    refined_metrics = base.evaluate(
        model, modules, originals, candidate_sets, choices,
        cal_w, test_w, fp_cal, fp_test
    )

    summary = {
        "model": args.model,
        "data_source": source,
        "start_choices": independent,
        "refined_choices": choices,
        "test_kl_ratio_refined_vs_independent": (
            refined_metrics["test_kl_to_fp"] / start_metrics["test_kl_to_fp"]
        ),
        "cal_kl_ratio_refined_vs_independent": (
            refined_metrics["cal_kl_to_fp"] / start_metrics["cal_kl_to_fp"]
        ),
        "weight_mse_ratio_refined_vs_independent": (
            refined_metrics["weight_mse"] / start_metrics["weight_mse"]
        ),
        "heldout_interaction_gain": bool(
            refined_metrics["test_kl_to_fp"] < start_metrics["test_kl_to_fp"]
        ),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({
        "summary": summary,
        "independent": start_metrics,
        "global_coordinate": refined_metrics,
        "history": history,
    }, indent=2))

    lines = [
        "# Exact cross-layer coordinate refinement",
        "",
        f"Model: {args.model}",
        f"Independent choices: {independent}",
        f"Refined choices: {choices}",
        "",
        f"Independent test KL: {start_metrics['test_kl_to_fp']:.8g}",
        f"Refined test KL: {refined_metrics['test_kl_to_fp']:.8g}",
        f"Test KL ratio: {summary['test_kl_ratio_refined_vs_independent']:.6f}",
        f"Calibration KL ratio: {summary['cal_kl_ratio_refined_vs_independent']:.6f}",
        f"Weight-MSE ratio: {summary['weight_mse_ratio_refined_vs_independent']:.6f}",
        f"Held-out interaction gain: {summary['heldout_interaction_gain']}",
    ]
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
