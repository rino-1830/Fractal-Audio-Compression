import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def softmax(x):
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def kl_fp_to_q(y_fp, y_q):
    p = softmax(y_fp)
    q = softmax(y_q)
    eps = 1e-12
    return float(np.mean(np.sum(p * (np.log(p + eps) - np.log(q + eps)), axis=-1)))


def forward(x, weights, out_w, gain):
    h = x
    for w in weights:
        h = h + gain * np.tanh(h @ w)
    return h @ out_w


def ternary_candidates(w):
    mean_abs = float(np.mean(np.abs(w))) + 1e-12
    threshold_factors = (0.45, 0.60, 0.75, 0.90, 1.05)
    scale_multipliers = (0.92, 1.00, 1.08)
    candidates = []
    for tf in threshold_factors:
        threshold = tf * mean_abs
        symbols = np.sign(w) * (np.abs(w) >= threshold)
        denom = float(np.sum(symbols * symbols))
        base_scale = mean_abs if denom == 0 else float(np.sum(w * symbols) / denom)
        p0 = float(np.mean(symbols == 0))
        pp = float(np.mean(symbols > 0))
        pm = float(np.mean(symbols < 0))
        entropy = 0.0
        for p in (p0, pp, pm):
            if p > 0:
                entropy -= p * math.log2(p)
        for sm in scale_multipliers:
            q = (base_scale * sm) * symbols
            candidates.append({
                "q": q,
                "threshold_factor": tf,
                "scale_multiplier": sm,
                "weight_mse": float(np.mean((w - q) ** 2)),
                "entropy_bpw": entropy,
                "zero_fraction": p0,
            })
    return candidates


def replace_one(weights, layer, q):
    out = list(weights)
    out[layer] = q
    return out


def apply_choices(weights, candidate_sets, choices):
    return [candidate_sets[i][choices[i]]["q"] for i in range(len(weights))]


def summarize_choice(weights, candidate_sets, choices):
    total_sq = 0.0
    total_n = 0
    entropy_weighted = 0.0
    zeros = 0.0
    for i, cidx in enumerate(choices):
        c = candidate_sets[i][cidx]
        total_sq += c["weight_mse"] * weights[i].size
        total_n += weights[i].size
        entropy_weighted += c["entropy_bpw"] * weights[i].size
        zeros += c["zero_fraction"] * weights[i].size
    return {
        "weight_mse": total_sq / total_n,
        "entropy_bpw": entropy_weighted / total_n,
        "zero_fraction": zeros / total_n,
    }


def output_metrics(y_fp, y_q):
    return {
        "output_mse": float(np.mean((y_fp - y_q) ** 2)),
        "kl": kl_fp_to_q(y_fp, y_q),
        "top1_agreement": float(np.mean(np.argmax(y_fp, axis=-1) == np.argmax(y_q, axis=-1))),
    }


def run_seed(seed, cfg):
    rng = np.random.default_rng(seed)
    d = cfg["dim"]
    layers = cfg["layers"]
    out_dim = cfg["out_dim"]
    gain = cfg["gain"]

    weights = [rng.normal(0, 1 / math.sqrt(d), size=(d, d)) for _ in range(layers)]
    out_w = rng.normal(0, 1 / math.sqrt(d), size=(d, out_dim))
    x_cal = rng.normal(size=(cfg["cal_samples"], d))
    x_test = rng.normal(size=(cfg["test_samples"], d))
    y_cal_fp = forward(x_cal, weights, out_w, gain)
    y_test_fp = forward(x_test, weights, out_w, gain)

    candidate_sets = [ternary_candidates(w) for w in weights]
    n_candidates = len(candidate_sets[0])

    local_choices = [int(np.argmin([c["weight_mse"] for c in cs])) for cs in candidate_sets]

    deltas = []
    individual_functional_scores = []
    for i, cs in enumerate(candidate_sets):
        layer_deltas = []
        layer_scores = []
        for c in cs:
            y = forward(x_cal, replace_one(weights, i, c["q"]), out_w, gain)
            delta = y - y_cal_fp
            layer_deltas.append(delta)
            layer_scores.append(float(np.mean(delta ** 2)))
        deltas.append(layer_deltas)
        individual_functional_scores.append(layer_scores)

    independent_choices = [int(np.argmin(scores)) for scores in individual_functional_scores]

    flat_dim = y_cal_fp.size
    sketch_dim = min(cfg["sketch_dim"], flat_dim)
    proj = rng.normal(0.0, 1.0 / math.sqrt(sketch_dim), size=(flat_dim, sketch_dim))
    sketches = []
    for layer_deltas in deltas:
        sketches.append([delta.reshape(-1) @ proj for delta in layer_deltas])

    beam = [(0.0, np.zeros(sketch_dim, dtype=np.float64), tuple())]
    for i in range(layers):
        expanded = []
        for _, accum, choices in beam:
            for cidx in range(n_candidates):
                new_accum = accum + sketches[i][cidx]
                score = float(np.dot(new_accum, new_accum))
                expanded.append((score, new_accum, choices + (cidx,)))
        expanded.sort(key=lambda t: t[0])
        beam = expanded[: cfg["beam_width"]]

    rerank = []
    for _, _, choices in beam[: cfg["rerank_top"]]:
        y = forward(x_cal, apply_choices(weights, candidate_sets, choices), out_w, gain)
        rerank.append((float(np.mean((y - y_cal_fp) ** 2)), tuple(choices)))
    rerank.sort(key=lambda t: t[0])
    fes_choices = list(rerank[0][1])

    for _ in range(cfg["coordinate_passes"]):
        changed = False
        for i in range(layers):
            best_choice = fes_choices[i]
            best_score = None
            for cidx in range(n_candidates):
                trial = list(fes_choices)
                trial[i] = cidx
                y = forward(x_cal, apply_choices(weights, candidate_sets, trial), out_w, gain)
                score = float(np.mean((y - y_cal_fp) ** 2))
                if best_score is None or score < best_score:
                    best_score = score
                    best_choice = cidx
            if best_choice != fes_choices[i]:
                fes_choices[i] = best_choice
                changed = True
        if not changed:
            break

    methods = {
        "local_weight_mse": local_choices,
        "independent_functional": independent_choices,
        "fes": fes_choices,
    }

    result = {"seed": seed, "methods": {}}
    for name, choices in methods.items():
        qweights = apply_choices(weights, candidate_sets, choices)
        y_cal = forward(x_cal, qweights, out_w, gain)
        y_test = forward(x_test, qweights, out_w, gain)
        entry = summarize_choice(weights, candidate_sets, choices)
        entry.update({f"cal_{k}": v for k, v in output_metrics(y_cal_fp, y_cal).items()})
        entry.update({f"test_{k}": v for k, v in output_metrics(y_test_fp, y_test).items()})

        selected_deltas = [deltas[i][choices[i]] for i in range(layers)]
        sum_individual_energy = float(sum(np.sum(d ** 2) for d in selected_deltas))
        summed_delta = np.sum(np.stack(selected_deltas, axis=0), axis=0)
        summed_energy = float(np.sum(summed_delta ** 2)) + 1e-30
        entry["linearized_cancellation_ratio"] = sum_individual_energy / summed_energy
        entry["choices"] = [int(x) for x in choices]
        result["methods"][name] = entry

    local = result["methods"]["local_weight_mse"]
    fes = result["methods"]["fes"]
    result["joint_hypothesis_success"] = bool(
        fes["test_kl"] < local["test_kl"] and fes["weight_mse"] > local["weight_mse"] * (1 + 1e-12)
    )
    result["fes_test_kl_ratio_vs_local"] = fes["test_kl"] / (local["test_kl"] + 1e-30)
    result["fes_weight_mse_ratio_vs_local"] = fes["weight_mse"] / (local["weight_mse"] + 1e-30)
    return result


def aggregate(results):
    method_names = list(results[0]["methods"].keys())
    agg = {}
    numeric_keys = [
        "weight_mse", "entropy_bpw", "zero_fraction", "cal_output_mse", "cal_kl",
        "cal_top1_agreement", "test_output_mse", "test_kl", "test_top1_agreement",
        "linearized_cancellation_ratio",
    ]
    for method in method_names:
        agg[method] = {}
        for key in numeric_keys:
            vals = np.array([r["methods"][method][key] for r in results], dtype=float)
            agg[method][key] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "std": float(np.std(vals)),
            }

    local_kl = np.array([r["methods"]["local_weight_mse"]["test_kl"] for r in results])
    fes_kl = np.array([r["methods"]["fes"]["test_kl"] for r in results])
    local_mse = np.array([r["methods"]["local_weight_mse"]["weight_mse"] for r in results])
    fes_mse = np.array([r["methods"]["fes"]["weight_mse"] for r in results])
    independent_kl = np.array([r["methods"]["independent_functional"]["test_kl"] for r in results])

    summary = {
        "seeds": len(results),
        "fes_test_kl_win_rate_vs_local": float(np.mean(fes_kl < local_kl)),
        "fes_test_kl_win_rate_vs_independent": float(np.mean(fes_kl < independent_kl)),
        "joint_hypothesis_success_rate": float(np.mean([r["joint_hypothesis_success"] for r in results])),
        "median_fes_test_kl_ratio_vs_local": float(np.median(fes_kl / (local_kl + 1e-30))),
        "mean_fes_test_kl_ratio_vs_local": float(np.mean(fes_kl / (local_kl + 1e-30))),
        "median_fes_weight_mse_ratio_vs_local": float(np.median(fes_mse / (local_mse + 1e-30))),
    }
    return agg, summary


def write_markdown(path, cfg, agg, summary):
    def m(method, key):
        return agg[method][key]["mean"]

    lines = [
        "# Functional Error Shaping proof-of-concept",
        "",
        "This experiment tests whether globally shaping/cancelling compression error in final-output space can preserve a model function better than minimizing each layer's local quantization error.",
        "",
        "## Configuration",
        "",
        f"- Seeds: {summary['seeds']}",
        f"- Residual layers: {cfg['layers']}, width: {cfg['dim']}, output width: {cfg['out_dim']}",
        "- Ternary candidates per layer: 15",
        f"- Calibration samples: {cfg['cal_samples']}; held-out test samples: {cfg['test_samples']}",
        f"- Output-error sketch dimension: {cfg['sketch_dim']}; beam width: {cfg['beam_width']}",
        "",
        "## Aggregate results",
        "",
        "| method | test KL ↓ | test output MSE ↓ | top-1 agreement ↑ | weight MSE ↓ | entropy bits/weight ↓ | cancellation ratio ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("local_weight_mse", "independent_functional", "fes"):
        lines.append(
            f"| {method} | {m(method,'test_kl'):.6g} | {m(method,'test_output_mse'):.6g} | "
            f"{m(method,'test_top1_agreement'):.4f} | {m(method,'weight_mse'):.6g} | "
            f"{m(method,'entropy_bpw'):.4f} | {m(method,'linearized_cancellation_ratio'):.3f} |"
        )
    lines += [
        "",
        "## Primary hypothesis checks",
        "",
        f"- FES beats local-MSE selection on held-out test KL in **{summary['fes_test_kl_win_rate_vs_local']*100:.1f}%** of seeds.",
        f"- FES beats independently output-aware per-layer selection in **{summary['fes_test_kl_win_rate_vs_independent']*100:.1f}%** of seeds.",
        f"- Joint condition (worse local weight MSE but better held-out KL) holds in **{summary['joint_hypothesis_success_rate']*100:.1f}%** of seeds.",
        f"- Median FES/local test-KL ratio: **{summary['median_fes_test_kl_ratio_vs_local']:.3f}** (<1 favors FES).",
        f"- Median FES/local weight-MSE ratio: **{summary['median_fes_weight_mse_ratio_vs_local']:.3f}** (>1 means FES deliberately accepts more local error).",
        "",
        "## Interpretation boundary",
        "",
        "This is a synthetic function-preservation test, not evidence that a language model will retain benchmark quality. A positive result only validates the narrower mechanism: cross-layer compression errors can be selected so their final-output effects cancel, even when local reconstruction error increases.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=24)
    args = ap.parse_args()

    cfg = {
        "layers": 8,
        "dim": 32,
        "out_dim": 16,
        "gain": 0.25,
        "cal_samples": 256,
        "test_samples": 2048,
        "sketch_dim": 128,
        "beam_width": 64,
        "rerank_top": 8,
        "coordinate_passes": 2,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = [run_seed(seed, cfg) for seed in range(args.seeds)]
    agg, summary = aggregate(results)

    payload = {"config": cfg, "summary": summary, "aggregate": agg, "per_seed": results}
    (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(out / "summary.md", cfg, agg, summary)

    with (out / "per_seed.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "local_test_kl", "independent_test_kl", "fes_test_kl", "fes/local_kl", "fes/local_weight_mse", "joint_success"])
        for r in results:
            writer.writerow([
                r["seed"],
                r["methods"]["local_weight_mse"]["test_kl"],
                r["methods"]["independent_functional"]["test_kl"],
                r["methods"]["fes"]["test_kl"],
                r["fes_test_kl_ratio_vs_local"],
                r["fes_weight_mse_ratio_vs_local"],
                int(r["joint_hypothesis_success"]),
            ])

    print((out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
