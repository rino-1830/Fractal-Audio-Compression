import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from holdout import MODEL_ID, run_one

SEEDS = [3, 7, 11]
LAYERS = [0, 2, 4]
POOL_N = 120
CALIB_N = 28
HOLDOUT_N = 28

def load_pool():
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    texts = []
    for row in ds:
        t = " ".join(row["text"].split())
        if len(t) >= 120 and not t.startswith("="):
            texts.append(t[:700])
        if len(texts) >= POOL_N:
            break
    if len(texts) < POOL_N:
        raise RuntimeError("not enough text")
    return texts

def main():
    torch.set_num_threads(2)
    pool = load_pool()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()

    runs = []
    split_meta = []
    for seed in SEEDS:
        rng = random.Random(seed)
        idx = list(range(len(pool)))
        rng.shuffle(idx)
        calib_idx = idx[:CALIB_N]
        hold_idx = idx[CALIB_N:CALIB_N + HOLDOUT_N]
        calib = [pool[i] for i in calib_idx]
        hold = [pool[i] for i in hold_idx]
        split_meta.append({"seed": seed, "calib_idx": calib_idx, "holdout_idx": hold_idx})
        for layer in LAYERS:
            runs.append(run_one(model, tok, calib, hold, layer, seed))

    rows = []
    for method in ["failure_conditioned", "fisher_all", "random"]:
        for rank in [1, 2, 4]:
            vals = [r["methods"][method]["low_rank"][str(rank)]["mean_delta_nll"] for r in runs]
            rows.append({
                "method": method,
                "rank": rank,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            })

    base = np.array([r["baseline_holdout"]["mean_delta_nll"] for r in runs])
    out = {
        "model": MODEL_ID,
        "seeds": SEEDS,
        "layers": LAYERS,
        "pool_n": POOL_N,
        "calib_n": CALIB_N,
        "holdout_n": HOLDOUT_N,
        "baseline_mean": float(base.mean()),
        "baseline_std": float(base.std()),
        "aggregate": rows,
        "splits": split_meta,
        "runs": runs,
    }

    outdir = Path("results-crossval")
    outdir.mkdir(exist_ok=True)
    (outdir / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "# Cross-split functional compression validation",
        "",
        "Each seed uses a different shuffled calibration/holdout split.",
        "",
        "| method | rank | mean holdout delta NLL | std | recovery vs baseline |",
        "|---|---:|---:|---:|---:|",
        f"| ternary baseline | 0 | {base.mean():.6f} | {base.std():.6f} | 0.00% |",
    ]
    for row in rows:
        recovery = 100.0 * (base.mean() - row["mean"]) / base.mean()
        lines.append(
            f"| {row['method']} | {row['rank']} | {row['mean']:.6f} | {row['std']:.6f} | {recovery:.2f}% |"
        )
    lines += [
        "",
        "Primary criterion: failure-conditioned rank-2/4 should beat Fisher and random across genuinely different data splits, not only repeated factorization noise.",
    ]
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((outdir / "summary.md").read_text())

if __name__ == "__main__":
    main()
