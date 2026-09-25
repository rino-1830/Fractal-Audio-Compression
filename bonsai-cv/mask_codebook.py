import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "prism-ml/Bonsai-1.7B-unpacked"
LAYER = int(os.environ.get("CASE_LAYER", "14"))
FRAC = float(os.environ.get("CASE_FRAC", "0.25"))
NUM_MASKS = int(os.environ.get("NUM_MASKS", "16"))
CAL_N = 4
HOLD_N = 12
MAX_LENGTH = 20
SKIP_ELIGIBLE = 220
GROUP_SIZE = 128
OUT = Path("results-mask-codebook")


def load_texts():
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    xs = []
    skipped = 0
    for row in ds:
        t = " ".join(row["text"].split())
        if len(t) < 100 or t.startswith("="):
            continue
        if skipped < SKIP_ELIGIBLE:
            skipped += 1
            continue
        xs.append(t[:500])
        if len(xs) >= CAL_N + HOLD_N:
            break
    return xs[:CAL_N], xs[CAL_N:CAL_N + HOLD_N]


def encode(tok, text):
    return tok(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)


@torch.no_grad()
def losses(model, tok, texts):
    vals = []
    for text in texts:
        x = encode(tok, text)
        vals.append(float(model(**x, labels=x["input_ids"]).loss))
    return np.array(vals, dtype=np.float64)


def make_masked(w, frac, code):
    x = w.detach().clone()
    gen = torch.Generator(device="cpu")
    # Global deterministic codebook. Only code index needs storage.
    gen.manual_seed(123456789 + code * 1000003 + LAYER * 1009 + int(FRAC * 1000))
    mask = torch.rand(x.shape, generator=gen) < frac
    x[mask] = 0
    return x, float(mask.float().mean())


def metrics(fp, vals):
    d = vals - fp
    return {
        "mean_delta_nll": float(d.mean()),
        "mean_positive_delta_nll": float(np.maximum(d, 0.0).mean()),
        "max_delta_nll": float(d.max()),
        "fraction_worse": float(np.mean(d > 0)),
        "per_sample_delta": d.tolist(),
    }


def score(m):
    # Harm first; signed mean breaks ties.
    return (m["mean_positive_delta_nll"], m["mean_delta_nll"], m["max_delta_nll"])


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    cal, hold = load_texts()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    target = model.model.layers[LAYER].mlp.down_proj.weight
    original = target.detach().clone()

    with torch.no_grad():
        target.copy_(original)
    cal_fp = losses(model, tok, cal)
    hold_fp = losses(model, tok, hold)

    candidates = []
    best_code = None
    best_score = None
    best_weight = None

    for code in range(NUM_MASKS):
        masked, actual = make_masked(original, FRAC, code)
        with torch.no_grad():
            target.copy_(masked)
        cal_vals = losses(model, tok, cal)
        m = metrics(cal_fp, cal_vals)
        s = score(m)
        candidates.append({
            "code": code,
            "actual_prune_fraction": actual,
            "calibration": m,
        })
        print("MASK", code, s, flush=True)
        if best_score is None or s < best_score:
            best_score = s
            best_code = code
            best_weight = masked.clone()

    # Compare selected code against code 0 and median calibration code on untouched holdout.
    by_score = sorted(candidates, key=lambda x: score(x["calibration"]))
    median_code = by_score[len(by_score)//2]["code"]
    compare_codes = sorted(set([0, best_code, median_code]))

    holdout = {}
    for code in compare_codes:
        masked, actual = make_masked(original, FRAC, code)
        with torch.no_grad():
            target.copy_(masked)
        vals = losses(model, tok, hold)
        holdout[str(code)] = {
            "actual_prune_fraction": actual,
            **metrics(hold_fp, vals),
        }

    with torch.no_grad():
        target.copy_(original)

    index_bits_per_matrix = float(np.ceil(np.log2(NUM_MASKS)))
    sign_payload_bpw = 1.0 - FRAC
    scale_bpw = 16.0 / GROUP_SIZE
    # Amortized code index is effectively zero, but report it exactly for this matrix.
    index_bpw = index_bits_per_matrix / original.numel()
    total_bpw = sign_payload_bpw + scale_bpw + index_bpw

    payload = {
        "model": MODEL_ID,
        "layer": LAYER,
        "prune_fraction": FRAC,
        "num_masks": NUM_MASKS,
        "best_code": best_code,
        "median_code": median_code,
        "code_index_bits_per_matrix": index_bits_per_matrix,
        "storage_bpw": total_bpw,
        "candidates": candidates,
        "holdout": holdout,
        "selected_holdout": holdout[str(best_code)],
        "fixed_code0_holdout": holdout["0"],
    }
    stem = f"l{LAYER}_p{int(FRAC*100)}"
    (OUT / f"{stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    selected = payload["selected_holdout"]
    fixed = payload["fixed_code0_holdout"]
    print(json.dumps({
        "best_code": best_code,
        "storage_bpw": total_bpw,
        "fixed_code0_positive_harm": fixed["mean_positive_delta_nll"],
        "selected_positive_harm": selected["mean_positive_delta_nll"],
        "fixed_code0_mean_delta": fixed["mean_delta_nll"],
        "selected_mean_delta": selected["mean_delta_nll"],
    }, indent=2))


if __name__ == "__main__":
    main()
