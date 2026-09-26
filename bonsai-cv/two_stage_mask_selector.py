import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "prism-ml/Bonsai-1.7B-unpacked"
LAYER = int(os.environ["CASE_LAYER"])
FOLD = int(os.environ["CASE_FOLD"])
FRAC = 0.50
NUM_MASKS = 16
SHORTLIST_K = 3
CAL_N = 8
GATE_N = 4
HOLD_N = 16
MAX_LENGTH = 20
BASE_SKIP = 500
FOLD_STRIDE = 40
GROUP_SIZE = 128
OUT = Path("results-two-stage-mask")


def load_texts():
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    xs = []
    eligible = 0
    start = BASE_SKIP + FOLD * FOLD_STRIDE
    need = CAL_N + GATE_N + HOLD_N
    for row in ds:
        t = " ".join(row["text"].split())
        if len(t) < 100 or t.startswith("="):
            continue
        if eligible < start:
            eligible += 1
            continue
        xs.append(t[:500])
        if len(xs) >= need:
            break
    if len(xs) < need:
        raise RuntimeError("not enough text")
    cal = xs[:CAL_N]
    gate = xs[CAL_N:CAL_N + GATE_N]
    hold = xs[CAL_N + GATE_N:]
    return cal, gate, hold


def encode(tok, text):
    return tok(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)


@torch.no_grad()
def losses(model, tok, texts):
    vals = []
    for text in texts:
        x = encode(tok, text)
        vals.append(float(model(**x, labels=x["input_ids"]).loss))
    return np.asarray(vals, dtype=np.float64)


def masked_weight(w, code):
    x = w.detach().clone()
    g = torch.Generator(device="cpu")
    # Same stable code family across layers/folds. Fold changes only data.
    g.manual_seed(246813579 + code * 1000003 + LAYER * 1009)
    mask = torch.rand(x.shape, generator=g) < FRAC
    x[mask] = 0
    return x


def metrics(fp, vals):
    d = vals - fp
    pos = np.maximum(d, 0.0)
    return {
        "mean_delta_nll": float(d.mean()),
        "mean_positive_delta_nll": float(pos.mean()),
        "max_delta_nll": float(d.max()),
        "fraction_worse": float(np.mean(d > 0)),
        "per_sample_delta": d.tolist(),
    }


def score(m):
    return (m["mean_positive_delta_nll"], m["mean_delta_nll"], m["max_delta_nll"])


def eval_code(target, original, code, model, tok, fp, texts):
    mw = masked_weight(original, code)
    with torch.no_grad():
        target.copy_(mw)
    vals = losses(model, tok, texts)
    return metrics(fp, vals)


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    cal, gate, hold = load_texts()

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
    gate_fp = losses(model, tok, gate)
    hold_fp = losses(model, tok, hold)

    candidates = []
    for code in range(NUM_MASKS):
        m = eval_code(target, original, code, model, tok, cal_fp, cal)
        candidates.append({"code": code, "calibration": m})
        print("CAL", code, score(m), flush=True)

    ranked = sorted(candidates, key=lambda x: score(x["calibration"]))
    shortlist = [x["code"] for x in ranked[:SHORTLIST_K]]
    calibration_best = shortlist[0]

    gate_rows = []
    for code in shortlist:
        m = eval_code(target, original, code, model, tok, gate_fp, gate)
        gate_rows.append({"code": code, "gate": m})
        print("GATE", code, score(m), flush=True)

    gate_ranked = sorted(gate_rows, key=lambda x: score(x["gate"]))
    selected_code = gate_ranked[0]["code"]

    compare_codes = sorted(set([0, calibration_best, selected_code] + shortlist))
    holdout = {}
    for code in compare_codes:
        holdout[str(code)] = eval_code(target, original, code, model, tok, hold_fp, hold)

    with torch.no_grad():
        target.copy_(original)

    code0 = holdout["0"]
    calbest = holdout[str(calibration_best)]
    selected = holdout[str(selected_code)]
    shortlist_oracle_code = min(shortlist, key=lambda c: score(holdout[str(c)]))
    shortlist_oracle = holdout[str(shortlist_oracle_code)]

    code_bits = int(np.ceil(np.log2(NUM_MASKS)))
    storage_bpw = (1.0 - FRAC) + 16.0 / GROUP_SIZE + code_bits / original.numel()

    payload = {
        "model": MODEL_ID,
        "layer": LAYER,
        "fold": FOLD,
        "prune_fraction": FRAC,
        "num_masks": NUM_MASKS,
        "shortlist_k": SHORTLIST_K,
        "calibration_n": CAL_N,
        "gate_n": GATE_N,
        "holdout_n": HOLD_N,
        "storage_bpw": float(storage_bpw),
        "calibration_ranked_codes": [x["code"] for x in ranked],
        "shortlist": shortlist,
        "gate_rows": gate_rows,
        "calibration_best_code": calibration_best,
        "selected_code": selected_code,
        "shortlist_oracle_code": shortlist_oracle_code,
        "holdout": holdout,
        "summary": {
            "code0_positive_harm": code0["mean_positive_delta_nll"],
            "calibration_best_positive_harm": calbest["mean_positive_delta_nll"],
            "two_stage_positive_harm": selected["mean_positive_delta_nll"],
            "shortlist_oracle_positive_harm": shortlist_oracle["mean_positive_delta_nll"],
            "code0_mean_delta": code0["mean_delta_nll"],
            "calibration_best_mean_delta": calbest["mean_delta_nll"],
            "two_stage_mean_delta": selected["mean_delta_nll"],
            "shortlist_oracle_mean_delta": shortlist_oracle["mean_delta_nll"],
        },
    }
    stem = f"l{LAYER}_f{FOLD}"
    (OUT / f"{stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"] | {
        "layer": LAYER,
        "fold": FOLD,
        "shortlist": shortlist,
        "selected_code": selected_code,
        "shortlist_oracle_code": shortlist_oracle_code,
        "storage_bpw": storage_bpw,
    }, indent=2))


if __name__ == "__main__":
    main()
