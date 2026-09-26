import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "prism-ml/Bonsai-1.7B-unpacked"
LAYER = int(os.environ["CASE_LAYER"])
FOLD = int(os.environ["CASE_FOLD"])
GROUP = 128
KEEP = 64
CODEBOOK_SIZES = [4, 8]
CAL_N = 8
HOLD_N = 20
MAX_LENGTH = 24
OUT = Path("results-activation-codebook")


def collect(split, skip, n):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    xs = []
    eligible = 0
    for row in ds:
        t = " ".join(row["text"].split())
        if len(t) < 100 or t.startswith("="):
            continue
        if eligible < skip:
            eligible += 1
            continue
        xs.append(t[:500])
        if len(xs) >= n:
            break
    if len(xs) < n:
        raise RuntimeError("not enough text")
    return xs


def enc(tok, t):
    return tok(t, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)


@torch.no_grad()
def losses(model, tok, texts):
    vals = []
    for t in texts:
        x = enc(tok, t)
        vals.append(float(model(**x, labels=x["input_ids"]).loss))
    return np.asarray(vals, dtype=np.float64)


def metric(fp, vals):
    d = vals - fp
    return {
        "mean_delta_nll": float(d.mean()),
        "mean_positive_delta_nll": float(np.maximum(d, 0).mean()),
        "median_delta_nll": float(np.median(d)),
        "max_delta_nll": float(d.max()),
        "fraction_worse": float(np.mean(d > 0)),
        "per_sample_delta": d.tolist(),
    }


def make_codebook(m):
    # Nested deterministic codebook: the first 4 masks of m=8 are identical to m=4.
    rows = []
    for code in range(m):
        g = torch.Generator(device="cpu")
        g.manual_seed(80808080 + code * 1000003)
        perm = torch.randperm(GROUP, generator=g)
        keep = torch.zeros(GROUP, dtype=torch.bool)
        keep[perm[:KEEP]] = True
        rows.append(keep)
    return torch.stack(rows, dim=0)


def collect_activation_energy(model, tok, texts, target):
    acc = None
    count = 0

    def hook(module, args):
        nonlocal acc, count
        x = args[0].detach().float().reshape(-1, args[0].shape[-1])
        s = x.square().sum(dim=0).cpu()
        acc = s if acc is None else acc + s
        count += x.shape[0]

    h = target.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            for t in texts:
                x = enc(tok, t)
                model(**x)
    finally:
        h.remove()
    return acc / max(count, 1)


def fixed_balanced(orig, alpha):
    rows, cols = orig.shape
    if cols % GROUP:
        raise RuntimeError(f"columns {cols} not divisible by group {GROUP}")
    codebook = make_codebook(1)
    keep = codebook[0].view(1, 1, GROUP).expand(rows, cols // GROUP, GROUP)
    x = orig.detach().float().reshape(rows, cols // GROUP, GROUP).clone()
    x[~keep] = 0
    x *= alpha
    return x.reshape_as(orig).to(orig.dtype)


def activation_codebook(orig, act2, m, adaptive_scale):
    rows, cols = orig.shape
    if cols % GROUP:
        raise RuntimeError(f"columns {cols} not divisible by group {GROUP}")
    blocks = cols // GROUP
    W = orig.detach().float().reshape(rows, blocks, GROUP)
    A = act2.float().reshape(blocks, GROUP)
    cb = make_codebook(m)

    # Local diagonal output-error proxy: w^2 * E[x^2].
    energy = W.square() * A.unsqueeze(0)
    prune = (~cb).float()
    cost = torch.einsum("rbg,mg->rbm", energy, prune)
    best = cost.argmin(dim=-1)  # rows x blocks
    keep = cb[best]             # rows x blocks x group

    out = W.clone()
    out[~keep] = 0

    if adaptive_scale:
        total = energy.sum(dim=-1)
        kept = (energy * keep.float()).sum(dim=-1).clamp_min(1e-12)
        alpha = torch.sqrt(total / kept).clamp(0.75, 2.5)
        out *= alpha.unsqueeze(-1)
        alpha_stats = {
            "mean": float(alpha.mean()),
            "std": float(alpha.std()),
            "min": float(alpha.min()),
            "max": float(alpha.max()),
        }
    else:
        out *= math.sqrt(2.0)
        alpha_stats = {"fixed": math.sqrt(2.0)}

    counts = torch.bincount(best.reshape(-1), minlength=m).tolist()
    return out.reshape_as(orig).to(orig.dtype), {
        "code_counts": counts,
        "alpha": alpha_stats,
    }


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)

    cal = collect("validation", 1200 + FOLD * 100, CAL_N)
    hold = collect("test", 2000 + FOLD * 100, HOLD_N)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()

    target = model.model.layers[LAYER].mlp.down_proj
    weight = target.weight
    orig = weight.detach().clone()

    with torch.no_grad():
        weight.copy_(orig)
    hold_fp = losses(model, tok, hold)

    act2 = collect_activation_energy(model, tok, cal, target)
    if act2.numel() != orig.shape[1]:
        raise RuntimeError(f"activation dim {act2.numel()} != weight cols {orig.shape[1]}")

    candidates = {}

    fixed = fixed_balanced(orig, math.sqrt(2.0))
    with torch.no_grad():
        weight.copy_(fixed)
    candidates["fixed_balanced_sqrt"] = {
        "metric": metric(hold_fp, losses(model, tok, hold)),
        "storage_bpw": 0.5 + 16.0 / GROUP,
    }

    for m in CODEBOOK_SIZES:
        bits = math.ceil(math.log2(m))
        storage = 0.5 + 16.0 / GROUP + bits / GROUP
        for adaptive in [False, True]:
            name = f"act_codebook_m{m}_" + ("adaptive_scale" if adaptive else "sqrt")
            sparse, meta = activation_codebook(orig, act2, m, adaptive)
            with torch.no_grad():
                weight.copy_(sparse)
            candidates[name] = {
                "metric": metric(hold_fp, losses(model, tok, hold)),
                "storage_bpw": float(storage),
                "meta": meta,
            }
            print(name, candidates[name]["metric"]["mean_positive_delta_nll"],
                  candidates[name]["metric"]["mean_delta_nll"], storage, flush=True)

    with torch.no_grad():
        weight.copy_(orig)

    payload = {
        "model": MODEL_ID,
        "layer": LAYER,
        "fold": FOLD,
        "group": GROUP,
        "keep": KEEP,
        "calibration_n": CAL_N,
        "holdout_n": HOLD_N,
        "activation_energy_stats": {
            "mean": float(act2.mean()),
            "std": float(act2.std()),
            "min": float(act2.min()),
            "max": float(act2.max()),
        },
        "candidates": candidates,
    }
    (OUT / f"l{LAYER}_f{FOLD}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
