import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy.linalg import eigh
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "EleutherAI/pythia-70m-deduped")
SEEDS = [3, 7, 11]
LAYERS = [0, 2, 4]
K = 4
CALIBRATION_N = 28
HOLDOUT_N = 28
FAIL_N = 6
STABLE_N = 6
LOW_RANKS = (1, 2, 4)
MAX_LENGTH = 64

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_texts():
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    texts = []
    for row in ds:
        t = " ".join(row["text"].split())
        if len(t) >= 120 and not t.startswith("="):
            texts.append(t[:700])
        if len(texts) >= CALIBRATION_N + HOLDOUT_N:
            break
    if len(texts) < CALIBRATION_N + HOLDOUT_N:
        raise RuntimeError("not enough WikiText samples")
    return texts[:CALIBRATION_N], texts[CALIBRATION_N:CALIBRATION_N+HOLDOUT_N]

def encode(tok, text):
    return tok(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)

@torch.no_grad()
def nll(model, tok, text):
    x = encode(tok, text)
    return float(model(**x, labels=x["input_ids"]).loss)

def eval_texts(model, tok, texts):
    return np.array([nll(model, tok, t) for t in texts], dtype=np.float64)

def rowwise_ternary(w):
    x = w.detach().float()
    mean_abs = x.abs().mean(dim=1, keepdim=True)
    threshold = 0.7 * mean_abs
    mask = x.abs() >= threshold
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
    scale = (x.abs() * mask).sum(dim=1, keepdim=True) / denom
    return (scale * x.sign() * mask).to(dtype=w.dtype)

def sample_grad(model, tok, target, text):
    model.zero_grad(set_to_none=True)
    x = encode(tok, text)
    loss = model(**x, labels=x["input_ids"]).loss
    g = torch.autograd.grad(loss, target, retain_graph=False, create_graph=False)[0]
    g = g.detach().float().reshape(-1)
    return g / g.norm().clamp_min(1e-12)

def orthonormalize(cols):
    q, _ = torch.linalg.qr(cols, mode="reduced")
    return q

def fisher_basis(G, k):
    Kmat = G @ G.T
    vals, vecs = torch.linalg.eigh(Kmat)
    order = torch.argsort(vals, descending=True)
    vals = vals[order][:k].clamp_min(1e-12)
    vecs = vecs[:, order][:, :k]
    return orthonormalize(G.T @ (vecs / torch.sqrt(vals).unsqueeze(0)))

def failure_basis(G, nf, k):
    n = G.shape[0]
    Kg = (G @ G.T).double().cpu().numpy()
    fi = np.arange(nf)
    si = np.arange(nf, n)
    A = Kg[:, fi] @ Kg[fi, :]
    B = Kg[:, si] @ Kg[si, :]
    reg = max(float(np.trace(Kg)) / max(n, 1), 1e-8) * 1e-2
    B = B + reg * np.eye(n)
    vals, vecs = eigh(A, B, check_finite=False)
    order = np.argsort(vals)[::-1][:k]
    coeff = torch.from_numpy(vecs[:, order]).to(dtype=G.dtype)
    return orthonormalize(G.T @ coeff)

def random_basis(p, k, seed):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 1000)
    return orthonormalize(torch.randn(p, k, generator=gen))

def low_rank_approx(mat, rank):
    q = min(rank + 2, min(mat.shape))
    U, S, V = torch.svd_lowrank(mat, q=q, niter=2)
    return (U[:, :rank] * S[:rank]) @ V[:, :rank].T

def summarize(fp, vals):
    d = vals - fp
    return {
        "mean_delta_nll": float(d.mean()),
        "median_delta_nll": float(np.median(d)),
        "p90_delta_nll": float(np.quantile(d, 0.9)),
        "max_delta_nll": float(d.max()),
    }

def run_one(model, tok, calib, holdout, layer_idx, seed):
    set_seed(seed)
    target = model.gpt_neox.layers[layer_idx].mlp.dense_h_to_4h.weight
    original = target.detach().clone()
    quantized = rowwise_ternary(original)

    with torch.no_grad():
        target.copy_(original)
    calib_fp = eval_texts(model, tok, calib)
    hold_fp = eval_texts(model, tok, holdout)

    with torch.no_grad():
        target.copy_(quantized)
    calib_q = eval_texts(model, tok, calib)
    hold_q = eval_texts(model, tok, holdout)
    calib_delta = calib_q - calib_fp
    fail_idx = np.argsort(calib_delta)[-FAIL_N:][::-1]
    stable_idx = np.argsort(np.abs(calib_delta))[:STABLE_N]
    selected = list(fail_idx) + list(stable_idx)

    with torch.no_grad():
        target.copy_(original)

    G = torch.stack([sample_grad(model, tok, target, calib[int(i)]) for i in selected])
    bases = {
        "failure_conditioned": failure_basis(G, FAIL_N, K),
        "fisher_all": fisher_basis(G, K),
        "random": random_basis(G.shape[1], K, seed),
    }

    residual = (quantized.float() - original.float()).reshape(-1)
    m, n = original.shape
    out = {
        "layer": layer_idx,
        "seed": seed,
        "baseline_holdout": summarize(hold_fp, hold_q),
        "selected_calibration_failure_delta_mean": float(calib_delta[fail_idx].mean()),
        "methods": {},
    }

    for name, U in bases.items():
        corr = (U @ (U.T @ residual)).reshape_as(original)
        exact = quantized.float() - corr
        with torch.no_grad():
            target.copy_(exact.to(target.dtype))
        exact_vals = eval_texts(model, tok, holdout)
        entry = {
            "correction_energy_fraction": float(corr.pow(2).sum() / residual.pow(2).sum().clamp_min(1e-12)),
            "exact": summarize(hold_fp, exact_vals),
            "low_rank": {},
        }
        for rank in LOW_RANKS:
            lr = low_rank_approx(corr, rank)
            candidate = quantized.float() - lr
            with torch.no_grad():
                target.copy_(candidate.to(target.dtype))
            vals = eval_texts(model, tok, holdout)
            overhead = 16.0 * rank * (m + n) / (m*n)
            entry["low_rank"][str(rank)] = {
                **summarize(hold_fp, vals),
                "factor_overhead_bpw": float(overhead),
                "nominal_ternary_plus_bpw": float(math.log2(3) + overhead),
            }
        out["methods"][name] = entry

    with torch.no_grad():
        target.copy_(original)
    return out

def main():
    torch.set_num_threads(2)
    outdir = Path("results-holdout")
    outdir.mkdir(exist_ok=True)
    calib, holdout = load_texts()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()

    runs = []
    for layer in LAYERS:
        for seed in SEEDS:
            runs.append(run_one(model, tok, calib, holdout, layer, seed))

    rows = []
    for method in ["failure_conditioned", "fisher_all", "random"]:
        for rank in LOW_RANKS:
            values = [r["methods"][method]["low_rank"][str(rank)]["mean_delta_nll"] for r in runs]
            rows.append({
                "method": method,
                "rank": rank,
                "mean_holdout_delta_nll": float(np.mean(values)),
                "std_holdout_delta_nll": float(np.std(values)),
            })

    baseline = np.array([r["baseline_holdout"]["mean_delta_nll"] for r in runs])
    payload = {
        "model": MODEL_ID,
        "calibration_n": len(calib),
        "holdout_n": len(holdout),
        "layers": LAYERS,
        "seeds": SEEDS,
        "k": K,
        "baseline_mean_holdout_delta_nll": float(baseline.mean()),
        "baseline_std": float(baseline.std()),
        "aggregate": rows,
        "runs": runs,
    }
    (outdir/"results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Holdout generalization probe",
        "",
        "Model: " + MODEL_ID,
        "Calibration and holdout sets are disjoint WikiText-2 validation passages.",
        "",
        "| method | rank | mean holdout delta NLL | std |",
        "|---|---:|---:|---:|",
        f"| ternary baseline | 0 | {baseline.mean():.6f} | {baseline.std():.6f} |",
    ]
    for row in rows:
        lines.append(f"| {row['method']} | {row['rank']} | {row['mean_holdout_delta_nll']:.6f} | {row['std_holdout_delta_nll']:.6f} |")
    lines += [
        "",
        "Support criterion: failure-conditioned must beat Fisher-all and random on disjoint holdout averaged across layers and seeds.",
    ]
    (outdir/"summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print((outdir/"summary.md").read_text())

if __name__ == "__main__":
    main()
