import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import eigh
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "EleutherAI/pythia-70m-deduped")
SEED = 7
K = 4
LOW_RANKS = (1, 2, 4)

TEXTS = [
    "The capital of France is Paris and the capital of Japan is Tokyo.",
    "Water freezes at zero degrees Celsius under standard atmospheric pressure.",
    "A triangle has three sides and the sum of its interior angles is 180 degrees in Euclidean geometry.",
    "The Earth orbits the Sun once each year.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "If Alice has three apples and buys two more, she has five apples.",
    "Ten divided by two equals five.",
    "The opposite of north is south.",
    "A kilogram contains one thousand grams.",
    "The Pacific Ocean is larger than the Atlantic Ocean.",
    "A prime number has exactly two positive divisors.",
    "The Moon reflects sunlight and does not produce visible light of its own.",
    "In binary notation, the decimal number two is written as 10.",
    "The derivative of x squared is two x.",
    "A square has four equal sides.",
    "Sound travels through air as a pressure wave.",
    "An hour contains sixty minutes.",
    "The chemical symbol for oxygen is O.",
    "Multiplying any finite number by zero gives zero.",
    "The human heart pumps blood through the circulatory system.",
]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def encode(tok, text):
    return tok(text, return_tensors="pt", truncation=True, max_length=64)

@torch.no_grad()
def nll(model, tok, text):
    x = encode(tok, text)
    out = model(**x, labels=x["input_ids"])
    return float(out.loss)

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
    out = model(**x, labels=x["input_ids"])
    g = torch.autograd.grad(out.loss, target, retain_graph=False, create_graph=False)[0]
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
    U = G.T @ (vecs / torch.sqrt(vals).unsqueeze(0))
    return orthonormalize(U)

def failure_basis(G, nf, k):
    # G rows are [failure gradients, stable gradients].
    n = G.shape[0]
    Kmat = (G @ G.T).double().cpu().numpy()
    fi = np.arange(nf)
    si = np.arange(nf, n)
    A = Kmat[:, fi] @ Kmat[fi, :]
    B = Kmat[:, si] @ Kmat[si, :]
    reg = max(np.trace(Kmat) / max(n, 1), 1e-8) * 1e-2
    B = B + reg * np.eye(n)
    vals, vecs = eigh(A, B, check_finite=False)
    order = np.argsort(vals)[::-1][:k]
    coeff = torch.from_numpy(vecs[:, order]).to(dtype=G.dtype)
    U = G.T @ coeff
    return orthonormalize(U), vals[order].tolist()

def random_basis(p, k):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(SEED + 100)
    return orthonormalize(torch.randn(p, k, generator=gen))

def directional_correction(residual_flat, U):
    return U @ (U.T @ residual_flat)

def low_rank_approx(mat, rank):
    # Randomized low-rank SVD; rank <= 4 here.
    q = min(max(rank + 2, rank), min(mat.shape))
    U, S, V = torch.svd_lowrank(mat, q=q, niter=2)
    return (U[:, :rank] * S[:rank]) @ V[:, :rank].T

def apply_and_eval(target, candidate, model, tok):
    with torch.no_grad():
        target.copy_(candidate.to(dtype=target.dtype))
    vals = eval_texts(model, tok, TEXTS)
    return vals

def summarize_delta(fp, vals, fail_idx, stable_idx):
    d = vals - fp
    return {
        "mean_delta_nll": float(d.mean()),
        "median_delta_nll": float(np.median(d)),
        "failure_mean_delta_nll": float(d[fail_idx].mean()),
        "stable_mean_delta_nll": float(d[stable_idx].mean()),
        "max_delta_nll": float(d.max()),
    }

def main():
    set_seed(SEED)
    torch.set_num_threads(2)
    outdir = Path("results")
    outdir.mkdir(exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()

    target = model.gpt_neox.layers[0].mlp.dense_h_to_4h.weight
    original = target.detach().clone()
    quantized = rowwise_ternary(original)

    fp = eval_texts(model, tok, TEXTS)
    with torch.no_grad():
        target.copy_(quantized)
    qvals = eval_texts(model, tok, TEXTS)
    qdelta = qvals - fp

    fail_idx = np.argsort(qdelta)[-4:][::-1]
    stable_idx = np.argsort(qdelta)[:4]
    selected = list(fail_idx) + list(stable_idx)

    with torch.no_grad():
        target.copy_(original)

    grads = []
    for idx in selected:
        grads.append(sample_grad(model, tok, target, TEXTS[int(idx)]))
    G = torch.stack(grads, dim=0)

    ufail, gen_eigs = failure_basis(G, nf=len(fail_idx), k=K)
    ufish = fisher_basis(G, K)
    urand = random_basis(G.shape[1], K)

    residual = (quantized.float() - original.float()).reshape(-1)
    methods = {
        "failure_conditioned": ufail,
        "fisher_all": ufish,
        "random": urand,
    }

    results = {
        "model": MODEL_ID,
        "target_shape": list(original.shape),
        "num_target_weights": int(original.numel()),
        "seed": SEED,
        "k": K,
        "failure_indices": [int(x) for x in fail_idx],
        "stable_indices": [int(x) for x in stable_idx],
        "failure_generalized_eigenvalues": gen_eigs,
        "baseline": {
            "fp_mean_nll": float(fp.mean()),
            "ternary": summarize_delta(fp, qvals, fail_idx, stable_idx),
        },
        "methods": {},
    }

    m, n = original.shape
    for name, U in methods.items():
        corr_flat = directional_correction(residual, U)
        corr = corr_flat.reshape_as(original)

        exact_candidate = quantized.float() - corr
        exact_vals = apply_and_eval(target, exact_candidate, model, tok)
        entry = {
            "exact_directional_correction": summarize_delta(fp, exact_vals, fail_idx, stable_idx),
            "correction_energy_fraction": float(corr_flat.pow(2).sum() / residual.pow(2).sum().clamp_min(1e-12)),
            "low_rank": {},
        }

        for rank in LOW_RANKS:
            lr_corr = low_rank_approx(corr, rank)
            candidate = quantized.float() - lr_corr
            vals = apply_and_eval(target, candidate, model, tok)
            overhead = 16.0 * rank * (m + n) / (m * n)
            entry["low_rank"][str(rank)] = {
                **summarize_delta(fp, vals, fail_idx, stable_idx),
                "fp16_factor_overhead_bits_per_target_weight": float(overhead),
                "ternary_plus_overhead_bpw_nominal": float(math.log2(3.0) + overhead),
            }
        results["methods"][name] = entry

    with torch.no_grad():
        target.copy_(original)

    (outdir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    base = results["baseline"]["ternary"]
    lines = [
        "# Functional compression probe",
        "",
        f"Model: \`{MODEL_ID}\`",
        f"Target: first MLP expansion matrix, shape {tuple(original.shape)}, {original.numel():,} weights",
        "",
        "The probe ternary-quantizes one matrix, identifies the four prompts with the largest NLL degradation and four with the smallest degradation, then compares equal-dimensional protected subspaces.",
        "",
        "| method | correction | mean ΔNLL | failure-set ΔNLL | overhead bpw (target matrix) |",
        "|---|---:|---:|---:|---:|",
        f"| ternary baseline | none | {base['mean_delta_nll']:.6f} | {base['failure_mean_delta_nll']:.6f} | 0 |",
    ]
    for name, entry in results["methods"].items():
        ex = entry["exact_directional_correction"]
        lines.append(f"| {name} | exact k={K} directional | {ex['mean_delta_nll']:.6f} | {ex['failure_mean_delta_nll']:.6f} | not storage-efficient |")
        for rank in LOW_RANKS:
            r = entry["low_rank"][str(rank)]
            lines.append(f"| {name} | rank-{rank} factor | {r['mean_delta_nll']:.6f} | {r['failure_mean_delta_nll']:.6f} | +{r['fp16_factor_overhead_bits_per_target_weight']:.4f} |")

    lines += [
        "",
        "Interpretation rule: the hypothesis gets preliminary support only if failure-conditioned correction consistently reduces failure-set degradation more than both Fisher-all and random at the same correction rank. A single small-model run is not sufficient to establish generality.",
    ]
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((outdir / "summary.md").read_text())

if __name__ == "__main__":
    main()
