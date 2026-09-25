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

MODEL_ID = "prism-ml/Ternary-Bonsai-1.7B-unpacked"
LAYER = 0
GROUP_SIZE = 128
K = 2
FAIL_N = 3
STABLE_N = 3
CALIBRATION_N = 8
HOLDOUT_N = 8
MAX_LENGTH = 24
SEED = 17

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_texts():
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    texts = []
    for row in ds:
        t = " ".join(row["text"].split())
        if len(t) >= 100 and not t.startswith("="):
            texts.append(t[:500])
        if len(texts) >= CALIBRATION_N + HOLDOUT_N:
            break
    if len(texts) < CALIBRATION_N + HOLDOUT_N:
        raise RuntimeError("not enough text")
    return texts[:CALIBRATION_N], texts[CALIBRATION_N:CALIBRATION_N+HOLDOUT_N]

def encode(tok, text):
    return tok(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)

@torch.no_grad()
def nll(model, tok, text):
    x = encode(tok, text)
    return float(model(**x, labels=x["input_ids"]).loss)

def eval_texts(model, tok, texts):
    return np.array([nll(model, tok, t) for t in texts], dtype=np.float64)

def deterministic_binary_from_ternary(w):
    x = w.detach().float()
    flat = x.reshape(-1)
    n = flat.numel()
    pad = (-n) % GROUP_SIZE
    if pad:
        padded = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
    else:
        padded = flat
    groups = padded.reshape(-1, GROUP_SIZE)
    scale = groups.abs().mean(dim=1, keepdim=True).clamp_min(1e-12)

    sign = groups.sign()
    z = sign == 0
    gen = torch.Generator(device="cpu")
    gen.manual_seed(SEED)
    rand_sign = torch.randint(0, 2, sign.shape, generator=gen, dtype=torch.int8).float() * 2 - 1
    sign[z] = rand_sign[z]
    q = sign * scale
    q = q.reshape(-1)[:n].reshape_as(x)
    return q.to(dtype=w.dtype)

def sample_grad(model, tok, target, text):
    model.zero_grad(set_to_none=True)
    x = encode(tok, text)
    loss = model(**x, labels=x["input_ids"]).loss
    g = torch.autograd.grad(loss, target, retain_graph=False, create_graph=False)[0]
    g = g.detach().float().reshape(-1)
    return g / g.norm().clamp_min(1e-12)

def orth(cols):
    q, _ = torch.linalg.qr(cols, mode="reduced")
    return q

def fisher_basis(G, k):
    kg = G @ G.T
    vals, vecs = torch.linalg.eigh(kg)
    order = torch.argsort(vals, descending=True)[:k]
    vals = vals[order].clamp_min(1e-12)
    vecs = vecs[:, order]
    return orth(G.T @ (vecs / torch.sqrt(vals).unsqueeze(0)))

def failure_basis(G, nf, k):
    n = G.shape[0]
    kg = (G @ G.T).double().cpu().numpy()
    fi = np.arange(nf)
    si = np.arange(nf, n)
    A = kg[:, fi] @ kg[fi, :]
    B = kg[:, si] @ kg[si, :]
    reg = max(float(np.trace(kg))/max(n,1), 1e-8) * 1e-2
    vals, vecs = eigh(A, B + reg*np.eye(n), check_finite=False)
    idx = np.argsort(vals)[::-1][:k]
    coeff = torch.from_numpy(vecs[:, idx]).to(dtype=G.dtype)
    return orth(G.T @ coeff), vals[idx].tolist()

def random_basis(p, k):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(SEED + 999)
    return orth(torch.randn(p, k, generator=gen))

def low_rank(mat, rank):
    q = min(rank + 2, min(mat.shape))
    U, S, V = torch.svd_lowrank(mat, q=q, niter=2)
    return (U[:, :rank] * S[:rank]) @ V[:, :rank].T

def stats(fp, vals):
    d = vals - fp
    return {
        "mean_delta_nll": float(d.mean()),
        "median_delta_nll": float(np.median(d)),
        "max_delta_nll": float(d.max()),
    }

def main():
    set_seed(SEED)
    torch.set_num_threads(2)
    outdir = Path("results-bonsai")
    outdir.mkdir(exist_ok=True)

    calib, holdout = load_texts()
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
    target.requires_grad_(True)
    original = target.detach().clone()
    binary = deterministic_binary_from_ternary(original)

    with torch.no_grad():
        target.copy_(original)
    calib_fp = eval_texts(model, tok, calib)
    hold_fp = eval_texts(model, tok, holdout)

    with torch.no_grad():
        target.copy_(binary)
    calib_bin = eval_texts(model, tok, calib)
    hold_bin = eval_texts(model, tok, holdout)

    delta = calib_bin - calib_fp
    fail_idx = np.argsort(delta)[-FAIL_N:][::-1]
    stable_idx = np.argsort(np.abs(delta))[:STABLE_N]
    chosen = list(fail_idx) + list(stable_idx)

    with torch.no_grad():
        target.copy_(original)
    G = torch.stack([sample_grad(model, tok, target, calib[int(i)]) for i in chosen])

    uf, eigs = failure_basis(G, FAIL_N, K)
    ui = fisher_basis(G, K)
    ur = random_basis(G.shape[1], K)

    residual = (binary.float() - original.float()).reshape(-1)
    m, n = original.shape

    result = {
        "model": MODEL_ID,
        "layer": LAYER,
        "target": "mlp.down_proj.weight",
        "shape": [m, n],
        "weights": int(original.numel()),
        "group_size": GROUP_SIZE,
        "baseline_binary": stats(hold_fp, hold_bin),
        "calibration_failure_delta_mean": float(delta[fail_idx].mean()),
        "generalized_eigenvalues": eigs,
        "methods": {},
    }

    for name, U in {
        "failure_conditioned": uf,
        "fisher_all": ui,
        "random": ur,
    }.items():
        corr = (U @ (U.T @ residual)).reshape_as(original)

        exact = binary.float() - corr
        with torch.no_grad():
            target.copy_(exact.to(target.dtype))
        exact_vals = eval_texts(model, tok, holdout)

        lr = low_rank(corr, 2)
        candidate = binary.float() - lr
        with torch.no_grad():
            target.copy_(candidate.to(target.dtype))
        rank2_vals = eval_texts(model, tok, holdout)

        overhead = 16.0 * 2 * (m+n)/(m*n)
        result["methods"][name] = {
            "correction_energy_fraction": float(corr.pow(2).sum()/residual.pow(2).sum().clamp_min(1e-12)),
            "exact": stats(hold_fp, exact_vals),
            "rank2": {
                **stats(hold_fp, rank2_vals),
                "factor_overhead_bpw": float(overhead),
                "binary_plus_overhead_nominal_bpw": float(1.0 + 16.0/GROUP_SIZE + overhead),
            },
        }

    with torch.no_grad():
        target.copy_(original)

    (outdir/"results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    b = result["baseline_binary"]["mean_delta_nll"]
    lines = [
        "# Ternary-Bonsai 1.7B further-compression probe",
        "",
        "Starting model: actual Prism ML Ternary-Bonsai-1.7B unpacked ternary weights.",
        "Perturbation: one MLP down-projection is collapsed from ternary to group-128 binary.",
        "Calibration and holdout WikiText passages are disjoint.",
        "",
        "| method | mean holdout delta NLL | rank-2 nominal bpw for target matrix |",
        "|---|---:|---:|",
        f"| binary baseline | {b:.6f} | {1.0 + 16.0/GROUP_SIZE:.6f} |",
    ]
    for name, entry in result["methods"].items():
        r = entry["rank2"]
        lines.append(f"| {name} | {r['mean_delta_nll']:.6f} | {r['binary_plus_overhead_nominal_bpw']:.6f} |")
    lines += [
        "",
        "Interpretation: this is a one-matrix stress test, not a whole-model compression result.",
    ]
    (outdir/"summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print((outdir/"summary.md").read_text())

if __name__ == "__main__":
    main()
