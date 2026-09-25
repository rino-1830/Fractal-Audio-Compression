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
SEEDS = (7, 19)
LAYERS = (0, 3)
K = 4
SELECT_N = 4
LOW_RANK = 2
CAL_N = 24
HOLD_N = 24
MODES = ("plain_ternary", "hadamard_ternary")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_corpus(n=96):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    texts = []
    for row in ds:
        t = " ".join(row["text"].split())
        if 80 <= len(t) <= 420:
            texts.append(t)
        if len(texts) >= n:
            break
    if len(texts) < n:
        raise RuntimeError(f"Only found {len(texts)} suitable corpus lines")
    return texts


def encode(tok, text):
    return tok(text, return_tensors="pt", truncation=True, max_length=80)


@torch.no_grad()
def nll(model, tok, text):
    x = encode(tok, text)
    return float(model(**x, labels=x["input_ids"]).loss)


def eval_texts(model, tok, texts):
    return np.array([nll(model, tok, t) for t in texts], dtype=np.float64)


def ternary_groupwise(w, group=128):
    x = w.detach().float()
    n = x.shape[-1]
    if n % group:
        raise ValueError(f"last dim {n} not divisible by group {group}")
    y = x.reshape(*x.shape[:-1], n // group, group)
    mean_abs = y.abs().mean(dim=-1, keepdim=True)
    threshold = 0.7 * mean_abs
    mask = y.abs() >= threshold
    denom = mask.sum(dim=-1, keepdim=True).clamp_min(1)
    scale = (y.abs() * mask).sum(dim=-1, keepdim=True) / denom
    q = scale * y.sign() * mask
    return q.reshape_as(x).to(dtype=w.dtype)


def fwht_last(x):
    n = x.shape[-1]
    if n & (n - 1):
        raise ValueError("Hadamard dimension must be a power of two")
    y = x.reshape(-1, n).clone()
    h = 1
    while h < n:
        v = y.view(-1, n // (2 * h), 2, h)
        a = v[:, :, 0, :].clone()
        b = v[:, :, 1, :].clone()
        v[:, :, 0, :] = a + b
        v[:, :, 1, :] = a - b
        y = v.view(-1, n)
        h *= 2
    return (y / math.sqrt(n)).reshape_as(x)


def quantize_effective(w, mode, sign_seed):
    x = w.detach().float()
    if mode == "plain_ternary":
        return ternary_groupwise(x)
    if mode != "hadamard_ternary":
        raise ValueError(mode)

    g = torch.Generator(device="cpu")
    g.manual_seed(sign_seed)
    signs = torch.randint(0, 2, (x.shape[-1],), generator=g, dtype=torch.int64)
    signs = signs.float().mul_(2).sub_(1)
    # R = H D; stored W' = W R^T = W D H.  Quantize W', then map
    # back to the original basis for ordinary PyTorch evaluation:
    # W_eff = Q(W D H) H D.
    rotated = fwht_last(x * signs)
    q_rot = ternary_groupwise(rotated)
    effective = fwht_last(q_rot) * signs
    return effective.to(dtype=w.dtype)


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
    U = G.T @ (vecs / torch.sqrt(vals).unsqueeze(0))
    return orthonormalize(U)


def failure_basis(G, nf, k):
    n = G.shape[0]
    Kmat = (G @ G.T).double().cpu().numpy()
    fi = np.arange(nf)
    si = np.arange(nf, n)
    A = Kmat[:, fi] @ Kmat[fi, :]
    B = Kmat[:, si] @ Kmat[si, :]
    reg = max(np.trace(Kmat) / max(n, 1), 1e-8) * 1e-2
    vals, vecs = eigh(A, B + reg * np.eye(n), check_finite=False)
    order = np.argsort(vals)[::-1][:k]
    coeff = torch.from_numpy(vecs[:, order]).to(dtype=G.dtype)
    return orthonormalize(G.T @ coeff), vals[order].tolist()


def random_basis(p, k, seed):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return orthonormalize(torch.randn(p, k, generator=gen))


def low_rank_approx(mat, rank):
    q = min(max(rank + 2, rank), min(mat.shape))
    U, S, V = torch.svd_lowrank(mat, q=q, niter=2)
    return (U[:, :rank] * S[:rank]) @ V[:, :rank].T


def directional_correction(residual_flat, U):
    return U @ (U.T @ residual_flat)


def apply_and_eval(target, candidate, model, tok, texts):
    with torch.no_grad():
        target.copy_(candidate.to(dtype=target.dtype))
    return eval_texts(model, tok, texts)


def metrics(fp, vals, baseline_delta):
    d = vals - fp
    top = np.argsort(baseline_delta)[-SELECT_N:]
    return {
        "mean_delta_nll": float(d.mean()),
        "median_delta_nll": float(np.median(d)),
        "baseline_failure_subset_delta_nll": float(d[top].mean()),
        "max_delta_nll": float(d.max()),
    }


def run_one(model, tok, corpus, layer_idx, mode, seed):
    set_seed(seed)
    perm = np.random.permutation(len(corpus))
    cal_texts = [corpus[int(i)] for i in perm[:CAL_N]]
    hold_texts = [corpus[int(i)] for i in perm[CAL_N:CAL_N + HOLD_N]]

    target = model.gpt_neox.layers[layer_idx].mlp.dense_h_to_4h.weight
    original = target.detach().clone()
    sign_seed = 1000 + 97 * layer_idx + seed
    quantized = quantize_effective(original, mode, sign_seed)

    with torch.no_grad():
        target.copy_(original)
    fp_cal = eval_texts(model, tok, cal_texts)
    fp_hold = eval_texts(model, tok, hold_texts)

    with torch.no_grad():
        target.copy_(quantized)
    q_cal = eval_texts(model, tok, cal_texts)
    q_hold = eval_texts(model, tok, hold_texts)
    cal_delta = q_cal - fp_cal
    hold_delta = q_hold - fp_hold

    fail_idx = np.argsort(cal_delta)[-SELECT_N:][::-1]
    stable_idx = np.argsort(cal_delta)[:SELECT_N]
    selected = list(fail_idx) + list(stable_idx)

    with torch.no_grad():
        target.copy_(original)
    G = torch.stack(
        [sample_grad(model, tok, target, cal_texts[int(i)]) for i in selected],
        dim=0,
    )

    ufail, eigs = failure_basis(G, nf=SELECT_N, k=K)
    ufish = fisher_basis(G, K)
    urand = random_basis(G.shape[1], K, seed + 5555)

    residual = (quantized.float() - original.float()).reshape(-1)
    methods = {
        "failure_conditioned": ufail,
        "fisher_all": ufish,
        "random": urand,
    }

    m, n = original.shape
    result = {
        "seed": seed,
        "layer": layer_idx,
        "mode": mode,
        "shape": list(original.shape),
        "baseline": metrics(fp_hold, q_hold, hold_delta),
        "calibration_failure_mean_delta_nll": float(cal_delta[fail_idx].mean()),
        "holdout_failure_mean_delta_nll": float(np.sort(hold_delta)[-SELECT_N:].mean()),
        "failure_generalized_eigenvalues": eigs,
        "methods": {},
    }

    for name, U in methods.items():
        corr_flat = directional_correction(residual, U)
        corr = corr_flat.reshape_as(original)

        exact = quantized.float() - corr
        exact_vals = apply_and_eval(target, exact, model, tok, hold_texts)

        lr_corr = low_rank_approx(corr, LOW_RANK)
        lr_candidate = quantized.float() - lr_corr
        lr_vals = apply_and_eval(target, lr_candidate, model, tok, hold_texts)

        overhead = 16.0 * LOW_RANK * (m + n) / (m * n)
        result["methods"][name] = {
            "correction_energy_fraction": float(
                corr_flat.pow(2).sum() / residual.pow(2).sum().clamp_min(1e-12)
            ),
            "exact": metrics(fp_hold, exact_vals, hold_delta),
            "rank2": {
                **metrics(fp_hold, lr_vals, hold_delta),
                "fp16_factor_overhead_bpw": float(overhead),
                "nominal_ternary_plus_overhead_bpw": float(math.log2(3.0) + overhead),
            },
        }

    with torch.no_grad():
        target.copy_(original)
    return result


def aggregate(runs):
    rows = []
    for mode in MODES:
        for layer in LAYERS:
            subset = [r for r in runs if r["mode"] == mode and r["layer"] == layer]
            base = np.array([r["baseline"]["baseline_failure_subset_delta_nll"] for r in subset])
            for method in ("failure_conditioned", "fisher_all", "random"):
                exact = np.array([
                    r["methods"][method]["exact"]["baseline_failure_subset_delta_nll"]
                    for r in subset
                ])
                rank2 = np.array([
                    r["methods"][method]["rank2"]["baseline_failure_subset_delta_nll"]
                    for r in subset
                ])
                rows.append({
                    "mode": mode,
                    "layer": layer,
                    "method": method,
                    "baseline_failure_delta_mean": float(base.mean()),
                    "exact_failure_delta_mean": float(exact.mean()),
                    "rank2_failure_delta_mean": float(rank2.mean()),
                    "rank2_recovery_fraction": float(
                        1.0 - rank2.mean() / max(base.mean(), 1e-12)
                    ),
                })
    return rows


def main():
    torch.set_num_threads(2)
    outdir = Path("results")
    outdir.mkdir(exist_ok=True)

    # self-test the orthonormal Hadamard transform
    probe = torch.randn(3, 512)
    err = (fwht_last(fwht_last(probe)) - probe).abs().max().item()
    if err > 2e-5:
        raise RuntimeError(f"FWHT self-test failed: {err}")

    corpus = load_corpus()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()

    runs = []
    for mode in MODES:
        for layer in LAYERS:
            for seed in SEEDS:
                print(f"RUN mode={mode} layer={layer} seed={seed}", flush=True)
                runs.append(run_one(model, tok, corpus, layer, mode, seed))

    agg = aggregate(runs)
    payload = {
        "model": MODEL_ID,
        "corpus": "Salesforce/wikitext wikitext-2-raw-v1 test",
        "calibration_n": CAL_N,
        "holdout_n": HOLD_N,
        "seeds": list(SEEDS),
        "layers": list(LAYERS),
        "k": K,
        "low_rank": LOW_RANK,
        "runs": runs,
        "aggregate": agg,
    }
    (outdir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Functional compression probe v2: held-out generalization",
        "",
        f"Model: \`{MODEL_ID}\`",
        f"Calibration/holdout per run: {CAL_N}/{HOLD_N} Wikitext lines; seeds {SEEDS}.",
        "",
        "The protected subspace is learned only from calibration examples. The reported failure subset is selected from the baseline quantization degradation on the unseen holdout set, before any correction is evaluated.",
        "",
        "| quantizer | layer | method | baseline failure ΔNLL | exact protected ΔNLL | rank-2 ΔNLL | rank-2 recovery |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in agg:
        lines.append(
            f"| {row['mode']} | {row['layer']} | {row['method']} | "
            f"{row['baseline_failure_delta_mean']:.6f} | "
            f"{row['exact_failure_delta_mean']:.6f} | "
            f"{row['rank2_failure_delta_mean']:.6f} | "
            f"{100*row['rank2_recovery_fraction']:.1f}% |"
        )
    lines += [
        "",
        "Interpretation: support requires failure-conditioned correction to beat Fisher-all and random on unseen holdout data across both ordinary ternary and Hadamard-rotated ternary settings. Negative recovery means the correction made holdout degradation worse.",
    ]
    summary = "\n".join(lines) + "\n"
    (outdir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
