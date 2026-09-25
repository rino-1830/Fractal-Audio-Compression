import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy.linalg import eigh
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "prism-ml/Bonsai-1.7B-unpacked"
LAYER = int(os.environ.get("CASE_LAYER", "27"))
FRAC = float(os.environ.get("CASE_FRAC", "0.50"))
NUM_MASKS = 16
SELECT_N = 4
BASIS_N = 8
HOLD_N = 12
FAIL_N = 3
STABLE_N = 3
K = 2
MAX_LENGTH = 20
GROUP_SIZE = 128
SKIP_ELIGIBLE = 320
OUT = Path("results-codebook-residual")


def load_texts():
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    xs = []
    skipped = 0
    need = SELECT_N + BASIS_N + HOLD_N
    for row in ds:
        t = " ".join(row["text"].split())
        if len(t) < 100 or t.startswith("="):
            continue
        if skipped < SKIP_ELIGIBLE:
            skipped += 1
            continue
        xs.append(t[:500])
        if len(xs) >= need:
            break
    return xs[:SELECT_N], xs[SELECT_N:SELECT_N+BASIS_N], xs[SELECT_N+BASIS_N:need]


def encode(tok, text):
    return tok(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)


@torch.no_grad()
def losses(model, tok, texts):
    out = []
    for text in texts:
        x = encode(tok, text)
        out.append(float(model(**x, labels=x["input_ids"]).loss))
    return np.array(out, dtype=np.float64)


def mask_weight(w, code):
    x = w.detach().clone()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(987654321 + code * 1000003 + LAYER * 1009 + int(FRAC * 1000))
    mask = torch.rand(x.shape, generator=gen) < FRAC
    x[mask] = 0
    return x


def metrics(fp, vals):
    d = vals - fp
    return {
        "mean_delta_nll": float(d.mean()),
        "mean_positive_delta_nll": float(np.maximum(d, 0.0).mean()),
        "max_delta_nll": float(d.max()),
        "fraction_worse": float(np.mean(d > 0)),
        "per_sample_delta": d.tolist(),
    }


def selection_score(m):
    return (m["mean_positive_delta_nll"], m["mean_delta_nll"], m["max_delta_nll"])


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


def failure_basis(G, nf, k):
    n = G.shape[0]
    kg = (G @ G.T).double().cpu().numpy()
    fi = np.arange(nf)
    si = np.arange(nf, n)
    A = kg[:, fi] @ kg[fi, :]
    B = kg[:, si] @ kg[si, :]
    reg = max(float(np.trace(kg)) / max(n, 1), 1e-8) * 1e-2
    vals, vecs = eigh(A, B + reg * np.eye(n), check_finite=False)
    idx = np.argsort(vals)[::-1][:k]
    coeff = torch.from_numpy(vecs[:, idx]).to(dtype=G.dtype)
    return orth(G.T @ coeff), vals[idx].tolist()


def low_rank(mat, rank=2):
    q = min(rank + 2, min(mat.shape))
    U, S, V = torch.svd_lowrank(mat, q=q, niter=2)
    return (U[:, :rank] * S[:rank]) @ V[:, :rank].T


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    select_texts, basis_texts, hold_texts = load_texts()

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
    target.requires_grad_(True)

    with torch.no_grad():
        target.copy_(original)
    select_fp = losses(model, tok, select_texts)
    basis_fp = losses(model, tok, basis_texts)
    hold_fp = losses(model, tok, hold_texts)

    search = []
    best_code = None
    best_score = None
    for code in range(NUM_MASKS):
        masked = mask_weight(original, code)
        with torch.no_grad():
            target.copy_(masked)
        vals = losses(model, tok, select_texts)
        m = metrics(select_fp, vals)
        s = selection_score(m)
        search.append({"code": code, "selection": m})
        if best_score is None or s < best_score:
            best_score = s
            best_code = code
        print("MASK", code, s, flush=True)

    fixed = mask_weight(original, 0)
    selected = mask_weight(original, best_code)

    with torch.no_grad():
        target.copy_(fixed)
    fixed_hold = metrics(hold_fp, losses(model, tok, hold_texts))

    with torch.no_grad():
        target.copy_(selected)
    selected_basis_vals = losses(model, tok, basis_texts)
    selected_hold_vals = losses(model, tok, hold_texts)
    selected_hold = metrics(hold_fp, selected_hold_vals)

    basis_delta = selected_basis_vals - basis_fp
    fail_idx = np.argsort(basis_delta)[-FAIL_N:][::-1]
    stable_idx = np.argsort(np.abs(basis_delta))[:STABLE_N]

    with torch.no_grad():
        target.copy_(original)

    G = torch.stack([
        sample_grad(model, tok, target, basis_texts[int(i)])
        for i in list(fail_idx) + list(stable_idx)
    ])
    U, eigs = failure_basis(G, FAIL_N, K)

    residual = (selected.float() - original.float()).reshape(-1)
    corr = (U @ (U.T @ residual)).reshape_as(original)
    lr = low_rank(corr, 2)
    corrected = selected.float() - lr

    with torch.no_grad():
        target.copy_(corrected.to(target.dtype))
    corrected_hold = metrics(hold_fp, losses(model, tok, hold_texts))

    m, n = original.shape
    factor_bpw = 16.0 * 2 * (m+n)/(m*n)
    index_bpw = float(np.ceil(np.log2(NUM_MASKS))) / original.numel()
    selected_bpw = (1.0 - FRAC) + 16.0/GROUP_SIZE + index_bpw
    corrected_bpw = selected_bpw + factor_bpw

    with torch.no_grad():
        target.copy_(original)
    target.requires_grad_(False)

    payload = {
        "model": MODEL_ID,
        "layer": LAYER,
        "prune_fraction": FRAC,
        "num_masks": NUM_MASKS,
        "best_code": best_code,
        "search": search,
        "failure_eigenvalues": eigs,
        "fixed_code0_holdout": fixed_hold,
        "selected_mask_holdout": selected_hold,
        "selected_plus_fc_holdout": corrected_hold,
        "selected_mask_bpw": float(selected_bpw),
        "selected_plus_fc_bpw": float(corrected_bpw),
        "correction_energy_fraction": float(corr.pow(2).sum()/residual.pow(2).sum().clamp_min(1e-12)),
    }

    stem = f"l{LAYER}_p{int(FRAC*100)}"
    (OUT/f"{stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "best_code": best_code,
        "fixed_positive_harm": fixed_hold["mean_positive_delta_nll"],
        "selected_positive_harm": selected_hold["mean_positive_delta_nll"],
        "selected_fc_positive_harm": corrected_hold["mean_positive_delta_nll"],
        "fixed_mean_delta": fixed_hold["mean_delta_nll"],
        "selected_mean_delta": selected_hold["mean_delta_nll"],
        "selected_fc_mean_delta": corrected_hold["mean_delta_nll"],
        "selected_bpw": selected_bpw,
        "selected_fc_bpw": corrected_bpw,
        "correction_energy_fraction": payload["correction_energy_fraction"],
    }, indent=2))


if __name__ == "__main__":
    main()
