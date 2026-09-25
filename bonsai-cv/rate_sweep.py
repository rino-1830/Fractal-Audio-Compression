import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy.linalg import eigh
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1] / "llm-functional-compression"))
import bonsai_probe as bp

MODEL_ID = bp.MODEL_ID
LAYERS = [0, 14, 27]
GROUP_SIZES = [128, 256, 512]
FOLDS = 3
CAL_N = 8
HOLD_N = 8
FAIL_N = 3
STABLE_N = 3
K = 2
MAX_LENGTH = 20
EIG_GATE = 10.0
OUT = Path("results-bonsai-rate-sweep")


def load_text_pool():
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    xs = []
    for row in ds:
        t = " ".join(row["text"].split())
        if len(t) >= 100 and not t.startswith("="):
            xs.append(t[:500])
        if len(xs) >= FOLDS * (CAL_N + HOLD_N):
            break
    if len(xs) < FOLDS * (CAL_N + HOLD_N):
        raise RuntimeError("not enough validation text")
    return xs


def encode(tok, text):
    return tok(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)


@torch.no_grad()
def nll(model, tok, text):
    x = encode(tok, text)
    return float(model(**x, labels=x["input_ids"]).loss)


def eval_texts(model, tok, texts):
    return np.array([nll(model, tok, t) for t in texts], dtype=np.float64)


def binary_from_ternary(w, group_size, seed):
    x = w.detach().float()
    flat = x.reshape(-1)
    n = flat.numel()
    pad = (-n) % group_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
    groups = flat.reshape(-1, group_size)
    scale = groups.abs().mean(dim=1, keepdim=True).clamp_min(1e-12)
    sign = groups.sign()
    z = sign == 0
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    fill = torch.randint(0, 2, sign.shape, generator=gen, dtype=torch.int8).float() * 2 - 1
    sign[z] = fill[z]
    return (sign * scale).reshape(-1)[:n].reshape_as(x).to(dtype=w.dtype)


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


def fisher_basis(G, k):
    kg = G @ G.T
    vals, vecs = torch.linalg.eigh(kg)
    order = torch.argsort(vals, descending=True)[:k]
    vals = vals[order].clamp_min(1e-12)
    vecs = vecs[:, order]
    return orth(G.T @ (vecs / torch.sqrt(vals).unsqueeze(0)))


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


def run_one(model, tok, calib, holdout, layer, group_size, fold):
    target = model.model.layers[layer].mlp.down_proj.weight
    target.requires_grad_(True)
    original = target.detach().clone()
    seed = 100000 + layer * 1000 + group_size + fold
    binary = binary_from_ternary(original, group_size, seed)

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

    with torch.no_grad():
        target.copy_(original)

    G = torch.stack([
        sample_grad(model, tok, target, calib[int(i)])
        for i in list(fail_idx) + list(stable_idx)
    ])
    uf, eigs = failure_basis(G, FAIL_N, K)
    ui = fisher_basis(G, K)

    residual = (binary.float() - original.float()).reshape(-1)
    m, n = original.shape
    eig_geom = float(math.sqrt(max(eigs[0], 0.0) * max(eigs[1], 0.0)))
    gate_open = eig_geom >= EIG_GATE

    out = {
        "layer": layer,
        "group_size": group_size,
        "fold": fold,
        "eigs": eigs,
        "eig_geom": eig_geom,
        "gate_open": gate_open,
        "baseline": stats(hold_fp, hold_bin),
        "methods": {},
    }

    for name, U in {"failure_conditioned": uf, "fisher_all": ui}.items():
        corr = (U @ (U.T @ residual)).reshape_as(original)
        entry = {
            "correction_energy_fraction": float(
                corr.pow(2).sum() / residual.pow(2).sum().clamp_min(1e-12)
            ),
            "ranks": {},
        }
        for rank in [1, 2]:
            lr = low_rank(corr, rank)
            candidate = binary.float() - lr
            with torch.no_grad():
                target.copy_(candidate.to(target.dtype))
            vals = eval_texts(model, tok, holdout)
            factor_bpw = 16.0 * rank * (m + n) / (m * n)
            nominal_bpw = 1.0 + 16.0 / group_size + factor_bpw
            entry["ranks"][str(rank)] = {
                **stats(hold_fp, vals),
                "factor_bpw": float(factor_bpw),
                "nominal_bpw": float(nominal_bpw),
            }
        out["methods"][name] = entry

    # Gated rank-2: if separation is weak, emit the binary baseline unchanged.
    if gate_open:
        gated = out["methods"]["failure_conditioned"]["ranks"]["2"].copy()
    else:
        gated = {
            **out["baseline"],
            "factor_bpw": 0.0,
            "nominal_bpw": float(1.0 + 16.0 / group_size),
        }
    out["gated_failure_rank2"] = gated

    with torch.no_grad():
        target.copy_(original)
    target.requires_grad_(False)
    return out


def mean_recovery(rows, key_fn):
    vals = []
    for r in rows:
        b = r["baseline"]["mean_delta_nll"]
        v = key_fn(r)
        if abs(b) > 1e-12:
            vals.append((b - v) / b * 100.0)
    return float(np.mean(vals)) if vals else float("nan")


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)

    xs = load_text_pool()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    rows = []
    for layer in LAYERS:
        for group_size in GROUP_SIZES:
            for fold in range(FOLDS):
                base = fold * (CAL_N + HOLD_N)
                calib = xs[base:base + CAL_N]
                holdout = xs[base + CAL_N:base + CAL_N + HOLD_N]
                print("START", layer, group_size, fold, flush=True)
                row = run_one(model, tok, calib, holdout, layer, group_size, fold)
                rows.append(row)
                print(
                    "RESULT", layer, group_size, fold,
                    row["baseline"]["mean_delta_nll"],
                    row["methods"]["failure_conditioned"]["ranks"]["2"]["mean_delta_nll"],
                    row["methods"]["fisher_all"]["ranks"]["2"]["mean_delta_nll"],
                    row["gated_failure_rank2"]["mean_delta_nll"],
                    row["eig_geom"], row["gate_open"],
                    flush=True,
                )

    aggregate = {}
    for group_size in GROUP_SIZES:
        subset = [r for r in rows if r["group_size"] == group_size]
        base = np.array([r["baseline"]["mean_delta_nll"] for r in subset])
        agg = {
            "baseline_mean_delta_nll": float(base.mean()),
            "binary_bpw": float(1.0 + 16.0 / group_size),
        }
        for rank in [1, 2]:
            for method in ["failure_conditioned", "fisher_all"]:
                vals = np.array([
                    r["methods"][method]["ranks"][str(rank)]["mean_delta_nll"]
                    for r in subset
                ])
                agg[f"{method}_rank{rank}"] = {
                    "mean_delta_nll": float(vals.mean()),
                    "mean_recovery_pct": mean_recovery(
                        subset,
                        lambda r, m=method, q=rank:
                            r["methods"][m]["ranks"][str(q)]["mean_delta_nll"],
                    ),
                    "nominal_bpw": float(
                        subset[0]["methods"][method]["ranks"][str(rank)]["nominal_bpw"]
                    ),
                }
        gated_vals = np.array([
            r["gated_failure_rank2"]["mean_delta_nll"] for r in subset
        ])
        agg["gated_failure_rank2"] = {
            "mean_delta_nll": float(gated_vals.mean()),
            "mean_recovery_pct": mean_recovery(
                subset, lambda r: r["gated_failure_rank2"]["mean_delta_nll"]
            ),
            "gate_open_fraction": float(np.mean([r["gate_open"] for r in subset])),
        }
        aggregate[str(group_size)] = agg

    payload = {
        "settings": {
            "layers": LAYERS,
            "group_sizes": GROUP_SIZES,
            "folds": FOLDS,
            "calibration_n": CAL_N,
            "holdout_n": HOLD_N,
            "rank_candidates": [1, 2],
            "eig_gate_geometric_mean_threshold": EIG_GATE,
            "data_split": "WikiText-2 validation",
        },
        "rows": rows,
        "aggregate": aggregate,
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Bonsai binary rate sweep with pre-registered eigengate",
        "",
        "The eigengate threshold is fixed at geometric-mean(lambda1,lambda2) >= 10 before this run.",
        "Data uses WikiText-2 validation, disjoint from the previous deep-CV test split.",
        "",
        "| group | binary bpw | FC r1 recovery | FC r1 bpw | FC r2 recovery | FC r2 bpw | Fisher r2 recovery | gated FC r2 recovery | gate open |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group_size in GROUP_SIZES:
        a = aggregate[str(group_size)]
        lines.append(
            f"| {group_size} | {a['binary_bpw']:.4f} | "
            f"{a['failure_conditioned_rank1']['mean_recovery_pct']:.2f}% | "
            f"{a['failure_conditioned_rank1']['nominal_bpw']:.4f} | "
            f"{a['failure_conditioned_rank2']['mean_recovery_pct']:.2f}% | "
            f"{a['failure_conditioned_rank2']['nominal_bpw']:.4f} | "
            f"{a['fisher_all_rank2']['mean_recovery_pct']:.2f}% | "
            f"{a['gated_failure_rank2']['mean_recovery_pct']:.2f}% | "
            f"{a['gated_failure_rank2']['gate_open_fraction']:.2f} |"
        )
    lines += [
        "",
        "This remains a per-matrix perturbation study; nominal bpw is for the tested down-projection matrix, not the complete model.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
