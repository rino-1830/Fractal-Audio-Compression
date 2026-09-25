import json
import math
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy.linalg import eigh
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "prism-ml/Bonsai-1.7B-unpacked"
LAYERS = [0, 14, 27]
PRUNE_FRACS = [0.25, 0.50]
FOLDS = 3
CAL_N = 8
HOLD_N = 8
FAIL_N = 3
STABLE_N = 3
K = 2
MAX_LENGTH = 20
EIG_GATE = 10.0
OUT = Path("results-onebit-thinning")


def load_text_pool():
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    xs = []
    for row in ds:
        t = " ".join(row["text"].split())
        if len(t) >= 100 and not t.startswith("="):
            xs.append(t[:500])
        if len(xs) >= FOLDS * (CAL_N + HOLD_N):
            break
    return xs


def encode(tok, text):
    return tok(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)


@torch.no_grad()
def nll(model, tok, text):
    x = encode(tok, text)
    return float(model(**x, labels=x["input_ids"]).loss)


def eval_texts(model, tok, texts):
    return np.array([nll(model, tok, t) for t in texts], dtype=np.float64)


def implicit_prune(w, frac, seed):
    x = w.detach().clone()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    mask = torch.rand(x.shape, generator=gen) < frac
    x[mask] = 0
    return x, float(mask.float().mean())


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


def low_rank(mat, rank=2):
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


def run_one(model, tok, calib, holdout, layer, frac, fold):
    target = model.model.layers[layer].mlp.down_proj.weight
    target.requires_grad_(True)
    original = target.detach().clone()
    seed = 700000 + layer * 1000 + int(frac * 100) * 10 + fold
    pruned, actual_frac = implicit_prune(original, frac, seed)

    with torch.no_grad():
        target.copy_(original)
    calib_fp = eval_texts(model, tok, calib)
    hold_fp = eval_texts(model, tok, holdout)

    with torch.no_grad():
        target.copy_(pruned)
    calib_p = eval_texts(model, tok, calib)
    hold_p = eval_texts(model, tok, holdout)

    delta = calib_p - calib_fp
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

    residual = (pruned.float() - original.float()).reshape(-1)
    m, n = original.shape
    eig_geom = float(math.sqrt(max(eigs[0], 0.0) * max(eigs[1], 0.0)))
    gate_open = eig_geom >= EIG_GATE

    out = {
        "layer": layer,
        "prune_fraction_requested": frac,
        "prune_fraction_actual": actual_frac,
        "fold": fold,
        "eigs": eigs,
        "eig_geom": eig_geom,
        "gate_open": gate_open,
        "baseline": stats(hold_fp, hold_p),
        "methods": {},
    }

    for name, U in {"failure_conditioned": uf, "fisher_all": ui}.items():
        corr = (U @ (U.T @ residual)).reshape_as(original)
        lr = low_rank(corr, 2)
        candidate = pruned.float() - lr
        with torch.no_grad():
            target.copy_(candidate.to(target.dtype))
        vals = eval_texts(model, tok, holdout)
        factor_bpw = 16.0 * 2 * (m + n) / (m * n)
        out["methods"][name] = {
            **stats(hold_fp, vals),
            "correction_energy_fraction": float(
                corr.pow(2).sum() / residual.pow(2).sum().clamp_min(1e-12)
            ),
            "residual_factor_bpw": float(factor_bpw),
            "sign_payload_plus_residual_bpw": float((1.0 - frac) + factor_bpw),
        }

    if gate_open:
        out["gated_failure_rank2"] = out["methods"]["failure_conditioned"].copy()
    else:
        out["gated_failure_rank2"] = {
            **out["baseline"],
            "correction_energy_fraction": 0.0,
            "residual_factor_bpw": 0.0,
            "sign_payload_plus_residual_bpw": float(1.0 - frac),
        }

    with torch.no_grad():
        target.copy_(original)
    target.requires_grad_(False)
    return out


def recovery(row, value):
    b = row["baseline"]["mean_delta_nll"]
    return (b - value) / b * 100.0 if abs(b) > 1e-12 else float("nan")


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
        for frac in PRUNE_FRACS:
            for fold in range(FOLDS):
                base = fold * (CAL_N + HOLD_N)
                calib = xs[base:base + CAL_N]
                holdout = xs[base + CAL_N:base + CAL_N + HOLD_N]
                print("START", layer, frac, fold, flush=True)
                row = run_one(model, tok, calib, holdout, layer, frac, fold)
                rows.append(row)
                print(
                    "RESULT", layer, frac, fold,
                    row["baseline"]["mean_delta_nll"],
                    row["methods"]["failure_conditioned"]["mean_delta_nll"],
                    row["methods"]["fisher_all"]["mean_delta_nll"],
                    row["gated_failure_rank2"]["mean_delta_nll"],
                    row["eig_geom"], row["gate_open"],
                    flush=True,
                )

    aggregate = {}
    for frac in PRUNE_FRACS:
        subset = [r for r in rows if r["prune_fraction_requested"] == frac]
        a = {
            "baseline_mean_delta_nll": float(np.mean([
                r["baseline"]["mean_delta_nll"] for r in subset
            ])),
        }
        for method in ["failure_conditioned", "fisher_all", "gated_failure_rank2"]:
            vals = []
            recs = []
            bpws = []
            for r in subset:
                x = r[method] if method == "gated_failure_rank2" else r["methods"][method]
                vals.append(x["mean_delta_nll"])
                recs.append(recovery(r, x["mean_delta_nll"]))
                bpws.append(x["sign_payload_plus_residual_bpw"])
            a[method] = {
                "mean_delta_nll": float(np.mean(vals)),
                "mean_recovery_pct": float(np.nanmean(recs)),
                "mean_sign_payload_plus_residual_bpw": float(np.mean(bpws)),
            }
        a["gate_open_fraction"] = float(np.mean([r["gate_open"] for r in subset]))
        aggregate[str(frac)] = a

    payload = {
        "model": MODEL_ID,
        "settings": {
            "layers": LAYERS,
            "prune_fractions": PRUNE_FRACS,
            "folds": FOLDS,
            "eig_gate": EIG_GATE,
            "mask": "implicit fixed PRNG; seed derived from layer/rate/fold",
            "bpw_note": "reported payload excludes the pre-existing packed Bonsai scale/metadata cost",
        },
        "rows": rows,
        "aggregate": aggregate,
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Official 1-bit Bonsai: implicit thinning + functional residual",
        "",
        "The pruning mask is deterministic from a seed and requires no per-weight mask storage.",
        "Reported bpw is retained sign payload + FP16 low-rank factor payload; existing Bonsai scale/metadata is excluded.",
        "",
        "| prune | baseline delta NLL | FC recovery | Fisher recovery | gated FC recovery | gated payload bpw | gate open |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for frac in PRUNE_FRACS:
        a = aggregate[str(frac)]
        lines.append(
            f"| {frac:.0%} | {a['baseline_mean_delta_nll']:.6f} | "
            f"{a['failure_conditioned']['mean_recovery_pct']:.2f}% | "
            f"{a['fisher_all']['mean_recovery_pct']:.2f}% | "
            f"{a['gated_failure_rank2']['mean_recovery_pct']:.2f}% | "
            f"{a['gated_failure_rank2']['mean_sign_payload_plus_residual_bpw']:.4f} | "
            f"{a['gate_open_fraction']:.2f} |"
        )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
