import json
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
BASIS_N = 8
GATE_N = 4
HOLD_N = 8
FAIL_N = 3
STABLE_N = 3
K = 2
MAX_LENGTH = 20
SKIP_ELIGIBLE = 100
GROUP_SIZE = 128
OUT = Path("results-onebit-validation-gate")


def load_text_pool():
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    xs = []
    skipped = 0
    need = FOLDS * (BASIS_N + GATE_N + HOLD_N)
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
    if len(xs) < need:
        raise RuntimeError("not enough text")
    return xs


def encode(tok, text):
    return tok(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)


@torch.no_grad()
def losses(model, tok, texts):
    out = []
    for text in texts:
        x = encode(tok, text)
        out.append(float(model(**x, labels=x["input_ids"]).loss))
    return np.array(out, dtype=np.float64)


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


def metrics(fp, vals):
    d = vals - fp
    return {
        "mean_delta_nll": float(d.mean()),
        "mean_positive_delta_nll": float(np.maximum(d, 0.0).mean()),
        "median_delta_nll": float(np.median(d)),
        "max_delta_nll": float(d.max()),
        "fraction_worse": float(np.mean(d > 0)),
        "per_sample_delta": d.tolist(),
    }


def eval_candidate(target, candidate, model, tok, fp, texts):
    with torch.no_grad():
        target.copy_(candidate.to(target.dtype))
    vals = losses(model, tok, texts)
    return vals, metrics(fp, vals)


def run_one(model, tok, basis_texts, gate_texts, hold_texts, layer, frac, fold):
    target = model.model.layers[layer].mlp.down_proj.weight
    target.requires_grad_(True)
    original = target.detach().clone()
    seed = 900000 + layer * 1000 + int(frac * 100) * 10 + fold
    pruned, actual_frac = implicit_prune(original, frac, seed)

    with torch.no_grad():
        target.copy_(original)
    basis_fp = losses(model, tok, basis_texts)
    gate_fp = losses(model, tok, gate_texts)
    hold_fp = losses(model, tok, hold_texts)

    with torch.no_grad():
        target.copy_(pruned)
    basis_p = losses(model, tok, basis_texts)
    gate_p = losses(model, tok, gate_texts)
    hold_p = losses(model, tok, hold_texts)

    basis_delta = basis_p - basis_fp
    fail_idx = np.argsort(basis_delta)[-FAIL_N:][::-1]
    stable_idx = np.argsort(np.abs(basis_delta))[:STABLE_N]

    with torch.no_grad():
        target.copy_(original)

    G = torch.stack([
        sample_grad(model, tok, target, basis_texts[int(i)])
        for i in list(fail_idx) + list(stable_idx)
    ])
    uf, eigs = failure_basis(G, FAIL_N, K)
    ui = fisher_basis(G, K)

    residual = (pruned.float() - original.float()).reshape(-1)
    m, n = original.shape
    factor_bpw = 16.0 * 2 * (m + n) / (m * n)
    base_storage_bpw = (1.0 - frac) + 16.0 / GROUP_SIZE
    corrected_storage_bpw = base_storage_bpw + factor_bpw

    candidates = {}
    for name, U in {"failure_conditioned": uf, "fisher_all": ui}.items():
        corr = (U @ (U.T @ residual)).reshape_as(original)
        lr = low_rank(corr, 2)
        candidate = pruned.float() - lr

        gate_vals, gate_m = eval_candidate(target, candidate, model, tok, gate_fp, gate_texts)
        hold_vals, hold_m = eval_candidate(target, candidate, model, tok, hold_fp, hold_texts)
        candidates[name] = {
            "gate": gate_m,
            "holdout": hold_m,
            "correction_energy_fraction": float(
                corr.pow(2).sum() / residual.pow(2).sum().clamp_min(1e-12)
            ),
            "storage_bpw": float(corrected_storage_bpw),
            "_candidate": candidate,
        }

    baseline_gate = metrics(gate_fp, gate_p)
    baseline_hold = metrics(hold_fp, hold_p)

    # Gate uses only the separate gate set.
    fc_gate_improves = (
        candidates["failure_conditioned"]["gate"]["mean_positive_delta_nll"]
        < baseline_gate["mean_positive_delta_nll"]
        and candidates["failure_conditioned"]["gate"]["mean_delta_nll"]
        <= baseline_gate["mean_delta_nll"]
    )

    if fc_gate_improves:
        selected_hold = candidates["failure_conditioned"]["holdout"].copy()
        selected_storage = corrected_storage_bpw
    else:
        selected_hold = baseline_hold.copy()
        selected_storage = base_storage_bpw

    # Drop in-memory tensors before serialization.
    for x in candidates.values():
        x.pop("_candidate", None)

    with torch.no_grad():
        target.copy_(original)
    target.requires_grad_(False)

    return {
        "layer": layer,
        "prune_fraction_requested": frac,
        "prune_fraction_actual": actual_frac,
        "fold": fold,
        "eigs": eigs,
        "baseline_gate": baseline_gate,
        "baseline_holdout": baseline_hold,
        "methods": candidates,
        "validation_gate_open": bool(fc_gate_improves),
        "validation_gated_holdout": selected_hold,
        "base_storage_bpw": float(base_storage_bpw),
        "corrected_storage_bpw": float(corrected_storage_bpw),
        "selected_storage_bpw": float(selected_storage),
    }


def aggregate_metrics(rows, getter):
    ds = []
    ps = []
    for r in rows:
        m = getter(r)
        ds.append(m["mean_delta_nll"])
        ps.append(m["mean_positive_delta_nll"])
    return {
        "mean_delta_nll": float(np.mean(ds)),
        "mean_positive_delta_nll": float(np.mean(ps)),
    }


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
    span = BASIS_N + GATE_N + HOLD_N
    for layer in LAYERS:
        for frac in PRUNE_FRACS:
            for fold in range(FOLDS):
                base = fold * span
                basis = xs[base:base + BASIS_N]
                gate = xs[base + BASIS_N:base + BASIS_N + GATE_N]
                hold = xs[base + BASIS_N + GATE_N:base + span]
                print("START", layer, frac, fold, flush=True)
                row = run_one(model, tok, basis, gate, hold, layer, frac, fold)
                rows.append(row)
                print(
                    "RESULT", layer, frac, fold,
                    row["baseline_holdout"]["mean_delta_nll"],
                    row["methods"]["failure_conditioned"]["holdout"]["mean_delta_nll"],
                    row["validation_gated_holdout"]["mean_delta_nll"],
                    row["validation_gate_open"],
                    flush=True,
                )

    aggregate = {}
    for frac in PRUNE_FRACS:
        subset = [r for r in rows if r["prune_fraction_requested"] == frac]
        aggregate[str(frac)] = {
            "baseline": aggregate_metrics(subset, lambda r: r["baseline_holdout"]),
            "always_failure": aggregate_metrics(
                subset, lambda r: r["methods"]["failure_conditioned"]["holdout"]
            ),
            "always_fisher": aggregate_metrics(
                subset, lambda r: r["methods"]["fisher_all"]["holdout"]
            ),
            "validation_gated": aggregate_metrics(
                subset, lambda r: r["validation_gated_holdout"]
            ),
            "gate_open_fraction": float(np.mean([r["validation_gate_open"] for r in subset])),
            "mean_selected_storage_bpw": float(np.mean([r["selected_storage_bpw"] for r in subset])),
            "base_storage_bpw": float(subset[0]["base_storage_bpw"]),
            "corrected_storage_bpw": float(subset[0]["corrected_storage_bpw"]),
        }

    payload = {
        "model": MODEL_ID,
        "settings": {
            "layers": LAYERS,
            "prune_fractions": PRUNE_FRACS,
            "folds": FOLDS,
            "basis_n": BASIS_N,
            "gate_n": GATE_N,
            "holdout_n": HOLD_N,
            "dataset": "WikiText-2 test, eligible examples after first 100",
            "storage_assumption": "official g128: 1 sign bit per retained weight + FP16 scale per 128 original weights + optional FP16 rank-2 factors",
        },
        "rows": rows,
        "aggregate": aggregate,
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# 1-bit Bonsai thinning with independent validation gate",
        "",
        "Basis construction, gate decision, and final holdout are disjoint.",
        "Mean-positive delta NLL measures only degradation and avoids cancellation by accidental improvements.",
        "",
        "| prune | baseline positive harm | always-FC positive harm | gated-FC positive harm | gate open | mean selected bpw |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for frac in PRUNE_FRACS:
        a = aggregate[str(frac)]
        lines.append(
            f"| {frac:.0%} | {a['baseline']['mean_positive_delta_nll']:.6f} | "
            f"{a['always_failure']['mean_positive_delta_nll']:.6f} | "
            f"{a['validation_gated']['mean_positive_delta_nll']:.6f} | "
            f"{a['gate_open_fraction']:.2f} | {a['mean_selected_storage_bpw']:.4f} |"
        )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
