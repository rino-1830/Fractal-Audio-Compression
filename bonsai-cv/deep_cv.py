import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1] / "llm-functional-compression"))
import bonsai_probe as bp

LAYERS = [21, 27]
FOLDS = 3
CAL_N = 8
HOLD_N = 8
FAIL_N = 3
STABLE_N = 3
MAX_LENGTH = 20
OUT = Path("results-bonsai-deep-cv")


def load_text_pool():
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
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


def sample_grad(model, tok, target, text):
    model.zero_grad(set_to_none=True)
    x = encode(tok, text)
    loss = model(**x, labels=x["input_ids"]).loss
    g = torch.autograd.grad(loss, target, retain_graph=False, create_graph=False)[0]
    g = g.detach().float().reshape(-1)
    return g / g.norm().clamp_min(1e-12)


def random_basis(p, k, seed):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return bp.orth(torch.randn(p, k, generator=gen)) if hasattr(bp, "orth") else bp.orthonormalize(torch.randn(p, k, generator=gen))


def stats(fp, vals):
    d = vals - fp
    return {
        "mean_delta_nll": float(d.mean()),
        "median_delta_nll": float(np.median(d)),
        "max_delta_nll": float(d.max()),
    }


def run_one(model, tok, calib, holdout, layer, fold):
    target = model.model.layers[layer].mlp.down_proj.weight
    target.requires_grad_(True)
    original = target.detach().clone()
    binary = bp.deterministic_binary_from_ternary(original)

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

    G = torch.stack([sample_grad(model, tok, target, calib[int(i)]) for i in list(fail_idx) + list(stable_idx)])
    uf, eigs = bp.failure_basis(G, FAIL_N, bp.K)
    ui = bp.fisher_basis(G, bp.K)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(9000 + layer * 10 + fold)
    ur = bp.orth(torch.randn(G.shape[1], bp.K, generator=gen)) if hasattr(bp, "orth") else bp.orthonormalize(torch.randn(G.shape[1], bp.K, generator=gen))

    residual = (binary.float() - original.float()).reshape(-1)
    m, n = original.shape

    out = {
        "layer": layer,
        "fold": fold,
        "baseline": stats(hold_fp, hold_bin),
        "eigs": eigs,
        "methods": {},
    }

    for name, U in {
        "failure_conditioned": uf,
        "fisher_all": ui,
        "random": ur,
    }.items():
        corr = (U @ (U.T @ residual)).reshape_as(original)
        lr = bp.low_rank(corr, 2)
        candidate = binary.float() - lr
        with torch.no_grad():
            target.copy_(candidate.to(target.dtype))
        vals = eval_texts(model, tok, holdout)
        out["methods"][name] = {
            **stats(hold_fp, vals),
            "correction_energy_fraction": float(corr.pow(2).sum() / residual.pow(2).sum().clamp_min(1e-12)),
            "bpw": float(1.0 + 16.0 / bp.GROUP_SIZE + 16.0 * 2 * (m + n) / (m * n)),
        }

    with torch.no_grad():
        target.copy_(original)
    target.requires_grad_(False)
    return out


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)

    xs = load_text_pool()
    tok = AutoTokenizer.from_pretrained(bp.MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        bp.MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    rows = []
    for layer in LAYERS:
        for fold in range(FOLDS):
            base = fold * (CAL_N + HOLD_N)
            calib = xs[base:base + CAL_N]
            holdout = xs[base + CAL_N:base + CAL_N + HOLD_N]
            print("START", layer, fold, flush=True)
            row = run_one(model, tok, calib, holdout, layer, fold)
            rows.append(row)
            b = row["baseline"]["mean_delta_nll"]
            f = row["methods"]["failure_conditioned"]["mean_delta_nll"]
            i = row["methods"]["fisher_all"]["mean_delta_nll"]
            r = row["methods"]["random"]["mean_delta_nll"]
            print("RESULT", layer, fold, b, f, i, r, flush=True)

    aggregate = {}
    for layer in LAYERS:
        subset = [x for x in rows if x["layer"] == layer]
        base = np.array([x["baseline"]["mean_delta_nll"] for x in subset])
        aggregate[str(layer)] = {"baseline_mean_delta_nll": float(base.mean())}
        for name in ["failure_conditioned", "fisher_all", "random"]:
            vals = np.array([x["methods"][name]["mean_delta_nll"] for x in subset])
            rec = (base - vals) / np.where(np.abs(base) > 1e-12, base, np.nan) * 100.0
            aggregate[str(layer)][name] = {
                "mean_delta_nll": float(vals.mean()),
                "mean_recovery_pct": float(np.nanmean(rec)),
                "fold_recovery_pct": rec.tolist(),
            }

    payload = {
        "settings": {
            "layers": LAYERS,
            "folds": FOLDS,
            "calibration_n": CAL_N,
            "holdout_n": HOLD_N,
            "max_length": MAX_LENGTH,
        },
        "rows": rows,
        "aggregate": aggregate,
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["# Bonsai deep-layer cross-validation", ""]
    for layer in LAYERS:
        a = aggregate[str(layer)]
        lines.append(f"## layer {layer}")
        lines.append(f"baseline mean delta NLL: {a['baseline_mean_delta_nll']:.6f}")
        for name in ["failure_conditioned", "fisher_all", "random"]:
            x = a[name]
            lines.append(
                f"- {name}: mean delta NLL {x['mean_delta_nll']:.6f}; "
                f"mean recovery {x['mean_recovery_pct']:.3f}%; "
                f"folds {x['fold_recovery_pct']}"
            )
        lines.append("")

    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
