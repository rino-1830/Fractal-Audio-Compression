import json
import math
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "prism-ml/Bonsai-1.7B-unpacked"
MODULES = ["gate_proj", "up_proj", "down_proj"]
LEVELS = [0.125, 0.25, 0.375, 0.50]
TARGETS = [0.125, 0.25, 0.375]
CAL_N = 4
HOLD_N = 16
MAX_LENGTH = 20
GROUP_SIZE = 128
OUT = Path("results-kfac-mlp-waterfill")


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


def enc(tok, text):
    return tok(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)


@torch.no_grad()
def losses(model, tok, texts):
    vals = []
    for text in texts:
        x = enc(tok, text)
        vals.append(float(model(**x, labels=x["input_ids"]).loss))
    return np.asarray(vals, dtype=np.float64)


def metric(fp, vals):
    d = vals - fp
    return {
        "mean_delta_nll": float(d.mean()),
        "mean_positive_delta_nll": float(np.maximum(d, 0.0).mean()),
        "median_delta_nll": float(np.median(d)),
        "p90_delta_nll": float(np.quantile(d, 0.9)),
        "max_delta_nll": float(d.max()),
        "fraction_worse": float(np.mean(d > 0)),
        "per_sample_delta": d.tolist(),
    }


def perturb_factor(p):
    # E[(m/sqrt(q)-1)^2] for m~Bernoulli(q), q=1-p.
    q = 1.0 - p
    return 2.0 * (1.0 - math.sqrt(q))


def make_thinned(orig, layer, module_name, p):
    if p <= 0:
        return orig
    names = MODULES
    gen = torch.Generator(device="cpu")
    gen.manual_seed(42424242 + layer * 1009 + names.index(module_name) * 100003)
    x = orig.detach().clone()
    mask = torch.rand(x.shape, generator=gen) < p
    x[mask] = 0
    x = x.float() / math.sqrt(1.0 - p)
    return x.to(orig.dtype)


def greedy_allocate(scores, target):
    # Weighted bit-allocation over all MLP matrices.
    keys = sorted(scores, key=lambda z: (z[0], z[1]))
    total_weights = sum(scores[k]["weights"] for k in keys)
    target_removed = target * total_weights
    state = {k: 0 for k in keys}  # index into [0]+LEVELS
    levels = [0.0] + LEVELS
    removed = 0.0
    trace = []

    while removed + 1e-9 < target_removed:
        options = []
        for k in keys:
            idx = state[k]
            if idx >= len(levels) - 1:
                continue
            p0 = levels[idx]
            p1 = levels[idx + 1]
            w = scores[k]["weights"]
            bits_saved = (p1 - p0) * w
            c0 = perturb_factor(p0) * scores[k]["kfac_energy"]
            c1 = perturb_factor(p1) * scores[k]["kfac_energy"]
            marginal = max(c1 - c0, 0.0)
            cost_per_bit = marginal / max(bits_saved, 1.0)
            options.append((cost_per_bit, marginal, -bits_saved, k, idx + 1))
        if not options:
            break
        _, marginal, neg_bits, k, new_idx = min(options)
        p0 = levels[state[k]]
        p1 = levels[new_idx]
        delta_bits = (p1 - p0) * scores[k]["weights"]
        state[k] = new_idx
        removed += delta_bits
        trace.append({
            "layer": k[0],
            "module": k[1],
            "new_rate": p1,
            "marginal_predicted_cost": marginal,
            "bits_saved": delta_bits,
        })

    rates = {k: levels[state[k]] for k in keys}
    actual = sum(rates[k] * scores[k]["weights"] for k in keys) / total_weights
    return rates, actual, trace


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)

    cal = collect("validation", 600, CAL_N)
    hold = collect("train", 12000, HOLD_N)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    targets = {}
    for li, block in enumerate(model.model.layers):
        for name in MODULES:
            targets[(li, name)] = getattr(block.mlp, name)

    accum = {}
    handles = []

    # Make the hidden-state graph differentiable without storing parameter grads.
    def embed_hook(module, inp, out):
        out.requires_grad_(True)
        return out

    handles.append(model.model.embed_tokens.register_forward_hook(embed_hook))

    for key, module in targets.items():
        in_dim = module.weight.shape[1]
        out_dim = module.weight.shape[0]
        accum[key] = {
            "a_sum": torch.zeros(in_dim, dtype=torch.float64),
            "g_sum": torch.zeros(out_dim, dtype=torch.float64),
            "a_count": 0,
            "g_count": 0,
        }

        def make_hook(k):
            def hook(mod, inp, out):
                x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                acc = accum[k]
                acc["a_sum"] += x.square().sum(dim=0).double().cpu()
                acc["a_count"] += x.shape[0]

                if out.requires_grad:
                    def grad_hook(grad):
                        g = grad.detach().float().reshape(-1, grad.shape[-1])
                        acc["g_sum"] += g.square().sum(dim=0).double().cpu()
                        acc["g_count"] += g.shape[0]
                    out.register_hook(grad_hook)
            return hook
        handles.append(module.register_forward_hook(make_hook(key)))

    for i, text in enumerate(cal):
        x = enc(tok, text)
        out = model(**x, labels=x["input_ids"])
        out.loss.backward()
        print("CAL", i, float(out.loss), flush=True)

    for h in handles:
        h.remove()

    scores = {}
    for key, module in targets.items():
        acc = accum[key]
        A = acc["a_sum"] / max(acc["a_count"], 1)
        G = acc["g_sum"] / max(acc["g_count"], 1)
        W2 = module.weight.detach().float().square().double()
        # G^T (W^2 A), diagonal-KFAC / Fisher-like functional energy.
        energy = float(torch.dot(G, torch.mv(W2, A)))
        scores[key] = {
            "kfac_energy": energy,
            "energy_per_weight": energy / module.weight.numel(),
            "weights": int(module.weight.numel()),
            "a_count": acc["a_count"],
            "g_count": acc["g_count"],
        }
        print("SCORE", key[0], key[1], scores[key]["energy_per_weight"], flush=True)

    # Drop accumulators before cloning model weights.
    del accum
    originals = {k: m.weight.detach().clone() for k, m in targets.items()}

    with torch.no_grad():
        for k, m in targets.items():
            m.weight.copy_(originals[k])
    hold_fp = losses(model, tok, hold)

    def restore():
        with torch.no_grad():
            for k, m in targets.items():
                m.weight.copy_(originals[k])

    def apply_rates(rates):
        restore()
        with torch.no_grad():
            for k, p in rates.items():
                if p <= 0:
                    continue
                li, name = k
                targets[k].weight.copy_(make_thinned(originals[k], li, name, p))

    total_model_weights = sum(p.numel() for p in model.parameters())
    mlp_weights = sum(w.numel() for w in originals.values())
    mlp_fraction = mlp_weights / total_model_weights

    results = {}
    for target in TARGETS:
        rates, actual, trace = greedy_allocate(scores, target)
        apply_rates(rates)
        kfac_m = metric(hold_fp, losses(model, tok, hold))

        # Uniform is exactly representable for these target values.
        uniform_rates = {k: target for k in scores}
        apply_rates(uniform_rates)
        uniform_m = metric(hold_fp, losses(model, tok, hold))

        hist = {}
        for p in [0.0] + LEVELS:
            hist[str(p)] = sum(1 for v in rates.values() if abs(v - p) < 1e-12)

        results[str(target)] = {
            "actual_weighted_prune": actual,
            "rate_histogram": hist,
            "kfac_waterfill": kfac_m,
            "uniform": uniform_m,
            "mlp_linear_bpw": 1.0 - actual + 16.0 / GROUP_SIZE,
            "global_sign_bpw_saving": actual * mlp_fraction,
            "trace": trace,
        }
        print(
            "RESULT", target, actual,
            kfac_m["mean_positive_delta_nll"],
            uniform_m["mean_positive_delta_nll"],
            results[str(target)]["global_sign_bpw_saving"],
            flush=True
        )

    restore()

    serial_scores = {
        f"{k[0]}:{k[1]}": v for k, v in scores.items()
    }
    payload = {
        "model": MODEL_ID,
        "modules": MODULES,
        "levels": LEVELS,
        "targets": TARGETS,
        "calibration_n": CAL_N,
        "holdout_n": HOLD_N,
        "mlp_weight_fraction": mlp_fraction,
        "scores": serial_scores,
        "results": results,
    }
    (OUT / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
