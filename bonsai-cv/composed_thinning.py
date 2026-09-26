import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "prism-ml/Bonsai-1.7B-unpacked"
LAYERS = [0, 7, 14, 21, 27]
# Chosen previously on calibration data only, using positive-harm ranking.
SELECTED_CODES = {0: 4, 7: 3, 14: 6, 21: 1, 27: 3}
FRAC = 0.50
N = 32
MAX_LENGTH = 20
SKIP_ELIGIBLE = 900
OUT = Path("results-composed-thinning")


def load_texts():
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    xs = []
    skipped = 0
    for row in ds:
        t = " ".join(row["text"].split())
        if len(t) < 100 or t.startswith("="):
            continue
        if skipped < SKIP_ELIGIBLE:
            skipped += 1
            continue
        xs.append(t[:500])
        if len(xs) >= N:
            break
    if len(xs) < N:
        raise RuntimeError("not enough text")
    return xs


def encode(tok, text):
    return tok(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)


@torch.no_grad()
def losses(model, tok, texts):
    vals = []
    for text in texts:
        x = encode(tok, text)
        vals.append(float(model(**x, labels=x["input_ids"]).loss))
    return np.asarray(vals, dtype=np.float64)


def mask_for(weight, layer, code):
    x = weight.detach().clone()
    g = torch.Generator(device="cpu")
    g.manual_seed(246813579 + code * 1000003 + layer * 1009)
    mask = torch.rand(x.shape, generator=g) < FRAC
    x[mask] = 0
    return x


def metrics(base, vals):
    d = vals - base
    pos = np.maximum(d, 0.0)
    return {
        "mean_delta_nll": float(d.mean()),
        "mean_positive_delta_nll": float(pos.mean()),
        "median_delta_nll": float(np.median(d)),
        "p90_delta_nll": float(np.quantile(d, 0.9)),
        "max_delta_nll": float(d.max()),
        "fraction_worse": float(np.mean(d > 0)),
        "per_sample_delta": d.tolist(),
    }


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    texts = load_texts()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    targets = {}
    originals = {}
    for layer in LAYERS:
        t = model.model.layers[layer].mlp.down_proj.weight
        targets[layer] = t
        originals[layer] = t.detach().clone()

    base = losses(model, tok, texts)

    def restore_all():
        with torch.no_grad():
            for layer in LAYERS:
                targets[layer].copy_(originals[layer])

    # Each selected layer individually on the same untouched evaluation data.
    individual = {}
    for layer in LAYERS:
        restore_all()
        with torch.no_grad():
            targets[layer].copy_(mask_for(originals[layer], layer, SELECTED_CODES[layer]))
        vals = losses(model, tok, texts)
        individual[str(layer)] = metrics(base, vals)
        print("IND", layer, individual[str(layer)]["mean_positive_delta_nll"], flush=True)

    # Five fixed-code0 masks simultaneously.
    restore_all()
    with torch.no_grad():
        for layer in LAYERS:
            targets[layer].copy_(mask_for(originals[layer], layer, 0))
    fixed_vals = losses(model, tok, texts)
    fixed = metrics(base, fixed_vals)

    # Five previously calibration-selected masks simultaneously.
    restore_all()
    with torch.no_grad():
        for layer in LAYERS:
            targets[layer].copy_(mask_for(originals[layer], layer, SELECTED_CODES[layer]))
    selected_vals = losses(model, tok, texts)
    selected = metrics(base, selected_vals)

    restore_all()

    sum_individual_mean = float(sum(x["mean_delta_nll"] for x in individual.values()))
    sum_individual_positive = float(sum(x["mean_positive_delta_nll"] for x in individual.values()))

    payload = {
        "model": MODEL_ID,
        "layers": LAYERS,
        "selected_codes": SELECTED_CODES,
        "prune_fraction_per_tested_matrix": FRAC,
        "evaluation_n": N,
        "data": "WikiText-2 test after eligible offset 900",
        "individual_selected": individual,
        "fixed_code0_composed": fixed,
        "selected_composed": selected,
        "sum_individual_mean_delta_nll": sum_individual_mean,
        "sum_individual_positive_harm": sum_individual_positive,
        "composition_ratio_mean": (
            selected["mean_delta_nll"] / sum_individual_mean
            if abs(sum_individual_mean) > 1e-12 else None
        ),
        "composition_ratio_positive": (
            selected["mean_positive_delta_nll"] / sum_individual_positive
            if sum_individual_positive > 1e-12 else None
        ),
    }

    (OUT / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "fixed_code0_composed": fixed,
        "selected_composed": selected,
        "sum_individual_mean_delta_nll": sum_individual_mean,
        "sum_individual_positive_harm": sum_individual_positive,
        "composition_ratio_mean": payload["composition_ratio_mean"],
        "composition_ratio_positive": payload["composition_ratio_positive"],
    }, indent=2))


if __name__ == "__main__":
    main()
