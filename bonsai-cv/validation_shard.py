import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import onebit_validation_gate as vg

LAYER = int(os.environ["CASE_LAYER"])
FRAC = float(os.environ["CASE_FRAC"])
FOLD = int(os.environ["CASE_FOLD"])
OUT = Path("results-validation-shard")


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)

    xs = vg.load_text_pool()
    span = vg.BASIS_N + vg.GATE_N + vg.HOLD_N
    base = FOLD * span
    basis = xs[base:base + vg.BASIS_N]
    gate = xs[base + vg.BASIS_N:base + vg.BASIS_N + vg.GATE_N]
    hold = xs[base + vg.BASIS_N + vg.GATE_N:base + span]

    tok = AutoTokenizer.from_pretrained(vg.MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        vg.MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    row = vg.run_one(model, tok, basis, gate, hold, LAYER, FRAC, FOLD)

    b = row["baseline_holdout"]
    fc = row["methods"]["failure_conditioned"]["holdout"]
    gated = row["validation_gated_holdout"]
    payload = {
        "case": {"layer": LAYER, "frac": FRAC, "fold": FOLD},
        "row": row,
        "summary": {
            "baseline_mean_delta_nll": b["mean_delta_nll"],
            "baseline_positive_harm": b["mean_positive_delta_nll"],
            "always_fc_mean_delta_nll": fc["mean_delta_nll"],
            "always_fc_positive_harm": fc["mean_positive_delta_nll"],
            "gate_open": row["validation_gate_open"],
            "gated_mean_delta_nll": gated["mean_delta_nll"],
            "gated_positive_harm": gated["mean_positive_delta_nll"],
            "selected_storage_bpw": row["selected_storage_bpw"],
        },
    }

    stem = f"l{LAYER}_p{int(FRAC*100)}_f{FOLD}"
    (OUT / f"{stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
