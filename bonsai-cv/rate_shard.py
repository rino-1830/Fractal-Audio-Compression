import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import rate_sweep as rs

LAYER = int(os.environ["CASE_LAYER"])
GROUP = int(os.environ["CASE_GROUP"])
FOLD = int(os.environ["CASE_FOLD"])
OUT = Path("results-rate-shard")


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)

    xs = rs.load_text_pool()
    base = FOLD * (rs.CAL_N + rs.HOLD_N)
    calib = xs[base:base + rs.CAL_N]
    holdout = xs[base + rs.CAL_N:base + rs.CAL_N + rs.HOLD_N]

    tok = AutoTokenizer.from_pretrained(rs.MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        rs.MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    row = rs.run_one(model, tok, calib, holdout, LAYER, GROUP, FOLD)
    stem = f"l{LAYER}_g{GROUP}_f{FOLD}"
    payload = {
        "case": {"layer": LAYER, "group": GROUP, "fold": FOLD},
        "row": row,
    }
    (OUT / f"{stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "baseline": row["baseline"]["mean_delta_nll"],
        "fc_rank1": row["methods"]["failure_conditioned"]["ranks"]["1"]["mean_delta_nll"],
        "fc_rank2": row["methods"]["failure_conditioned"]["ranks"]["2"]["mean_delta_nll"],
        "fisher_rank2": row["methods"]["fisher_all"]["ranks"]["2"]["mean_delta_nll"],
        "gated_rank2": row["gated_failure_rank2"]["mean_delta_nll"],
        "gate_open": row["gate_open"],
        "eig_geom": row["eig_geom"],
        "rank1_bpw": row["methods"]["failure_conditioned"]["ranks"]["1"]["nominal_bpw"],
        "rank2_bpw": row["methods"]["failure_conditioned"]["ranks"]["2"]["nominal_bpw"],
    }, indent=2))


if __name__ == "__main__":
    main()
