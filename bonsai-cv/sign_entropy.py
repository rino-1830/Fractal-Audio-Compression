import json
import math
import zlib
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

MODEL_ID = "prism-ml/Bonsai-1.7B-unpacked"
OUT = Path("results-bonsai-sign-entropy")


def h2(p):
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def entropy_from_counts(counts):
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def analyze_bits(bits):
    bits = np.asarray(bits, dtype=np.uint8)
    m, n = bits.shape
    total = bits.size
    p = float(bits.mean())
    iid_bpw = h2(p)

    if n > 1:
        xor_h = np.bitwise_xor(bits[:, 1:], bits[:, :-1])
        ph = float(xor_h.mean())
        markov_h_bpw = h2(ph) + m / total
    else:
        ph = 0.0
        markov_h_bpw = 1.0

    if m > 1:
        xor_v = np.bitwise_xor(bits[1:, :], bits[:-1, :])
        pv = float(xor_v.mean())
        markov_v_bpw = h2(pv) + n / total
    else:
        pv = 0.0
        markov_v_bpw = 1.0

    me = (m // 2) * 2
    ne = (n // 2) * 2
    if me and ne:
        b = bits[:me, :ne]
        codes = (
            b[0::2, 0::2]
            | (b[0::2, 1::2] << 1)
            | (b[1::2, 0::2] << 2)
            | (b[1::2, 1::2] << 3)
        )
        counts = np.bincount(codes.reshape(-1), minlength=16)
        block2_bpw = entropy_from_counts(counts) / 4.0
    else:
        block2_bpw = 1.0

    packed = np.packbits(bits.reshape(-1), bitorder="little").tobytes()
    comp = zlib.compress(packed, 9)
    zlib_bpw = 8.0 * len(comp) / total

    return {
        "weights": int(total),
        "positive_fraction": p,
        "iid_entropy_bpw": iid_bpw,
        "horizontal_transition_fraction": ph,
        "horizontal_markov_est_bpw": markov_h_bpw,
        "vertical_transition_fraction": pv,
        "vertical_markov_est_bpw": markov_v_bpw,
        "block2x2_entropy_bpw": block2_bpw,
        "zlib_bpw": zlib_bpw,
    }


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    rows = []
    for name, p in model.named_parameters():
        if p.ndim != 2 or p.numel() < 65536:
            continue
        arr = p.detach().cpu().numpy()
        bits = (arr > 0).astype(np.uint8, copy=False)
        s = analyze_bits(bits)
        s["name"] = name
        s["shape"] = list(arr.shape)
        rows.append(s)
        print(name, s["weights"], s["iid_entropy_bpw"], s["horizontal_markov_est_bpw"], s["block2x2_entropy_bpw"], s["zlib_bpw"], flush=True)

    total = sum(r["weights"] for r in rows)
    def weighted(key):
        return sum(r["weights"] * r[key] for r in rows) / total

    aggregate = {
        "analyzed_weights": total,
        "num_matrices": len(rows),
        "iid_entropy_bpw": weighted("iid_entropy_bpw"),
        "horizontal_markov_est_bpw": weighted("horizontal_markov_est_bpw"),
        "vertical_markov_est_bpw": weighted("vertical_markov_est_bpw"),
        "block2x2_entropy_bpw": weighted("block2x2_entropy_bpw"),
        "zlib_bpw": weighted("zlib_bpw"),
    }

    best = sorted(rows, key=lambda r: r["block2x2_entropy_bpw"])[:10]
    worst = sorted(rows, key=lambda r: r["block2x2_entropy_bpw"], reverse=True)[:10]

    payload = {"model": MODEL_ID, "aggregate": aggregate, "best_block2x2": best, "worst_block2x2": worst, "rows": rows}
    (OUT / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# 1-bit Bonsai sign-structure audit",
        "",
        f"Analyzed {aggregate['analyzed_weights']:,} weights across {aggregate['num_matrices']} large 2-D matrices.",
        "",
        "| coding model | estimated / measured bits per sign weight |",
        "|---|---:|",
        f"| IID entropy lower bound from sign bias | {aggregate['iid_entropy_bpw']:.6f} |",
        f"| horizontal first-order Markov estimate | {aggregate['horizontal_markov_est_bpw']:.6f} |",
        f"| vertical first-order Markov estimate | {aggregate['vertical_markov_est_bpw']:.6f} |",
        f"| empirical 2x2 block entropy | {aggregate['block2x2_entropy_bpw']:.6f} |",
        f"| zlib on bit-packed signs | {aggregate['zlib_bpw']:.6f} |",
        "",
        "Values substantially below 1.0 would indicate exploitable lossless structure in the binary signs. Values near 1.0 imply that further compression must be lossy/functional rather than ordinary entropy coding.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
