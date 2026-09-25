#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def get_target_layer(model):
    # Pythia / GPT-NeoX: down-projection in a middle MLP block.
    layers = model.gpt_neox.layers
    idx = len(layers) // 2
    return layers[idx].mlp.dense_4h_to_h, idx


def make_blocks(tokenizer, seq_len: int, total_blocks: int, seed: int):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n".join(x["text"] for x in ds if x["text"].strip())
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    # Sample blocks across the validation corpus rather than taking one contiguous region.
    starts = list(range(256, max(257, ids.numel() - seq_len), seq_len))
    if len(starts) < total_blocks:
        raise RuntimeError(f"Not enough candidate blocks: have {len(starts)}, need {total_blocks}")
    rng = random.Random(seed)
    chosen = rng.sample(starts, total_blocks)
    return [ids[s : s + seq_len].unsqueeze(0) for s in chosen]


@torch.no_grad()
def ternarize_mse(weight: torch.Tensor):
    w = weight.detach().float()
    a = w.abs().flatten()
    qs = torch.linspace(0.05, 0.85, 17)
    thresholds = torch.quantile(a, qs)
    best = None
    for qv, threshold in zip(qs.tolist(), thresholds.tolist()):
        mask = w.abs() > threshold
        if not mask.any():
            continue
        alpha = w.abs()[mask].mean()
        tern = torch.sign(w) * mask * alpha
        mse = torch.mean((w - tern) ** 2).item()
        if best is None or mse < best["mse"]:
            best = {
                "q": float(qv),
                "threshold": float(threshold),
                "alpha": float(alpha),
                "mse": float(mse),
                "weight": tern,
                "codes": (torch.sign(w) * mask).to(torch.int8),
            }
    if best is None:
        raise RuntimeError("Ternary search failed")
    return best


def symbol_entropy(codes: torch.Tensor) -> float:
    vals = []
    for s in (-1, 0, 1):
        p = (codes == s).float().mean().item()
        if p > 0:
            vals.append(-p * math.log2(p))
    return float(sum(vals))


@torch.no_grad()
def cache_teacher(model, target, original_weight, blocks):
    target.weight.copy_(original_weight)
    model.eval()
    out = []
    for x in blocks:
        logits = model(input_ids=x).logits[:, :-1, :].detach().float().cpu()
        out.append(logits)
    return out


def kl_to_teacher(student_logits, teacher_logits):
    t = teacher_logits.to(student_logits.device, dtype=torch.float32)
    s = student_logits.float()
    tp = torch.softmax(t, dim=-1)
    return F.kl_div(torch.log_softmax(s, dim=-1), tp, reduction="none").sum(-1).mean()


@torch.no_grad()
def evaluate(model, target, weight, blocks, teacher_logits):
    target.weight.copy_(weight)
    model.eval()
    total_kl = 0.0
    agree = 0
    count = 0
    nll_sum = 0.0
    tok_count = 0
    for x, tlog in zip(blocks, teacher_logits):
        logits_all = model(input_ids=x).logits
        logits = logits_all[:, :-1, :]
        kl = kl_to_teacher(logits, tlog)
        total_kl += float(kl)
        pred_s = logits.argmax(dim=-1).cpu()
        pred_t = tlog.argmax(dim=-1)
        agree += int((pred_s == pred_t).sum())
        count += pred_s.numel()
        labels = x[:, 1:]
        nll = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="sum",
        )
        nll_sum += float(nll)
        tok_count += labels.numel()
    return {
        "kl": total_kl / len(blocks),
        "top1_agreement": agree / max(1, count),
        "nll": nll_sum / max(1, tok_count),
        "ppl": math.exp(min(20.0, nll_sum / max(1, tok_count))),
    }


def calibration_scores(model, target, qweight, blocks, teacher_logits):
    scores = []
    with torch.no_grad():
        target.weight.copy_(qweight)
        model.eval()
        for i, (x, tlog) in enumerate(zip(blocks, teacher_logits)):
            logits = model(input_ids=x).logits[:, :-1, :]
            scores.append((i, float(kl_to_teacher(logits, tlog))))
    return scores


def collect_gradients(model, target, qweight, blocks, teacher_logits, indices):
    for p in model.parameters():
        p.requires_grad_(False)
    target.weight.requires_grad_(True)
    target.weight.data.copy_(qweight)
    model.eval()
    grads = []
    raw_norms = []
    losses = []
    for idx in indices:
        model.zero_grad(set_to_none=True)
        x = blocks[idx]
        logits = model(input_ids=x).logits[:, :-1, :]
        loss = kl_to_teacher(logits, teacher_logits[idx])
        loss.backward()
        g = target.weight.grad.detach().float().clone().reshape(-1)
        norm = torch.linalg.vector_norm(g)
        raw_norms.append(float(norm))
        losses.append(float(loss.detach()))
        g = g / (norm + 1e-12)
        grads.append(g.cpu())
    target.weight.requires_grad_(False)
    return torch.stack(grads, dim=0), raw_norms, losses


def span_projection(error: torch.Tensor, grads: torch.Tensor, ridge=1e-5):
    e = error.reshape(-1).cpu().float()
    G = grads.cpu().float()
    gram = G @ G.T
    rhs = G @ e
    scale = float(torch.trace(gram) / max(1, gram.shape[0]))
    reg = ridge * max(scale, 1e-8)
    coeff = torch.linalg.solve(gram + reg * torch.eye(gram.shape[0]), rhs)
    delta = (G.T @ coeff).reshape_as(error)
    energy = float((delta.square().sum() / (error.square().sum() + 1e-12)))
    return delta, energy


def spectrum_stats(grads: torch.Tensor):
    gram = grads @ grads.T
    eig = torch.linalg.eigvalsh(gram).clamp_min(0).flip(0)
    p = eig / (eig.sum() + 1e-12)
    entropy = -(p[p > 0] * torch.log(p[p > 0])).sum()
    eff_rank = float(torch.exp(entropy))
    cumulative = torch.cumsum(p, dim=0)
    return {
        "eigenvalues": [float(x) for x in eig],
        "effective_rank": eff_rank,
        "top1_energy": float(cumulative[min(0, len(cumulative)-1)]),
        "top2_energy": float(cumulative[min(1, len(cumulative)-1)]),
        "top4_energy": float(cumulative[min(3, len(cumulative)-1)]),
    }


def lowrank_factors(matrix: torch.Tensor, max_rank: int):
    q = min(max_rank, min(matrix.shape) - 1)
    if q < 1:
        raise RuntimeError("Matrix too small for low-rank factorization")
    U, S, V = torch.pca_lowrank(matrix.float(), q=q, center=False, niter=4)
    return U, S, V


def reconstruct(U, S, V, rank: int):
    r = min(rank, S.numel())
    return (U[:, :r] * S[:r]) @ V[:, :r].T


def random_lowrank_projection(error: torch.Tensor, rank: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    m, n = error.shape
    U = torch.randn(m, rank, generator=gen)
    V = torch.randn(n, rank, generator=gen)
    U = torch.linalg.qr(U, mode="reduced").Q
    V = torch.linalg.qr(V, mode="reduced").Q
    coeff = torch.diagonal(U.T @ error @ V)
    return (U * coeff) @ V.T


def residual_bpw(shape, rank: int, include_core=False):
    m, n = shape
    params = rank * (m + n) + (rank * rank if include_core else rank)
    return 16.0 * params / (m * n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--calib-blocks", type=int, default=24)
    ap.add_argument("--eval-blocks", type=int, default=12)
    ap.add_argument("--extreme-count", type=int, default=8)
    ap.add_argument("--ranks", default="1,2,4,8")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output-dir", default="results")
    args = ap.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to("cpu")
    model.eval()

    target, layer_idx = get_target_layer(model)
    original = target.weight.detach().clone().float()
    qinfo = ternarize_mse(original)
    qweight = qinfo.pop("weight")
    codes = qinfo.pop("codes")
    entropy_bpw = symbol_entropy(codes)

    total = args.calib_blocks + args.eval_blocks
    blocks = make_blocks(tokenizer, args.seq_len, total, args.seed)
    calib = blocks[: args.calib_blocks]
    eval_blocks = blocks[args.calib_blocks :]

    print(f"Model: {args.model}")
    print(f"Target layer: GPT-NeoX layer {layer_idx} dense_4h_to_h {tuple(original.shape)}")
    print(f"Ternary entropy: {entropy_bpw:.4f} bpw")
    print(f"Ternary MSE: {qinfo['mse']:.6e}, alpha={qinfo['alpha']:.6e}")

    teacher_calib = cache_teacher(model, target, original, calib)
    teacher_eval = cache_teacher(model, target, original, eval_blocks)

    scores = calibration_scores(model, target, qweight, calib, teacher_calib)
    ordered = sorted(scores, key=lambda x: x[1])
    k = min(args.extreme_count, len(ordered) // 2)
    success_idx = [i for i, _ in ordered[:k]]
    failure_idx = [i for i, _ in ordered[-k:]]

    failure_grads, failure_norms, failure_losses = collect_gradients(
        model, target, qweight, calib, teacher_calib, failure_idx
    )
    success_grads, success_norms, success_losses = collect_gradients(
        model, target, qweight, calib, teacher_calib, success_idx
    )

    error = original - qweight
    failure_delta, failure_proj_energy = span_projection(error, failure_grads)
    success_delta, success_proj_energy = span_projection(error, success_grads)

    ranks = sorted({int(x) for x in args.ranks.split(",") if x.strip()})
    max_rank = max(ranks)
    Uf, Sf, Vf = lowrank_factors(failure_delta, max_rank)
    Us, Ss, Vs = lowrank_factors(success_delta, max_rank)
    Ue, Se, Ve = lowrank_factors(error, max_rank)

    baseline_teacher = evaluate(model, target, original, eval_blocks, teacher_eval)
    baseline_quant = evaluate(model, target, qweight, eval_blocks, teacher_eval)

    rows = []
    methods = ["failure_span", "success_span", "weight_svd", "random"]
    for rank in ranks:
        candidates = {
            "failure_span": reconstruct(Uf, Sf, Vf, rank),
            "success_span": reconstruct(Us, Ss, Vs, rank),
            "weight_svd": reconstruct(Ue, Se, Ve, rank),
            "random": random_lowrank_projection(error, rank, args.seed + rank),
        }
        for method in methods:
            corr = candidates[method]
            metrics = evaluate(model, target, qweight + corr, eval_blocks, teacher_eval)
            qkl = baseline_quant["kl"]
            recovery = (qkl - metrics["kl"]) / qkl if qkl > 1e-12 else 0.0
            nll_denom = baseline_quant["nll"] - baseline_teacher["nll"]
            nll_recovery = (
                (baseline_quant["nll"] - metrics["nll"]) / nll_denom
                if abs(nll_denom) > 1e-12
                else 0.0
            )
            rows.append({
                "method": method,
                "rank": rank,
                "kl": metrics["kl"],
                "kl_recovery": recovery,
                "nll_recovery": nll_recovery,
                "top1_agreement": metrics["top1_agreement"],
                "nll": metrics["nll"],
                "ppl": metrics["ppl"],
                "residual_bpw_fp16": residual_bpw(original.shape, rank),
                "total_estimated_bpw": entropy_bpw + residual_bpw(original.shape, rank),
                "correction_energy_fraction": float(corr.square().sum() / (error.square().sum() + 1e-12)),
            })

    result = {
        "model": args.model,
        "target_layer_index": layer_idx,
        "target_shape": list(original.shape),
        "seq_len": args.seq_len,
        "calib_blocks": args.calib_blocks,
        "eval_blocks": args.eval_blocks,
        "extreme_count": k,
        "ternary": {**qinfo, "entropy_bpw": entropy_bpw},
        "calibration_kl": [{"index": i, "kl": s} for i, s in scores],
        "failure_indices": failure_idx,
        "success_indices": success_idx,
        "failure_gradient_norms": failure_norms,
        "success_gradient_norms": success_norms,
        "failure_gradient_losses": failure_losses,
        "success_gradient_losses": success_losses,
        "failure_spectrum": spectrum_stats(failure_grads),
        "success_spectrum": spectrum_stats(success_grads),
        "failure_projection_energy": failure_proj_energy,
        "success_projection_energy": success_proj_energy,
        "teacher_eval": baseline_teacher,
        "ternary_eval": baseline_quant,
        "results": rows,
    }

    (outdir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    header = [
        "# Functional-subspace compression PoC",
        "",
        f"- Model: \`{args.model}\`",
        f"- Target: middle MLP down-projection, shape \`{tuple(original.shape)}\`",
        f"- Ternary symbol entropy: **{entropy_bpw:.4f} bit/weight**",
        f"- Ternary held-out KL to FP model: **{baseline_quant['kl']:.6g}**",
        f"- Ternary held-out top-1 agreement: **{baseline_quant['top1_agreement']:.4%}**",
        f"- Failure-gradient effective rank ({k} samples): **{result['failure_spectrum']['effective_rank']:.3f}**",
        f"- Success-gradient effective rank ({k} samples): **{result['success_spectrum']['effective_rank']:.3f}**",
        f"- Quantization-error energy inside failure span: **{failure_proj_energy:.4%}**",
        f"- Quantization-error energy inside success span: **{success_proj_energy:.4%}**",
        "",
        "| method | rank | KL ↓ | KL recovery ↑ | NLL recovery ↑ | top-1 agreement ↑ | residual bpw | estimated total bpw |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        header.append(
            f"| {row['method']} | {row['rank']} | {row['kl']:.6g} | "
            f"{row['kl_recovery']:.2%} | {row['nll_recovery']:.2%} | "
            f"{row['top1_agreement']:.4%} | {row['residual_bpw_fp16']:.4f} | "
            f"{row['total_estimated_bpw']:.4f} |"
        )
    header += [
        "",
        "## Interpretation",
        "",
        "The hypothesis is supported only if the failure-conditioned residual consistently "
        "recovers more held-out functional KL than the success-conditioned and random controls "
        "at the same rank/bit overhead. Weight-SVD is a strong baseline: beating it would indicate "
        "that functional sensitivity contains compression-relevant information not captured by "
        "weight reconstruction error alone.",
        "",
        "This is a one-layer, small-model falsification test. A positive result is evidence to scale "
        "the experiment, not evidence that the same gain will hold for large models or full-model quantization.",
    ]
    (outdir / "summary.md").write_text("\n".join(header) + "\n", encoding="utf-8")
    print("\n".join(header))


if __name__ == "__main__":
    main()
