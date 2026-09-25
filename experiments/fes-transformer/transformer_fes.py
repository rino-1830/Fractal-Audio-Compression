#!/usr/bin/env python3
import argparse, json, math, os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

FALLBACK_TEXT = """
Language models predict the next token from the tokens that came before it. Compression changes
the numerical representation of a model, but a useful compression method should preserve the
function computed by the original network. Quantization maps many distinct parameter values to a
small set of representable values. The reconstruction error in one layer does not necessarily
measure the effect of that error on the final output. Residual networks make this distinction
especially important because perturbations from different layers can interact and partially cancel.

Scientific experiments should separate calibration data from held-out evaluation data. A method
that only reduces error on calibration examples can overfit those examples. A stronger result is
an improvement on held-out sequences that were not used to choose the compressed weights.
Information theory provides a second perspective. The number of stored bits matters, but so does
where approximation error is placed. A representation can tolerate substantial error in directions
that have little effect on observable outputs.

Transformers contain attention blocks and feed-forward blocks connected by residual streams.
The feed-forward layers contain large matrices and are common targets for compression. Ternary
quantization constrains weights to negative scale, zero, or positive scale. Different thresholds
produce different sparsity levels and different approximation errors. If several layers are
compressed together, choosing each layer independently may not minimize the error of the network
as a whole. This experiment tests whether jointly choosing the quantization alternatives can improve
the final language-model distribution even when some individual layers are reconstructed less
accurately.

The held-out portion contains different prose. Engineers often optimize systems by measuring the
quantity that users actually observe instead of a convenient internal proxy. In a neural network,
the final probability distribution is observable while individual parameter errors are internal.
A compression method can therefore benefit from modeling how internal perturbations propagate.
The central question is whether the sign and direction of errors from multiple layers can be chosen
so that their effects cancel at the output.
""" * 8

def get_blocks(tokenizer, seq_len, cal_blocks, test_blocks):
    total = cal_blocks + test_blocks
    text = FALLBACK_TEXT
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    need = total * seq_len + 1
    if ids.numel() < need:
        reps = math.ceil(need / ids.numel())
        ids = ids.repeat(reps)
    ids = ids[: total * seq_len]
    blocks = ids.view(total, seq_len)
    return blocks[:cal_blocks], blocks[cal_blocks:cal_blocks+test_blocks]

def ternary_candidates(w, threshold_factors, scale_multipliers):
    wf = w.float().cpu()
    mean_abs = wf.abs().mean().item() + 1e-12
    out = []
    for tf in threshold_factors:
        symbols = torch.sign(wf) * (wf.abs() >= tf * mean_abs)
        denom = (symbols * symbols).sum().item()
        base = mean_abs if denom == 0 else (wf * symbols).sum().item() / denom
        probs = [(symbols == v).float().mean().item() for v in (-1, 0, 1)]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        for sm in scale_multipliers:
            q = (base * sm) * symbols
            out.append({
                "q": q.to(dtype=w.dtype),
                "tf": tf,
                "sm": sm,
                "weight_mse": float(torch.mean((wf - q.float()) ** 2).item()),
                "entropy_bpw": entropy,
                "zero_fraction": probs[1],
            })
    return out

@torch.inference_mode()
def logits_for(model, blocks, batch_size):
    chunks = []
    for i in range(0, len(blocks), batch_size):
        x = blocks[i:i+batch_size]
        chunks.append(model(x).logits.cpu())
    return torch.cat(chunks, dim=0)

def eval_metrics(base_logits, q_logits, blocks):
    # Predict token t+1 from position t.
    bp = base_logits[:, :-1, :].float()
    qp = q_logits[:, :-1, :].float()
    labels = blocks[:, 1:]
    logp = F.log_softmax(bp, dim=-1)
    logq = F.log_softmax(qp, dim=-1)
    p = logp.exp()
    kl = torch.sum(p * (logp - logq), dim=-1).mean().item()
    ce = F.cross_entropy(qp.reshape(-1, qp.shape[-1]), labels.reshape(-1)).item()
    base_ce = F.cross_entropy(bp.reshape(-1, bp.shape[-1]), labels.reshape(-1)).item()
    mse = torch.mean((bp - qp) ** 2).item()
    agree = (bp.argmax(-1) == qp.argmax(-1)).float().mean().item()
    return {
        "kl": float(kl),
        "ce": float(ce),
        "base_ce": float(base_ce),
        "ppl": float(math.exp(min(ce, 20))),
        "base_ppl": float(math.exp(min(base_ce, 20))),
        "logit_mse": float(mse),
        "top1_agreement": float(agree),
    }

def build_probe(base_logits, topk, positions_per_block):
    # Fixed observable coordinates: top-k baseline logits at the last positions.
    b, s, _ = base_logits.shape
    start = max(0, s - positions_per_block)
    sel = base_logits[:, start:, :].float()
    idx = torch.topk(sel, k=topk, dim=-1).indices
    base_vals = torch.gather(sel, -1, idx)
    return start, idx, base_vals

def probe_vector(logits, start, idx, base_vals):
    vals = torch.gather(logits[:, start:, :].float(), -1, idx)
    return (vals - base_vals).reshape(-1)

def apply_choices(params, candidate_sets, choices):
    for p, cs, cidx in zip(params, candidate_sets, choices):
        p.data.copy_(cs[cidx]["q"].to(device=p.device, dtype=p.dtype))

def restore(params, originals):
    for p, w in zip(params, originals):
        p.data.copy_(w.to(device=p.device, dtype=p.dtype))

def choice_stats(candidate_sets, choices):
    mses = [candidate_sets[i][c]["weight_mse"] for i, c in enumerate(choices)]
    ent = [candidate_sets[i][c]["entropy_bpw"] for i, c in enumerate(choices)]
    zero = [candidate_sets[i][c]["zero_fraction"] for i, c in enumerate(choices)]
    return {
        "weight_mse_mean": float(sum(mses)/len(mses)),
        "entropy_bpw_mean": float(sum(ent)/len(ent)),
        "zero_fraction_mean": float(sum(zero)/len(zero)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--cal-blocks", type=int, default=4)
    ap.add_argument("--test-blocks", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--beam-width", type=int, default=64)
    ap.add_argument("--rerank", type=int, default=6)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--probe-positions", type=int, default=12)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results/transformer_fes.json")
    args = ap.parse_args()

    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "4"))))
    torch.manual_seed(0)

    if args.quick:
        args.layers = min(args.layers, 2)
        args.cal_blocks = min(args.cal_blocks, 2)
        args.test_blocks = min(args.test_blocks, 3)
        args.beam_width = min(args.beam_width, 16)
        args.rerank = min(args.rerank, 3)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    model.to("cpu")

    cal, test = get_blocks(tok, args.seq_len, args.cal_blocks, args.test_blocks)
    base_cal = logits_for(model, cal, args.batch_size)
    base_test = logits_for(model, test, args.batch_size)
    start, probe_idx, base_probe = build_probe(base_cal, args.topk, args.probe_positions)

    blocks = model.transformer.h
    layer_ids = list(range(min(args.layers, len(blocks))))
    params = [blocks[i].mlp.c_fc.weight for i in layer_ids]
    originals = [p.detach().cpu().clone() for p in params]

    tfs = [0.55, 0.75, 0.95, 1.15]
    sms = [0.92, 1.00, 1.08]
    candidate_sets = [ternary_candidates(w, tfs, sms) for w in originals]
    n_candidates = len(candidate_sets[0])

    local_choices = [
        min(range(n_candidates), key=lambda c: candidate_sets[i][c]["weight_mse"])
        for i in range(len(params))
    ]

    sketches = []
    independent_choices = []
    candidate_probe_mse = []
    for li, p in enumerate(params):
        layer_vecs = []
        scores = []
        for ci, cand in enumerate(candidate_sets[li]):
            p.data.copy_(cand["q"].to(dtype=p.dtype))
            qlog = logits_for(model, cal, args.batch_size)
            vec = probe_vector(qlog, start, probe_idx, base_probe)
            layer_vecs.append(vec)
            scores.append(float(torch.mean(vec * vec).item()))
            p.data.copy_(originals[li].to(dtype=p.dtype))
        sketches.append(layer_vecs)
        candidate_probe_mse.append(scores)
        independent_choices.append(int(min(range(n_candidates), key=lambda c: scores[c])))

    # Beam search over additive output-error sketches.
    beam = [(0.0, torch.zeros_like(sketches[0][0]), tuple())]
    for li in range(len(params)):
        nxt = []
        for _, acc, choices in beam:
            for ci in range(n_candidates):
                v = acc + sketches[li][ci]
                score = float(torch.dot(v, v).item())
                nxt.append((score, v, choices + (ci,)))
        nxt.sort(key=lambda x: x[0])
        beam = nxt[:args.beam_width]

    # Exact calibration rerank compensates for nonlinear interactions.
    reranked = []
    for _, _, choices in beam[:args.rerank]:
        apply_choices(params, candidate_sets, choices)
        qcal = logits_for(model, cal, args.batch_size)
        metrics = eval_metrics(base_cal, qcal, cal)
        reranked.append((metrics["kl"], list(choices), metrics))
        restore(params, originals)
    reranked.sort(key=lambda x: x[0])
    fes_choices = reranked[0][1]

    methods = {
        "local_weight_mse": local_choices,
        "independent_functional": independent_choices,
        "fes": fes_choices,
    }

    result = {
        "model": args.model,
        "layer_ids": layer_ids,
        "candidate_count_per_layer": n_candidates,
        "config": vars(args),
        "methods": {},
        "fes_rerank": [{"cal_kl": x[0], "choices": x[1]} for x in reranked],
    }

    for name, choices in methods.items():
        apply_choices(params, candidate_sets, choices)
        qcal = logits_for(model, cal, args.batch_size)
        qtest = logits_for(model, test, args.batch_size)
        entry = choice_stats(candidate_sets, choices)
        entry["choices"] = choices
        entry["cal"] = eval_metrics(base_cal, qcal, cal)
        entry["test"] = eval_metrics(base_test, qtest, test)
        # Linearized cancellation: sum individual energies / energy of their vector sum.
        vecs = [sketches[i][choices[i]] for i in range(len(params))]
        denom = torch.dot(sum(vecs[1:], vecs[0].clone()), sum(vecs[1:], vecs[0].clone())).item() + 1e-30
        numer = sum(torch.dot(v, v).item() for v in vecs)
        entry["linearized_cancellation_ratio"] = float(numer / denom)
        result["methods"][name] = entry
        restore(params, originals)

    local = result["methods"]["local_weight_mse"]["test"]["kl"]
    indep = result["methods"]["independent_functional"]["test"]["kl"]
    fes = result["methods"]["fes"]["test"]["kl"]
    result["summary"] = {
        "fes_vs_local_test_kl_ratio": float(fes/(local+1e-30)),
        "fes_vs_independent_test_kl_ratio": float(fes/(indep+1e-30)),
        "fes_beats_local": bool(fes < local),
        "fes_beats_independent": bool(fes < indep),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    for name, e in result["methods"].items():
        print(name, json.dumps({
            "test_kl": e["test"]["kl"],
            "test_ce": e["test"]["ce"],
            "top1": e["test"]["top1_agreement"],
            "weight_mse": e["weight_mse_mean"],
            "cancel": e["linearized_cancellation_ratio"],
        }, indent=2))

if __name__ == "__main__":
    main()

