# Functional Error Shaping (FES) proof-of-concept

This isolated experiment tests one narrow hypothesis for neural-network compression:

> A combination of per-layer ternary quantization choices can have larger local weight-reconstruction error but smaller held-out final-output error when the layer errors are selected to cancel in output space.

The experiment compares three selectors over exactly the same 15 ternary candidates per layer:

1. **local_weight_mse** — independently minimizes weight reconstruction MSE.
2. **independent_functional** — independently minimizes each layer's final-output perturbation.
3. **fes** — treats each candidate's final-output perturbation as a vector, projects those vectors into a compact sketch, beam-searches combinations whose vectors cancel, reranks exactly, then performs a small functional coordinate refinement.

The model is intentionally synthetic: an 8-layer residual nonlinear network with random fixed weights. It measures function preservation rather than language-model quality. This makes the first test cheap, deterministic, and independent of model downloads.

Primary success condition: on held-out inputs, FES should reduce KL divergence versus local-MSE selection while deliberately accepting higher local weight MSE.

Run locally:

```bash
python experiments/fes-proof/fes_experiment.py --out results --seeds 24
```

GitHub Actions uploads `results.json`, `summary.md`, and `per_seed.csv` as an artifact.
