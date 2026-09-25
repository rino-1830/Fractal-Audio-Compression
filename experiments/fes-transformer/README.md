# FES pretrained Transformer experiment

Tests Functional Error Shaping on `EleutherAI/pythia-14m`.

Each transformer block's MLP up-projection receives the same 15 ternary candidates. The experiment compares local weight-MSE selection, per-layer functional selection, and cross-layer FES vector balancing followed by exact KL reranking.

This is a partial-model test before a full Bonsai-style ternary compression experiment.
