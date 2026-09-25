# FES on a pretrained Transformer

This experiment tests Functional Error Shaping on pretrained `distilgpt2`.

It ternary-quantizes the MLP `c_fc` matrices in the first four Transformer blocks and compares three selectors over the same candidate set:

1. local weight-MSE minimization;
2. independent per-layer functional minimization;
3. joint FES selection using output-error-vector cancellation plus exact calibration reranking.

The primary metric is held-out KL divergence from the original model. The experiment also reports cross-entropy, top-1 logit agreement, weight MSE, ternary entropy, sparsity, and a linearized cancellation ratio.

A smaller two-layer smoke test was run on the host first. It produced held-out KL values 0.709 (local), 0.624 (independent functional), and 0.604 (FES).
