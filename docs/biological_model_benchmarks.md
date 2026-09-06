# Biological-model counterfactual benchmarks

The benchmark isolates the search primitive while retaining CORAL's augmented-Lagrangian constraint semantics. The primary comparison is **ST-Gumbel + ALM** versus **whole-sequence particle search + the same ALM**, with and without an approximation to `h_t`. Ledidi remains the external joint-penalty baseline.

## GB1

We use the four-site GB1 IgG-binding landscape from Wu et al. (2016), positions V39, D40, G41 and V54. Only experimentally profiled genotypes are used for biological validation or experimental edit regret. Values from subsequently inferred complete landscapes are never substituted for missing experiments.

The predictor is a frozen ESM2 protein language model plus a linear ridge head. The head is fit only on measured WT, single and double mutants. Triple and quadruple variants are out-of-training-order tests. This is intentionally stricter than fitting a flexible predictor to the entire landscape: useful high-order extrapolation must already be linearly accessible in the frozen PLM representation.

For a global measured log-fitness threshold, source genotypes are selected from higher-order variants whose **experimental** minimum edit distance to the target is two. Before optimization, the frozen model is exhaustively evaluated over each source's one- and two-edit shells. This gives an exact model optimum of one or two edits without evaluating or imputing all 160,000 genotypes.

Each returned CF records hard model feasibility, edit count and exact model edit regret, whether the endpoint was experimentally measured, measured fitness/target satisfaction when available, biological-model forward/backward examples, and wall time. An unmeasured endpoint is marked experimentally unresolved rather than assigned an inferred fitness.

Methods:

1. `particle_local`: whole-sequence local proposals from the current ALM energy gradient, without future-success guidance.
2. `particle_h`: identical proposals/ALM state, twisted by a shared one-step rollout approximation to `h_t`.
3. `st_alm`: straight-through Gumbel-Softmax with the same ALM inequality penalty and hard endpoint feasibility check.

The candidate-specific `h_t` estimator is a teacher/diagnostic. The shared rollout reuses the parent's gradient-ranked edit operations, so it adds forward evaluations without candidate-specific biological-model backward passes. The scalable target is an amortized `log_h_fn` distilled from those teacher rollouts.

A counterfactual is first an explanation of the frozen predictor. GB1 lookup can establish functional support but does not prove a unique biological mechanism. Likewise, successful high-order extrapolation from a linear low-order head is evidence that useful interaction information is readable from the frozen PLM representation, not proof that pretraining alone learned a specific causal rule.

## ENCODE BPNet / ChromBPNet

`bpnet-lite` converts official TensorFlow/Keras BPNet and ChromBPNet checkpoints to differentiable PyTorch models while preserving predictions. The first ENCODE experiment should use real held-out genomic windows, fixed controls, and a scalar count increase/decrease constraint. Once feasibility/edit/compute behavior is stable, add profile and multi-model constraints.
