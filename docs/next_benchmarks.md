# Next benchmarks: measured combinatorics, then genomic transfer

This document records the planning stage after the MPRA pilot. The subsequent
[biological-model implementation and findings](biological_optimizer.md) now
include native ESMC-300M, GB1 assay audits and verified out-of-fold BPNet
inference. Proto inference remains unexecuted. Read the newer report for the
implemented protocol and evidence; the proposals below preserve that earlier
planning context.

## What the MPRA pilot leaves unresolved

Eight assayed states per context are enough to check h mathematics, dependence
between edits, clause-body exclusions and biological counterexamples. They are
too easy to establish a lower-edit or compute-efficiency advantage: the reference
already retrieves the minimum-edit model CF in every run. The next benchmark
should have a larger measured combinatorial domain, while a separate genomic
benchmark should retain a realistic differentiable sequence predictor.

## ENCODE-BPNet: the genomic application benchmark

User-suggested collection:
[ENCODE BPNet models](https://huggingface.co/collections/kundajelab/encode-bpnet-models).
One concrete inspected starting point is
[FOXA1 ChIP-seq in HepG2, ENCSR865RXA / ENCSR337KST](https://huggingface.co/kundajelab/encode-bpnet-FOXA1-ChIP-seq-HepG2-ENCSR865RXA-ENCSR337KST).
The model card reports passed QC, a direct motif in counts and profile outputs,
five cross-validation folds, hg38, and Keras / TensorFlow SavedModel releases.
Its biosample is genetically modified HepG2, which must remain in the provenance.

The declared model inputs are 2,114bp one-hot DNA, a 1,000-by-2 control profile
and two control counts. Outputs are 1,000-by-2 profile logits and total logcounts.
Profile normalization is joint across positions and strands, not independent
per-strand softmax. The card permits zero control inputs. For a sequence CF,
controls should stay fixed between source and endpoint, with the chosen control
convention recorded and sensitivity to it tested.

This is a binding-profile target, not the reporter-expression target in the
230bp MPRA. Do not pad those reporters to 2,114bp and treat BPNet disagreement
as a matched experimental validation. Use genuine genomic windows and assay
coordinates for this separate task.

Proposed protocol:

1. Freeze an out-of-fold model for each eligible genomic region, respecting the
   released training split. Do not call agreement across five folds independent
   validation when some folds trained on the tested regions.
2. Match CORAL, Ledidi and h-guided search on the same differentiable target,
   starts, target margins and nucleotide action space. Preserve and ablate the
   original straight-through hardening / tau schedule rather than removing it.
   Charge model forward/backward calls, rollout calls, reference-prior calls,
   elapsed time and h-training cost separately.
3. Separate a total-binding-change target from a spatial profile-change target.
   Check whether clauses explain motif orientation/spacing effects beyond
   merely increasing the count head or adding motif copies.
4. Audit CFs with a different model only as a robustness screen. Withheld ChIP-seq
   can validate reference-window predictions, not occupancy of a newly mutated
   sequence. Prospective occupancy or reporter perturbations, with the endpoint
   explicitly matched, are needed for new biological claims.
5. If porting TensorFlow weights into CORAL's Torch workflow, verify counts,
   profile logits and input gradients numerically before comparing editors.

The Genomic Intelligence catalogue was inspected. Its available enhancer models
are Drosophila DeepSTARR models; they are not an appropriate human HepG2 activity
validator. No mismatched prediction was submitted. A DeepSEA-style chromatin
profile would also be corroborating model evidence, not a reporter assay.

## Protein option: GB1 as the next optimizer stress test

[Wu et al. (2016), Adaptation in protein fitness landscapes is facilitated by
indirect paths](https://elifesciences.org/articles/16965), studies a four-site
GB1 landscape with 20^4 possible variants. The measured score reflects both
folding and IgG-Fc binding in the selection assay, not organismal fitness or an
isolated biochemical binding constant. The study explicitly investigates sign
epistasis, indirect paths, reversions and higher-order interactions, making it
well matched to a future-success guide rather than purely immediate improvement.

Important data distinction: the article's complete landscape includes 10,639
imputed variants lacking adequate input read counts. Use the 149,361 measured
variants as experimental audit data; never label those imputed values as measured
ground truth. Missing nodes are unknown, not biologically impossible. The
[SaProtHub release](https://huggingface.co/datasets/SaProtHub/Dataset-GB1-fitness)
is a convenient candidate mirror, but its provenance, transformations and splits
need auditing against the primary release before implementation.

Start with fully measured 16-state binary slices to reuse the current h tests,
then move to the measured categorical graph. The current `FiniteEditGraph` is
dense and computes all pairwise distances; **do not feed the full GB1 library
into it**. The larger stage needs sparse neighbour lists / a matrix-free kernel
and categorical guidance features. Missing variants need explicit treatment in
the reference support.

Train a functional surrogate on a separated measurement subset; keep audit
responses out of h targets and clause induction. A protein language-model score
can define reference proposals or a plausibility covariate, not the functional
ground truth. Design blocks and mutation-combination splits are preferable to
interpreting an easy random split as compositional generalisation. Generalising
to a second protein remains a separate test.

The Proto catalogue exposes ESM-C embeddings/logits, ESM-C SAE features and
ESM-IF1/ProteinDPO structure-conditioned scores, but these tools reported
`needs_deploy` when inspected. No deployment or paid inference was launched.
Structure confidence, likelihood and pathogenicity scores are not substitutes
for the GB1 selection measurement. SAE features could later provide an
alternative contestable predicate vocabulary; their functional interpretation
would still require held-out perturbation tests.

## Experiments that would justify advancing the method

| Question | Comparison | Decisive endpoint |
|---|---|---|
| Does h solve the hard-search problem? | Local gradients, raw/defensive rollouts, learned h; exact h only where tractable | Minimum-edit regret and measured-success coverage versus actual compute, not only ESS |
| Does the reference geometry help? | Uniform mutation, nucleotide/residue costs, independent biological prior | Matched-target work and audit quality; explicitly account for any change in reference endpoint law |
| Is a joint representation necessary? | Product marginals, mixtures, whole-state particles | Invalid-combination mass and recovered alternative edit sets at matched work |
| Does blocking improve explanations? | Restarts, exact-sequence blocking, edit-set blocking, clause-body blocking | New experimentally supported regions per call, false exclusions and counterexamples |
| Is a conjunction an interaction? | Measured factorial contrasts and higher-order residuals | Non-additivity and background dependence, not merely thresholded additive effects |
| Can we trust a returned rule? | Model support, withheld assay support, new-context transfer | Report each separately; allow abstention rather than treating predictor agreement as certainty |

For directional claims, audit paired response signs. For quantitative claims,
audit the full requested effect. For a compositional claim, test both the
conjunction and its matched component interventions; an AND-shaped threshold
rule can arise from additive biology. With three MPRA replicates, all-replicate
agreement is useful evidence but not a calibrated 95% guarantee.

The eventual reportable primitive should therefore be a scoped rule plus its
intervention contrast, measured support, counterexamples and transfer domain.
That retains the user's contestable-vocabulary idea while making the evidential
status of each explanation explicit.
