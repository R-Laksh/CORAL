# Distributional counterfactuals on measured HepG2 regulatory variants

Research branch: `research/distributional-cf-mpra`.
Base CORAL commit: `04e33cd274d793f09ac9566820a3db9fb0fb2e77`.
This is a finite-state calibration prototype. It is not evidence that the new
method outperforms Ledidi or the existing CORAL nucleotide optimiser.
See [the executed pilot findings](mpra_findings.md) for the results and
[the ENCODE / protein follow-on](next_benchmarks.md) for the next model choices.

## Dataset and why it fits

The primary dataset is Georgakopoulos-Soares et al. (2023), **Transcription factor
binding site orientation and order are major drivers of gene regulatory activity**,
Nature Communications 14, 2333. [Paper](https://doi.org/10.1038/s41467-023-37960-5),
[author data and code](https://github.com/IliasGeoSo/TFBSs_grammar).

The public release contains 209,440 designed sequences, a descriptive FASTA,
and three independent tables of normalised RNA, DNA, RNA/DNA ratios and barcode
counts. The loader is pinned to author commit
`2a65724010cac893e5dd667f198145a7c7c9803f` and checks all four source hashes.
The FASTA's one-based record index maps to `seq_N` in the response tables. Four
values reproduced from the authors' published plotting data independently check
this mapping. We read all three replicate files separately; the public `qc.py`
contains a repeated filename and is not reused as a preprocessing pipeline.

The first benchmark holds TF identity, TF order, motif positions, inter-motif
spacing, background scaffold and sequence length fixed, allowing only the
orientations of three inserted motifs to change. The loader verifies motif
strings against the actual DNA at each annotated position (including the 15bp
cloning prefix). There are:

- 85,750 triplet design records, each 230bp long;
- 11,664 annotated orientation neighbourhoods;
- 1,838 reduced/incomplete design cubes, excluded rather than padded;
- 2,466 complete cubes failing the minimum coverage rule;
- **7,360 complete eight-state cubes / 58,880 distinct assayed sequences**, with
  at least three barcodes in each of the three replicate tables.

Palindromic motifs do not supply distinguishable orientation interventions and
are excluded by the complete/distinct-sequence requirement. Order experiments
need a later matched design: order changes in the triplet library can also change
positions and spacing. An orientation intervention changes several nucleotides
for some motifs and only one for others; we always count actual nucleotide
Hamming changes, rather than treating every orientation flip as one base edit.

This is real experimental response data on synthetic constructs, not natural
genomic sequence variation. It validates reporter activity in the studied assay
context. It does not establish endogenous enhancer function, TF occupancy or a
biochemical interaction between the named proteins.

## Implemented stages

| Stage | Implemented component | What it establishes |
|---|---|---|
| Whole-sequence alternatives | Complete-state particles, feasible archive, product-of-marginals diagnostic | Whether independent marginals recreate invalid combinations |
| Local reference dynamics | Coordinate-change graph, exact nucleotide cost and a fixed stochastic kernel | A specified reference process without an endpoint edit-count cap |
| Immediate guidance | Myopic soft target-violation twist | A local guidance reference, not a replacement claim for ST-Gumbel gradients |
| Lookahead | Monte Carlo estimates of future terminal weight | Whether future-success estimates help preserve usable particle mass |
| Exact h | Backward dynamic programming | A calibration target and finite-state upper reference |
| Learned h | Regression of log h from local model information on training families | Whether guidance can transfer to held-out TF combinations |
| Mechanism blocking | Generate a CF, derive a local sufficient conjunction, exclude its body, rerun | A working interface and finite-support exhaustion check |
| Biological audit | Withheld paired source/endpoint replicate responses | Whether model CFs produce the requested observed activity change |

The new implementation is in `coral/optimizers/distributional.py` and the
experiment is in `coral/runners/run_tfbs_mpra.py`. It is separate from the original
ALM code. Package exports are lazy so this research backend can run without
installing PyTorch, Transformers or Ledidi.

## Target, h, and importance corrections

For start x0, model-feasible residual set F, cost d and reference kernel K, the
terminal weight is G(x) = exp(-beta d(x,x0)) 1[x in F]. The exact endpoint target
is proportional to K^T(x0,x) G(x); it is **not** simply exp(-beta d) times the
indicator, because the reference endpoint probabilities matter.

The backward recursion is h_T = G and h_t = K h_(t+1). With a positive
intermediate approximation psi, the proposal is

`q_t(x'|x) = K(x,x') psi_(t+1)(x') / (K psi_(t+1))(x)`.

The incremental importance weight is `(K psi_(t+1))(x) / psi_t(x)`.
Starting the normalizer estimate with psi_0(x0) makes these terms telescope to
the reference path measure times G. The terminal twist remains exactly G.
Approximate rollout zeros get a positive intermediate floor to preserve support.
An all-zero particle population is reported as a sampling failure, never as a
proof that the target is impossible. Exact h gives constant incremental weights.
The additional defensive-rollout ablation uses 0.95 times the raw rollout twist
plus 0.05 times the constant reference twist at intermediate times, keeping the
terminal condition unchanged. It was added after the first exploratory runs
exposed heavy-tailed normalizer estimates; it is not a pre-registered result.

Tests verify the recursion against exhaustive path enumeration, the transformed
kernel's endpoint law, normalizer estimation with an approximate twist, hard
exclusion, and preservation of overlapping alternatives during blocking.

The six-step default permits every state in the three-dimensional cube. This
is a restricted intervention vocabulary backed by existing assays; it is not
an unrestricted 230bp search and cannot certify a minimum over arbitrary DNA.
Likewise, low relative-entropy action is not identical to minimum nucleotide
edit count. Both the reference dynamics and endpoint cost are recorded.

The reference used here contains mutation-distance preferences and the
measured intervention domain. It is **not a learned DNA plausibility model**.
A frozen, independently trained biological prior is a later ablation.

## Model, splits, and audit separation

All orders, orientations, spacings and backgrounds of the same unordered TF
multiset are assigned to one split by a stable hash. The retained cubes split
into 5,175 train, 1,236 validation and 949 test neighbourhoods under seed 20260905.
Identical DNA across retained neighbourhoods is checked and rejected; it cannot
silently leak across splits.

For this pilot we compare a histogram gradient-boosted model with regularised
pairwise motif/orientation models. The predictor is selected using the lowest
**within-neighbourhood** validation RMSE, not global test correlation. These
are structured screening models, not nucleotide CNNs or foundation models.
Their representations already use the experiment's vocabulary, so recovery of
that vocabulary is not an independent discovery claim.

The h teacher uses model predictions, never experimental responses from test
families. Training inputs contain remaining time, current/one-move model margins,
distances, orientations and reference transition probabilities. Infeasible
training problems are counted and excluded from value fitting. The held-out
benchmark chooses neighbourhoods, source states and up/down directions by a
stable hash independent of assay outcomes. Requested gain is relative to the
source prediction. Model-infeasible tasks remain in the denominator and are
reported separately rather than silently dropped.

The experiment reports both global prediction and centred within-neighbourhood
prediction quality. High overall correlation can reflect differences in TF
identity or background while failing to predict the small changes relevant to
counterfactuals.

Only after search targets are defined do we audit endpoints using the observed
responses. The response summary is log2(mean of the three RNA/DNA ratios),
matching the authors' plotting summaries. Three endpoint checks are distinct:

1. The aggregate measured change reaches the requested gain.
2. All three paired replicate differences have the requested direction.
3. All three paired replicate differences reach the requested gain.

The last is a conservative operational acceptance rule, **not** a calibrated
95% biological-confidence statement. There are only three assay replicates and
no barcode-level observations in the processed tables. We do not fabricate
barcode standard errors or treat barcodes, backgrounds or particles as
independent biological replicates. Summary uncertainty resamples TF-multiset
families, after averaging repeated sampler runs within each case.

## What constitutes an explanation

A clause is a conjunction of orientation literals scoped to its fixed TF
identities, positions, spacing, background, source and requested direction.
The prototype finds the smallest body whose completions all satisfy the model
target within the supplied cube. This is a local finite-state clause learner,
not a Popper/MagicPopper integration or a globally valid biological rule.

Blocking a body A AND B excludes that conjunction. It retains A AND C when B
is absent. The implementation reruns search under accumulated exclusions and
distinguishes particle failure, a round limit, and exhaustion of the finite
reference support. The global ILP stage should replace this local learner while
retaining the exclusion interface and independent counterexample checks.

Learned h is currently trained on unblocked tasks; it is not yet trained to
reason about arbitrary sets of clauses. Exact guidance supplies the blocking
calibration experiment. Future learned guidance needs the residual condition as
an input or new rollouts after each block.

## Validation programme beyond this pilot

| Question | Experiment and gate |
|---|---|
| Do gradients help the actual editor? | Adapt the current CORAL and Ledidi to the same orientation actions and frozen differentiable sequence model; hold tau schedules, margins, starts and postprocessing fixed where applicable. Then compare on the same assayed endpoints. |
| Does h help beyond local gradients? | Compare immediate gradients, rollout h, learned h and exact h. Report model calls, rollout calls, memory, elapsed time, normalizer stability, ESS and mechanism coverage. Charge training and reference-model access separately. |
| Does geometry matter? | Compare equal-cost edges, nucleotide-distance edges, KL/mirror updates and context-dependent biological proposals with the same target and scorer. Do not call a Hamming OT distance to a point source a new objective. |
| Does the representation matter? | Product categorical vs mixtures vs full-state particles, with matched work. Include held-out cases with competing and overlapping successful edit sets. The marginal-projection diagnostic is not an optimiser benchmark. |
| Do clauses improve discovery? | Independent restarts, exact-sequence blocking, edit-set blocking and clause-body blocking at matched evaluation effort. Challenge broad clauses before exclusion; report coverage and false exclusions against the finite catalogue. |
| Do explanations transfer? | Train the sequence model and ILP on one set of TF multisets; audit unseen combinations, positions and scaffolds. Test the second background separately rather than treating two backgrounds as a universal generalisation test. |
| Does plausibility improve? | Add a separately trained DNA prior and audit held-out reporter effects. Prior likelihood alone is not the biological validity criterion. |
| Does it scale? | Move from measured orientation cubes to larger assayed pair/order libraries with verified positional matching, then unrestricted nucleotide edits. Unmeasured generated endpoints require new assays or must remain unvalidated hypotheses. |

For the next meaningful model stage, train a differentiable sequence model on
the training constructs, select it using within-neighbourhood validation effects,
and keep an untouched experimental audit split. A larger real-sequence task can
use [DeepSTARR](https://doi.org/10.1038/s41588-022-01048-5) as an external model and
regulatory context, but its broad training sequences do not automatically give
measured outcomes for arbitrary generated CFs. Dense measured perturbation
families remain necessary for the decisive audit.

Direct enumeration is the practical strongest baseline in an eight-state
neighbourhood. We use small graphs to test the mathematics and the biological
audit, not to claim a runtime win over enumeration or a result on unrestricted
genomic counterfactuals.

## Reproduction and outputs

```bash
python -m pip install -r requirements-research.txt
python -m unittest discover -s tests -v
python -m coral.runners.run_tfbs_mpra --download --gain 0.1 --cases 300 --h-train-cases 2000 --repeats 16 --particles 16 --output research_results/tfbs_mpra_gain010
```

`summary.json` records configuration, hashes, model selection, splits, audit
metrics and bootstrap intervals. `case_metrics.csv` averages sampler repeats
within each case; `neighbourhood_diagnostics.csv` includes model-infeasible cases
and product-marginal diagnostics. `local_clauses.json` contains scoped bodies and
their experimental audits. `benchmark.png` is an aggregate scientific figure.
No newly designed DNA sequences are emitted.

## Related optimiser and logic work

- [DRAKES](https://arxiv.org/abs/2410.13643): discrete trajectory optimisation with a reference process and Gumbel gradients.
- [Discrete Guidance](https://arxiv.org/abs/2406.01572): conditional rates and efficient local guidance approximations.
- [STGFlow](https://arxiv.org/abs/2503.17361): straight-through guidance for biological sequence flows.
- [Discrete Wasserstein natural gradients](https://ww3.math.ucla.edu/camreport/cam18-18.pdf): a graph-based distribution geometry.
- [Semantic loss](https://proceedings.mlr.press/v80/xu18h.html): logical satisfaction mass and weighted model counting.
- [Minimal correction subset enumeration](https://www.ijcai.org/Proceedings/13/Papers/098.pdf): blocking-based enumeration with explicit finite-domain guarantees.
