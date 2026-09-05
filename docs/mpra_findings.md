# Executed MPRA pilot: what h guidance did and did not improve

Date: 2026-09-05. These are exploratory runs, not a pre-registered test or a
comparison with the existing CORAL and Ledidi editors. See the
[protocol](mpra_prototype.md) for preprocessing, mathematical target and splits.

## Setup

The filtered author data provide 7,360 complete three-motif orientation cubes,
each containing eight distinct, experimentally assayed constructs. Train,
validation and test contain 585, 140 and 117 unordered TF-multiset families,
respectively. All orders, spacings and backgrounds of a family stay together.
The 300 selected test contexts span 103 families. Starts and requested up/down
directions are chosen without looking at their assay responses.

A pairwise motif/orientation ridge model, selected using within-neighbourhood
validation error, predicts activity. Its test Pearson correlation is 0.872
globally but 0.675 after centring within each neighbourhood. Test contrast-scale
RMSE is 0.139 log2 units for centred states; this is not a calibrated uncertainty
bound for every pairwise difference.

An initial histogram-model probe at gain 0.35 found only two model-feasible
cases among 300. That exposed the inadequacy of global prediction quality as a
model-selection criterion. We then compared the structured models on validation
families. This is a development history, not an independent replication.

For each gain, h is fitted using only training-family model scores and exact
backward messages. The 0.10 run has 1,055 feasible h-training tasks / 50,640
state-time examples; the 0.35 run has 269 / 12,912. Every method receives the
same eight cached activity predictions per test context. Each uses 16 particles,
six transitions and 16 independent repeats. Approximate guidance is
importance-corrected to the same terminal target.

## Main results

| Requested log2 gain | Model-feasible tasks / selected | Reference TV | Learned-h TV | Reference ESS/N | Learned-h ESS/N |
|---|---:|---:|---:|---:|---:|
| 0.10 | 154 / 300 | 0.1279 | 0.1056 | 0.4764 | 0.8737 |
| 0.35 | 40 / 300 | 0.1067 | 0.0827 | 0.3889 | 0.8460 |

TV is total variation distance between the finite particle endpoint distribution
and the exact target, averaged over repeats and contexts. Exact-h particle
sampling still has finite-sample TV (0.1005 and 0.0804); exact enumeration itself
has none. The reference method uses a constant intermediate twist, with the same
hard terminal weighting as the other methods; it is not unconstrained random
sequence generation.

The paired learned-minus-reference TV differences are -0.0223 (family-bootstrap
95% interval -0.0267 to -0.0180) and -0.0240 (-0.0353 to -0.0139). These intervals
describe the exploratory context sample, not uncertainty across all possible
model-training runs or the entire biological domain.

Crucially, this is **not an edit-count result**. The corresponding mean endpoint
edit differences are -0.0087 bases (-0.0734 to 0.0555) and +0.0019 bases
(-0.1272 to 0.1284). The reference already finds a minimum-edit feasible state
in every run in both tests. It also archives slightly more unique CF states on
average. Learned guidance costs about eight to nine times as much per run in this tiny
CPU setup, excluding model scoring and training. Direct enumeration remains
the sensible practical solver for eight states.

Thus this experiment supports improved distributional fidelity and particle
weight balance at a fixed particle count, not improved search success,
mechanism discovery, wall-clock efficiency or biological quality.

## Assay audit: the important separation

| Requested gain | Learned CF mass: aggregate observed gain passes | All three paired replicates have correct direction | All three paired replicates reach requested gain |
|---|---:|---:|---:|
| 0.10 | 61.4% | 47.1% | 32.0% |
| 0.35 | 72.7% | 89.1% | 40.9% |

These are weighted endpoint-mass averages over model-feasible contexts, not a
percentage of all 300 attempted tasks. Audit metrics are conditional on a
surviving particle run. The other methods give essentially the same biological
results, as expected when targeting the same model-defined distribution.

At the larger requested effect, the qualitative direction is considerably more
reproducible than the full requested magnitude. This distinction matters for
directional relational rules. Neither result is a calibrated confidence level.

The converse model error also matters. At gain 0.10, 31 of the 146 contexts with
no model-feasible solution contain an assayed solution reaching the target in
all three replicates; 85 have a solution passing the aggregate criterion. At
gain 0.35 these counts are 11 and 56 out of 260. Exact exhaustion of a model's
finite CF support is therefore demonstrably not biological impossibility.

## Blocking and the representation diagnostic

The generate/generalise/block/rerun loop exhausts the model's finite target
support in all model-feasible test cases. This is a verified property of the
specified cubes, not global ILP completeness or discovery of biochemical routes.

At gain 0.10, the deterministic local catalogue has 226 clauses across 154
contexts; 64 contexts have more than one catalogue region. All covered
constructs pass the aggregate experimental criterion for 113 clauses and the
all-three-replicate magnitude criterion for only 43. At gain 0.35 the respective
counts are 48 clauses, 30 aggregate-supported and 14 strictly supported.

The saved clause records include scope, source, published IDs for every assayed
completion, minimum observed contrast and minimum paired-replicate contrast.
They remain model explanations even when contradicted by the assay, but must
not then be called validated biological mechanisms.

The analytical three-round coverage reference is 1.239 local catalogue regions
for three independent target draws versus 1.468 for three exact blocking rounds
at gain 0.10. This is a finite-catalogue illustration, **not a matched-compute
superiority result**: constructing and checking clauses has a cost.

Projecting the exact joint CF law to independent orientation marginals puts an
average 7.41% of probability on model-invalid combinations at gain 0.10 (median
zero, maximum 50%). At gain 0.35 the mean is 3.22%. This diagnoses the potential
loss from factorisation; it does not measure the actual Gumbel/ALM optimiser or
establish that it causes CORAL's observed overediting.

## Rollout stability check

Raw Monte Carlo h estimates can be zero even when future success is possible.
A tiny positive support floor avoids deleting paths but leaves very large
importance-weight ratios. In the exploratory tests the raw rollout normalizer
estimates were unstable (mean estimate / exact value 1.149 at gain 0.10 and
0.761 at gain 0.35). A finite family bootstrap cannot reliably diagnose an
unseen heavy tail. Neither mathematical unbiasedness nor improved ESS alone
establishes practical normalizer accuracy.

The branch therefore also runs `defensive_rollout`: intermediate twists are
0.95 times the raw estimate plus 0.05 times the constant reference twist. The
terminal target and importance corrections are unchanged. This is an explicit
proposal-stability ablation added after inspecting the initial runs, not a
novelty claim or a new endpoint edit budget. Both raw and defensive results are
retained in the machine-readable reports.

The defensive version's mean normalizer ratios are 0.995 and 1.010, and its TV
distances are 0.1099 and 0.0880. Both are improvements over the raw-rollout
estimates in these runs, despite lower final ESS/N (0.716 and 0.625). This is a
useful warning against treating final ESS as a sufficient quality metric. It
does not establish low variance on all rare-event tasks or substitute for a
fresh confirmatory benchmark.

## Reproduction

```bash
python -m unittest discover -s tests -v
python -m coral.runners.run_tfbs_mpra --download --gain 0.1 --cases 300 --h-train-cases 2000 --repeats 16 --particles 16 --output research_results/tfbs_mpra_gain010
python -m coral.runners.run_tfbs_mpra --download --gain 0.35 --cases 300 --h-train-cases 2000 --repeats 16 --particles 16 --output research_results/tfbs_mpra_gain035
```

Fourteen unit tests pass, including exhaustive path/Doob identities, approximate
importance normalizers, defensive guidance, clause blocking, preservation of
existing exclusions, split grouping and motif-coordinate integrity. Both real
data runs complete with the pinned four-file source release. The original
Torch-based editor was not run: this environment lacks its deep-learning
dependencies, and this branch intentionally keeps that implementation separate.

Environment: Python 3.12.13, NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3,
scikit-learn 1.8.0. All reported non-timing metrics are seeded. Reports are in
`research_results/tfbs_mpra_gain010/` and `research_results/tfbs_mpra_gain035/`.
