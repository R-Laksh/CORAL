CORAL explores counterfactual explanations for genomic sequence models.

This research branch adds a CPU prototype for distributional counterfactual
search on experimentally measured HepG2 motif-orientation neighbourhoods.
The existing augmented-Lagrangian nucleotide optimisers are preserved.

Read [the benchmark protocol](docs/mpra_prototype.md) for the dataset, assumptions,
implemented stages, validation plan and limits of the current results.
The [measured findings](docs/mpra_findings.md) report both pilot runs, including
the lack of edit-count or biological-validity improvement.

The [biological-model study](docs/biological_optimizer.md) extends the engine to
native ESMC-300M with a GB1 functional head and an ENCODE FOXA1 BPNet checkpoint.
It reports corrected h-guided ALM search, actual upstream Ledidi, explicit
discrete feasibility checks, measured GB1 edit regret and component interventions.
The pilot does not establish an advantage from h guidance or GPU-scale speed.

With Python 3.11 or newer:

```bash
python -m pip install -r requirements-research.txt
python -m unittest discover -s tests -v
python -m coral.runners.run_tfbs_mpra --download --gain 0.1 --h-train-cases 2000 --repeats 16
```

The downloader retrieves four public files from a pinned author repository and
verifies SHA-256 hashes. Downloaded data are ignored by git. You can instead use
`--source-dir /path/to/TFBSs_grammar` with an existing copy of the author release.
The MPRA result files contain aggregate metrics and published construct IDs, not
newly designed sequences.
