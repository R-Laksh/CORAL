"""ENCODE engineering smoke test on an actual hg38 ChIP-seq peak window.

This is not an out-of-fold biological efficacy benchmark: model-fold membership
must first be checked against ENCODE's released training/test-region archive.
Mutated-window occupancy is unmeasured, regardless of reference ChIP evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from coral.models.bpnet import ZeroControlBPNet, MODEL_ID, MODEL_REVISION
from coral.optimizers.alm_sequence import Constraints, SequenceOracle, run_h_alm, run_st_alm, run_ledidi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--window-json", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--budget", type=int, default=1024)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    window = json.loads(Path(args.window_json).read_text())
    sequence = window["dna"].upper()
    if len(sequence) != 2114 or set(sequence) - set("ACGT"):
        raise ValueError("Expected a complete canonical 2114bp window")
    source = np.array(["ACGT".index(a) for a in sequence])
    model = ZeroControlBPNet(args.checkpoint)
    x = torch.nn.functional.one_hot(torch.tensor(source), 4).T.float()[None]
    with torch.no_grad():
        initial = float(model(x)[0, 0])
    # Fixed 128bp around the observed summit: no model-gradient preselection.
    mutable = np.arange(1057 - 64, 1057 + 64)
    constraints = Constraints([initial - np.log(2)], [-1], [.25])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "fold": 0,
                "window": {k: window[k] for k in ("genome", "chrom", "start", "end")},
                "source": sequence, "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                "peak_accession": "ENCFF081USG", "experiment": "ENCSR865RXA",
                "window_selection": "first chr1 IDR peak in coordinate order, summit +/-1057bp",
                "model_fold_membership": "UNVERIFIED: training/test archive download blocked by automatic approval review (usage limit)",
                "assay_status": "reference peak measured; mutant occupancy unmeasured",
                "target": "decrease predicted total counts two-fold at fixed zero controls",
                "initial_logcount": initial, "threshold_logcount": float(constraints.threshold[0]),
                "mutable_positions_zero_based": [int(mutable[0]), int(mutable[-1])],
                "kernel": "independent per-site reversible substitutions; full endpoint support; expected 0.5 substitutions per step",
                "purpose": "engineering smoke test; not a generalisation or biological-validation result"}
    (output / "window_protocol.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records = []
    for seed in range(2):
        for method in ("alm_st_gumbel", "alm_h_rollout_control", "alm_h_myopic", "alm_h_rollout", "ledidi_hinge"):
            oracle = SequenceOracle(model, source, mutable, 4, constraints, args.budget)
            if method == "alm_st_gumbel":
                result = run_st_alm(oracle, seed=seed, steps=args.budget // 8 + 8, lr=1.)
            elif method == "ledidi_hinge":
                result = run_ledidi(oracle, seed=seed, steps=args.budget // 8 + 8, edit_weight=.1)
            else:
                result = run_h_alm(oracle, seed=seed, guidance=method.removeprefix("alm_h_"),
                                   episodes=20, horizon=8, kernel="independent")
            result["seed"] = seed
            if result["best"]:
                result["best"]["sequence"] = "".join("ACGT"[a] for a in result["best"]["ids"])
                result["best"]["predicted_count_ratio"] = float(np.exp(result["best"]["scores"][0] - initial))
            records.append(result)
            print(method, seed, None if result["best"] is None else result["best"]["edits"], flush=True)
            (output / "smoke_runs.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
