"""BPNet CF search with explicit split and task-feasibility audits.

Mutated-window occupancy is unmeasured, regardless of reference ChIP evidence.
An optional exhaustive single-edit screen constructs a feasible control task;
its candidates and scores are never supplied to the optimizers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from coral.models.bpnet import ZeroControlBPNet, MODEL_ID, MODEL_REVISION, MODEL_SHA256
from coral.datasets.encode import verify_fold_window
from coral.optimizers.alm_sequence import Constraints, SequenceOracle, run_h_alm, run_st_alm, run_ledidi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--window-json", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--budget", type=int, default=1024)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--split-archive", required=True)
    p.add_argument("--task", choices=["twofold", "single-edit-control"], default="twofold")
    p.add_argument("--seeds", type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    window = json.loads(Path(args.window_json).read_text())
    sequence = window["dna"].upper()
    if len(sequence) != 2114 or set(sequence) - set("ACGT"):
        raise ValueError("Expected a complete canonical 2114bp window")
    source = np.array(["ACGT".index(a) for a in sequence])
    split_validation = verify_fold_window(args.split_archive, window)
    if hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError("Expected the validated fold-0 BPNet checkpoint")
    model = ZeroControlBPNet(args.checkpoint)
    x = torch.nn.functional.one_hot(torch.tensor(source), 4).T.float()[None]
    with torch.no_grad():
        initial = float(model(x)[0, 0])
    # Fixed 128bp around the observed summit: no model-gradient preselection.
    mutable = np.arange(1057 - 64, 1057 + 64)
    constraints = Constraints([initial - np.log(2)], [-1], [.25])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    calibration = None
    if args.task == "single-edit-control":
        tic = time.perf_counter()
        variants, edits = [], []
        for pos in mutable:
            for base in range(4):
                if base != source[pos]:
                    state = source.copy()
                    state[pos] = base
                    variants.append(state)
                    edits.append({"position_zero_based": int(pos), "base": "ACGT"[base]})
        scores = []
        with torch.no_grad():
            for start in range(0, len(variants), 32):
                batch = torch.nn.functional.one_hot(torch.tensor(np.asarray(variants[start:start + 32])), 4).transpose(1, 2).float()
                scores.extend(model(batch)[:, 0].tolist())
        best = min(scores)
        if initial - best < 1e-5:
            raise ValueError("No numerically separated single-edit decrease: choose another control window")
        threshold = (initial + best) / 2
        constraints = Constraints([threshold], [-1], [.25])
        calibration = {"task_construction": "threshold halfway between source and best single-edit logcount",
                       "certified_model_minimum_edits": 1,
                       "initial_logcount": initial, "best_single_edit_logcount": best,
                       "threshold_logcount": threshold,
                       "feasible_single_edits": sum(s <= threshold for s in scores),
                       "additional_work": {"forward_sequences": len(variants), "backward_sequences": 0,
                                           "seconds": time.perf_counter() - tic},
                       "oracle_separation": "Only the scalar threshold is passed to search; no candidates, scores, caches or gradients",
                       "single_edits": [dict(edit, logcount=score) for edit, score in zip(edits, scores)]}
        (output / "task_calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    (output / "split_validation.json").write_text(json.dumps(split_validation, indent=2) + "\n")
    manifest = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "fold": 0,
                "window": {k: window[k] for k in ("genome", "chrom", "start", "end")},
                "source": sequence, "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                "peak_accession": "ENCFF081USG", "experiment": "ENCSR865RXA",
                "window_selection": "first chr1 IDR peak in coordinate order, summit +/-1057bp",
                "model_fold_membership": "verified_test_window; see split_validation.json",
                "assay_status": "reference peak measured; mutant occupancy unmeasured",
                "task": args.task,
                "target": "two-fold count decrease" if args.task == "twofold" else "feasibility-certified single-edit control",
                "fixed_controls": "zero control profile and zero native logcontrol counts; native logsumexp = log(2)",
                "initial_logcount": initial, "threshold_logcount": float(constraints.threshold[0]),
                "mutable_positions_zero_based": [int(mutable[0]), int(mutable[-1])],
                "kernel": "independent per-site reversible substitutions; full endpoint support; expected 0.5 substitutions per step",
                "guide_defense": "normalised proposal mixture; eta=0.02; corrected importance weights",
                "budget": args.budget, "budget_meaning": "search forward sequences; backward and task construction reported separately",
                "ledidi_edit_weight": .1, "ledidi_tuning": "not tuned for this engineering control",
                "initial_score_work": {"forward_sequences": 1},
                "purpose": "engineering smoke test; not a generalisation or biological-validation result"}
    (output / "window_protocol.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records = []
    for seed in range(args.seeds):
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
            result["model_edit_regret"] = (result["best"]["edits"] - 1
                                           if calibration and result["best"] is not None else None)
            result["biological_validation"] = "mutant occupancy unmeasured"
            records.append(result)
            print(method, seed, None if result["best"] is None else result["best"]["edits"], flush=True)
            (output / "smoke_runs.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
