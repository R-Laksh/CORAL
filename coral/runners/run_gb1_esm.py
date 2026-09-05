"""Matched forward-evaluation ceilings, with backward work reported separately.

All methods query the same frozen transformer and linear functional head. The
observed assay table is opened for auditing, never supplied to the search oracle.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from coral.datasets.gb1 import load_gb1, SITES, decode
from coral.models.esm import load_backbone, ESMFunctionalPredictor
from coral.optimizers.alm_sequence import Constraints, SequenceOracle, run_h_alm, run_st_alm, run_ledidi


def summarize(records):
    groups = {}
    for method in sorted({r["method"] for r in records}):
        rows = [r for r in records if r["method"] == method]
        successful = [r for r in rows if r["best"] is not None]
        measured = [r for r in successful if r["audit"]["measured"]]
        valid = [r for r in measured if r["audit"]["assay_feasible"]]
        mean = lambda x: float(np.mean(x)) if x else None
        groups[method] = {
            "runs": len(rows), "model_feasible": len(successful), "assay_measured": len(measured),
            "assay_feasible": len(valid),
            "assay_feasible_not_in_head_train": sum(not r["audit"]["head_training_member"] for r in valid),
            "mean_edits_model_feasible": mean([r["best"]["edits"] for r in successful]),
            "mean_assay_edit_regret_when_valid": mean([r["audit"]["assay_edit_regret"] for r in valid]),
            "mean_forward_sequences": mean([r["work"]["forward_sequences"] for r in rows]),
            "mean_backward_sequences": mean([r["work"]["backward_sequences"] for r in rows]),
            "mean_seconds": mean([r["seconds"] for r in rows]),
        }
    return groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--head-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cases", type=int, default=4)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--budget", type=int, default=384)
    p.add_argument("--target-fitness", type=float, default=.5)
    p.add_argument("--constraint-scale", type=float, default=.25)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--methods", nargs="+", default=["alm_st_gumbel", "alm_h_rollout_control", "alm_h_myopic", "alm_h_rollout", "ledidi_hinge"])
    p.add_argument("--validation-cases", type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    out, head_dir = Path(args.output), Path(args.head_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = load_gb1(args.data)
    metadata = json.loads((head_dir / "model_validation.json").read_text())
    with np.load(head_dir / "split_predictions.npz") as predictions:
        indices, scores = predictions["indices"], predictions["prediction"]
        n, v = int(predictions["n_train"]), int(predictions["n_validation"])
    train_set = set(int(i) for i in indices[:n])
    target = float(np.log1p(args.target_fitness))
    validation = indices[n:n + v][scores[n:n + v] < target][:args.validation_cases]
    test = indices[n + v:][scores[n + v:] < target][:args.cases]
    if len(test) != args.cases or len(validation) != args.validation_cases:
        raise ValueError("Not enough preordered model-ineligible origins")
    predictor = ESMFunctionalPredictor(load_backbone(metadata.get("family", "esm2"),
                                      args.checkpoint, sites=SITES, device=args.device), head_dir / "head.npz")
    predictor.backbone.amp = metadata["runtime"].get("amp_bf16", False)
    constraint = Constraints([target], [1], [args.constraint_scale])
    assay_index = {s: i for i, s in enumerate(data.sequences)}

    def run(method, row, seed, edit_weight=.1):
        oracle = SequenceOracle(predictor, data.ids[row], SITES, 20, constraint,
                                args.budget, batch_size=32, device=args.device)
        if method == "alm_st_gumbel":
            result = run_st_alm(oracle, seed=seed, steps=args.budget // 8 + 8, lr=1.)
        elif method == "ledidi_hinge":
            result = run_ledidi(oracle, seed=seed, steps=args.budget // 8 + 8, edit_weight=edit_weight)
        else:
            result = run_h_alm(oracle, seed=seed, guidance=method.removeprefix("alm_h_"),
                               episodes=12, horizon=4, particles=8, pool_size=3, rollouts=2)
        # Common final FP32 gate. Its model evaluations are separate, disclosed
        # audit work, and never feed back into proposal selection or assay choice.
        if predictor.backbone.amp:
            result["search_best"] = result["best"]
            pool = [(key, score) for key, score in oracle.cache.items()
                    if (constraint.residual(score) <= 0).all()]
            pool.sort(key=lambda item: (np.count_nonzero(np.asarray(item[0]) != oracle.source),
                                       float(constraint.residual(item[1]).max()), item[0]))
            # Always check x0; numerical precision must not create a false task.
            selected_ids = list(dict.fromkeys([tuple(oracle.source)] + [key for key, _ in pool[:16]]))
            predictor.backbone.amp = False
            try:
                x = torch.nn.functional.one_hot(torch.tensor(selected_ids, device=args.device), 20).transpose(1, 2).float()
                with torch.no_grad():
                    exact_scores = predictor(x).cpu().numpy()
            finally:
                predictor.backbone.amp = True
            accepted = []
            for key, score in zip(selected_ids, exact_scores):
                residual = constraint.residual(score)
                if (residual <= 0).all():
                    accepted.append({"ids": [int(a) for a in key], "scores": score.tolist(),
                                     "edits": int(np.count_nonzero(np.asarray(key) != oracle.source)),
                                     "margin": float(-residual.max())})
            result["best"] = min(accepted, key=lambda b: (b["edits"], -b["margin"], b["ids"])) if accepted else None
            result["precision_audit"] = {"forward_sequences": len(selected_ids), "precision": "float32",
                                          "source_score": exact_scores[0].tolist(),
                                          "max_candidates": 16,
                                          "accepted_candidates": len(accepted),
                                          "search_used_amp_bf16": True}
        result.update({"source_index": int(row), "source": str(data.sequences[row]), "seed": seed,
                       "source_assay_fitness": float(data.fitness[row]), "target_fitness": args.target_fitness,
                       "source_model_score": float(oracle.cache[tuple(data.ids[row])][0]),
                       "assay_measured_minimum_edits": data.measured_optimum(data.ids[row], args.target_fitness)})
        if result["best"] is not None:
            sequence = decode([result["best"]["ids"]])[0]
            observed = assay_index.get(sequence)
            result["best"]["sequence"] = sequence
            measured = observed is not None
            feasible = bool(data.fitness[observed] >= args.target_fitness) if measured else None
            result["audit"] = {"measured": measured,
                               "fitness": float(data.fitness[observed]) if measured else None,
                               "assay_feasible": feasible,
                               "head_training_member": observed in train_set if measured else None,
                               "mutation_order": int(data.mutation_order[observed]) if measured else None,
                               "assay_edit_regret": (result["best"]["edits"] - result["assay_measured_minimum_edits"])
                                                    if feasible else None,
                               "model_edit_regret": None,
                               "model_optimum_status": "not exhaustively certified"}
        else:
            result["audit"] = None
        print(f"{method} row={row} seed={seed} feasible={result['best'] is not None} "
              f"edits={None if result['best'] is None else result['best']['edits']} "
              f"forwards={result['work']['forward_sequences']} seconds={result['seconds']:.1f}", flush=True)
        return result

    # Tune the actual upstream Ledidi penalty on distinct validation origins.
    tune_path = out / "ledidi_validation.json"
    validation_runs = []
    if "ledidi_hinge" in args.methods:
        if tune_path.exists():
            saved = json.loads(tune_path.read_text())
            if saved["budget"] != args.budget or saved["target_fitness"] != args.target_fitness:
                raise ValueError("Validation checkpoint settings differ")
            validation_runs = saved["runs"]
        else:
            for weight in (.01, .1, 1.):
                for row in validation:
                    result = run("ledidi_hinge", row, 123, edit_weight=weight)
                    result["edit_weight"] = weight
                    validation_runs.append(result)
            tune_path.write_text(json.dumps({"runs": validation_runs, "budget": args.budget,
                                             "target_fitness": args.target_fitness}, indent=2) + "\n")
        def rank(weight):
            rows = [r for r in validation_runs if r["edit_weight"] == weight]
            good = [r for r in rows if r["best"] is not None]
            return (-len(good), np.mean([r["best"]["edits"] for r in good]) if good else float("inf"), weight)
        edit_weight = min((.01, .1, 1.), key=rank)
    else:
        edit_weight = .1
    protocol = {"args": vars(args), "head_sha256": hashlib.sha256((head_dir / "head.npz").read_bytes()).hexdigest(),
                "model_revision": metadata["model_revision"], "test_indices": test.tolist(),
                "validation_indices": validation.tolist(), "ledidi_edit_weight": edit_weight,
                "ledidi_commit": "beeee38f81bc00f902d41cc02695cec485b41cb9",
                "selection": "first hash-ordered origins with prediction below target; assay values not used",
                "budget_meaning": "forward sequences, not equal FLOPs or wall time; backward reported separately",
                "archive": "best jointly model-feasible discrete sequence across all queries, including rollout leaves",
                "endpoint_domain": "all 20^4 sequences at experimental sites; no hard edit budget",
                "model_validation": metadata["selected"][0]}
    protocol_path = out / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing output uses a different protocol")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    records, path = [], out / "runs.jsonl"
    if path.exists():
        records = [json.loads(line) for line in path.read_text().splitlines()]
    done = {(r["method"], r["source_index"], r["seed"]) for r in records}
    for row in test:
        for seed in range(args.seeds):
            # Rotate execution order to reduce systematic CPU warm-up bias.
            offset = seed % len(args.methods)
            for method in args.methods[offset:] + args.methods[:offset]:
                if (method, int(row), seed) in done:
                    continue
                result = run(method, row, seed, edit_weight=edit_weight)
                records.append(result)
                with path.open("a") as handle:
                    handle.write(json.dumps(result) + "\n")
                (out / "summary.json").write_text(json.dumps(summarize(records), indent=2) + "\n")
    print(json.dumps(summarize(records), indent=2), flush=True)


if __name__ == "__main__":
    main()
