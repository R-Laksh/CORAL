"""Post-search FP32 model-regret bounds and measured component interventions.

Audit queries never feed back into search. Missing assay measurements stay null.
The exact assay optimum is over measured variants; the exact *model* optimum
is reported only if all smaller Hamming shells are certified infeasible.
"""
import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

from coral.datasets.gb1 import load_gb1, SITES, decode
from coral.models.esm import load_backbone, ESMFunctionalPredictor


def subsets(source, endpoint):
    edited = np.flatnonzero(source != endpoint)
    variants, masks = [], []
    for bits in itertools.product((0, 1), repeat=len(edited)):
        x = source.copy()
        selected = edited[np.array(bits, dtype=bool)]
        x[selected] = endpoint[selected]
        variants.append(x)
        masks.append(bits)
    return np.asarray(variants), masks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--head-dir", required=True)
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    data, head_dir, directory = load_gb1(args.data), Path(args.head_dir), Path(args.runs_dir)
    records = [json.loads(line) for line in (directory / "runs.jsonl").read_text().splitlines()]
    metadata = json.loads((head_dir / "model_validation.json").read_text())
    model = ESMFunctionalPredictor(load_backbone(metadata.get("family", "esm2"),
                                  args.checkpoint, sites=SITES, device=args.device), head_dir / "head.npz")
    queries, shells = {}, {}
    for row in sorted({r["source_index"] for r in records}):
        source = data.ids[row]
        shell = [source.copy()]
        for pos in SITES:
            for aa in range(20):
                if source[pos] == aa:
                    continue
                x = source.copy()
                x[pos] = aa
                shell.append(x)
        shells[row] = shell
        queries.update({tuple(x): x for x in shell})
    for result in records:
        if result["best"] is not None:
            variants, _ = subsets(data.ids[result["source_index"]], np.asarray(result["best"]["ids"]))
            queries.update({tuple(x): x for x in variants})
    states = list(queries.values())
    tic, chunks = time.perf_counter(), []
    for start in range(0, len(states), 32):
        x = torch.nn.functional.one_hot(torch.tensor(np.asarray(states[start:start + 32]), device=args.device), 20).transpose(1, 2).float()
        with torch.no_grad():
            chunks.append(model(x).cpu().numpy().ravel())
    score_map = dict(zip(queries, np.concatenate(chunks)))
    assay_map = {s: i for i, s in enumerate(data.sequences)}
    with np.load(head_dir / "split_predictions.npz") as split:
        training = set(int(x) for x in split["indices"][:int(split["n_train"])])
    bounds, evidence = {}, []
    for row, shell in shells.items():
        group = [r for r in records if r["source_index"] == row]
        target = np.log1p(group[0]["target_fitness"])
        shell_feasible = [x for x in shell if score_map[tuple(x)] >= target]
        if shell_feasible:
            minimum = min(int(np.count_nonzero(x != data.ids[row])) for x in shell_feasible)
            lower, upper = minimum, minimum
        else:
            lower = 2
            known = [r["best"]["edits"] for r in group if r["best"] is not None]
            upper = min(known) if known else None
        bounds[row] = {"lower": lower, "upper": upper,
                       "certified_optimum": lower if upper == lower else None,
                       "certification": "source plus all 76 single substitutions; known feasible returned endpoints"}
    for result in records:
        if result["best"] is None:
            continue
        row, endpoint = result["source_index"], np.asarray(result["best"]["ids"])
        source = data.ids[row]
        variants, masks = subsets(source, endpoint)
        d = int(np.count_nonzero(source != endpoint))
        sequences = decode(variants)
        assay_rows = [assay_map.get(s) for s in sequences]
        measured = [float(data.fitness[i]) if i is not None else None for i in assay_rows]
        predicted = np.array([score_map[tuple(x)] for x in variants])
        target = result["target_fitness"]
        leave_one_out = np.array([sum(mask) == d - 1 for mask in masks])
        all_measured = all(y is not None for y in measured)
        signs = np.array([(-1) ** (d - sum(mask)) for mask in masks])
        bound = bounds[row]
        evidence.append({"method": result["method"], "source_index": row, "seed": result["seed"],
                         "edits": d, "model_optimum": bound,
                         "model_edit_regret": d - bound["certified_optimum"] if bound["certified_optimum"] is not None else None,
                         "model_edit_regret_bounds": [d - bound["upper"] if bound["upper"] is not None else None, d - bound["lower"]],
                         "model_all_edits_necessary": bool((predicted[leave_one_out] < np.log1p(target)).all()) if d else None,
                         "assay_all_edits_necessary": bool(measured[-1] >= target and
                             (np.array(measured)[leave_one_out] < target).all()) if all_measured and d else None,
                         "highest_order_interaction": {
                             "order": d,
                             "model_log1p": float(signs @ predicted) if d >= 2 else None,
                             "assay_raw_fitness": float(signs @ np.array(measured)) if all_measured and d >= 2 else None,
                             "assay_log1p": float(signs @ np.log1p(measured)) if all_measured and d >= 2 else None},
                         "interventions": [{"mask": list(mask), "sequence": seq, "prediction_log1p": float(y),
                                            "assay_fitness": observed, "head_training_member": index in training if index is not None else None}
                                           for mask, seq, y, observed, index in zip(masks, sequences, predicted, measured, assay_rows)]})
    report = {"model_bounds_by_source": bounds, "returned_cf_evidence": evidence,
              "additional_audit_work": {"forward_sequences": len(states), "backward_sequences": 0,
                                        "seconds": time.perf_counter() - tic, "precision": "float32"},
              "scope": "Finite GB1 four-site domain; assay regret and model regret are different quantities",
              "interaction_warning": "Contrasts depend on measurement scale; an AND threshold alone does not establish non-additivity"}
    (directory / "component_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"bounds": bounds, "audited_cfs": len(evidence), "work": report["additional_audit_work"]}, indent=2))


if __name__ == "__main__":
    main()
