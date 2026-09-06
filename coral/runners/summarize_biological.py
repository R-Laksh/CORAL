"""Summarize saved biological-model experiments without making model queries."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from coral.datasets.gb1 import SITES


METHODS = ["alm_st_gumbel", "ledidi_hinge", "alm_h_rollout_control", "alm_h_myopic", "alm_h_rollout"]
LABELS = ["ST-ALM", "Ledidi", "Rollout\ncontrol", "Myopic", "Rollout h"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="research_results")
    args = p.parse_args()
    root = Path(args.results)
    read = lambda path: json.loads((root / path).read_text())
    gb1 = read("gb1_esmc300m_comparison/summary.json")
    audit = read("gb1_esmc300m_comparison/component_audit.json")
    for method in METHODS:
        evidence = [r for r in audit["returned_cf_evidence"] if r["method"] == method]
        values = [r["model_edit_regret"] for r in evidence if r["model_edit_regret"] is not None]
        gb1[method]["mean_certified_model_regret_when_feasible"] = float(np.mean(values)) if values else None
    encode = {}
    for task in ("control", "twofold_verified"):
        runs = read(f"encode_foxa1_{task}/smoke_runs.json")
        assert len(runs) == 10, "Do not summarize a partial ENCODE run"
        encode[task] = {}
        for method in METHODS:
            group = [r for r in runs if r["method"] == method]
            good = [r for r in group if r["best"] is not None]
            regret = [r["model_edit_regret"] for r in good if r["model_edit_regret"] is not None]
            encode[task][method] = {"runs": len(group), "model_feasible": len(good),
                                   "edits_when_feasible": [r["best"]["edits"] for r in good],
                                   "mean_model_regret_when_feasible": float(np.mean(regret)) if regret else None,
                                   "mean_seconds": float(np.mean([r["seconds"] for r in group])),
                                   "assay_status": "unmeasured mutants"}
    report = {"gb1": gb1, "gb1_model_optima": audit["model_bounds_by_source"],
              "encode": encode, "scope": "Exploratory CPU pilot: four GB1 origins x one seed; one ENCODE window x two seeds"}
    (root / "biological_summary.json").write_text(json.dumps(report, indent=2) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.3), gridspec_kw={"width_ratios": [1.12, 1]})
    ax = axes[0]
    heldout = np.array([gb1[m]["assay_feasible_not_in_head_train"] for m in METHODS])
    trained = np.array([gb1[m]["assay_feasible"] for m in METHODS]) - heldout
    failed = np.array([gb1[m]["model_feasible"] - gb1[m]["assay_feasible"] for m in METHODS])
    absent = 4 - heldout - trained - failed
    bottom = np.zeros(5)
    for values, color, label in ((heldout, "#236b53", "Assay valid; outside head training"),
                                 (trained, "#89b8a5", "Assay valid; in head training"),
                                 (failed, "#e5ae78", "Model CF fails assay target"),
                                 (absent, "#e0e3e8", "No model CF returned")):
        ax.bar(np.arange(5), values, bottom=bottom, width=.65, color=color, label=label)
        bottom += values
    ax.set_xticks(np.arange(5), LABELS)
    ax.set_yticks(range(5))
    ax.set_ylim(0, 4.3)
    ax.set_ylabel("Starting sequences (four per method)")
    ax.set_title("A  Model success and measured success", loc="left", fontweight="bold", pad=15)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -.18), fontsize=9)

    example = next(r for r in audit["returned_cf_evidence"]
                   if r["method"] == "alm_h_rollout" and r["source_index"] == 115108)
    interventions = {tuple(r["mask"]): r for r in example["interventions"]}
    source = interventions[(0, 0)]["sequence"]
    endpoint = interventions[(1, 1)]["sequence"]
    positions = [i for i, (a, b) in enumerate(zip(source, endpoint)) if a != b]
    assert len(positions) == 2, "The illustrated example must have exactly two edits"
    edits = [f"{source[i]}{i + 1}{endpoint[i]}" for i in positions]
    masks = ((0, 0), (1, 0), (0, 1), (1, 1))
    values = [interventions[m]["assay_fitness"] for m in ((0, 0), (1, 0), (0, 1), (1, 1))]
    ax = axes[1]
    ax.bar(range(4), values, color=["#b7bdc6", "#78a4c9", "#78a4c9", "#236b53"], width=.65)
    ax.axhline(.5, color="#695142", ls="--", lw=1.2)
    ax.text(3.42, .54, "Target 0.5", ha="right", va="bottom", fontsize=9)
    for i, value in enumerate(values):
        label_y = .64 if abs(value + .09 - .5) < .1 else value + .09
        ax.text(i, label_y, f"{value:.4f}", ha="center", fontsize=10)
    ax.set_xticks(range(4), [label + "\n" + "".join(interventions[mask]["sequence"][i] for i in SITES)
                            for label, mask in zip(["Source", *edits, "Both"], masks)])
    ax.set_ylim(0, 4.45)
    ax.set_ylabel("Measured WT-normalized selection fitness")
    ax.set_title("B  A measured component-edit audit", loc="left", fontweight="bold", pad=15)
    ax.text(0, -.22, "All four measurements are outside head training.\nSelected example; other residues are fixed.",
            transform=ax.transAxes, fontsize=9, va="top")
    fig.suptitle("CORAL with native ESMC-300M: exploratory GB1 pilot", x=.07, ha="left", fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=.07, right=.98, bottom=.29, top=.83, wspace=.3)
    fig.savefig(root / "biological_findings.png", dpi=180, facecolor="white")
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
