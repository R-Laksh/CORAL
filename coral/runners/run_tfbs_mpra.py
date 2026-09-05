"""Reproducible measured-neighbourhood experiment; no newly designed DNA output.

Run from the repository root:
  python -m coral.runners.run_tfbs_mpra --source-dir /path/to/TFBSs_grammar
"""
import argparse
from collections import Counter
import hashlib
import json
import platform
from pathlib import Path
import time
from urllib.request import urlopen

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from threadpoolctl import threadpool_limits

from coral.datasets.tfbs_mpra import (
    SOURCE_COMMIT, SOURCE_HASHES, load_orientation_neighbourhoods, predictor_features,
)
from coral.optimizers.distributional import (
    DistributionalCFOptimizer, FiniteEditGraph, backward_messages,
    factorized_projection, guidance_features, minimal_sufficient_clause, enumerate_clauses,
)


def fetch_source(directory):
    """Download a pinned public release, checking content hashes before use."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in SOURCE_HASHES.items():
        destination = directory / name
        if not destination.exists():
            url = f"https://raw.githubusercontent.com/IliasGeoSo/TFBSs_grammar/{SOURCE_COMMIT}/MPRA_library_data/{name}"
            temporary = destination.with_suffix(destination.suffix + ".part")
            with urlopen(url, timeout=120) as response, temporary.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            with temporary.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != expected:
                temporary.unlink()
                raise ValueError(f"download checksum mismatch: {name}")
            temporary.replace(destination)
    return directory


def build_problem(neighbourhood, scores, args):
    """Choose source/direction from a stable hash, never from assay outcomes."""
    code = int(neighbourhood.identifier, 16)
    start = code % 8
    direction = 1 if (code // 8) % 2 == 0 else -1
    graph = FiniteEditGraph.from_sequences(neighbourhood.states, neighbourhood.sequences,
                                         stay=args.stay, edge_cost=args.edge_cost)
    return DistributionalCFOptimizer(graph, scores, start,
        threshold=scores[start] + direction * args.gain, direction=direction,
        beta=args.beta, steps=args.steps)


def activity_metrics(actual, predicted):
    return {
        "pearson": float(pearsonr(actual, predicted).statistic),
        "spearman": float(spearmanr(actual, predicted).statistic),
        "r2": float(r2_score(actual, predicted)),
        "rmse_log2": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def train_activity_model(neighbourhoods, args):
    """Structured screening model with explicit TF-multiset holdouts.

    All responses in validation/test families are excluded from fitting. No
    automatically selected random row-level early-stopping split is used.
    This model is NOT claimed to be a foundation model or nucleotide CNN.
    """
    vocabulary = sorted({factor for n in neighbourhoods for factor in n.factors})
    features = predictor_features(neighbourhoods, vocabulary)
    expression = np.concatenate([n.expression for n in neighbourhoods])
    split = np.repeat([n.split(args.seed) for n in neighbourhoods], 8)
    model = HistGradientBoostingRegressor(
        max_iter=180, max_leaf_nodes=15, learning_rate=.07, l2_regularization=10,
        min_samples_leaf=30, categorical_features=list(range(7)),
        early_stopping=False, random_state=args.seed,
    )
    model.fit(features[split == "train"], expression[split == "train"])
    candidates = {"histogram": model.predict(features)}
    # Pairwise motif-orientation terms can generalise to unseen TF triplets.
    # Vocabulary fitting and coefficient fitting both exclude validation/test.
    dictionaries = []
    for n in neighbourhoods:
        for state in n.states:
            row = {f"background={n.background}": 1.}
            for i, (factor, orientation) in enumerate(zip(n.factors, state)):
                key = f"slot={i}:TF={factor}:strand={orientation}"
                row[key] = 1.
                row[f"bg={n.background}:{key}"] = 1.
                row[f"position:{key}"] = n.positions[i] / 200
            for i in range(3):
                for j in range(i + 1, 3):
                    key = f"pair={i},{j}:{n.factors[i]}:{state[i]}:{n.factors[j]}:{state[j]}"
                    row[key] = 1.
                    row[f"bg={n.background}:{key}"] = 1.
                    row[f"distance:{key}"] = (n.positions[j]-n.positions[i]) / 50
            dictionaries.append(row)
    vectorizer = DictVectorizer(dtype=np.float64)
    vectorizer.fit([row for row, s in zip(dictionaries, split) if s == "train"])
    pair_features = vectorizer.transform(dictionaries)
    for alpha in (10., 100.):
        ridge = Ridge(alpha=alpha, solver="lsqr", tol=1e-6)
        ridge.fit(pair_features[split == "train"], expression[split == "train"])
        candidates[f"pair_ridge_{alpha:g}"] = ridge.predict(pair_features)
    all_metrics = {}
    for name, prediction in candidates.items():
        metrics = {}
        for s in ("train", "validation", "test"):
            y = expression[split == s].reshape(-1, 8)
            p = prediction[split == s].reshape(-1, 8)
            metrics[s] = activity_metrics(y.ravel(), p.ravel())
            metrics[s]["within_neighbourhood_pearson"] = float(pearsonr(
                (y-y.mean(axis=1, keepdims=True)).ravel(),
                (p-p.mean(axis=1, keepdims=True)).ravel()).statistic)
            metrics[s]["within_neighbourhood_rmse"] = float(np.sqrt(np.mean(
                ((y-y.mean(axis=1, keepdims=True))-(p-p.mean(axis=1, keepdims=True)))**2)))
        all_metrics[name] = metrics
    chosen = min(all_metrics, key=lambda name: all_metrics[name]["validation"]["within_neighbourhood_rmse"])
    return candidates[chosen].reshape(-1, 8), {
        "selected": chosen, "selection": "minimum within-neighbourhood RMSE on validation TF families",
        "candidates": all_metrics, "selected_metrics": all_metrics[chosen],
    }, vocabulary


def train_log_h(neighbourhoods, prediction, args):
    eligible = sorted((i for i, n in enumerate(neighbourhoods) if n.split(args.seed) == "train"),
                      key=lambda i: neighbourhoods[i].identifier)[:args.h_train_cases]
    xs, ys, feasible_cases = [], [], 0
    for index in eligible:
        problem = build_problem(neighbourhoods[index], prediction[index], args)
        if not problem.feasible.any():
            continue
        feasible_cases += 1
        exact = backward_messages(problem.graph.kernel, problem.terminal, args.steps)
        xs.append(guidance_features(problem))
        ys.append(np.log(np.maximum(exact[:-1].ravel(), 1e-12)))
    if feasible_cases < 5:
        raise ValueError("too few training problems for learned h; increase --h-train-cases")
    model = HistGradientBoostingRegressor(
        max_iter=150, max_leaf_nodes=20, learning_rate=.07, l2_regularization=5,
        min_samples_leaf=30, early_stopping=False, random_state=args.seed,
    )
    model.fit(np.concatenate(xs), np.concatenate(ys))
    return model, {"candidate_training_neighbourhoods": len(eligible),
                   "feasible_training_neighbourhoods": feasible_cases,
                   "training_state_time_examples": sum(len(x) for x in xs)}


def finite_clause_catalogue(problem):
    """Exact local clause catalogue for assessing the blocking interface."""
    remaining = problem.feasible.copy()
    catalogue = []
    while remaining.any():
        candidates = np.flatnonzero(remaining)
        witness = min(candidates, key=lambda s: (problem.cost[s], int(s)))
        clause = minimal_sufficient_clause(problem.graph.states, witness, problem.feasible)
        region = remaining & clause.matches(problem.graph.states)
        if not region.any():
            raise RuntimeError("blocking must remove the witness")
        catalogue.append((clause, region))
        remaining &= ~clause.matches(problem.graph.states)
    return catalogue


def audit_endpoints(neighbourhood, problem, gain):
    """Audit data are only accessed after the target/guidance have been defined."""
    delta = problem.direction * (neighbourhood.replicate_log2 -
                                  neighbourhood.replicate_log2[problem.start])
    mean_delta = problem.direction * (neighbourhood.expression -
                                       neighbourhood.expression[problem.start])
    return {
        "mean_meets_gain": mean_delta >= gain,
        "all_replicates_positive": (delta > 0).all(axis=1),
        "all_replicates_meet_gain": (delta >= gain).all(axis=1),
        "observed_gain": mean_delta,
        "replicate_gain": delta,
    }


def bootstrap_cluster_mean(frame, metric, rng, draws=1000):
    """Resample TF-multiset families, retaining every case in each sampled family."""
    groups = frame.groupby("family")[metric].agg(["sum", "count"])
    if not len(groups) or groups["count"].sum() == 0:
        return {"mean": None, "ci95": [None, None]}
    values = groups.to_numpy()
    indices = rng.integers(0, len(groups), size=(draws, len(groups)))
    sampled = values[indices].sum(axis=1)
    valid = sampled[:, 1] > 0
    distribution = sampled[valid, 0] / sampled[valid, 1]
    return {"mean": float(frame[metric].mean()),
            "ci95": np.quantile(distribution, [.025, .975]).tolist()}


def run_experiment(args):
    started = time.perf_counter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source = fetch_source(args.source_dir) if args.download else Path(args.source_dir)
    neighbourhoods, provenance = load_orientation_neighbourhoods(source, args.min_tags)
    print(f"Loaded {len(neighbourhoods)} complete measured neighbourhoods", flush=True)
    prediction, predictor_metrics, vocabulary = train_activity_model(neighbourhoods, args)
    print("Selected activity model:", predictor_metrics["selected"], flush=True)
    print("Activity prediction on held-out families:", predictor_metrics["selected_metrics"]["test"], flush=True)
    fitted_h, training_h = train_log_h(neighbourhoods, prediction, args)
    print("Fitted h on training families:", training_h, flush=True)
    methods = ("reference", "myopic", "rollout", "learned", "exact", "defensive_rollout")
    selected = sorted((i for i, n in enumerate(neighbourhoods) if n.split(args.seed) == "test"),
                      key=lambda i: neighbourhoods[i].identifier)[:args.cases]
    all_cases, diagnostics, clause_records = [], [], []
    infeasible = 0
    for case_number, index in enumerate(selected):
        neighbourhood = neighbourhoods[index]
        problem = build_problem(neighbourhood, prediction[index], args)
        target, normalizer = problem.exact_target()
        audit = audit_endpoints(neighbourhood, problem, args.gain)
        identity = {"case": neighbourhood.identifier, "family": ",".join(neighbourhood.family),
                    "background": neighbourhood.background,
                    "source_id": neighbourhood.ids[problem.start], "direction": problem.direction}
        if normalizer == 0:
            infeasible += 1
            diagnostics.append({**identity, "model_feasible": False,
                "assay_mean_feasible": bool(audit["mean_meets_gain"].any()),
                "assay_strict_feasible": bool(audit["all_replicates_meet_gain"].any())})
            continue
        product = factorized_projection(problem.graph.states, target)
        catalogue = finite_clause_catalogue(problem)
        blocking_run = enumerate_clauses(problem, particles=args.particles,
            rng=np.random.default_rng([args.seed, int(neighbourhood.identifier[:8], 16)]))
        minimum = float(problem.cost[problem.feasible].min())
        region_mass = np.array([target[region].sum() for _, region in catalogue])
        diagnostics.append({**identity, "model_feasible": True,
            "assay_mean_feasible": bool(audit["mean_meets_gain"].any()),
            "assay_strict_feasible": bool(audit["all_replicates_meet_gain"].any()),
            "feasible_states": int(problem.feasible.sum()), "normalizer": normalizer,
            "minimum_feasible_base_edits": minimum,
            "exact_target_base_edits": float(target @ problem.cost),
            "factorized_invalid_mass": float(product[~problem.feasible].sum()),
            "exact_target_assay_mean": float(target @ audit["mean_meets_gain"]),
            "exact_target_assay_strict": float(target @ audit["all_replicates_meet_gain"]),
            "local_clause_count": len(catalogue),
            "blocking_run_clauses": len(blocking_run["clauses"]),
            "blocking_run_complete": blocking_run["status"] == "finite_support_exhausted",
            "expected_regions_3_independent_draws": float((1 - (1 - region_mass)**3).sum()),
            "regions_3_exact_blocking_rounds": min(3, len(catalogue)),
        })
        for number, (clause, region) in enumerate(catalogue):
            full_support = clause.matches(neighbourhood.states)
            clause_records.append({**identity, "clause_number": number,
                "scope_factors_in_order": ",".join(neighbourhood.factors),
                "scope_positions": list(neighbourhood.positions),
                "literals": [list(literal) for literal in clause.literals],
                "new_states_covered": int(region.sum()), "full_support": int(full_support.sum()),
                "assay_mean_precision": float(audit["mean_meets_gain"][full_support].mean()),
                "assay_strict_precision": float(audit["all_replicates_meet_gain"][full_support].mean()),
                "all_support_assay_mean_pass": bool(audit["mean_meets_gain"][full_support].all()),
                "all_support_assay_strict_pass": bool(audit["all_replicates_meet_gain"][full_support].all()),
                "minimum_measured_gain": float(audit["observed_gain"][full_support].min()),
                "minimum_paired_replicate_gain": float(audit["replicate_gain"][full_support].min()),
                "assayed_support_ids": [str(i) for i in np.asarray(neighbourhood.ids)[full_support]],
                "status": "finite-model sufficient clause; experimentally audited, not global ILP",
            })
        for method_number, method in enumerate(methods):
            rows = []
            for repeat in range(args.repeats):
                rng = np.random.default_rng(np.random.SeedSequence(
                    [args.seed, int(neighbourhood.identifier[:8], 16), method_number, repeat]))
                before = time.perf_counter()
                guide = problem.guidance(method, rng, rollouts=args.rollouts, learned=fitted_h)
                result = problem.sample(guide, particles=args.particles, rng=rng)
                elapsed = time.perf_counter() - before
                histogram = result.endpoint_probabilities
                archive = list(result.archive)
                row = {
                    "smc_survived": float(not result.failed), "found_cf": float(bool(archive)),
                    "terminal_tv": float(.5 * abs(histogram - target).sum()) if not result.failed else 1.,
                    "ess_fraction": result.effective_sample_size / args.particles,
                    "normalizer_relative_estimate": float(np.exp(result.log_normalizer) / normalizer),
                    "unique_cf": float(len(archive)), "seconds": elapsed,
                    "best_base_edit_regret": min(problem.cost[archive]) - minimum if archive else np.nan,
                    "endpoint_mean_meets_gain": float(histogram @ audit["mean_meets_gain"]) if not result.failed else np.nan,
                    "endpoint_all_replicates_positive": float(histogram @ audit["all_replicates_positive"]) if not result.failed else np.nan,
                    "endpoint_all_replicates_meet_gain": float(histogram @ audit["all_replicates_meet_gain"]) if not result.failed else np.nan,
                    "endpoint_base_edits": float(histogram @ problem.cost) if not result.failed else np.nan,
                }
                rows.append(row)
            all_cases.append({**identity, "method": method,
                              **pd.DataFrame(rows).mean().to_dict()})
        if (case_number + 1) % 50 == 0:
            print(f"Evaluated {case_number+1}/{len(selected)} test neighbourhoods", flush=True)
    cases = pd.DataFrame(all_cases)
    diagnostic = pd.DataFrame(diagnostics)
    if cases.empty:
        raise ValueError("no model-feasible test cases; inspect the predictor and target")
    cases.to_csv(output / "case_metrics.csv", index=False)
    diagnostic.to_csv(output / "neighbourhood_diagnostics.csv", index=False)
    (output / "local_clauses.json").write_text(json.dumps(clause_records, indent=2) + "\n")
    selected_metrics = ["smc_survived", "found_cf", "terminal_tv", "ess_fraction",
                        "unique_cf", "best_base_edit_regret", "endpoint_mean_meets_gain",
                        "endpoint_all_replicates_positive", "endpoint_all_replicates_meet_gain",
                        "endpoint_base_edits", "normalizer_relative_estimate", "seconds"]
    summary = {
        "status": "exploratory finite measured-state prototype; not a Ledidi/CORAL superiority claim",
        "data": provenance, "configuration": vars(args), "factor_vocabulary": vocabulary,
        "split_neighbourhood_counts": dict(Counter(n.split(args.seed) for n in neighbourhoods)),
        "split_family_counts": {s: len({n.family for n in neighbourhoods if n.split(args.seed) == s})
                                for s in ("train", "validation", "test")},
        "activity_predictor": {"type": "validation-selected structured screening model",
                               "metrics": predictor_metrics},
        "learned_h_training": training_h,
        "test": {"neighbourhoods_selected_without_assay_selection": len(selected),
                 "model_infeasible": infeasible, "model_feasible": len(selected)-infeasible,
                 "mean_factorized_invalid_mass": float(diagnostic.factorized_invalid_mass.mean()),
                 "mean_exact_target_assay_mean": float(diagnostic.exact_target_assay_mean.mean()),
                 "mean_exact_target_assay_strict": float(diagnostic.exact_target_assay_strict.mean()),
                 "mean_local_clauses": float(diagnostic.local_clause_count.mean()),
                 "blocking_run_completeness": float(diagnostic.blocking_run_complete.mean()),
                 "expected_regions_3_independent_draws": float(diagnostic.expected_regions_3_independent_draws.mean()),
                 "regions_3_exact_blocking_rounds": float(diagnostic.regions_3_exact_blocking_rounds.mean())},
        "methods": {}, "total_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "interpretation": [
            "Exact h and direct enumeration are finite-state references, not scalable optimiser baselines.",
            "All methods receive the same eight cached activity predictions; timing excludes scoring and training.",
            "SMC survival and model-valid endpoints do not imply experimental validity.",
            "Assay audit metrics are conditional on SMC survival; failures are reported separately.",
            "Intervals resample TF-multiset families after averaging repeats; they are exploratory.",
            "The reference is an edit-distance kernel, not a learned biological prior.",
            "Only orientation interventions in fixed experimental contexts are validated.",
            "Local clauses are scoped finite-model summaries, not causal TF mechanisms or trained ILP outputs.",
        ],
    }
    for method in methods:
        subset = cases[cases.method == method]
        summary["methods"][method] = {metric: bootstrap_cluster_mean(
            subset, metric, np.random.default_rng(args.seed)) for metric in selected_metrics}
    summary["paired_differences_vs_reference"] = {}
    reference = cases[cases.method == "reference"].set_index("case")
    for method in methods[1:]:
        subset = cases[cases.method == method].set_index("case")
        differences = subset[selected_metrics] - reference[selected_metrics]
        differences["family"] = subset.family
        summary["paired_differences_vs_reference"][method] = {
            metric: bootstrap_cluster_mean(differences, metric, np.random.default_rng(args.seed))
            for metric in selected_metrics if metric != "seconds"
        }
    summary["clause_audit"] = {
        "clauses": len(clause_records),
        "all_support_assay_mean_pass": sum(r["all_support_assay_mean_pass"] for r in clause_records),
        "all_support_assay_strict_pass": sum(r["all_support_assay_strict_pass"] for r in clause_records),
        "interpretation": "Observed support checks within fixed contexts; not confidence intervals or transferable causal mechanisms.",
    }
    missed = diagnostic[~diagnostic.model_feasible]
    summary["test"]["model_infeasible_but_assay_mean_feasible"] = int(missed.assay_mean_feasible.sum())
    summary["test"]["model_infeasible_but_assay_strict_feasible"] = int(missed.assay_strict_feasible.sum())
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    make_plot(summary, output)
    print(json.dumps({"test": summary["test"], "methods": {
        name: {key: value["mean"] for key, value in metrics.items()}
        for name, metrics in summary["methods"].items()}}, indent=2), flush=True)


def make_plot(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = list(summary["methods"])
    metrics = [("smc_survived", "Particle runs retaining terminal mass"),
               ("terminal_tv", "Distance from exact CF distribution (lower is better)"),
               ("endpoint_all_replicates_meet_gain", "Endpoints meeting gain in all 3 assay replicates")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (key, title) in zip(axes, metrics):
        values = [summary["methods"][method][key]["mean"] for method in methods]
        intervals = np.array([summary["methods"][method][key]["ci95"] for method in methods])
        labels = [m.replace("defensive_rollout", "defensive") for m in methods]
        ax.bar(labels, values, color=["#8e99a4", "#8aa6c1", "#53929b", "#b58c58", "#665888", "#465a70"])
        errors = np.maximum(0, np.array([np.array(values)-intervals[:, 0], intervals[:, 1]-values]))
        ax.errorbar(range(len(methods)), values, yerr=errors, fmt="none", color="#222222", capsize=3)
        ax.set_title(title, fontsize=10, wrap=True)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=30)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("CORAL prototype: measured HepG2 orientation neighbourhoods", fontsize=13)
    fig.text(.5, .02, "Exploratory test on held-out TF multisets; exact h is a finite-state reference. Error bars: family bootstrap.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .95))
    fig.savefig(output / "benchmark.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="research_data/tfbs_mpra")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", default="research_results/tfbs_mpra_gain010")
    parser.add_argument("--cases", type=int, default=300)
    parser.add_argument("--h-train-cases", type=int, default=600)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--particles", type=int, default=16)
    parser.add_argument("--rollouts", type=int, default=16)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--gain", type=float, default=.1)
    parser.add_argument("--beta", type=float, default=.1)
    parser.add_argument("--stay", type=float, default=.65)
    parser.add_argument("--edge-cost", type=float, default=.03)
    parser.add_argument("--min-tags", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    if min(args.cases, args.h_train_cases, args.repeats, args.particles, args.rollouts) < 1:
        parser.error("case, repeat, particle and rollout counts must be positive")
    if args.steps < 3:
        parser.error("use at least three steps so the whole orientation cube is reachable")
    if args.gain <= 0:
        parser.error("gain must be positive")
    with threadpool_limits(limits=2):
        run_experiment(args)


if __name__ == "__main__":
    main()
