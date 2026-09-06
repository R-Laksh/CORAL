"""GB1 counterfactual benchmark with a frozen ESM2 protein language model.

The functional head is trained only on measured WT/single/double mutants. No
imputed GB1 fitness values are used for edit regret or biological validation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from coral.datasets.gb1 import (
    AA_ORDER, EDIT_POSITIONS, WT_GENOTYPE, genotype_from_ids, genotype_hamming,
    genotype_to_full_sequence, hamming_shell, load_gb1_processed, sequence_to_ids,
)
from coral.models.biological import FairESMSoftSequenceRegressor, ThresholdConstraint
from coral.optimizers.alm_h import ALMHTwistedSearch, ALMState, SearchConfig
from coral.optimizers.st_alm import STALMConfig, STGumbelALMSearch


def _jsonable(x):
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.floating): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    raise TypeError(type(x).__name__)


def load_fair_esm(model_name: str, device: str):
    import esm
    loader = getattr(esm.pretrained, model_name, None)
    if loader is None:
        raise ValueError(f"fair-esm has no pretrained loader named {model_name!r}")
    model, alphabet = loader()
    return model.eval().to(device), alphabet


def embed_full_sequences(model, alphabet, sequences, batch_size: int, device: str):
    converter = alphabet.get_batch_converter()
    layer = int(model.num_layers)
    chunks = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start:start + batch_size]
        _, _, tokens = converter([(str(start+i), seq) for i, seq in enumerate(batch)])
        with torch.inference_mode():
            out = model(tokens.to(device), repr_layers=[layer], return_contacts=False)
            rep = out["representations"][layer][:, 1:1+len(batch[0])].mean(dim=1)
        chunks.append(rep.cpu())
    return torch.cat(chunks, dim=0) if chunks else torch.empty((0, int(model.embed_dim)))


def fit_ridge_head(emb: np.ndarray, y: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y)); split = max(1, int(round(0.8 * len(y))))
    tr, va = idx[:split], idx[split:]
    trials, best_alpha, best_rmse = [], None, float("inf")
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        reg = Ridge(alpha=alpha).fit(emb[tr], y[tr])
        pred = reg.predict(emb[va])
        rmse = float(mean_squared_error(y[va], pred) ** 0.5)
        trials.append({"alpha": alpha, "rmse": rmse})
        if rmse < best_rmse: best_rmse, best_alpha = rmse, alpha
    probe = Ridge(alpha=best_alpha).fit(emb[tr], y[tr])
    val_pred = probe.predict(emb[va])
    final = Ridge(alpha=best_alpha).fit(emb, y)
    metrics = {
        "best_alpha": float(best_alpha), "validation_rmse": best_rmse,
        "validation_pearson": float(pearsonr(y[va], val_pred).statistic),
        "validation_spearman": float(spearmanr(y[va], val_pred).statistic),
        "alpha_trials": trials,
    }
    return final, metrics


def ridge_to_torch_head(reg: Ridge):
    head = torch.nn.Linear(reg.coef_.shape[0], 1)
    with torch.no_grad():
        head.weight.copy_(torch.as_tensor(reg.coef_[None, :], dtype=torch.float32))
        head.bias.copy_(torch.as_tensor([reg.intercept_], dtype=torch.float32))
    return head


def discrete_scores(model, alphabet, head, genotypes, batch_size: int, device: str):
    emb = embed_full_sequences(
        model, alphabet, [genotype_to_full_sequence(g) for g in genotypes], batch_size, device)
    with torch.inference_mode():
        return head(emb.to(device)).reshape(-1).cpu().numpy()


def experimental_optimum(source: str, high_genotypes: np.ndarray):
    if not len(high_genotypes): return None
    src = np.asarray(list(source)); arr = np.asarray([list(g) for g in high_genotypes])
    return int((arr != src).sum(axis=1).min())


def select_tasks(df, model, alphabet, head, target, n_tasks, candidate_scan,
                 batch_size, device, seed):
    high = df.loc[df.fitness >= target, "genotype"].to_numpy()
    candidates = []
    for row in df.itertuples(index=False):
        if row.fitness >= target or genotype_hamming(row.genotype, WT_GENOTYPE) < 3:
            continue
        opt = experimental_optimum(row.genotype, high)
        if opt == 2: candidates.append((row.genotype, float(row.fitness), opt))
    rng = random.Random(seed); rng.shuffle(candidates)
    selected, cache = [], {}

    def score_many(gs):
        missing = [g for g in gs if g not in cache]
        if missing:
            vals = discrete_scores(model, alphabet, head, missing, batch_size, device)
            cache.update({g: float(v) for g, v in zip(missing, vals)})
        return np.asarray([cache[g] for g in gs], dtype=float)

    for source, exp_fit, exp_opt in candidates[:candidate_scan]:
        p0 = score_many([source])[0]
        if p0 >= target: continue
        shell1 = hamming_shell(source, 1); p1 = score_many(shell1)
        if np.any(p1 >= target):
            model_opt = 1
        else:
            shell2 = hamming_shell(source, 2); p2 = score_many(shell2)
            if not np.any(p2 >= target): continue
            model_opt = 2
        selected.append({
            "source": source, "source_experimental_fitness": exp_fit,
            "source_model_fitness": float(p0), "experimental_optimum": int(exp_opt),
            "model_optimum": int(model_opt), "gateway": bool(model_opt == 2),
            "best_single_model_fitness": float(np.max(p1)),
        })
        if len(selected) >= n_tasks: break
    return selected


def soft_adapter_parity(model, alphabet, head, device):
    aa_ids = [int(alphabet.get_idx(a)) for a in AA_ORDER]
    soft = FairESMSoftSequenceRegressor(model, aa_ids, head).to(device).eval()
    seq = genotype_to_full_sequence(WT_GENOTYPE)
    ids = torch.as_tensor(sequence_to_ids(seq), dtype=torch.long, device=device)
    x = F.one_hot(ids, num_classes=len(AA_ORDER)).float()[None, :]
    with torch.inference_mode():
        s_soft = float(soft(x).item())
        s_discrete = float(head(embed_full_sequences(model, alphabet, [seq], 1, device).to(device)).item())
    return soft, abs(s_soft-s_discrete), s_soft, s_discrete


def run_optimizer_task(method, source, target, soft_score_model, rho, seed, device):
    constraint = ThresholdConstraint(soft_score_model, target, "increase")
    source_full = genotype_to_full_sequence(source)
    x0 = torch.as_tensor(sequence_to_ids(source_full), dtype=torch.long, device=device)
    editable = torch.zeros(len(source_full), dtype=torch.bool, device=device)
    editable[list(EDIT_POSITIONS)] = True
    state = ALMState(rho=rho, rho_min=rho)
    t0 = time.perf_counter()
    if method == "st_alm":
        cfg = STALMConfig(steps=100, lr=0.25, tau_start=1.5, tau_end=0.2,
                           mc_samples=2, original_logit_bias=2.0,
                           discrete_eval_every=2, seed=seed)
        result = STGumbelALMSearch(constraint, len(AA_ORDER), cfg, editable).run(x0, state)
    elif method in {"particle_local", "particle_h"}:
        cfg = SearchConfig(
            particles=4, episodes=2, horizon=2, proposal_width=8, lookahead_width=4,
            guidance_strength=0.0 if method == "particle_local" else 4.0,
            guidance_estimator="shared_rollout", gradient_proposal_strength=0.25,
            move_temperature=0.75, energy_temperature=1.0, seed=seed)
        result = ALMHTwistedSearch(constraint, len(AA_ORDER), cfg, editable).run(x0, state)
    else:
        raise ValueError(method)
    seconds = time.perf_counter() - t0
    candidate = None if result.best_ids is None else genotype_from_ids(result.best_ids.cpu().numpy())
    return result, candidate, seconds


def aggregate_runs(rows):
    out = {}
    for method in sorted({r["method"] for r in rows}):
        rr = [r for r in rows if r["method"] == method]
        edits = [r["edits"] for r in rr if r["edits"] is not None]
        regrets = [r["model_edit_regret"] for r in rr if r["model_edit_regret"] is not None]
        exp = [r["experimental_success"] for r in rr if r["experimental_success"] is not None]
        out[method] = {
            "n": len(rr), "model_success_rate": float(np.mean([r["model_success"] for r in rr])),
            "mean_edits_on_success": None if not edits else float(np.mean(edits)),
            "mean_model_edit_regret": None if not regrets else float(np.mean(regrets)),
            "experimental_success_rate_when_measured": None if not exp else float(np.mean(exp)),
            "measured_endpoint_fraction": float(np.mean([r["candidate_measured"] for r in rr])),
            "mean_forward_evals": float(np.mean([r["forward_evals"] for r in rr])),
            "mean_backward_evals": float(np.mean([r["backward_evals"] for r in rr])),
            "mean_seconds": float(np.mean([r["seconds"] for r in rr])),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="esm2_t6_8M_UR50D"); ap.add_argument("--device", default="cpu")
    ap.add_argument("--embedding-batch-size", type=int, default=64)
    ap.add_argument("--high-order-eval", type=int, default=3000)
    ap.add_argument("--target-quantile", type=float, default=0.80)
    ap.add_argument("--tasks", type=int, default=4); ap.add_argument("--candidate-scan", type=int, default=12)
    ap.add_argument("--repeats", type=int, default=2); ap.add_argument("--rho", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=17); args = ap.parse_args()

    outdir = Path(args.output); outdir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    df = load_gb1_processed(args.data)
    if len(df) < 140_000: raise RuntimeError(f"Expected measured GB1 landscape; found {len(df)} rows")
    target = float(df.fitness.quantile(args.target_quantile))

    t0 = time.perf_counter(); model, alphabet = load_fair_esm(args.model, args.device)
    load_seconds = time.perf_counter() - t0
    low = df[df.genotype.map(lambda g: genotype_hamming(g, WT_GENOTYPE) <= 2)].copy()
    t1 = time.perf_counter()
    low_emb = embed_full_sequences(model, alphabet,
        [genotype_to_full_sequence(g) for g in low.genotype], args.embedding_batch_size, args.device).numpy()
    embedding_seconds = time.perf_counter() - t1
    ridge, head_metrics = fit_ridge_head(low_emb, low.fitness.to_numpy(), args.seed)
    head = ridge_to_torch_head(ridge).to(args.device).eval()

    high_order = df[df.genotype.map(lambda g: genotype_hamming(g, WT_GENOTYPE) >= 3)].copy()
    high_order = high_order.sample(n=min(args.high_order_eval, len(high_order)), random_state=args.seed)
    high_pred = discrete_scores(model, alphabet, head, high_order.genotype.tolist(),
                                args.embedding_batch_size, args.device)
    head_metrics.update({
        "train_low_order_n": int(len(low)), "high_order_test_n": int(len(high_order)),
        "high_order_rmse": float(mean_squared_error(high_order.fitness, high_pred) ** 0.5),
        "high_order_pearson": float(pearsonr(high_order.fitness, high_pred).statistic),
        "high_order_spearman": float(spearmanr(high_order.fitness, high_pred).statistic),
    })

    soft_model, parity_diff, soft_wt, discrete_wt = soft_adapter_parity(model, alphabet, head, args.device)
    if parity_diff > 2e-4: raise RuntimeError(f"Soft fair-ESM adapter parity failed: {parity_diff}")
    tasks = select_tasks(df, model, alphabet, head, target, args.tasks, args.candidate_scan,
                         args.embedding_batch_size, args.device, args.seed)
    if not tasks:
        raise RuntimeError("No source with an exactly verified <=2-edit model path; change target/candidate scan.")

    fitness_map = dict(zip(df.genotype, df.fitness)); rows = []
    methods = ["particle_local", "particle_h", "st_alm"]
    for task_i, task in enumerate(tasks):
        for method in methods:
            for rep in range(args.repeats):
                seed = args.seed + 1000*task_i + 100*rep + methods.index(method)
                result, candidate, seconds = run_optimizer_task(
                    method, task["source"], target, soft_model, args.rho, seed, args.device)
                measured = candidate in fitness_map if candidate is not None else False
                exp_fit = float(fitness_map[candidate]) if measured else None
                exp_success = None if exp_fit is None else bool(exp_fit >= target)
                edits = result.best_hamming
                rows.append({
                    "task": task_i, "source": task["source"], "method": method, "repeat": rep,
                    "candidate": candidate, "model_success": bool(result.feasible_found), "edits": edits,
                    "model_optimum": task["model_optimum"],
                    "model_edit_regret": None if edits is None else int(edits-task["model_optimum"]),
                    "experimental_optimum": task["experimental_optimum"], "candidate_measured": bool(measured),
                    "experimental_fitness": exp_fit, "experimental_success": exp_success,
                    "hard_constraint": result.best_constraint, "forward_evals": int(result.model_forward_evals),
                    "backward_evals": int(result.model_backward_evals), "seconds": float(seconds),
                })

    pd.DataFrame(rows).to_csv(outdir / "runs.csv", index=False)
    summary = {
        "git_sha": os.environ.get("GITHUB_SHA"),
        "dataset": {"rows_measured": int(len(df)), "unique_genotypes": int(df.genotype.nunique()),
                    "wild_type": WT_GENOTYPE, "target_quantile": args.target_quantile,
                    "target_log_fitness": target, "imputed_fitness_used": False},
        "model": {"name": args.model, "load_seconds": load_seconds,
                  "low_order_embedding_seconds": embedding_seconds,
                  "soft_adapter_abs_parity_error_wt": parity_diff,
                  "soft_adapter_wt_score": soft_wt, "discrete_wt_score": discrete_wt},
        "head": head_metrics, "tasks": tasks,
        "optimizer": {"rho_initial": args.rho, "methods": aggregate_runs(rows)},
        "versions": {"python": os.sys.version, "torch": torch.__version__},
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=_jsonable))
    print(json.dumps(summary, indent=2, default=_jsonable))


if __name__ == "__main__": main()
