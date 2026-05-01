"""Runner script for SeqGra counterfactual experiments across multiple grammars."""
import argparse
import inspect
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from coral.datasets.seqgra_dataset import SeqgraDataset, decode_idx, decode_idx_batch
from coral.grammar import GrammarChecker
from coral.models.seqgra_models import CNN1D
from coral.optimizers.seqgra import SeqgraCORALOptimizer, LedidiSeqgraCFOptimizer
from coral.utils import set_seed

torch.set_float32_matmul_precision('high')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest.csv")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["batch", "sequential"], default="batch")
    parser.add_argument("--n_samples", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--n_starts", type=int, default=1)
    parser.add_argument("--method", type=str, choices=["coral", "ledidi"], default="coral")
    parser.add_argument("--rho", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--mc_samples", type=int, default=64)
    parser.add_argument("--robust_mode", choices=["mean", "worst"], default="mean")
    parser.add_argument("--ledidi_l", type=float, default=0.01)
    parser.add_argument("--ledidi_tau", type=float, default=1.0)
    parser.add_argument("--ledidi_batch_size", type=int, default=256)
    parser.add_argument("--ledidi_early_stopping", type=int, default=500)
    parser.add_argument("--ledidi_output_loss", type=str, default="margin",
                        choices=["margin", "bce", "mse", "l1"])
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} | Mode: {args.mode.upper()} | Method: {args.method.upper()}")

    df_manifest = pd.read_csv(args.manifest)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_outdir = Path(args.outdir) / f"run_{timestamp}_{args.mode.upper()}"
    run_outdir.mkdir(parents=True, exist_ok=True)

    params = vars(args).copy()
    params["timestamp"] = timestamp
    params["device"] = device
    if args.method != "ledidi":
        sig = inspect.signature(SeqgraCORALOptimizer.generate_cf_batch)
        params["optimizer_defaults"] = {
            k: v.default for k, v in sig.parameters.items()
            if v.default is not inspect.Parameter.empty and k not in ("self", "x_onehot")
        }
    with open(run_outdir / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    all_metrics = []

    for _, row in df_manifest.iterrows():
        dataset_name = row["dataset_name"]
        grammar = row["grammar"]
        data_root = Path(row["data_root"])
        model_path = row["model_path"]
        label_pos = row["label_pos"]

        print(f"\n--- Processing {dataset_name} ({grammar}) ---")

        dataset = SeqgraDataset(data_root / "test.txt", label_pos=label_pos)

        if args.n_samples is not None and args.n_samples < len(dataset):
            pos_idx = np.where(dataset.y == 1)[0]
            neg_idx = np.where(dataset.y == 0)[0]
            n_pos = min(args.n_samples // 2, len(pos_idx))
            n_neg = min(args.n_samples - n_pos, len(neg_idx))
            np.random.shuffle(pos_idx)
            np.random.shuffle(neg_idx)
            chosen_idx = np.concatenate([pos_idx[:n_pos], neg_idx[:n_neg]])
            np.random.shuffle(chosen_idx)
            subset = Subset(dataset, chosen_idx)
        else:
            subset = dataset

        loader = DataLoader(
            subset, batch_size=args.batch_size, shuffle=False, num_workers=4,
            pin_memory=(device == "cuda"),
            persistent_workers=(device == "cuda"),
        )

        L = dataset.x.shape[-1]
        model = CNN1D(L)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device).eval()

        if args.method == "ledidi":
            cf_opt = LedidiSeqgraCFOptimizer(model, device=device)
        else:
            cf_opt = SeqgraCORALOptimizer(model, device=device)
        checker = GrammarChecker(grammar)

        results_df = []

        for batch_x, batch_y in tqdm(loader, desc=f"Evaluating {dataset_name}"):
            batch_x = batch_x.to(device, non_blocking=True)
            B = batch_x.shape[0]
            orig_seqs = decode_idx_batch(batch_x.argmax(dim=1).cpu())

            if args.mode == "sequential":
                for i in range(B):
                    x1 = batch_x[i]
                    y_orig = float(batch_y[i].item())
                    res = cf_opt.generate_cf(
                        x_onehot=x1, y_orig=y_orig, motif_mask=None,
                        steps=args.steps, verbose=False, margin=args.margin,
                        rho=args.rho, mc_samples=args.mc_samples, robust_mode=args.robust_mode,
                    )
                    o_seq = orig_seqs[i]
                    c_seq = decode_idx(res["cf_idx"]) if res["success"] else None
                    o_dpos, o_dneg = checker.get_distances(o_seq)
                    c_dpos, c_dneg = checker.get_distances(c_seq) if res["success"] else (None, None)
                    results_df.append({
                        "orig_seq": o_seq, "cf_seq": c_seq,
                        "orig_pred": res["orig_pred"], "target_label": res["target_label"],
                        "success": res["success"],
                        "edit_distance": res["edit_distance"] if res["success"] else None,
                        "orig_dist_pos": o_dpos, "orig_dist_neg": o_dneg,
                        "cf_dist_pos": c_dpos, "cf_dist_neg": c_dneg,
                    })
            else:
                if args.method == "ledidi":
                    cf_res = cf_opt.generate_cf_batch(
                        x_onehot=batch_x, l_input=args.ledidi_l, tau=args.ledidi_tau,
                        batch_size=args.ledidi_batch_size, max_iter=args.steps,
                        early_stopping_iter=args.ledidi_early_stopping,
                        verbose=False, margin=args.margin, output_loss=args.ledidi_output_loss,
                    )
                elif args.n_starts > 1:
                    cf_res = cf_opt.generate_cf_batch_multistart(
                        x_onehot=batch_x, steps=args.steps, margin=args.margin,
                        n_starts=args.n_starts, base_seed=args.seed,
                    )
                else:
                    cf_res = cf_opt.generate_cf_batch(
                        x_onehot=batch_x, steps=args.steps, margin=args.margin,
                        k_gumbel=500, rho=args.rho, mc_samples=args.mc_samples,
                        robust_mode=args.robust_mode, alpha=args.alpha,
                    )

                orig_seqs = decode_idx_batch(cf_res["orig_idx"])
                cf_seqs = decode_idx_batch(cf_res["cf_idx"])
                success = cf_res["success"].numpy()
                target_labels = cf_res["target_label"].numpy()
                orig_preds = cf_res["orig_pred"].numpy()
                edit_dists = cf_res["edit_distance"].numpy()

                for i in range(B):
                    o_seq = orig_seqs[i]
                    c_seq = cf_seqs[i] if success[i] else None
                    o_dpos, o_dneg = checker.get_distances(o_seq)
                    c_dpos, c_dneg = checker.get_distances(c_seq) if success[i] else (None, None)
                    results_df.append({
                        "orig_seq": o_seq, "cf_seq": c_seq,
                        "orig_pred": orig_preds[i], "target_label": target_labels[i],
                        "success": success[i],
                        "edit_distance": edit_dists[i] if success[i] else None,
                        "orig_dist_pos": o_dpos, "orig_dist_neg": o_dneg,
                        "cf_dist_pos": c_dpos, "cf_dist_neg": c_dneg,
                    })

        df_res = pd.DataFrame(results_df)
        df_res.to_csv(run_outdir / f"{dataset_name}_cf_results.csv", index=False)

        pos2neg = df_res[(df_res["orig_pred"] == 1) & (df_res["target_label"] == 0)]
        neg2pos = df_res[(df_res["orig_pred"] == 0) & (df_res["target_label"] == 1)]

        def safe_mean(df, col):
            return float(df[col].mean()) if len(df) > 0 else 0.0

        metrics = {
            "dataset": str(dataset_name), "grammar": str(grammar),
            "pos2neg_total": int(len(pos2neg)),
            "pos2neg_success_rate": float(pos2neg["success"].mean()) if len(pos2neg) > 0 else 0.0,
            "pos2neg_mean_edits": safe_mean(pos2neg[pos2neg["success"]], "edit_distance"),
            "pos2neg_mean_dist_to_neg_grammar": safe_mean(pos2neg[pos2neg["success"]], "cf_dist_neg"),
            "neg2pos_total": int(len(neg2pos)),
            "neg2pos_success_rate": float(neg2pos["success"].mean()) if len(neg2pos) > 0 else 0.0,
            "neg2pos_mean_edits": safe_mean(neg2pos[neg2pos["success"]], "edit_distance"),
            "neg2pos_mean_dist_to_pos_grammar": safe_mean(neg2pos[neg2pos["success"]], "cf_dist_pos"),
        }
        all_metrics.append(metrics)
        print(json.dumps(metrics, indent=4))

    df_metrics = pd.DataFrame(all_metrics)
    summary_path = run_outdir / "summary_metrics.csv"
    df_metrics.to_csv(summary_path, index=False)
    print(f"\nEvaluation complete. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
