"""Runner script for GC-content counterfactual experiments."""
import argparse
import inspect
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from coral.models.gc_model import GenomicGCModel, train_head
from coral.optimizers.gc import LearnedGCOptimizer, LedidiGCOptimizer
from coral.datasets.gc_dataset import generate_benchmark_sequences
from coral.utils import set_seed, get_theoretical_min_edits, classify_edits

torch.set_float32_matmul_precision('high')

def main():
    parser = argparse.ArgumentParser(description="GC Content Counterfactual Optimizer")
    parser.add_argument("--outdir", type=str, default="gc_cf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--n_sequences", type=int, default=16)
    parser.add_argument("--seq_length", type=int, default=48,
                        help="Sequence length in bases (must be multiple of 6)")
    parser.add_argument("--target_gc", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--method", choices=["coral", "ledidi"], default="coral")
    parser.add_argument("--mode", choices=["batch", "sequential"], default="batch")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=10.0)
    parser.add_argument("--mc_samples", type=int, default=64)
    parser.add_argument("--robust_mode", choices=["mean", "worst"], default="mean")
    parser.add_argument("--ledidi_l", type=float, default=0.001)
    parser.add_argument("--ledidi_tau", type=float, default=1.0)
    parser.add_argument("--ledidi_batch_size", type=int, default=16)
    parser.add_argument("--ledidi_early_stopping", type=int, default=2000)
    args = parser.parse_args()

    assert args.seq_length % 6 == 0, f"seq_length must be multiple of 6, got {args.seq_length}"

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} | Method: {args.method.upper()} | Mode: {args.mode.upper()}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_outdir = Path(args.outdir) / f"run_{timestamp}_{args.method.upper()}"
    run_outdir.mkdir(parents=True, exist_ok=True)

    torch.cuda.empty_cache()
    if args.load_model:
        print(f"Loading pre-trained head from {args.load_model}")
        tokenizer = AutoTokenizer.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True
        )
        gc_model = GenomicGCModel(device=device)
        gc_model.head.load_state_dict(torch.load(args.load_model, map_location=device))
        print("Head weights loaded.")
    else:
        gc_model, tokenizer = train_head()

    if args.save_model:
        model_path = run_outdir / "gc_head.pt"
        torch.save(gc_model.head.state_dict(), model_path)
        print(f"Saved GC head weights to {model_path}")

    params = vars(args).copy()
    params["timestamp"] = timestamp
    params["device"] = device
    if args.method == "coral":
        sig = inspect.signature(LearnedGCOptimizer.generate_cf_batch)
        params["optimizer_defaults"] = {
            k: v.default for k, v in sig.parameters.items()
            if v.default is not inspect.Parameter.empty and k not in ("self", "sequences", "target_gc")
        }
    with open(run_outdir / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    target_gc = args.target_gc
    sequences = generate_benchmark_sequences(args.n_sequences, args.seq_length, seed=args.seed, target_gc=target_gc)

    if args.method == "coral":
        optimizer = LearnedGCOptimizer(gc_model, tokenizer, device=device)
    else:
        optimizer = LedidiGCOptimizer(gc_model, tokenizer, device=device)

    eligible = []
    for i, seq in enumerate(sequences):
        orig_gc = (seq.count("G") + seq.count("C")) / len(seq)
        min_edits = get_theoretical_min_edits(seq, orig_gc, target_gc)
        eligible.append((i, seq, orig_gc, min_edits))

    print(f"Eligible sequences: {len(eligible)}")
    records = []
    chunks = [eligible[i:i + args.batch_size] for i in range(0, len(eligible), args.batch_size)]
    processed = 0

    for chunk in tqdm(chunks, desc=f"{args.method.upper()} ({args.mode})"):
        chunk_seqs = [s for _, s, _, _ in chunk]
        results_for_chunk = []

        if args.mode == "batch":
            if args.method == "coral":
                batch_res = optimizer.generate_cf_batch(
                    chunk_seqs, target_gc, steps=args.steps, margin=args.margin,
                    rho=args.rho, mc_samples=args.mc_samples, robust_mode=args.robust_mode, lr=args.lr,
                )
                for j in range(len(chunk_seqs)):
                    results_for_chunk.append((
                        batch_res["cf_seqs"][j], float(batch_res["cf_true_gc"][j]),
                        float(batch_res["cf_pred_gc"][j]), float(batch_res["edit_distance"][j]),
                        bool(batch_res["success"][j]), bool(batch_res["pred_success"][j]),
                    ))
            else:
                batch_res = optimizer.generate_cf_batch(
                    chunk_seqs, target_gc, l_input=args.ledidi_l, tau=args.ledidi_tau,
                    batch_size=args.ledidi_batch_size, max_iter=args.steps,
                    early_stopping_iter=args.ledidi_early_stopping,
                )
                for res in batch_res:
                    results_for_chunk.append((res["seq"], res["gc"], res["pred_gc"],
                                              res["hamming"], res["success"], res["pred_success"]))
        else:
            for i, seq in enumerate(chunk_seqs):
                if args.method == "coral":
                    res = optimizer.generate_cf(
                        seq, target_gc=target_gc, steps=args.steps,
                        robust_mode=args.robust_mode, mc_samples=args.mc_samples,
                        lr=args.lr, rho_init=args.rho, margin=args.margin, verbose=(i < 3),
                    )
                    results_for_chunk.append((res["seq"], res["gc"], res["pred_gc"],
                                              res["dist"], res["success"], res["pred_success"]))
                else:
                    res = optimizer.generate_cf(
                        seq, target_gc=target_gc, l_input=args.ledidi_l, tau=args.ledidi_tau,
                        batch_size=args.ledidi_batch_size, max_iter=args.steps,
                        early_stopping_iter=args.ledidi_early_stopping, verbose=True,
                    )
                    results_for_chunk.append((res["seq"], res["gc"], res["pred_gc"],
                                              res["hamming"], res["success"], res["pred_success"]))

        for j, (cf_seq, cf_gc, pred_gc, edit_distance, success, pred_success) in enumerate(results_for_chunk):
            _, orig_seq, orig_gc, min_edits = chunk[j]
            target_label = 1 if orig_gc < target_gc else 0
            n_meaningful, n_redundant, n_counterproductive = classify_edits(orig_seq, cf_seq, target_label)
            records.append({
                "orig_seq": orig_seq, "cf_seq": cf_seq,
                "orig_gc": round(orig_gc, 4), "target_gc": target_gc,
                "cf_gc": round(float(cf_gc), 4), "pred_gc": round(float(pred_gc), 4),
                "edit_distance": float(edit_distance),
                "min_theoretical_edits": min_edits,
                "meaningful_edits": n_meaningful,
                "redundant_edits": n_redundant,
                "counterproductive_edits": n_counterproductive,
                "success": success, "pred_success": pred_success,
            })
            print(f"[{processed + j + 1}/{len(eligible)}] Succ={success} | "
                  f"GC: {orig_gc:.3f}->{float(cf_gc):.3f} (pred={float(pred_gc):.3f}) | "
                  f"Edits: {float(edit_distance):.0f} (min={min_edits}) | "
                  f"M={n_meaningful} R={n_redundant} C={n_counterproductive}")
        processed += len(chunk_seqs)

    df = pd.DataFrame(records)
    csv_path = run_outdir / "gc_cf_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    if records:
        n_total = len(records)
        n_success = sum(1 for r in records if r["success"])
        successful = [r for r in records if r["success"]]
        total_m = sum(r["meaningful_edits"] for r in records)
        total_r = sum(r["redundant_edits"] for r in records)
        total_c = sum(r["counterproductive_edits"] for r in records)
        total_cls = total_m + total_r + total_c

        summary = {
            "method": args.method, "mode": args.mode, "n_sequences": n_total,
            "seq_length": args.seq_length, "target_gc": target_gc,
            "success_rate": round(n_success / n_total, 4),
            "mean_edits": round(float(np.mean([r["edit_distance"] for r in records])), 2),
            "mean_edits_success": round(float(np.mean([r["edit_distance"] for r in successful])), 2) if successful else 0.0,
            "mean_gc_error": round(float(np.mean([abs(r["cf_gc"] - r["target_gc"]) for r in records])), 4),
            "meaningful_pct": round(total_m / total_cls * 100, 1) if total_cls > 0 else 0.0,
            "redundant_pct": round(total_r / total_cls * 100, 1) if total_cls > 0 else 0.0,
            "counterproductive_pct": round(total_c / total_cls * 100, 1) if total_cls > 0 else 0.0,
        }
        with open(run_outdir / "summary_metrics.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print(f"SUMMARY ({args.method.upper()}, {args.mode.upper()})")
        print(f"{'='*60}")
        print(f"  Sequences:     {n_total}")
        print(f"  Success rate:  {summary['success_rate']:.1%}")
        print(f"  Mean edits:    {summary['mean_edits']:.1f} (successful: {summary['mean_edits_success']:.1f})")
        print(f"  Mean GC error: {summary['mean_gc_error']:.4f}")
        print(f"  Edit quality — Meaningful: {summary['meaningful_pct']:.1f}% | "
              f"Redundant: {summary['redundant_pct']:.1f}% | "
              f"Counterproductive: {summary['counterproductive_pct']:.1f}%")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
