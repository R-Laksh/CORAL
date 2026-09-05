"""Fit and audit a frozen ESM-2 linear head on a mutation-order split."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch
from torch.nn.functional import one_hot

from coral.datasets.gb1 import load_gb1, SITES, SHA256, URL
from coral.models.esm import load_backbone, ESMC_CODE_REVISION


def metrics(y, predicted):
    return {"n": len(y), "r2": float(r2_score(y, predicted)),
            "spearman": float(spearmanr(y, predicted).statistic),
            "rmse_log1p": float(mean_squared_error(y, predicted) ** .5)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--family", choices=["esm2", "esmc"], default="esm2")
    p.add_argument("--amp-bf16", action="store_true", help="Accelerate GEMMs; retain FP32 weights and acceptance audits")
    p.add_argument("--output", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--validation-size", type=int, default=512)
    p.add_argument("--test-size", type=int, default=512)
    p.add_argument("--train-triples", type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    out, cache = Path(args.output), Path(args.cache)
    out.mkdir(parents=True, exist_ok=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    data = load_gb1(args.data)
    train, val, test = data.split(args.validation_size, args.test_size, args.train_triples)
    indices = np.concatenate((train, val, test))
    model = load_backbone(args.family, args.checkpoint, sites=SITES, device=args.device)
    ids = torch.tensor(data.ids[indices[:2]], device=args.device)
    probabilities = one_hot(ids, 20).transpose(1, 2).float().requires_grad_()
    with torch.no_grad():
        native = model.native_features(ids)
        adapted = model(probabilities)
    parity = float((native - adapted).abs().max())
    if parity > 1e-5:
        raise AssertionError(f"Native ESM parity failed: {parity}")
    # Directional derivative along a valid probability-simplex edge.
    direction = torch.zeros_like(probabilities)
    site, source = int(SITES[0]), int(ids[0, SITES[0]])
    direction[0, source, site] = -1
    direction[0, (source + 1) % 20, site] = 1
    midpoint = (probabilities.detach() + .1 * direction).requires_grad_()
    feature = model(midpoint)[:, :model.encoder.config.hidden_size].square().mean()
    grad = torch.autograd.grad(feature, midpoint)[0]
    analytic = float((grad * direction).sum())
    eps = 1e-2
    with torch.no_grad():
        plus = model(midpoint + eps * direction)[:, :model.encoder.config.hidden_size].square().mean()
        minus = model(midpoint - eps * direction)[:, :model.encoder.config.hidden_size].square().mean()
    finite_difference = float((plus - minus) / (2 * eps))
    gradient_error = abs(analytic - finite_difference)
    if gradient_error > 2e-4 + .1 * abs(finite_difference):
        raise AssertionError("ESM input-gradient finite difference failed")
    model.amp = args.amp_bf16
    with torch.no_grad():
        runtime_features = model(probabilities)
        runtime_native = model.native_features(ids)
    runtime_parity = float((runtime_features - runtime_native).abs().max())
    precision_drift = float((runtime_features - native).abs().max())
    if runtime_parity > 1e-5:
        raise AssertionError("Runtime native-token parity failed")
    checkpoint_sha = hashlib.sha256((Path(args.checkpoint) / "model.safetensors").read_bytes()).hexdigest()
    if cache.exists():
        with np.load(cache) as saved:
            if (not np.array_equal(saved["indices"], indices) or str(saved["checkpoint_sha"]) != checkpoint_sha
                or bool(saved.get("amp_bf16", False)) != args.amp_bf16):
                raise ValueError("Feature cache provenance does not match this run")
            features = saved["features"]
            embedding_seconds = float(saved["seconds"])
    else:
        tic = time.perf_counter()
        chunks = []
        for start in range(0, len(indices), args.batch_size):
            x = torch.tensor(data.ids[indices[start:start + args.batch_size]], device=args.device)
            with torch.no_grad():
                chunks.append(model.native_features(x).cpu().numpy())
            if start % (args.batch_size * 10) == 0:
                print(f"ESM features {min(start + args.batch_size, len(indices))}/{len(indices)}; "
                      f"{time.perf_counter() - tic:.1f}s", flush=True)
        features = np.concatenate(chunks)
        embedding_seconds = time.perf_counter() - tic
        np.savez_compressed(cache, indices=indices, features=features, seconds=embedding_seconds,
                            checkpoint_sha=checkpoint_sha, amp_bf16=args.amp_bf16)
    n, v = len(train), len(val)
    y = np.log1p(data.fitness[indices])
    candidates, fitted = [], []
    for name, x in (("esm_mean", features[:, :model.encoder.config.hidden_size]),
                    ("esm_mean_and_sites", features),
                    ("onehot_additive", np.eye(20)[data.ids[indices][:, SITES]].reshape(len(indices), -1))):
        scaler = StandardScaler().fit(x[:n])
        z = scaler.transform(x).astype(np.float64)
        for alpha in (1., 10., 100., 1000., 10000.):
            head = Ridge(alpha=alpha).fit(z[:n], y[:n])
            record = {"features": name, "alpha": alpha,
                      "validation": metrics(y[n:n + v], head.predict(z[n:n + v]))}
            candidates.append(record)
            fitted.append((scaler, head, z))
    # Selection uses validation RMSE only; test evaluated after selection.
    esm_indices = [i for i, c in enumerate(candidates) if c["features"].startswith("esm")]
    winner = min(esm_indices, key=lambda i: candidates[i]["validation"]["rmse_log1p"])
    baseline = min((i for i, c in enumerate(candidates) if c["features"] == "onehot_additive"),
                   key=lambda i: candidates[i]["validation"]["rmse_log1p"])
    selected = []
    for i in (winner, baseline):
        scaler, head, z = fitted[i]
        selected.append({**candidates[i], "test": metrics(y[n + v:], head.predict(z[n + v:])),
                         "train": metrics(y[:n], head.predict(z[:n]))})
    scaler, head, z = fitted[winner]
    np.savez(out / "head.npz", mean=scaler.mean_, scale=scaler.scale_,
             coef=head.coef_, intercept=head.intercept_)
    predicted = head.predict(z)
    np.savez_compressed(out / "split_predictions.npz", indices=indices,
                        prediction=predicted, observed_log1p=y, n_train=n, n_validation=v)
    manifest = {"dataset_url": URL, "dataset_sha256": SHA256, "measured_n": len(data.ids),
                "missing_of_20pow4": 20 ** 4 - len(data.ids), "model_id": model.model_id,
                "family": args.family, "code_revision": ESMC_CODE_REVISION if args.family == "esmc" else None,
                "model_revision": model.model_revision, "checkpoint_sha256": checkpoint_sha,
                "encoder_parameters": sum(p.numel() for p in model.parameters()),
                "target": "log1p(WT-normalised selection fitness)",
                "split": {"train_order_0_to_2": int((data.mutation_order[train] <= 2).sum()),
                          "train_order_3": args.train_triples, "train_total": n, "validation_order_3": v,
                          "test_order_4": len(test), "hash_salt": "coral-gb1-v1"},
                "selected": selected, "validation_candidates": candidates,
                "embedding_seconds": embedding_seconds,
                "checks": {"native_max_abs_error": parity,
                           "runtime_native_max_abs_error": runtime_parity,
                           "runtime_vs_fp32_feature_max_abs_error": precision_drift,
                           "directional_derivative": analytic, "finite_difference": finite_difference,
                           "absolute_gradient_error": gradient_error,
                           "all_encoder_weights_frozen": all(not p.requires_grad for p in model.parameters())},
                "runtime": {"torch": torch.__version__, "device": args.device,
                            "threads": args.threads, "batch_size": args.batch_size,
                            "amp_bf16": args.amp_bf16}}
    (out / "model_validation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"selected": selected, "checks": manifest["checks"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
