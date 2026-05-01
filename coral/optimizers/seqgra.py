import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn
import torch.nn.functional as F
from ledidi import Ledidi
from tqdm import tqdm

from coral.losses.losses import ProbToLogitMarginLoss
from coral.optimizers.coral import CORALOptimizer

class SeqgraCORALOptimizer(CORALOptimizer):
    """CORAL optimizer for SeqGra binary-classification counterfactuals."""

    def __init__(self, model: nn.Module, device="cuda"):
        self.model = torch.compile(model.to(device).eval())
        self.device = device

    @staticmethod
    def prob_to_logit(p: float) -> float:
        eps = 1e-6
        p = min(max(p, eps), 1 - eps)
        return math.log(p / (1 - p))

    @staticmethod
    def constraint_from_logit(z, target_label, p_target=0.5):
        m = SeqgraCORALOptimizer.prob_to_logit(p_target)
        if isinstance(target_label, torch.Tensor):
            s = 2.0 * target_label.to(z.dtype) - 1.0
        else:
            s = 2.0 * float(target_label) - 1.0
        return m - s * z

    def _init_state(self, x_onehot, target_info: dict,
                    rho_init: float, steps: int, alpha: float) -> dict:
        device = self.device
        margin = target_info.get("margin", 0.5)
        x0 = x_onehot.to(device)
        B, _, L = x0.shape
        orig_idx = x0.argmax(dim=1)                           # (B, L)
        editable_mask = torch.ones((B, L), dtype=torch.bool, device=device)

        with torch.inference_mode():
            z_orig = self.model(x0).squeeze(-1)
            p_orig = torch.sigmoid(z_orig)
            pred_orig = (p_orig >= 0.5).long()
        target_label = 1.0 - pred_orig.float()

        Z_init_log = torch.log(x0.transpose(1, 2).contiguous() + 1e-4)
        Z = torch.nn.Parameter(Z_init_log + torch.randn_like(Z_init_log) * 0.05)

        return {
            "Z": Z, "Z_init": Z_init_log,
            "orig_idx": orig_idx, "orig_ids": orig_idx,   # alias for shared code
            "editable_mask": editable_mask,
            "target_label": target_label, "margin": margin,
            "pred_orig": pred_orig, "B": B,
            "lambda_gc": torch.zeros(B, device=device),
            "rho_vec": torch.full((B,), rho_init, device=device),
            "rho_init": rho_init,
            "last_improvement_step": torch.zeros(B, dtype=torch.long, device=device),
            "STAGNATION_THRESH": steps // 5,
            "alpha": alpha,
            # Best-solution trackers
            "best_dist": torch.full((B,), float("inf"), device=device),
            "best_seq_idx": torch.zeros((B, L), dtype=torch.long, device=device),
            "best_success": torch.zeros(B, dtype=torch.bool, device=device),
            "best_probs": torch.zeros(B, device=device),
            "best_attempt_probs": torch.where(target_label == 1.0,
                                              torch.zeros(B, device=device),
                                              torch.ones(B, device=device)),
            "best_attempt_idx": orig_idx.clone(),
        }

    def _enforce_constraints(self, Z, state):
        pass  # all positions editable; no masking needed

    def _clean_logits(self, Z, state):
        return Z  # no vocabulary masking for 4-base one-hot

    def _forward(self, logits_clean, tau, mode, K, state):
        B = state["B"]
        _, L, _ = logits_clean.shape
        target_label = state["target_label"]
        margin = state["margin"]

        if mode == "soft":
            pi = F.softmax(logits_clean / tau, dim=-1)           # (B, L, 4)
            x_in = pi.transpose(1, 2)                             # (B, 4, L)
            z_samples = self.model(x_in).view(B, 1)
            probs_4d = pi.unsqueeze(1)                            # (B, 1, L, 4)

        elif mode == "gumbel":
            logits_mc = logits_clean.unsqueeze(1).expand(B, K, L, 4).reshape(B * K, L, 4)
            pi = F.gumbel_softmax(logits_mc, tau=tau, hard=False, dim=-1)
            y = F.one_hot(pi.argmax(dim=-1), 4).float() + (pi - pi.detach())
            z_samples = self.model(y.transpose(1, 2)).view(B, K)
            probs_4d = pi.view(B, K, L, 4)

        else:  # st_det
            pi = F.softmax(logits_clean / tau, dim=-1)
            y = (F.one_hot(pi.argmax(dim=-1), 4).float() + (pi - pi.detach())).unsqueeze(1)
            z_samples = self.model(y.reshape(B, L, 4).transpose(1, 2)).view(B, 1)
            probs_4d = pi.unsqueeze(1)

        g_each = self.constraint_from_logit(z_samples, target_label.unsqueeze(1), p_target=margin)
        return probs_4d, g_each

    def _expected_hamming(self, probs_4d, state):
        """Expected Hamming for one-hot 4-base vocabulary. Returns (B, K)."""
        orig_idx = state["orig_idx"]                              # (B, L)
        editable_mask = state["editable_mask"]                    # (B, L)
        B, K, L, C = probs_4d.shape
        gather_idx = orig_idx.unsqueeze(1).unsqueeze(-1).expand(B, K, L, 1)
        p_same = torch.gather(probs_4d, 3, gather_idx).squeeze(-1)  # (B, K, L)
        mask = editable_mask.unsqueeze(1)                            # (B, 1, L)
        return torch.where(mask, 1.0 - p_same, torch.zeros_like(p_same)).sum(dim=2)  # (B, K)

    def _eval_discrete(self, Z, state, step, opt):
        device = self.device
        orig_idx = state["orig_idx"]
        editable_mask = state["editable_mask"]
        target_label = state["target_label"]
        margin = state["margin"]
        alpha = state["alpha"]
        Z_init_log = state["Z_init"]

        idx_new = Z.argmax(dim=-1)                                # (B, L)
        dist_disc = ((idx_new != orig_idx) & editable_mask).sum(dim=1).float()
        x_cf = F.one_hot(idx_new, 4).float().transpose(1, 2)
        z_cf = self.model(x_cf).view(-1)
        p_cf = torch.sigmoid(z_cf)
        threshold = self.prob_to_logit(margin)
        success = torch.where(target_label == 1.0, z_cf >= threshold, z_cf <= -threshold)

        update_mask = success & ((~state["best_success"]) | (dist_disc < state["best_dist"])) & (dist_disc > 0)
        state["best_dist"] = torch.where(update_mask, dist_disc, state["best_dist"])
        state["best_seq_idx"] = torch.where(update_mask.unsqueeze(1), idx_new, state["best_seq_idx"])
        state["best_success"] = state["best_success"] | update_mask
        state["best_probs"] = torch.where(update_mask, p_cf, state["best_probs"])
        state["last_improvement_step"] = torch.where(update_mask, step, state["last_improvement_step"])

        closer = torch.where(target_label == 1.0, p_cf > state["best_attempt_probs"], p_cf < state["best_attempt_probs"])
        attempt_update = ~success & closer
        state["best_attempt_probs"] = torch.where(attempt_update, p_cf, state["best_attempt_probs"])
        state["best_attempt_idx"] = torch.where(attempt_update.unsqueeze(1), idx_new, state["best_attempt_idx"])
        state["last_improvement_step"] = torch.where(attempt_update, step, state["last_improvement_step"])

        if step % 50 == 0:
            stagnant = (step - state["last_improvement_step"]) > state["STAGNATION_THRESH"]
            restart_mask = stagnant & ~state["best_success"]
            if restart_mask.any():
                Z.data[restart_mask] = Z_init_log[restart_mask] + torch.randn_like(Z.data[restart_mask]) * 0.1
                state["lambda_gc"][restart_mask] = 0.0
                state["rho_vec"][restart_mask] = state["rho_init"]
                state["last_improvement_step"][restart_mask] = step
                s = opt.state[Z]
                if "exp_avg" in s:
                    s["exp_avg"][restart_mask] = 0.0
                    s["exp_avg_sq"][restart_mask] = 0.0

            refine_mask = stagnant & state["best_success"] & (state["best_dist"] > 1)
            if refine_mask.any():
                best_cf_oh = F.one_hot(state["best_seq_idx"][refine_mask], 4).float()
                Z_cf = torch.log(best_cf_oh + 1e-4)
                Z_orig = Z_init_log[refine_mask]
                Z.data[refine_mask] = alpha * Z_cf + (1 - alpha) * Z_orig + torch.randn_like(Z.data[refine_mask]) * 0.1
                state["last_improvement_step"][refine_mask] = step
                s = opt.state[Z]
                if "exp_avg" in s:
                    s["exp_avg"][refine_mask] = 0.0
                    s["exp_avg_sq"][refine_mask] = 0.0

    def _build_output(self, state):
        orig_idx = state["orig_idx"]
        editable_mask = state["editable_mask"]
        best_success = state["best_success"]

        final_idx = torch.where(best_success.unsqueeze(1), state["best_seq_idx"], state["best_attempt_idx"])
        final_prob = torch.where(best_success, state["best_probs"], state["best_attempt_probs"])
        final_dist = torch.where(
            best_success,
            state["best_dist"],
            ((final_idx != orig_idx) & editable_mask).sum(dim=1).float(),
        )
        return {
            "success": best_success.cpu(),
            "orig_pred": state["pred_orig"].cpu(),
            "target_label": state["target_label"].cpu(),
            "cf_prob": final_prob.cpu(),
            "cf_pred": (final_prob >= 0.5).long().cpu(),
            "orig_idx": orig_idx.cpu(),
            "cf_idx": final_idx.cpu(),
            "edit_distance": final_dist.cpu(),
        }


    def generate_cf_batch(self, x_onehot: torch.Tensor, steps=800,
                          tau_max=2.5, tau_min=0.01, k_soft=10, k_gumbel=400,
                          rho=10.0, margin=0.5, alpha=0.7, lr=0.5, mc_samples=64,
                          robust_mode="mean"):
        target_info = {"margin": margin}
        return self._run_batch(
            inputs=x_onehot, target_info=target_info,
            steps=steps, tau_max=tau_max, tau_min=tau_min,
            k_soft=k_soft, k_gumbel=k_gumbel, rho=rho,
            mc_samples=mc_samples, robust_mode=robust_mode,
            lr=lr, alpha=alpha, tau_burst=1.0,
        )

    def generate_cf(self, x_onehot: torch.Tensor, y_orig: float = None,
                    motif_mask=None, steps=800, tau_max=1.5, tau_min=0.1,
                    k_soft=10, k_gumbel=400, rho=10.0, margin=0.5, lr=5e-3,
                    verbose=False, mc_samples=8, robust_mode="mean"):
        """Single-sequence CORAL optimization (to be deprecated)."""
        device = self.device
        x0 = x_onehot.to(device)
        L = x0.shape[1]
        orig_idx = x0.argmax(dim=0)
        editable_mask = torch.ones(L, dtype=torch.bool, device=device)

        with torch.inference_mode():
            z_orig = self.model(x0.unsqueeze(0)).squeeze(0)
            p_orig = torch.sigmoid(z_orig).item()
        pred_orig = int(p_orig >= 0.5)
        target_label = 1 - pred_orig

        Z = torch.nn.Parameter(torch.zeros(L, 4, device=device))
        with torch.no_grad():
            Z.normal_(mean=0.0, std=0.01)
            Z[torch.arange(L), orig_idx] += 2.0

        opt = torch.optim.AdamW([Z], lr=lr, fused=True)
        lambda_gc = 0.0

        best = {"seq_idx": None, "dist": float("inf"), "logit": None,
                "pred": None, "label": None, "success": False}

        burst_every, burst_len, tau_burst = 200, 50, 1.0

        for step in range(steps):
            opt.zero_grad()
            tau, mode = CORALOptimizer._temperature_and_mode(
                step, k_gumbel, k_soft, tau_min, tau_max, burst_every, burst_len, tau_burst
            )

            if mode == "soft":
                pi = F.softmax(Z / tau, dim=-1)
                x_in = pi.unsqueeze(0).transpose(1, 2)
                z_new = self.model(x_in).squeeze(0)
                probs_for_dist = pi.unsqueeze(0)

            elif mode == "gumbel":
                K = mc_samples
                logits_mc = Z.unsqueeze(0).expand(K, -1, -1)
                pi = F.gumbel_softmax(logits_mc, tau=tau, hard=False, dim=-1)
                y = F.one_hot(pi.argmax(-1), 4).float() + (pi - pi.detach())
                z_samples = self.model(y.transpose(1, 2)).view(K)
                z_new = z_samples.mean() if robust_mode == "mean" else (z_samples.min() if target_label == 1 else z_samples.max())
                probs_for_dist = pi

            else:  # st_det
                pi = F.softmax(Z / tau, dim=-1)
                y = (F.one_hot(pi.argmax(-1), 4).float() + (pi - pi.detach())).unsqueeze(0)
                z_new = self.model(y.transpose(1, 2)).squeeze(0)
                probs_for_dist = pi.unsqueeze(0)

            if mode != "gumbel":
                z_new_scalar = z_new if z_new.dim() == 0 else z_new.mean()
            else:
                z_new_scalar = z_new

            g_label = self.constraint_from_logit(z_new_scalar, target_label, p_target=margin)
            aug_gc = (0.5 / rho) * F.relu(lambda_gc + rho * g_label).pow(2)

            # simple expected hamming for 2D/3D probs
            gather_idx = orig_idx.unsqueeze(-1)
            if probs_for_dist.dim() == 3:
                p_same = probs_for_dist[:, torch.arange(L), orig_idx]  # (K, L) or (1, L)
                d_edit = (1.0 - p_same).sum(dim=1).mean()
            else:
                p_same = probs_for_dist[torch.arange(L), orig_idx]
                d_edit = (1.0 - p_same).sum()

            loss = d_edit + aug_gc
            loss.backward()
            torch.nn.utils.clip_grad_norm_([Z], 1.0)
            opt.step()

            with torch.no_grad():
                lambda_gc = max(0.0, lambda_gc + rho * g_label.item())
                if step % 50 == 0 and step < (steps - 100):
                    rho = min(100.0, rho * 1.1)

            with torch.inference_mode():
                idx_new = Z.argmax(dim=-1)
                x_cf = F.one_hot(idx_new, 4).float().T.unsqueeze(0)
                z_cf = self.model(x_cf).squeeze(0)
                p_cf = torch.sigmoid(z_cf).item()
                pred_cf = int(p_cf >= 0.5)
                dist_disc = int(((idx_new != orig_idx) & editable_mask).sum().item())
                success = (pred_cf == target_label)
                if success and dist_disc > 0 and dist_disc < best["dist"]:
                    best.update({"seq_idx": idx_new.clone(), "dist": dist_disc,
                                 "logit": z_cf.item(), "pred": p_cf, "label": pred_cf, "success": True})

        return {
            "success": best["success"], "target_label": target_label,
            "orig_pred": pred_orig,
            "cf_idx": best["seq_idx"].cpu() if best["seq_idx"] is not None else None,
            "edit_distance": best["dist"],
        }

    def generate_cf_batch_multistart(self, x_onehot: torch.Tensor, steps: int,
                                      n_starts=4, base_seed=0, **kwargs):
        """Run generate_cf_batch n_starts times; merge per-sample best results."""
        device = self.device
        x0 = x_onehot.to(device)
        B, _, L = x0.shape
        orig_idx = x0.argmax(dim=1)

        with torch.inference_mode():
            z_orig = self.model(x0).squeeze(-1)
            pred_orig = (torch.sigmoid(z_orig) >= 0.5).long()
        target_label = 1.0 - pred_orig.float()

        combined_success = torch.zeros(B, dtype=torch.bool, device=device)
        combined_dist = torch.full((B,), float("inf"), device=device)
        combined_cf_idx = orig_idx.clone()
        combined_probs = torch.where(target_label == 1.0, torch.zeros(B, device=device), torch.ones(B, device=device))

        import numpy as np
        for start_idx in range(n_starts):
            if combined_success.all():
                print(f"  [multistart] all {B} samples succeeded after {start_idx} start(s).")
                break
            seed = base_seed + start_idx * 7919
            torch.manual_seed(seed)
            np.random.seed(seed % (2**31))
            res = self.generate_cf_batch(x_onehot=x_onehot, steps=steps,
                                         k_gumbel=steps // 2, **kwargs)
            s = res["success"].to(device)
            d = res["edit_distance"].to(device)
            p = res["cf_prob"].to(device)
            ci = res["cf_idx"].to(device)
            improve = s & (~combined_success | (d < combined_dist))
            combined_success[improve] = True
            combined_dist[improve] = d[improve]
            combined_cf_idx[improve] = ci[improve]
            combined_probs[improve] = p[improve]
            n_done = int(combined_success.sum().item())
            print(f"  [multistart {start_idx+1}/{n_starts}] cumulative success: {n_done}/{B} ({100*n_done/B:.1f}%)")

        return {
            "success": combined_success.cpu(),
            "orig_pred": pred_orig.cpu(),
            "target_label": target_label.cpu(),
            "cf_prob": combined_probs.cpu(),
            "cf_pred": (combined_probs >= 0.5).long().cpu(),
            "orig_idx": orig_idx.cpu(),
            "cf_idx": combined_cf_idx.cpu(),
            "edit_distance": combined_dist.cpu(),
        }

    def generate_cf_batch_parallel_candidates(self, x_onehot: torch.Tensor, n_copies=8, **kwargs):
        """Run generate_cf_batch with n_copies duplicates; pick best candidate per sequence."""
        B, C, L = x_onehot.shape
        x_expanded = x_onehot.repeat_interleave(n_copies, dim=0)
        res = self.generate_cf_batch(x_expanded, **kwargs)

        success = res["success"].view(B, n_copies)
        dists = res["edit_distance"].view(B, n_copies)
        probs = res["cf_prob"].view(B, n_copies)
        cf_idx = res["cf_idx"].view(B, n_copies, L)
        target_labels = res["target_label"].view(B, n_copies)[:, 0]
        orig_preds = res["orig_pred"].view(B, n_copies)[:, 0]
        orig_idx = res["orig_idx"].view(B, n_copies, L)[:, 0, :]

        feasible = success & (dists > 0)
        any_feasible = feasible.any(dim=1)
        inf = torch.tensor(float("inf"), device=dists.device, dtype=dists.dtype)
        masked_dists = torch.where(feasible, dists, inf)
        min_dist = masked_dists.min(dim=1).values

        tied = feasible & (dists == min_dist.unsqueeze(1))
        target_score = torch.where(target_labels.unsqueeze(1) == 1, probs, -probs)
        neg_inf = torch.tensor(float("-inf"), device=probs.device, dtype=probs.dtype)
        best_idx_feasible = torch.where(tied, target_score, neg_inf).argmax(dim=1)
        best_idx_fallback = target_score.argmax(dim=1)
        best_idx = torch.where(any_feasible, best_idx_feasible, best_idx_fallback)

        row = torch.arange(B, device=best_idx.device)
        return {
            "success": success[row, best_idx],
            "orig_pred": orig_preds,
            "target_label": target_labels,
            "cf_prob": probs[row, best_idx],
            "cf_pred": (probs[row, best_idx] >= 0.5).long(),
            "orig_idx": orig_idx,
            "cf_idx": cf_idx[row, best_idx],
            "edit_distance": dists[row, best_idx],
        }


# ── Ledidi adapter ───────────────────────────────────────────────────────────

class SeqgraOracleOneHot(nn.Module):
    """Wraps a SeqGra CNN so Ledidi can interface with it.

    Input:  X_onehot (B, 4, L)
    Output: probabilities (B, 1)
    """

    def __init__(self, model: nn.Module, device="cuda"):
        super().__init__()
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, X_onehot: torch.Tensor) -> torch.Tensor:
        if X_onehot.dim() == 2:
            X_onehot = X_onehot.unsqueeze(0)
        logits = self.model(X_onehot)
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
        return torch.sigmoid(logits).unsqueeze(-1)


# ── Ledidi SeqGra optimizer ──────────────────────────────────────────────────

class LedidiSeqgraCFOptimizer:
    """Ledidi-based baseline for SeqGra binary-classification counterfactuals."""

    def __init__(self, model: nn.Module, device="cuda"):
        self.device = device
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.oracle = SeqgraOracleOneHot(self.model).to(device)

    @staticmethod
    def _target_score(prob: torch.Tensor, target_label: int) -> torch.Tensor:
        return prob if int(target_label) == 1 else (1.0 - prob)

    def _pick_best_candidate(self, X_hat, orig_idx, target_label, margin):
        with torch.inference_mode():
            probs = self.oracle(X_hat).squeeze(-1).view(-1)
            idx_all = X_hat.argmax(dim=1)
            dists = (idx_all != orig_idx.unsqueeze(0)).sum(dim=1)
            preds = (probs >= margin).long()
            success = preds.eq(int(target_label))
            score = self._target_score(probs, int(target_label))

        feasible = success & (dists > 0)
        if feasible.any():
            feasible_idx = feasible.nonzero(as_tuple=False).flatten()
            min_dist = dists[feasible_idx].min()
            tied = feasible_idx[dists[feasible_idx] == min_dist]
            best_i = tied[torch.argmax(score[tied])].item() if tied.numel() > 1 else tied.item()
            return {"success": True, "cf_idx": idx_all[best_i].detach().cpu(),
                    "edit_distance": int(dists[best_i].item()),
                    "best_feasible_prob": float(probs[best_i].item()),
                    "best_attempt_prob": float(probs[best_i].item()),
                    "n_candidates": int(X_hat.shape[0])}
        best_i = torch.argmax(score).item()
        return {"success": False, "cf_idx": None, "edit_distance": float("inf"),
                "best_feasible_prob": None, "best_attempt_prob": float(probs[best_i].item()),
                "n_candidates": int(X_hat.shape[0])}

    def generate_cf(self, x_onehot: torch.Tensor, y_orig: float = None,
                    l_input=0.01, tau=1.0, batch_size=256, max_iter=2000,
                    early_stopping_iter=250, report_iter=100, lr=1.0,
                    output_loss="margin", verbose=True, editable_mask=None, margin=0.5):
        device = self.device
        x0 = x_onehot.to(device).float()
        if x0.dim() != 2:
            raise ValueError(f"Expected x_onehot shape (4,L), got {tuple(x0.shape)}")

        X = x0.unsqueeze(0)
        L = X.shape[-1]
        orig_idx = x0.argmax(dim=0)

        with torch.no_grad():
            p_orig = self.oracle(X).squeeze().item()
        pred_orig = int(p_orig >= 0.5)
        target_label = 1 - pred_orig

        y_bar_val = margin if target_label == 1 else 1.0 - margin
        y_bar = torch.tensor([[y_bar_val]], dtype=torch.float32, device=device)

        if editable_mask is None:
            input_mask = torch.zeros(L, dtype=torch.bool, device=device)
        else:
            input_mask = (~editable_mask.bool().to(device)).clone()

        loss_obj = ProbToLogitMarginLoss(target_label=target_label, margin=margin, squared=True).to(device)

        designer = Ledidi(
            model=self.oracle, shape=X.shape[-2:], target=None,
            tau=tau, l=l_input, batch_size=batch_size,
            max_iter=max_iter, early_stopping_iter=early_stopping_iter,
            report_iter=report_iter, lr=lr,
            input_mask=input_mask, initial_weights=None,
            return_history=False, verbose=verbose, output_loss=loss_obj,
        ).to(device)

        X_hat = designer.fit_transform(X, y_bar)
        if X_hat.dim() == 2:
            X_hat = X_hat.unsqueeze(0)

        picked = self._pick_best_candidate(X_hat, orig_idx, target_label, margin)
        return {
            "success": picked["success"], "target_label": target_label,
            "orig_pred": pred_orig, "cf_idx": picked["cf_idx"],
            "edit_distance": picked["edit_distance"],
            "best_feasible_prob": picked["best_feasible_prob"],
            "best_attempt_prob": picked["best_attempt_prob"],
            "first_flip_step": -1, "stage1_success": picked["success"],
            "fallback_used": False, "fallback_success": False,
            "fallback_evals": 0, "fallback_best_prob": float("nan"),
            "solver_stage": "ledidi" if picked["success"] else "none",
            "n_candidates": picked["n_candidates"],
        }

    def generate_cf_batch(self, x_onehot: torch.Tensor, y_orig=None,
                          l_input=0.01, tau=1.0, batch_size=256, max_iter=1000,
                          early_stopping_iter=250, report_iter=100, lr=1.0,
                          output_loss="margin", verbose=True, max_workers=16, margin=0.5):
        device = self.device
        x0 = x_onehot.to(device).float()
        if x0.dim() != 3:
            raise ValueError(f"Expected x_onehot shape (B,4,L), got {tuple(x0.shape)}")
        B, _, L = x0.shape
        orig_idx = x0.argmax(dim=1)

        def _process_single(i):
            yi = None if y_orig is None else float(y_orig[i].item())
            return self.generate_cf(
                x_onehot=x0[i], y_orig=yi, l_input=l_input, tau=tau,
                batch_size=batch_size, max_iter=max_iter,
                early_stopping_iter=early_stopping_iter,
                report_iter=report_iter, lr=lr,
                output_loss=output_loss, verbose=False, margin=margin,
            )

        results_ordered = [None] * B
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_single, i): i for i in range(B)}
            for future in tqdm(as_completed(futures), total=B, desc=f"Ledidi Batch (Threads: {max_workers})"):
                idx = futures[future]
                try:
                    results_ordered[idx] = future.result()
                except Exception as e:
                    print(f"Error processing sequence {idx}: {e}")
                    raise

        success_list, orig_pred_list, target_label_list = [], [], []
        cf_prob_list, cf_pred_list, cf_idx_list, edit_dist_list = [], [], [], []
        best_attempt_prob_list, stage1_success_list = [], []
        fallback_used_list, fallback_success_list, fallback_evals_list = [], [], []
        fallback_best_prob_list, solver_stage_list = [], []

        for i, res in enumerate(results_ordered):
            success_list.append(bool(res["success"]))
            orig_pred_list.append(int(res["orig_pred"]))
            target_label_list.append(float(res["target_label"]))
            if res["success"]:
                cf_idx = res["cf_idx"].to(device)
                cf_prob = float(res["best_feasible_prob"])
                cf_pred = 1 if cf_prob >= 0.5 else 0
                edit_dist = float(res["edit_distance"])
            else:
                cf_idx = orig_idx[i].clone()
                cf_prob = float(res["best_attempt_prob"])
                cf_pred = 1 if cf_prob >= 0.5 else 0
                edit_dist = float("inf")
            cf_idx_list.append(cf_idx)
            cf_prob_list.append(cf_prob)
            cf_pred_list.append(cf_pred)
            edit_dist_list.append(edit_dist)
            best_attempt_prob_list.append(float(res["best_attempt_prob"]))
            stage1_success_list.append(bool(res["stage1_success"]))
            fallback_used_list.append(False)
            fallback_success_list.append(False)
            fallback_evals_list.append(0)
            fallback_best_prob_list.append(float("nan"))
            solver_stage_list.append(res["solver_stage"])

        return {
            "success": torch.tensor(success_list, dtype=torch.bool),
            "orig_pred": torch.tensor(orig_pred_list, dtype=torch.long),
            "target_label": torch.tensor(target_label_list, dtype=torch.float32),
            "cf_prob": torch.tensor(cf_prob_list, dtype=torch.float32),
            "cf_pred": torch.tensor(cf_pred_list, dtype=torch.long),
            "orig_idx": orig_idx.detach().cpu(),
            "cf_idx": torch.stack(cf_idx_list).detach().cpu(),
            "edit_distance": torch.tensor(edit_dist_list, dtype=torch.float32),
            "best_attempt_prob": torch.tensor(best_attempt_prob_list, dtype=torch.float32),
            "stage1_success": torch.tensor(stage1_success_list, dtype=torch.bool),
            "fallback_used": torch.tensor(fallback_used_list, dtype=torch.bool),
            "fallback_success": torch.tensor(fallback_success_list, dtype=torch.bool),
            "fallback_evals": torch.tensor(fallback_evals_list, dtype=torch.long),
            "fallback_best_prob": torch.tensor(fallback_best_prob_list, dtype=torch.float32),
            "solver_stage": solver_stage_list,
        }
