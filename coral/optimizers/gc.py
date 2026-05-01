"""GC-content counterfactual optimizers: CORAL (LearnedGCOptimizer) and Ledidi (LedidiGCOptimizer)."""
import math
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn
import torch.nn.functional as F
from ledidi import Ledidi
from tqdm import tqdm

from coral.losses.losses import HammingTableLoss, GCMarginLoss
from coral.optimizers.coral import CORALOptimizer

class LearnedGCOptimizer(CORALOptimizer):
    """CORAL optimizer for GC-content counterfactuals using 6-mer token vocabulary."""

    def __init__(self, gc_model, tokenizer, device="cuda"):
        self.gc_model = torch.compile(gc_model.to(device).eval())
        self.tokenizer = tokenizer
        self.specials = set(self.tokenizer.all_special_tokens)
        self.device = device
        self.embedding_matrix = self.gc_model.backbone.get_input_embeddings().weight.detach()
        self.V_full = tokenizer.vocab_size
        self._precompute_tables()

    def _precompute_tables(self):
        vocab = self.tokenizer.get_vocab()
        pure_token_ids_list = sorted(
            tid for tok, tid in vocab.items() if re.fullmatch(r"[ACGT]{6}", tok)
        )
        self.V_pure = len(pure_token_ids_list)
        self.pure_token_ids = torch.tensor(pure_token_ids_list, device=self.device, dtype=torch.long)

        self.valid_token_mask = torch.zeros(self.V_full, dtype=torch.bool, device=self.device)
        self.valid_token_mask[self.pure_token_ids] = True

        self.global2pure = torch.full((self.V_full,), -1, dtype=torch.long, device=self.device)
        self.global2pure[self.pure_token_ids] = torch.arange(self.V_pure, dtype=torch.long, device=self.device)

        id2tok = {v: k for k, v in vocab.items()}
        self.gc_per_token = torch.zeros(self.V_full, device=self.device, dtype=torch.float32)
        for i in range(self.V_full):
            tok = id2tok.get(i, "")
            seq_chars = [c for c in tok if c in "ACGTacgt"]
            if seq_chars:
                self.gc_per_token[i] = sum(c in "GCgc" for c in seq_chars) / len(seq_chars)

        pure_seqs = [id2tok[tid] for tid in pure_token_ids_list]
        table_vals = torch.tensor([["ACGT".index(c) for c in seq] for seq in pure_seqs], dtype=torch.long)
        self.hamming_table = (table_vals.unsqueeze(1) != table_vals.unsqueeze(0)).sum(-1).to(self.device).float()


    def _init_state(self, sequences, target_info: dict,
                    rho_init: float, steps: int, alpha: float) -> dict:
        device = self.device
        target_gc = target_info["target_gc"]
        margin = target_info.get("margin", 0.05)
        verbose = target_info.get("verbose", True)
        B = len(sequences)

        tok_out = self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=True).to(device)
        orig_ids = tok_out["input_ids"]
        attention_mask = tok_out["attention_mask"]
        _, L = orig_ids.shape

        editable_mask = (self.global2pure[orig_ids] >= 0) & attention_mask.bool()
        non_editable_mask = ~editable_mask & attention_mask.bool()
        pad_mask = ~attention_mask.bool()
        nm_b, nm_l = non_editable_mask.nonzero(as_tuple=True)

        orig_gc_list = [(s.count("G") + s.count("C")) / len(s) for s in sequences]
        orig_gc_tensor = torch.tensor(orig_gc_list, device=device, dtype=torch.float32)
        target_label = (orig_gc_tensor < target_gc).float()

        Z = torch.nn.Parameter(torch.zeros((B, L, self.V_full), device=device))
        with torch.no_grad():
            for b in range(B):
                em = editable_mask[b]
                edit_oh = F.one_hot(orig_ids[b, em], num_classes=self.V_full).float()
                Z.data[b, em] = torch.log(edit_oh + 1e-1)
                Z.data[b, em] = Z.data[b, em].masked_fill(~self.valid_token_mask, -1e9)
            Z.data[non_editable_mask] = -1e9
            Z.data[nm_b, nm_l, orig_ids[nm_b, nm_l]] = 0.0
            Z.data[pad_mask] = -1e9
            Z_init = Z.data.clone()
            Z.data[editable_mask] += torch.randn_like(Z.data[editable_mask]) * 0.05
            Z.data[editable_mask].masked_fill_(~self.valid_token_mask, -1e9)

        inf = float("inf")
        return {
            "Z": Z, "Z_init": Z_init,
            "orig_ids": orig_ids, "editable_mask": editable_mask,
            "non_editable_mask": non_editable_mask,
            "attention_mask": attention_mask,
            "nm_b": nm_b, "nm_l": nm_l, "pad_mask": pad_mask,
            "target_label": target_label, "target_gc": target_gc,
            "margin": margin, "orig_gc_tensor": orig_gc_tensor,
            "B": B,
            "lambda_gc": torch.zeros(B, device=device),
            "rho_vec": torch.full((B,), rho_init, device=device),
            "rho_init": rho_init,
            "last_improvement_step": torch.zeros(B, dtype=torch.long, device=device),
            "STAGNATION_THRESH": steps // 5,
            "alpha": alpha,
            "verbose": verbose,
            # Best-solution trackers
            "best_dist": torch.full((B,), inf, device=device),
            "best_seq_ids": orig_ids.clone(),
            "best_pred_success": torch.zeros(B, dtype=torch.bool, device=device),
            "best_true_success": torch.zeros(B, dtype=torch.bool, device=device),
            "best_pred_gc": torch.where(target_label == 1.0,
                                         torch.zeros(B, device=device),
                                         torch.ones(B, device=device)),
            "best_true_gc": orig_gc_tensor.clone(),
            "best_attempt_pred": torch.where(target_label == 1.0,
                                              torch.zeros(B, device=device),
                                              torch.ones(B, device=device)),
            "best_attempt_ids": orig_ids.clone(),
        }

    def _enforce_constraints(self, Z, state):
        nm_b, nm_l = state["nm_b"], state["nm_l"]
        Z.data[state["non_editable_mask"]] = -1e9
        Z.data[nm_b, nm_l, state["orig_ids"][nm_b, nm_l]] = 0.0
        Z.data[state["pad_mask"]] = -1e9

    def _clean_logits(self, Z, state):
        logits_clean = Z.clone()
        for b in range(state["B"]):
            em = state["editable_mask"][b]
            logits_clean[b, em] = logits_clean[b, em].masked_fill(~self.valid_token_mask, -1e9)
        logits_clean[state["non_editable_mask"]] = -1e9
        nm_b, nm_l = state["nm_b"], state["nm_l"]
        logits_clean[nm_b, nm_l, state["orig_ids"][nm_b, nm_l]] = 0.0
        logits_clean[state["pad_mask"]] = -1e9
        return logits_clean

    def _forward(self, logits_clean, tau, mode, K, state):
        B = state["B"]
        L = state["orig_ids"].shape[1]
        attention_mask = state["attention_mask"]
        target_label = state["target_label"]
        target_gc = state["target_gc"]
        margin = state["margin"]

        if mode == "soft":
            pi = F.softmax(logits_clean / tau, dim=-1)          # (B, L, V)
            embeds = torch.matmul(pi, self.embedding_matrix)
            pred_batch = self.gc_model(inputs_embeds=embeds,
                                       attention_mask=attention_mask).unsqueeze(1)  # (B, 1)
            probs_4d = pi.unsqueeze(1)                                              # (B, 1, L, V)

        elif mode == "gumbel":
            logits_mc = logits_clean.unsqueeze(1).expand(B, K, L, self.V_full).reshape(B * K, L, self.V_full)
            pi = F.gumbel_softmax(logits_mc, tau=tau, hard=False, dim=-1)
            y = F.one_hot(pi.argmax(dim=-1), self.V_full).float() + (pi - pi.detach())
            embeds = torch.matmul(y, self.embedding_matrix)
            attn_exp = attention_mask.unsqueeze(1).expand(B, K, L).reshape(B * K, L)
            pred_batch = self.gc_model(inputs_embeds=embeds,
                                       attention_mask=attn_exp).view(B, K)          # (B, K)
            probs_4d = pi.view(B, K, L, self.V_full)

        else:  # st_det
            pi = F.softmax(logits_clean / tau, dim=-1)
            y = F.one_hot(pi.argmax(dim=-1), self.V_full).float() + (pi - pi.detach())
            embeds = torch.matmul(y, self.embedding_matrix)
            pred_batch = self.gc_model(inputs_embeds=embeds,
                                       attention_mask=attention_mask).unsqueeze(1)   # (B, 1)
            probs_4d = pi.unsqueeze(1)

        g_each = torch.where(
            target_label.unsqueeze(1) == 1.0,
            target_gc + margin - pred_batch,
            pred_batch - (target_gc - margin),
        )
        return probs_4d, g_each

    def _expected_hamming(self, probs_4d, state):
        """Expected base-level Hamming via 6-mer hamming_table. Returns (B, K)."""
        pi = probs_4d                                            # (B, K, L, V_full)
        B, K, L, _ = pi.shape
        orig_ids = state["orig_ids"]
        editable_mask = state["editable_mask"]

        P_pure = pi[..., self.pure_token_ids]                   # (B, K, L, V_pure)
        orig_pure = self.global2pure[orig_ids]                  # (B, L)
        results = []
        for b in range(B):
            mask_b = editable_mask[b]
            H_rows = self.hamming_table[orig_pure[b][mask_b]]   # (#edit, V_pure)
            P_edit = P_pure[b, :, mask_b, :]                    # (K, #edit, V_pure)
            results.append((P_edit * H_rows.unsqueeze(0)).sum(-1).sum(-1))
        return torch.stack(results, dim=0)                       # (B, K)

    def _discrete_hamming(self, new_ids, orig_ids, editable_mask):
        """Discrete base-level Hamming using 6-mer table. Supports (B, L)."""
        B = new_ids.shape[0]
        dists = torch.zeros(B, device=new_ids.device)
        for b in range(B):
            idx_new = self.global2pure[new_ids[b]]
            idx_old = self.global2pure[orig_ids[b]]
            mask_b = editable_mask[b] if editable_mask.dim() == 2 else editable_mask
            mask_valid = (idx_new >= 0) & (idx_old >= 0) & mask_b
            dists[b] = self.hamming_table[idx_old[mask_valid], idx_new[mask_valid]].sum()
        return dists

    def _decode_ids(self, ids):
        toks = self.tokenizer.convert_ids_to_tokens(ids)
        return "".join(t for t in toks if t not in self.specials)

    def _true_gc(self, seq):
        bases = [c for c in seq if c in "ACGT"]
        return sum(1 for c in bases if c in "GC") / max(1, len(bases))

    def _eval_discrete(self, Z, state, step, opt):
        device = self.device
        B = state["B"]
        orig_ids = state["orig_ids"]
        editable_mask = state["editable_mask"]
        attention_mask = state["attention_mask"]
        target_label = state["target_label"]
        target_gc = state["target_gc"]
        margin = state["margin"]

        idx_new = Z.argmax(dim=-1)
        idx_new = torch.where(editable_mask, idx_new, orig_ids)
        idx_new = torch.where(attention_mask.bool(), idx_new, orig_ids)

        disc_embeds = self.embedding_matrix[idx_new]
        real_gc_pred = self.gc_model(inputs_embeds=disc_embeds, attention_mask=attention_mask)
        dist_disc = self._discrete_hamming(idx_new, orig_ids, editable_mask)

        true_gc_batch = torch.zeros(B, device=device)
        for b in range(B):
            true_gc_batch[b] = self._true_gc(self._decode_ids(idx_new[b]))

        pred_success = torch.where(
            target_label == 1.0,
            real_gc_pred >= (target_gc + margin),
            real_gc_pred <= (target_gc - margin),
        )
        true_success = torch.where(
            target_label == 1.0,
            true_gc_batch >= target_gc,
            true_gc_batch <= target_gc,
        )

        update_mask = pred_success & ((~state["best_pred_success"]) | (dist_disc < state["best_dist"])) & (dist_disc > 0)
        state["best_dist"] = torch.where(update_mask, dist_disc, state["best_dist"])
        state["best_seq_ids"] = torch.where(update_mask.unsqueeze(1), idx_new, state["best_seq_ids"])
        state["best_pred_success"] = state["best_pred_success"] | update_mask
        state["best_true_success"] = torch.where(update_mask, true_success, state["best_true_success"])
        state["best_pred_gc"] = torch.where(update_mask, real_gc_pred, state["best_pred_gc"])
        state["best_true_gc"] = torch.where(update_mask, true_gc_batch, state["best_true_gc"])
        state["last_improvement_step"] = torch.where(update_mask, step, state["last_improvement_step"])

        closer = torch.where(
            target_label == 1.0,
            real_gc_pred > state["best_attempt_pred"],
            real_gc_pred < state["best_attempt_pred"],
        )
        attempt_update = ~pred_success & closer
        state["best_attempt_pred"] = torch.where(attempt_update, real_gc_pred, state["best_attempt_pred"])
        state["best_attempt_ids"] = torch.where(attempt_update.unsqueeze(1), idx_new, state["best_attempt_ids"])
        state["last_improvement_step"] = torch.where(attempt_update, step, state["last_improvement_step"])

        if state["verbose"] and step % 200 == 0:
            tsr = state["best_true_success"].float().mean().item()
            psr = pred_success.float().mean().item()
            md = state["best_dist"][state["best_pred_success"]].mean().item() if state["best_pred_success"].any() else 0.0
            print(f"  Step {step:04d} | True SR: {tsr:.1%} | Pred SR: {psr:.1%} | Mean Dist: {md:.1f}")

    def _build_output(self, state):
        device = self.device
        orig_ids = state["orig_ids"]
        editable_mask = state["editable_mask"]
        attention_mask = state["attention_mask"]
        target_label = state["target_label"]
        target_gc = state["target_gc"]
        B = state["B"]

        final_ids = torch.where(state["best_pred_success"].unsqueeze(1), state["best_seq_ids"], state["best_attempt_ids"])
        final_dist = self._discrete_hamming(final_ids, orig_ids, editable_mask)

        final_embeds = self.embedding_matrix[final_ids]
        final_pred_gc = self.gc_model(inputs_embeds=final_embeds, attention_mask=attention_mask)
        cf_seqs = [self._decode_ids(final_ids[b]) for b in range(B)]
        cf_true_gc = torch.tensor([self._true_gc(s) for s in cf_seqs], device=device)

        final_true_success = torch.where(target_label == 1.0, cf_true_gc >= target_gc, cf_true_gc <= target_gc)
        final_pred_success = torch.where(target_label == 1.0, final_pred_gc >= target_gc, final_pred_gc <= target_gc)

        return {
            "success": final_true_success.cpu(),
            "pred_success": final_pred_success.cpu(),
            "orig_gc": state["orig_gc_tensor"].cpu(),
            "cf_pred_gc": final_pred_gc.cpu(),
            "cf_true_gc": cf_true_gc.cpu(),
            "orig_ids": orig_ids.cpu(),
            "cf_ids": final_ids.cpu(),
            "edit_distance": final_dist.cpu(),
            "cf_seqs": cf_seqs,
        }

    def generate_cf_batch(self, sequences, target_gc, steps=800,
                          tau_max=2.5, tau_min=0.1, k_soft=10, k_gumbel=400,
                          rho=10.0, margin=0.05, lr=0.5, mc_samples=64,
                          robust_mode="mean", verbose=True):
        """Batch CORAL optimizer for GC counterfactuals."""
        target_info = {"target_gc": target_gc, "margin": margin, "verbose": verbose}
        return self._run_batch(
            inputs=sequences, target_info=target_info,
            steps=steps, tau_max=tau_max, tau_min=tau_min,
            k_soft=k_soft, k_gumbel=k_gumbel, rho=rho,
            mc_samples=mc_samples, robust_mode=robust_mode,
            lr=lr, alpha=0.7, tau_burst=1.5,
        )

    def generate_cf(self, sequence, target_gc=0.6, steps=800, robust_mode="mean",
                 mc_samples=64, lr=0.5, rho_init=20.0, margin=0.05,
                 k_soft=10, k_gumbel=400, tau_max=2.5, tau_min=0.1, verbose=True):
        """Single-sequence CORAL optimization (to be deprecated)."""
        tokens = self.tokenizer(sequence, return_tensors="pt")["input_ids"].to(self.device)[0]
        L = tokens.shape[0]
        orig_ids = tokens.clone()
        editable_mask = (self.global2pure[tokens] >= 0)
        non_editable_mask = ~editable_mask

        Z = torch.nn.Parameter(torch.zeros(L, self.V_full, device=self.device))
        with torch.no_grad():
            Z.data.fill_(math.log(1e-4))
            edit_oh = F.one_hot(orig_ids[editable_mask], num_classes=self.V_full).float()
            Z.data[editable_mask] = torch.log(edit_oh + 1e-4)
            Z.data[editable_mask] = Z.data[editable_mask].masked_fill(~self.valid_token_mask, -1e9)
            Z.data[non_editable_mask] = -1e9
            Z.data[non_editable_mask, orig_ids[non_editable_mask]] = 0.0
            Z.data[editable_mask] += torch.randn_like(Z.data[editable_mask]) * 0.05
        Z_init = Z.data.clone()

        opt = torch.optim.AdamW([Z], lr=lr, fused=True)
        orig_gc = sum(1 for c in sequence if c in "GC") / len(sequence)
        orig_pred_gc = self.gc_model(input_ids=orig_ids.unsqueeze(0),
                                     attention_mask=torch.ones((1, L), device=self.device))
        target_label = 1 if orig_gc < target_gc else 0

        lambda_gc = 0.0
        rho_gc = rho_init
        last_improvement_step = 0
        STAGNATION_THRESH = steps // 4
        burst_every, burst_len, tau_burst = 200, 50, 1.0

        best_sol = {
            "ids": orig_ids.clone(), "dist": float("inf"),
            "pred_gc": orig_pred_gc.item(), "gc": orig_gc,
            "pred_success": False, "success": False, "seq": sequence,
        }
        best_attempt_pred = orig_pred_gc.item()

        if verbose:
            print(f"Seq Len: {len(sequence)} bp | Orig GC: {orig_gc:.3f} | Target: {'>' if target_label else '<'} {target_gc}")

        for step in range(steps):
            opt.zero_grad()
            with torch.no_grad():
                Z.data[non_editable_mask] = -1e9
                Z.data[non_editable_mask, orig_ids[non_editable_mask]] = 0.0

            tau, mode = CORALOptimizer._temperature_and_mode(
                step, k_gumbel, k_soft, tau_min, tau_max, burst_every, burst_len, tau_burst
            )

            logits_clean = Z.clone()
            logits_clean[editable_mask] = logits_clean[editable_mask].masked_fill(~self.valid_token_mask, -1e9)
            logits_clean[non_editable_mask] = -1e9
            logits_clean[non_editable_mask, orig_ids[non_editable_mask]] = 0.0

            if mode == "soft":
                probs = F.softmax(logits_clean / tau, dim=-1)
                embeds = (probs @ self.embedding_matrix).unsqueeze(0)
                pred = self.gc_model(inputs_embeds=embeds, attention_mask=torch.ones((1, L), device=self.device))
                g_each = self.gc_constraint_scalar(pred, target_label, target_gc, margin).unsqueeze(0)
                probs_for_dist = probs.unsqueeze(0)

            elif mode == "gumbel":
                K = mc_samples
                logits_batch = logits_clean.unsqueeze(0).expand(K, -1, -1)
                probs = F.gumbel_softmax(logits_batch, tau=tau, hard=False, dim=-1)
                y = F.one_hot(probs.argmax(-1), self.V_full).float() + (probs - probs.detach())
                embeds = torch.matmul(y, self.embedding_matrix)
                pred = self.gc_model(inputs_embeds=embeds, attention_mask=torch.ones((K, L), device=self.device))
                g_each = self.gc_constraint_scalar(pred, target_label, target_gc, margin)
                probs_for_dist = probs

            else:  # st_det
                probs = F.softmax(logits_clean, dim=-1)
                y = F.one_hot(probs.argmax(-1), self.V_full).float() + (probs - probs.detach())
                embeds = (y @ self.embedding_matrix).unsqueeze(0)
                pred = self.gc_model(inputs_embeds=embeds, attention_mask=torch.ones((1, L), device=self.device))
                g_each = self.gc_constraint_scalar(pred, target_label, target_gc, margin).unsqueeze(0)
                probs_for_dist = probs.unsqueeze(0)

            penalty = (0.5 / rho_gc) * F.relu(lambda_gc + rho_gc * g_each).pow(2)
            if robust_mode == "mean":
                aug_gc, g_dual = penalty.mean(), g_each.mean()
            else:
                aug_gc, g_dual = penalty.max(), g_each.max()

            # Single-sequence: drop batch dim, use 3D _expected_hamming variant
            exp_ham = self._expected_hamming_single(probs_for_dist, orig_ids, editable_mask)
            loss = exp_ham + aug_gc
            loss.backward()
            torch.nn.utils.clip_grad_norm_([Z], 1.0)
            opt.step()

            with torch.no_grad():
                if step % 10 == 0:
                    lambda_gc = max(0.0, lambda_gc + rho_gc * g_dual.item())
                if step % 50 == 0 and step < (steps - 100):
                    viol = max(0.0, g_dual.item())
                    if viol > 0.1:
                        rho_gc = min(100.0, rho_gc * 1.15)
                    elif viol < 0.01:
                        rho_gc = max(rho_init, rho_gc * 0.9)

            with torch.no_grad():
                ids_discrete = logits_clean.argmax(-1)
                ids_discrete[non_editable_mask] = orig_ids[non_editable_mask]
                disc_embeds = self.embedding_matrix[ids_discrete].unsqueeze(0)
                real_gc_pred = self.gc_model(inputs_embeds=disc_embeds,
                                             attention_mask=torch.ones((1, ids_discrete.shape[0]), device=self.device))
                full_seq = self._decode_ids(ids_discrete)
                curr_gc_val = self._true_gc(full_seq)
                dist_val = self._discrete_hamming_single(ids_discrete, orig_ids, editable_mask)

                is_pred_success = bool((real_gc_pred >= target_gc).item() if target_label == 1 else (real_gc_pred <= target_gc).item())
                is_true_success = (curr_gc_val >= target_gc) if target_label == 1 else (curr_gc_val <= target_gc)

                if is_pred_success and not best_sol["pred_success"]:
                    best_sol = {"ids": ids_discrete.clone(), "dist": dist_val, "pred_gc": real_gc_pred.item(),
                                "gc": curr_gc_val, "pred_success": True, "success": is_true_success, "seq": full_seq}
                    last_improvement_step = step
                elif is_pred_success and best_sol["pred_success"] and dist_val < best_sol["dist"]:
                    best_sol = {"ids": ids_discrete.clone(), "dist": dist_val, "pred_gc": real_gc_pred.item(),
                                "gc": curr_gc_val, "pred_success": True, "success": is_true_success, "seq": full_seq}
                    last_improvement_step = step
                elif not is_pred_success and not best_sol["pred_success"]:
                    closer = (real_gc_pred.item() > best_attempt_pred) if target_label == 1 else (real_gc_pred.item() < best_attempt_pred)
                    if closer:
                        best_attempt_pred = real_gc_pred.item()
                        last_improvement_step = step

                if step > k_gumbel and step % 50 == 0:
                    stagnant = (step - last_improvement_step) > STAGNATION_THRESH
                    if stagnant and not best_sol["pred_success"]:
                        if verbose:
                            print(f"  [restart @ step {step}]")
                        Z.data = Z_init.clone() + torch.randn_like(Z.data) * 0.3
                        lambda_gc = 0.0
                        rho_gc = rho_init
                        last_improvement_step = step
                        s = opt.state.get(Z, {})
                        if "exp_avg" in s:
                            s["exp_avg"].zero_()
                            s["exp_avg_sq"].zero_()
                    elif stagnant and best_sol["pred_success"] and best_sol["dist"] > 1:
                        if verbose:
                            print(f"  [refine @ step {step}]")
                        best_cf_oh = F.one_hot(best_sol["ids"], num_classes=self.V_full).float()
                        Z_cf = torch.log(best_cf_oh + 1e-4)
                        Z_cf[editable_mask] = Z_cf[editable_mask].masked_fill(~self.valid_token_mask, -1e9)
                        Z.data = 0.7 * Z_cf + 0.3 * Z_init + torch.randn_like(Z.data) * 0.2
                        last_improvement_step = step
                        s = opt.state.get(Z, {})
                        if "exp_avg" in s:
                            s["exp_avg"].zero_()
                            s["exp_avg_sq"].zero_()

            if verbose and step % 100 == 0:
                print(f"Step {step:03d} | Dist: {dist_val:.1f} | GC: {curr_gc_val:.3f} | PredGC: {real_gc_pred.item():.3f} | Loss: {loss.item():.2f}")

        return best_sol

    def _expected_hamming_single(self, probs_for_dist, orig_ids, editable_mask):
        """Expected Hamming for single-sequence optimize() — handles (1,L,V) and (K,L,V)."""
        if probs_for_dist.dim() == 3:  # (K, L, V)
            K = probs_for_dist.shape[0]
            P_pure = probs_for_dist[..., self.pure_token_ids]  # (K, L, V_pure)
            orig_pure = self.global2pure[orig_ids]
            H_rows = self.hamming_table[orig_pure[editable_mask]]
            P_edit = P_pure[:, editable_mask, :]
            return (P_edit * H_rows.unsqueeze(0)).sum(dim=-1).sum(dim=-1).mean()
        raise ValueError(f"Unsupported shape: {probs_for_dist.shape}")

    def _discrete_hamming_single(self, new_ids, orig_ids, editable_mask):
        """Discrete base-level Hamming for single sequences."""
        idx_new = self.global2pure[new_ids]
        idx_old = self.global2pure[orig_ids]
        mask_valid = (idx_new >= 0) & (idx_old >= 0) & editable_mask
        return self.hamming_table[idx_old[mask_valid], idx_new[mask_valid]].sum().item()

    def gc_constraint_scalar(self, gc_value, target_label, gc_threshold, margin=0.1):
        if target_label == 1:
            return gc_threshold + margin - gc_value
        else:
            return gc_value - (gc_threshold - margin)

    def _get_theoretical_min_edits(self, sequence, current_gc, target_gc_threshold, target_label):
        L_bases = len(sequence)
        current_gc_bases = current_gc * L_bases
        if target_label == 1:
            needed = math.ceil(target_gc_threshold * L_bases) - current_gc_bases
        else:
            needed = current_gc_bases - math.floor(target_gc_threshold * L_bases)
        return max(0, needed)

class GCOracleOneHot(nn.Module):
    """Wraps GenomicGCModel so Ledidi can interface with it.

    Input:  X_edit (B, V_pure, L_edit) — one-hot over pure-ACGT 6-mer vocab
    Output: GC prediction (B, 1)
    """

    def __init__(self, gc_model, embedding_matrix, pure_token_ids, device="cuda"):
        super().__init__()
        self.gc_model = gc_model.eval()
        for p in self.gc_model.parameters():
            p.requires_grad = False
        self.register_buffer("embedding_matrix", embedding_matrix.clone().detach())
        self.register_buffer("pure_token_ids", pure_token_ids.clone().detach())
        self.device = device
        self.orig_ids = None
        self.editable_positions = None
        self.L_full = None

    def set_context(self, orig_ids: torch.Tensor, editable_positions: torch.Tensor):
        self.orig_ids = orig_ids.detach().to(self.embedding_matrix.device)
        self.editable_positions = editable_positions.detach().to(self.embedding_matrix.device)
        self.L_full = int(self.orig_ids.shape[0])

    def clear_context(self):
        self.orig_ids = None
        self.editable_positions = None
        self.L_full = None

    def forward(self, X_edit: torch.Tensor) -> torch.Tensor:
        if self.orig_ids is None:
            raise RuntimeError("Call set_context() before forward().")
        B, V_pure, L_edit = X_edit.shape
        embedding_subset = self.embedding_matrix[self.pure_token_ids]          # (V_pure, D)
        edit_embeds = torch.einsum("bvl,vd->bld", X_edit, embedding_subset)   # (B, L_edit, D)
        orig_embeds = self.embedding_matrix[self.orig_ids]                     # (L_full, D)
        full_embeds = orig_embeds.unsqueeze(0).expand(B, -1, -1).clone()
        full_embeds[:, self.editable_positions, :] = edit_embeds
        attn = torch.ones((B, self.L_full), dtype=torch.long, device=X_edit.device)
        preds = self.gc_model(inputs_embeds=full_embeds, attention_mask=attn)
        if preds.ndim == 1:
            preds = preds.unsqueeze(-1)
        return preds

class LedidiGCOptimizer:
    """Ledidi-based baseline for GC-content counterfactuals."""

    def __init__(self, gc_model, tokenizer, device="cuda"):
        self.gc_model = gc_model
        self.tokenizer = tokenizer
        self.specials = set(self.tokenizer.all_special_tokens)
        self.device = device
        self.vocab_size = tokenizer.vocab_size
        self.embedding_matrix = gc_model.backbone.get_input_embeddings().weight.detach()
        self._precompute_tables()
        self.oracle = GCOracleOneHot(
            gc_model=self.gc_model,
            embedding_matrix=self.embedding_matrix,
            pure_token_ids=self.pure_token_ids,
            device=self.device,
        ).to(self.device)

    def _precompute_tables(self):
        vocab = self.tokenizer.get_vocab()
        pure_token_ids_list = sorted(
            tid for tok, tid in vocab.items() if re.fullmatch(r"[ACGT]{6}", tok)
        )
        self.pure_token_ids = torch.tensor(pure_token_ids_list, device=self.device, dtype=torch.long)
        self.global2pure = torch.full((self.vocab_size,), -1, dtype=torch.long, device=self.device)
        self.global2pure[self.pure_token_ids] = torch.arange(len(pure_token_ids_list), dtype=torch.long, device=self.device)
        self.valid_token_mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.device)
        self.valid_token_mask[self.pure_token_ids] = True

        id2tok = {v: k for k, v in vocab.items()}
        pure_seqs = [id2tok[tid] for tid in pure_token_ids_list]
        table_vals = torch.tensor([["ACGT".index(c) for c in seq] for seq in pure_seqs], dtype=torch.long)
        self.hamming_table = (table_vals.unsqueeze(1) != table_vals.unsqueeze(0)).sum(-1).to(self.device).float()

    def _sequence_to_onehot(self, sequence: str):
        ids = self.tokenizer(sequence, return_tensors="pt")["input_ids"].to(self.device)[0]
        L = ids.shape[0]
        X = torch.zeros(1, self.vocab_size, L, dtype=torch.float32, device=self.device)
        X[0, ids, torch.arange(L, device=self.device)] = 1.0
        return X, ids

    def _compute_input_mask(self, token_ids: torch.Tensor):
        toks = self.tokenizer.convert_ids_to_tokens(token_ids.tolist())
        editable = [bool(re.fullmatch(r"[ACGT]{6}", t)) for t in toks]
        editable_mask = torch.tensor(editable, dtype=torch.bool, device=self.device)
        return ~editable_mask, editable_mask

    def _discrete_hamming(self, new_ids, orig_ids, editable_mask):
        if new_ids.dim() == 1:
            idx_new = self.global2pure[new_ids]
            idx_old = self.global2pure[orig_ids]
            mask_valid = (idx_new >= 0) & (idx_old >= 0) & editable_mask
            return self.hamming_table[idx_old[mask_valid], idx_new[mask_valid]].sum().item()
        B = new_ids.shape[0]
        dists = torch.zeros(B, device=new_ids.device)
        for b in range(B):
            idx_new = self.global2pure[new_ids[b]]
            idx_old = self.global2pure[orig_ids[b]]
            mask_b = editable_mask[b] if editable_mask.dim() == 2 else editable_mask
            mask_valid = (idx_new >= 0) & (idx_old >= 0) & mask_b
            dists[b] = self.hamming_table[idx_old[mask_valid], idx_new[mask_valid]].sum()
        return dists

    def _true_gc(self, seq):
        bases = [c for c in seq if c in "ACGT"]
        return sum(1 for c in bases if c in "GC") / max(1, len(bases))

    def _ids_to_seq(self, ids: torch.Tensor) -> str:
        toks = self.tokenizer.convert_ids_to_tokens(ids.tolist())
        return "".join(t for t in toks if t not in self.specials)

    def _pick_best_candidate(self, X_hat, orig_ids, editable_positions, target_label, target_gc):
        if X_hat.dim() == 2:
            X_hat = X_hat.unsqueeze(0)
        N = X_hat.shape[0]
        full_ids_list = []
        for i in range(N):
            new_edit_ids = self.pure_token_ids[X_hat[i].argmax(dim=0)]
            full_ids = orig_ids.clone()
            full_ids[editable_positions] = new_edit_ids
            full_ids_list.append(full_ids)
        cand_ids = torch.stack(full_ids_list, dim=0)

        with torch.inference_mode():
            embeds = self.embedding_matrix[cand_ids]
            attn = torch.ones((cand_ids.shape[0], cand_ids.shape[1]), dtype=torch.long, device=self.device)
            pred_gc = self.gc_model(inputs_embeds=embeds, attention_mask=attn).view(-1)

            editable_mask_full = torch.zeros_like(orig_ids, dtype=torch.bool)
            editable_mask_full[editable_positions] = True
            dists = self._discrete_hamming(cand_ids, orig_ids.unsqueeze(0).expand_as(cand_ids), editable_mask_full)

            seqs = [self._ids_to_seq(cand_ids[i]) for i in range(N)]
            true_gc = torch.tensor([self._true_gc(s) for s in seqs], device=self.device)
            pred_success = (pred_gc >= target_gc) if target_label == 1 else (pred_gc <= target_gc)
            true_success = (true_gc >= target_gc) if target_label == 1 else (true_gc <= target_gc)
            score = -torch.abs(pred_gc - target_gc)

        feasible = pred_success & (dists > 0)
        if feasible.any():
            feasible_idx = feasible.nonzero(as_tuple=False).flatten()
            min_dist = dists[feasible_idx].min()
            tied = feasible_idx[dists[feasible_idx] == min_dist]
            best_i = tied[torch.argmax(score[tied])].item()
        else:
            best_i = torch.argmax(score).item()

        return {
            "token_ids": cand_ids[best_i],
            "seq": seqs[best_i],
            "gc": float(true_gc[best_i].item()),
            "pred_gc": float(pred_gc[best_i].item()),
            "hamming": float(dists[best_i].item()),
            "success": bool(true_success[best_i].item()),
            "pred_success": bool(pred_success[best_i].item()),
            "n_candidates": N,
        }

    def _get_theoretical_min_edits(self, sequence, current_gc, target_gc_threshold, target_label):
        L_bases = len(sequence)
        current_gc_bases = current_gc * L_bases
        if target_label == 1:
            needed = math.ceil(target_gc_threshold * L_bases) - current_gc_bases
        else:
            needed = current_gc_bases - math.floor(target_gc_threshold * L_bases)
        return max(0, needed)

    def generate_cf(self, sequence: str, target_gc: float, l_input=0.1, tau=1.0,
                    batch_size=16, max_iter=1000, early_stopping_iter=1000, verbose=True):
        orig_gc = sum(1 for c in sequence if c in "GC") / len(sequence)
        target_label = 1 if orig_gc < target_gc else 0

        X, token_ids = self._sequence_to_onehot(sequence)
        _, editable_mask = self._compute_input_mask(token_ids)
        editable_positions = editable_mask.nonzero(as_tuple=False).flatten()
        orig_local = self.global2pure[token_ids[editable_positions]]
        assert (orig_local >= 0).all()
        L_edit = editable_positions.numel()
        V_pure = self.pure_token_ids.numel()

        self.oracle.set_context(orig_ids=token_ids, editable_positions=editable_positions)
        X_edit = torch.zeros(1, V_pure, L_edit, dtype=torch.float32, device=self.device)
        X_edit[0, orig_local, torch.arange(L_edit, device=self.device)] = 1.0
        y_bar = torch.tensor([[target_gc]], dtype=torch.float32, device=self.device)

        designer = Ledidi(
            model=self.oracle,
            shape=X_edit.shape[-2:],
            target=None,
            tau=tau, l=l_input, batch_size=batch_size,
            max_iter=max_iter, early_stopping_iter=early_stopping_iter,
            report_iter=100, lr=1.0,
            input_mask=torch.zeros(L_edit, dtype=torch.bool, device=self.device),
            initial_weights=None, return_history=False, verbose=verbose,
            input_loss=HammingTableLoss(self.hamming_table).to(self.device),
            output_loss=GCMarginLoss(target_label=target_label, target_gc=target_gc, squared=True),
        ).to(self.device)

        X_hat = designer.fit_transform(X_edit, y_bar)
        picked = self._pick_best_candidate(X_hat, token_ids, editable_positions, target_label, target_gc)
        self.oracle.clear_context()
        return picked

    def generate_cf_batch(self, sequences: list, target_gc: float, l_input=0.01, tau=1.0,
                          batch_size=16, max_iter=1000, early_stopping_iter=1000,
                          max_workers=8, verbose=True):
        """Run Ledidi in parallel using ThreadPoolExecutor (awaiting batched implementation)."""
        with torch.no_grad():
            tok = self.tokenizer(sequences[0], return_tensors="pt")
            self.gc_model(input_ids=tok["input_ids"].to(self.device),
                         attention_mask=tok["attention_mask"].to(self.device))

        def _process_single(i):
            return self.generate_cf(sequences[i], target_gc=target_gc, l_input=l_input,
                                    tau=tau, batch_size=batch_size, max_iter=max_iter,
                                    early_stopping_iter=early_stopping_iter, verbose=verbose)

        B = len(sequences)
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
        return results_ordered