import math
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

class CORALOptimizer(ABC):
    @abstractmethod
    def _init_state(self, inputs, target_info: dict,
                    rho_init: float, steps: int, alpha: float) -> dict:
        """Set up all optimization state for a batch.

        Returns a dict with at minimum:
          Z                   : nn.Parameter  — logit tensor
          Z_init              : Tensor         — copy for restart/refine
          orig_ids            : Tensor (B, L)  — original token/base indices
          editable_mask       : Tensor (B, L)  — bool, editable positions
          lambda_gc           : Tensor (B,)    — dual variables
          rho_vec             : Tensor (B,)    — penalty weights
          rho_init            : float
          last_improvement_step: Tensor (B,)  — step of last improvement
          STAGNATION_THRESH   : int
          alpha               : float          — refine interpolation weight
          B                   : int
          + domain-specific best-solution trackers
        """

    @abstractmethod
    def _enforce_constraints(self, Z: torch.nn.Parameter, state: dict) -> None:
        """Clamp hard constraints on Z in-place (non-editable positions, padding)."""

    @abstractmethod
    def _clean_logits(self, Z: torch.Tensor, state: dict) -> torch.Tensor:
        """Mask invalid vocabulary entries; return logits_clean."""

    @abstractmethod
    def _forward(self, logits_clean: torch.Tensor, tau: float, mode: str,
                 K: int, state: dict) -> tuple:
        """Domain-specific forward pass.

        Returns:
          probs_4d : (B, K, L, V) — soft token distributions for Hamming
          g_each   : (B, K)       — constraint violation (>0 = violated)
        """

    @abstractmethod
    def _expected_hamming(self, probs_4d: torch.Tensor, state: dict) -> torch.Tensor:
        """Expected Hamming distance. Returns (B, K)."""

    @abstractmethod
    def _eval_discrete(self, Z: torch.Tensor, state: dict,
                       step: int, opt: torch.optim.Optimizer) -> None:
        """Evaluate argmax solution; update best-solution fields in state in-place."""

    @abstractmethod
    def _build_output(self, state: dict) -> dict:
        """Assemble and return the final result dict."""

    @staticmethod
    def _temperature_and_mode(step: int, k_gumbel: int, k_soft: int,
                               tau_min: float, tau_max: float,
                               burst_every: int = 200, burst_len: int = 50,
                               tau_burst: float = 1.0):
        """Cosine-annealed temperature with burst re-exploration after k_gumbel."""
        if step < k_gumbel:
            progress = step / k_gumbel
            tau = tau_min + (tau_max - tau_min) * 0.5 * (1 + math.cos(math.pi * progress))
            mode = "soft" if step < k_soft else "gumbel"
        else:
            tau = tau_min
            mode = "st_det"
            if (((step - k_gumbel) // burst_every) % 2 == 0
                    and (step - k_gumbel) % burst_every < burst_len):
                mode = "gumbel"
                tau = tau_burst
        return tau, mode

    @staticmethod
    def _alm_penalty(g_each: torch.Tensor, lambda_gc: torch.Tensor,
                     rho_vec: torch.Tensor, robust_mode: str):
        """ALM augmented penalty.

        Returns:
          aug_gc : (B,) — per-sample penalty
          g_dual : (B,) — dual gradient signal
        """
        rho_bk = rho_vec.unsqueeze(1)
        lam_bk = lambda_gc.unsqueeze(1)
        penalty_each = (0.5 / rho_bk) * F.relu(lam_bk + rho_bk * g_each).pow(2)
        if robust_mode == "mean":
            return penalty_each.mean(dim=1), g_each.mean(dim=1)
        elif robust_mode == "worst":
            return penalty_each.max(dim=1).values, g_each.max(dim=1).values
        else:
            raise ValueError(f"Unknown robust_mode: {robust_mode}")

    @staticmethod
    def _clip_gradients(Z: torch.nn.Parameter, B: int) -> None:
        """Per-sample gradient norm clipping (in-place)."""
        clip_coef = 1.0 / (Z.grad.reshape(B, -1).norm(p=2, dim=1) + 1e-6)
        clip_coef = torch.clamp(clip_coef, max=1.0)
        Z.grad.mul_(clip_coef.view(B, 1, 1))

    @staticmethod
    def _dual_update(state: dict, g_dual: torch.Tensor, step: int) -> None:
        """Update dual variables lambda_gc and rho_vec in state in-place."""
        rho_init = state["rho_init"]
        if step % 10 == 0:
            state["lambda_gc"] = F.relu(
                state["lambda_gc"] + state["rho_vec"].detach() * g_dual.detach()
            )
        if step % 50 == 0:
            viol = F.relu(g_dual.detach())
            state["rho_vec"] = torch.where(
                viol > 0.1,
                torch.clamp(state["rho_vec"] * 1.15, max=100.0),
                state["rho_vec"],
            )
            state["rho_vec"] = torch.where(
                viol < 0.01,
                torch.clamp(state["rho_vec"] * 0.9, min=rho_init),
                state["rho_vec"],
            )

    def _run_batch(self, inputs, target_info: dict, steps: int,
                   tau_max: float, tau_min: float, k_soft: int, k_gumbel: int,
                   rho: float, mc_samples: int, robust_mode: str,
                   lr: float, alpha: float, tau_burst: float) -> dict:
        state = self._init_state(inputs, target_info, rho_init=rho, steps=steps, alpha=alpha)
        Z = state["Z"]
        B = state["B"]
        opt = torch.optim.AdamW([Z], lr=lr, fused=True)

        for step in range(steps):
            opt.zero_grad()
            self._enforce_constraints(Z, state)

            tau, mode = self._temperature_and_mode(
                step, k_gumbel, k_soft, tau_min, tau_max, tau_burst=tau_burst
            )
            logits_clean = self._clean_logits(Z, state)
            probs_4d, g_each = self._forward(logits_clean, tau, mode, mc_samples, state)

            aug_gc, g_dual = self._alm_penalty(
                g_each, state["lambda_gc"], state["rho_vec"], robust_mode
            )
            d_edit = self._expected_hamming(probs_4d, state).mean(dim=1)

            loss = (d_edit + aug_gc).sum()
            loss.backward()

            with torch.no_grad():
                self._clip_gradients(Z, B)
            opt.step()

            with torch.no_grad():
                self._dual_update(state, g_dual, step)

            with torch.inference_mode():
                self._eval_discrete(Z, state, step, opt)

        return self._build_output(state)
