"""Generic straight-through Gumbel-Softmax + ALM search baseline.

This keeps the functional constraint machinery identical to the h-guided search
so experiments can isolate the search primitive. It is a predictor-agnostic
benchmark adapter, not a replacement for CORAL's domain-specific optimizers.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .alm_h import ALMState, SearchResult, augmented_penalty

Tensor = torch.Tensor


@dataclass
class STALMConfig:
    steps: int = 250
    lr: float = 0.2
    tau_start: float = 1.5
    tau_end: float = 0.2
    mc_samples: int = 4
    original_logit_bias: float = 3.0
    dual_every: int = 10
    rho_every: int = 50
    rho_growth: float = 1.15
    rho_decay: float = 0.9
    violation_high: float = 0.1
    violation_low: float = 0.01
    discrete_eval_every: int = 1
    seed: int = 0


class STGumbelALMSearch:
    """Straight-through Gumbel categorical optimization with CORAL-style ALM."""

    def __init__(self, constraint_fn, vocab_size: int, config: STALMConfig | None = None,
                 editable_mask: Tensor | None = None) -> None:
        self.constraint_fn = constraint_fn
        self.vocab_size = int(vocab_size)
        self.config = config or STALMConfig()
        self.editable_mask = editable_mask
        self._forward_evals = 0
        self._backward_evals = 0

    def _masked_logits(self, logits: Tensor, x0: Tensor) -> Tensor:
        if self.editable_mask is None:
            return logits
        em = self.editable_mask.to(logits.device).bool()
        if em.ndim != 1 or em.shape[0] != x0.shape[0]:
            raise ValueError("editable_mask must have shape (L,)")
        allowed = torch.ones_like(logits, dtype=torch.bool)
        fixed = ~em
        if fixed.any():
            allowed[fixed] = False
            allowed[fixed, x0[fixed]] = True
        return logits.masked_fill(~allowed, -1e9)

    @staticmethod
    def _temperature(step: int, cfg: STALMConfig) -> float:
        if cfg.steps <= 1:
            return cfg.tau_end
        p = step / (cfg.steps - 1)
        return cfg.tau_start * (cfg.tau_end / cfg.tau_start) ** p

    def _hard_constraint(self, ids: Tensor) -> Tensor:
        x = F.one_hot(ids.long(), num_classes=self.vocab_size).to(torch.float32)
        with torch.no_grad():
            g = self.constraint_fn(x).reshape(-1)
        self._forward_evals += int(ids.shape[0])
        return g

    @staticmethod
    def _dual_update(state: ALMState, g_signal: float, cfg: STALMConfig, step: int) -> None:
        if (step + 1) % cfg.dual_every == 0:
            state.lam = max(0.0, state.lam + state.rho * g_signal)
        if (step + 1) % cfg.rho_every == 0:
            viol = max(0.0, g_signal)
            if viol > cfg.violation_high:
                state.rho = min(state.rho * cfg.rho_growth, state.rho_max)
            elif viol < cfg.violation_low:
                state.rho = max(state.rho * cfg.rho_decay, state.rho_min)

    def run(self, x0: Tensor, state: ALMState | None = None) -> SearchResult:
        if x0.ndim != 1:
            raise ValueError("Prototype expects one initial sequence at a time")
        cfg = self.config
        torch.manual_seed(cfg.seed)
        state = state or ALMState()
        self._forward_evals = 0
        self._backward_evals = 0
        L, V = x0.shape[0], self.vocab_size
        base = torch.zeros((L, V), dtype=torch.float32, device=x0.device)
        base.scatter_(1, x0[:, None], float(cfg.original_logit_bias))
        delta = torch.nn.Parameter(torch.zeros_like(base))
        opt = torch.optim.Adam([delta], lr=cfg.lr)
        archive: dict[tuple[int, ...], float] = {}

        g0 = self._hard_constraint(x0[None, :])
        if float(g0[0]) <= 0:
            archive[tuple(int(v) for v in x0.tolist())] = float(g0[0])
        original_oh = F.one_hot(x0.long(), num_classes=V).to(torch.float32)

        for step in range(cfg.steps):
            opt.zero_grad(set_to_none=True)
            logits = self._masked_logits(base + delta, x0)
            tau = self._temperature(step, cfg)
            samples = F.gumbel_softmax(
                logits.unsqueeze(0).expand(cfg.mc_samples, -1, -1),
                tau=tau, hard=True, dim=-1,
            )
            g = self.constraint_fn(samples).reshape(cfg.mc_samples)
            self._forward_evals += cfg.mc_samples
            d_edit = (1.0 - (samples * original_oh.unsqueeze(0)).sum(dim=-1)).sum(dim=-1)
            loss = (d_edit + augmented_penalty(g, state.lam, state.rho)).mean()
            loss.backward()
            self._backward_evals += cfg.mc_samples
            torch.nn.utils.clip_grad_norm_([delta], 1.0)
            opt.step()
            self._dual_update(state, float(g.detach().mean().item()), cfg, step)

            if (step + 1) % cfg.discrete_eval_every == 0 or step + 1 == cfg.steps:
                with torch.no_grad():
                    hard_ids = self._masked_logits(base + delta, x0).argmax(dim=-1)
                g_hard = self._hard_constraint(hard_ids[None, :])
                if float(g_hard[0]) <= 0:
                    key = tuple(int(v) for v in hard_ids.tolist())
                    archive[key] = min(float(g_hard[0]), archive.get(key, float("inf")))

        if archive:
            x0_list = x0.tolist()
            def rank(key):
                return sum(int(a != b) for a, b in zip(key, x0_list)), archive[key]
            keys = sorted(archive, key=rank)
            seqs = [torch.tensor(k, dtype=x0.dtype, device=x0.device) for k in keys]
            best = seqs[0]
            return SearchResult(
                best_ids=best,
                best_hamming=int((best != x0).sum().item()),
                best_constraint=float(archive[keys[0]]),
                feasible_found=True,
                archive_ids=seqs,
                archive_constraint=[archive[k] for k in keys],
                final_state=state,
                model_forward_evals=self._forward_evals,
                model_backward_evals=self._backward_evals,
            )
        return SearchResult(
            best_ids=None, best_hamming=None, best_constraint=None,
            feasible_found=False, archive_ids=[], archive_constraint=[],
            final_state=state, model_forward_evals=self._forward_evals,
            model_backward_evals=self._backward_evals,
        )
