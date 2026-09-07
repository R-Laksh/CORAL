"""Gradient-informed multi-edit proposals for future-aware counterfactual search."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


@dataclass
class MultiEditProposalConfig:
    candidates_per_parent: int = 8
    expected_edits: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    expected_edit_probs: tuple[float, ...] = (0.15, 0.25, 0.35, 0.25)
    functional_strength: float = 1.0
    distance_strength: float = 0.35
    gradient_clip: float = 4.0
    random_mix: float = 0.10
    bias_low: float = -24.0
    bias_high: float = 24.0
    bias_steps: int = 28

    def __post_init__(self) -> None:
        if self.candidates_per_parent < 1:
            raise ValueError("candidates_per_parent must be >= 1")
        if len(self.expected_edits) != len(self.expected_edit_probs) or not self.expected_edits:
            raise ValueError("expected_edits and expected_edit_probs must have equal nonzero length")
        if any(k < 0 for k in self.expected_edits):
            raise ValueError("expected edit counts must be non-negative")
        if any(p < 0 for p in self.expected_edit_probs) or sum(self.expected_edit_probs) <= 0:
            raise ValueError("expected_edit_probs must be non-negative with positive sum")
        if not 0.0 <= self.random_mix <= 1.0:
            raise ValueError("random_mix must lie in [0, 1]")


@dataclass
class ProposalBatch:
    ids: Tensor
    owner: Tensor
    log_q_action: Tensor
    target_expected_edits: Tensor
    use_gradient: Tensor


class GradientMultiEditProposal:
    """Sample complete hard sequences that may change many positions at once.

    Gradients of the *raw constraint* determine mutation preference.  A separate
    change bias is solved so the proposal has a requested expected cardinality.
    This deliberately prevents the magnitude of an ALM multiplier from causing
    proposal size to explode.
    """

    def __init__(self, vocab_size: int, config: MultiEditProposalConfig,
                 editable_mask: Optional[Tensor] = None) -> None:
        self.vocab_size = int(vocab_size)
        self.config = config
        self.editable_mask = editable_mask

    def _editable(self, ids: Tensor) -> Tensor:
        n, L = ids.shape
        if self.editable_mask is None:
            return torch.ones((n, L), dtype=torch.bool, device=ids.device)
        em = self.editable_mask.to(ids.device).bool()
        if em.ndim == 1:
            em = em.unsqueeze(0).expand(n, -1)
        elif em.shape[0] == 1:
            em = em.expand(n, -1)
        if em.shape != ids.shape:
            raise ValueError("editable_mask must broadcast to (N, L)")
        return em

    def _normalized_gain(self, ids: Tensor, grad_g: Tensor, editable: Tensor) -> Tensor:
        current = grad_g.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        gain = -(grad_g - current.unsqueeze(-1))
        current_mask = F.one_hot(ids.long(), num_classes=self.vocab_size).bool()
        allowed = editable.unsqueeze(-1) & ~current_mask
        flat = gain.masked_fill(~allowed, float("nan")).reshape(gain.shape[0], -1)
        mean = torch.nanmean(flat, dim=1)
        var = torch.nanmean((flat - mean[:, None]).square(), dim=1)
        scale = torch.sqrt(var.clamp_min(1e-8))
        norm = (gain - mean[:, None, None]) / scale[:, None, None]
        norm = torch.nan_to_num(norm, nan=0.0,
                                posinf=self.config.gradient_clip,
                                neginf=-self.config.gradient_clip)
        return norm.clamp(-self.config.gradient_clip, self.config.gradient_clip)

    def _base_logits(self, ids: Tensor, grad_g: Tensor, edit_costs: Tensor,
                     use_gradient: Tensor) -> tuple[Tensor, Tensor]:
        editable = self._editable(ids)
        gain = self._normalized_gain(ids, grad_g, editable)
        costs = edit_costs.to(ids.device, dtype=torch.float32)[None, :, :].expand(ids.shape[0], -1, -1)
        current_cost = costs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        delta_cost = costs - current_cost.unsqueeze(-1)
        functional = self.config.functional_strength * gain * use_gradient[:, None, None].to(gain.dtype)
        logits = functional - self.config.distance_strength * delta_cost
        token_ids = torch.arange(self.vocab_size, device=ids.device)[None, None, :]
        allowed = editable.unsqueeze(-1) | (token_ids == ids[:, :, None])
        return logits.masked_fill(~allowed, -1e9), editable

    def _change_bias(self, base: Tensor, ids: Tensor, editable: Tensor,
                     target_k: Tensor) -> Tensor:
        """Solve a scalar bias per sample so E[number changed] ~= target_k."""
        B, _, V = base.shape
        current_mask = F.one_hot(ids.long(), num_classes=V).bool()
        target = torch.minimum(target_k.float(), editable.sum(1).float()).clamp_min(0.0)
        lo = torch.full((B,), self.config.bias_low, device=ids.device)
        hi = torch.full((B,), self.config.bias_high, device=ids.device)
        for _ in range(self.config.bias_steps):
            mid = 0.5 * (lo + hi)
            logits = base + (~current_mask).to(base.dtype) * mid[:, None, None]
            p = F.softmax(logits, dim=-1)
            p_stay = p.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
            e = ((1.0 - p_stay) * editable.to(p.dtype)).sum(1)
            too_low = e < target
            lo = torch.where(too_low, mid, lo)
            hi = torch.where(too_low, hi, mid)
        return 0.5 * (lo + hi)

    def sample(self, parent_ids: Tensor, grad_g: Tensor, edit_costs: Tensor,
               generator: torch.Generator,
               candidates_per_parent: Optional[int] = None) -> ProposalBatch:
        """Draw hard full-sequence candidates from a tractable auxiliary proposal.

        ``log_q_action`` is the joint log-probability of the sampled cardinality
        component, gradient/random mixture component, and all categorical token
        draws. Keeping those auxiliaries makes later importance correction possible
        without pretending to know the many-to-one marginal probability of a hard
        sequence.
        """
        n, L = parent_ids.shape
        m = int(candidates_per_parent or self.config.candidates_per_parent)
        owner = torch.arange(n, device=parent_ids.device).repeat_interleave(m)
        ids, grads = parent_ids[owner], grad_g[owner]

        k_probs = torch.as_tensor(self.config.expected_edit_probs, dtype=torch.float32,
                                  device=ids.device)
        k_probs /= k_probs.sum()
        k_idx = torch.multinomial(k_probs, n * m, replacement=True, generator=generator)
        k_values = torch.as_tensor(self.config.expected_edits, dtype=torch.float32, device=ids.device)
        target_k = k_values[k_idx]
        log_p_k = torch.log(k_probs[k_idx].clamp_min(1e-30))

        r = self.config.random_mix
        if r <= 0:
            use_gradient = torch.ones(n * m, dtype=torch.bool, device=ids.device)
            log_p_component = torch.zeros(n * m, device=ids.device)
        elif r >= 1:
            use_gradient = torch.zeros(n * m, dtype=torch.bool, device=ids.device)
            log_p_component = torch.zeros(n * m, device=ids.device)
        else:
            use_gradient = torch.rand(n * m, generator=generator, device=ids.device) >= r
            pg = torch.full((n * m,), 1.0 - r, device=ids.device)
            pr = torch.full((n * m,), r, device=ids.device)
            log_p_component = torch.where(use_gradient, pg.log(), pr.log())

        base, editable = self._base_logits(ids, grads, edit_costs, use_gradient)
        bias = self._change_bias(base, ids, editable, target_k)
        current_mask = F.one_hot(ids.long(), num_classes=self.vocab_size).bool()
        logits = base + (~current_mask).to(base.dtype) * bias[:, None, None]
        log_probs = F.log_softmax(logits, dim=-1)
        sampled = torch.multinomial(log_probs.exp().reshape(n * m * L, self.vocab_size),
                                    1, generator=generator).reshape(n * m, L)
        log_q_tokens = log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1).sum(-1)
        return ProposalBatch(sampled, owner,
                             log_p_k + log_p_component + log_q_tokens,
                             target_k, use_gradient)
