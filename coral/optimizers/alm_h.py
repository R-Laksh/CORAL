"""ALM-compatible, h-guided search over discrete sequence edits.

This module is deliberately predictor-agnostic. A caller supplies a differentiable
constraint function ``g(x)`` where ``g <= 0`` means the requested functional target
is satisfied. Search uses whole discrete sequences, an edit-local reference kernel,
and an optional approximation to the backward information function h_t.

The augmented-Lagrangian state is held fixed while one particle episode runs and is
updated only between episodes. This keeps the episode's twisted target stationary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

Tensor = torch.Tensor
ConstraintFn = Callable[[Tensor], Tensor]
LogHFn = Callable[[Tensor, Tensor, "ALMState", int], Tensor]


@dataclass
class ALMState:
    lam: float = 0.0
    rho: float = 1.0
    rho_min: float = 1.0
    rho_max: float = 100.0


@dataclass
class SearchConfig:
    particles: int = 32
    episodes: int = 8
    horizon: int = 6
    proposal_width: int = 24
    lookahead_width: int = 8
    guidance_strength: float = 1.0
    guidance_estimator: str = "shared_rollout"
    gradient_proposal_strength: float = 0.5
    move_temperature: float = 1.0
    energy_temperature: float = 1.0
    dual_every: int = 1
    rho_growth: float = 1.15
    rho_decay: float = 0.9
    violation_high: float = 0.1
    violation_low: float = 0.01
    seed: int = 0


@dataclass
class SearchResult:
    best_ids: Optional[Tensor]
    best_hamming: Optional[int]
    best_constraint: Optional[float]
    feasible_found: bool
    archive_ids: list[Tensor]
    archive_constraint: list[float]
    final_state: ALMState
    model_forward_evals: int
    model_backward_evals: int


def augmented_penalty(g: Tensor, lam: float | Tensor, rho: float | Tensor) -> Tensor:
    """Rockafellar-style inequality ALM penalty used by CORAL.

    ``g <= 0`` is feasible. This is the same functional form as
    ``CORALOptimizer._alm_penalty`` before reduction over Monte Carlo samples.
    """
    lam_t = torch.as_tensor(lam, dtype=g.dtype, device=g.device)
    rho_t = torch.as_tensor(rho, dtype=g.dtype, device=g.device)
    return 0.5 / rho_t * F.relu(lam_t + rho_t * g).square()


class ALMHTwistedSearch:
    """Particle search with an ALM terminal energy and optional h-guidance.

    The state space is categorical sequences. A local transition changes at most one
    coordinate. The *horizon* is a search-computation horizon, not a hard
    admissibility constraint: no-op and reversion transitions are allowed, and final
    candidates are ranked by actual Hamming distance to ``x0``.

    ``constraint_fn`` receives one-hot tensors with shape ``(N, L, V)`` and returns
    one scalar violation per sequence. It may wrap BPNet, a protein LM + head, or a
    synthetic oracle, as long as gradients with respect to the one-hot input exist.
    """

    def __init__(
        self,
        constraint_fn: ConstraintFn,
        vocab_size: int,
        config: SearchConfig | None = None,
        editable_mask: Tensor | None = None,
        log_h_fn: LogHFn | None = None,
    ) -> None:
        self.constraint_fn = constraint_fn
        self.vocab_size = int(vocab_size)
        self.config = config or SearchConfig()
        self.editable_mask = editable_mask
        self.log_h_fn = log_h_fn
        self._forward_evals = 0
        self._backward_evals = 0

    def _one_hot(self, ids: Tensor, requires_grad: bool = False) -> Tensor:
        x = F.one_hot(ids.long(), num_classes=self.vocab_size).to(torch.float32)
        if requires_grad:
            x = x.detach().requires_grad_(True)
        return x

    @staticmethod
    def _hamming(ids: Tensor, x0: Tensor) -> Tensor:
        return (ids != x0.unsqueeze(0)).sum(dim=-1).to(torch.float32)

    def _energy_ids(self, ids: Tensor, x0: Tensor, state: ALMState) -> tuple[Tensor, Tensor]:
        x = self._one_hot(ids)
        with torch.no_grad():
            g = self.constraint_fn(x).reshape(-1)
        self._forward_evals += int(ids.shape[0])
        e = self._hamming(ids, x0) + augmented_penalty(g, state.lam, state.rho)
        return e, g

    def _energy_and_gradient(
        self, ids: Tensor, x0: Tensor, state: ALMState
    ) -> tuple[Tensor, Tensor, Tensor]:
        x = self._one_hot(ids, requires_grad=True)
        g = self.constraint_fn(x).reshape(-1)
        self._forward_evals += int(ids.shape[0])

        orig = F.one_hot(x0.long(), num_classes=self.vocab_size).to(x.dtype)
        d_soft = (1.0 - (x * orig.unsqueeze(0)).sum(dim=-1)).sum(dim=-1)
        e = d_soft + augmented_penalty(g, state.lam, state.rho)
        grad = torch.autograd.grad(e.sum(), x, create_graph=False, retain_graph=False)[0]
        self._backward_evals += int(ids.shape[0])
        return e.detach(), g.detach(), grad.detach()

    def _top_moves(
        self,
        ids: Tensor,
        grad: Tensor,
        x0: Tensor,
        width: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return top single-coordinate moves and first-order energy deltas."""
        n, L = ids.shape
        V = self.vocab_size
        current_grad = grad.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        delta = grad - current_grad.unsqueeze(-1)

        mask = torch.zeros_like(delta, dtype=torch.bool)
        mask.scatter_(-1, ids.unsqueeze(-1), True)
        editable = self.editable_mask
        if editable is not None:
            em = editable.to(ids.device).bool()
            if em.ndim == 1:
                em = em.unsqueeze(0).expand(n, -1)
            elif em.shape[0] == 1:
                em = em.expand(n, -1)
            mask |= ~em.unsqueeze(-1)
        delta = delta.masked_fill(mask, float("inf"))

        flat = delta.reshape(n, L * V)
        k = min(int(width), L * (V - 1))
        vals, idx = torch.topk(flat, k=k, dim=-1, largest=False)
        pos = idx // V
        tok = idx % V

        owner = torch.arange(n, device=ids.device).repeat_interleave(k)
        next_ids = ids[owner].clone()
        next_ids[torch.arange(n * k, device=ids.device), pos.reshape(-1)] = tok.reshape(-1)
        return next_ids, vals.reshape(-1), owner

    def _reference_logits(
        self,
        directional_delta: Tensor,
        next_ids: Tensor,
        owner: Tensor,
        parent_ids: Tensor,
        x0: Tensor,
    ) -> Tensor:
        cfg = self.config
        before = self._hamming(parent_ids, x0)
        after = self._hamming(next_ids, x0)
        step_delta_h = after - before[owner]
        return (
            -step_delta_h / max(cfg.move_temperature, 1e-6)
            - cfg.gradient_proposal_strength * directional_delta
        )

    def estimate_log_h_one_step(
        self,
        candidate_ids: Tensor,
        x0: Tensor,
        state: ALMState,
    ) -> Tensor:
        """Expensive teacher approximation to one-step log h_t.

        For each candidate y, approximate

            h(y) = E_{Z ~ P0(.|y)} [ exp(-E_ALM(Z) / tau_E) ]

        over a local reference kernel containing a no-op plus top single edits.
        Candidate-level gradients make this useful for debugging but too expensive
        for the intended large biological-model implementation.
        """
        cfg = self.config
        e_y, _, grad_y = self._energy_and_gradient(candidate_ids, x0, state)
        moves, dgrad, owner = self._top_moves(
            candidate_ids, grad_y, x0, width=cfg.lookahead_width
        )
        e_z, _ = self._energy_ids(moves, x0, state)

        n = candidate_ids.shape[0]
        k = moves.shape[0] // n
        move_logits = self._reference_logits(
            dgrad, moves, owner, candidate_ids, x0
        ).reshape(n, k)
        ref_logits = torch.cat(
            [torch.zeros((n, 1), device=e_y.device), move_logits], dim=1
        )
        log_p = F.log_softmax(ref_logits, dim=1)
        energies = torch.cat([e_y[:, None], e_z.reshape(n, k)], dim=1)
        return torch.logsumexp(
            log_p - energies / max(cfg.energy_temperature, 1e-6), dim=1
        )

    def estimate_log_h_shared_rollout(
        self,
        candidate_ids: Tensor,
        candidate_owner: Tensor,
        parent_ids: Tensor,
        parent_moves: Tensor,
        parent_move_delta: Tensor,
        parent_move_owner: Tensor,
        x0: Tensor,
        state: ALMState,
    ) -> Tensor:
        """Cheap one-step lookahead using a proposal shared with each parent.

        This estimator does not run a backward pass from every candidate. It reuses
        the parent's gradient-ranked edit operations, applies a small set as possible
        continuations of each candidate, and evaluates all resulting hard endpoints
        in one batched forward pass. The approximation can miss a second edit that
        only becomes attractive after the first edit.
        """
        cfg = self.config
        n_parent = parent_ids.shape[0]
        moves_per_parent = parent_moves.shape[0] // n_parent
        k2 = min(cfg.lookahead_width, moves_per_parent)

        pm = parent_moves.reshape(n_parent, moves_per_parent, -1)[:, :k2]
        pd = parent_move_delta.reshape(n_parent, moves_per_parent)[:, :k2]
        pbase = parent_ids[:, None, :].expand_as(pm)
        diff = pm != pbase
        pos = diff.to(torch.int64).argmax(dim=-1)
        tok = pm.gather(-1, pos.unsqueeze(-1)).squeeze(-1)

        n_cand = candidate_ids.shape[0]
        cand_parent = candidate_owner.long()
        pos_c = pos[cand_parent]
        tok_c = tok[cand_parent]
        dgrad_c = pd[cand_parent]

        endpoints = candidate_ids[:, None, :].repeat(1, k2, 1)
        rows = torch.arange(n_cand, device=candidate_ids.device)[:, None].expand(-1, k2)
        cols = torch.arange(k2, device=candidate_ids.device)[None, :].expand(n_cand, -1)
        endpoints[rows, cols, pos_c] = tok_c
        endpoints_flat = endpoints.reshape(n_cand * k2, -1)

        before = self._hamming(candidate_ids, x0)
        after = self._hamming(endpoints_flat, x0).reshape(n_cand, k2)
        step_delta_h = after - before[:, None]
        move_logits = (
            -step_delta_h / max(cfg.move_temperature, 1e-6)
            - cfg.gradient_proposal_strength * dgrad_c
        )
        ref_logits = torch.cat(
            [torch.zeros((n_cand, 1), device=candidate_ids.device), move_logits], dim=1
        )
        log_p = F.log_softmax(ref_logits, dim=1)

        all_endpoints = torch.cat(
            [candidate_ids[:, None, :], endpoints], dim=1
        ).reshape(n_cand * (k2 + 1), -1)
        e, _ = self._energy_ids(all_endpoints, x0, state)
        energies = e.reshape(n_cand, k2 + 1)
        return torch.logsumexp(
            log_p - energies / max(cfg.energy_temperature, 1e-6), dim=1
        )

    @staticmethod
    def _sample_grouped(
        logits: Tensor, owner: Tensor, n_owner: int, generator: torch.Generator
    ) -> Tensor:
        counts = torch.bincount(owner, minlength=n_owner)
        if not torch.all(counts == counts[0]):
            raise ValueError("Grouped proposal currently expects equal candidate counts")
        k = int(counts[0].item())
        probs = F.softmax(logits.reshape(n_owner, k), dim=1)
        choice = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        return torch.arange(n_owner, device=owner.device) * k + choice

    def _update_archive(
        self,
        ids: Tensor,
        g: Tensor,
        x0: Tensor,
        archive: dict[tuple[int, ...], float],
    ) -> None:
        feasible = g <= 0
        for seq, gv in zip(ids[feasible], g[feasible]):
            key = tuple(int(v) for v in seq.tolist())
            archive[key] = min(float(gv.item()), archive.get(key, float("inf")))

    def _dual_update(self, state: ALMState, g_signal: float) -> None:
        cfg = self.config
        state.lam = max(0.0, state.lam + state.rho * g_signal)
        viol = max(0.0, g_signal)
        if viol > cfg.violation_high:
            state.rho = min(state.rho * cfg.rho_growth, state.rho_max)
        elif viol < cfg.violation_low:
            state.rho = max(state.rho * cfg.rho_decay, state.rho_min)

    def run(self, x0: Tensor, state: ALMState | None = None) -> SearchResult:
        """Search from one initial categorical sequence ``x0`` of shape ``(L,)``."""
        if x0.ndim != 1:
            raise ValueError("Prototype expects one initial sequence at a time")
        cfg = self.config
        if cfg.guidance_estimator not in {"shared_rollout", "exact_one_step"}:
            raise ValueError(
                "guidance_estimator must be 'shared_rollout' or 'exact_one_step'"
            )
        self._forward_evals = 0
        self._backward_evals = 0
        device = x0.device
        gen = torch.Generator(device=device)
        gen.manual_seed(cfg.seed)
        state = state or ALMState()
        particles = x0.unsqueeze(0).repeat(cfg.particles, 1)
        archive: dict[tuple[int, ...], float] = {}

        _, g0 = self._energy_ids(particles[:1], x0, state)
        self._update_archive(particles[:1], g0, x0, archive)

        for ep in range(cfg.episodes):
            # Freeze dual state inside an episode so h refers to one objective.
            for _ in range(cfg.horizon):
                _, _, grad = self._energy_and_gradient(particles, x0, state)
                moves, dgrad, owner = self._top_moves(
                    particles, grad, x0, width=cfg.proposal_width
                )
                move_ref_logits = self._reference_logits(
                    dgrad, moves, owner, particles, x0
                )
                k = moves.shape[0] // cfg.particles

                candidates = torch.cat(
                    [particles[:, None, :], moves.reshape(cfg.particles, k, -1)], dim=1
                ).reshape(cfg.particles * (k + 1), -1)
                cand_owner = torch.arange(
                    cfg.particles, device=device
                ).repeat_interleave(k + 1)
                ref_logits = torch.cat(
                    [
                        torch.zeros((cfg.particles, 1), device=device),
                        move_ref_logits.reshape(cfg.particles, k),
                    ],
                    dim=1,
                ).reshape(-1)

                if cfg.guidance_strength != 0.0:
                    if self.log_h_fn is not None:
                        log_h = self.log_h_fn(
                            candidates, x0, state, cfg.horizon
                        ).reshape(-1)
                    elif cfg.guidance_estimator == "shared_rollout":
                        log_h = self.estimate_log_h_shared_rollout(
                            candidates,
                            cand_owner,
                            particles,
                            moves,
                            dgrad,
                            owner,
                            x0,
                            state,
                        )
                    else:
                        log_h = self.estimate_log_h_one_step(candidates, x0, state)
                    proposal_logits = ref_logits + cfg.guidance_strength * log_h
                else:
                    proposal_logits = ref_logits

                picked = self._sample_grouped(
                    proposal_logits, cand_owner, cfg.particles, gen
                )
                particles = candidates[picked]
                _, g_step = self._energy_ids(particles, x0, state)
                self._update_archive(particles, g_step, x0, archive)

            _, g_final = self._energy_ids(particles, x0, state)
            self._update_archive(particles, g_final, x0, archive)
            if (ep + 1) % cfg.dual_every == 0:
                self._dual_update(state, float(g_final.mean().item()))

        if archive:
            seqs = [torch.tensor(k, dtype=x0.dtype, device=device) for k in archive]
            seqs.sort(
                key=lambda s: (
                    int((s != x0).sum().item()),
                    archive[tuple(int(v) for v in s.tolist())],
                )
            )
            best = seqs[0]
            best_key = tuple(int(v) for v in best.tolist())
            return SearchResult(
                best_ids=best,
                best_hamming=int((best != x0).sum().item()),
                best_constraint=float(archive[best_key]),
                feasible_found=True,
                archive_ids=seqs,
                archive_constraint=[
                    archive[tuple(int(v) for v in s.tolist())] for s in seqs
                ],
                final_state=state,
                model_forward_evals=self._forward_evals,
                model_backward_evals=self._backward_evals,
            )
        return SearchResult(
            best_ids=None,
            best_hamming=None,
            best_constraint=None,
            feasible_found=False,
            archive_ids=[],
            archive_constraint=[],
            final_state=state,
            model_forward_evals=self._forward_evals,
            model_backward_evals=self._backward_evals,
        )
