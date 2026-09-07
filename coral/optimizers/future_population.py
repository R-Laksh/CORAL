"""Future-aware multi-edit population search for constrained counterfactuals.

Particles are complete hard sequences; one transition can change many positions.
The immediate search energy is the same edit-cost + ALM objective used by CORAL,
while a lambda-independent rollout value gives proposal credit to states with good
future completions. ALM multipliers are frozen within each episode and updated only
between episodes.

This is intentionally an optimizer, not yet an exact twisted-SMC sampler. The
experience/proposal interfaces retain the information needed for later path-space
importance correction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
import math

import torch
import torch.nn.functional as F

from .alm_h import ALMState, augmented_penalty
from .future_experience import SequenceExperienceGraph
from .future_proposal import GradientMultiEditProposal, MultiEditProposalConfig

Tensor = torch.Tensor
ConstraintFn = Callable[[Tensor], Tensor]


@dataclass
class TeacherConfig:
    rollout_depth: int = 2
    rollout_width: int = 4
    rollout_max_children: int = 12
    rollout_add_per_revisit: int = 2
    roots_per_parent: int = 3
    random_roots_per_parent: int = 1
    beta_violation: float = 4.0
    beta_edit: float = 0.0
    feasibility_bonus: float = 4.0
    violation_scale: Optional[float] = None

    def __post_init__(self) -> None:
        if self.rollout_depth < 0 or self.rollout_width < 0:
            raise ValueError("rollout_depth/rollout_width must be non-negative")
        if self.rollout_max_children < self.rollout_width:
            raise ValueError("rollout_max_children must be >= rollout_width")
        if self.rollout_add_per_revisit < 0:
            raise ValueError("rollout_add_per_revisit must be non-negative")
        if self.roots_per_parent < 0:
            raise ValueError("roots_per_parent must be non-negative")
        if not 0 <= self.random_roots_per_parent <= self.roots_per_parent:
            raise ValueError("random_roots_per_parent must be in [0, roots_per_parent]")


@dataclass
class PopulationSearchConfig:
    particles: int = 24
    episodes: int = 5
    rounds_per_episode: int = 4
    guidance_strength: float = 1.5
    energy_temperature: float = 1.0
    resample_ess_fraction: float = 0.50
    retain_parents: bool = True
    dual_every: int = 1
    rho_growth: float = 1.15
    rho_decay: float = 0.90
    violation_high: float = 0.10
    violation_low: float = 0.01
    seed: int = 0
    proposal: MultiEditProposalConfig = field(default_factory=MultiEditProposalConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)


@dataclass
class EpisodeStats:
    episode: int
    lambda_start: float
    lambda_end: float
    rho_start: float
    rho_end: float
    mean_g: float
    min_g: float
    feasible_fraction: float
    best_feasible_edits: Optional[int]
    best_feasible_edit_cost: Optional[float]
    mean_selection_ess: float
    mean_proposal_changes: float
    p95_proposal_changes: float
    mean_selected_hamming: float
    graph_nodes: int
    forward_evals: int
    backward_evals: int
    forward_batches: int
    backward_batches: int
    cache_g_hits: int
    cache_grad_hits: int


@dataclass
class PopulationSearchResult:
    best_ids: Optional[Tensor]
    best_hamming: Optional[int]
    best_edit_cost: Optional[float]
    best_constraint: Optional[float]
    feasible_found: bool
    archive_ids: list[Tensor]
    archive_constraint: list[float]
    final_state: ALMState
    history: list[EpisodeStats]
    model_forward_evals: int
    model_backward_evals: int
    model_forward_batches: int
    model_backward_batches: int
    graph_nodes: int
    cache_g_hits: int
    cache_grad_hits: int


class FutureAwarePopulationSearch:
    """ALM-constrained population optimizer with multi-edit q0 and rollout h."""

    def __init__(self, constraint_fn: ConstraintFn, vocab_size: int,
                 config: Optional[PopulationSearchConfig] = None,
                 editable_mask: Optional[Tensor] = None,
                 edit_costs: Optional[Tensor] = None) -> None:
        self.constraint_fn = constraint_fn
        self.vocab_size = int(vocab_size)
        self.config = config or PopulationSearchConfig()
        self.editable_mask = editable_mask
        self.edit_costs = edit_costs
        self.proposal = GradientMultiEditProposal(
            self.vocab_size, self.config.proposal, editable_mask=editable_mask)
        self.graph: Optional[SequenceExperienceGraph] = None
        self._teacher_scale = 1.0
        self._forward_evals = self._backward_evals = 0
        self._forward_batches = self._backward_batches = 0
        self._cache_g_hits = self._cache_grad_hits = 0

    def _one_hot(self, ids: Tensor, requires_grad: bool = False) -> Tensor:
        x = F.one_hot(ids.long(), num_classes=self.vocab_size).float()
        return x.detach().requires_grad_(True) if requires_grad else x

    def _evaluate_hard(self, ids: Tensor) -> Tensor:
        assert self.graph is not None
        out = torch.empty(ids.shape[0], dtype=torch.float32, device=ids.device)
        missing, keys = [], []
        positions: dict[tuple[int, ...], list[int]] = {}
        for i, row in enumerate(ids):
            node = self.graph.get_or_add(row)
            key = self.graph.key(row)
            positions.setdefault(key, []).append(i)
            if node.g is not None:
                self._cache_g_hits += 1
                out[i] = node.g
            elif key not in keys:
                missing.append(row); keys.append(key)
        if missing:
            batch = torch.stack(missing)
            with torch.no_grad():
                g = self.constraint_fn(self._one_hot(batch)).reshape(-1)
            self._forward_evals += len(missing); self._forward_batches += 1
            for key, gv in zip(keys, g):
                self.graph.nodes[key].g = float(gv.item())
                for i in positions[key]: out[i] = gv
        return out

    def _evaluate_gradient(self, ids: Tensor) -> tuple[Tensor, Tensor]:
        assert self.graph is not None
        g_out = torch.empty(ids.shape[0], dtype=torch.float32, device=ids.device)
        grad_out = torch.empty((*ids.shape, self.vocab_size), dtype=torch.float32, device=ids.device)
        missing, keys = [], []
        positions: dict[tuple[int, ...], list[int]] = {}
        for i, row in enumerate(ids):
            node = self.graph.get_or_add(row); key = self.graph.key(row)
            positions.setdefault(key, []).append(i)
            if node.g is not None and node.grad_g is not None:
                self._cache_grad_hits += 1
                g_out[i] = node.g; grad_out[i] = node.grad_g.to(ids.device)
            elif key not in keys:
                missing.append(row); keys.append(key)
        if missing:
            batch = torch.stack(missing); x = self._one_hot(batch, requires_grad=True)
            g = self.constraint_fn(x).reshape(-1)
            grad = torch.autograd.grad(g.sum(), x)[0]
            self._forward_evals += len(missing); self._backward_evals += len(missing)
            self._forward_batches += 1; self._backward_batches += 1
            for key, gv, gr in zip(keys, g.detach(), grad.detach()):
                node = self.graph.nodes[key]; node.g = float(gv.item()); node.grad_g = gr.cpu()
                for i in positions[key]: g_out[i] = gv; grad_out[i] = gr
        return g_out, grad_out

    def _log_merit_from_g(self, ids: Tensor, g: Tensor) -> Tensor:
        """Lambda-independent teacher signal: future feasibility, optionally edit-aware."""
        assert self.graph is not None
        c = self.config.teacher
        edit = torch.tensor([self.graph.get_or_add(x).edit_cost for x in ids],
                            dtype=torch.float32, device=ids.device)
        viol = F.relu(g) / max(self._teacher_scale, 1e-8)
        return (-c.beta_edit * edit - c.beta_violation * viol
                + c.feasibility_bonus * (g <= 0).float())

    def _log_merit(self, ids: Tensor) -> Tensor:
        return self._log_merit_from_g(ids, self._evaluate_hard(ids))

    def _ensure_children(self, row: Tensor, generator: torch.Generator,
                         expanded: set[tuple[int, ...]]) -> None:
        """Grow, rather than replace, the Monte Carlo support cached at a state.

        The first visit draws ``rollout_width`` samples. Later top-level teacher
        calls add a small number of fresh q0 samples up to ``rollout_max_children``.
        ``expanded`` prevents repeated recursive visits in one backup from consuming
        the entire growth budget immediately.
        """
        assert self.graph is not None
        key = self.graph.key(row)
        if key in expanded:
            return
        expanded.add(key)
        node = self.graph.get_or_add(row)
        c = self.config.teacher
        if len(node.children) == 0:
            need = min(c.rollout_width, c.rollout_max_children)
        else:
            need = min(c.rollout_add_per_revisit,
                       max(0, c.rollout_max_children - len(node.children)))
        if need <= 0:
            return
        _, grad = self._evaluate_gradient(row[None, :])
        batch = self.proposal.sample(row[None, :], grad, self.graph.edit_costs,
                                     generator, candidates_per_parent=need)
        self._evaluate_hard(batch.ids)
        self.graph.add_edges(row, batch)

    def _log_h_single(self, ids: Tensor, depth: int, generator: torch.Generator,
                      memo: dict[tuple[tuple[int, ...], int, int], float],
                      expanded: set[tuple[int, ...]]) -> float:
        assert self.graph is not None
        node = self.graph.get_or_add(ids)
        base = float(self._log_merit(ids[None, :])[0])
        if depth <= 0 or (node.g is not None and node.g <= 0):
            return base
        self._ensure_children(ids, generator, expanded)
        edges = node.children[:self.config.teacher.rollout_max_children]
        if not edges:
            return base
        key = (self.graph.key(ids), depth, len(edges))
        if key in memo:
            return memo[key]
        vals = [self._log_h_single(self.graph.nodes[e.child].ids.to(ids.device), depth - 1,
                                   generator, memo, expanded) for e in edges]
        future = torch.logsumexp(torch.tensor(vals), 0).item() - math.log(len(vals))
        # Explicit no-op/current-state branch makes the backup conservative when
        # all sampled futures are poor.
        value = torch.logsumexp(torch.tensor([future, base]), 0).item() - math.log(2.0)
        memo[key] = float(value)
        return float(value)

    def _candidate_log_h(self, candidates: Tensor, owner: Tensor, immediate: Tensor,
                         generator: torch.Generator) -> Tensor:
        """Refine a small mixture of greedy and exploratory roots per parent.

        Greedy roots exploit the shaped feasibility signal. Random roots protect
        against exactly the epistatic valley case where the useful first macro-edit
        looks neutral or deleterious before a later coordinated completion.
        """
        c = self.config.teacher
        if c.rollout_depth <= 0 or c.roots_per_parent <= 0:
            return immediate
        out = immediate.clone()
        n = int(owner.max()) + 1
        memo: dict[tuple[tuple[int, ...], int, int], float] = {}
        expanded: set[tuple[int, ...]] = set()
        for p in range(n):
            group = torch.where(owner == p)[0]
            if not len(group):
                continue
            roots = min(c.roots_per_parent, len(group))
            n_random = min(c.random_roots_per_parent, roots)
            n_top = roots - n_random
            chosen_global: list[int] = []
            if n_top:
                local_top = torch.topk(immediate[group], n_top).indices
                chosen_global.extend(group[local_top].tolist())
            if n_random:
                chosen_set = set(chosen_global)
                remaining = torch.tensor([int(j) for j in group.tolist() if int(j) not in chosen_set],
                                         dtype=torch.long, device=candidates.device)
                if len(remaining):
                    perm = torch.randperm(len(remaining), generator=generator,
                                          device=candidates.device)
                    chosen_global.extend(remaining[perm[:min(n_random, len(remaining))]].tolist())
            for idx in chosen_global:
                idx = int(idx)
                out[idx] = self._log_h_single(candidates[idx], c.rollout_depth,
                                              generator, memo, expanded)
        return out

    def _energy(self, ids: Tensor, g: Tensor, state: ALMState) -> Tensor:
        assert self.graph is not None
        edit = torch.tensor([self.graph.get_or_add(x).edit_cost for x in ids],
                            dtype=torch.float32, device=ids.device)
        return edit + augmented_penalty(g, state.lam, state.rho)

    @staticmethod
    def _ess(weights: Tensor) -> float:
        w = weights / weights.sum().clamp_min(1e-12)
        return float(1.0 / w.square().sum().clamp_min(1e-12))

    def _select(self, candidates: Tensor, logw: Tensor,
                generator: torch.Generator) -> tuple[Tensor, float]:
        stable = logw - logw.max(); weights = stable.exp().clamp_min(1e-30)
        ess = self._ess(weights); probs = weights / weights.sum()
        replace = ess < self.config.resample_ess_fraction * len(candidates)
        idx = torch.multinomial(probs, self.config.particles,
                                replacement=replace, generator=generator)
        return candidates[idx], ess

    def _archive(self, ids: Tensor, g: Tensor, archive: dict[tuple[int, ...], float]) -> None:
        assert self.graph is not None
        for row, gv in zip(ids[g <= 0], g[g <= 0]):
            key = self.graph.key(row)
            archive[key] = min(float(gv), archive.get(key, float("inf")))

    def _dual_update(self, state: ALMState, mean_g: float) -> None:
        c = self.config
        state.lam = max(0.0, state.lam + state.rho * mean_g)
        viol = max(0.0, mean_g)
        if viol > c.violation_high: state.rho = min(state.rho * c.rho_growth, state.rho_max)
        elif viol < c.violation_low: state.rho = max(state.rho * c.rho_decay, state.rho_min)

    def run(self, x0: Tensor, state: Optional[ALMState] = None) -> PopulationSearchResult:
        if x0.ndim != 1: raise ValueError("x0 must have shape (L,)")
        c = self.config; device = x0.device
        proposal_rng = torch.Generator(device=device).manual_seed(c.seed)
        teacher_rng = torch.Generator(device=device).manual_seed(c.seed + 1_000_003)
        select_rng = torch.Generator(device=device).manual_seed(c.seed + 2_000_003)
        if self.edit_costs is None:
            costs = torch.ones((len(x0), self.vocab_size), device=device)
            costs.scatter_(1, x0[:, None], 0.0)
        else:
            costs = self.edit_costs.to(device=device, dtype=torch.float32)
            if costs.shape != (len(x0), self.vocab_size):
                raise ValueError("edit_costs must have shape (L, vocab_size)")
        self.graph = SequenceExperienceGraph(x0, costs)
        self._forward_evals = self._backward_evals = self._forward_batches = self._backward_batches = 0
        self._cache_g_hits = self._cache_grad_hits = 0
        state = state or ALMState(); archive: dict[tuple[int, ...], float] = {}; history = []
        g0 = self._evaluate_hard(x0[None, :])
        self._teacher_scale = (float(c.teacher.violation_scale) if c.teacher.violation_scale is not None
                               else max(abs(float(g0[0])), 1e-3))
        self._archive(x0[None, :], g0, archive)
        particles = x0[None, :].repeat(c.particles, 1)

        for ep in range(c.episodes):
            lam0, rho0 = state.lam, state.rho; esss = []; changes = []; selected_h = []
            for _ in range(c.rounds_per_episode):  # dual state frozen in this block
                _, grad = self._evaluate_gradient(particles)
                batch = self.proposal.sample(particles, grad, self.graph.edit_costs, proposal_rng)
                candidates, owner = batch.ids, batch.owner
                if c.retain_parents:
                    candidates = torch.cat([candidates, particles], dim=0)
                    owner = torch.cat([owner, torch.arange(c.particles, device=device)], dim=0)
                g = self._evaluate_hard(candidates); self._archive(candidates, g, archive)
                logw = -self._energy(candidates, g, state) / max(c.energy_temperature, 1e-8)
                if c.guidance_strength:
                    immediate = self._log_merit_from_g(candidates, g)
                    logh = self._candidate_log_h(candidates, owner, immediate, teacher_rng)
                    logw = logw + c.guidance_strength * logh
                changes.append((candidates != particles[owner]).sum(-1).float().cpu())
                particles, ess = self._select(candidates, logw, select_rng); esss.append(ess)
                selected_h.append((particles != x0).sum(-1).float().cpu())

            g = self._evaluate_hard(particles); self._archive(particles, g, archive)
            mean_g = float(g.mean())
            if (ep + 1) % c.dual_every == 0: self._dual_update(state, mean_g)
            best_h = min((self.graph.nodes[k].hamming for k in archive), default=None)
            best_c = min((self.graph.nodes[k].edit_cost for k in archive), default=None)
            ch = torch.cat(changes) if changes else torch.zeros(1)
            sh = torch.cat(selected_h) if selected_h else torch.zeros(1)
            history.append(EpisodeStats(
                ep, float(lam0), float(state.lam), float(rho0), float(state.rho), mean_g,
                float(g.min()), float((g <= 0).float().mean()), best_h, best_c,
                float(sum(esss) / max(len(esss), 1)), float(ch.mean()),
                float(torch.quantile(ch, .95)), float(sh.mean()), len(self.graph.nodes),
                self._forward_evals, self._backward_evals, self._forward_batches,
                self._backward_batches, self._cache_g_hits, self._cache_grad_hits))

        if archive:
            keys = sorted(archive, key=lambda k: (self.graph.nodes[k].edit_cost,
                                                  self.graph.nodes[k].hamming, archive[k]))
            seqs = [self.graph.nodes[k].ids.to(device) for k in keys]; key = keys[0]
            return PopulationSearchResult(
                seqs[0], self.graph.nodes[key].hamming, self.graph.nodes[key].edit_cost,
                archive[key], True, seqs, [archive[k] for k in keys], state, history,
                self._forward_evals, self._backward_evals, self._forward_batches,
                self._backward_batches, len(self.graph.nodes), self._cache_g_hits,
                self._cache_grad_hits)
        return PopulationSearchResult(
            None, None, None, None, False, [], [], state, history,
            self._forward_evals, self._backward_evals, self._forward_batches,
            self._backward_batches, len(self.graph.nodes), self._cache_g_hits,
            self._cache_grad_hits)
