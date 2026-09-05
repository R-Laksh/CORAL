"""Finite-state calibration backend for counterfactual distribution search.

Exact backward messages enumerate the supplied state space. Approximate twists
change proposals; importance corrections retain the same terminal target.
Experimental responses never enter this module. This CPU research backend is
separate from CORAL's existing nucleotide optimiser.
"""
from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass
class FiniteEditGraph:
    states: np.ndarray
    sequences: tuple
    kernel: np.ndarray
    distances: np.ndarray

    def __post_init__(self):
        self.states = np.asarray(self.states, dtype=int)
        self.kernel = np.asarray(self.kernel, dtype=float)
        self.distances = np.asarray(self.distances, dtype=float)
        n = len(self.states)
        if self.states.ndim != 2 or len({tuple(s) for s in self.states}) != n:
            raise ValueError("states must be a matrix of distinct categorical states")
        if len(self.sequences) != n or self.kernel.shape != (n, n):
            raise ValueError("sequence and kernel dimensions must match states")
        if (not np.isfinite(self.kernel).all() or (self.kernel < 0).any()
                or not np.allclose(self.kernel.sum(axis=1), 1)):
            raise ValueError("kernel must be finite, nonnegative and row stochastic")
        if (self.distances.shape != (n, n) or not np.isfinite(self.distances).all()
                or (self.distances < 0).any()):
            raise ValueError("distances must be a finite nonnegative square matrix")

    @classmethod
    def from_sequences(cls, states, sequences, stay=0.65, edge_cost=0.03):
        """One-coordinate moves weighted by nucleotide Hamming cost.

        The horizon defines reference dynamics, not an endpoint edit threshold.
        All eight MPRA states are reachable in three moves; the default is six.
        """
        if not 0 <= stay < 1 or edge_cost < 0:
            raise ValueError("require 0 <= stay < 1 and nonnegative edge_cost")
        states = np.asarray(states, dtype=int)
        sequences = tuple(sequences)
        if len({len(s) for s in sequences}) != 1:
            raise ValueError("Hamming cost requires equal sequence lengths")
        seq = np.array([list(s) for s in sequences])
        distances = (seq[:, None, :] != seq[None, :, :]).sum(axis=2).astype(float)
        adjacent = (states[:, None, :] != states[None, :, :]).sum(axis=2) == 1
        weights = adjacent * np.exp(-edge_cost * distances)
        total = weights.sum(axis=1)
        kernel = np.zeros_like(weights)
        connected = total > 0
        kernel[connected] = (1 - stay) * weights[connected] / total[connected, None]
        np.fill_diagonal(kernel, np.where(connected, stay, 1.0))
        return cls(states, sequences, kernel, distances)


def backward_messages(kernel, terminal_weight, steps):
    """Exact h_t = K h_(t+1); rows of K are source states."""
    if steps < 1:
        raise ValueError("steps must be positive")
    terminal = np.asarray(terminal_weight, dtype=float)
    if terminal.ndim != 1 or not np.isfinite(terminal).all() or (terminal < 0).any():
        raise ValueError("terminal weights must be a finite nonnegative vector")
    h = np.empty((steps + 1, len(terminal)))
    h[-1] = terminal
    for t in range(steps - 1, -1, -1):
        h[t] = kernel @ h[t + 1]
    return h


def rollout_messages(kernel, terminal_weight, steps, rollouts, rng):
    """Monte Carlo h estimates, with a positive intermediate support floor.

    Finite rollout failures must not delete legitimate paths. The terminal
    condition remains hard. Work is O(states * rollouts * steps**2).
    """
    if rollouts < 1:
        raise ValueError("rollouts must be positive")
    n = len(terminal_weight)
    h = np.empty((steps + 1, n))
    h[-1] = terminal_weight
    floor = max(float(np.max(terminal_weight)), 1e-30) * 1e-5
    cdf = np.cumsum(kernel, axis=1)
    cdf[:, -1] = 1
    for t in range(steps):
        particles = np.repeat(np.arange(n), rollouts)
        for _ in range(steps - t):
            u = rng.random(len(particles))
            particles = (u[:, None] > cdf[particles]).sum(axis=1)
        estimates = terminal_weight[particles].reshape(n, rollouts).mean(axis=1)
        h[t] = np.maximum(estimates, floor)
    return h


@dataclass(frozen=True)
class Clause:
    """Conjunction of (coordinate, value) literals scoped to one edit graph."""
    literals: tuple

    def matches(self, states):
        result = np.ones(len(states), dtype=bool)
        for coordinate, value in self.literals:
            result &= states[:, coordinate] == value
        return result


def minimal_sufficient_clause(states, witness, target_mask):
    """Smallest sufficient body in this finite graph; not a global ILP rule.

    All satisfying completions must meet the model target. Experimental
    validation belongs to the independent audit layer.
    """
    states = np.asarray(states)
    target_mask = np.asarray(target_mask, dtype=bool)
    if not target_mask[witness]:
        raise ValueError("witness must satisfy the target")
    for size in range(states.shape[1] + 1):
        for coords in combinations(range(states.shape[1]), size):
            clause = Clause(tuple((int(i), int(states[witness, i])) for i in coords))
            if np.all(target_mask[clause.matches(states)]):
                return clause
    raise RuntimeError("distinct states must admit a singleton explanation")


def factorized_projection(states, probabilities):
    """Product of categorical marginals on a complete Cartesian state space."""
    probabilities = np.asarray(probabilities, dtype=float)
    if (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1):
        raise ValueError("probabilities must sum to one")
    expected = int(np.prod([len(np.unique(states[:, j])) for j in range(states.shape[1])]))
    if len(states) != expected or len({tuple(s) for s in states}) != expected:
        raise ValueError("factorized projection requires the complete product space")
    result = np.ones(len(states))
    for j in range(states.shape[1]):
        for value in np.unique(states[:, j]):
            mask = states[:, j] == value
            result[mask] *= probabilities[mask].sum()
    return result


@dataclass
class ParticleResult:
    endpoint_probabilities: np.ndarray
    archive: tuple
    log_normalizer: float
    effective_sample_size: float
    resamplings: int
    failed: bool


class DistributionalCFOptimizer:
    """Twisted SMC for a fixed graph and frozen model score vector."""

    def __init__(self, graph, scores, start, threshold, beta=0.1, steps=6,
                 direction=1, blocked=()):
        self.graph = graph
        self.scores = np.asarray(scores, dtype=float)
        if self.scores.shape != (len(graph.states),) or not np.isfinite(self.scores).all():
            raise ValueError("scores must give a finite prediction for every state")
        if not 0 <= start < len(graph.states) or steps < 1 or beta < 0:
            raise ValueError("invalid start, steps or beta")
        if direction not in (-1, 1) or not np.isfinite(threshold):
            raise ValueError("direction must be +/-1 and threshold must be finite")
        self.start, self.steps, self.beta = int(start), int(steps), float(beta)
        self.threshold, self.direction = float(threshold), direction
        self.blocked = tuple(blocked)
        self.margin = direction * (self.scores - threshold)
        self.feasible = self.margin >= 0
        for clause in self.blocked:
            self.feasible &= ~clause.matches(graph.states)
        self.cost = graph.distances[start]
        self.terminal = np.exp(-beta * self.cost) * self.feasible

    def exact_target(self):
        p = np.zeros(len(self.scores))
        p[self.start] = 1
        for _ in range(self.steps):
            p = p @ self.graph.kernel
        p *= self.terminal
        normalizer = p.sum()
        return (p / normalizer if normalizer > 0 else p), float(normalizer)

    def guidance(self, method, rng, rollouts=16, learned=None):
        if method == "exact":
            return backward_messages(self.graph.kernel, self.terminal, self.steps)
        if method in ("rollout", "defensive_rollout"):
            guide = rollout_messages(self.graph.kernel, self.terminal, self.steps, rollouts, rng)
            if method == "defensive_rollout":
                # Shrink noisy future-success estimates toward a positive
                # reference twist. This changes proposals, not the target.
                guide[:-1] = .95 * guide[:-1] + .05
            return guide
        guide = np.ones((self.steps + 1, len(self.scores)))
        if method == "myopic":
            guide[:-1] = np.exp(-4 * np.maximum(-self.margin, 0) - self.beta * self.cost)
        elif method == "learned":
            if learned is None:
                raise ValueError("learned guidance needs a fitted log-h predictor")
            guide[:-1] = np.exp(np.clip(learned.predict(guidance_features(self)), -30, 0)).reshape(
                self.steps, len(self.scores))
        elif method != "reference":
            raise ValueError(f"unknown guidance method: {method}")
        guide[:-1] = np.maximum(guide[:-1], 1e-30)
        guide[-1] = self.terminal
        return guide

    def sample(self, guide, particles=16, rng=None, resample_fraction=0.5):
        """Propose K(s,s') psi_(t+1)(s'); correct by (K psi_(t+1))/psi_t.

        The intermediate targets telescope to the fixed terminal target.
        Zero-mass rows kill particles; they are not silently reset. Failure of
        a finite particle run is not a certificate of infeasibility.
        """
        if particles < 1 or not 0 <= resample_fraction <= 1:
            raise ValueError("invalid particle count or resampling fraction")
        rng = np.random.default_rng() if rng is None else rng
        guide = np.asarray(guide, dtype=float)
        if (guide.shape != (self.steps + 1, len(self.scores))
                or not np.isfinite(guide).all() or (guide < 0).any()
                or not np.allclose(guide[-1], self.terminal, rtol=1e-12, atol=0)):
            raise ValueError("guide must be finite, nonnegative and end at the terminal weight")
        indices = np.full(particles, self.start, dtype=int)
        weights = np.full(particles, 1 / particles)
        archive = set()
        n_resample, ess = 0, float(particles)
        log_z = np.log(guide[0, self.start]) if guide[0, self.start] > 0 else -np.inf
        if not np.isfinite(log_z):
            return ParticleResult(np.zeros(len(self.scores)), (), -np.inf, 0., 0, True)
        for t in range(self.steps):
            proposals = self.graph.kernel[indices] * guide[t + 1]
            mass = proposals.sum(axis=1)
            denom = guide[t, indices]
            increment = np.divide(mass, denom, out=np.zeros_like(mass), where=denom > 0)
            weights *= increment
            total = weights.sum()
            if total <= 0 or not np.isfinite(total):
                return ParticleResult(np.zeros(len(self.scores)), tuple(sorted(archive)),
                                      -np.inf, 0., n_resample, True)
            weights /= total
            log_z += np.log(total)
            live = mass > 0
            cdf = np.cumsum(proposals[live] / mass[live, None], axis=1)
            cdf[:, -1] = 1
            indices[live] = (rng.random(live.sum())[:, None] > cdf).sum(axis=1)
            archive.update(int(s) for s in indices[live & (weights > 0)] if self.feasible[s])
            ess = float(1 / np.square(weights).sum())
            if t < self.steps - 1 and ess < resample_fraction * particles:
                positions = (rng.random() + np.arange(particles)) / particles
                cdf_w = np.cumsum(weights)
                cdf_w[-1] = 1
                chosen = np.searchsorted(cdf_w, positions, side="right")
                indices = indices[chosen]
                weights.fill(1 / particles)
                n_resample += 1
        histogram = np.bincount(indices, weights=weights, minlength=len(self.scores))
        return ParticleResult(histogram, tuple(sorted(archive)), float(log_z), ess,
                              n_resample, False)


def guidance_features(problem):
    """Local model information for amortised log-h approximation.

    Inputs are current/one-move predictions, costs, orientations, remaining time
    and reference transitions. No experimental labels or backward messages are
    inputs. Exact messages supply training targets on training graphs only.
    """
    graph = problem.graph
    n, dimensions = graph.states.shape
    features = []
    for t in range(problem.steps):
        for s in range(n):
            neighbours = np.flatnonzero((graph.states != graph.states[s]).sum(axis=1) == 1)
            local = []
            for coordinate in range(dimensions):
                matching = [j for j in neighbours if graph.states[j, coordinate] != graph.states[s, coordinate]]
                if len(matching) != 1:
                    raise ValueError("learned-h features require a complete binary cube")
                j = matching[0]
                local.extend([problem.margin[j], problem.cost[j] / 100, graph.kernel[s, j]])
            features.append([
                (problem.steps - t) / problem.steps, problem.steps - t,
                problem.margin[s], problem.cost[s] / 100, problem.beta,
                *graph.states[s], *graph.states[problem.start], *local,
            ])
    return np.asarray(features, dtype=float)


def enumerate_clauses(problem, particles=16, rng=None, method="exact", learned=None,
                      max_rounds=None, rollouts=16):
    """Generate, generalise, block the body, and rerun distributional search.

    Completeness is only relative to the supplied finite graph, target and
    horizon. A particle failure is reported separately from zero residual mass.
    The clause learner here exhaustively examines local categorical conjunctions;
    callers can use the same blocking interface with an external ILP learner.
    """
    rng = np.random.default_rng() if rng is None else rng
    clauses, witnesses = [], []
    limit = len(problem.scores) if max_rounds is None else max_rounds
    if limit < 1:
        raise ValueError("max_rounds must be positive")
    for _ in range(limit):
        residual = DistributionalCFOptimizer(
            problem.graph, problem.scores, problem.start, problem.threshold,
            beta=problem.beta, steps=problem.steps, direction=problem.direction,
            blocked=problem.blocked + tuple(clauses),
        )
        if residual.exact_target()[1] == 0:
            return {"clauses": clauses, "witnesses": witnesses, "status": "finite_support_exhausted"}
        guide = residual.guidance(method, rng, rollouts=rollouts, learned=learned)
        result = residual.sample(guide, particles=particles, rng=rng)
        if not result.archive:
            return {"clauses": clauses, "witnesses": witnesses, "status": "particle_search_failed"}
        witness = min(result.archive, key=lambda s: (problem.cost[s], int(s)))
        clause = minimal_sufficient_clause(problem.graph.states, witness, problem.feasible)
        clauses.append(clause)
        witnesses.append(witness)
    remaining = problem.feasible.copy()
    for clause in clauses:
        remaining &= ~clause.matches(problem.graph.states)
    status = "finite_support_exhausted" if not remaining.any() else "round_limit"
    return {"clauses": clauses, "witnesses": witnesses, "status": status}
