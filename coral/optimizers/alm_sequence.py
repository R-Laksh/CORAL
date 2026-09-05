"""Sequence-oracle ALM search without enumerating the sequence state space.

Positive approximate h messages guide candidate-pool SMC. Multipliers are
frozen within each episode. The resampling weights correct the proposal; they
do not turn a finite ALM penalty into a pointwise feasibility guarantee.
Only explicitly scored, discrete, jointly feasible candidates enter the CF
archive. Reference paths can undo edits; horizon is not an endpoint edit cap.
"""
from dataclasses import dataclass, asdict
import math
import time

import numpy as np
from scipy.special import logsumexp
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Constraints:
    threshold: np.ndarray
    direction: np.ndarray  # +1 for increase, -1 for decrease
    scale: np.ndarray

    def __post_init__(self):
        self.threshold = np.atleast_1d(self.threshold).astype(float)
        self.direction = np.atleast_1d(self.direction).astype(float)
        self.scale = np.atleast_1d(self.scale).astype(float)
        if not (self.threshold.shape == self.direction.shape == self.scale.shape):
            raise ValueError("Constraint arrays must have identical shapes")
        if not np.isfinite(np.concatenate((self.threshold, self.direction, self.scale))).all():
            raise ValueError("Constraints must be finite")
        if (self.scale <= 0).any() or not np.isin(self.direction, [-1, 1]).all():
            raise ValueError("Positive scales and directions +/-1 required")

    def residual(self, scores):
        return self.direction * (self.threshold - scores) / self.scale

    def torch_residual(self, scores):
        convert = lambda x: torch.as_tensor(x, dtype=scores.dtype, device=scores.device)
        return convert(self.direction) * (convert(self.threshold) - scores) / convert(self.scale)


def alm_energy(edits, residual, multipliers, rho):
    return edits + np.square(np.maximum(0, multipliers + rho * residual)).sum(axis=-1) / (2 * rho)


def candidate_pool_select(candidates, log_psi, previous_log_psi, rng):
    """Extended-space importance step for iid candidate draws from K.

    Pick j proportional to psi_j, then multiply weight by mean(psi_j)/psi_old.
    No evaluation of all K neighbours is needed. Carry the selected *realised*
    message into the next step; recomputing a stochastic denominator is wrong.
    """
    normalizer = logsumexp(log_psi, axis=1)
    probabilities = np.exp(log_psi - normalizer[:, None])
    u = rng.random(len(candidates))
    chosen = (u[:, None] > np.cumsum(probabilities, axis=1)).sum(axis=1)
    chosen = np.minimum(chosen, candidates.shape[1] - 1)
    rows = np.arange(len(candidates))
    return (candidates[rows, chosen], log_psi[rows, chosen],
            normalizer - math.log(candidates.shape[1]) - previous_log_psi)


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Work:
    forward_batches: int = 0
    forward_sequences: int = 0
    backward_batches: int = 0
    backward_sequences: int = 0
    cached_sequences: int = 0


class SequenceOracle(nn.Module):
    """Shared accounting/cache policy and hard-feasibility archive for all methods."""
    def __init__(self, model, source, mutable, alphabet_size, constraints, budget,
                 batch_size=32, device="cpu"):
        super().__init__()
        self.model = model.eval()
        self.source = np.asarray(source, dtype=np.int64)
        self.mutable = np.asarray(mutable, dtype=np.int64)
        if len(self.source.shape) != 1 or len(self.mutable) == 0:
            raise ValueError("A sequence and nonempty editable region are required")
        if len(set(self.mutable)) != len(self.mutable) or (self.mutable < 0).any() or (self.mutable >= len(source)).any():
            raise ValueError("Invalid editable positions")
        self.immutable = np.ones(len(source), dtype=bool)
        self.immutable[self.mutable] = False
        self.alphabet_size = alphabet_size
        self.constraints, self.budget = constraints, int(budget)
        self.batch_size, self.device = batch_size, torch.device(device)
        self.work, self.cache = Work(), {}
        self.best, self.improvements = None, []

    def _observe(self, ids, scores):
        for state, score in zip(ids, scores):
            if not np.array_equal(state[self.immutable], self.source[self.immutable]):
                raise ValueError("Attempt to edit an immutable position")
            if not np.isfinite(score).all():
                raise ValueError("Nonfinite predictor output")
            key = tuple(int(x) for x in state)
            self.cache[key] = np.asarray(score).copy()
            residual = self.constraints.residual(score)
            if (residual <= 0).all():
                edits = int(np.count_nonzero(state != self.source))
                margin = float(-residual.max())
                rank = (edits, -margin, key)
                if self.best is None or rank < self.best["rank"]:
                    self.best = {"ids": state.copy(), "scores": score.copy(),
                                 "edits": edits, "margin": margin, "rank": rank}
                    self.improvements.append({"forward_sequences": self.work.forward_sequences,
                                              "backward_sequences": self.work.backward_sequences,
                                              "edits": edits, "margin": margin})

    def _predict(self, probabilities):
        n = len(probabilities)
        if self.work.forward_sequences + n > self.budget:
            raise BudgetExceeded("Forward-sequence budget exhausted before this batch")
        output = self.model(probabilities)
        if output.ndim == 1:
            output = output[:, None]
        if output.shape != (n, len(self.constraints.threshold)):
            raise ValueError("Predictor/constraint output dimensions disagree")
        self.work.forward_batches += 1
        self.work.forward_sequences += n
        if output.requires_grad:
            def count(gradient):
                self.work.backward_batches += 1
                self.work.backward_sequences += n
                return gradient
            output.register_hook(count)
        return output

    def score_ids(self, ids):
        ids = np.asarray(ids, dtype=np.int64).reshape(-1, len(self.source))
        keys = [tuple(int(x) for x in s) for s in ids]
        missing = list(dict.fromkeys(k for k in keys if k not in self.cache))
        self.work.cached_sequences += len(ids) - len(missing)
        for start in range(0, len(missing), self.batch_size):
            states = np.asarray(missing[start:start + self.batch_size])
            x = F.one_hot(torch.tensor(states, device=self.device), self.alphabet_size).transpose(1, 2).float()
            with torch.no_grad():
                scores = self._predict(x).cpu().numpy()
            self._observe(states, scores)
        return np.array([self.cache[k] for k in keys])

    def forward(self, probabilities):
        detached = probabilities.detach()
        hard = (torch.allclose(detached.sum(1), torch.ones_like(detached[:, 0]), atol=1e-6)
                and bool((torch.minimum(detached.abs(), (detached - 1).abs()) < 1e-6).all()))
        ids = detached.argmax(1).cpu().numpy() if hard else None
        if hard and not probabilities.requires_grad:
            return torch.tensor(self.score_ids(ids), device=self.device, dtype=probabilities.dtype)
        output = self._predict(probabilities)
        if hard:
            self._observe(ids, output.detach().cpu().numpy())
        return output

    def result(self, method, seconds, status, history):
        best = None if self.best is None else {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                              for k, v in self.best.items() if k != "rank"}
        return {"method": method, "status": status, "best": best,
                "work": asdict(self.work), "seconds": seconds,
                "improvements": self.improvements, "history": history}


def reference_step(states, mutable, alphabet_size, rng, stay_probability=.5, kernel="single"):
    """Lazy symmetric single-substitution kernel with reversions allowed."""
    result = states.copy()
    if kernel == "independent":
        # Positive probability of changing any subset in one step. This gives
        # full endpoint support on long editable regions without a hard cap.
        positions = np.asarray(mutable)
        old = result[:, positions]
        new = rng.integers(alphabet_size - 1, size=old.shape)
        new += new >= old
        change = rng.random(old.shape) < (1 - stay_probability) / len(positions)
        result[:, positions] = np.where(change, new, old)
        return result
    if kernel != "single":
        raise ValueError("Unknown reference kernel")
    rows = np.arange(len(states))
    positions = rng.choice(mutable, len(states))
    old = result[rows, positions]
    new = rng.integers(alphabet_size - 1, size=len(states))
    new += new >= old
    change = rng.random(len(states)) >= stay_probability
    result[rows[change], positions[change]] = new[change]
    return result


def run_h_alm(oracle, seed=0, guidance="rollout", particles=8, pool_size=3,
              rollouts=2, horizon=4, episodes=4, temperature=.75,
              rho_initial=10., rho_growth=2., defensive_mix=.02, kernel="single"):
    if guidance not in ("rollout", "rollout_control", "myopic", "reference"):
        raise ValueError("Unknown message estimator")
    if min(particles, pool_size, rollouts, horizon, episodes) < 1 or temperature <= 0:
        raise ValueError("Positive search dimensions and temperature required")
    if rho_initial <= 0 or rho_growth < 1 or not 0 <= defensive_mix < 1:
        raise ValueError("Invalid ALM or defensive-mixture configuration")
    if kernel == "single" and horizon < len(oracle.mutable):
        raise ValueError("Horizon must reach every editable endpoint; do not impose a hidden edit cap")
    rng = np.random.default_rng(seed)
    multipliers = np.zeros(len(oracle.constraints.threshold))
    rho, history, status = rho_initial, [], "completed"
    tic = time.perf_counter()
    try:
        oracle.score_ids(oracle.source[None])
        for episode in range(episodes):
            # Restart at x0: each episode has one explicitly defined path target.
            states = np.repeat(oracle.source[None], particles, axis=0)
            log_weights, previous = np.zeros(particles), np.zeros(particles)
            episode_ess, log_z = [], 0.

            def log_terminal(x):
                score = oracle.score_ids(x)
                distance = np.count_nonzero(x != oracle.source, axis=1)
                return -alm_energy(distance, oracle.constraints.residual(score), multipliers, rho) / temperature

            for step in range(1, horizon + 1):
                pool = reference_step(np.repeat(states, pool_size, axis=0), oracle.mutable,
                                      oracle.alphabet_size, rng, kernel=kernel).reshape(particles, pool_size, -1)
                flat = pool.reshape(-1, len(oracle.source))
                if step == horizon:
                    message = log_terminal(flat)
                elif guidance == "reference":
                    message = np.zeros(len(flat))
                elif guidance == "myopic":
                    message = log_terminal(flat)
                else:
                    leaves = np.repeat(flat, rollouts, axis=0)
                    for _ in range(horizon - step):
                        leaves = reference_step(leaves, oracle.mutable, oracle.alphabet_size, rng, kernel=kernel)
                    values = log_terminal(leaves).reshape(len(flat), rollouts)
                    message = logsumexp(values, axis=1) - math.log(rollouts)
                    if guidance == "rollout_control":
                        message = np.zeros(len(flat))
                if step < horizon and defensive_mix:
                    message = np.logaddexp(math.log(defensive_mix), math.log1p(-defensive_mix) + message)
                states, previous, increment = candidate_pool_select(
                    pool, message.reshape(particles, pool_size), previous, rng)
                log_weights += increment
                weights = np.exp(log_weights - logsumexp(log_weights))
                ess = float(1 / np.square(weights).sum())
                episode_ess.append(ess)
                if step < horizon and ess < particles / 2:
                    log_z += logsumexp(log_weights) - math.log(particles)
                    chosen = rng.choice(particles, particles, p=weights)
                    states, previous = states[chosen], previous[chosen]
                    log_weights = np.zeros(particles)
            endpoint_scores = oracle.score_ids(states)
            residual = (weights[:, None] * oracle.constraints.residual(endpoint_scores)).sum(axis=0)
            history.append({"episode": episode, "rho": rho, "multipliers": multipliers.tolist(),
                            "weighted_residual": residual.tolist(), "ess": episode_ess,
                            "log_normalizer_estimate": float(log_z + logsumexp(log_weights) - math.log(particles))})
            multipliers = np.maximum(0, multipliers + rho * residual)
            if np.maximum(residual, 0).max() > .01:
                rho *= rho_growth
    except BudgetExceeded:
        status = "budget_exhausted"
    if oracle.device.type == "cuda":
        torch.cuda.synchronize(oracle.device)
    return oracle.result("alm_h_" + guidance, time.perf_counter() - tic, status, history)


def run_st_alm(oracle, seed=0, steps=96, batch_size=8, lr=.3,
               tau_max=1., tau_min=.15, dual_every=16, rho_initial=10., rho_growth=2.):
    """Generic ST-Gumbel ALM comparator with annealing and mean MC reduction.

    Uses CORAL's PHR inequality penalty, adapted to arbitrary alphabets and
    vector-valued frozen predictors. This is not the legacy seqgra runner.
    """
    generator = torch.Generator(device=oracle.device).manual_seed(seed)
    source = torch.tensor(oracle.source, device=oracle.device)
    x0 = F.one_hot(source, oracle.alphabet_size).T.float()[None]
    mutable = torch.tensor(oracle.mutable, device=oracle.device)
    logits = nn.Parameter(torch.log(x0[:, :, mutable] + 1e-4))
    optimizer = torch.optim.AdamW([logits], lr=lr)
    multipliers = torch.zeros(len(oracle.constraints.threshold), device=oracle.device)
    rho, history, status, residuals = rho_initial, [], "completed", []
    tic = time.perf_counter()
    try:
        oracle.score_ids(oracle.source[None])
        for step in range(steps):
            tau = tau_min + .5 * (tau_max - tau_min) * (1 + math.cos(math.pi * step / max(steps - 1, 1)))
            noise = -torch.empty((batch_size, oracle.alphabet_size, len(mutable)), device=oracle.device).exponential_(generator=generator).log()
            soft = ((logits + noise) / tau).softmax(1)
            hard = F.one_hot(soft.argmax(1), oracle.alphabet_size).transpose(1, 2).float()
            st = hard - soft.detach() + soft
            full = x0.expand(batch_size, -1, -1).clone()
            full[:, :, mutable] = st
            score = oracle(full)
            residual = oracle.constraints.torch_residual(score)
            edit_cost = (1 - (soft * x0[:, :, mutable]).sum(1)).sum(1)
            penalty = torch.relu(multipliers + rho * residual).square().sum(1) / (2 * rho)
            loss = (edit_cost + penalty).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([logits], 10.)
            optimizer.step()
            residuals.append(residual.detach().mean(0))
            # Audit deterministic hardening too, without accepting soft successes.
            current = oracle.source.copy()
            current[oracle.mutable] = logits.detach().argmax(1)[0].cpu().numpy()
            oracle.score_ids(current[None])
            if (step + 1) % dual_every == 0:
                g = torch.stack(residuals).mean(0)
                history.append({"step": step + 1, "tau": tau, "rho": rho,
                                "multipliers": multipliers.tolist(), "mean_residual": g.tolist()})
                multipliers = torch.relu(multipliers + rho * g)
                if g.clamp_min(0).max() > .01:
                    rho *= rho_growth
                residuals = []
    except BudgetExceeded:
        status = "budget_exhausted"
    if oracle.device.type == "cuda":
        torch.cuda.synchronize(oracle.device)
    return oracle.result("alm_st_gumbel", time.perf_counter() - tic, status, history)


def run_ledidi(oracle, seed=0, steps=96, batch_size=8, edit_weight=.1, lr=1.):
    """Call upstream Ledidi with a one-sided functional hinge loss."""
    from ledidi import Ledidi
    x0 = F.one_hot(torch.tensor(oracle.source, device=oracle.device), oracle.alphabet_size).T.float()[None]
    mask = torch.tensor(oracle.immutable, device=oracle.device)
    hinge = lambda prediction, target: oracle.constraints.torch_residual(prediction).relu().mean()
    editor = Ledidi(oracle, x0.shape[1:], output_loss=hinge, l=edit_weight, tau=1.,
                    batch_size=batch_size, max_iter=steps, early_stopping_iter=steps + 1,
                    input_mask=mask, lr=lr, random_state=seed, verbose=False).to(oracle.device)
    tic, status = time.perf_counter(), "completed"
    try:
        result = editor.fit_transform(x0, torch.tensor(oracle.constraints.threshold,
                                      dtype=torch.float32, device=oracle.device)[None])
        oracle.score_ids(result.detach().argmax(1).cpu().numpy())
    except BudgetExceeded:
        status = "budget_exhausted"
    if oracle.device.type == "cuda":
        torch.cuda.synchronize(oracle.device)
    return oracle.result("ledidi_hinge", time.perf_counter() - tic, status,
                         [{"edit_weight": edit_weight, "tau": 1., "lr": lr}])
