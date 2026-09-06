"""Exact edit-regret utilities for finite measured fitness landscapes."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np


@dataclass(frozen=True)
class EditRegret:
    achieved_edits: int | None
    optimal_edits: int | None
    regret: int | None
    target_feasible_in_measured_landscape: bool


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        raise ValueError("Sequences must have equal length")
    return sum(x != y for x, y in zip(a, b))


def exact_min_edits(
    source: str,
    sequences: Iterable[str],
    measured_scores: Iterable[float],
    target: float,
    direction: str = "increase",
) -> int | None:
    """Minimum Hamming distance to any *measured* endpoint satisfying the target."""
    seqs = list(sequences)
    scores = np.asarray(list(measured_scores), dtype=float)
    if len(seqs) != len(scores):
        raise ValueError("sequences and scores must have the same length")
    if direction == "increase":
        ok = scores >= float(target)
    elif direction == "decrease":
        ok = scores <= float(target)
    else:
        raise ValueError("direction must be 'increase' or 'decrease'")
    if not np.any(ok):
        return None
    return min(hamming(source, seq) for seq, good in zip(seqs, ok) if bool(good))


def edit_regret(
    source: str,
    candidate: str | None,
    sequences: Iterable[str],
    measured_scores: Iterable[float],
    target: float,
    direction: str = "increase",
) -> EditRegret:
    optimum = exact_min_edits(source, sequences, measured_scores, target, direction)
    if optimum is None:
        return EditRegret(None if candidate is None else hamming(source, candidate), None, None, False)
    if candidate is None:
        return EditRegret(None, optimum, None, True)
    achieved = hamming(source, candidate)
    return EditRegret(achieved, optimum, achieved - optimum, True)
