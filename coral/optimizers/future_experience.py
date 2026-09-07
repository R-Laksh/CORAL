"""Reusable hard-sequence experience graph for future-aware search."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from .future_proposal import ProposalBatch

Tensor = torch.Tensor


@dataclass(frozen=True)
class ExperienceEdge:
    child: tuple[int, ...]
    log_q_action: float
    target_expected_edits: float
    use_gradient: bool


@dataclass
class ExperienceNode:
    ids: Tensor
    g: Optional[float] = None
    grad_g: Optional[Tensor] = None
    hamming: Optional[int] = None
    edit_cost: Optional[float] = None
    children: list[ExperienceEdge] = field(default_factory=list)


class SequenceExperienceGraph:
    """Run-local cache of model evaluations and sampled proposal transitions.

    Nodes are deduplicated by hard sequence. Edges intentionally retain duplicate
    draws because they are Monte Carlo samples from q0; collapsing duplicates would
    bias rollout expectations.
    """

    def __init__(self, x0: Tensor, edit_costs: Tensor):
        self.x0 = x0.detach().clone()
        self.edit_costs = edit_costs.detach().clone()
        self.nodes: dict[tuple[int, ...], ExperienceNode] = {}

    @staticmethod
    def key(ids: Tensor) -> tuple[int, ...]:
        return tuple(int(v) for v in ids.tolist())

    def get_or_add(self, ids: Tensor) -> ExperienceNode:
        key = self.key(ids)
        node = self.nodes.get(key)
        if node is None:
            node = ExperienceNode(ids=ids.detach().clone())
            node.hamming = int((ids != self.x0).sum().item())
            pos = torch.arange(ids.shape[0], device=ids.device)
            costs = self.edit_costs.to(ids.device)
            node.edit_cost = float(costs[pos, ids.long()].sum().item())
            self.nodes[key] = node
        return node

    def add_edges(self, parent: Tensor, proposal: ProposalBatch) -> None:
        node = self.get_or_add(parent)
        for child, logq, target_k, use_grad in zip(
            proposal.ids, proposal.log_q_action,
            proposal.target_expected_edits, proposal.use_gradient,
        ):
            key = self.key(child)
            self.get_or_add(child)
            node.children.append(ExperienceEdge(
                child=key, log_q_action=float(logq.item()),
                target_expected_edits=float(target_k.item()),
                use_gradient=bool(use_grad.item()),
            ))
