import statistics

import torch

from coral.optimizers.alm_h import ALMState
from coral.optimizers.future_population import (
    FutureAwarePopulationSearch,
    GradientMultiEditProposal,
    MultiEditProposalConfig,
    PopulationSearchConfig,
    SequenceExperienceGraph,
    TeacherConfig,
)


def _costs(x0: torch.Tensor, vocab_size: int = 2) -> torch.Tensor:
    out = torch.ones((x0.numel(), vocab_size), dtype=torch.float32)
    out.scatter_(1, x0[:, None], 0.0)
    return out


def _and_constraint(x: torch.Tensor) -> torch.Tensor:
    score = x[:, 0, 1] * x[:, 1, 1]
    return 0.8 - score


def _additive_constraint(x: torch.Tensor) -> torch.Tensor:
    return 2.5 - x[:, :, 1].sum(dim=-1)


def test_multi_edit_proposal_controls_expected_cardinality():
    cfg = MultiEditProposalConfig(
        candidates_per_parent=1000,
        expected_edits=(4.0,), expected_edit_probs=(1.0,),
        random_mix=1.0, distance_strength=0.0,
    )
    proposal = GradientMultiEditProposal(2, cfg)
    x0 = torch.zeros(20, dtype=torch.long)
    parents = x0[None, :]
    grad = torch.zeros((1, 20, 2))
    batch = proposal.sample(
        parents, grad, _costs(x0), torch.Generator().manual_seed(3)
    )
    edits = (batch.ids != x0).sum(dim=-1).float()
    assert abs(float(edits.mean()) - 4.0) < 0.3
    assert torch.isfinite(batch.log_q_action).all()


def test_custom_edit_cost_is_distinct_from_changed_position_count():
    x0 = torch.zeros(4, dtype=torch.long)
    costs = torch.zeros((4, 2), dtype=torch.float32)
    costs[:, 1] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    graph = SequenceExperienceGraph(x0, costs)
    node = graph.get_or_add(torch.tensor([1, 1, 0, 0]))
    assert node.hamming == 2
    assert node.edit_cost == 3.0


def test_experience_graph_preserves_duplicate_rollout_samples():
    x0 = torch.zeros(3, dtype=torch.long)
    graph = SequenceExperienceGraph(x0, _costs(x0))
    parent = x0.clone()
    child = torch.tensor([1, 0, 0])
    from coral.optimizers.future_proposal import ProposalBatch
    batch = ProposalBatch(
        ids=torch.stack([child, child]), owner=torch.zeros(2, dtype=torch.long),
        log_q_action=torch.tensor([-1.0, -1.0]),
        target_expected_edits=torch.ones(2),
        use_gradient=torch.ones(2, dtype=torch.bool),
    )
    graph.add_edges(parent, batch)
    assert len(graph.get_or_add(parent).children) == 2
    assert len(graph.nodes) == 2


def test_hard_and_gradient_evaluations_are_cached():
    cfg = PopulationSearchConfig(
        particles=2, episodes=1, rounds_per_episode=1,
        proposal=MultiEditProposalConfig(
            candidates_per_parent=2, expected_edits=(1.0,),
            expected_edit_probs=(1.0,), random_mix=1.0,
        ),
        teacher=TeacherConfig(rollout_depth=0),
    )
    search = FutureAwarePopulationSearch(_additive_constraint, 2, cfg)
    x0 = torch.zeros(6, dtype=torch.long)
    search.graph = SequenceExperienceGraph(x0, _costs(x0))
    repeated = torch.stack([x0, x0])

    search._evaluate_hard(repeated)
    forward = search._forward_evals
    search._evaluate_hard(repeated)
    assert search._forward_evals == forward

    search._evaluate_gradient(repeated)
    backward = search._backward_evals
    search._evaluate_gradient(repeated)
    assert search._backward_evals == backward
    assert search._cache_g_hits > 0
    assert search._cache_grad_hits > 0


def test_rollout_value_credits_partial_gateway_state():
    cfg = PopulationSearchConfig(
        particles=2, episodes=1, rounds_per_episode=1,
        proposal=MultiEditProposalConfig(
            candidates_per_parent=4, expected_edits=(1.0,),
            expected_edit_probs=(1.0,), random_mix=1.0,
            distance_strength=0.0,
        ),
        teacher=TeacherConfig(
            rollout_depth=1, rollout_width=64, rollout_max_children=64,
            roots_per_parent=2,
            beta_violation=4.0, beta_edit=0.0, feasibility_bonus=4.0,
        ),
    )
    search = FutureAwarePopulationSearch(_and_constraint, 2, cfg)
    x0 = torch.zeros(4, dtype=torch.long)
    search.graph = SequenceExperienceGraph(x0, _costs(x0))
    search._teacher_scale = 0.8
    gen = torch.Generator().manual_seed(2)
    start = search._log_h_single(torch.tensor([0, 0, 0, 0]), 1, gen, {}, set())
    partial = search._log_h_single(torch.tensor([1, 0, 0, 0]), 1, gen, {}, set())
    assert partial > start


def _gateway_run(guidance: float, seed: int):
    cfg = PopulationSearchConfig(
        particles=6, episodes=3, rounds_per_episode=2,
        guidance_strength=guidance, seed=seed,
        proposal=MultiEditProposalConfig(
            candidates_per_parent=3, expected_edits=(1.0,),
            expected_edit_probs=(1.0,), random_mix=1.0,
            distance_strength=0.2,
        ),
        teacher=TeacherConfig(
            rollout_depth=1, rollout_width=6, roots_per_parent=2,
            beta_violation=4.0, beta_edit=0.0, feasibility_bonus=4.0,
        ),
    )
    return FutureAwarePopulationSearch(_and_constraint, 2, cfg).run(
        torch.zeros(4, dtype=torch.long), ALMState(rho=1.0, rho_min=1.0)
    )


def test_future_guidance_reduces_dual_pressure_on_gateway():
    local_peak, guided_peak = [], []
    for seed in range(10):
        local = _gateway_run(0.0, seed)
        guided = _gateway_run(2.0, seed)
        assert local.feasible_found and guided.feasible_found
        assert local.best_hamming == guided.best_hamming == 2
        local_peak.append(max(h.lambda_end for h in local.history))
        guided_peak.append(max(h.lambda_end for h in guided.history))
    assert statistics.mean(guided_peak) < 0.25 * statistics.mean(local_peak)


def test_search_returns_minimum_hard_feasible_edit_set():
    cfg = PopulationSearchConfig(
        particles=20, episodes=3, rounds_per_episode=3,
        guidance_strength=1.0, seed=4,
        proposal=MultiEditProposalConfig(
            candidates_per_parent=6,
            expected_edits=(1.0, 2.0, 3.0),
            expected_edit_probs=(0.2, 0.4, 0.4), random_mix=0.1,
        ),
        teacher=TeacherConfig(rollout_depth=1, rollout_width=3, roots_per_parent=1),
    )
    result = FutureAwarePopulationSearch(_additive_constraint, 2, cfg).run(
        torch.zeros(8, dtype=torch.long), ALMState(rho=5.0, rho_min=1.0)
    )
    assert result.feasible_found
    assert result.best_hamming == 3
    assert result.best_edit_cost == 3.0
    assert result.model_forward_batches > 0
    assert result.model_backward_batches > 0


def test_teacher_support_grows_on_revisit_without_discarding_samples():
    cfg = PopulationSearchConfig(
        particles=2, episodes=1, rounds_per_episode=1,
        proposal=MultiEditProposalConfig(
            candidates_per_parent=2, expected_edits=(1.0,),
            expected_edit_probs=(1.0,), random_mix=1.0, distance_strength=0.0,
        ),
        teacher=TeacherConfig(
            rollout_depth=1, rollout_width=3, rollout_max_children=5,
            rollout_add_per_revisit=2, roots_per_parent=1, random_roots_per_parent=0,
        ),
    )
    search = FutureAwarePopulationSearch(_additive_constraint, 2, cfg)
    x0 = torch.zeros(6, dtype=torch.long)
    search.graph = SequenceExperienceGraph(x0, _costs(x0))
    search._teacher_scale = 2.5
    gen = torch.Generator().manual_seed(7)
    search._log_h_single(x0, 1, gen, {}, set())
    first = len(search.graph.get_or_add(x0).children)
    search._log_h_single(x0, 1, gen, {}, set())
    second = len(search.graph.get_or_add(x0).children)
    assert first == 3
    assert second == 5


def test_zero_guidance_does_not_pay_teacher_rollout_cost():
    base = dict(
        particles=4, episodes=1, rounds_per_episode=1, seed=13,
        proposal=MultiEditProposalConfig(
            candidates_per_parent=3, expected_edits=(1.0,),
            expected_edit_probs=(1.0,), random_mix=1.0, distance_strength=0.0,
        ),
        teacher=TeacherConfig(
            rollout_depth=2, rollout_width=3, rollout_max_children=3,
            rollout_add_per_revisit=0, roots_per_parent=2, random_roots_per_parent=1,
        ),
    )
    local = FutureAwarePopulationSearch(
        _and_constraint, 2, PopulationSearchConfig(guidance_strength=0.0, **base)
    ).run(torch.zeros(4, dtype=torch.long), ALMState(rho=1.0))
    guided = FutureAwarePopulationSearch(
        _and_constraint, 2, PopulationSearchConfig(guidance_strength=2.0, **base)
    ).run(torch.zeros(4, dtype=torch.long), ALMState(rho=1.0))
    assert guided.model_forward_evals > local.model_forward_evals
    assert guided.graph_nodes > local.graph_nodes
