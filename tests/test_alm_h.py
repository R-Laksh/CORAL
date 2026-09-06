import torch

from coral.optimizers.alm_h import ALMHTwistedSearch, ALMState, SearchConfig, augmented_penalty


def _and_constraint(x):
    # target: token 1 at positions 0 and 1 together.  No single change satisfies it.
    score = x[:, 0, 1] * x[:, 1, 1]
    return 0.9 - score


def test_augmented_penalty_matches_coral_form():
    g = torch.tensor([-1.0, 0.2])
    got = augmented_penalty(g, lam=0.5, rho=2.0)
    expected = 0.5 / 2.0 * torch.relu(torch.tensor(0.5) + 2.0 * g).square()
    assert torch.allclose(got, expected)


def test_search_finds_minimum_two_edit_conjunction():
    cfg = SearchConfig(
        particles=64,
        episodes=2,
        horizon=2,
        proposal_width=4,
        lookahead_width=4,
        guidance_strength=6.0,
        seed=3,
    )
    opt = ALMHTwistedSearch(_and_constraint, vocab_size=2, config=cfg)
    res = opt.run(torch.zeros(4, dtype=torch.long), ALMState(rho=4.0, rho_min=4.0))
    assert res.feasible_found
    assert res.best_hamming == 2
    assert int(res.best_ids[0]) == 1 and int(res.best_ids[1]) == 1
    assert res.best_constraint <= 0


def test_dual_state_updates_between_episodes():
    def impossible(x):
        return torch.ones(x.shape[0])

    cfg = SearchConfig(
        particles=4, episodes=2, horizon=1, proposal_width=2,
        lookahead_width=1, guidance_strength=0.0, seed=0,
    )
    state = ALMState(lam=0.0, rho=1.0, rho_min=1.0)
    res = ALMHTwistedSearch(impossible, 2, cfg).run(torch.zeros(3, dtype=torch.long), state)
    assert not res.feasible_found
    assert res.final_state.lam > 0
    assert res.final_state.rho > 1


def test_amortized_h_hook_receives_remaining_horizon_and_avoids_teacher_backwards():
    calls = []

    def hook(candidates, x0, state, remaining):
        calls.append((candidates.shape[0], remaining))
        # Favor sequences with both required edits.
        return 5.0 * (candidates[:, 0].float() + candidates[:, 1].float())

    cfg = SearchConfig(
        particles=8, episodes=1, horizon=2, proposal_width=4,
        guidance_strength=2.0, seed=1,
    )
    opt = ALMHTwistedSearch(_and_constraint, 2, cfg, log_h_fn=hook)
    res = opt.run(torch.zeros(4, dtype=torch.long), ALMState(rho=4.0, rho_min=4.0))
    assert calls and calls[0][1] == 1 and calls[-1][1] == 0
    # One biological-model backward per particle per search step; the h hook itself
    # does not trigger candidate-specific biological-model backwards.
    assert res.model_backward_evals == cfg.particles * cfg.horizon * cfg.episodes


def test_shared_rollout_improves_gateway_success_across_fixed_seeds():
    successes = {"unguided": 0, "guided": 0}
    for seed in range(20):
        common = dict(
            particles=4, episodes=1, horizon=2, proposal_width=4,
            lookahead_width=4, seed=seed, move_temperature=0.5,
            gradient_proposal_strength=0.5,
        )
        u = ALMHTwistedSearch(
            _and_constraint, 2, SearchConfig(**common, guidance_strength=0.0)
        ).run(torch.zeros(4, dtype=torch.long), ALMState(rho=20.0, rho_min=20.0))
        g = ALMHTwistedSearch(
            _and_constraint, 2,
            SearchConfig(**common, guidance_strength=8.0, guidance_estimator="shared_rollout"),
        ).run(torch.zeros(4, dtype=torch.long), ALMState(rho=20.0, rho_min=20.0))
        successes["unguided"] += int(u.feasible_found)
        successes["guided"] += int(g.feasible_found)
    assert successes["guided"] >= 18
    assert successes["guided"] > successes["unguided"]
