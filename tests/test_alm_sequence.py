"""Necessary probability, domain, and discrete-acceptance checks."""
import itertools
import unittest

import numpy as np
import torch

from coral.optimizers.alm_sequence import (
    Constraints, SequenceOracle, alm_energy, candidate_pool_select, reference_step,
    run_h_alm, run_st_alm, run_ledidi,
)


class TwoOutput(torch.nn.Module):
    def forward(self, x):
        return torch.stack((x[:, 1, 0] * x[:, 1, 1], x[:, 1, 2]), dim=1)


class ALMSequenceTests(unittest.TestCase):
    def test_candidate_pool_importance_identity(self):
        # Exhaustive integration over every candidate pool and selected index.
        kernel, psi, f, old = np.array([.2, .3, .5]), np.array([.1, 3., .7]), np.array([2., -1., 4.]), .8
        expected = (kernel * psi * f).sum() / old
        total = 0.
        for pool in itertools.product(range(3), repeat=3):
            pool = np.array(pool)
            pool_probability = kernel[pool].prod()
            chosen = psi[pool] / psi[pool].sum()
            weight = psi[pool].mean() / old
            total += pool_probability * (chosen * f[pool]).sum() * weight
        self.assertAlmostEqual(total, expected, places=13)

    def test_pool_sampler_and_realised_denominator(self):
        rng = np.random.default_rng(2026)
        n = 60000
        candidates = np.tile(np.array([[[0], [1]]]), (n, 1, 1))
        messages = np.log(np.tile([1., 3.], (n, 1)))
        selected, carried, weight = candidate_pool_select(candidates, messages, np.full(n, np.log(.4)), rng)
        self.assertAlmostEqual(selected.mean(), .75, delta=.006)
        np.testing.assert_allclose(carried, np.log(np.where(selected[:, 0] == 0, 1., 3.)))
        np.testing.assert_allclose(weight, np.log(5.))

    def test_reference_reversions_and_immutable_sites(self):
        source = np.tile([1, 0, 1], (30000, 1))
        moved = reference_step(source, np.array([1]), 2, np.random.default_rng(1), 0.)
        np.testing.assert_array_equal(moved, np.tile([1, 1, 1], (30000, 1)))
        back = reference_step(moved, np.array([1]), 2, np.random.default_rng(2), 0.)
        np.testing.assert_array_equal(back, source)

    def test_defensive_proposal_survives_large_alm_offsets(self):
        pool = np.tile(np.array([[[0], [1]]]), (10000, 1, 1))
        message = np.tile([-1., -4.], (len(pool), 1))
        old = np.zeros(len(pool))
        a = candidate_pool_select(pool, message, old, np.random.default_rng(8), .02)
        b = candidate_pool_select(pool, message - 1e6, old - 1e6, np.random.default_rng(8), .02)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_allclose(a[2], b[2], atol=1e-9)
        self.assertLess(a[0].mean(), .1)

    def test_defensive_mixture_has_correct_importance_weights(self):
        pool = np.tile(np.array([[[0], [1]]]), (80000, 1, 1))
        message = np.log(np.tile([1., 3.], (len(pool), 1)))
        chosen, carried, weights = candidate_pool_select(pool, message, np.zeros(len(pool)),
                                                         np.random.default_rng(9), .25)
        f = np.where(chosen[:, 0] == 0, 2., -1.)
        self.assertAlmostEqual(float((np.exp(weights) * f).mean()), -.5, delta=.025)

    def test_independent_reference_can_exceed_one_edit_without_a_cap(self):
        source = np.zeros((5000, 16), dtype=np.int64)
        moved = reference_step(source, np.arange(12), 2, np.random.default_rng(3), kernel="independent")
        self.assertGreater(np.count_nonzero(moved, axis=1).max(), 1)
        self.assertFalse(moved[:, 12:].any())

    def test_mean_feasibility_does_not_accept_joint_failure(self):
        constraints = Constraints([.5, .5], [1, -1], [1, 1])
        oracle = SequenceOracle(TwoOutput(), [0, 0, 0], [0, 1, 2], 2, constraints, 10)
        oracle._observe(np.array([[0, 0, 0], [1, 1, 1]]), np.array([[0., 0.], [1., 1.]]))
        self.assertIsNone(oracle.best)
        oracle.score_ids([[1, 1, 0]])
        self.assertEqual(oracle.best["edits"], 2)

    def test_soft_prediction_cannot_enter_archive(self):
        oracle = SequenceOracle(TwoOutput(), [0, 0, 0], [0, 1], 2,
                                Constraints([.1, .5], [1, -1], [1, 1]), 10)
        soft = torch.full((1, 2, 3), .5, requires_grad=True)
        oracle(soft).sum().backward()
        self.assertIsNone(oracle.best)
        self.assertEqual(oracle.work.backward_sequences, 1)

    def test_all_engines_respect_domain_and_recheck_constraints(self):
        for runner in (run_h_alm, run_st_alm, run_ledidi):
            oracle = SequenceOracle(TwoOutput(), [0, 0, 0], [0, 1], 2,
                                    Constraints([.5, .5], [1, -1], [1, 1]), 512)
            kwargs = {"horizon": 3, "particles": 32} if runner == run_h_alm else {"steps": 32}
            result = runner(oracle, seed=7, **kwargs)
            self.assertLessEqual(result["work"]["forward_sequences"], 512)
            if result["best"] is not None:
                self.assertEqual(result["best"]["ids"], [1, 1, 0])
                self.assertEqual(result["best"]["edits"], 2)
            if runner == run_h_alm:
                self.assertIsNotNone(result["best"])

    def test_phr_constant_is_irrelevant_but_penalty_not_indicator(self):
        residual = np.array([[-1.], [1.]])
        value = alm_energy(np.zeros(2), residual, np.array([0.]), 10.)
        np.testing.assert_array_equal(value, [0., 5.])
        self.assertGreater(np.exp(-value[1]), 0.)

    def test_hidden_edit_cap_rejected(self):
        oracle = SequenceOracle(TwoOutput(), [0, 0, 0], [0, 1, 2], 2,
                                Constraints([.5, .5], [1, -1], [1, 1]), 10)
        with self.assertRaises(ValueError):
            run_h_alm(oracle, horizon=2)

    def test_multistep_stochastic_messages_preserve_normalizer(self):
        states = np.array(list(itertools.product((0, 1), repeat=3)))
        distance = np.count_nonzero(states[:, None] != states[None], axis=2)
        kernel = .5 * np.eye(8) + (distance == 1) / 6
        constraints = Constraints([.5, .5], [1, -1], [1, 1])
        scores = np.column_stack((states[:, 0] * states[:, 1], states[:, 2]))
        terminal = np.exp(-alm_energy(states.sum(1), constraints.residual(scores), np.zeros(2), 10.) / .75)
        exact = np.linalg.matrix_power(kernel, 3)[0] @ terminal
        for guidance in ("rollout", "rollout_control", "myopic", "reference"):
            estimates = []
            for seed in range(100):
                oracle = SequenceOracle(TwoOutput(), [0, 0, 0], [0, 1, 2], 2, constraints, 32)
                result = run_h_alm(oracle, seed=seed, guidance=guidance, horizon=3,
                                   episodes=1, particles=64)
                estimates.append(np.exp(result["history"][0]["log_normalizer_estimate"]))
            se = np.std(estimates, ddof=1) / np.sqrt(len(estimates))
            self.assertLess(abs(np.mean(estimates) - exact), 5 * se + 1e-5, guidance)


if __name__ == "__main__":
    unittest.main()
