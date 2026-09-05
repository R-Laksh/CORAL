"""Mathematical and data-contract checks, using only the research dependencies."""
import itertools
import unittest

import numpy as np

from coral.optimizers.distributional import (
    Clause, DistributionalCFOptimizer, FiniteEditGraph, backward_messages,
    factorized_projection, minimal_sufficient_clause, enumerate_clauses,
)
from coral.datasets.tfbs_mpra import _parse_triplet, OrientationNeighbourhood


class DistributionalTests(unittest.TestCase):
    def setUp(self):
        self.states = np.array(list(itertools.product((0, 1), repeat=3)))
        self.graph = FiniteEditGraph.from_sequences(
            self.states, ["".join("C" if b else "A" for b in s) for s in self.states])
        # Two alternative conjunctions sharing the first feature.
        self.scores = (self.states[:, 0] & (self.states[:, 1] | self.states[:, 2])).astype(float)
        self.problem = DistributionalCFOptimizer(self.graph, self.scores, 0, .5, steps=4)

    def test_backward_matches_exhaustive_paths(self):
        h = backward_messages(self.graph.kernel, self.problem.terminal, 4)
        expected = 0.
        for path in itertools.product(range(8), repeat=4):
            weight, previous = 1., 0
            for current in path:
                weight *= self.graph.kernel[previous, current]
                previous = current
            expected += weight * self.problem.terminal[previous]
        self.assertAlmostEqual(h[0, 0], expected, places=13)
        q, z = self.problem.exact_target()
        self.assertAlmostEqual(z, expected, places=13)
        self.assertAlmostEqual(q.sum(), 1.)

    def test_exact_doob_kernel_telescopes_to_target(self):
        h = backward_messages(self.graph.kernel, self.problem.terminal, 4)
        p = np.eye(8)[0]
        for t in range(4):
            numerator = self.graph.kernel * h[t + 1]
            transition = np.divide(numerator, h[t, :, None], out=np.zeros_like(numerator),
                                   where=h[t, :, None] > 0)
            p = p @ transition
        np.testing.assert_allclose(p, self.problem.exact_target()[0], atol=1e-13)

    def test_smc_exact_has_constant_weight_and_no_forbidden_endpoints(self):
        h = self.problem.guidance("exact", np.random.default_rng(1))
        result = self.problem.sample(h, particles=20000, rng=np.random.default_rng(2))
        self.assertFalse(result.failed)
        self.assertAlmostEqual(result.effective_sample_size, 20000., places=7)
        self.assertAlmostEqual(np.exp(result.log_normalizer), self.problem.exact_target()[1], places=13)
        np.testing.assert_allclose(result.endpoint_probabilities, self.problem.exact_target()[0], atol=.012)
        self.assertEqual(result.endpoint_probabilities[~self.problem.feasible].sum(), 0.)

    def test_approximate_twist_importance_normalizer(self):
        # Non-exact proposals must still estimate the same unnormalised target.
        estimates = []
        for seed in range(300):
            rng = np.random.default_rng(seed)
            result = self.problem.sample(self.problem.guidance("myopic", rng), particles=128, rng=rng)
            estimates.append(np.exp(result.log_normalizer))
        error = abs(np.mean(estimates) - self.problem.exact_target()[1])
        self.assertLess(error, 5 * np.std(estimates, ddof=1) / np.sqrt(len(estimates)))

    def test_block_shared_conjunction_preserves_other_route(self):
        block = Clause(((0, 1), (1, 1)))
        p = DistributionalCFOptimizer(self.graph, self.scores, 0, .5, blocked=[block])
        self.assertEqual(np.flatnonzero(p.feasible).tolist(), [5])  # A and C, without B
        self.assertEqual(self.problem.exact_target()[0][~self.problem.feasible].sum(), 0.)

    def test_defensive_rollout_preserves_terminal_target_and_normalizer(self):
        estimates = []
        for seed in range(300):
            rng = np.random.default_rng(seed)
            guide = self.problem.guidance("defensive_rollout", rng, rollouts=4)
            self.assertTrue((guide[:-1] >= .05).all())
            np.testing.assert_array_equal(guide[-1], self.problem.terminal)
            result = self.problem.sample(guide, particles=128, rng=rng)
            estimates.append(np.exp(result.log_normalizer))
        error = abs(np.mean(estimates) - self.problem.exact_target()[1])
        self.assertLess(error, 5 * np.std(estimates, ddof=1) / np.sqrt(len(estimates)))

    def test_minimal_clause_is_sufficient_and_not_literal_banning(self):
        clause = minimal_sufficient_clause(self.states, 6, self.problem.feasible)
        self.assertEqual(clause.literals, ((0, 1), (1, 1)))
        self.assertTrue(np.all(self.problem.feasible[clause.matches(self.states)]))

    def test_factorized_projection_loses_alternative_dependency(self):
        states = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        product = factorized_projection(states, np.array([0., .5, .5, 0.]))
        np.testing.assert_allclose(product, .25)
        with self.assertRaises(ValueError):
            factorized_projection(states[:3], np.array([0., .5, .5]))

    def test_impossible_event_is_explicit(self):
        problem = DistributionalCFOptimizer(self.graph, self.scores, 0, 2.)
        self.assertEqual(problem.exact_target()[1], 0.)
        self.assertTrue(problem.sample(problem.guidance("exact", np.random.default_rng(1))).failed)

    def test_generate_block_rerun_exhausts_finite_alternatives(self):
        result = enumerate_clauses(self.problem, rng=np.random.default_rng(1))
        self.assertEqual(result["status"], "finite_support_exhausted")
        self.assertEqual(len(result["clauses"]), 2)
        covered = np.zeros(8, dtype=bool)
        for clause, witness in zip(result["clauses"], result["witnesses"]):
            self.assertFalse(covered[witness])
            covered |= clause.matches(self.states)
        np.testing.assert_array_equal(covered, self.problem.feasible)

    def test_invalid_kernel_is_rejected(self):
        with self.assertRaises(ValueError):
            FiniteEditGraph(self.states, self.graph.sequences, np.ones((8, 8)), self.graph.distances)

    def test_enumeration_preserves_existing_exclusions(self):
        block = Clause(((0, 1), (1, 1)))
        residual = DistributionalCFOptimizer(self.graph, self.scores, 0, .5, blocked=[block])
        result = enumerate_clauses(residual, rng=np.random.default_rng(3))
        self.assertEqual(result["status"], "finite_support_exhausted")
        self.assertEqual(result["witnesses"], [5])
        self.assertEqual(len(result["clauses"]), 1)

    def test_split_groups_all_backgrounds_and_orders(self):
        def make(factors, background):
            return OrientationNeighbourhood("test", background, factors, (0, 10, 20), (2, 2), (),
                np.empty((0, 3)), (), np.empty((0, 3)), np.empty((0, 3)), np.empty(0))
        a = make(("A", "B", "C"), 1)
        b = make(("C", "A", "B"), 2)
        self.assertEqual(a.split(), b.split())

    def test_coordinates_are_checked_against_actual_dna(self):
        header = ">Construct1 Three Motifs A,B,C, Non-Template:AAA Template:CCC Non-Template:GGG Pos1:0 Pos2:5 Pos3:10 Distance1:2 Distance2:2"
        sequence = "T" * 15 + "AAATTCCCTTGGG" + "T" * 202
        row = _parse_triplet(header, sequence, 1)
        self.assertEqual(row["state"], (0, 1, 0))
        with self.assertRaises(ValueError):
            _parse_triplet(header, "T" * 230, 1)


if __name__ == "__main__":
    unittest.main()
