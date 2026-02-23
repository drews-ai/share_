#!/usr/bin/env python3
"""Smoke/regression tests for ShareWith simulation engine."""

from __future__ import annotations

import statistics
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import server  # noqa: E402


class ShareWithEngineTests(unittest.TestCase):
    def test_balanced_local_produces_cycles_with_tuned_defaults(self) -> None:
        cycle_counts = []
        completed_counts = []
        for seed in range(40):
            payload = server.run_scenario("balanced_local", seed=seed, params_override=None)
            metrics = payload["metrics"]
            cycle_counts.append(metrics["cycleCount"])
            completed_counts.append(metrics["completedCycles"])

        self.assertGreaterEqual(statistics.fmean(cycle_counts), 1.0)
        self.assertGreaterEqual(statistics.fmean(completed_counts), 0.55)

    def test_rank_solutions_handles_string_pains(self) -> None:
        ranked = server.rank_solutions(
            pains=["choice_declines", "low_cycle_survival"],
            pain_levels={"choice_declines": "5"},
        )
        self.assertGreater(len(ranked), 0)
        self.assertIn("score", ranked[0])

    def test_metrics_include_unmatched_alias(self) -> None:
        payload = server.run_scenario("choice_friction", seed=7, params_override=None)
        metrics = payload["metrics"]
        self.assertIn("unmatchedWants", metrics)
        self.assertEqual(metrics["unmatchedWants"], metrics["unmatchedWantsAfterMatch"])

    def test_optimizer_improves_or_matches_balanced_objective(self) -> None:
        result = server.optimize_parameters("balanced_local", budget=12, seeds_per_candidate=4)
        self.assertGreaterEqual(result["best"]["objectiveMean"], result["baseline"]["objectiveMean"])

    def test_high_trust_network_has_high_trust_signal(self) -> None:
        payload = server.run_scenario("high_trust_network", seed=3, params_override=None)
        metrics = payload["metrics"]
        self.assertGreater(metrics["highTrustEdgeShare"], 0.5)
        self.assertGreater(metrics["highTrustCycleShare"], 0.5)

    def test_high_trust_optimizer_profile_improves_or_matches_baseline(self) -> None:
        result = server.optimize_parameters(
            "high_trust_network",
            budget=16,
            seeds_per_candidate=4,
            objective_profile="high_trust_network",
        )
        self.assertGreaterEqual(result["best"]["objectiveMean"], result["baseline"]["objectiveMean"])

    def test_uncapped_cycle_length_mode_accepts_nonpositive_cap(self) -> None:
        capped = server.run_scenario("balanced_local", seed=11, params_override={"max_cycle_length": 2})
        uncapped = server.run_scenario("balanced_local", seed=11, params_override={"max_cycle_length": 0})
        self.assertLessEqual(capped["metrics"]["maxCycleLength"], 2)
        self.assertGreaterEqual(uncapped["metrics"]["maxCycleLength"], capped["metrics"]["maxCycleLength"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
