#!/usr/bin/env python3
"""
TDD validation for thesis-aligned wave tessellation behavior.
"""

from __future__ import annotations

from pathlib import Path
import unittest

import server
import wave_tessellation_fulltilt as wave


def _agent(
    agent_id: str,
    offer_skill: str,
    want_skill: str,
    x: float,
    y: float,
    patience: int = 0,
    context: server.Context = server.Context.SOCIAL,
) -> server.Agent:
    return server.Agent(
        agent_id=agent_id,
        name=agent_id,
        mode=server.Mode.AUTO,
        offers=[server.Offer(skill=offer_skill, context=context, tags=(offer_skill,))],
        wants=[server.Want(skill=want_skill, context=context, tags=(want_skill,))],
        location=server.Location(x=x, y=y),
        trust=server.TrustMetrics(completion=0.92, quality=0.91),
        patience=patience,
        gave=10,
        received=8,
    )


def _edge(
    edge_id: str,
    provider_id: str,
    receiver_id: str,
    weight: float,
    distance: float,
    distance_allowance: float,
    proximity: float = 0.75,
    context: server.Context = server.Context.SOCIAL,
) -> server.Edge:
    return server.Edge(
        edge_id=edge_id,
        offer_node=f"o_{provider_id}",
        want_node=f"w_{receiver_id}",
        provider_id=provider_id,
        provider_name=provider_id,
        receiver_id=receiver_id,
        receiver_name=receiver_id,
        skill="haircut",
        context=context.value,
        mode=server.Mode.AUTO.value,
        fit_score=0.9,
        proximity=proximity,
        base_quality=0.9,
        generosity_bonus=0.0,
        choice_bonus=0.0,
        patience_bonus=0.0,
        high_trust_bonus=0.0,
        trust_multiplier=1.0,
        weight=weight,
        admissible=True,
        distance=distance,
        distance_allowance=distance_allowance,
    )


class WaveThesisTDDTests(unittest.TestCase):
    def test_baseline_profile_keeps_single_offer_and_want(self) -> None:
        cfg = wave.SimulationConfig(
            grid=wave.CellConfig(width=2, height=2),
            agents_per_cell=25,
            weeks=1,
            active_rate=0.2,
            max_hop=3,
            patience_per_hop=1,
            panel_mode="fixed",
            variant="C",
            seed=7,
            outdir=Path("/tmp"),
        )
        agents, _, _ = wave._build_agents(cfg)
        self.assertTrue(all(len(agent.offers) == 1 for agent in agents))
        self.assertTrue(all(len(agent.wants) == 1 for agent in agents))

    def test_thesis_ideation_profile_generates_multi_need_contracts(self) -> None:
        cfg = wave.SimulationConfig(
            grid=wave.CellConfig(width=2, height=2),
            agents_per_cell=40,
            weeks=1,
            active_rate=0.2,
            max_hop=3,
            patience_per_hop=1,
            panel_mode="fixed",
            variant="D",
            seed=17,
            outdir=Path("/tmp"),
            max_offers_per_agent=4,
            max_wants_per_agent=4,
            decomposition_rate=0.55,
            cross_context_want_rate=0.2,
            spice_profile="thesis_ideation",
        )
        agents, _, _ = wave._build_agents(cfg)

        self.assertTrue(any(len(agent.wants) > 1 for agent in agents))
        self.assertTrue(any(len(agent.offers) > 1 for agent in agents))
        self.assertTrue(all(1 <= len(agent.wants) <= 4 for agent in agents))
        self.assertTrue(all(1 <= len(agent.offers) <= 4 for agent in agents))

    def test_reach_hops_is_monotonic_and_clamped(self) -> None:
        values = [wave._reach_hops(patience=p, max_hop=3, patience_per_hop=2) for p in range(0, 12)]
        self.assertTrue(all(a <= b for a, b in zip(values, values[1:])))
        self.assertEqual(values[0], 0)
        self.assertEqual(values[-1], 3)

    def test_cells_within_touching_includes_diagonals(self) -> None:
        cfg = wave.CellConfig(width=3, height=3)
        cells = set(wave._cells_within((1, 1), hop=1, cfg=cfg))
        self.assertIn((0, 0), cells)
        self.assertIn((2, 2), cells)
        self.assertIn((1, 0), cells)
        self.assertEqual(len(cells), 9)

    def test_hop_zero_blocks_cross_tessellation_edges(self) -> None:
        agents = [
            _agent("p0", offer_skill="haircut", want_skill="music", x=0.1, y=0.5),
            _agent("r1", offer_skill="music", want_skill="haircut", x=1.1, y=0.5),
            _agent("p2", offer_skill="haircut", want_skill="music", x=2.1, y=0.5),
        ]
        home = {"p0": (0, 0), "r1": (1, 0), "p2": (2, 0)}
        reach = {"p0": 0, "r1": 0, "p2": 0}

        _, feasible, _, _ = wave._build_edges_wave(
            agents=agents,
            params=server.DEFAULT_PARAMS,
            home_cell=home,
            reach_by_agent=reach,
            grid=wave.CellConfig(width=3, height=1),
        )

        # r1 can only "see" own tessellation at hop 0, so no incoming haircut edges from p0/p2.
        incoming_to_r1 = [e for e in feasible if e.receiver_id == "r1"]
        self.assertEqual(incoming_to_r1, [])

    def test_hop_one_allows_touching_tessellations(self) -> None:
        agents = [
            _agent("p0", offer_skill="haircut", want_skill="music", x=0.1, y=0.5),
            _agent("r1", offer_skill="music", want_skill="haircut", x=1.1, y=0.5),
            _agent("p2", offer_skill="haircut", want_skill="music", x=2.1, y=0.5),
        ]
        home = {"p0": (0, 0), "r1": (1, 0), "p2": (2, 0)}
        reach = {"p0": 0, "r1": 1, "p2": 0}

        _, feasible, _, _ = wave._build_edges_wave(
            agents=agents,
            params=server.DEFAULT_PARAMS,
            home_cell=home,
            reach_by_agent=reach,
            grid=wave.CellConfig(width=3, height=1),
        )

        incoming_to_r1 = sorted(e.provider_id for e in feasible if e.receiver_id == "r1" and e.skill == "haircut")
        self.assertEqual(incoming_to_r1, ["p0", "p2"])

    def test_hop_limit_prevents_non_touching(self) -> None:
        agents = [
            _agent("p0", offer_skill="haircut", want_skill="music", x=0.1, y=0.5),
            _agent("r1", offer_skill="music", want_skill="haircut", x=1.1, y=0.5),
            _agent("p3", offer_skill="haircut", want_skill="music", x=3.1, y=0.5),
        ]
        home = {"p0": (0, 0), "r1": (1, 0), "p3": (3, 0)}
        reach = {"p0": 0, "r1": 1, "p3": 0}

        _, feasible, _, _ = wave._build_edges_wave(
            agents=agents,
            params=server.DEFAULT_PARAMS,
            home_cell=home,
            reach_by_agent=reach,
            grid=wave.CellConfig(width=4, height=1),
        )

        incoming_to_r1 = [e.provider_id for e in feasible if e.receiver_id == "r1" and e.skill == "haircut"]
        self.assertNotIn("p3", incoming_to_r1)

    def test_all_edges_respect_receiver_wave_radius(self) -> None:
        agents = [
            _agent("a0", offer_skill="haircut", want_skill="music", x=0.1, y=0.1),
            _agent("a1", offer_skill="music", want_skill="haircut", x=1.1, y=0.1),
            _agent("a2", offer_skill="haircut", want_skill="music", x=2.1, y=0.1),
            _agent("a3", offer_skill="music", want_skill="haircut", x=3.1, y=0.1),
        ]
        home = {"a0": (0, 0), "a1": (1, 0), "a2": (2, 0), "a3": (3, 0)}
        reach = {"a0": 0, "a1": 1, "a2": 2, "a3": 0}

        _, feasible, _, _ = wave._build_edges_wave(
            agents=agents,
            params=server.DEFAULT_PARAMS,
            home_cell=home,
            reach_by_agent=reach,
            grid=wave.CellConfig(width=4, height=1),
        )

        for edge in feasible:
            receiver_cell = home[edge.receiver_id]
            provider_cell = home[edge.provider_id]
            hop = wave._hop_distance(provider_cell, receiver_cell)
            self.assertLessEqual(hop, reach[edge.receiver_id])

    def test_variant_d_penalizes_higher_failure_risk(self) -> None:
        hi_provider = _agent("hi", "haircut", "music", 0.1, 0.1)
        lo_provider = _agent("lo", "haircut", "music", 0.1, 0.1)
        receiver = _agent("rx", "music", "haircut", 0.2, 0.2)
        hi_provider.trust.completion = 0.96
        hi_provider.trust.quality = 0.95
        lo_provider.trust.completion = 0.45
        lo_provider.trust.quality = 0.42
        receiver.trust.completion = 0.75

        safe_edge = _edge("e_safe", "hi", "rx", weight=1.4, distance=1.0, distance_allowance=6.0, proximity=0.88)
        risky_edge = _edge("e_risky", "lo", "rx", weight=1.4, distance=5.8, distance_allowance=6.0, proximity=0.18)

        wave._apply_variant_weight(edge=safe_edge, provider=hi_provider, receiver=receiver, variant="D")
        wave._apply_variant_weight(edge=risky_edge, provider=lo_provider, receiver=receiver, variant="D")

        self.assertGreater(safe_edge.weight, risky_edge.weight)

    def test_variant_e_increases_priority_for_waiting_agents(self) -> None:
        provider = _agent("px", "haircut", "music", 0.1, 0.1)
        impatient = _agent("r0", "music", "haircut", 0.2, 0.2, patience=0)
        waiting = _agent("r1", "music", "haircut", 0.2, 0.2, patience=6)
        impatient.trust.completion = 0.92
        waiting.trust.completion = 0.72

        edge_impatient = _edge("e0", "px", "r0", weight=1.0, distance=2.0, distance_allowance=6.0)
        edge_waiting = _edge("e1", "px", "r1", weight=1.0, distance=2.0, distance_allowance=6.0)

        wave._apply_variant_weight(edge=edge_impatient, provider=provider, receiver=impatient, variant="E")
        wave._apply_variant_weight(edge=edge_waiting, provider=provider, receiver=waiting, variant="E")

        self.assertGreater(edge_waiting.weight, edge_impatient.weight)

    def test_variant_f_applies_ant_trail_bonus(self) -> None:
        provider = _agent("p", "haircut", "music", 0.1, 0.1)
        receiver = _agent("r", "music", "haircut", 1.1, 0.1)
        edge = _edge("ef", "p", "r", weight=1.0, distance=2.0, distance_allowance=6.0)
        home = {"p": (0, 0), "r": (1, 0)}
        key = wave._trail_key(edge, home_cell=home)
        trails = {key: 12.0}

        wave._apply_variant_weight(
            edge=edge,
            provider=provider,
            receiver=receiver,
            variant="F",
            home_cell=home,
            trail_memory=trails,
        )

        self.assertGreater(edge.weight, 1.0)

    def test_partition_bridge_mode_reports_stage_cycle_counts(self) -> None:
        agents = [
            _agent("l0", "haircut", "music", 0.15, 0.25, patience=1),
            _agent("l1", "music", "haircut", 0.35, 0.35, patience=1),
            _agent("b0", "plumbing", "transport", 0.25, 0.75, patience=1, context=server.Context.SAFETY),
            _agent("b1", "transport", "plumbing", 1.25, 0.75, patience=1, context=server.Context.SAFETY),
        ]
        home = {
            "l0": (0, 0),
            "l1": (0, 0),
            "b0": (0, 0),
            "b1": (1, 0),
        }
        metrics, projection, matching, cycles, _ = wave._run_wave_week(
            active_agents=agents,
            params=server.DEFAULT_PARAMS,
            home_cell=home,
            grid=wave.CellConfig(width=2, height=1),
            max_hop=1,
            patience_per_hop=1,
            seed=404,
            variant="C",
            matching_mode="partition_bridge",
            bridge_hop=1,
            fallback_rounds=1,
        )

        self.assertEqual(metrics["matchingMode"], "partition_bridge")
        self.assertGreaterEqual(metrics["localCycleCount"], 1)
        self.assertGreaterEqual(metrics["bridgeCycleCount"], 1)
        self.assertGreaterEqual(metrics["cycleCount"], metrics["localCycleCount"] + metrics["bridgeCycleCount"])
        self.assertIn("projectedCompletionTrust", projection)
        self.assertTrue(len(matching) >= 2)
        self.assertTrue(len(cycles) >= 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
