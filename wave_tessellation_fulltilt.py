#!/usr/bin/env python3
"""
Wave-style tessellation simulation:
- Agents start matching in home tessellation (hop 0).
- As patience grows, each agent's visibility expands to touching tessellations.
- No global graph merge; visibility expands as a wave from home cell.

Outputs under --outdir:
- weekly_metrics.csv
- summary.json
- report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import server


ROOT = Path(__file__).resolve().parent
TrailKey = Tuple[int, int, int, int, str, str]


@dataclass(frozen=True)
class CellConfig:
    width: int
    height: int


@dataclass
class SimulationConfig:
    grid: CellConfig
    agents_per_cell: int
    weeks: int
    active_rate: float
    max_hop: int
    patience_per_hop: int
    panel_mode: str
    variant: str
    seed: int
    outdir: Path
    max_offers_per_agent: int = 1
    max_wants_per_agent: int = 1
    decomposition_rate: float = 0.0
    cross_context_want_rate: float = 0.0
    spice_profile: str = "none"
    matching_mode: str = "global"
    bridge_hop: int = 1
    fallback_rounds: int = 0


@dataclass
class StageRunResult:
    name: str
    matching: List[server.Edge]
    cycles_detected: List[Dict[str, Any]]
    cycles_selected: List[Dict[str, Any]]
    choice_declines: int
    atomic_abort_count: int
    execution_outcomes: Dict[str, Any]
    consumed_offer_nodes: set[str]
    consumed_want_nodes: set[str]


SKILL_POOLS: Dict[server.Context, Tuple[str, ...]] = {
    server.Context.SURVIVAL: ("mealprep", "nursing", "eldercare", "childcare", "transport", "shelter"),
    server.Context.SAFETY: ("plumbing", "electrical", "legal", "repair", "security", "mediation"),
    server.Context.SOCIAL: ("haircut", "music", "fitness", "photo", "language", "events"),
    server.Context.GROWTH: ("coding", "design", "tutoring", "marketing", "career", "writing"),
    server.Context.LUXURY: ("travel", "coaching", "styling", "concierge", "wellness", "artisan"),
}

CONTEXT_NEIGHBORS: Dict[server.Context, Tuple[server.Context, ...]] = {
    server.Context.SURVIVAL: (server.Context.SAFETY, server.Context.SOCIAL),
    server.Context.SAFETY: (server.Context.SURVIVAL, server.Context.GROWTH),
    server.Context.SOCIAL: (server.Context.GROWTH, server.Context.LUXURY, server.Context.SURVIVAL),
    server.Context.GROWTH: (server.Context.SOCIAL, server.Context.SAFETY),
    server.Context.LUXURY: (server.Context.SOCIAL, server.Context.GROWTH),
}

# Thesis-inspired decompositions: one high-level need can fan out to multiple cycle-relevant sub-needs.
NEED_BUNDLE_TEMPLATES: Tuple[Tuple[str, Tuple[Tuple[server.Context, str], ...]], ...] = (
    (
        "bathroom_repair",
        (
            (server.Context.SAFETY, "plumbing"),
            (server.Context.SAFETY, "electrical"),
            (server.Context.SAFETY, "repair"),
            (server.Context.GROWTH, "design"),
            (server.Context.SURVIVAL, "shelter"),
        ),
    ),
    (
        "caregiving_stack",
        (
            (server.Context.SURVIVAL, "nursing"),
            (server.Context.SURVIVAL, "childcare"),
            (server.Context.SURVIVAL, "mealprep"),
            (server.Context.SURVIVAL, "transport"),
            (server.Context.SOCIAL, "events"),
        ),
    ),
    (
        "career_transition",
        (
            (server.Context.GROWTH, "coding"),
            (server.Context.GROWTH, "tutoring"),
            (server.Context.GROWTH, "career"),
            (server.Context.SAFETY, "legal"),
            (server.Context.SOCIAL, "language"),
        ),
    ),
    (
        "vacation_deconstruction",
        (
            (server.Context.LUXURY, "travel"),
            (server.Context.SURVIVAL, "transport"),
            (server.Context.SURVIVAL, "shelter"),
            (server.Context.SURVIVAL, "mealprep"),
            (server.Context.LUXURY, "concierge"),
            (server.Context.SOCIAL, "events"),
        ),
    ),
)


def _mean(values: Iterable[float]) -> float:
    data = list(values)
    return sum(data) / len(data) if data else 0.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _parse_grid(raw: str) -> CellConfig:
    parts = [part.strip() for part in raw.lower().replace("x", ",").split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("grid must look like '3x3' or '3,3'")
    return CellConfig(width=int(parts[0]), height=int(parts[1]))


def _sample_declaration_count(rng: random.Random, max_count: int, decomposition_rate: float) -> int:
    max_count = max(1, max_count)
    if max_count <= 1:
        return 1

    if rng.random() >= decomposition_rate:
        return 2 if rng.random() < 0.18 else 1

    counts = [k for k in range(2, max_count + 1)]
    base_weights = [0.42, 0.31, 0.18, 0.09]
    weights = base_weights[: len(counts)]
    total = sum(weights)
    normalized = [w / total for w in weights]
    return int(rng.choices(counts, weights=normalized, k=1)[0])


def _want_from_skill(context: server.Context, skill: str, bundle_tag: str = "") -> server.Want:
    return server.Want(skill=skill, context=context, tags=(skill, context.value.lower()))


def _offer_from_skill(context: server.Context, skill: str, supply_tag: str = "") -> server.Offer:
    return server.Offer(skill=skill, context=context, tags=(skill, context.value.lower()))


def _build_wants_with_ideation(
    rng: random.Random,
    primary_context: server.Context,
    cfg: SimulationConfig,
) -> List[server.Want]:
    target_wants = _sample_declaration_count(rng, cfg.max_wants_per_agent, cfg.decomposition_rate)
    wants: List[server.Want] = []

    use_bundle = target_wants > 1 and rng.random() < max(0.4, cfg.decomposition_rate)
    bundle_tag = ""
    if use_bundle:
        bundle_name, components = rng.choice(NEED_BUNDLE_TEMPLATES)
        bundle_tag = f"bundle:{bundle_name}"
        component_list = list(components)
        rng.shuffle(component_list)
        for context, skill in component_list:
            wants.append(_want_from_skill(context=context, skill=skill, bundle_tag=bundle_tag))
            if len(wants) >= target_wants:
                break

    while len(wants) < target_wants:
        want_context = primary_context
        if rng.random() < cfg.cross_context_want_rate:
            want_context = rng.choice(CONTEXT_NEIGHBORS[primary_context])
        want_skill = rng.choice(SKILL_POOLS[want_context])
        wants.append(_want_from_skill(context=want_context, skill=want_skill, bundle_tag=bundle_tag))

    return wants


def _build_offers_with_ideation(
    rng: random.Random,
    primary_context: server.Context,
    wants: List[server.Want],
    cfg: SimulationConfig,
) -> List[server.Offer]:
    if cfg.max_offers_per_agent <= 1:
        target_offers = 1
    else:
        if len(wants) > 1:
            target_offers = max(1, len(wants) + rng.choice((-1, 0, 0, 1)))
        else:
            target_offers = 2 if rng.random() < 0.2 else 1
        target_offers = min(cfg.max_offers_per_agent, target_offers)

    offers: List[server.Offer] = []
    anchor_context = primary_context
    anchor_skill = rng.choice(SKILL_POOLS[anchor_context])
    offers.append(_offer_from_skill(context=anchor_context, skill=anchor_skill, supply_tag="capacity:anchor"))

    want_contexts = [want.context for want in wants]
    offer_context_pool = [primary_context, *want_contexts, *CONTEXT_NEIGHBORS[primary_context]]

    while len(offers) < target_offers:
        offer_context = rng.choice(offer_context_pool)
        # Thesis spirit: complex needs should be met with proportional reciprocal capacity.
        if len(wants) > 1 and rng.random() < 0.32:
            offer_context = anchor_context
            offer_skill = anchor_skill
            supply_tag = "capacity:repeat"
        else:
            offer_skill = rng.choice(SKILL_POOLS[offer_context])
            supply_tag = ""
        offers.append(_offer_from_skill(context=offer_context, skill=offer_skill, supply_tag=supply_tag))

    return offers


def _build_agents(cfg: SimulationConfig) -> Tuple[List[server.Agent], Dict[str, Tuple[int, int]], List[Tuple[int, int]]]:
    rng = random.Random(cfg.seed)
    agents: List[server.Agent] = []
    home_cell: Dict[str, Tuple[int, int]] = {}
    all_cells: List[Tuple[int, int]] = []

    aid = 0
    contexts = (
        server.Context.SAFETY,
        server.Context.SOCIAL,
        server.Context.GROWTH,
        server.Context.SURVIVAL,
        server.Context.LUXURY,
    )
    context_weights = (0.33, 0.24, 0.2, 0.16, 0.07)
    baseline_declarations = (
        cfg.max_offers_per_agent <= 1
        and cfg.max_wants_per_agent <= 1
        and cfg.decomposition_rate <= 1e-12
        and cfg.cross_context_want_rate <= 1e-12
    )

    for cx in range(cfg.grid.width):
        for cy in range(cfg.grid.height):
            all_cells.append((cx, cy))
            for i in range(cfg.agents_per_cell):
                context = rng.choices(contexts, weights=context_weights)[0]
                if baseline_declarations:
                    pool = SKILL_POOLS[context]
                    pidx = rng.randrange(len(pool))
                    widx = (pidx + rng.randint(1, len(pool) - 1)) % len(pool)
                    offer_skill = pool[pidx]
                    want_skill = pool[widx]
                    offers = [server.Offer(skill=offer_skill, context=context, tags=(offer_skill, context.value.lower()))]
                    wants = [server.Want(skill=want_skill, context=context, tags=(want_skill, context.value.lower()))]
                else:
                    wants = _build_wants_with_ideation(rng=rng, primary_context=context, cfg=cfg)
                    offers = _build_offers_with_ideation(rng=rng, primary_context=context, wants=wants, cfg=cfg)
                agent_id = f"a{aid}"
                agent = server.Agent(
                    agent_id=agent_id,
                    name=f"A{aid}",
                    mode=server.Mode.CHOICE if rng.random() < 0.22 else server.Mode.AUTO,
                    offers=offers,
                    wants=wants,
                    location=server.Location(
                        x=cx + rng.uniform(0.05, 0.95),
                        y=cy + rng.uniform(0.05, 0.95),
                    ),
                    trust=server.TrustMetrics(
                        completion=max(0.35, min(0.99, rng.gauss(0.77, 0.1))),
                        quality=max(0.35, min(0.99, rng.gauss(0.79, 0.1))),
                    ),
                    patience=0,
                    gave=rng.randint(1, 22),
                    received=rng.randint(1, 22),
                )
                agents.append(agent)
                home_cell[agent_id] = (cx, cy)
                aid += 1
    return agents, home_cell, all_cells


def _hop_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    # "Touching tessellations" includes diagonals in this grid approximation.
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _cells_within(cell: Tuple[int, int], hop: int, cfg: CellConfig) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for dx in range(-hop, hop + 1):
        for dy in range(-hop, hop + 1):
            nx = cell[0] + dx
            ny = cell[1] + dy
            if 0 <= nx < cfg.width and 0 <= ny < cfg.height:
                out.append((nx, ny))
    return out


def _reach_hops(patience: int, max_hop: int, patience_per_hop: int) -> int:
    if patience_per_hop <= 0:
        return max_hop
    return int(_clamp(float(patience // patience_per_hop), 0.0, float(max_hop)))


def _build_edges_wave(
    agents: List[server.Agent],
    params: server.EngineParams,
    home_cell: Dict[str, Tuple[int, int]],
    reach_by_agent: Dict[str, int],
    grid: CellConfig,
    variant: str = "C",
    trail_memory: Dict[TrailKey, float] | None = None,
) -> Tuple[List[server.Edge], List[server.Edge], List[server.OfferNode], List[server.WantNode]]:
    offer_nodes, want_nodes = server._build_nodes(agents)
    agents_by_id = server._build_agent_index(agents)
    centrality = server._centrality_map(agents)

    # Context index avoids scanning offers that would be rejected by context-gating anyway.
    offers_by_cell_ctx: Dict[Tuple[int, int, server.Context], List[server.OfferNode]] = defaultdict(list)
    for offer_node in offer_nodes:
        cell = home_cell[offer_node.agent_id]
        offers_by_cell_ctx[(cell[0], cell[1], offer_node.offer.context)].append(offer_node)

    cell_cache: Dict[Tuple[Tuple[int, int], int], List[Tuple[int, int]]] = {}
    potential_edges: List[server.Edge] = []
    feasible_edges: List[server.Edge] = []

    for want_node in want_nodes:
        receiver = agents_by_id[want_node.agent_id]
        receiver_cell = home_cell[receiver.agent_id]
        receiver_hop = reach_by_agent[receiver.agent_id]
        want_context = want_node.want.context
        key = (receiver_cell, receiver_hop)
        if key not in cell_cache:
            cell_cache[key] = _cells_within(receiver_cell, receiver_hop, cfg=grid)
        for cell in cell_cache[key]:
            for offer_node in offers_by_cell_ctx.get((cell[0], cell[1], want_context), []):
                if offer_node.agent_id == want_node.agent_id:
                    continue
                provider = agents_by_id[offer_node.agent_id]
                # Receiver-centric wave filter: receiver can only "see" providers within hop radius.
                if _hop_distance(home_cell[provider.agent_id], receiver_cell) > receiver_hop:
                    continue
                edge = server._compute_edge(
                    provider=provider,
                    receiver=receiver,
                    offer_node=offer_node,
                    want_node=want_node,
                    centrality=centrality,
                    params=params,
                )
                if edge is None:
                    continue
                _apply_variant_weight(
                    edge=edge,
                    provider=provider,
                    receiver=receiver,
                    variant=variant,
                    home_cell=home_cell,
                    trail_memory=trail_memory,
                )
                potential_edges.append(edge)
                feasible_edges.append(edge)

    feasible_edges.sort(key=lambda edge: edge.weight, reverse=True)
    return potential_edges, feasible_edges, offer_nodes, want_nodes


def _edge_success_proxy(edge: server.Edge, provider: server.Agent, receiver: server.Agent) -> float:
    # Proxy for pre-transplant reliability (failure-aware literature inspired).
    allowance = max(edge.distance_allowance, 1e-6)
    distance_ratio = edge.distance / allowance
    p = (
        0.18
        + 0.34 * provider.trust.completion
        + 0.18 * provider.trust.quality
        + 0.16 * receiver.trust.completion
        + 0.18 * edge.proximity
        - 0.12 * distance_ratio
    )
    return _clamp(p, 0.03, 0.995)


def _apply_variant_weight(
    edge: server.Edge,
    provider: server.Agent,
    receiver: server.Agent,
    variant: str,
    home_cell: Dict[str, Tuple[int, int]] | None = None,
    trail_memory: Dict[TrailKey, float] | None = None,
) -> None:
    if variant == "C":
        return

    if variant == "D":
        # Failure-aware expected utility + tail-risk penalty (inspired by failure-aware / CVaR work).
        q = _edge_success_proxy(edge=edge, provider=provider, receiver=receiver)
        risk = 1.0 - q
        risk_penalty = 0.38 * risk + 0.22 * (risk * risk)
        edge.weight = edge.weight * q - risk_penalty
        return

    if variant == "E":
        # Hybrid fairness: utilitarian base + bounded patience/disadvantage bonus.
        # Inspired by lexicographic fairness + dynamic time-fairness papers.
        waiting_signal = math.log1p(max(receiver.patience, 0))
        disadvantage_signal = 1.0 - receiver.trust.completion
        raw_bonus = 0.22 * waiting_signal + 0.28 * waiting_signal * disadvantage_signal
        cap = 0.35 * max(abs(edge.weight), 0.2)
        fairness_bonus = min(raw_bonus, cap)
        edge.weight = edge.weight + fairness_bonus
        return

    if variant == "F":
        # Ant-colony inspired reinforcement:
        # - keep utilitarian base objective
        # - add bounded pheromone bonus from prior completed edge archetypes
        if home_cell is None or trail_memory is None:
            return
        key = _trail_key(edge, home_cell=home_cell)
        pheromone = max(0.0, float(trail_memory.get(key, 0.0)))
        pheromone_bonus = 0.18 * math.log1p(pheromone)
        cap = 0.32 * max(abs(edge.weight), 0.25)
        edge.weight = edge.weight + min(pheromone_bonus, cap)
        return

    raise ValueError(f"Unknown variant: {variant}")


def _trail_key(edge: server.Edge, home_cell: Dict[str, Tuple[int, int]]) -> TrailKey:
    p = home_cell[edge.provider_id]
    r = home_cell[edge.receiver_id]
    return (p[0], p[1], r[0], r[1], edge.context, edge.skill)


def _update_trail_memory(
    trail_memory: Dict[TrailKey, float],
    cycles: Sequence[Dict[str, Any]],
    home_cell: Dict[str, Tuple[int, int]],
    evaporation: float = 0.86,
) -> None:
    # Evaporation prevents lock-in and keeps adaptation to current network demand.
    for key in list(trail_memory.keys()):
        trail_memory[key] *= evaporation
        if trail_memory[key] < 1e-4:
            del trail_memory[key]

    for cycle in cycles:
        if not cycle.get("completed", False):
            continue
        trust = float(cycle.get("trustCohesion", 0.0))
        for edge in cycle.get("edges", []):
            key = _trail_key(edge, home_cell=home_cell)
            trail_memory[key] = trail_memory.get(key, 0.0) + (0.9 + 0.6 * trust)


def _merge_execution_outcomes(execution_outcomes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ratings_by_provider: Dict[str, List[float]] = defaultdict(list)
    completed_cycles = 0
    failed_execution_cycles = 0
    rolled_back_cycles = 0

    for outcomes in execution_outcomes:
        completed_cycles += int(outcomes.get("completedCycles", 0))
        failed_execution_cycles += int(outcomes.get("failedExecutionCycles", 0))
        rolled_back_cycles += int(outcomes.get("rolledBackCycles", 0))
        for provider_id, ratings in outcomes.get("ratingsByProvider", {}).items():
            ratings_by_provider[provider_id].extend(float(r) for r in ratings)

    return {
        "completedCycles": completed_cycles,
        "failedExecutionCycles": failed_execution_cycles,
        "rolledBackCycles": rolled_back_cycles,
        "ratingsByProvider": dict(ratings_by_provider),
    }


def _run_stage_match(
    *,
    stage_name: str,
    stage_edges: Sequence[server.Edge],
    available_offer_ids: set[str],
    available_want_ids: set[str],
    offer_node_by_id: Dict[str, server.OfferNode],
    want_node_by_id: Dict[str, server.WantNode],
    agents_by_id: Dict[str, server.Agent],
    params: server.EngineParams,
    seed: int,
) -> StageRunResult:
    candidate_edges = [
        edge
        for edge in stage_edges
        if edge.offer_node in available_offer_ids and edge.want_node in available_want_ids
    ]

    if not candidate_edges:
        return StageRunResult(
            name=stage_name,
            matching=[],
            cycles_detected=[],
            cycles_selected=[],
            choice_declines=0,
            atomic_abort_count=0,
            execution_outcomes={"completedCycles": 0, "failedExecutionCycles": 0, "rolledBackCycles": 0, "ratingsByProvider": {}},
            consumed_offer_nodes=set(),
            consumed_want_nodes=set(),
        )

    stage_offer_ids = sorted({edge.offer_node for edge in candidate_edges})
    stage_want_ids = sorted({edge.want_node for edge in candidate_edges})
    stage_offer_nodes = [offer_node_by_id[node_id] for node_id in stage_offer_ids]
    stage_want_nodes = [want_node_by_id[node_id] for node_id in stage_want_ids]

    matching = server._max_weight_matching(candidate_edges, stage_offer_nodes, stage_want_nodes, params)
    cycles_detected = server._detect_cycles(
        matching_edges=matching,
        max_cycle_len=params.max_cycle_length,
        agents_by_id=agents_by_id,
        params=params,
    )
    cycles_selected = server._select_edge_disjoint_cycles(cycles_detected)
    for cycle in cycles_selected:
        cycle["stageName"] = stage_name

    rng = random.Random(seed)
    choice_declines = server._confirm_cycles(cycles_selected, agents_by_id, params, rng)
    atomic_abort_count = server._apply_atomic_commit(cycles_selected)
    execution_outcomes = server._simulate_execution(cycles_selected, agents_by_id, params, rng)

    consumed_offer_nodes: set[str] = set()
    consumed_want_nodes: set[str] = set()
    for cycle in cycles_selected:
        if not cycle.get("committed", False):
            continue
        for edge in cycle.get("edges", []):
            consumed_offer_nodes.add(edge.offer_node)
            consumed_want_nodes.add(edge.want_node)

    return StageRunResult(
        name=stage_name,
        matching=matching,
        cycles_detected=cycles_detected,
        cycles_selected=cycles_selected,
        choice_declines=choice_declines,
        atomic_abort_count=atomic_abort_count,
        execution_outcomes=execution_outcomes,
        consumed_offer_nodes=consumed_offer_nodes,
        consumed_want_nodes=consumed_want_nodes,
    )


def _run_wave_week_partition_bridge(
    *,
    active_agents: List[server.Agent],
    params: server.EngineParams,
    home_cell: Dict[str, Tuple[int, int]],
    grid: CellConfig,
    max_hop: int,
    patience_per_hop: int,
    seed: int,
    variant: str,
    trail_memory: Dict[TrailKey, float] | None,
    bridge_hop: int,
    fallback_rounds: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[server.Edge], List[Dict[str, Any]], Dict[str, int]]:
    agents_by_id = server._build_agent_index(active_agents)
    reach_by_agent = {
        agent.agent_id: _reach_hops(
            patience=agent.patience,
            max_hop=max_hop,
            patience_per_hop=patience_per_hop,
        )
        for agent in active_agents
    }

    potential_edges, feasible_edges, offer_nodes, want_nodes = _build_edges_wave(
        agents=active_agents,
        params=params,
        home_cell=home_cell,
        reach_by_agent=reach_by_agent,
        grid=grid,
        variant=variant,
        trail_memory=trail_memory,
    )

    offer_node_by_id = {node.node_id: node for node in offer_nodes}
    want_node_by_id = {node.node_id: node for node in want_nodes}
    available_offer_ids = {node.node_id for node in offer_nodes}
    available_want_ids = {node.node_id for node in want_nodes}

    local_edges: List[server.Edge] = []
    bridge_edges: List[server.Edge] = []
    for edge in feasible_edges:
        hop = _hop_distance(home_cell[edge.provider_id], home_cell[edge.receiver_id])
        if hop == 0:
            local_edges.append(edge)
        elif hop <= max(1, bridge_hop):
            bridge_edges.append(edge)

    stage_results: List[StageRunResult] = []
    stage_seed = seed * 100_003 + 73

    for stage_name, stage_edges in (("local", local_edges), ("bridge", bridge_edges)):
        result = _run_stage_match(
            stage_name=stage_name,
            stage_edges=stage_edges,
            available_offer_ids=available_offer_ids,
            available_want_ids=available_want_ids,
            offer_node_by_id=offer_node_by_id,
            want_node_by_id=want_node_by_id,
            agents_by_id=agents_by_id,
            params=params,
            seed=stage_seed,
        )
        stage_results.append(result)
        available_offer_ids -= result.consumed_offer_nodes
        available_want_ids -= result.consumed_want_nodes
        stage_seed += 1

    fallback_used = 0
    for ridx in range(max(0, fallback_rounds)):
        if not available_offer_ids or not available_want_ids:
            break
        result = _run_stage_match(
            stage_name=f"fallback_{ridx + 1}",
            stage_edges=feasible_edges,
            available_offer_ids=available_offer_ids,
            available_want_ids=available_want_ids,
            offer_node_by_id=offer_node_by_id,
            want_node_by_id=want_node_by_id,
            agents_by_id=agents_by_id,
            params=params,
            seed=stage_seed,
        )
        if not result.cycles_selected:
            break
        stage_results.append(result)
        available_offer_ids -= result.consumed_offer_nodes
        available_want_ids -= result.consumed_want_nodes
        fallback_used += 1
        stage_seed += 1

    matching: List[server.Edge] = []
    cycles_detected: List[Dict[str, Any]] = []
    cycles_selected: List[Dict[str, Any]] = []
    choice_declines = 0
    atomic_abort_count = 0
    execution_chunks: List[Dict[str, Any]] = []

    for result in stage_results:
        matching.extend(result.matching)
        cycles_detected.extend(result.cycles_detected)
        cycles_selected.extend(result.cycles_selected)
        choice_declines += result.choice_declines
        atomic_abort_count += result.atomic_abort_count
        execution_chunks.append(result.execution_outcomes)

    execution_outcomes = _merge_execution_outcomes(execution_chunks)

    for idx, cycle in enumerate(cycles_selected, start=1):
        cycle["cycleId"] = f"cycle_{idx:03d}"

    projection = server._project_trust_and_patience(
        agents=active_agents,
        cycles=cycles_selected,
        params=params,
        ratings_by_provider=execution_outcomes["ratingsByProvider"],
    )
    metrics = server._derive_metrics(
        agents=active_agents,
        offer_nodes=offer_nodes,
        want_nodes=want_nodes,
        potential_edges=potential_edges,
        feasible_edges=feasible_edges,
        matching=matching,
        cycles_detected=cycles_detected,
        cycles_selected=cycles_selected,
        choice_declines=choice_declines,
        atomic_abort_count=atomic_abort_count,
        execution_outcomes=execution_outcomes,
        projection=projection,
    )

    metrics["matchingMode"] = "partition_bridge"
    metrics["localCycleCount"] = sum(1 for cycle in cycles_selected if cycle.get("stageName") == "local")
    metrics["bridgeCycleCount"] = sum(1 for cycle in cycles_selected if cycle.get("stageName") == "bridge")
    metrics["fallbackCycleCount"] = sum(1 for cycle in cycles_selected if str(cycle.get("stageName", "")).startswith("fallback_"))
    metrics["fallbackRoundsUsed"] = fallback_used

    return metrics, projection, matching, cycles_selected, reach_by_agent


def _run_wave_week(
    active_agents: List[server.Agent],
    params: server.EngineParams,
    home_cell: Dict[str, Tuple[int, int]],
    grid: CellConfig,
    max_hop: int,
    patience_per_hop: int,
    seed: int,
    variant: str = "C",
    trail_memory: Dict[TrailKey, float] | None = None,
    matching_mode: str = "global",
    bridge_hop: int = 1,
    fallback_rounds: int = 0,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[server.Edge], List[Dict[str, Any]], Dict[str, int]]:
    if matching_mode == "partition_bridge":
        return _run_wave_week_partition_bridge(
            active_agents=active_agents,
            params=params,
            home_cell=home_cell,
            grid=grid,
            max_hop=max_hop,
            patience_per_hop=patience_per_hop,
            seed=seed,
            variant=variant,
            trail_memory=trail_memory,
            bridge_hop=bridge_hop,
            fallback_rounds=fallback_rounds,
        )
    if matching_mode != "global":
        raise ValueError(f"Unknown matching_mode: {matching_mode}")

    agents_by_id = server._build_agent_index(active_agents)
    reach_by_agent = {
        agent.agent_id: _reach_hops(
            patience=agent.patience,
            max_hop=max_hop,
            patience_per_hop=patience_per_hop,
        )
        for agent in active_agents
    }

    potential_edges, feasible_edges, offer_nodes, want_nodes = _build_edges_wave(
        agents=active_agents,
        params=params,
        home_cell=home_cell,
        reach_by_agent=reach_by_agent,
        grid=grid,
        variant=variant,
        trail_memory=trail_memory,
    )

    matching = server._max_weight_matching(feasible_edges, offer_nodes, want_nodes, params)
    cycles_detected = server._detect_cycles(
        matching_edges=matching,
        max_cycle_len=params.max_cycle_length,
        agents_by_id=agents_by_id,
        params=params,
    )
    cycles_selected = server._select_edge_disjoint_cycles(cycles_detected)

    rng = random.Random(seed)
    choice_declines = server._confirm_cycles(cycles_selected, agents_by_id, params, rng)
    atomic_abort_count = server._apply_atomic_commit(cycles_selected)
    execution_outcomes = server._simulate_execution(cycles_selected, agents_by_id, params, rng)
    projection = server._project_trust_and_patience(
        agents=active_agents,
        cycles=cycles_selected,
        params=params,
        ratings_by_provider=execution_outcomes["ratingsByProvider"],
    )
    metrics = server._derive_metrics(
        agents=active_agents,
        offer_nodes=offer_nodes,
        want_nodes=want_nodes,
        potential_edges=potential_edges,
        feasible_edges=feasible_edges,
        matching=matching,
        cycles_detected=cycles_detected,
        cycles_selected=cycles_selected,
        choice_declines=choice_declines,
        atomic_abort_count=atomic_abort_count,
        execution_outcomes=execution_outcomes,
        projection=projection,
    )
    metrics["matchingMode"] = "global"
    metrics["localCycleCount"] = int(metrics["cycleCount"])
    metrics["bridgeCycleCount"] = 0
    metrics["fallbackCycleCount"] = 0
    metrics["fallbackRoundsUsed"] = 0
    return metrics, projection, matching, cycles_selected, reach_by_agent


def _cross_cell_stats(
    matching: Sequence[server.Edge],
    cycles: Sequence[Dict[str, Any]],
    home_cell: Dict[str, Tuple[int, int]],
) -> Dict[str, float]:
    matching_cross = 0
    matching_hops: List[int] = []
    for edge in matching:
        hop = _hop_distance(home_cell[edge.provider_id], home_cell[edge.receiver_id])
        matching_hops.append(float(hop))
        if hop > 0:
            matching_cross += 1

    completed_edges: List[server.Edge] = []
    for cycle in cycles:
        if cycle.get("completed", False):
            completed_edges.extend(cycle["edges"])
    completed_cross = 0
    completed_hops: List[int] = []
    for edge in completed_edges:
        hop = _hop_distance(home_cell[edge.provider_id], home_cell[edge.receiver_id])
        completed_hops.append(float(hop))
        if hop > 0:
            completed_cross += 1

    return {
        "matching_cross_share": matching_cross / len(matching) if matching else 0.0,
        "matching_avg_hop": _mean(matching_hops),
        "completed_cross_share": completed_cross / len(completed_edges) if completed_edges else 0.0,
        "completed_avg_hop": _mean(completed_hops),
    }


def _update_agents_from_projection(
    active_agents: List[server.Agent],
    projection: Dict[str, Any],
) -> None:
    completion = projection["projectedCompletionTrust"]
    quality = projection["projectedQualityTrust"]
    patience = projection["projectedPatience"]

    for agent in active_agents:
        aid = agent.agent_id
        agent.trust.completion = float(completion[aid])
        agent.trust.quality = float(quality[aid])
        agent.patience = int(patience[aid])


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_table(rows: List[Dict[str, Any]], columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _build_report(cfg: SimulationConfig, weekly_rows: List[Dict[str, Any]], summary: Dict[str, Any], elapsed_s: float) -> str:
    report: List[str] = []
    report.append("# Wave Tessellation Full-Tilt Report")
    report.append("")
    report.append(f"Generated UTC: `{datetime.now(timezone.utc).isoformat()}`")
    report.append("")
    report.append("## Configuration")
    report.append("")
    report.append(f"- Grid: `{cfg.grid.width}x{cfg.grid.height}` tessellations")
    report.append(f"- Agents per tessellation: `{cfg.agents_per_cell}`")
    report.append(f"- Total agents: `{cfg.grid.width * cfg.grid.height * cfg.agents_per_cell}`")
    report.append(f"- Weeks: `{cfg.weeks}`")
    report.append(f"- Active declaration rate per week: `{cfg.active_rate:.4f}`")
    report.append(f"- Max wave hop: `{cfg.max_hop}`")
    report.append(f"- Patience per hop: `{cfg.patience_per_hop}`")
    report.append(f"- Matching mode: `{cfg.matching_mode}`")
    report.append(f"- Bridge hop: `{cfg.bridge_hop}`")
    report.append(f"- Fallback rounds: `{cfg.fallback_rounds}`")
    report.append(f"- Spice profile: `{cfg.spice_profile}`")
    report.append(f"- Max offers per agent: `{cfg.max_offers_per_agent}`")
    report.append(f"- Max wants per agent: `{cfg.max_wants_per_agent}`")
    report.append(f"- Need decomposition rate: `{cfg.decomposition_rate:.3f}`")
    report.append(f"- Cross-context want rate: `{cfg.cross_context_want_rate:.3f}`")
    report.append(f"- Runtime seconds: `{elapsed_s:.3f}`")
    report.append("")
    report.append("## Summary")
    report.append("")
    for key, value in summary.items():
        if isinstance(value, float):
            report.append(f"- {key}: `{value:.6f}`")
        else:
            report.append(f"- {key}: `{value}`")
    report.append("")
    report.append("## Weekly Metrics")
    report.append("")
    report.append(
        _format_table(
            weekly_rows,
            (
                "week",
                "active_agents",
                "avg_reach_hops",
                "share_reach_ge1",
                "share_reach_ge2",
                "feasible_edges",
                "cycle_count",
                "completed_cycles",
                "completed_per_1000_active",
                "unmet_ratio",
                "avg_trust",
                "next_avg_trust",
                "avg_patience",
                "next_avg_patience",
                "matching_cross_share",
                "completed_cross_share",
                "local_cycle_count",
                "bridge_cycle_count",
                "fallback_cycle_count",
            ),
        )
    )
    report.append("")
    return "\n".join(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run large-cohort wave-style tessellation simulation.")
    parser.add_argument("--grid", default="3x3", help="Grid dimensions, e.g. 3x3")
    parser.add_argument("--agents-per-cell", type=int, default=17000)
    parser.add_argument("--weeks", type=int, default=6)
    parser.add_argument("--active-rate", type=float, default=0.02)
    parser.add_argument("--max-hop", type=int, default=4)
    parser.add_argument("--patience-per-hop", type=int, default=1)
    parser.add_argument(
        "--matching-mode",
        choices=("global", "partition_bridge"),
        default="global",
        help="global: one solve over current visible graph; partition_bridge: local pass + adjacent bridge pass + fallback rounds.",
    )
    parser.add_argument(
        "--bridge-hop",
        type=int,
        default=1,
        help="For partition_bridge mode: maximum cross-tessellation hop used in bridge stage.",
    )
    parser.add_argument(
        "--fallback-rounds",
        type=int,
        default=1,
        help="For partition_bridge mode: retry rounds using remaining declarations after local/bridge stages.",
    )
    parser.add_argument(
        "--panel-mode",
        choices=("fixed", "resample"),
        default="fixed",
        help="fixed: same active cohort each week (best for wave signal); resample: new cohort each week.",
    )
    parser.add_argument(
        "--variant",
        choices=("C", "D", "E", "F"),
        default="C",
        help="C=baseline wave, D=failure-aware risk-adjusted, E=hybrid fairness/time-priority, F=ant-trail reinforced.",
    )
    parser.add_argument(
        "--spice-profile",
        choices=("none", "thesis_ideation", "thesis_spice"),
        default="none",
        help="Declaration generator profile: thesis_ideation introduces multi-need decomposition.",
    )
    parser.add_argument("--max-offers-per-agent", type=int, default=None, help="Override max offers per agent (default from spice profile).")
    parser.add_argument("--max-wants-per-agent", type=int, default=None, help="Override max wants per agent (default from spice profile).")
    parser.add_argument("--decomposition-rate", type=float, default=None, help="Probability that a declaration becomes a decomposed multi-need bundle.")
    parser.add_argument("--cross-context-want-rate", type=float, default=None, help="Probability a want is sampled from adjacent context.")
    parser.add_argument(
        "--params-overrides-file",
        type=Path,
        default=None,
        help="Optional JSON file containing EngineParams overrides (or {\"params_overrides\": {...}}).",
    )
    parser.add_argument("--seed", type=int, default=20260218)
    parser.add_argument("--outdir", type=Path, default=ROOT / "artifacts" / "wave_fulltilt")
    return parser.parse_args()


def _resolve_spice_parameters(args: argparse.Namespace) -> Tuple[int, int, float, float]:
    presets = {
        "none": (1, 1, 0.0, 0.0),
        "thesis_ideation": (4, 4, 0.36, 0.18),
        "thesis_spice": (5, 5, 0.58, 0.26),
    }
    max_offers, max_wants, decomposition_rate, cross_context_rate = presets[args.spice_profile]

    if args.max_offers_per_agent is not None:
        max_offers = max(1, int(args.max_offers_per_agent))
    if args.max_wants_per_agent is not None:
        max_wants = max(1, int(args.max_wants_per_agent))
    if args.decomposition_rate is not None:
        decomposition_rate = _clamp(float(args.decomposition_rate), 0.0, 1.0)
    if args.cross_context_want_rate is not None:
        cross_context_rate = _clamp(float(args.cross_context_want_rate), 0.0, 1.0)

    return max_offers, max_wants, decomposition_rate, cross_context_rate


def main() -> None:
    args = parse_args()
    max_offers, max_wants, decomposition_rate, cross_context_rate = _resolve_spice_parameters(args)
    cfg = SimulationConfig(
        grid=_parse_grid(args.grid),
        agents_per_cell=args.agents_per_cell,
        weeks=args.weeks,
        active_rate=args.active_rate,
        max_hop=args.max_hop,
        patience_per_hop=args.patience_per_hop,
        panel_mode=args.panel_mode,
        variant=args.variant,
        seed=args.seed,
        outdir=args.outdir.resolve(),
        max_offers_per_agent=max_offers,
        max_wants_per_agent=max_wants,
        decomposition_rate=decomposition_rate,
        cross_context_want_rate=cross_context_rate,
        spice_profile=args.spice_profile,
        matching_mode=args.matching_mode,
        bridge_hop=max(1, int(args.bridge_hop)),
        fallback_rounds=max(0, int(args.fallback_rounds)),
    )
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    agents, home_cell, _ = _build_agents(cfg)
    print(
        f"Built {len(agents)} agents across {cfg.grid.width * cfg.grid.height} tessellations "
        f"({cfg.agents_per_cell} per tessellation)."
    )
    offer_counts = [len(agent.offers) for agent in agents]
    want_counts = [len(agent.wants) for agent in agents]
    print(
        f"Declaration model: spice={cfg.spice_profile} offers(mean)={_mean(float(v) for v in offer_counts):.2f} "
        f"wants(mean)={_mean(float(v) for v in want_counts):.2f} multi-need-share="
        f"{(sum(1 for v in want_counts if v > 1) / len(want_counts)):.3f}"
    )

    params = server.DEFAULT_PARAMS
    if args.params_overrides_file is not None:
        overrides_payload = json.loads(args.params_overrides_file.read_text(encoding="utf-8"))
        if isinstance(overrides_payload, dict) and "params_overrides" in overrides_payload:
            overrides = overrides_payload.get("params_overrides")
        else:
            overrides = overrides_payload
        if not isinstance(overrides, dict):
            raise ValueError("params-overrides-file must contain an object or {\"params_overrides\": {...}}")
        params = params.with_overrides(overrides)
    rng = random.Random(cfg.seed + 99)

    weekly_rows: List[Dict[str, Any]] = []
    trail_memory: Dict[TrailKey, float] = {}
    n_active = max(2, int(len(agents) * cfg.active_rate))
    fixed_indices: List[int] = rng.sample(range(len(agents)), n_active) if cfg.panel_mode == "fixed" else []

    for week in range(1, cfg.weeks + 1):
        week_seed = cfg.seed + week * 1009
        if cfg.panel_mode == "fixed":
            active_indices = fixed_indices
        else:
            active_indices = rng.sample(range(len(agents)), n_active)
        active_agents = [agents[idx] for idx in active_indices]
        t0 = time.perf_counter()

        metrics, projection, matching, cycles, reach_by_agent = _run_wave_week(
            active_agents=active_agents,
            params=params,
            home_cell=home_cell,
            grid=cfg.grid,
            max_hop=cfg.max_hop,
            patience_per_hop=cfg.patience_per_hop,
            seed=week_seed,
            variant=cfg.variant,
            trail_memory=trail_memory if cfg.variant == "F" else None,
            matching_mode=cfg.matching_mode,
            bridge_hop=cfg.bridge_hop,
            fallback_rounds=cfg.fallback_rounds,
        )
        if cfg.variant == "F":
            _update_trail_memory(trail_memory=trail_memory, cycles=cycles, home_cell=home_cell)
        _update_agents_from_projection(active_agents, projection)
        wave_stats = _cross_cell_stats(matching=matching, cycles=cycles, home_cell=home_cell)
        week_elapsed = time.perf_counter() - t0

        reaches = [reach_by_agent[a.agent_id] for a in active_agents]
        row = {
            "week": week,
            "week_runtime_s": week_elapsed,
            "active_agents": len(active_agents),
            "avg_reach_hops": _mean(float(r) for r in reaches),
            "share_reach_ge1": sum(1 for r in reaches if r >= 1) / len(reaches),
            "share_reach_ge2": sum(1 for r in reaches if r >= 2) / len(reaches),
            "share_reach_ge3": sum(1 for r in reaches if r >= 3) / len(reaches),
            "feasible_edges": metrics["feasibleEdges"],
            "matching_size": metrics["matchingSize"],
            "cycle_count": metrics["cycleCount"],
            "completed_cycles": metrics["completedCycles"],
            "completed_per_1000_active": metrics["completedCycles"] * 1000.0 / len(active_agents),
            "unmet_ratio": metrics["unmetWantsAfterExecution"] / max(metrics["totalWants"], 1),
            "avg_trust": metrics["avgTrust"],
            "next_avg_trust": metrics["nextAvgTrust"],
            "avg_patience": metrics["avgPatience"],
            "next_avg_patience": metrics["nextAvgPatience"],
            "cycle_survival": metrics["cycleSurvival"],
            "high_trust_cycle_share": metrics["highTrustCycleShare"],
            "high_trust_edge_share": metrics["highTrustEdgeShare"],
            "matching_cross_share": wave_stats["matching_cross_share"],
            "matching_avg_hop": wave_stats["matching_avg_hop"],
            "completed_cross_share": wave_stats["completed_cross_share"],
            "completed_avg_hop": wave_stats["completed_avg_hop"],
            "local_cycle_count": metrics.get("localCycleCount", metrics["cycleCount"]),
            "bridge_cycle_count": metrics.get("bridgeCycleCount", 0),
            "fallback_cycle_count": metrics.get("fallbackCycleCount", 0),
            "fallback_rounds_used": metrics.get("fallbackRoundsUsed", 0),
            "trail_mass": float(sum(trail_memory.values())) if cfg.variant == "F" else 0.0,
        }
        weekly_rows.append(row)
        print(
            f"week={week:02d} active={len(active_agents)} runtime={week_elapsed:.2f}s "
            f"reach>=1={row['share_reach_ge1']:.3f} completed/1000={row['completed_per_1000_active']:.3f} "
            f"unmet={row['unmet_ratio']:.3f} cross_completed={row['completed_cross_share']:.3f}"
        )

    elapsed_s = time.perf_counter() - start

    summary = {
        "total_agents": len(agents),
        "tessellations": cfg.grid.width * cfg.grid.height,
        "agents_per_tessellation": cfg.agents_per_cell,
        "weeks": cfg.weeks,
        "active_rate": cfg.active_rate,
        "panel_mode": cfg.panel_mode,
        "variant": cfg.variant,
        "matching_mode": cfg.matching_mode,
        "bridge_hop": cfg.bridge_hop,
        "fallback_rounds": cfg.fallback_rounds,
        "spice_profile": cfg.spice_profile,
        "params_overrides_file": str(args.params_overrides_file.resolve()) if args.params_overrides_file else "",
        "max_offers_per_agent": cfg.max_offers_per_agent,
        "max_wants_per_agent": cfg.max_wants_per_agent,
        "decomposition_rate": cfg.decomposition_rate,
        "cross_context_want_rate": cfg.cross_context_want_rate,
        "mean_offers_per_agent": _mean(float(v) for v in offer_counts),
        "mean_wants_per_agent": _mean(float(v) for v in want_counts),
        "share_agents_multi_need": sum(1 for v in want_counts if v > 1) / len(want_counts),
        "active_agents_per_week": n_active,
        "mean_week_runtime_s": _mean(float(row["week_runtime_s"]) for row in weekly_rows),
        "total_runtime_s": elapsed_s,
        "mean_completed_per_1000_active": _mean(float(row["completed_per_1000_active"]) for row in weekly_rows),
        "mean_unmet_ratio": _mean(float(row["unmet_ratio"]) for row in weekly_rows),
        "mean_avg_reach_hops": _mean(float(row["avg_reach_hops"]) for row in weekly_rows),
        "final_share_reach_ge1": float(weekly_rows[-1]["share_reach_ge1"]) if weekly_rows else 0.0,
        "final_share_reach_ge2": float(weekly_rows[-1]["share_reach_ge2"]) if weekly_rows else 0.0,
        "mean_matching_cross_share": _mean(float(row["matching_cross_share"]) for row in weekly_rows),
        "mean_completed_cross_share": _mean(float(row["completed_cross_share"]) for row in weekly_rows),
        "mean_local_cycle_count": _mean(float(row["local_cycle_count"]) for row in weekly_rows),
        "mean_bridge_cycle_count": _mean(float(row["bridge_cycle_count"]) for row in weekly_rows),
        "mean_fallback_cycle_count": _mean(float(row["fallback_cycle_count"]) for row in weekly_rows),
        "final_trail_mass": float(weekly_rows[-1]["trail_mass"]) if weekly_rows else 0.0,
    }

    _write_csv(cfg.outdir / "weekly_metrics.csv", weekly_rows)
    (cfg.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = _build_report(cfg, weekly_rows, summary, elapsed_s=elapsed_s)
    (cfg.outdir / "report.md").write_text(report, encoding="utf-8")

    print("Wave full-tilt run complete.")
    print(f"Output directory: {cfg.outdir}")
    print(f" - {cfg.outdir / 'weekly_metrics.csv'}")
    print(f" - {cfg.outdir / 'summary.json'}")
    print(f" - {cfg.outdir / 'report.md'}")


if __name__ == "__main__":
    main()
