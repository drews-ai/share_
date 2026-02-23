#!/usr/bin/env python3
"""
SHAREWITH CYCLE TRADING SERVER
==============================

Purpose:
- Serve a wireframe UI for ShareWith cycle trading
- Run trust-aware, patience-first cycle simulations on the Python server
- Surface pain points and rank candidate interventions
- Optimize algorithm parameters using repeatable parameter search

Run:
  python3 server.py --host 127.0.0.1 --port 8080

API:
  GET  /api/scenarios
  GET  /api/default-params
  POST /api/run       {"scenario_key": "balanced_local", "seed": 42, "params": {...}}
  POST /api/rank      {"pains": [...], "pain_levels": {...}}
  POST /api/optimize  {"scenario_key": "balanced_local", "budget": 40, "seeds": 6, "objective_profile": "balanced|high_trust_network"}
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse


# ============================================================
# DOMAIN ENUMS + ENTITIES
# ============================================================


class Mode(Enum):
    AUTO = "AUTO"
    CHOICE = "CHOICE"


class Context(Enum):
    SURVIVAL = "SURVIVAL"
    SAFETY = "SAFETY"
    SOCIAL = "SOCIAL"
    GROWTH = "GROWTH"
    LUXURY = "LUXURY"


@dataclass
class Offer:
    skill: str
    context: Context
    tags: Tuple[str, ...] = ()


@dataclass
class Want:
    skill: str
    context: Context
    tags: Tuple[str, ...] = ()


@dataclass
class TrustMetrics:
    completion: float
    quality: float


@dataclass
class Location:
    x: float
    y: float


@dataclass
class Agent:
    agent_id: str
    name: str
    mode: Mode
    offers: List[Offer]
    wants: List[Want]
    location: Location
    trust: TrustMetrics
    patience: int
    gave: int
    received: int


@dataclass
class Scenario:
    key: str
    name: str
    note: str
    agents: List[Agent]


@dataclass
class OfferNode:
    node_id: str
    agent_id: str
    agent_name: str
    offer: Offer


@dataclass
class WantNode:
    node_id: str
    agent_id: str
    agent_name: str
    want: Want


@dataclass
class Edge:
    edge_id: str
    offer_node: str
    want_node: str
    provider_id: str
    provider_name: str
    receiver_id: str
    receiver_name: str
    skill: str
    context: str
    mode: str

    fit_score: float
    proximity: float
    base_quality: float
    generosity_bonus: float
    choice_bonus: float
    patience_bonus: float
    high_trust_bonus: float
    trust_multiplier: float

    weight: float
    admissible: bool
    distance: float
    distance_allowance: float

    def to_wire(self) -> Dict[str, Any]:
        return {
            "id": self.edge_id,
            "offerNode": self.offer_node,
            "wantNode": self.want_node,
            "providerId": self.provider_id,
            "providerName": self.provider_name,
            "receiverId": self.receiver_id,
            "receiverName": self.receiver_name,
            "skill": self.skill,
            "context": self.context,
            "mode": self.mode,
            "fitScore": self.fit_score,
            "proximity": self.proximity,
            "baseQuality": self.base_quality,
            "generosityBonus": self.generosity_bonus,
            "choiceBonus": self.choice_bonus,
            "patienceBonus": self.patience_bonus,
            "highTrustBonus": self.high_trust_bonus,
            "trustMultiplier": self.trust_multiplier,
            "weight": self.weight,
            "admissible": self.admissible,
            "distance": self.distance,
            "distanceAllowance": self.distance_allowance,
        }


@dataclass
class EngineParams:
    context_base_distance: Dict[Context, float] = field(
        default_factory=lambda: {
            Context.SURVIVAL: 2.0,
            Context.SAFETY: 5.0,
            Context.SOCIAL: 10.0,
            Context.GROWTH: 10.0,
            Context.LUXURY: 15.0,
        }
    )
    context_upsilon: Dict[Context, float] = field(
        default_factory=lambda: {
            Context.SURVIVAL: 1.5,
            Context.SAFETY: 2.0,
            Context.SOCIAL: 1.8,
            Context.GROWTH: 2.5,
            Context.LUXURY: 3.0,
        }
    )
    context_omega: Dict[Context, float] = field(
        default_factory=lambda: {
            Context.SURVIVAL: 0.28,
            Context.SAFETY: 0.24,
            Context.SOCIAL: 0.2,
            Context.GROWTH: 0.18,
            Context.LUXURY: 0.14,
        }
    )
    min_provider_completion: Dict[Context, float] = field(
        default_factory=lambda: {
            Context.SURVIVAL: 0.4,
            Context.SAFETY: 0.45,
            Context.SOCIAL: 0.35,
            Context.GROWTH: 0.35,
            Context.LUXURY: 0.3,
        }
    )
    min_provider_quality: Dict[Context, float] = field(
        default_factory=lambda: {
            Context.SURVIVAL: 0.45,
            Context.SAFETY: 0.45,
            Context.SOCIAL: 0.35,
            Context.GROWTH: 0.35,
            Context.LUXURY: 0.3,
        }
    )
    min_receiver_completion: Dict[Context, float] = field(
        default_factory=lambda: {
            Context.SURVIVAL: 0.3,
            Context.SAFETY: 0.35,
            Context.SOCIAL: 0.0,
            Context.GROWTH: 0.0,
            Context.LUXURY: 0.0,
        }
    )

    alpha_patience_expansion: float = 0.48
    distance_kernel_alpha: float = 0.22
    generosity_beta: float = 0.22
    choice_penalty: float = -0.08
    trust_kappa: float = 0.55
    trust_ema_lambda: float = 0.85
    high_trust_completion_threshold: float = 0.78
    high_trust_quality_threshold: float = 0.76
    high_trust_edge_bonus: float = 0.18
    high_trust_distance_boost: float = 0.24
    high_trust_cycle_bonus: float = 0.28
    high_trust_confirmation_bonus: float = 0.1
    high_trust_execution_bonus: float = 0.08

    min_fit: float = 0.25
    max_cycle_length: int = 5
    invalid_edge_weight: float = -1_000_000.0

    execution_base: float = 0.52
    execution_trust_weight: float = 0.34
    execution_quality_weight: float = 0.22
    execution_length_penalty: float = 0.09
    execution_distance_penalty: float = 0.012

    def to_wire(self) -> Dict[str, Any]:
        return {
            "context_base_distance": {ctx.value: value for ctx, value in self.context_base_distance.items()},
            "context_upsilon": {ctx.value: value for ctx, value in self.context_upsilon.items()},
            "context_omega": {ctx.value: value for ctx, value in self.context_omega.items()},
            "min_provider_completion": {ctx.value: value for ctx, value in self.min_provider_completion.items()},
            "min_provider_quality": {ctx.value: value for ctx, value in self.min_provider_quality.items()},
            "min_receiver_completion": {ctx.value: value for ctx, value in self.min_receiver_completion.items()},
            "alpha_patience_expansion": self.alpha_patience_expansion,
            "distance_kernel_alpha": self.distance_kernel_alpha,
            "generosity_beta": self.generosity_beta,
            "choice_penalty": self.choice_penalty,
            "trust_kappa": self.trust_kappa,
            "trust_ema_lambda": self.trust_ema_lambda,
            "high_trust_completion_threshold": self.high_trust_completion_threshold,
            "high_trust_quality_threshold": self.high_trust_quality_threshold,
            "high_trust_edge_bonus": self.high_trust_edge_bonus,
            "high_trust_distance_boost": self.high_trust_distance_boost,
            "high_trust_cycle_bonus": self.high_trust_cycle_bonus,
            "high_trust_confirmation_bonus": self.high_trust_confirmation_bonus,
            "high_trust_execution_bonus": self.high_trust_execution_bonus,
            "min_fit": self.min_fit,
            "max_cycle_length": self.max_cycle_length,
            "execution_base": self.execution_base,
            "execution_trust_weight": self.execution_trust_weight,
            "execution_quality_weight": self.execution_quality_weight,
            "execution_length_penalty": self.execution_length_penalty,
            "execution_distance_penalty": self.execution_distance_penalty,
        }

    def with_overrides(self, overrides: Optional[Dict[str, Any]]) -> "EngineParams":
        if not overrides:
            return copy.deepcopy(self)

        params = copy.deepcopy(self)
        context_maps = {
            "context_base_distance": params.context_base_distance,
            "context_upsilon": params.context_upsilon,
            "context_omega": params.context_omega,
            "min_provider_completion": params.min_provider_completion,
            "min_provider_quality": params.min_provider_quality,
            "min_receiver_completion": params.min_receiver_completion,
        }

        for key, value in overrides.items():
            if key in context_maps and isinstance(value, dict):
                target = context_maps[key]
                for raw_context, raw_val in value.items():
                    context = _context_from_value(raw_context)
                    if context is None:
                        continue
                    target[context] = float(raw_val)
                continue

            if not hasattr(params, key):
                continue

            current = getattr(params, key)
            if isinstance(current, float):
                setattr(params, key, float(value))
            elif isinstance(current, int):
                setattr(params, key, int(value))
            else:
                setattr(params, key, value)

        return params


# ============================================================
# STATIC CATALOGS
# ============================================================

SOLUTION_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "fallback_edges",
        "title": "Fallback Edge Reserve",
        "detail": "Pre-compute backup counterparties so one CHOICE decline does not force a full rerun.",
        "effort": 2,
        "impact": 4,
        "addresses": ["choice_declines", "low_cycle_survival"],
    },
    {
        "id": "staged_atomicity",
        "title": "Staged Atomic Bundles",
        "detail": "Split large all-or-nothing bundles into milestones to reduce blast radius from one decline.",
        "effort": 4,
        "impact": 5,
        "addresses": ["atomic_abort", "unmatched_pressure"],
    },
    {
        "id": "guild_floor",
        "title": "Guild Trust Floor",
        "detail": "Apply a trust floor for verified cohorts so new participants are not filtered out immediately.",
        "effort": 2,
        "impact": 4,
        "addresses": ["cold_start_trust", "edge_dropoff"],
    },
    {
        "id": "distance_lanes",
        "title": "Context Distance Lanes",
        "detail": "Show context radius lanes in declaration UX so people understand why edges get pruned.",
        "effort": 2,
        "impact": 3,
        "addresses": ["edge_dropoff", "patience_drift"],
    },
    {
        "id": "choice_sla",
        "title": "Choice SLA Nudges",
        "detail": "Countdown reminders and temporary AUTO fallback after repeated CHOICE timeouts.",
        "effort": 3,
        "impact": 4,
        "addresses": ["choice_declines", "low_cycle_survival"],
    },
    {
        "id": "supply_seeding",
        "title": "Supply Seeding",
        "detail": "Seed trusted service pods in under-supplied skills before resolution day.",
        "effort": 4,
        "impact": 4,
        "addresses": ["unmatched_pressure", "edge_dropoff"],
    },
    {
        "id": "patience_hotlane",
        "title": "Patience Hotlane",
        "detail": "Route high-patience declarations to adjacent tessellations earlier.",
        "effort": 3,
        "impact": 3,
        "addresses": ["patience_drift", "unmatched_pressure"],
    },
    {
        "id": "trust_coaching",
        "title": "Trust Coaching Loop",
        "detail": "After failures, provide directed feedback to increase completion reliability.",
        "effort": 2,
        "impact": 3,
        "addresses": ["cold_start_trust", "low_cycle_survival"],
    },
]


# ============================================================
# SCENARIOS
# ============================================================


def _offer(skill: str, context: Context, tags: Tuple[str, ...] = ()) -> Offer:
    return Offer(skill=skill, context=context, tags=tags)


def _want(skill: str, context: Context, tags: Tuple[str, ...] = ()) -> Want:
    return Want(skill=skill, context=context, tags=tags)


def _agent(
    agent_id: str,
    name: str,
    mode: Mode,
    offers: List[Offer],
    wants: List[Want],
    x: float,
    y: float,
    completion: float,
    quality: float,
    patience: int,
    gave: int,
    received: int,
) -> Agent:
    return Agent(
        agent_id=agent_id,
        name=name,
        mode=mode,
        offers=offers,
        wants=wants,
        location=Location(x=x, y=y),
        trust=TrustMetrics(completion=completion, quality=quality),
        patience=patience,
        gave=gave,
        received=received,
    )


def build_scenarios() -> Dict[str, Scenario]:
    scenarios: List[Scenario] = [
        Scenario(
            key="balanced_local",
            name="Balanced Local 3-Cycle",
            note="Baseline sanity check with short distances and moderate trust.",
            agents=[
                _agent(
                    "a",
                    "Alice",
                    Mode.AUTO,
                    [_offer("haircut", Context.SOCIAL, ("grooming", "styling"))],
                    [_want("plumbing", Context.SAFETY, ("repair", "sink"))],
                    1.2,
                    1.5,
                    0.86,
                    0.83,
                    1,
                    9,
                    7,
                ),
                _agent(
                    "b",
                    "Bob",
                    Mode.AUTO,
                    [_offer("plumbing", Context.SAFETY, ("repair", "pipes"))],
                    [_want("yard", Context.SOCIAL, ("yard", "lawn"))],
                    2.1,
                    1.3,
                    0.83,
                    0.79,
                    0,
                    6,
                    6,
                ),
                _agent(
                    "c",
                    "Carol",
                    Mode.CHOICE,
                    [_offer("yard", Context.SOCIAL, ("landscaping", "lawn"))],
                    [_want("haircut", Context.SOCIAL, ("grooming",))],
                    1.8,
                    2.2,
                    0.75,
                    0.78,
                    1,
                    5,
                    6,
                ),
            ],
        ),
        Scenario(
            key="choice_friction",
            name="Choice Friction + Declines",
            note="CHOICE-heavy participants where confirmation becomes the bottleneck.",
            agents=[
                _agent(
                    "a",
                    "Alice",
                    Mode.CHOICE,
                    [_offer("haircut", Context.SOCIAL, ("grooming",))],
                    [_want("plumbing", Context.SAFETY, ("repair",))],
                    2.2,
                    1.4,
                    0.58,
                    0.69,
                    2,
                    4,
                    5,
                ),
                _agent(
                    "b",
                    "Bob",
                    Mode.CHOICE,
                    [_offer("plumbing", Context.SAFETY, ("repair", "pipes"))],
                    [_want("childcare", Context.SAFETY, ("care",))],
                    2.6,
                    2.3,
                    0.55,
                    0.64,
                    3,
                    2,
                    4,
                ),
                _agent(
                    "c",
                    "Carol",
                    Mode.CHOICE,
                    [_offer("childcare", Context.SAFETY, ("care", "supervision"))],
                    [_want("transport", Context.GROWTH, ("rides",))],
                    3.0,
                    2.9,
                    0.63,
                    0.68,
                    2,
                    5,
                    6,
                ),
                _agent(
                    "d",
                    "Diego",
                    Mode.AUTO,
                    [_offer("transport", Context.GROWTH, ("rides", "logistics"))],
                    [_want("haircut", Context.SOCIAL, ("grooming",))],
                    1.9,
                    2.1,
                    0.78,
                    0.77,
                    1,
                    7,
                    5,
                ),
            ],
        ),
        Scenario(
            key="atomic_dependency",
            name="Atomic Bundle Dependency",
            note="One participant is present in multiple linked cycles. A single decline can roll back all.",
            agents=[
                _agent(
                    "a",
                    "Alice",
                    Mode.CHOICE,
                    [
                        _offer("haircut", Context.SOCIAL, ("grooming",)),
                        _offer("haircut", Context.SOCIAL, ("styling",)),
                    ],
                    [
                        _want("plumbing", Context.SAFETY, ("repair",)),
                        _want("electrical", Context.SAFETY, ("wiring",)),
                    ],
                    1.4,
                    1.2,
                    0.66,
                    0.71,
                    3,
                    8,
                    9,
                ),
                _agent(
                    "b",
                    "Bob",
                    Mode.AUTO,
                    [_offer("plumbing", Context.SAFETY, ("repair",))],
                    [_want("yard", Context.SOCIAL, ("lawn",))],
                    1.7,
                    1.4,
                    0.82,
                    0.81,
                    1,
                    9,
                    6,
                ),
                _agent(
                    "c",
                    "Carol",
                    Mode.AUTO,
                    [_offer("yard", Context.SOCIAL, ("landscaping",))],
                    [_want("haircut", Context.SOCIAL, ("grooming",))],
                    2.2,
                    1.8,
                    0.8,
                    0.79,
                    1,
                    7,
                    7,
                ),
                _agent(
                    "d",
                    "Dante",
                    Mode.AUTO,
                    [_offer("electrical", Context.SAFETY, ("wiring",))],
                    [_want("mealprep", Context.SURVIVAL, ("food",))],
                    2.4,
                    2.0,
                    0.76,
                    0.72,
                    2,
                    4,
                    3,
                ),
                _agent(
                    "e",
                    "Eden",
                    Mode.CHOICE,
                    [_offer("mealprep", Context.SURVIVAL, ("food", "cooking"))],
                    [_want("haircut", Context.SOCIAL, ("styling",))],
                    1.8,
                    2.3,
                    0.59,
                    0.66,
                    4,
                    3,
                    6,
                ),
            ],
        ),
        Scenario(
            key="trust_coldstart",
            name="Trust Cold-Start + Distance Pruning",
            note="Low trust plus larger distances leaves too few admissible edges.",
            agents=[
                _agent(
                    "a",
                    "Alice",
                    Mode.AUTO,
                    [_offer("nursing", Context.SURVIVAL, ("care", "health"))],
                    [_want("transport", Context.SAFETY, ("rides",))],
                    1.0,
                    1.0,
                    0.41,
                    0.56,
                    4,
                    2,
                    4,
                ),
                _agent(
                    "b",
                    "Ben",
                    Mode.AUTO,
                    [_offer("transport", Context.SAFETY, ("rides", "logistics"))],
                    [_want("repair", Context.SAFETY, ("maintenance",))],
                    7.8,
                    5.9,
                    0.48,
                    0.59,
                    5,
                    1,
                    2,
                ),
                _agent(
                    "c",
                    "Cora",
                    Mode.CHOICE,
                    [_offer("repair", Context.SAFETY, ("maintenance", "fix"))],
                    [_want("mealprep", Context.SURVIVAL, ("food",))],
                    8.5,
                    7.1,
                    0.45,
                    0.54,
                    6,
                    2,
                    5,
                ),
                _agent(
                    "d",
                    "Dev",
                    Mode.CHOICE,
                    [_offer("mealprep", Context.SURVIVAL, ("food",))],
                    [_want("nursing", Context.SURVIVAL, ("care",))],
                    5.4,
                    6.2,
                    0.46,
                    0.57,
                    6,
                    2,
                    6,
                ),
            ],
        ),
        Scenario(
            key="high_trust_network",
            name="High-Trust Network Throughput",
            note="Dense, high-trust local network to stress throughput and multi-cycle reliability.",
            agents=[
                _agent("a", "Ari", Mode.CHOICE, [_offer("childcare", Context.SAFETY, ("care",))], [_want("legal", Context.SAFETY, ("forms",))], 1.0, 1.1, 0.93, 0.92, 1, 13, 10),
                _agent("b", "Bela", Mode.AUTO, [_offer("legal", Context.SAFETY, ("compliance",))], [_want("plumbing", Context.SAFETY, ("repair",))], 1.3, 1.0, 0.9, 0.89, 1, 12, 9),
                _agent("c", "Cian", Mode.AUTO, [_offer("plumbing", Context.SAFETY, ("repair",))], [_want("electrical", Context.SAFETY, ("wiring",))], 1.6, 1.2, 0.91, 0.9, 1, 11, 8),
                _agent("d", "Dina", Mode.CHOICE, [_offer("electrical", Context.SAFETY, ("wiring",))], [_want("childcare", Context.SAFETY, ("care",))], 1.8, 1.4, 0.89, 0.9, 2, 10, 9),
                _agent("e", "Evan", Mode.AUTO, [_offer("mediation", Context.SAFETY, ("conflict",))], [_want("transport", Context.SAFETY, ("rides",))], 2.2, 1.8, 0.9, 0.88, 1, 11, 8),
                _agent("f", "Fara", Mode.CHOICE, [_offer("transport", Context.SAFETY, ("rides",))], [_want("accounting", Context.SAFETY, ("books",))], 2.5, 2.1, 0.88, 0.9, 2, 10, 8),
                _agent("g", "Gabe", Mode.AUTO, [_offer("accounting", Context.SAFETY, ("books",))], [_want("translation", Context.SAFETY, ("language",))], 2.8, 2.2, 0.92, 0.9, 1, 12, 9),
                _agent("h", "Hana", Mode.AUTO, [_offer("translation", Context.SAFETY, ("language",))], [_want("mediation", Context.SAFETY, ("conflict",))], 3.0, 2.0, 0.9, 0.91, 1, 13, 10),
            ],
        ),
        Scenario(
            key="mesh_pressure",
            name="Mesh Pressure (8-Agent Stress)",
            note="High-declaration density stress test to force multi-cycle overlap and atomic cascades.",
            agents=[
                _agent("a", "Aria", Mode.CHOICE, [_offer("plumbing", Context.SAFETY), _offer("childcare", Context.SAFETY)], [_want("electrical", Context.SAFETY), _want("transport", Context.GROWTH)], 1.0, 1.0, 0.67, 0.7, 3, 7, 8),
                _agent("b", "Bex", Mode.AUTO, [_offer("electrical", Context.SAFETY)], [_want("mealprep", Context.SURVIVAL)], 1.4, 1.2, 0.78, 0.74, 2, 8, 6),
                _agent("c", "Cruz", Mode.CHOICE, [_offer("mealprep", Context.SURVIVAL)], [_want("haircut", Context.SOCIAL)], 1.8, 1.4, 0.62, 0.69, 3, 5, 7),
                _agent("d", "Dax", Mode.AUTO, [_offer("haircut", Context.SOCIAL)], [_want("plumbing", Context.SAFETY)], 2.1, 1.8, 0.81, 0.76, 2, 9, 8),
                _agent("e", "Eli", Mode.CHOICE, [_offer("transport", Context.GROWTH)], [_want("design", Context.GROWTH)], 2.5, 2.0, 0.64, 0.66, 4, 4, 6),
                _agent("f", "Faye", Mode.AUTO, [_offer("design", Context.GROWTH)], [_want("childcare", Context.SAFETY)], 2.8, 2.4, 0.77, 0.72, 2, 8, 7),
                _agent("g", "Gia", Mode.AUTO, [_offer("legal", Context.SAFETY)], [_want("plumbing", Context.SAFETY)], 3.1, 2.7, 0.73, 0.75, 3, 6, 6),
                _agent("h", "Hale", Mode.CHOICE, [_offer("plumbing", Context.SAFETY)], [_want("legal", Context.SAFETY)], 3.4, 3.1, 0.58, 0.63, 5, 3, 5),
            ],
        ),
    ]
    return {scenario.key: scenario for scenario in scenarios}


SCENARIOS = build_scenarios()
DEFAULT_PARAMS = EngineParams()


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def _context_from_value(value: Any) -> Optional[Context]:
    if isinstance(value, Context):
        return value
    if not isinstance(value, str):
        return None

    normalized = value.strip().upper()
    for context in Context:
        if context.value == normalized:
            return context
    return None


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _average(values: Iterable[float]) -> float:
    values_list = list(values)
    return sum(values_list) / len(values_list) if values_list else 0.0


def _distance_miles(a: Location, b: Location) -> float:
    dx = (a.x - b.x) * 7.5
    dy = (a.y - b.y) * 7.5
    return math.sqrt(dx * dx + dy * dy)


def _tokenize(text: str) -> set[str]:
    normalized = text.lower().replace("_", " ").replace("-", " ")
    return {token for token in normalized.split() if token}


def _fit_score(offer: Offer, want: Want) -> float:
    if offer.skill.lower() == want.skill.lower():
        return 1.0

    offer_tokens = _tokenize(offer.skill)
    want_tokens = _tokenize(want.skill)
    token_union = offer_tokens | want_tokens
    token_overlap = offer_tokens & want_tokens
    token_jaccard = len(token_overlap) / len(token_union) if token_union else 0.0

    offer_tags = set(offer.tags)
    want_tags = set(want.tags)
    tag_union = offer_tags | want_tags
    tag_overlap = offer_tags & want_tags
    tag_jaccard = len(tag_overlap) / len(tag_union) if tag_union else 0.0

    return max(token_jaccard, tag_jaccard)


# ============================================================
# EDGE + MATCHING PIPELINE
# ============================================================


def _build_nodes(agents: List[Agent]) -> Tuple[List[OfferNode], List[WantNode]]:
    offer_nodes: List[OfferNode] = []
    want_nodes: List[WantNode] = []

    for agent in agents:
        for offer_index, offer in enumerate(agent.offers):
            offer_nodes.append(
                OfferNode(
                    node_id=f"{agent.agent_id}:o{offer_index}",
                    agent_id=agent.agent_id,
                    agent_name=agent.name,
                    offer=offer,
                )
            )
        for want_index, want in enumerate(agent.wants):
            want_nodes.append(
                WantNode(
                    node_id=f"{agent.agent_id}:w{want_index}",
                    agent_id=agent.agent_id,
                    agent_name=agent.name,
                    want=want,
                )
            )

    return offer_nodes, want_nodes


def _build_agent_index(agents: List[Agent]) -> Dict[str, Agent]:
    return {agent.agent_id: agent for agent in agents}


def _centrality_map(agents: List[Agent]) -> Dict[str, float]:
    raw_scores: Dict[str, float] = {}
    for agent in agents:
        generosity = agent.gave / (agent.received + 1)
        raw_scores[agent.agent_id] = agent.trust.completion * agent.trust.quality * (1.0 + 0.15 * math.log1p(generosity))

    max_score = max(raw_scores.values()) if raw_scores else 1.0
    if max_score <= 0:
        return {agent_id: 0.0 for agent_id in raw_scores}
    return {agent_id: value / max_score for agent_id, value in raw_scores.items()}


def _trust_multiplier(provider: Agent, receiver: Agent, centrality: Dict[str, float], params: EngineParams) -> float:
    completion_similarity = 1.0 - abs(provider.trust.completion - receiver.trust.completion)
    quality_similarity = 1.0 - abs(provider.trust.quality - receiver.trust.quality)
    theta = _clamp(0.55 * completion_similarity + 0.45 * quality_similarity, 0.0, 1.0)

    tau_provider = centrality.get(provider.agent_id, 0.0)
    tau_receiver = centrality.get(receiver.agent_id, 0.0)
    tau_avg = 0.5 * (tau_provider + tau_receiver)

    psi = math.sqrt(max(theta * tau_avg, 0.0))
    return 1.0 + params.trust_kappa * psi


def _high_trust_signal(provider: Agent, receiver: Agent, params: EngineParams) -> float:
    completion_floor = min(provider.trust.completion, receiver.trust.completion)
    quality_floor = min(provider.trust.quality, receiver.trust.quality)

    completion_den = max(1.0 - params.high_trust_completion_threshold, 1e-9)
    quality_den = max(1.0 - params.high_trust_quality_threshold, 1e-9)

    completion_strength = _clamp((completion_floor - params.high_trust_completion_threshold) / completion_den, 0.0, 1.0)
    quality_strength = _clamp((quality_floor - params.high_trust_quality_threshold) / quality_den, 0.0, 1.0)
    return 0.5 * (completion_strength + quality_strength)


def _compute_edge(
    provider: Agent,
    receiver: Agent,
    offer_node: OfferNode,
    want_node: WantNode,
    centrality: Dict[str, float],
    params: EngineParams,
) -> Optional[Edge]:
    offer = offer_node.offer
    want = want_node.want

    if offer.context is not want.context:
        return None

    fit = _fit_score(offer, want)
    if fit < params.min_fit:
        return None

    context = offer.context

    if provider.trust.completion < params.min_provider_completion[context]:
        return None
    if provider.trust.quality < params.min_provider_quality[context]:
        return None
    if receiver.trust.completion < params.min_receiver_completion[context]:
        return None

    distance = _distance_miles(provider.location, receiver.location)
    trust_multiplier = _trust_multiplier(provider, receiver, centrality, params)
    high_trust_signal = _high_trust_signal(provider, receiver, params)

    base_distance = params.context_base_distance[context]
    upsilon = params.context_upsilon[context]
    expansion = params.alpha_patience_expansion * math.log1p(receiver.patience)
    distance_allowance = (
        base_distance
        * (1 + expansion * upsilon * trust_multiplier)
        * (1 + params.high_trust_distance_boost * high_trust_signal)
    )

    if distance > distance_allowance:
        return None

    slack = distance_allowance - distance
    proximity = 1.0 / (1.0 + math.exp(-params.distance_kernel_alpha * slack))

    generosity = provider.gave / (provider.received + 1)
    generosity_bonus = params.generosity_beta * math.log1p(max(generosity, 0.0))
    choice_bonus = params.choice_penalty if receiver.mode is Mode.CHOICE else 0.0
    patience_bonus = params.context_omega[context] * math.log1p(receiver.patience)
    high_trust_bonus = params.high_trust_edge_bonus * high_trust_signal

    base_quality = (
        0.4 * fit
        + 0.28 * provider.trust.quality
        + 0.16 * provider.trust.completion
        + 0.16 * receiver.trust.completion
    )

    weight = base_quality * proximity + generosity_bonus + choice_bonus + patience_bonus + high_trust_bonus

    return Edge(
        edge_id=f"{offer_node.node_id}->{want_node.node_id}",
        offer_node=offer_node.node_id,
        want_node=want_node.node_id,
        provider_id=provider.agent_id,
        provider_name=provider.name,
        receiver_id=receiver.agent_id,
        receiver_name=receiver.name,
        skill=offer.skill,
        context=context.value,
        mode=receiver.mode.value,
        fit_score=fit,
        proximity=proximity,
        base_quality=base_quality,
        generosity_bonus=generosity_bonus,
        choice_bonus=choice_bonus,
        patience_bonus=patience_bonus,
        high_trust_bonus=high_trust_bonus,
        trust_multiplier=trust_multiplier,
        weight=weight,
        admissible=True,
        distance=distance,
        distance_allowance=distance_allowance,
    )


def _build_edges(
    agents: List[Agent],
    offer_nodes: List[OfferNode],
    want_nodes: List[WantNode],
    params: EngineParams,
) -> Tuple[List[Edge], List[Edge]]:
    by_agent = _build_agent_index(agents)
    centrality = _centrality_map(agents)

    potential_edges: List[Edge] = []
    feasible_edges: List[Edge] = []

    for offer_node in offer_nodes:
        for want_node in want_nodes:
            if offer_node.agent_id == want_node.agent_id:
                continue

            provider = by_agent[offer_node.agent_id]
            receiver = by_agent[want_node.agent_id]

            edge = _compute_edge(provider, receiver, offer_node, want_node, centrality, params)
            if edge is None:
                continue

            potential_edges.append(edge)
            feasible_edges.append(edge)

    feasible_edges.sort(key=lambda edge: edge.weight, reverse=True)
    return potential_edges, feasible_edges


def _hungarian_min_cost(cost: List[List[float]]) -> List[int]:
    n = len(cost)
    m = len(cost[0]) if n else 0
    if n == 0 or m == 0:
        return []

    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [math.inf] * (m + 1)
        used = [False] * (m + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = math.inf
            j1 = 0

            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j] > 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def _hungarian_maximize(weights: List[List[float]]) -> List[int]:
    rows = len(weights)
    cols = len(weights[0]) if rows else 0
    if rows == 0 or cols == 0:
        return []

    transposed = False
    work_weights = weights

    if rows > cols:
        transposed = True
        work_weights = [list(row) for row in zip(*weights)]

    cost = [[-value for value in row] for row in work_weights]
    assignment = _hungarian_min_cost(cost)

    if not transposed:
        return assignment

    # assignment maps transposed rows (original columns) -> transposed cols (original rows)
    # Convert to original row -> original column mapping.
    original_assignment = [-1] * rows
    for trans_row, trans_col in enumerate(assignment):
        if trans_col >= 0:
            original_assignment[trans_col] = trans_row
    return original_assignment


def _max_weight_matching(
    edges: List[Edge],
    offer_nodes: List[OfferNode],
    want_nodes: List[WantNode],
    params: EngineParams,
) -> List[Edge]:
    if not offer_nodes or not want_nodes or not edges:
        return []

    edge_lookup: Dict[Tuple[str, str], Edge] = {
        (edge.offer_node, edge.want_node): edge for edge in edges
    }

    real_cols = len(want_nodes)
    pass_cols = len(offer_nodes)

    matrix: List[List[float]] = []
    for offer_node in offer_nodes:
        row = [params.invalid_edge_weight] * (real_cols + pass_cols)

        for want_index, want_node in enumerate(want_nodes):
            edge = edge_lookup.get((offer_node.node_id, want_node.node_id))
            if edge is not None:
                row[want_index] = edge.weight

        for pass_index in range(pass_cols):
            row[real_cols + pass_index] = 0.0

        matrix.append(row)

    assignment = _hungarian_maximize(matrix)

    matching: List[Edge] = []
    for offer_index, col in enumerate(assignment):
        if col < 0 or col >= real_cols:
            continue

        offer_node = offer_nodes[offer_index]
        want_node = want_nodes[col]
        edge = edge_lookup.get((offer_node.node_id, want_node.node_id))
        if edge is not None:
            matching.append(edge)

    matching.sort(key=lambda edge: edge.weight, reverse=True)
    return matching


# ============================================================
# CYCLE DETECTION + EXECUTION
# ============================================================


def _cycle_trust_cohesion(
    member_ids: List[str],
    agents_by_id: Dict[str, Agent],
    params: EngineParams,
) -> float:
    members = [agents_by_id[agent_id] for agent_id in member_ids]
    completion_floor = min((agent.trust.completion for agent in members), default=0.0)
    quality_floor = min((agent.trust.quality for agent in members), default=0.0)

    completion_den = max(1.0 - params.high_trust_completion_threshold, 1e-9)
    quality_den = max(1.0 - params.high_trust_quality_threshold, 1e-9)

    completion_strength = _clamp((completion_floor - params.high_trust_completion_threshold) / completion_den, 0.0, 1.0)
    quality_strength = _clamp((quality_floor - params.high_trust_quality_threshold) / quality_den, 0.0, 1.0)
    return 0.5 * (completion_strength + quality_strength)


def _detect_cycles(
    matching_edges: List[Edge],
    max_cycle_len: int,
    agents_by_id: Dict[str, Agent],
    params: EngineParams,
) -> List[Dict[str, Any]]:
    if not matching_edges:
        return []

    edges_by_provider: Dict[str, List[int]] = {}
    for idx, edge in enumerate(matching_edges):
        edges_by_provider.setdefault(edge.provider_id, []).append(idx)

    discovered: List[Dict[str, Any]] = []
    seen_signatures: set[Tuple[int, ...]] = set()

    def dfs(start_agent: str, current_agent: str, path_edge_ids: List[int], visited_agents: set[str]) -> None:
        for edge_idx in edges_by_provider.get(current_agent, []):
            edge = matching_edges[edge_idx]
            nxt = edge.receiver_id

            if nxt == start_agent and path_edge_ids:
                cycle_edge_ids = path_edge_ids + [edge_idx]
                if len(cycle_edge_ids) < 2 or len(cycle_edge_ids) > max_cycle_len:
                    continue

                signature = tuple(sorted(cycle_edge_ids))
                if signature in seen_signatures:
                    continue

                seen_signatures.add(signature)
                cycle_edges = [matching_edges[idx] for idx in cycle_edge_ids]
                member_ids = sorted({e.provider_id for e in cycle_edges} | {e.receiver_id for e in cycle_edges})
                trust_cohesion = _cycle_trust_cohesion(member_ids, agents_by_id, params)
                total_weight = sum(e.weight for e in cycle_edges)
                selection_score = total_weight + params.high_trust_cycle_bonus * trust_cohesion
                discovered.append(
                    {
                        "edgeIndices": cycle_edge_ids,
                        "edges": cycle_edges,
                        "memberIds": member_ids,
                        "length": len(cycle_edge_ids),
                        "totalWeight": total_weight,
                        "trustCohesion": trust_cohesion,
                        "selectionScore": selection_score,
                        "avgDistance": _average(e.distance for e in cycle_edges),
                    }
                )
                continue

            if nxt in visited_agents or len(path_edge_ids) + 1 >= max_cycle_len:
                continue

            visited_agents.add(nxt)
            path_edge_ids.append(edge_idx)
            dfs(start_agent, nxt, path_edge_ids, visited_agents)
            path_edge_ids.pop()
            visited_agents.remove(nxt)

    for start_agent in edges_by_provider.keys():
        dfs(start_agent, start_agent, [], {start_agent})

    discovered.sort(key=lambda cycle: cycle["selectionScore"], reverse=True)
    return discovered


def _select_edge_disjoint_cycles(cycles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    used_edges: set[int] = set()

    for cycle in cycles:
        edge_ids = set(cycle["edgeIndices"])
        if edge_ids & used_edges:
            continue

        selected.append(cycle)
        used_edges.update(edge_ids)

    for index, cycle in enumerate(selected, start=1):
        cycle["cycleId"] = f"cycle_{index:03d}"

    return selected


def _agent_participation_map(cycles: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    participation: Dict[str, List[int]] = {}
    for idx, cycle in enumerate(cycles):
        for agent_id in cycle["memberIds"]:
            participation.setdefault(agent_id, []).append(idx)
    return participation


def _confirm_cycles(
    cycles: List[Dict[str, Any]],
    agents_by_id: Dict[str, Agent],
    params: EngineParams,
    rng: random.Random,
) -> int:
    choice_declines = 0
    participation = _agent_participation_map(cycles)

    for cycle in cycles:
        declined_by: List[str] = []

        for agent_id in cycle["memberIds"]:
            agent = agents_by_id[agent_id]
            if agent.mode is not Mode.CHOICE:
                continue

            counterparties = [
                agents_by_id[other_id].trust.completion
                for other_id in cycle["memberIds"]
                if other_id != agent_id
            ]
            trust_in_counterparties = _average(counterparties)
            load_penalty = 0.05 * max(participation.get(agent_id, [0]).__len__() - 1, 0)
            high_trust_bonus = params.high_trust_confirmation_bonus * float(cycle.get("trustCohesion", 0.0))
            confirm_probability = _clamp(
                0.62 + 0.24 * trust_in_counterparties + 0.08 * agent.trust.quality + high_trust_bonus - load_penalty,
                0.25,
                0.98,
            )

            if rng.random() > confirm_probability:
                declined_by.append(agent.name)
                choice_declines += 1

        cycle["declinedBy"] = declined_by
        cycle["confirmed"] = len(declined_by) == 0

    return choice_declines


def _apply_atomic_commit(cycles: List[Dict[str, Any]]) -> int:
    participation = _agent_participation_map(cycles)
    commit_state = [bool(cycle.get("confirmed", False)) for cycle in cycles]

    changed = True
    while changed:
        changed = False
        for cycle_indices in participation.values():
            if all(commit_state[idx] for idx in cycle_indices):
                continue
            for idx in cycle_indices:
                if commit_state[idx]:
                    commit_state[idx] = False
                    changed = True

    atomic_abort_count = 0
    for idx, cycle in enumerate(cycles):
        cycle["atomicBlocked"] = bool(cycle.get("confirmed", False) and not commit_state[idx])
        cycle["committed"] = bool(commit_state[idx])
        if cycle["atomicBlocked"]:
            atomic_abort_count += 1

    return atomic_abort_count


def _simulate_execution(
    cycles: List[Dict[str, Any]],
    agents_by_id: Dict[str, Agent],
    params: EngineParams,
    rng: random.Random,
) -> Dict[str, Any]:
    participation = _agent_participation_map(cycles)

    completion_state = [False] * len(cycles)
    ratings_by_provider: Dict[str, List[float]] = {}

    for idx, cycle in enumerate(cycles):
        if not cycle.get("committed", False):
            cycle["executionProbability"] = 0.0
            cycle["completedInitial"] = False
            continue

        edges = cycle["edges"]
        members = [agents_by_id[agent_id] for agent_id in cycle["memberIds"]]

        avg_completion = _average(agent.trust.completion for agent in members)
        avg_quality = _average(agent.trust.quality for agent in members)
        avg_distance = _average(edge.distance for edge in edges)
        high_trust_bonus = params.high_trust_execution_bonus * float(cycle.get("trustCohesion", 0.0))

        completion_prob = _clamp(
            params.execution_base
            + params.execution_trust_weight * avg_completion
            + params.execution_quality_weight * avg_quality
            + high_trust_bonus
            - params.execution_length_penalty * max(cycle["length"] - 2, 0)
            - params.execution_distance_penalty * avg_distance,
            0.1,
            0.98,
        )

        cycle["executionProbability"] = completion_prob
        cycle["completedInitial"] = rng.random() <= completion_prob
        completion_state[idx] = bool(cycle["completedInitial"])

    # Atomic completion cascade: if any cycle in an agent's committed set fails, all fail.
    changed = True
    while changed:
        changed = False
        for cycle_indices in participation.values():
            committed_indices = [idx for idx in cycle_indices if cycles[idx].get("committed", False)]
            if not committed_indices:
                continue
            if all(completion_state[idx] for idx in committed_indices):
                continue
            for idx in committed_indices:
                if completion_state[idx]:
                    completion_state[idx] = False
                    changed = True

    completed_cycles = 0
    failed_execution_cycles = 0
    rolled_back_cycles = 0

    for idx, cycle in enumerate(cycles):
        if not cycle.get("committed", False):
            cycle["completed"] = False
            cycle["executionRolledBack"] = False
            continue

        completed = completion_state[idx]
        rolled_back = bool(cycle.get("completedInitial", False) and not completed)

        cycle["completed"] = completed
        cycle["executionRolledBack"] = rolled_back

        if completed:
            completed_cycles += 1
            for edge in cycle["edges"]:
                provider = agents_by_id[edge.provider_id]
                rating = _clamp(rng.gauss(3.1 + provider.trust.quality * 1.7, 0.45), 1.0, 5.0)
                ratings_by_provider.setdefault(provider.agent_id, []).append(rating)
        else:
            failed_execution_cycles += 1
            if rolled_back:
                rolled_back_cycles += 1

    return {
        "completedCycles": completed_cycles,
        "failedExecutionCycles": failed_execution_cycles,
        "rolledBackCycles": rolled_back_cycles,
        "ratingsByProvider": ratings_by_provider,
    }


# ============================================================
# TRUST + PATIENCE PROJECTION
# ============================================================


def _project_trust_and_patience(
    agents: List[Agent],
    cycles: List[Dict[str, Any]],
    params: EngineParams,
    ratings_by_provider: Dict[str, List[float]],
) -> Dict[str, Any]:
    projected_completion: Dict[str, float] = {agent.agent_id: agent.trust.completion for agent in agents}
    projected_quality: Dict[str, float] = {agent.agent_id: agent.trust.quality for agent in agents}
    projected_patience: Dict[str, int] = {agent.agent_id: agent.patience for agent in agents}

    outcomes_by_agent: Dict[str, List[float]] = {}
    fulfilled_wants: set[str] = set()

    for cycle in cycles:
        if not cycle.get("committed", False):
            continue

        outcome_value = 1.0 if cycle.get("completed", False) else 0.0
        for agent_id in cycle["memberIds"]:
            outcomes_by_agent.setdefault(agent_id, []).append(outcome_value)

        if cycle.get("completed", False):
            for edge in cycle["edges"]:
                fulfilled_wants.add(edge.want_node)

    for agent_id, outcomes in outcomes_by_agent.items():
        completion = projected_completion[agent_id]
        for outcome in outcomes:
            completion = params.trust_ema_lambda * completion + (1 - params.trust_ema_lambda) * outcome
        projected_completion[agent_id] = _clamp(completion, 0.0, 1.0)

    for provider_id, ratings in ratings_by_provider.items():
        quality = projected_quality[provider_id]
        for rating in ratings:
            normalized = (rating - 1.0) / 4.0
            quality = params.trust_ema_lambda * quality + (1 - params.trust_ema_lambda) * normalized
        projected_quality[provider_id] = _clamp(quality, 0.0, 1.0)

    total_wants = 0
    unmet_after_execution = 0
    for agent in agents:
        total_wants += len(agent.wants)
        agent_want_nodes = {f"{agent.agent_id}:w{idx}" for idx, _ in enumerate(agent.wants)}
        agent_fulfilled = len(agent_want_nodes & fulfilled_wants)

        if agent_fulfilled == len(agent.wants) and agent.wants:
            projected_patience[agent.agent_id] = 0
        elif agent.wants:
            projected_patience[agent.agent_id] = agent.patience + 1
            unmet_after_execution += len(agent.wants) - agent_fulfilled

    avg_completion_before = _average(agent.trust.completion for agent in agents)
    avg_completion_after = _average(projected_completion.values())
    avg_patience_before = _average(float(agent.patience) for agent in agents)
    avg_patience_after = _average(float(value) for value in projected_patience.values())

    return {
        "projectedCompletionTrust": projected_completion,
        "projectedQualityTrust": projected_quality,
        "projectedPatience": projected_patience,
        "avgCompletionBefore": avg_completion_before,
        "avgCompletionAfter": avg_completion_after,
        "avgPatienceBefore": avg_patience_before,
        "avgPatienceAfter": avg_patience_after,
        "unmetWantsAfterExecution": unmet_after_execution,
        "fulfilledWantNodes": fulfilled_wants,
        "totalWants": total_wants,
    }


# ============================================================
# METRICS, PAINS, RANKING
# ============================================================


def _derive_metrics(
    agents: List[Agent],
    offer_nodes: List[OfferNode],
    want_nodes: List[WantNode],
    potential_edges: List[Edge],
    feasible_edges: List[Edge],
    matching: List[Edge],
    cycles_detected: List[Dict[str, Any]],
    cycles_selected: List[Dict[str, Any]],
    choice_declines: int,
    atomic_abort_count: int,
    execution_outcomes: Dict[str, Any],
    projection: Dict[str, Any],
) -> Dict[str, Any]:
    total_offers = len(offer_nodes)
    total_wants = len(want_nodes)

    matched_wants = {edge.want_node for edge in matching}
    high_trust_edges = sum(1 for edge in matching if edge.high_trust_bonus > 1e-9)
    high_trust_cycles = sum(1 for cycle in cycles_selected if float(cycle.get("trustCohesion", 0.0)) > 1e-9)

    cycle_count = len(cycles_selected)
    confirmed_cycles = sum(1 for cycle in cycles_selected if cycle.get("confirmed", False))
    committed_cycles = sum(1 for cycle in cycles_selected if cycle.get("committed", False))
    completed_cycles = execution_outcomes["completedCycles"]
    failed_execution_cycles = execution_outcomes["failedExecutionCycles"]

    avg_cycle_length = _average(cycle["length"] for cycle in cycles_selected)
    max_cycle_length = max((cycle["length"] for cycle in cycles_selected), default=0)

    avg_distance_matched = _average(edge.distance for edge in matching)
    avg_distance_completed = _average(
        edge.distance for cycle in cycles_selected if cycle.get("completed", False) for edge in cycle["edges"]
    )

    return {
        "totalOffers": total_offers,
        "totalWants": total_wants,
        "potentialEdges": len(potential_edges),
        "feasibleEdges": len(feasible_edges),
        "edgeRatio": len(feasible_edges) / len(potential_edges) if potential_edges else 0.0,
        "matchingSize": len(matching),
        "highTrustEdgeShare": high_trust_edges / len(matching) if matching else 0.0,
        "cyclesDetected": len(cycles_detected),
        "cycleCount": cycle_count,
        "highTrustCycleShare": high_trust_cycles / cycle_count if cycle_count else 0.0,
        "confirmedCycles": confirmed_cycles,
        "committedCycles": committed_cycles,
        "completedCycles": completed_cycles,
        "failedCycles": cycle_count - completed_cycles,
        "failedConfirmationCycles": cycle_count - confirmed_cycles,
        "failedExecutionCycles": failed_execution_cycles,
        "cycleSurvival": completed_cycles / cycle_count if cycle_count else 0.0,
        "atomicAbortCount": atomic_abort_count,
        "choiceDeclines": choice_declines,
        "unmatchedWantsAfterMatch": max(total_wants - len(matched_wants), 0),
        "unmatchedWants": max(total_wants - len(matched_wants), 0),
        "unmetWantsAfterExecution": projection["unmetWantsAfterExecution"],
        "avgPatience": projection["avgPatienceBefore"],
        "nextAvgPatience": projection["avgPatienceAfter"],
        "avgTrust": projection["avgCompletionBefore"],
        "nextAvgTrust": projection["avgCompletionAfter"],
        "avgCycleLength": avg_cycle_length,
        "maxCycleLength": max_cycle_length,
        "avgDistanceMatched": avg_distance_matched,
        "avgDistanceCompleted": avg_distance_completed,
        "rolledBackCycles": execution_outcomes["rolledBackCycles"],
    }


def _detect_pains(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    pains: List[Dict[str, Any]] = []

    if metrics["edgeRatio"] < 0.45 and metrics["potentialEdges"] > 0:
        pains.append(
            {
                "id": "edge_dropoff",
                "title": "Edge graph shrinks too fast",
                "description": "Distance and trust admissibility prune too many potential trades before matching.",
                "severity": int(_clamp(round((0.45 - metrics["edgeRatio"]) * 8) + 2, 2, 5)),
            }
        )

    if metrics["choiceDeclines"] > 0:
        pains.append(
            {
                "id": "choice_declines",
                "title": "CHOICE confirmation friction",
                "description": "CHOICE declines/timeouts are collapsing otherwise valid cycles.",
                "severity": int(_clamp(2 + metrics["choiceDeclines"], 2, 5)),
            }
        )

    if metrics["atomicAbortCount"] > 0 or metrics["rolledBackCycles"] > 0:
        pains.append(
            {
                "id": "atomic_abort",
                "title": "Atomic bundle fragility",
                "description": "One failure in a linked cycle-set is rolling back additional committed work.",
                "severity": int(_clamp(2 + metrics["atomicAbortCount"] + metrics["rolledBackCycles"], 2, 5)),
            }
        )

    if metrics["avgTrust"] < 0.62:
        pains.append(
            {
                "id": "cold_start_trust",
                "title": "Trust floor too low",
                "description": "Low completion trust drags edge weights and discourages confirmations.",
                "severity": int(_clamp(round((0.62 - metrics["avgTrust"]) * 10) + 2, 2, 5)),
            }
        )

    unmet_rate = metrics["unmetWantsAfterExecution"] / max(metrics["totalWants"], 1)
    if unmet_rate > 0.34:
        pains.append(
            {
                "id": "unmatched_pressure",
                "title": "Unmatched demand pressure",
                "description": "Too many wants stay unresolved after execution, creating backlog pressure.",
                "severity": int(_clamp(round(unmet_rate * 6), 2, 5)),
            }
        )

    if metrics["avgPatience"] > 2.7:
        pains.append(
            {
                "id": "patience_drift",
                "title": "Patience accumulation",
                "description": "Long wait times are forcing weak long-distance fits and lower confidence.",
                "severity": int(_clamp(round(metrics["avgPatience"] / 1.4), 2, 5)),
            }
        )

    if metrics["cycleCount"] > 0 and metrics["cycleSurvival"] < 0.5:
        pains.append(
            {
                "id": "low_cycle_survival",
                "title": "Low cycle execution survival",
                "description": "Detected cycles are not surviving confirmation and execution to completion.",
                "severity": int(_clamp(round((0.5 - metrics["cycleSurvival"]) * 10) + 2, 2, 5)),
            }
        )

    if not pains:
        pains.append(
            {
                "id": "stable_baseline",
                "title": "Stable baseline",
                "description": "No severe bottleneck fired in this run. Switch scenarios to stress failure modes.",
                "severity": 1,
            }
        )

    return pains


def rank_solutions(pains: List[Dict[str, Any]], pain_levels: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
    pain_levels = pain_levels or {}
    severity_by_id: Dict[str, int] = {}
    for pain in pains:
        if isinstance(pain, dict):
            pain_id = str(pain.get("id", "")).strip()
            base_value = pain.get("severity", 3)
        else:
            pain_id = str(pain).strip()
            base_value = 3

        if not pain_id:
            continue

        raw_level = pain_levels.get(pain_id, base_value)
        try:
            severity = int(raw_level)
        except (TypeError, ValueError):
            try:
                severity = int(base_value)
            except (TypeError, ValueError):
                severity = 3

        severity_by_id[pain_id] = int(_clamp(float(severity), 1, 5))

    ranked: List[Dict[str, Any]] = []
    for solution in SOLUTION_CATALOG:
        weight = sum(severity_by_id.get(pain_id, 0) for pain_id in solution["addresses"])
        score = weight * solution["impact"] - solution["effort"] * 2.2
        ranked.append({**solution, "score": round(score, 3)})

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def _build_workflow(metrics: Dict[str, Any]) -> List[Dict[str, str]]:
    edge_state = "state-warn" if metrics["edgeRatio"] < 0.45 else "state-ok"
    cycle_state = "state-fail" if metrics["cycleCount"] == 0 else "state-ok"
    confirmation_state = "state-warn" if metrics["failedConfirmationCycles"] > 0 else "state-ok"
    atomic_state = "state-warn" if metrics["atomicAbortCount"] > 0 or metrics["rolledBackCycles"] > 0 else "state-ok"
    execution_state = "state-warn" if metrics["failedExecutionCycles"] > 0 else "state-ok"

    return [
        {
            "title": "1) Declaration Phase",
            "detail": f"{metrics['totalOffers']} offers and {metrics['totalWants']} wants submitted",
            "className": "state-ok",
        },
        {
            "title": "2) Edge Construction",
            "detail": f"{metrics['feasibleEdges']}/{metrics['potentialEdges']} admissible edges survived gates",
            "className": edge_state,
        },
        {
            "title": "3) Hungarian Assignment + Cycle Detection",
            "detail": f"{metrics['matchingSize']} matched declarations, {metrics['cycleCount']} selected cycles",
            "className": cycle_state,
        },
        {
            "title": "4) Confirmation",
            "detail": (
                f"{metrics['failedConfirmationCycles']} cycles failed in confirmation"
                if metrics["failedConfirmationCycles"]
                else "All selected cycles confirmed"
            ),
            "className": confirmation_state,
        },
        {
            "title": "5) Atomic Commit",
            "detail": f"{metrics['committedCycles']} committed, {metrics['atomicAbortCount']} blocked by atomic dependency",
            "className": atomic_state,
        },
        {
            "title": "6) Execution + Trust Update",
            "detail": f"{metrics['completedCycles']} completed, {metrics['failedExecutionCycles']} failed in execution",
            "className": execution_state,
        },
    ]


# ============================================================
# RUNTIME PIPELINE
# ============================================================


def _agent_to_wire(agent: Agent) -> Dict[str, Any]:
    return {
        "id": agent.agent_id,
        "name": agent.name,
        "mode": agent.mode.value,
        "offers": [{"skill": offer.skill, "context": offer.context.value, "tags": list(offer.tags)} for offer in agent.offers],
        "wants": [{"skill": want.skill, "context": want.context.value, "tags": list(want.tags)} for want in agent.wants],
        "location": {"x": agent.location.x, "y": agent.location.y},
        "trust": {"completion": agent.trust.completion, "quality": agent.trust.quality},
        "patience": agent.patience,
        "gave": agent.gave,
        "received": agent.received,
    }


def _cycle_path(cycle: Dict[str, Any], agents_by_id: Dict[str, Agent]) -> List[str]:
    edges = cycle["edges"]
    if not edges:
        return []
    path = [agents_by_id[edges[0].provider_id].name]
    for edge in edges:
        path.append(agents_by_id[edge.receiver_id].name)
    return path


def _run_pipeline(scenario: Scenario, params: EngineParams, seed: Optional[int]) -> Dict[str, Any]:
    agents = copy.deepcopy(scenario.agents)
    agents_by_id = _build_agent_index(agents)

    offer_nodes, want_nodes = _build_nodes(agents)
    potential_edges, feasible_edges = _build_edges(agents, offer_nodes, want_nodes, params)

    matching = _max_weight_matching(feasible_edges, offer_nodes, want_nodes, params)

    cycles_detected = _detect_cycles(
        matching,
        max_cycle_len=params.max_cycle_length,
        agents_by_id=agents_by_id,
        params=params,
    )
    cycles_selected = _select_edge_disjoint_cycles(cycles_detected)

    effective_seed = seed if seed is not None else random.randint(1, 1_000_000_000)
    rng = random.Random(effective_seed)

    choice_declines = _confirm_cycles(cycles_selected, agents_by_id, params, rng)
    atomic_abort_count = _apply_atomic_commit(cycles_selected)
    execution_outcomes = _simulate_execution(cycles_selected, agents_by_id, params, rng)
    projection = _project_trust_and_patience(
        agents,
        cycles_selected,
        params,
        execution_outcomes["ratingsByProvider"],
    )

    metrics = _derive_metrics(
        agents,
        offer_nodes,
        want_nodes,
        potential_edges,
        feasible_edges,
        matching,
        cycles_detected,
        cycles_selected,
        choice_declines,
        atomic_abort_count,
        execution_outcomes,
        projection,
    )

    pains = _detect_pains(metrics)
    ranked_solutions = rank_solutions(pains)
    workflow = _build_workflow(metrics)

    cycles_wire: List[Dict[str, Any]] = []
    for cycle in cycles_selected:
        status = "Completed" if cycle.get("completed", False) else "Failed"
        if not cycle.get("confirmed", False):
            status = "Declined by " + ", ".join(cycle.get("declinedBy", []))
        elif cycle.get("atomicBlocked", False):
            status = "Aborted by atomic dependency"
        elif cycle.get("executionRolledBack", False):
            status = "Rolled back by atomic execution"
        elif cycle.get("committed", False) and not cycle.get("completed", False):
            status = "Committed but failed execution"

        cycles_wire.append(
            {
                "cycleId": cycle["cycleId"],
                "path": _cycle_path(cycle, agents_by_id),
                "status": status,
                "length": cycle["length"],
                "totalWeight": cycle["totalWeight"],
                "selectionScore": cycle.get("selectionScore", cycle["totalWeight"]),
                "trustCohesion": cycle.get("trustCohesion", 0.0),
                "avgDistance": cycle["avgDistance"],
                "confirmed": cycle.get("confirmed", False),
                "committed": cycle.get("committed", False),
                "completed": cycle.get("completed", False),
                "atomicBlocked": cycle.get("atomicBlocked", False),
                "executionRolledBack": cycle.get("executionRolledBack", False),
                "declinedBy": cycle.get("declinedBy", []),
            }
        )

    return {
        "scenarioKey": scenario.key,
        "scenarioName": scenario.name,
        "scenarioNote": scenario.note,
        "seed": effective_seed,
        "paramsUsed": params.to_wire(),
        "agents": [_agent_to_wire(agent) for agent in agents],
        "matching": [edge.to_wire() for edge in matching],
        "cycles": cycles_wire,
        "metrics": metrics,
        "workflow": workflow,
        "pains": pains,
        "solutionCatalog": SOLUTION_CATALOG,
        "rankedSolutions": ranked_solutions,
        "projections": {
            "nextAvgTrust": projection["avgCompletionAfter"],
            "nextAvgPatience": projection["avgPatienceAfter"],
        },
    }


def run_scenario(scenario_key: str, seed: Optional[int], params_override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if scenario_key not in SCENARIOS:
        raise ValueError(f"Unknown scenario_key: {scenario_key}")

    scenario = SCENARIOS[scenario_key]
    params = DEFAULT_PARAMS.with_overrides(params_override)
    return _run_pipeline(scenario, params, seed)


# ============================================================
# PARAMETER OPTIMIZATION
# ============================================================


OBJECTIVE_PROFILES = {"balanced", "high_trust_network"}


def _objective(metrics: Dict[str, Any], profile: str = "balanced") -> float:
    if profile == "high_trust_network":
        return (
            6.0 * metrics["completedCycles"]
            + 1.6 * metrics["matchingSize"]
            + 2.8 * metrics["nextAvgTrust"]
            + 2.0 * metrics["cycleSurvival"]
            - 2.4 * metrics["unmetWantsAfterExecution"]
            - 2.2 * metrics["failedExecutionCycles"]
            - 1.6 * metrics["failedConfirmationCycles"]
            - 0.05 * metrics["avgDistanceCompleted"]
        )

    return (
        5.0 * metrics["completedCycles"]
        + 1.3 * metrics["matchingSize"]
        - 2.8 * metrics["unmetWantsAfterExecution"]
        - 1.6 * metrics["failedExecutionCycles"]
        - 1.2 * metrics["failedConfirmationCycles"]
        - 0.08 * metrics["avgDistanceCompleted"]
    )


def _sample_params(base: EngineParams, rng: random.Random) -> Dict[str, Any]:
    sampled = base.to_wire()

    sampled["alpha_patience_expansion"] = _clamp(base.alpha_patience_expansion * rng.uniform(0.7, 1.35), 0.2, 0.95)
    sampled["distance_kernel_alpha"] = _clamp(base.distance_kernel_alpha * rng.uniform(0.6, 1.6), 0.08, 0.5)
    sampled["generosity_beta"] = _clamp(base.generosity_beta * rng.uniform(0.5, 1.8), 0.05, 0.5)
    sampled["choice_penalty"] = _clamp(base.choice_penalty + rng.uniform(-0.08, 0.07), -0.25, 0.05)
    sampled["trust_kappa"] = _clamp(base.trust_kappa * rng.uniform(0.55, 1.45), 0.1, 0.95)
    sampled["high_trust_edge_bonus"] = _clamp(base.high_trust_edge_bonus + rng.uniform(-0.14, 0.2), 0.0, 0.7)
    sampled["high_trust_distance_boost"] = _clamp(base.high_trust_distance_boost + rng.uniform(-0.1, 0.25), 0.0, 0.9)
    sampled["high_trust_cycle_bonus"] = _clamp(base.high_trust_cycle_bonus + rng.uniform(-0.14, 0.24), 0.0, 0.9)
    sampled["high_trust_confirmation_bonus"] = _clamp(base.high_trust_confirmation_bonus + rng.uniform(-0.07, 0.14), 0.0, 0.45)
    sampled["high_trust_execution_bonus"] = _clamp(base.high_trust_execution_bonus + rng.uniform(-0.06, 0.14), 0.0, 0.45)
    sampled["high_trust_completion_threshold"] = _clamp(
        base.high_trust_completion_threshold + rng.uniform(-0.12, 0.08),
        0.55,
        0.95,
    )
    sampled["high_trust_quality_threshold"] = _clamp(
        base.high_trust_quality_threshold + rng.uniform(-0.12, 0.08),
        0.55,
        0.95,
    )

    sampled["execution_base"] = _clamp(base.execution_base + rng.uniform(-0.08, 0.1), 0.35, 0.78)
    sampled["execution_trust_weight"] = _clamp(base.execution_trust_weight + rng.uniform(-0.08, 0.1), 0.12, 0.55)
    sampled["execution_quality_weight"] = _clamp(base.execution_quality_weight + rng.uniform(-0.08, 0.09), 0.08, 0.45)
    sampled["execution_length_penalty"] = _clamp(base.execution_length_penalty + rng.uniform(-0.04, 0.05), 0.02, 0.18)
    sampled["execution_distance_penalty"] = _clamp(base.execution_distance_penalty + rng.uniform(-0.006, 0.007), 0.002, 0.03)

    sampled["context_base_distance"] = {
        context.value: _clamp(value * rng.uniform(0.7, 2.3), 1.0, 60.0)
        for context, value in base.context_base_distance.items()
    }
    sampled["context_upsilon"] = {
        context.value: _clamp(value * rng.uniform(0.7, 1.9), 0.8, 6.0)
        for context, value in base.context_upsilon.items()
    }
    sampled["context_omega"] = {
        context.value: _clamp(value * rng.uniform(0.65, 1.5), 0.05, 0.6)
        for context, value in base.context_omega.items()
    }

    sampled["min_provider_completion"] = {
        context.value: _clamp(value + rng.uniform(-0.09, 0.09), 0.2, 0.85)
        for context, value in base.min_provider_completion.items()
    }
    sampled["min_provider_quality"] = {
        context.value: _clamp(value + rng.uniform(-0.09, 0.09), 0.2, 0.85)
        for context, value in base.min_provider_quality.items()
    }
    sampled["min_receiver_completion"] = {
        context.value: _clamp(value + rng.uniform(-0.08, 0.08), 0.0, 0.75)
        for context, value in base.min_receiver_completion.items()
    }

    return sampled


def optimize_parameters(
    scenario_key: str,
    budget: int = 40,
    seeds_per_candidate: int = 6,
    objective_profile: str = "balanced",
) -> Dict[str, Any]:
    if scenario_key not in SCENARIOS:
        raise ValueError(f"Unknown scenario_key: {scenario_key}")
    if objective_profile not in OBJECTIVE_PROFILES:
        raise ValueError(f"Unknown objective_profile: {objective_profile}")

    budget = int(_clamp(float(budget), 4, 120))
    seeds_per_candidate = int(_clamp(float(seeds_per_candidate), 3, 16))

    rng = random.Random(1337)
    base = DEFAULT_PARAMS
    candidate_overrides: List[Dict[str, Any]] = [base.to_wire()]

    for _ in range(max(budget - 1, 0)):
        candidate_overrides.append(_sample_params(base, rng))

    scored_candidates: List[Dict[str, Any]] = []

    for candidate_index, override in enumerate(candidate_overrides):
        params = base.with_overrides(override)
        objectives: List[float] = []
        completed: List[float] = []
        unmatched: List[float] = []
        failures: List[float] = []

        for seed_offset in range(seeds_per_candidate):
            payload = _run_pipeline(
                SCENARIOS[scenario_key],
                params,
                seed=10_000 + candidate_index * 101 + seed_offset,
            )
            metrics = payload["metrics"]
            objectives.append(_objective(metrics, profile=objective_profile))
            completed.append(float(metrics["completedCycles"]))
            unmatched.append(float(metrics["unmetWantsAfterExecution"]))
            failures.append(float(metrics["failedExecutionCycles"] + metrics["failedConfirmationCycles"]))

        mean_objective = _average(objectives)
        scored_candidates.append(
            {
                "rank": 0,
                "candidateIndex": candidate_index,
                "objectiveMean": mean_objective,
                "objectiveMin": min(objectives),
                "objectiveMax": max(objectives),
                "completedCyclesMean": _average(completed),
                "unmetWantsMean": _average(unmatched),
                "failureMean": _average(failures),
                "params": params.to_wire(),
            }
        )

    scored_candidates.sort(key=lambda candidate: candidate["objectiveMean"], reverse=True)
    for rank, candidate in enumerate(scored_candidates, start=1):
        candidate["rank"] = rank

    best = scored_candidates[0]

    return {
        "scenarioKey": scenario_key,
        "objectiveProfile": objective_profile,
        "budget": budget,
        "seedsPerCandidate": seeds_per_candidate,
        "best": best,
        "topCandidates": scored_candidates[:5],
        "baseline": next(candidate for candidate in scored_candidates if candidate["candidateIndex"] == 0),
    }


# ============================================================
# HTTP SERVER
# ============================================================


class ShareWithHandler(BaseHTTPRequestHandler):
    server_version = "ShareWithPOC/0.2"

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, status: int = 400) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_index(self) -> None:
        index_path = Path(__file__).resolve().parent / "index.html"
        if not index_path.exists():
            self._send_text("Missing index.html in server directory", status=500)
            return

        body = index_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> Dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}

        raw = self.rfile.read(content_length)
        if not raw:
            return {}

        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        path = urlparse(self.path).path

        if path in ("/", "/index.html"):
            self._serve_index()
            return

        if path == "/api/scenarios":
            self._send_json(
                {
                    "scenarios": [
                        {
                            "key": scenario.key,
                            "name": scenario.name,
                            "note": scenario.note,
                        }
                        for scenario in SCENARIOS.values()
                    ]
                }
            )
            return

        if path == "/api/default-params":
            self._send_json({"params": DEFAULT_PARAMS.to_wire()})
            return

        if path == "/healthz":
            self._send_json({"ok": True})
            return

        self._send_text("Not Found", status=404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path

        if path == "/api/run":
            try:
                body = self._read_json_body()
            except json.JSONDecodeError:
                self._send_text("Invalid JSON payload", status=400)
                return

            scenario_key = body.get("scenario_key", "balanced_local")
            seed_input = body.get("seed")
            params_override = body.get("params")

            seed: Optional[int] = None
            if seed_input is not None:
                try:
                    seed = int(seed_input)
                except (TypeError, ValueError):
                    self._send_text("seed must be an integer", status=400)
                    return

            if params_override is not None and not isinstance(params_override, dict):
                self._send_text("params must be an object", status=400)
                return

            try:
                payload = run_scenario(scenario_key, seed, params_override)
            except ValueError as exc:
                self._send_text(str(exc), status=400)
                return

            self._send_json(payload)
            return

        if path == "/api/rank":
            try:
                body = self._read_json_body()
            except json.JSONDecodeError:
                self._send_text("Invalid JSON payload", status=400)
                return

            pains = body.get("pains", [])
            pain_levels = body.get("pain_levels", {})
            if not isinstance(pains, list) or not isinstance(pain_levels, dict):
                self._send_text("Expected pains:list and pain_levels:object", status=400)
                return

            normalized_levels: Dict[str, int] = {}
            for key, value in pain_levels.items():
                try:
                    normalized_levels[str(key)] = int(value)
                except (TypeError, ValueError):
                    continue

            ranked = rank_solutions(pains, normalized_levels)
            self._send_json({"rankedSolutions": ranked})
            return

        if path == "/api/optimize":
            try:
                body = self._read_json_body()
            except json.JSONDecodeError:
                self._send_text("Invalid JSON payload", status=400)
                return

            scenario_key = body.get("scenario_key", "balanced_local")
            budget = body.get("budget", 40)
            seeds = body.get("seeds", 6)
            objective_profile = body.get("objective_profile", "balanced")

            try:
                payload = optimize_parameters(
                    scenario_key=scenario_key,
                    budget=int(budget),
                    seeds_per_candidate=int(seeds),
                    objective_profile=str(objective_profile),
                )
            except ValueError as exc:
                self._send_text(str(exc), status=400)
                return

            self._send_json(payload)
            return

        self._send_text("Not Found", status=404)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}")


# ============================================================
# ENTRYPOINT
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ShareWith cycle trading wireframe server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind (default: 8080)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), ShareWithHandler)
    print(f"ShareWith server running on http://{args.host}:{args.port}")
    print("Endpoints: GET /api/scenarios | GET /api/default-params | POST /api/run | POST /api/rank | POST /api/optimize")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
