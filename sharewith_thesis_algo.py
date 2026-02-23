r"""
# ShareWith Thesis Algorithm (Single-File Canonical Edition)

This file is the canonical one-file implementation of the ShareWith thesis algorithm.
It intentionally keeps core logic in one place for reading, editing, testing, and extension.

## Thesis Math (inline)

Edge weight:
\[
W(e) = Q(e)\cdot\Pi(e) + B_{gen}(e) + C(e) + \omega\log(1 + patience) + B_{high\_trust}
\]

Distance allowance:
\[
D^* = D_{base}(c)\cdot\left(1 + \alpha\log(1 + patience)\cdot\Upsilon(c)\cdot T_{mult}\right)
\]

Parallel tessellation complexity argument:
\[
O(N^3) \rightarrow O(m\cdot (N/m)^3) = O(N^3/m^2)
\]

Cycle atomicity invariant:
- If an agent is in multiple linked cycles, either all linked cycles commit/complete or none do.

Variants:
- C: baseline utilitarian objective
- D: failure-aware (risk-adjusted expected utility)
- E: fairness/time-priority (waiting/disadvantage bonuses)
- F: ant-trail reinforcement (bounded pheromone memory)
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

try:
    from scipy.optimize import linear_sum_assignment as _scipy_linear_sum_assignment
    _SCIPY_AVAILABLE = True
except Exception:
    _scipy_linear_sum_assignment = None
    _SCIPY_AVAILABLE = False


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
    matching_solver: str = "python"

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
            "matching_solver": self.matching_solver,
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


def _scipy_maximize(weights: List[List[float]]) -> List[int]:
    if _scipy_linear_sum_assignment is None:
        return _hungarian_maximize(weights)

    rows = len(weights)
    cols = len(weights[0]) if rows else 0
    if rows == 0 or cols == 0:
        return []

    # linear_sum_assignment minimizes; negate to maximize.
    cost = [[-value for value in row] for row in weights]
    try:
        row_idx, col_idx = _scipy_linear_sum_assignment(cost)
    except Exception:
        # Preserve robustness if scipy backend fails on edge-case matrices.
        return _hungarian_maximize(weights)

    assignment = [-1] * rows
    for r, c in zip(row_idx, col_idx):
        assignment[int(r)] = int(c)
    return assignment


def _maximize_assignment(weights: List[List[float]], solver: str) -> List[int]:
    solver_key = str(solver).strip().lower()
    if solver_key == "scipy" and _SCIPY_AVAILABLE:
        return _scipy_maximize(weights)
    return _hungarian_maximize(weights)


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

    solver_name = str(getattr(params, "matching_solver", "python"))
    assignment = _maximize_assignment(matrix, solver=solver_name)

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

    unique_agents = {edge.provider_id for edge in matching_edges}
    unique_agents.update(edge.receiver_id for edge in matching_edges)
    if max_cycle_len <= 0:
        # "Uncapped" mode: allow cycles up to active graph size.
        effective_max_cycle_len = max(2, len(unique_agents))
    else:
        effective_max_cycle_len = max(2, int(max_cycle_len))

    discovered: List[Dict[str, Any]] = []
    seen_signatures: set[Tuple[int, ...]] = set()

    def dfs(start_agent: str, current_agent: str, path_edge_ids: List[int], visited_agents: set[str]) -> None:
        for edge_idx in edges_by_provider.get(current_agent, []):
            edge = matching_edges[edge_idx]
            nxt = edge.receiver_id

            if nxt == start_agent and path_edge_ids:
                cycle_edge_ids = path_edge_ids + [edge_idx]
                if len(cycle_edge_ids) < 2 or len(cycle_edge_ids) > effective_max_cycle_len:
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

            if nxt in visited_agents or len(path_edge_ids) + 1 >= effective_max_cycle_len:
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
# MONOLITH BRIDGE FOR WAVE/TESSELLATION SECTION
# ============================================================

import csv
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Sequence


class _ServerNamespace:
    """Compatibility namespace so wave functions can call server.* in this same file."""


server = _ServerNamespace()
for _name, _value in list(globals().items()):
    if _name.startswith("__"):
        continue
    setattr(server, _name, _value)


# ============================================================
# WAVE + TESSELLATION + VARIANT RUNNER (inlined from wave_tessellation_fulltilt.py)
# ============================================================

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
    tessellation_topology: str = "hex_voronoi"
    wave_expansion_mode: str = "week_ring"
    context_weights: Optional[Dict[server.Context, float]] = None
    skill_weights: Optional[Dict[server.Context, Dict[str, float]]] = None


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


def _sample_context(rng: random.Random, cfg: SimulationConfig) -> server.Context:
    contexts = (
        server.Context.SAFETY,
        server.Context.SOCIAL,
        server.Context.GROWTH,
        server.Context.SURVIVAL,
        server.Context.LUXURY,
    )
    fallback_weights = (0.33, 0.24, 0.2, 0.16, 0.07)

    if not cfg.context_weights:
        return rng.choices(contexts, weights=fallback_weights)[0]

    weights = [max(0.0, float(cfg.context_weights.get(ctx, 0.0))) for ctx in contexts]
    if sum(weights) <= 0:
        return rng.choices(contexts, weights=fallback_weights)[0]
    return rng.choices(contexts, weights=weights)[0]


def _sample_skill(rng: random.Random, cfg: SimulationConfig, context: server.Context) -> str:
    pool = SKILL_POOLS[context]
    if not cfg.skill_weights or context not in cfg.skill_weights:
        return rng.choice(pool)

    mapping = cfg.skill_weights[context]
    weights = [max(0.0, float(mapping.get(skill, 0.0))) for skill in pool]
    if sum(weights) <= 0:
        return rng.choice(pool)
    return str(rng.choices(pool, weights=weights, k=1)[0])


def _sample_distinct_skill(
    rng: random.Random,
    cfg: SimulationConfig,
    context: server.Context,
    *,
    avoid: str,
) -> str:
    pool = SKILL_POOLS[context]
    if len(pool) <= 1:
        return avoid
    for _ in range(6):
        choice = _sample_skill(rng, cfg, context)
        if choice != avoid:
            return choice
    alternatives = [skill for skill in pool if skill != avoid]
    return str(rng.choice(alternatives))


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
        want_skill = _sample_skill(rng=rng, cfg=cfg, context=want_context)
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
    anchor_skill = _sample_skill(rng=rng, cfg=cfg, context=anchor_context)
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
            offer_skill = _sample_skill(rng=rng, cfg=cfg, context=offer_context)
            supply_tag = ""
        offers.append(_offer_from_skill(context=offer_context, skill=offer_skill, supply_tag=supply_tag))

    return offers


def _build_agents(cfg: SimulationConfig) -> Tuple[List[server.Agent], Dict[str, Tuple[int, int]], List[Tuple[int, int]]]:
    rng = random.Random(cfg.seed)
    agents: List[server.Agent] = []
    home_cell: Dict[str, Tuple[int, int]] = {}
    all_cells: List[Tuple[int, int]] = []

    aid = 0
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
                context = _sample_context(rng=rng, cfg=cfg)
                if baseline_declarations:
                    offer_skill = _sample_skill(rng=rng, cfg=cfg, context=context)
                    want_skill = _sample_distinct_skill(rng=rng, cfg=cfg, context=context, avoid=offer_skill)
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


def _hop_distance(
    a: Tuple[int, int],
    b: Tuple[int, int],
    *,
    topology: str = "hex_voronoi",
) -> int:
    # Hex axial distance approximates Voronoi-neighbor ring expansion.
    if topology == "hex_voronoi":
        dq = a[0] - b[0]
        dr = a[1] - b[1]
        return int((abs(dq) + abs(dq + dr) + abs(dr)) / 2)
    if topology == "grid_queen":
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
    raise ValueError(f"Unknown tessellation topology: {topology}")


def _cells_within(
    cell: Tuple[int, int],
    hop: int,
    cfg: CellConfig,
    *,
    topology: str = "hex_voronoi",
) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    x0 = max(0, cell[0] - hop)
    x1 = min(cfg.width - 1, cell[0] + hop)
    y0 = max(0, cell[1] - hop)
    y1 = min(cfg.height - 1, cell[1] + hop)

    for nx in range(x0, x1 + 1):
        for ny in range(y0, y1 + 1):
            if _hop_distance((nx, ny), cell, topology=topology) <= hop:
                out.append((nx, ny))
    return out


def _reach_hops(
    patience: int,
    max_hop: int,
    patience_per_hop: int,
    *,
    week_index: int = 1,
    wave_expansion_mode: str = "week_ring",
) -> int:
    week_hop = max(0, int(week_index) - 1)
    if patience_per_hop <= 0:
        patience_hop = max_hop
    else:
        patience_hop = int(_clamp(float(patience // patience_per_hop), 0.0, float(max_hop)))

    if wave_expansion_mode == "week_ring":
        return int(_clamp(float(week_hop), 0.0, float(max_hop)))
    if wave_expansion_mode == "patience":
        return patience_hop
    if wave_expansion_mode == "week_plus_patience":
        return int(_clamp(float(max(week_hop, patience_hop)), 0.0, float(max_hop)))
    raise ValueError(f"Unknown wave_expansion_mode: {wave_expansion_mode}")


def _build_edges_wave(
    agents: List[server.Agent],
    params: server.EngineParams,
    home_cell: Dict[str, Tuple[int, int]],
    reach_by_agent: Dict[str, int],
    grid: CellConfig,
    topology: str = "hex_voronoi",
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

    cell_cache: Dict[Tuple[Tuple[int, int], int, str], List[Tuple[int, int]]] = {}
    potential_edges: List[server.Edge] = []
    feasible_edges: List[server.Edge] = []

    for want_node in want_nodes:
        receiver = agents_by_id[want_node.agent_id]
        receiver_cell = home_cell[receiver.agent_id]
        receiver_hop = reach_by_agent[receiver.agent_id]
        want_context = want_node.want.context
        key = (receiver_cell, receiver_hop, topology)
        if key not in cell_cache:
            cell_cache[key] = _cells_within(receiver_cell, receiver_hop, cfg=grid, topology=topology)
        for cell in cell_cache[key]:
            for offer_node in offers_by_cell_ctx.get((cell[0], cell[1], want_context), []):
                if offer_node.agent_id == want_node.agent_id:
                    continue
                provider = agents_by_id[offer_node.agent_id]
                # Receiver-centric wave filter: receiver can only "see" providers within hop radius.
                if _hop_distance(home_cell[provider.agent_id], receiver_cell, topology=topology) > receiver_hop:
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
    week_index: int,
    wave_expansion_mode: str,
    topology: str,
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
            week_index=week_index,
            wave_expansion_mode=wave_expansion_mode,
        )
        for agent in active_agents
    }

    potential_edges, feasible_edges, offer_nodes, want_nodes = _build_edges_wave(
        agents=active_agents,
        params=params,
        home_cell=home_cell,
        reach_by_agent=reach_by_agent,
        grid=grid,
        topology=topology,
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
        hop = _hop_distance(home_cell[edge.provider_id], home_cell[edge.receiver_id], topology=topology)
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
    week_index: int = 1,
    wave_expansion_mode: str = "week_ring",
    topology: str = "hex_voronoi",
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
            week_index=week_index,
            wave_expansion_mode=wave_expansion_mode,
            topology=topology,
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
            week_index=week_index,
            wave_expansion_mode=wave_expansion_mode,
        )
        for agent in active_agents
    }

    potential_edges, feasible_edges, offer_nodes, want_nodes = _build_edges_wave(
        agents=active_agents,
        params=params,
        home_cell=home_cell,
        reach_by_agent=reach_by_agent,
        grid=grid,
        topology=topology,
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
    *,
    topology: str = "hex_voronoi",
) -> Dict[str, float]:
    matching_cross = 0
    matching_hops: List[int] = []
    for edge in matching:
        hop = _hop_distance(home_cell[edge.provider_id], home_cell[edge.receiver_id], topology=topology)
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
        hop = _hop_distance(home_cell[edge.provider_id], home_cell[edge.receiver_id], topology=topology)
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
    report.append(
        f"- Matching solver (requested -> resolved): "
        f"`{summary.get('solver_requested', 'python')} -> {summary.get('solver_resolved', 'python')}`"
    )
    report.append(f"- SciPy available: `{summary.get('scipy_available', False)}`")
    report.append(f"- Matching mode: `{cfg.matching_mode}`")
    report.append(f"- Bridge hop: `{cfg.bridge_hop}`")
    report.append(f"- Fallback rounds: `{cfg.fallback_rounds}`")
    report.append(f"- Tessellation topology: `{cfg.tessellation_topology}`")
    report.append(f"- Wave expansion mode: `{cfg.wave_expansion_mode}`")
    report.append(f"- Spice profile: `{cfg.spice_profile}`")
    report.append(f"- Max offers per agent: `{cfg.max_offers_per_agent}`")
    report.append(f"- Max wants per agent: `{cfg.max_wants_per_agent}`")
    report.append(f"- Need decomposition rate: `{cfg.decomposition_rate:.3f}`")
    report.append(f"- Cross-context want rate: `{cfg.cross_context_want_rate:.3f}`")
    report.append(f"- Max cycle length setting: `{summary.get('max_cycle_length_setting', 'n/a')}`")
    report.append(f"- Unbounded cycle length mode: `{summary.get('unbounded_cycle_length_mode', False)}`")
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
                "max_cycle_length_observed",
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


_RUN_CONFIG_ALLOWED_KEYS = {
    "grid",
    "agents_per_cell",
    "weeks",
    "active_rate",
    "max_hop",
    "patience_per_hop",
    "matching_mode",
    "bridge_hop",
    "fallback_rounds",
    "tessellation_topology",
    "wave_expansion_mode",
    "panel_mode",
    "variant",
    "spice_profile",
    "max_offers_per_agent",
    "max_wants_per_agent",
    "decomposition_rate",
    "cross_context_want_rate",
    "solver",
    "max_cycle_length",
    "params_overrides_file",
    "sparkov_dataset_dir",
    "sparkov_download",
    "sparkov_max_rows",
    "sparkov_category_map_file",
    "sparkov_profile_out",
    "sparkov_apply_recommendations",
    "seed",
    "outdir",
}
_RUN_CONFIG_PATH_KEYS = {
    "params_overrides_file",
    "outdir",
    "sparkov_dataset_dir",
    "sparkov_category_map_file",
    "sparkov_profile_out",
}

SPARKOV_DEFAULT_HANDLE = "kartik2112/fraud-detection"
SPARKOV_DEFAULT_CATEGORY_MAP_FILE = ROOT / "run_presets" / "sparkov_category_skill_map.json"


def _normalize_weight_map(raw: Dict[str, float], keys: Sequence[str]) -> Dict[str, float]:
    prepared = {key: max(0.0, float(raw.get(key, 0.0))) for key in keys}
    total = sum(prepared.values())
    if total <= 0:
        uniform = 1.0 / max(len(keys), 1)
        return {key: uniform for key in keys}
    return {key: value / total for key, value in prepared.items()}


def _load_sparkov_category_map(map_file: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    resolved = (map_file or SPARKOV_DEFAULT_CATEGORY_MAP_FILE).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Sparkov category map file not found: {resolved}")

    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Sparkov category map must be a JSON object.")

    normalized: Dict[str, Dict[str, Any]] = {}
    for category, cfg in payload.items():
        if not isinstance(cfg, dict):
            continue
        context = _context_from_value(cfg.get("context"))
        if context is None:
            continue
        skills_raw = cfg.get("skills", {})
        if not isinstance(skills_raw, dict):
            continue
        valid_skills = {skill: float(weight) for skill, weight in skills_raw.items() if skill in SKILL_POOLS[context]}
        if not valid_skills:
            continue
        normalized[category] = {
            "context": context.value,
            "skills": _normalize_weight_map(valid_skills, SKILL_POOLS[context]),
        }

    if not normalized:
        raise ValueError(f"Sparkov category map produced no valid category mappings: {resolved}")
    return normalized


def _download_sparkov_dataset() -> Path:
    try:
        import kagglehub  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "kagglehub is required for --sparkov-download. Install with `pip install kagglehub`."
        ) from exc

    dataset_path = kagglehub.dataset_download(SPARKOV_DEFAULT_HANDLE)
    return Path(str(dataset_path)).expanduser().resolve()


def _resolve_sparkov_dataset_dir(args: argparse.Namespace) -> Optional[Path]:
    if args.sparkov_dataset_dir is not None:
        return args.sparkov_dataset_dir.expanduser().resolve()
    if args.sparkov_download:
        return _download_sparkov_dataset()
    return None


def _derive_sparkov_profile(
    dataset_dir: Path,
    category_map: Dict[str, Dict[str, Any]],
    *,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    csv_files: List[Path] = []
    for filename in ("fraudTrain.csv", "fraudTest.csv"):
        path = dataset_dir / filename
        if path.exists():
            csv_files.append(path)
    if not csv_files:
        raise FileNotFoundError(f"No Sparkov CSV files found under {dataset_dir}. Expected fraudTrain.csv/fraudTest.csv")

    category_counts: Counter[str] = Counter()
    context_counts: Counter[str] = Counter()
    skill_scores: Dict[str, Counter[str]] = {ctx.value: Counter() for ctx in server.Context}

    mapped_rows = 0
    rows = 0
    fraud_rows = 0
    amount_sum = 0.0
    min_unix: Optional[int] = None
    max_unix: Optional[int] = None

    unique_cards: set[str] = set()
    unique_merchants: set[str] = set()
    card_categories: Dict[str, set[str]] = defaultdict(set)
    card_contexts: Dict[str, set[str]] = defaultdict(set)

    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if max_rows is not None and rows >= max_rows:
                    break
                category = str(row.get("category", "")).strip()
                if not category:
                    continue

                rows += 1
                category_counts[category] += 1

                try:
                    amount_sum += float(row.get("amt", 0.0) or 0.0)
                except ValueError:
                    pass

                try:
                    is_fraud = int(float(row.get("is_fraud", 0) or 0))
                    fraud_rows += 1 if is_fraud > 0 else 0
                except ValueError:
                    pass

                try:
                    unix_time = int(float(row.get("unix_time", 0) or 0))
                except ValueError:
                    unix_time = 0
                if unix_time > 0:
                    min_unix = unix_time if min_unix is None else min(min_unix, unix_time)
                    max_unix = unix_time if max_unix is None else max(max_unix, unix_time)

                card = str(row.get("cc_num", "")).strip()
                if card:
                    unique_cards.add(card)
                    card_categories[card].add(category)

                merchant = str(row.get("merchant", "")).strip()
                if merchant:
                    unique_merchants.add(merchant)

                mapping = category_map.get(category)
                if mapping is None:
                    continue

                mapped_rows += 1
                context_name = mapping["context"]
                context_counts[context_name] += 1
                if card:
                    card_contexts[card].add(context_name)

                for skill, weight in mapping["skills"].items():
                    skill_scores[context_name][skill] += float(weight)

            if max_rows is not None and rows >= max_rows:
                break

    if rows <= 0:
        raise ValueError(f"Sparkov profile derivation found zero rows in {dataset_dir}")

    default_context = {
        server.Context.SAFETY.value: 0.33,
        server.Context.SOCIAL.value: 0.24,
        server.Context.GROWTH.value: 0.2,
        server.Context.SURVIVAL.value: 0.16,
        server.Context.LUXURY.value: 0.07,
    }
    context_weights = _normalize_weight_map(
        {ctx.value: float(context_counts.get(ctx.value, 0.0)) for ctx in server.Context},
        [ctx.value for ctx in server.Context],
    )
    if mapped_rows <= 0:
        context_weights = default_context

    normalized_skill_weights: Dict[str, Dict[str, float]] = {}
    for context in server.Context:
        pool = SKILL_POOLS[context]
        raw_scores = skill_scores[context.value]
        if not raw_scores:
            normalized_skill_weights[context.value] = _normalize_weight_map({}, pool)
            continue
        normalized_skill_weights[context.value] = _normalize_weight_map(
            {skill: float(raw_scores.get(skill, 0.0)) for skill in pool},
            pool,
        )

    cards_count = max(1, len(unique_cards))
    categories_per_card = [len(v) for v in card_categories.values()] or [1]
    contexts_per_card = [len(v) for v in card_contexts.values()] or [1]

    mean_categories_per_card = _mean(float(v) for v in categories_per_card)
    mean_contexts_per_card = _mean(float(v) for v in contexts_per_card)
    share_multi_category_cards = sum(1 for v in categories_per_card if v > 1) / len(categories_per_card)

    if min_unix is not None and max_unix is not None:
        days_span = max(1.0, float(max_unix - min_unix) / 86_400.0)
    else:
        days_span = 1.0

    tx_per_card_per_day = float(rows) / float(cards_count) / days_span
    recommended_max_wants = int(_clamp(round(mean_categories_per_card / 3.2), 1.0, 5.0))
    recommended_max_offers = int(_clamp(round(mean_contexts_per_card / 1.7), 1.0, 5.0))
    recommended_decomposition = _clamp((mean_categories_per_card - 1.0) / 10.0, 0.05, 0.75)
    recommended_cross_context = _clamp((mean_contexts_per_card - 1.0) / 6.0, 0.02, 0.45)
    recommended_active_rate = _clamp(tx_per_card_per_day / 3.0, 0.01, 0.3)

    return {
        "dataset_dir": str(dataset_dir),
        "files": [path.name for path in csv_files],
        "rows_profiled": rows,
        "rows_mapped": mapped_rows,
        "mapping_coverage": float(mapped_rows) / float(rows),
        "unique_cards": len(unique_cards),
        "unique_merchants": len(unique_merchants),
        "fraud_rate": float(fraud_rows) / float(rows),
        "mean_amount": amount_sum / float(rows),
        "days_span": days_span,
        "tx_per_card_per_day": tx_per_card_per_day,
        "mean_categories_per_card": mean_categories_per_card,
        "mean_contexts_per_card": mean_contexts_per_card,
        "share_multi_category_cards": share_multi_category_cards,
        "category_counts": dict(sorted(category_counts.items(), key=lambda kv: kv[1], reverse=True)),
        "context_counts": dict(context_counts),
        "context_weights": context_weights,
        "skill_weights": normalized_skill_weights,
        "recommended": {
            "max_offers_per_agent": recommended_max_offers,
            "max_wants_per_agent": recommended_max_wants,
            "decomposition_rate": recommended_decomposition,
            "cross_context_want_rate": recommended_cross_context,
            "active_rate_hint": recommended_active_rate,
        },
    }


def _sparkov_cfg_context_weights(profile: Dict[str, Any]) -> Dict[server.Context, float]:
    raw = profile.get("context_weights", {})
    return {ctx: float(raw.get(ctx.value, 0.0)) for ctx in server.Context}


def _sparkov_cfg_skill_weights(profile: Dict[str, Any]) -> Dict[server.Context, Dict[str, float]]:
    raw = profile.get("skill_weights", {})
    out: Dict[server.Context, Dict[str, float]] = {}
    for ctx in server.Context:
        ctx_raw = raw.get(ctx.value, {})
        out[ctx] = {skill: float(ctx_raw.get(skill, 0.0)) for skill in SKILL_POOLS[ctx]}
    return out


def _apply_sparkov_recommendations(args: argparse.Namespace, profile: Dict[str, Any]) -> None:
    rec = profile.get("recommended", {})
    if args.max_offers_per_agent is None:
        args.max_offers_per_agent = int(rec.get("max_offers_per_agent", 1))
    if args.max_wants_per_agent is None:
        args.max_wants_per_agent = int(rec.get("max_wants_per_agent", 1))
    if args.decomposition_rate is None:
        args.decomposition_rate = float(rec.get("decomposition_rate", 0.0))
    if args.cross_context_want_rate is None:
        args.cross_context_want_rate = float(rec.get("cross_context_want_rate", 0.0))


def _normalize_run_config_key(raw_key: str) -> str:
    return raw_key.strip().replace("-", "_")


def _load_run_config(config_file: Path) -> Dict[str, Any]:
    payload = json.loads(config_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config-file must contain a JSON object.")

    if "run" in payload:
        run_section = payload["run"]
        if not isinstance(run_section, dict):
            raise ValueError("config-file field 'run' must be a JSON object.")
        payload = run_section

    normalized: Dict[str, Any] = {}
    for key, value in payload.items():
        normalized[_normalize_run_config_key(key)] = value

    unknown_keys = sorted(key for key in normalized if key not in _RUN_CONFIG_ALLOWED_KEYS)
    if unknown_keys:
        allowed_keys = ", ".join(sorted(_RUN_CONFIG_ALLOWED_KEYS))
        raise ValueError(f"Unknown config-file keys: {unknown_keys}. Allowed keys: {allowed_keys}")

    for key in _RUN_CONFIG_PATH_KEYS:
        if key not in normalized:
            continue
        value = normalized[key]
        if value in (None, ""):
            normalized[key] = None
            continue
        path_value = Path(str(value)).expanduser()
        if not path_value.is_absolute():
            path_value = (config_file.parent / path_value).resolve()
        normalized[key] = path_value

    return normalized


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config-file", type=Path, default=None, help=argparse.SUPPRESS)
    bootstrap_args, _ = bootstrap.parse_known_args(argv)

    config_defaults: Dict[str, Any] = {}
    resolved_config_file: Optional[Path] = None
    if bootstrap_args.config_file is not None:
        resolved_config_file = bootstrap_args.config_file.expanduser().resolve()
        config_defaults = _load_run_config(resolved_config_file)

    parser = argparse.ArgumentParser(description="Run large-cohort wave-style tessellation simulation.")
    parser.add_argument(
        "--config-file",
        type=Path,
        default=resolved_config_file,
        help="Optional JSON run preset. Any explicit CLI flag overrides file values.",
    )
    parser.add_argument("--grid", default="3x3", help="Grid dimensions, e.g. 3x3")
    parser.add_argument("--agents-per-cell", type=int, default=17000)
    parser.add_argument("--weeks", type=int, default=6)
    parser.add_argument("--active-rate", type=float, default=0.02)
    parser.add_argument("--max-hop", type=int, default=4)
    parser.add_argument("--patience-per-hop", type=int, default=1)
    parser.add_argument(
        "--solver",
        choices=("python", "scipy"),
        default="python",
        help="Assignment solver backend: python (built-in Hungarian) or scipy (linear_sum_assignment).",
    )
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
        "--tessellation-topology",
        choices=("hex_voronoi", "grid_queen"),
        default="hex_voronoi",
        help="hex_voronoi uses 6-neighbor ring hops (Voronoi-like); grid_queen keeps legacy square/diagonal hops.",
    )
    parser.add_argument(
        "--wave-expansion-mode",
        choices=("week_ring", "week_plus_patience", "patience"),
        default="week_ring",
        help="week_ring: week1=home, week2=ring1, ...; week_plus_patience: max(week ring, patience); patience: legacy patience-only.",
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
        "--max-cycle-length",
        type=int,
        default=None,
        help="Maximum agents per cycle. Use 0 for uncapped mode (bounded only by active graph size).",
    )
    parser.add_argument(
        "--params-overrides-file",
        type=Path,
        default=None,
        help="Optional JSON file containing EngineParams overrides (or {\"params_overrides\": {...}}).",
    )
    parser.add_argument(
        "--sparkov-dataset-dir",
        type=Path,
        default=None,
        help="Path containing Sparkov CSVs (fraudTrain.csv / fraudTest.csv).",
    )
    parser.add_argument(
        "--sparkov-download",
        action="store_true",
        help=f"Download Sparkov dataset from Kaggle handle '{SPARKOV_DEFAULT_HANDLE}' via kagglehub.",
    )
    parser.add_argument(
        "--sparkov-max-rows",
        type=int,
        default=None,
        help="Optional row cap for Sparkov profiling (for quick tests).",
    )
    parser.add_argument(
        "--sparkov-category-map-file",
        type=Path,
        default=SPARKOV_DEFAULT_CATEGORY_MAP_FILE,
        help="JSON file mapping Sparkov categories to ShareWith contexts/skills.",
    )
    parser.add_argument(
        "--sparkov-profile-out",
        type=Path,
        default=None,
        help="Optional output path for derived Sparkov calibration profile JSON.",
    )
    parser.add_argument(
        "--sparkov-apply-recommendations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply Sparkov-derived recommended declaration settings when explicit CLI values are absent.",
    )
    parser.add_argument("--seed", type=int, default=20260218)
    parser.add_argument("--outdir", type=Path, default=ROOT / "artifacts" / "wave_fulltilt")
    if config_defaults:
        parser.set_defaults(**config_defaults)
    args = parser.parse_args(argv)
    if args.config_file is not None:
        args.config_file = args.config_file.expanduser().resolve()
    if isinstance(args.params_overrides_file, str):
        args.params_overrides_file = Path(args.params_overrides_file)
    if isinstance(args.sparkov_dataset_dir, str):
        args.sparkov_dataset_dir = Path(args.sparkov_dataset_dir)
    if isinstance(args.sparkov_category_map_file, str):
        args.sparkov_category_map_file = Path(args.sparkov_category_map_file)
    if isinstance(args.sparkov_profile_out, str):
        args.sparkov_profile_out = Path(args.sparkov_profile_out)
    if isinstance(args.outdir, str):
        args.outdir = Path(args.outdir)
    return args


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
    if args.config_file is not None:
        print(f"Run preset: {args.config_file}")

    sparkov_profile: Optional[Dict[str, Any]] = None
    sparkov_dataset_dir = _resolve_sparkov_dataset_dir(args)
    if sparkov_dataset_dir is not None:
        print(f"Sparkov dataset: {sparkov_dataset_dir}")
        category_map = _load_sparkov_category_map(args.sparkov_category_map_file)
        sparkov_profile = _derive_sparkov_profile(
            sparkov_dataset_dir,
            category_map,
            max_rows=args.sparkov_max_rows,
        )
        if args.sparkov_apply_recommendations:
            _apply_sparkov_recommendations(args, sparkov_profile)
            rec = sparkov_profile["recommended"]
            print(
                "Applied Sparkov recommendations: "
                f"offers={rec['max_offers_per_agent']} wants={rec['max_wants_per_agent']} "
                f"decomp={rec['decomposition_rate']:.3f} cross={rec['cross_context_want_rate']:.3f}"
            )
        sparkov_profile_out = args.sparkov_profile_out
        if sparkov_profile_out is not None:
            sparkov_profile_out = sparkov_profile_out.expanduser().resolve()
            sparkov_profile_out.parent.mkdir(parents=True, exist_ok=True)
            sparkov_profile_out.write_text(json.dumps(sparkov_profile, indent=2), encoding="utf-8")
            print(f"Sparkov profile saved: {sparkov_profile_out}")

    max_offers, max_wants, decomposition_rate, cross_context_rate = _resolve_spice_parameters(args)
    sparkov_context_weights_cfg = _sparkov_cfg_context_weights(sparkov_profile) if sparkov_profile else None
    sparkov_skill_weights_cfg = _sparkov_cfg_skill_weights(sparkov_profile) if sparkov_profile else None
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
        tessellation_topology=str(args.tessellation_topology),
        wave_expansion_mode=str(args.wave_expansion_mode),
        context_weights=sparkov_context_weights_cfg,
        skill_weights=sparkov_skill_weights_cfg,
    )
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    solver_requested = str(args.solver).strip().lower()
    solver_resolved = solver_requested
    if solver_requested == "scipy" and not _SCIPY_AVAILABLE:
        print("solver=scipy requested but scipy is unavailable; falling back to python.")
        solver_resolved = "python"

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
    print(f"Matching solver: requested={solver_requested} resolved={solver_resolved} scipy_available={_SCIPY_AVAILABLE}")

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
    if args.max_cycle_length is not None:
        params = params.with_overrides({"max_cycle_length": int(args.max_cycle_length)})
    params = params.with_overrides({"matching_solver": solver_resolved})
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
            week_index=week,
            wave_expansion_mode=cfg.wave_expansion_mode,
            topology=cfg.tessellation_topology,
            variant=cfg.variant,
            trail_memory=trail_memory if cfg.variant == "F" else None,
            matching_mode=cfg.matching_mode,
            bridge_hop=cfg.bridge_hop,
            fallback_rounds=cfg.fallback_rounds,
        )
        if cfg.variant == "F":
            _update_trail_memory(trail_memory=trail_memory, cycles=cycles, home_cell=home_cell)
        _update_agents_from_projection(active_agents, projection)
        wave_stats = _cross_cell_stats(
            matching=matching,
            cycles=cycles,
            home_cell=home_cell,
            topology=cfg.tessellation_topology,
        )
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
            "max_cycle_length_observed": metrics["maxCycleLength"],
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
            f"max_cycle={row['max_cycle_length_observed']} "
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
        "solver_requested": solver_requested,
        "solver_resolved": solver_resolved,
        "scipy_available": _SCIPY_AVAILABLE,
        "matching_mode": cfg.matching_mode,
        "bridge_hop": cfg.bridge_hop,
        "fallback_rounds": cfg.fallback_rounds,
        "max_cycle_length_setting": int(params.max_cycle_length),
        "unbounded_cycle_length_mode": bool(params.max_cycle_length <= 0),
        "tessellation_topology": cfg.tessellation_topology,
        "wave_expansion_mode": cfg.wave_expansion_mode,
        "spice_profile": cfg.spice_profile,
        "config_file": str(args.config_file) if args.config_file else "",
        "params_overrides_file": str(args.params_overrides_file.resolve()) if args.params_overrides_file else "",
        "sparkov_dataset_dir": str(sparkov_dataset_dir) if sparkov_dataset_dir else "",
        "sparkov_rows_profiled": int(sparkov_profile["rows_profiled"]) if sparkov_profile else 0,
        "sparkov_mapping_coverage": float(sparkov_profile["mapping_coverage"]) if sparkov_profile else 0.0,
        "sparkov_fraud_rate": float(sparkov_profile["fraud_rate"]) if sparkov_profile else 0.0,
        "sparkov_mean_categories_per_card": float(sparkov_profile["mean_categories_per_card"]) if sparkov_profile else 0.0,
        "sparkov_mean_contexts_per_card": float(sparkov_profile["mean_contexts_per_card"]) if sparkov_profile else 0.0,
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
        "max_cycle_length_observed": max(int(row["max_cycle_length_observed"]) for row in weekly_rows) if weekly_rows else 0,
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
