#!/usr/bin/env python3
"""
Translate Aproximal transition_policy.v1 into ShareWith parameter overrides.

This adapter is contract-only and keeps ShareWith decoupled from Aproximal internals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import server


CONTEXTS = [
    server.Context.SURVIVAL,
    server.Context.SAFETY,
    server.Context.SOCIAL,
    server.Context.GROWTH,
    server.Context.LUXURY,
]


def _validate_policy(policy: Dict[str, Any], schema_path: Path | None) -> None:
    if schema_path is None:
        return
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    try:
        import jsonschema  # type: ignore

        jsonschema.validate(instance=policy, schema=schema)
    except ImportError:
        pass


def build_overrides(policy: Dict[str, Any]) -> Dict[str, Any]:
    if policy.get("schema_version") != "transition_policy.v1":
        raise ValueError("Expected transition_policy.v1 payload")

    bias = policy.get("matching_bias") or {}
    context_weights = bias.get("context_weights") or {}
    patience_multiplier = float(bias.get("patience_multiplier") or 1.0)
    trust_floor = float(bias.get("trust_floor") or 0.0)
    max_hop_override = bias.get("max_hop_override")

    defaults = server.DEFAULT_PARAMS
    overrides: Dict[str, Any] = {
        "context_omega": {},
        "context_base_distance": {},
        "min_provider_completion": {},
        "min_provider_quality": {},
        "min_receiver_completion": {},
    }

    for context in CONTEXTS:
        key = context.value
        weight = float(context_weights.get(key, 1.0))
        weight = max(0.5, min(2.5, weight))

        overrides["context_omega"][key] = round(defaults.context_omega[context] * weight, 6)

        # If Aproximal says a context has high vortex pressure, increase early-distance flexibility.
        distance_scale = 1.0 + max(0.0, weight - 1.0) * 0.55
        overrides["context_base_distance"][key] = round(defaults.context_base_distance[context] * distance_scale, 6)

        provider_floor = max(defaults.min_provider_completion[context], max(0.0, min(1.0, trust_floor - 0.08)))
        quality_floor = max(defaults.min_provider_quality[context], max(0.0, min(1.0, trust_floor - 0.06)))
        receiver_floor = max(defaults.min_receiver_completion[context], max(0.0, min(1.0, trust_floor - 0.2)))

        overrides["min_provider_completion"][key] = round(provider_floor, 6)
        overrides["min_provider_quality"][key] = round(quality_floor, 6)
        overrides["min_receiver_completion"][key] = round(receiver_floor, 6)

    overrides["alpha_patience_expansion"] = round(
        defaults.alpha_patience_expansion * max(0.5, min(3.0, patience_multiplier)),
        6,
    )

    payload = {
        "schema_version": "sharewith_params_override.v1",
        "source_policy_id": str(policy.get("policy_id") or ""),
        "generated_from": "transition_policy.v1",
        "params_overrides": overrides,
        "wave_hints": {
            "max_hop": int(max_hop_override) if isinstance(max_hop_override, int) else None,
            "note": "Apply via wave_tessellation_fulltilt.py --params-overrides-file",
        },
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ShareWith parameter overrides from Aproximal policy")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--policy-schema",
        type=Path,
        default=Path("/Users/drewprescott/Desktop/share_labor_food/aproximal/schemas/transition_policy.v1.schema.json"),
    )
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    if not isinstance(policy, dict):
        raise ValueError("Policy payload must be a JSON object")

    _validate_policy(policy, None if args.skip_validation else args.policy_schema)
    payload = build_overrides(policy)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote ShareWith overrides: {args.out}")


if __name__ == "__main__":
    main()
