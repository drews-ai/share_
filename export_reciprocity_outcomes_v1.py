#!/usr/bin/env python3
"""Export ShareWith run artifacts into reciprocity_outcomes.v1 contract."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _validate(payload: Dict[str, Any], schema_path: Path | None) -> None:
    if schema_path is None:
        return
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    try:
        import jsonschema  # type: ignore

        jsonschema.validate(instance=payload, schema=schema)
    except ImportError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ShareWith artifacts to reciprocity_outcomes.v1")
    parser.add_argument("--summary", type=Path, required=True, help="Path to summary.json")
    parser.add_argument("--weekly", type=Path, required=True, help="Path to weekly_metrics.csv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--region", default="unspecified")
    parser.add_argument("--policy-id", default="")
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("/Users/drewprescott/Desktop/share_labor_food/aproximal/schemas/reciprocity_outcomes.v1.schema.json"),
    )
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("summary payload must be an object")

    weekly_rows: List[Dict[str, Any]] = []
    with args.weekly.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            weekly_rows.append(row)

    weekly = [
        {
            "week": _to_int(row.get("week"), 0),
            "active_agents": _to_int(row.get("active_agents"), 0),
            "completed_cycles": _to_float(row.get("completed_cycles"), 0.0),
            "unmet_ratio": _to_float(row.get("unmet_ratio"), 0.0),
            "cycle_survival": _to_float(row.get("cycle_survival"), 0.0),
            "week_runtime_s": _to_float(row.get("week_runtime_s"), 0.0),
            "matching_cross_share": _to_float(row.get("matching_cross_share"), 0.0),
            "completed_cross_share": _to_float(row.get("completed_cross_share"), 0.0),
            "local_cycle_count": _to_float(row.get("local_cycle_count"), 0.0),
            "bridge_cycle_count": _to_float(row.get("bridge_cycle_count"), 0.0),
            "fallback_cycle_count": _to_float(row.get("fallback_cycle_count"), 0.0),
        }
        for row in weekly_rows
    ]
    completed_cycles_total = sum(_to_float(row.get("completed_cycles"), 0.0) for row in weekly_rows)
    mean_unmet_ratio = (
        sum(_to_float(row.get("unmet_ratio"), 0.0) for row in weekly_rows) / max(len(weekly_rows), 1)
    )
    mean_cycle_survival = (
        sum(_to_float(row.get("cycle_survival"), 0.0) for row in weekly_rows) / max(len(weekly_rows), 1)
    )

    run_id = f"sharewith-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    payload: Dict[str, Any] = {
        "schema_version": "reciprocity_outcomes.v1",
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "region": str(args.region),
        "period": {
            "weeks": _to_int(summary.get("weeks"), len(weekly)),
            "start_week": 1,
            "end_week": len(weekly),
        },
        "metrics": {
            "completed_cycles": completed_cycles_total,
            "unmet_ratio": _to_float(summary.get("mean_unmet_ratio"), mean_unmet_ratio),
            "cycle_survival": mean_cycle_survival,
            "mean_week_runtime_s": _to_float(summary.get("mean_week_runtime_s"), 0.0),
            "mean_completed_per_1000_active": _to_float(summary.get("mean_completed_per_1000_active"), 0.0),
            "mean_matching_cross_share": _to_float(summary.get("mean_matching_cross_share"), 0.0),
            "mean_completed_cross_share": _to_float(summary.get("mean_completed_cross_share"), 0.0),
            "mean_local_cycle_count": _to_float(summary.get("mean_local_cycle_count"), 0.0),
            "mean_bridge_cycle_count": _to_float(summary.get("mean_bridge_cycle_count"), 0.0),
            "mean_fallback_cycle_count": _to_float(summary.get("mean_fallback_cycle_count"), 0.0),
        },
        "weekly": weekly,
        "intervention_results": [],
    }

    if args.policy_id:
        payload["policy_id"] = str(args.policy_id)

    _validate(payload, None if args.skip_validation else args.schema)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote reciprocity outcomes: {args.out}")


if __name__ == "__main__":
    main()
