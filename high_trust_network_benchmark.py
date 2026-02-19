#!/usr/bin/env python3
"""Targeted optimization and evaluation for high-trust network scenarios."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import server  # noqa: E402


def _mean(values: Iterable[float]) -> float:
    data = list(values)
    return statistics.fmean(data) if data else 0.0


def _stdev(values: Iterable[float]) -> float:
    data = list(values)
    if len(data) < 2:
        return 0.0
    return statistics.pstdev(data)


def _evaluate_config(
    scenario_key: str,
    seeds: int,
    params_override: Dict[str, Any] | None,
) -> Dict[str, float]:
    balanced_objectives: List[float] = []
    high_trust_objectives: List[float] = []
    completed: List[float] = []
    cycles: List[float] = []
    failures: List[float] = []
    unmet: List[float] = []
    next_avg_trust: List[float] = []
    high_trust_edge_share: List[float] = []
    high_trust_cycle_share: List[float] = []

    for seed in range(seeds):
        payload = server.run_scenario(scenario_key, seed=seed, params_override=params_override)
        metrics = payload["metrics"]
        balanced_objectives.append(server._objective(metrics, profile="balanced"))
        high_trust_objectives.append(server._objective(metrics, profile="high_trust_network"))
        completed.append(float(metrics["completedCycles"]))
        cycles.append(float(metrics["cycleCount"]))
        failures.append(float(metrics["failedExecutionCycles"] + metrics["failedConfirmationCycles"]))
        unmet.append(float(metrics["unmetWantsAfterExecution"]))
        next_avg_trust.append(float(metrics["nextAvgTrust"]))
        high_trust_edge_share.append(float(metrics.get("highTrustEdgeShare", 0.0)))
        high_trust_cycle_share.append(float(metrics.get("highTrustCycleShare", 0.0)))

    return {
        "balancedObjectiveMean": _mean(balanced_objectives),
        "balancedObjectiveStd": _stdev(balanced_objectives),
        "highTrustObjectiveMean": _mean(high_trust_objectives),
        "highTrustObjectiveStd": _stdev(high_trust_objectives),
        "completedCyclesMean": _mean(completed),
        "cycleCountMean": _mean(cycles),
        "failureMean": _mean(failures),
        "unmetWantsMean": _mean(unmet),
        "nextAvgTrustMean": _mean(next_avg_trust),
        "highTrustEdgeShareMean": _mean(high_trust_edge_share),
        "highTrustCycleShareMean": _mean(high_trust_cycle_share),
    }


def _generalization_table(
    baseline_params: Dict[str, Any] | None,
    tuned_params: Dict[str, Any],
    seeds: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scenario_key in server.SCENARIOS.keys():
        base_balanced: List[float] = []
        tuned_balanced: List[float] = []
        base_completed: List[float] = []
        tuned_completed: List[float] = []

        for seed in range(seeds):
            base_metrics = server.run_scenario(scenario_key, seed=seed, params_override=baseline_params)["metrics"]
            tuned_metrics = server.run_scenario(scenario_key, seed=seed, params_override=tuned_params)["metrics"]
            base_balanced.append(server._objective(base_metrics, profile="balanced"))
            tuned_balanced.append(server._objective(tuned_metrics, profile="balanced"))
            base_completed.append(float(base_metrics["completedCycles"]))
            tuned_completed.append(float(tuned_metrics["completedCycles"]))

        rows.append(
            {
                "scenarioKey": scenario_key,
                "baselineBalancedObjectiveMean": _mean(base_balanced),
                "tunedBalancedObjectiveMean": _mean(tuned_balanced),
                "deltaBalancedObjectiveMean": _mean(tuned_balanced) - _mean(base_balanced),
                "baselineCompletedMean": _mean(base_completed),
                "tunedCompletedMean": _mean(tuned_completed),
                "deltaCompletedMean": _mean(tuned_completed) - _mean(base_completed),
            }
        )
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _bar(draw: ImageDraw.ImageDraw, left: int, top: int, width: int, value: float, scale: float, color: tuple[int, int, int]) -> None:
    draw.rectangle((left, top, left + width, top + 20), outline=(205, 210, 214), fill=(246, 248, 250))
    if scale <= 0:
        return
    fill_w = int(max(0.0, min(1.0, value / scale)) * width)
    draw.rectangle((left, top, left + fill_w, top + 20), fill=color)


def _render_dashboard(path: Path, rows: List[Dict[str, Any]], scenario_key: str, seeds: int) -> None:
    img = Image.new("RGB", (1500, 900), (247, 248, 245))
    draw = ImageDraw.Draw(img)
    font_title = ImageFont.load_default()
    font_body = ImageFont.load_default()

    draw.rectangle((24, 24, 1476, 876), fill=(252, 252, 251), outline=(212, 216, 214), width=2)
    draw.text((44, 42), f"High-Trust Network Optimization Dashboard ({scenario_key})", fill=(22, 28, 32), font=font_title)
    draw.text((44, 60), f"Seeds: {seeds} | Generated UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}", fill=(78, 86, 93), font=font_body)

    labels = [row["config"] for row in rows]
    metrics = [
        ("High-trust objective mean", "highTrustObjectiveMean", (42, 140, 116)),
        ("Completed cycles mean", "completedCyclesMean", (48, 120, 176)),
        ("Next avg trust mean", "nextAvgTrustMean", (170, 114, 64)),
        ("High-trust edge share", "highTrustEdgeShareMean", (158, 74, 74)),
    ]

    y = 110
    for metric_title, metric_key, color in metrics:
        draw.text((44, y), metric_title, fill=(32, 38, 42), font=font_body)
        values = [float(row[metric_key]) for row in rows]
        scale = max(values) if values else 1.0
        row_y = y + 20
        for idx, label in enumerate(labels):
            draw.text((64, row_y + idx * 28 + 5), label, fill=(62, 70, 76), font=font_body)
            _bar(draw, 280, row_y + idx * 28, 520, values[idx], scale, color)
            draw.text((810, row_y + idx * 28 + 5), f"{values[idx]:.4f}", fill=(44, 52, 58), font=font_body)
        y += 120

    draw.text((44, 620), "Balanced objective mean (for cross-profile sanity)", fill=(32, 38, 42), font=font_body)
    b_vals = [float(row["balancedObjectiveMean"]) for row in rows]
    min_v = min(b_vals) if b_vals else 0.0
    max_v = max(b_vals) if b_vals else 1.0
    span = max(max_v - min_v, 1e-9)
    base_left = 280
    for idx, label in enumerate(labels):
        y0 = 648 + idx * 30
        draw.text((64, y0 + 5), label, fill=(62, 70, 76), font=font_body)
        draw.rectangle((base_left, y0, base_left + 520, y0 + 20), outline=(205, 210, 214), fill=(246, 248, 250))
        ratio = (b_vals[idx] - min_v) / span
        draw.rectangle((base_left, y0, base_left + int(ratio * 520), y0 + 20), fill=(122, 134, 145))
        draw.text((810, y0 + 5), f"{b_vals[idx]:.4f}", fill=(44, 52, 58), font=font_body)

    img.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize/evaluate high-trust network strategy.")
    parser.add_argument("--scenario", default="high_trust_network", help="Scenario key to optimize for.")
    parser.add_argument("--eval-seeds", type=int, default=500, help="Deterministic seeds for final evaluation.")
    parser.add_argument("--opt-budget", type=int, default=80, help="Candidate budget for optimization.")
    parser.add_argument("--opt-seeds", type=int, default=12, help="Seeds per candidate during optimization.")
    parser.add_argument("--outdir", type=Path, default=ROOT / "artifacts" / "high_trust")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.scenario not in server.SCENARIOS:
        raise SystemExit(f"Unknown scenario: {args.scenario}")

    balanced_opt = server.optimize_parameters(
        scenario_key=args.scenario,
        budget=args.opt_budget,
        seeds_per_candidate=args.opt_seeds,
        objective_profile="balanced",
    )
    high_trust_opt = server.optimize_parameters(
        scenario_key=args.scenario,
        budget=args.opt_budget,
        seeds_per_candidate=args.opt_seeds,
        objective_profile="high_trust_network",
    )

    configs = [
        ("default", None, "Current default params"),
        ("balanced_optimized", balanced_opt["best"]["params"], "Optimized with balanced objective"),
        ("high_trust_optimized", high_trust_opt["best"]["params"], "Optimized with high_trust_network objective"),
    ]

    rows: List[Dict[str, Any]] = []
    for name, params_override, note in configs:
        stats = _evaluate_config(args.scenario, args.eval_seeds, params_override)
        rows.append(
            {
                "config": name,
                "note": note,
                **stats,
            }
        )

    _write_csv(outdir / "high_trust_scorecard.csv", rows)
    _write_csv(
        outdir / "high_trust_optimizer_candidates.csv",
        [
            {
                "objectiveProfile": "balanced",
                "scenarioKey": balanced_opt["scenarioKey"],
                "baselineObjectiveMean": balanced_opt["baseline"]["objectiveMean"],
                "bestObjectiveMean": balanced_opt["best"]["objectiveMean"],
                "deltaObjectiveMean": balanced_opt["best"]["objectiveMean"] - balanced_opt["baseline"]["objectiveMean"],
                "bestCandidateIndex": balanced_opt["best"]["candidateIndex"],
            },
            {
                "objectiveProfile": "high_trust_network",
                "scenarioKey": high_trust_opt["scenarioKey"],
                "baselineObjectiveMean": high_trust_opt["baseline"]["objectiveMean"],
                "bestObjectiveMean": high_trust_opt["best"]["objectiveMean"],
                "deltaObjectiveMean": high_trust_opt["best"]["objectiveMean"] - high_trust_opt["baseline"]["objectiveMean"],
                "bestCandidateIndex": high_trust_opt["best"]["candidateIndex"],
            },
        ],
    )

    summary = {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "scenarioKey": args.scenario,
        "evalSeeds": args.eval_seeds,
        "optBudget": args.opt_budget,
        "optSeedsPerCandidate": args.opt_seeds,
        "balancedOptimization": balanced_opt,
        "highTrustOptimization": high_trust_opt,
        "scorecard": rows,
    }

    generalization_rows = _generalization_table(
        baseline_params=None,
        tuned_params=high_trust_opt["best"]["params"],
        seeds=max(120, min(args.eval_seeds, 240)),
    )
    _write_csv(outdir / "high_trust_generalization.csv", generalization_rows)
    summary["generalization"] = generalization_rows

    (outdir / "high_trust_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _render_dashboard(
        path=outdir / "high_trust_dashboard.png",
        rows=rows,
        scenario_key=args.scenario,
        seeds=args.eval_seeds,
    )

    print("High-trust benchmark complete.")
    print(f"Artifacts: {outdir}")
    for filename in [
        "high_trust_scorecard.csv",
        "high_trust_optimizer_candidates.csv",
        "high_trust_generalization.csv",
        "high_trust_summary.json",
        "high_trust_dashboard.png",
    ]:
        print(f" - {outdir / filename}")


if __name__ == "__main__":
    main()
