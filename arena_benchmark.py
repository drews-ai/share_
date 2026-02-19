#!/usr/bin/env python3
"""
Run deterministic benchmark sweeps for the ShareWith cycle trading engine.

Artifacts written under ./artifacts:
- benchmark_runs.csv
- benchmark_scorecard.csv
- benchmark_summary.json
- optimizer_summary.json
- benchmark_dashboard.png
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"Pillow is required to render charts: {exc}") from exc


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import server  # noqa: E402


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    params_override: Optional[Dict[str, Any]]
    note: str


CONFIGS: Tuple[BenchmarkConfig, ...] = (
    BenchmarkConfig(
        name="legacy_social_3",
        params_override={"context_base_distance": {"SOCIAL": 3.0}},
        note="Strict social radius (matches pre-tuned default).",
    ),
    BenchmarkConfig(
        name="tuned_default",
        params_override=None,
        note="Current tuned default parameters.",
    ),
)


def _mean(values: Iterable[float]) -> float:
    data = list(values)
    return statistics.fmean(data) if data else 0.0


def _safe_stdev(values: Iterable[float]) -> float:
    data = list(values)
    if len(data) <= 1:
        return 0.0
    return statistics.pstdev(data)


def run_benchmark(seeds_per_scenario: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    run_rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str], Dict[str, List[float]]] = {}

    scenario_keys = list(server.SCENARIOS.keys())
    for config in CONFIGS:
        for scenario_key in scenario_keys:
            for seed in range(seeds_per_scenario):
                payload = server.run_scenario(
                    scenario_key=scenario_key,
                    seed=seed,
                    params_override=config.params_override,
                )
                metrics = payload["metrics"]
                objective = server._objective(metrics)

                row = {
                    "config": config.name,
                    "scenarioKey": scenario_key,
                    "seed": seed,
                    "objective": objective,
                    "matchingSize": metrics["matchingSize"],
                    "cycleCount": metrics["cycleCount"],
                    "completedCycles": metrics["completedCycles"],
                    "failedConfirmationCycles": metrics["failedConfirmationCycles"],
                    "failedExecutionCycles": metrics["failedExecutionCycles"],
                    "rolledBackCycles": metrics["rolledBackCycles"],
                    "unmatchedWantsAfterMatch": metrics["unmatchedWantsAfterMatch"],
                    "unmetWantsAfterExecution": metrics["unmetWantsAfterExecution"],
                    "edgeRatio": metrics["edgeRatio"],
                    "avgDistanceMatched": metrics["avgDistanceMatched"],
                    "avgDistanceCompleted": metrics["avgDistanceCompleted"],
                }
                run_rows.append(row)

                bucket = grouped.setdefault((config.name, scenario_key), {key: [] for key in row if key not in ("config", "scenarioKey", "seed")})
                for metric_key, metric_value in row.items():
                    if metric_key in ("config", "scenarioKey", "seed"):
                        continue
                    bucket[metric_key].append(float(metric_value))

    scorecard_rows: List[Dict[str, Any]] = []
    for config in CONFIGS:
        for scenario_key in scenario_keys:
            metrics_group = grouped[(config.name, scenario_key)]
            scorecard_rows.append(
                {
                    "config": config.name,
                    "scenarioKey": scenario_key,
                    "objectiveMean": _mean(metrics_group["objective"]),
                    "objectiveStd": _safe_stdev(metrics_group["objective"]),
                    "matchingMean": _mean(metrics_group["matchingSize"]),
                    "cycleMean": _mean(metrics_group["cycleCount"]),
                    "completedMean": _mean(metrics_group["completedCycles"]),
                    "confirmFailMean": _mean(metrics_group["failedConfirmationCycles"]),
                    "executionFailMean": _mean(metrics_group["failedExecutionCycles"]),
                    "rollbackMean": _mean(metrics_group["rolledBackCycles"]),
                    "unmatchedAfterMatchMean": _mean(metrics_group["unmatchedWantsAfterMatch"]),
                    "unmetAfterExecutionMean": _mean(metrics_group["unmetWantsAfterExecution"]),
                    "edgeRatioMean": _mean(metrics_group["edgeRatio"]),
                }
            )

    summary = {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "seedsPerScenario": seeds_per_scenario,
        "configs": [{"name": c.name, "note": c.note} for c in CONFIGS],
        "scenarios": [
            {"key": key, "name": server.SCENARIOS[key].name, "note": server.SCENARIOS[key].note}
            for key in scenario_keys
        ],
        "scorecard": scorecard_rows,
        "globalObjectiveByConfig": {
            config.name: _mean(row["objectiveMean"] for row in scorecard_rows if row["config"] == config.name)
            for config in CONFIGS
        },
    }
    return run_rows, scorecard_rows, summary


def run_optimizer_summary(budget: int, seeds_per_candidate: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scenario_key in server.SCENARIOS.keys():
        result = server.optimize_parameters(
            scenario_key=scenario_key,
            budget=budget,
            seeds_per_candidate=seeds_per_candidate,
        )
        baseline = result["baseline"]
        best = result["best"]
        rows.append(
            {
                "scenarioKey": scenario_key,
                "budget": result["budget"],
                "seedsPerCandidate": result["seedsPerCandidate"],
                "baselineObjectiveMean": baseline["objectiveMean"],
                "bestObjectiveMean": best["objectiveMean"],
                "deltaObjectiveMean": best["objectiveMean"] - baseline["objectiveMean"],
                "baselineCompletedMean": baseline["completedCyclesMean"],
                "bestCompletedMean": best["completedCyclesMean"],
                "baselineUnmetMean": baseline["unmetWantsMean"],
                "bestUnmetMean": best["unmetWantsMean"],
                "bestCandidateIndex": best["candidateIndex"],
            }
        )
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _draw_grouped_horizontal_bars(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    scenario_keys: List[str],
    values_a: List[float],
    values_b: List[float],
    label_a: str,
    label_b: str,
    color_a: Tuple[int, int, int],
    color_b: Tuple[int, int, int],
    font: ImageFont.ImageFont,
    font_small: ImageFont.ImageFont,
) -> None:
    draw.text((x, y), title, fill=(30, 36, 38), font=font)
    chart_top = y + 24
    chart_bottom = y + h
    chart_left = x + 170
    chart_right = x + w - 20

    all_values = values_a + values_b + [0.0]
    v_min = min(all_values)
    v_max = max(all_values)
    if v_min == v_max:
        v_min -= 1.0
        v_max += 1.0

    def map_x(value: float) -> float:
        ratio = (value - v_min) / (v_max - v_min)
        return chart_left + ratio * (chart_right - chart_left)

    zero_x = map_x(0.0)
    draw.line((zero_x, chart_top, zero_x, chart_bottom), fill=(180, 188, 194), width=1)

    groups = len(scenario_keys)
    row_h = (chart_bottom - chart_top) / max(groups, 1)
    bar_h = max(6, int(row_h * 0.3))
    gap = max(2, int(row_h * 0.1))

    for idx, scenario_key in enumerate(scenario_keys):
        row_y = chart_top + idx * row_h
        draw.text((x, int(row_y + row_h * 0.3)), scenario_key, fill=(72, 82, 88), font=font_small)

        bars = [(values_a[idx], color_a, 0), (values_b[idx], color_b, bar_h + gap)]
        for value, color, offset in bars:
            top = int(row_y + row_h * 0.15 + offset)
            bottom = top + bar_h
            value_x = map_x(value)
            left = min(zero_x, value_x)
            right = max(zero_x, value_x)
            draw.rectangle((left, top, right, bottom), fill=color, outline=(0, 0, 0, 0))

    legend_y = chart_bottom + 6
    draw.rectangle((chart_left, legend_y, chart_left + 12, legend_y + 12), fill=color_a)
    draw.text((chart_left + 16, legend_y - 1), label_a, fill=(72, 82, 88), font=font_small)
    draw.rectangle((chart_left + 180, legend_y, chart_left + 192, legend_y + 12), fill=color_b)
    draw.text((chart_left + 196, legend_y - 1), label_b, fill=(72, 82, 88), font=font_small)


def render_dashboard(
    path: Path,
    scorecard_rows: List[Dict[str, Any]],
    optimizer_rows: List[Dict[str, Any]],
    seeds_per_scenario: int,
) -> None:
    scenario_keys = list(server.SCENARIOS.keys())
    by_config_scenario = {
        (row["config"], row["scenarioKey"]): row
        for row in scorecard_rows
    }

    legacy_values_obj = [by_config_scenario[("legacy_social_3", sk)]["objectiveMean"] for sk in scenario_keys]
    tuned_values_obj = [by_config_scenario[("tuned_default", sk)]["objectiveMean"] for sk in scenario_keys]
    legacy_values_completed = [by_config_scenario[("legacy_social_3", sk)]["completedMean"] for sk in scenario_keys]
    tuned_values_completed = [by_config_scenario[("tuned_default", sk)]["completedMean"] for sk in scenario_keys]

    image = Image.new("RGB", (1700, 980), (246, 247, 244))
    draw = ImageDraw.Draw(image)
    font_title = ImageFont.load_default()
    font_body = ImageFont.load_default()
    font_small = ImageFont.load_default()

    draw.rectangle((30, 26, 1670, 954), outline=(210, 216, 212), width=2, fill=(252, 252, 250))
    draw.text((50, 44), "ShareWith Simulation Arena Benchmark", fill=(28, 34, 36), font=font_title)
    draw.text(
        (50, 62),
        f"Seeds per scenario: {seeds_per_scenario} | Generated UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
        fill=(86, 96, 102),
        font=font_small,
    )

    _draw_grouped_horizontal_bars(
        draw=draw,
        x=50,
        y=94,
        w=1580,
        h=300,
        title="Objective Mean by Scenario (higher is better)",
        scenario_keys=scenario_keys,
        values_a=legacy_values_obj,
        values_b=tuned_values_obj,
        label_a="legacy_social_3",
        label_b="tuned_default",
        color_a=(200, 105, 86),
        color_b=(48, 136, 113),
        font=font_body,
        font_small=font_small,
    )

    _draw_grouped_horizontal_bars(
        draw=draw,
        x=50,
        y=430,
        w=1580,
        h=300,
        title="Completed Cycles Mean by Scenario",
        scenario_keys=scenario_keys,
        values_a=legacy_values_completed,
        values_b=tuned_values_completed,
        label_a="legacy_social_3",
        label_b="tuned_default",
        color_a=(214, 146, 82),
        color_b=(34, 122, 164),
        font=font_body,
        font_small=font_small,
    )

    table_top = 770
    draw.text((50, table_top), "Optimizer Delta (best minus baseline objective)", fill=(28, 34, 36), font=font_body)
    y = table_top + 20
    for row in optimizer_rows:
        scenario_key = row["scenarioKey"]
        delta = row["deltaObjectiveMean"]
        color = (34, 128, 87) if delta >= 0 else (170, 70, 70)
        line = (
            f"{scenario_key:16s}  delta={delta:+.3f}  "
            f"baseline={row['baselineObjectiveMean']:.3f}  best={row['bestObjectiveMean']:.3f}  "
            f"candidate={row['bestCandidateIndex']}"
        )
        draw.text((64, y), line, fill=color, font=font_small)
        y += 16

    image.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ShareWith simulation benchmark and render artifacts.")
    parser.add_argument("--seeds", type=int, default=120, help="Number of deterministic seeds per scenario/config.")
    parser.add_argument("--opt-budget", type=int, default=36, help="Parameter candidates for optimizer summary.")
    parser.add_argument("--opt-seeds", type=int, default=8, help="Seeds per optimizer candidate.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "artifacts",
        help="Output directory for benchmark artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    run_rows, scorecard_rows, summary = run_benchmark(seeds_per_scenario=args.seeds)
    optimizer_rows = run_optimizer_summary(budget=args.opt_budget, seeds_per_candidate=args.opt_seeds)

    _write_csv(outdir / "benchmark_runs.csv", run_rows)
    _write_csv(outdir / "benchmark_scorecard.csv", scorecard_rows)
    _write_csv(outdir / "optimizer_summary.csv", optimizer_rows)

    (outdir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (outdir / "optimizer_summary.json").write_text(json.dumps(optimizer_rows, indent=2), encoding="utf-8")

    render_dashboard(
        path=outdir / "benchmark_dashboard.png",
        scorecard_rows=scorecard_rows,
        optimizer_rows=optimizer_rows,
        seeds_per_scenario=args.seeds,
    )

    print("Benchmark complete.")
    print(f"Artifacts directory: {outdir}")
    print(f"Runs: {len(run_rows)}")
    print(f"Scorecard rows: {len(scorecard_rows)}")
    print(f"Optimizer rows: {len(optimizer_rows)}")
    print("Generated files:")
    for filename in [
        "benchmark_runs.csv",
        "benchmark_scorecard.csv",
        "benchmark_summary.json",
        "optimizer_summary.csv",
        "optimizer_summary.json",
        "benchmark_dashboard.png",
    ]:
        print(f" - {outdir / filename}")


if __name__ == "__main__":
    main()
