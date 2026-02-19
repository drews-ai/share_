#!/usr/bin/env python3
"""
Monte Carlo benchmark for thesis-aligned wave tessellation variants C/D/E.

Outputs under --outdir:
- cde_raw_runs.csv
- cde_group_summary.csv
- cde_pairwise_vs_c.csv
- cde_summary.json
- cde_report.md
- cde_dashboard.png
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt

import server
import wave_tessellation_fulltilt as wave


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class BenchmarkConfig:
    grid: wave.CellConfig
    agents_per_cell: int
    weeks: int
    active_rate: float
    max_hop: int
    patience_per_hop: int
    panel_mode: str
    spice_profile: str
    max_offers_per_agent: int
    max_wants_per_agent: int
    decomposition_rate: float
    cross_context_want_rate: float
    variants: Tuple[str, ...]
    seeds: int
    seed_start: int
    bootstrap_samples: int
    permutation_samples: int
    outdir: Path


METRICS = (
    "mean_week_runtime_s",
    "mean_completed_per_1000_active",
    "mean_unmet_ratio",
    "mean_next_avg_trust",
    "mean_next_avg_patience",
    "mean_matching_cross_share",
    "mean_completed_cross_share",
    "mean_cycle_survival",
    "mean_high_trust_edge_share",
    "mean_high_trust_cycle_share",
    "final_trail_mass",
    "high_trust_objective",
)


def _mean(values: Iterable[float]) -> float:
    data = list(values)
    return statistics.fmean(data) if data else 0.0


def _std(values: Iterable[float]) -> float:
    data = list(values)
    if len(data) < 2:
        return 0.0
    return statistics.stdev(data)


def _bootstrap_ci(values: Sequence[float], rng: random.Random, n_boot: int = 2000, alpha: float = 0.05) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    n = len(values)
    means: List[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(_mean(sample))
    means.sort()
    lo_idx = int((alpha / 2) * (len(means) - 1))
    hi_idx = int((1 - alpha / 2) * (len(means) - 1))
    return means[lo_idx], means[hi_idx]


def _cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    wins = 0
    losses = 0
    for x in a:
        for y in b:
            if x > y:
                wins += 1
            elif x < y:
                losses += 1
    total = len(a) * len(b)
    return (wins - losses) / total if total else 0.0


def _permutation_pvalue(a: Sequence[float], b: Sequence[float], rng: random.Random, n_perm: int = 5000) -> float:
    if not a or not b:
        return 1.0
    observed = abs(_mean(a) - _mean(b))
    combined = list(a) + list(b)
    n_a = len(a)
    extreme = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        diff = abs(_mean(perm_a) - _mean(perm_b))
        if diff >= observed:
            extreme += 1
    return (extreme + 1) / (n_perm + 1)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _run_single_variant_seed(cfg: BenchmarkConfig, variant: str, seed: int) -> Dict[str, Any]:
    sim_cfg = wave.SimulationConfig(
        grid=cfg.grid,
        agents_per_cell=cfg.agents_per_cell,
        weeks=cfg.weeks,
        active_rate=cfg.active_rate,
        max_hop=cfg.max_hop,
        patience_per_hop=cfg.patience_per_hop,
        panel_mode=cfg.panel_mode,
        variant=variant,
        seed=seed,
        outdir=cfg.outdir,
        max_offers_per_agent=cfg.max_offers_per_agent,
        max_wants_per_agent=cfg.max_wants_per_agent,
        decomposition_rate=cfg.decomposition_rate,
        cross_context_want_rate=cfg.cross_context_want_rate,
        spice_profile=cfg.spice_profile,
    )

    params = server.DEFAULT_PARAMS
    agents, home_cell, _ = wave._build_agents(sim_cfg)
    rng = random.Random(seed + 99)
    trail_memory: Dict[wave.TrailKey, float] = {}
    n_active = max(2, int(len(agents) * cfg.active_rate))
    fixed_indices = rng.sample(range(len(agents)), n_active) if cfg.panel_mode == "fixed" else []

    weekly_rows: List[Dict[str, float]] = []
    started = time.perf_counter()
    for week in range(1, cfg.weeks + 1):
        week_seed = seed + week * 1009
        if cfg.panel_mode == "fixed":
            active_indices = fixed_indices
        else:
            active_indices = rng.sample(range(len(agents)), n_active)
        active_agents = [agents[idx] for idx in active_indices]

        week_started = time.perf_counter()
        metrics, projection, matching, cycles, reach_by_agent = wave._run_wave_week(
            active_agents=active_agents,
            params=params,
            home_cell=home_cell,
            grid=cfg.grid,
            max_hop=cfg.max_hop,
            patience_per_hop=cfg.patience_per_hop,
            seed=week_seed,
            variant=variant,
            trail_memory=trail_memory if variant == "F" else None,
        )
        if variant == "F":
            wave._update_trail_memory(trail_memory=trail_memory, cycles=cycles, home_cell=home_cell)
        wave._update_agents_from_projection(active_agents, projection)
        wave_stats = wave._cross_cell_stats(matching=matching, cycles=cycles, home_cell=home_cell)
        reaches = [reach_by_agent[a.agent_id] for a in active_agents]

        weekly_rows.append(
            {
                "week_runtime_s": time.perf_counter() - week_started,
                "completed_per_1000_active": metrics["completedCycles"] * 1000.0 / len(active_agents),
                "unmet_ratio": metrics["unmetWantsAfterExecution"] / max(metrics["totalWants"], 1),
                "next_avg_trust": float(metrics["nextAvgTrust"]),
                "next_avg_patience": float(metrics["nextAvgPatience"]),
                "matching_cross_share": wave_stats["matching_cross_share"],
                "completed_cross_share": wave_stats["completed_cross_share"],
                "cycle_survival": float(metrics["cycleSurvival"]),
                "high_trust_edge_share": float(metrics["highTrustEdgeShare"]),
                "high_trust_cycle_share": float(metrics["highTrustCycleShare"]),
                "share_reach_ge1": sum(1 for r in reaches if r >= 1) / len(reaches),
                "share_reach_ge2": sum(1 for r in reaches if r >= 2) / len(reaches),
            }
        )

    high_trust_objective = (
        0.85 * _mean(r["completed_per_1000_active"] for r in weekly_rows)
        - 12.0 * _mean(r["unmet_ratio"] for r in weekly_rows)
        + 8.0 * _mean(r["next_avg_trust"] for r in weekly_rows)
        + 4.0 * _mean(r["high_trust_cycle_share"] for r in weekly_rows)
    )

    return {
        "variant": variant,
        "seed": seed,
        "total_agents": len(agents),
        "active_agents_per_week": n_active,
        "weeks": cfg.weeks,
        "total_runtime_s": time.perf_counter() - started,
        "mean_week_runtime_s": _mean(r["week_runtime_s"] for r in weekly_rows),
        "mean_completed_per_1000_active": _mean(r["completed_per_1000_active"] for r in weekly_rows),
        "mean_unmet_ratio": _mean(r["unmet_ratio"] for r in weekly_rows),
        "mean_next_avg_trust": _mean(r["next_avg_trust"] for r in weekly_rows),
        "mean_next_avg_patience": _mean(r["next_avg_patience"] for r in weekly_rows),
        "mean_matching_cross_share": _mean(r["matching_cross_share"] for r in weekly_rows),
        "mean_completed_cross_share": _mean(r["completed_cross_share"] for r in weekly_rows),
        "mean_cycle_survival": _mean(r["cycle_survival"] for r in weekly_rows),
        "mean_high_trust_edge_share": _mean(r["high_trust_edge_share"] for r in weekly_rows),
        "mean_high_trust_cycle_share": _mean(r["high_trust_cycle_share"] for r in weekly_rows),
        "final_share_reach_ge1": weekly_rows[-1]["share_reach_ge1"] if weekly_rows else 0.0,
        "final_share_reach_ge2": weekly_rows[-1]["share_reach_ge2"] if weekly_rows else 0.0,
        "final_trail_mass": float(sum(trail_memory.values())) if variant == "F" else 0.0,
        "high_trust_objective": high_trust_objective,
    }


def _summarize_by_variant(cfg: BenchmarkConfig, raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rng = random.Random(cfg.seed_start + 700_001)
    rows: List[Dict[str, Any]] = []
    for variant in cfg.variants:
        bucket = [row for row in raw_rows if row["variant"] == variant]
        out: Dict[str, Any] = {"variant": variant, "runs": len(bucket)}
        for metric in METRICS:
            vals = [float(row[metric]) for row in bucket]
            lo, hi = _bootstrap_ci(vals, rng=rng, n_boot=cfg.bootstrap_samples)
            out[f"{metric}_mean"] = _mean(vals)
            out[f"{metric}_std"] = _std(vals)
            out[f"{metric}_ci95_lo"] = lo
            out[f"{metric}_ci95_hi"] = hi
        rows.append(out)
    return rows


def _pairwise_vs_c(cfg: BenchmarkConfig, raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    baseline = [row for row in raw_rows if row["variant"] == "C"]
    rng = random.Random(cfg.seed_start + 900_123)
    pairs: List[Dict[str, Any]] = []
    for variant in cfg.variants:
        if variant == "C":
            continue
        comp = [row for row in raw_rows if row["variant"] == variant]
        for metric in METRICS:
            a = [float(row[metric]) for row in comp]
            b = [float(row[metric]) for row in baseline]
            pairs.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "delta_mean_variant_minus_c": _mean(a) - _mean(b),
                    "cliffs_delta": _cliffs_delta(a, b),
                    "permutation_pvalue": _permutation_pvalue(a, b, rng=rng, n_perm=cfg.permutation_samples),
                }
            )
    return pairs


def _render_dashboard(path: Path, summary_rows: List[Dict[str, Any]]) -> None:
    by_variant = {row["variant"]: row for row in summary_rows}
    variants = sorted(by_variant.keys())
    x = list(range(len(variants)))

    specs = [
        ("mean_completed_per_1000_active", "Completed / 1000 active (higher)", "#2b8cbe"),
        ("mean_unmet_ratio", "Unmet ratio (lower)", "#d95f0e"),
        ("mean_next_avg_trust", "Next avg trust (higher)", "#1b9e77"),
        ("mean_completed_cross_share", "Completed cross-cell share (higher)", "#756bb1"),
        ("high_trust_objective", "High-trust objective (higher)", "#a6761d"),
        ("mean_week_runtime_s", "Week runtime seconds (lower)", "#636363"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Wave C/D/E Benchmark (95% bootstrap CI)", fontsize=14, fontweight="bold")
    for ax, (metric, title, color) in zip(axes.flat, specs):
        means = [float(by_variant[v][f"{metric}_mean"]) for v in variants]
        los = [float(by_variant[v][f"{metric}_ci95_lo"]) for v in variants]
        his = [float(by_variant[v][f"{metric}_ci95_hi"]) for v in variants]
        err = [[m - lo for m, lo in zip(means, los)], [hi - m for m, hi in zip(means, his)]]
        bars = ax.bar(x, means, yerr=err, capsize=5, color=color, alpha=0.88)
        ax.set_xticks(x)
        ax.set_xticklabels(variants)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
        for idx, bar in enumerate(bars):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{means[idx]:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=190)
    plt.close(fig)


def _format_table(rows: List[Dict[str, Any]], columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(f"{val:.6f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _build_report(cfg: BenchmarkConfig, summary_rows: List[Dict[str, Any]], pair_rows: List[Dict[str, Any]]) -> str:
    report: List[str] = []
    report.append("# Wave Variant Benchmark Report")
    report.append("")
    report.append(f"Generated UTC: `{datetime.now(timezone.utc).isoformat()}`")
    report.append("")
    report.append("## Configuration")
    report.append("")
    report.append(f"- Grid: `{cfg.grid.width}x{cfg.grid.height}`")
    report.append(f"- Agents per tessellation: `{cfg.agents_per_cell}`")
    report.append(f"- Weeks: `{cfg.weeks}`")
    report.append(f"- Active rate: `{cfg.active_rate}`")
    report.append(f"- Max hop: `{cfg.max_hop}`")
    report.append(f"- Patience per hop: `{cfg.patience_per_hop}`")
    report.append(f"- Panel mode: `{cfg.panel_mode}`")
    report.append(f"- Spice profile: `{cfg.spice_profile}`")
    report.append(f"- Max offers per agent: `{cfg.max_offers_per_agent}`")
    report.append(f"- Max wants per agent: `{cfg.max_wants_per_agent}`")
    report.append(f"- Need decomposition rate: `{cfg.decomposition_rate}`")
    report.append(f"- Cross-context want rate: `{cfg.cross_context_want_rate}`")
    report.append(f"- Variants: `{', '.join(cfg.variants)}`")
    report.append(f"- Seeds per variant: `{cfg.seeds}`")
    report.append("")
    report.append("## Group Summary")
    report.append("")
    report.append(
        _format_table(
            [
                {
                    "variant": r["variant"],
                    "runs": r["runs"],
                    "completed_per_1000_mean": r["mean_completed_per_1000_active_mean"],
                    "completed_per_1000_ci95_lo": r["mean_completed_per_1000_active_ci95_lo"],
                    "completed_per_1000_ci95_hi": r["mean_completed_per_1000_active_ci95_hi"],
                    "unmet_ratio_mean": r["mean_unmet_ratio_mean"],
                    "next_avg_trust_mean": r["mean_next_avg_trust_mean"],
                    "cross_completed_mean": r["mean_completed_cross_share_mean"],
                    "objective_mean": r["high_trust_objective_mean"],
                }
                for r in summary_rows
            ],
            (
                "variant",
                "runs",
                "completed_per_1000_mean",
                "completed_per_1000_ci95_lo",
                "completed_per_1000_ci95_hi",
                "unmet_ratio_mean",
                "next_avg_trust_mean",
                "cross_completed_mean",
                "objective_mean",
            ),
        )
    )
    report.append("")
    report.append("## Pairwise (Variant vs C)")
    report.append("")
    selected_pairs = [
        row
        for row in pair_rows
        if row["metric"] in ("mean_completed_per_1000_active", "mean_unmet_ratio", "mean_next_avg_trust", "high_trust_objective")
    ]
    report.append(
        _format_table(
            selected_pairs,
            ("variant", "metric", "delta_mean_variant_minus_c", "cliffs_delta", "permutation_pvalue"),
        )
    )
    report.append("")
    return "\n".join(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run C/D/E Monte Carlo benchmark for wave tessellation simulation.")
    parser.add_argument("--grid", default="3x3")
    parser.add_argument("--agents-per-cell", type=int, default=3000)
    parser.add_argument("--weeks", type=int, default=6)
    parser.add_argument("--active-rate", type=float, default=0.02)
    parser.add_argument("--max-hop", type=int, default=5)
    parser.add_argument("--patience-per-hop", type=int, default=1)
    parser.add_argument("--panel-mode", choices=("fixed", "resample"), default="fixed")
    parser.add_argument("--spice-profile", choices=("none", "thesis_ideation", "thesis_spice"), default="none")
    parser.add_argument("--max-offers-per-agent", type=int, default=None)
    parser.add_argument("--max-wants-per-agent", type=int, default=None)
    parser.add_argument("--decomposition-rate", type=float, default=None)
    parser.add_argument("--cross-context-want-rate", type=float, default=None)
    parser.add_argument("--variants", default="C,D,E", help="Comma-separated, e.g. C,D,E,F")
    parser.add_argument("--seeds", type=int, default=10, help="Replications per variant")
    parser.add_argument("--seed-start", type=int, default=20260219)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--permutation-samples", type=int, default=3000)
    parser.add_argument("--outdir", type=Path, default=ROOT / "artifacts" / "wave_cde_benchmark")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = tuple(part.strip().upper() for part in args.variants.split(",") if part.strip())
    for variant in variants:
        if variant not in {"C", "D", "E", "F"}:
            raise ValueError(f"Unsupported variant '{variant}'. Choose from C, D, E, F.")

    max_offers, max_wants, decomposition_rate, cross_context_rate = wave._resolve_spice_parameters(args)
    cfg = BenchmarkConfig(
        grid=wave._parse_grid(args.grid),
        agents_per_cell=args.agents_per_cell,
        weeks=args.weeks,
        active_rate=args.active_rate,
        max_hop=args.max_hop,
        patience_per_hop=args.patience_per_hop,
        panel_mode=args.panel_mode,
        spice_profile=args.spice_profile,
        max_offers_per_agent=max_offers,
        max_wants_per_agent=max_wants,
        decomposition_rate=decomposition_rate,
        cross_context_want_rate=cross_context_rate,
        variants=variants,
        seeds=args.seeds,
        seed_start=args.seed_start,
        bootstrap_samples=args.bootstrap_samples,
        permutation_samples=args.permutation_samples,
        outdir=args.outdir.resolve(),
    )
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    raw_rows: List[Dict[str, Any]] = []
    total_runs = len(cfg.variants) * cfg.seeds
    run_idx = 0
    started = time.perf_counter()
    for variant in cfg.variants:
        for offset in range(cfg.seeds):
            run_idx += 1
            seed = cfg.seed_start + offset
            t0 = time.perf_counter()
            row = _run_single_variant_seed(cfg, variant=variant, seed=seed)
            raw_rows.append(row)
            print(
                f"[{run_idx:03d}/{total_runs:03d}] variant={variant} seed={seed} "
                f"obj={row['high_trust_objective']:.3f} completed/1000={row['mean_completed_per_1000_active']:.3f} "
                f"unmet={row['mean_unmet_ratio']:.3f} trust={row['mean_next_avg_trust']:.3f} "
                f"runtime={time.perf_counter() - t0:.2f}s"
            )

    summary_rows = _summarize_by_variant(cfg, raw_rows)
    pair_rows = _pairwise_vs_c(cfg, raw_rows)
    total_elapsed = time.perf_counter() - started

    summary_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "grid": f"{cfg.grid.width}x{cfg.grid.height}",
            "agents_per_cell": cfg.agents_per_cell,
            "weeks": cfg.weeks,
            "active_rate": cfg.active_rate,
            "max_hop": cfg.max_hop,
            "patience_per_hop": cfg.patience_per_hop,
            "panel_mode": cfg.panel_mode,
            "spice_profile": cfg.spice_profile,
            "max_offers_per_agent": cfg.max_offers_per_agent,
            "max_wants_per_agent": cfg.max_wants_per_agent,
            "decomposition_rate": cfg.decomposition_rate,
            "cross_context_want_rate": cfg.cross_context_want_rate,
            "variants": cfg.variants,
            "seeds": cfg.seeds,
            "seed_start": cfg.seed_start,
        },
        "runtime_s": total_elapsed,
        "summary_rows": summary_rows,
        "pairwise_vs_c": pair_rows,
    }

    _write_csv(cfg.outdir / "cde_raw_runs.csv", raw_rows)
    _write_csv(cfg.outdir / "cde_group_summary.csv", summary_rows)
    _write_csv(cfg.outdir / "cde_pairwise_vs_c.csv", pair_rows)
    (cfg.outdir / "cde_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    (cfg.outdir / "cde_report.md").write_text(_build_report(cfg, summary_rows, pair_rows), encoding="utf-8")
    _render_dashboard(cfg.outdir / "cde_dashboard.png", summary_rows)

    print("Wave variant benchmark complete.")
    print(f"Output directory: {cfg.outdir}")
    print("Artifacts:")
    for name in (
        "cde_raw_runs.csv",
        "cde_group_summary.csv",
        "cde_pairwise_vs_c.csv",
        "cde_summary.json",
        "cde_report.md",
        "cde_dashboard.png",
    ):
        print(f" - {cfg.outdir / name}")


if __name__ == "__main__":
    main()
