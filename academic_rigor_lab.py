#!/usr/bin/env python3
"""
Academic-style experiment harness for ShareWith cycle trading.

Outputs under ./artifacts/academic:
- raw_runs.csv
- group_summary.csv
- pairwise_tests.csv
- rigor_report.md
- rigor_report.html
- rigor_summary.json
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
import resource
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import server  # noqa: E402


@dataclass(frozen=True)
class RunSpec:
    regime: str
    architecture: str
    cohort_size: int
    repetition: int
    seed: int


KEY_METRICS = (
    "runtime_s",
    "matched_ratio",
    "cycle_count",
    "completed_cycles",
    "completed_per_1000",
    "completion_ratio",
    "unmet_ratio",
    "avg_distance_completed",
    "avg_trust",
    "next_avg_trust",
    "trust_delta",
    "avg_patience",
    "next_avg_patience",
    "patience_delta",
    "high_trust_edge_share",
    "high_trust_cycle_share",
    "cycle_survival",
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
    means = []
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


def _token(i: int) -> str:
    return f"z{i}q{i * i + 17}"


def _build_structured_agents(n_agents: int, seed: int) -> List[server.Agent]:
    rng = random.Random(seed)
    n_agents = n_agents - (n_agents % 3)
    agents: List[server.Agent] = []
    aid = 0
    for gid in range(n_agents // 3):
        s1 = _token(gid * 3 + 1)
        s2 = _token(gid * 3 + 2)
        s3 = _token(gid * 3 + 3)
        triple = (
            (s1, s3, 0.12, 0.10),
            (s2, s1, 0.14, 0.12),
            (s3, s2, 0.11, 0.14),
        )
        for offer_skill, want_skill, bx, by in triple:
            jitter_x = bx + rng.uniform(-0.01, 0.01)
            jitter_y = by + rng.uniform(-0.01, 0.01)
            agents.append(
                server.Agent(
                    agent_id=f"a{aid}",
                    name=f"A{aid}",
                    mode=server.Mode.AUTO if rng.random() < 0.88 else server.Mode.CHOICE,
                    offers=[server.Offer(skill=offer_skill, context=server.Context.SOCIAL, tags=(offer_skill,))],
                    wants=[server.Want(skill=want_skill, context=server.Context.SOCIAL, tags=(want_skill,))],
                    location=server.Location(x=jitter_x, y=jitter_y),
                    trust=server.TrustMetrics(
                        completion=max(0.75, min(0.98, rng.gauss(0.90, 0.04))),
                        quality=max(0.75, min(0.98, rng.gauss(0.90, 0.04))),
                    ),
                    patience=rng.randint(0, 2),
                    gave=rng.randint(7, 16),
                    received=rng.randint(6, 16),
                )
            )
            aid += 1
    return agents


def _build_messy_agents(n_agents: int, seed: int) -> List[server.Agent]:
    rng = random.Random(seed)
    contexts = (server.Context.SAFETY, server.Context.SOCIAL, server.Context.GROWTH)
    skills = {
        server.Context.SAFETY: ("plumbing", "electrical", "legal", "repair", "childcare", "transport"),
        server.Context.SOCIAL: ("haircut", "yard", "music", "fitness", "photo", "language"),
        server.Context.GROWTH: ("design", "coding", "tutoring", "marketing", "career", "writing"),
    }

    agents: List[server.Agent] = []
    for i in range(n_agents):
        context = rng.choices(contexts, weights=(0.45, 0.35, 0.20))[0]
        pool = skills[context]
        idx = rng.randrange(len(pool))
        offer_skill = pool[idx]
        want_skill = pool[(idx + 1) % len(pool)]
        agents.append(
            server.Agent(
                agent_id=f"a{i}",
                name=f"A{i}",
                mode=server.Mode.CHOICE if rng.random() < 0.24 else server.Mode.AUTO,
                offers=[server.Offer(skill=offer_skill, context=context, tags=(offer_skill, context.value.lower()))],
                wants=[server.Want(skill=want_skill, context=context, tags=(want_skill, context.value.lower()))],
                location=server.Location(x=rng.uniform(0, 2), y=rng.uniform(0, 2)),
                trust=server.TrustMetrics(
                    completion=max(0.35, min(0.98, rng.gauss(0.77, 0.10))),
                    quality=max(0.35, min(0.98, rng.gauss(0.79, 0.10))),
                ),
                patience=rng.randint(0, 4),
                gave=rng.randint(2, 20),
                received=rng.randint(2, 20),
            )
        )
    return agents


def _metrics_from_payload(payload: Dict[str, Any], cohort_size: int, runtime_s: float, peak_rss_mb: float, n_cells: int) -> Dict[str, Any]:
    metrics = payload["metrics"]
    cycle_count = float(metrics["cycleCount"])
    completed_cycles = float(metrics["completedCycles"])
    matching_size = float(metrics["matchingSize"])
    unmet_wants = float(metrics["unmetWantsAfterExecution"])
    avg_distance_completed = float(metrics.get("avgDistanceCompleted", 0.0))
    avg_trust = float(metrics.get("avgTrust", 0.0))
    next_avg_trust = float(metrics.get("nextAvgTrust", avg_trust))
    avg_patience = float(metrics.get("avgPatience", 0.0))
    next_avg_patience = float(metrics.get("nextAvgPatience", avg_patience))
    high_trust_edge_share = float(metrics.get("highTrustEdgeShare", 0.0))
    high_trust_cycle_share = float(metrics.get("highTrustCycleShare", 0.0))
    cycle_survival = float(metrics.get("cycleSurvival", 0.0))

    return {
        "runtime_s": runtime_s,
        "peak_rss_mb": peak_rss_mb,
        "n_cells": n_cells,
        "matched_ratio": matching_size / max(cohort_size, 1),
        "cycle_count": cycle_count,
        "completed_cycles": completed_cycles,
        "completed_per_1000": completed_cycles * 1000.0 / max(cohort_size, 1),
        "completion_ratio": completed_cycles / max(cycle_count, 1.0),
        "unmet_ratio": unmet_wants / max(cohort_size, 1),
        "avg_distance_completed": avg_distance_completed,
        "avg_trust": avg_trust,
        "next_avg_trust": next_avg_trust,
        "trust_delta": next_avg_trust - avg_trust,
        "avg_patience": avg_patience,
        "next_avg_patience": next_avg_patience,
        "patience_delta": next_avg_patience - avg_patience,
        "high_trust_edge_share": high_trust_edge_share,
        "high_trust_cycle_share": high_trust_cycle_share,
        "cycle_survival": cycle_survival,
    }


def _run_global(agents: List[server.Agent], seed: int) -> Tuple[Dict[str, Any], float]:
    scenario = server.Scenario(key="cohort", name="cohort", note="global", agents=agents)
    start = time.perf_counter()
    payload = server._run_pipeline(scenario, server.DEFAULT_PARAMS, seed=seed)
    elapsed = time.perf_counter() - start
    return payload, elapsed


def _run_partitioned(agents: List[server.Agent], seed: int, cell_size: int) -> Tuple[Dict[str, Any], float, int]:
    cells = [agents[i : i + cell_size] for i in range(0, len(agents), cell_size)]
    start = time.perf_counter()
    agg = {
        "n_agents": 0,
        "totalOffers": 0,
        "totalWants": 0,
        "potentialEdges": 0,
        "feasibleEdges": 0,
        "matchingSize": 0,
        "cycleCount": 0,
        "completedCycles": 0,
        "failedConfirmationCycles": 0,
        "failedExecutionCycles": 0,
        "unmetWantsAfterExecution": 0,
        "avgDistanceCompleted_weighted_sum": 0.0,
        "avgTrust_weighted_sum": 0.0,
        "nextAvgTrust_weighted_sum": 0.0,
        "avgPatience_weighted_sum": 0.0,
        "nextAvgPatience_weighted_sum": 0.0,
        "highTrustEdge_weighted_sum": 0.0,
        "highTrustCycle_weighted_sum": 0.0,
    }
    for idx, cell_agents in enumerate(cells):
        scenario = server.Scenario(key=f"cell_{idx}", name="cell", note="partition", agents=cell_agents)
        payload = server._run_pipeline(scenario, server.DEFAULT_PARAMS, seed=seed + idx * 97)
        m = payload["metrics"]
        n_cell_agents = len(cell_agents)
        agg["n_agents"] += n_cell_agents
        agg["totalOffers"] += int(m["totalOffers"])
        agg["totalWants"] += int(m["totalWants"])
        agg["potentialEdges"] += int(m["potentialEdges"])
        agg["feasibleEdges"] += int(m["feasibleEdges"])
        agg["matchingSize"] += int(m["matchingSize"])
        agg["cycleCount"] += int(m["cycleCount"])
        agg["completedCycles"] += int(m["completedCycles"])
        agg["failedConfirmationCycles"] += int(m["failedConfirmationCycles"])
        agg["failedExecutionCycles"] += int(m["failedExecutionCycles"])
        agg["unmetWantsAfterExecution"] += int(m["unmetWantsAfterExecution"])
        agg["avgDistanceCompleted_weighted_sum"] += float(m.get("avgDistanceCompleted", 0.0)) * int(m["completedCycles"])
        agg["avgTrust_weighted_sum"] += float(m.get("avgTrust", 0.0)) * n_cell_agents
        agg["nextAvgTrust_weighted_sum"] += float(m.get("nextAvgTrust", m.get("avgTrust", 0.0))) * n_cell_agents
        agg["avgPatience_weighted_sum"] += float(m.get("avgPatience", 0.0)) * n_cell_agents
        agg["nextAvgPatience_weighted_sum"] += float(m.get("nextAvgPatience", m.get("avgPatience", 0.0))) * n_cell_agents
        agg["highTrustEdge_weighted_sum"] += float(m.get("highTrustEdgeShare", 0.0)) * int(m["matchingSize"])
        agg["highTrustCycle_weighted_sum"] += float(m.get("highTrustCycleShare", 0.0)) * int(m["cycleCount"])
    elapsed = time.perf_counter() - start
    avg_distance_completed = (
        agg["avgDistanceCompleted_weighted_sum"] / max(agg["completedCycles"], 1)
        if agg["completedCycles"] > 0
        else 0.0
    )
    avg_trust = agg["avgTrust_weighted_sum"] / max(agg["n_agents"], 1)
    next_avg_trust = agg["nextAvgTrust_weighted_sum"] / max(agg["n_agents"], 1)
    avg_patience = agg["avgPatience_weighted_sum"] / max(agg["n_agents"], 1)
    next_avg_patience = agg["nextAvgPatience_weighted_sum"] / max(agg["n_agents"], 1)
    high_trust_edge_share = agg["highTrustEdge_weighted_sum"] / max(agg["matchingSize"], 1)
    high_trust_cycle_share = agg["highTrustCycle_weighted_sum"] / max(agg["cycleCount"], 1)
    cycle_survival = agg["completedCycles"] / max(agg["cycleCount"], 1)
    payload_like = {
        "metrics": {
            "cycleCount": agg["cycleCount"],
            "completedCycles": agg["completedCycles"],
            "matchingSize": agg["matchingSize"],
            "unmetWantsAfterExecution": agg["unmetWantsAfterExecution"],
            "avgDistanceCompleted": avg_distance_completed,
            "avgTrust": avg_trust,
            "nextAvgTrust": next_avg_trust,
            "avgPatience": avg_patience,
            "nextAvgPatience": next_avg_patience,
            "highTrustEdgeShare": high_trust_edge_share,
            "highTrustCycleShare": high_trust_cycle_share,
            "cycleSurvival": cycle_survival,
        }
    }
    return payload_like, elapsed, len(cells)


def _run_spec(spec: RunSpec, cell_size: int) -> Dict[str, Any]:
    if spec.regime == "structured":
        agents = _build_structured_agents(spec.cohort_size, spec.seed)
    elif spec.regime == "messy":
        agents = _build_messy_agents(spec.cohort_size, spec.seed)
    else:
        raise ValueError(f"Unknown regime: {spec.regime}")

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if spec.architecture == "global":
        payload, runtime_s = _run_global(agents, spec.seed)
        n_cells = 1
    elif spec.architecture == "partitioned":
        payload, runtime_s, n_cells = _run_partitioned(agents, spec.seed, cell_size=cell_size)
    else:
        raise ValueError(f"Unknown architecture: {spec.architecture}")
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_mb = max(rss_before, rss_after) / 1024.0

    metrics = _metrics_from_payload(
        payload=payload,
        cohort_size=spec.cohort_size,
        runtime_s=runtime_s,
        peak_rss_mb=peak_rss_mb,
        n_cells=n_cells,
    )
    return {
        "regime": spec.regime,
        "architecture": spec.architecture,
        "cohort_size": spec.cohort_size,
        "repetition": spec.repetition,
        "seed": spec.seed,
        **metrics,
    }


def _summarize_groups(rows: List[Dict[str, Any]], bootstrap_samples: int, rng: random.Random) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["regime"]), str(row["architecture"]), int(row["cohort_size"]))
        grouped.setdefault(key, []).append(row)

    summary: List[Dict[str, Any]] = []
    for (regime, architecture, cohort_size), group_rows in sorted(grouped.items(), key=lambda t: (t[0][0], t[0][2], t[0][1])):
        base = {
            "regime": regime,
            "architecture": architecture,
            "cohort_size": cohort_size,
            "n_runs": len(group_rows),
        }
        for metric in KEY_METRICS:
            values = [float(r[metric]) for r in group_rows]
            ci_lo, ci_hi = _bootstrap_ci(values, rng=rng, n_boot=bootstrap_samples)
            base[f"{metric}_mean"] = _mean(values)
            base[f"{metric}_std"] = _std(values)
            base[f"{metric}_ci95_lo"] = ci_lo
            base[f"{metric}_ci95_hi"] = ci_hi
        summary.append(base)
    return summary


def _pairwise_tests(rows: List[Dict[str, Any]], permutations: int, rng: random.Random) -> List[Dict[str, Any]]:
    by_group: Dict[Tuple[str, int], Dict[str, List[Dict[str, Any]]]] = {}
    for row in rows:
        key = (str(row["regime"]), int(row["cohort_size"]))
        by_group.setdefault(key, {}).setdefault(str(row["architecture"]), []).append(row)

    metrics_to_test = (
        "runtime_s",
        "completed_per_1000",
        "unmet_ratio",
        "next_avg_trust",
        "next_avg_patience",
        "high_trust_cycle_share",
    )
    tests: List[Dict[str, Any]] = []
    for (regime, cohort_size), arch_map in sorted(by_group.items(), key=lambda t: (t[0][0], t[0][1])):
        if "global" not in arch_map or "partitioned" not in arch_map:
            continue
        global_rows = arch_map["global"]
        partition_rows = arch_map["partitioned"]
        for metric in metrics_to_test:
            a = [float(r[metric]) for r in global_rows]
            b = [float(r[metric]) for r in partition_rows]
            tests.append(
                {
                    "regime": regime,
                    "cohort_size": cohort_size,
                    "metric": metric,
                    "global_mean": _mean(a),
                    "partitioned_mean": _mean(b),
                    "delta_partition_minus_global": _mean(b) - _mean(a),
                    "cliffs_delta_global_vs_partitioned": _cliffs_delta(a, b),
                    "permutation_pvalue": _permutation_pvalue(a, b, rng=rng, n_perm=permutations),
                }
            )
    return tests


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


def _build_markdown_report(
    outdir: Path,
    summary_rows: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    concise_summary = []
    for row in summary_rows:
        concise_summary.append(
            {
                "regime": row["regime"],
                "architecture": row["architecture"],
                "cohort_size": row["cohort_size"],
                "n_runs": row["n_runs"],
                "runtime_mean_s": row["runtime_s_mean"],
                "runtime_ci95_lo": row["runtime_s_ci95_lo"],
                "runtime_ci95_hi": row["runtime_s_ci95_hi"],
                "completed_per_1000_mean": row["completed_per_1000_mean"],
                "completed_per_1000_ci95_lo": row["completed_per_1000_ci95_lo"],
                "completed_per_1000_ci95_hi": row["completed_per_1000_ci95_hi"],
                "unmet_ratio_mean": row["unmet_ratio_mean"],
                "unmet_ratio_ci95_lo": row["unmet_ratio_ci95_lo"],
                "unmet_ratio_ci95_hi": row["unmet_ratio_ci95_hi"],
                "next_avg_trust_mean": row["next_avg_trust_mean"],
                "next_avg_trust_ci95_lo": row["next_avg_trust_ci95_lo"],
                "next_avg_trust_ci95_hi": row["next_avg_trust_ci95_hi"],
                "next_avg_patience_mean": row["next_avg_patience_mean"],
                "next_avg_patience_ci95_lo": row["next_avg_patience_ci95_lo"],
                "next_avg_patience_ci95_hi": row["next_avg_patience_ci95_hi"],
                "high_trust_cycle_share_mean": row["high_trust_cycle_share_mean"],
                "high_trust_cycle_share_ci95_lo": row["high_trust_cycle_share_ci95_lo"],
                "high_trust_cycle_share_ci95_hi": row["high_trust_cycle_share_ci95_hi"],
            }
        )

    report = []
    report.append("# ShareWith Academic Rigor Report")
    report.append("")
    report.append(f"Generated UTC: `{datetime.now(timezone.utc).isoformat()}`")
    report.append("")
    report.append("## Methods")
    report.append("")
    report.append(f"- Structured cohort sizes: `{args.sizes_structured}`")
    report.append(f"- Messy cohort sizes: `{args.sizes_messy}`")
    report.append(f"- Repetitions per condition: `{args.repetitions}`")
    report.append(f"- Partition cell size: `{args.partition_cell_size}`")
    report.append(f"- Bootstrap samples (95% CI): `{args.bootstrap_samples}`")
    report.append(f"- Permutation test samples: `{args.permutations}`")
    report.append("- Comparison axes: `regime x architecture x cohort_size`")
    report.append("- Tested metrics: `runtime_s`, `completed_per_1000`, `unmet_ratio`, `next_avg_trust`, `next_avg_patience`, `high_trust_cycle_share`")
    report.append("")
    report.append("## Group Summary")
    report.append("")
    report.append(
        _format_table(
            concise_summary,
            (
                "regime",
                "architecture",
                "cohort_size",
                "n_runs",
                "runtime_mean_s",
                "runtime_ci95_lo",
                "runtime_ci95_hi",
                "completed_per_1000_mean",
                "completed_per_1000_ci95_lo",
                "completed_per_1000_ci95_hi",
                "unmet_ratio_mean",
                "unmet_ratio_ci95_lo",
                "unmet_ratio_ci95_hi",
                "next_avg_trust_mean",
                "next_avg_trust_ci95_lo",
                "next_avg_trust_ci95_hi",
                "next_avg_patience_mean",
                "next_avg_patience_ci95_lo",
                "next_avg_patience_ci95_hi",
                "high_trust_cycle_share_mean",
                "high_trust_cycle_share_ci95_lo",
                "high_trust_cycle_share_ci95_hi",
            ),
        )
    )
    report.append("")
    report.append("## Pairwise Tests (Partitioned minus Global)")
    report.append("")
    report.append(
        _format_table(
            test_rows,
            (
                "regime",
                "cohort_size",
                "metric",
                "global_mean",
                "partitioned_mean",
                "delta_partition_minus_global",
                "cliffs_delta_global_vs_partitioned",
                "permutation_pvalue",
            ),
        )
    )
    report.append("")
    report.append("## Output Files")
    report.append("")
    report.append(f"- `{outdir / 'raw_runs.csv'}`")
    report.append(f"- `{outdir / 'group_summary.csv'}`")
    report.append(f"- `{outdir / 'pairwise_tests.csv'}`")
    report.append(f"- `{outdir / 'rigor_summary.json'}`")
    report.append(f"- `{outdir / 'rigor_report.md'}`")
    report.append(f"- `{outdir / 'rigor_report.html'}`")
    return "\n".join(report)


def _build_html_report(markdown_text: str) -> str:
    # Simple markdown-to-html for headings/lists/code/table without external deps.
    lines = markdown_text.splitlines()
    out = []
    out.append("<!doctype html><html><head><meta charset='utf-8'><title>ShareWith Rigor Report</title>")
    out.append(
        "<style>body{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;max-width:1200px;margin:24px auto;padding:0 16px;color:#1d2428;background:#f7f8f6;}h1,h2{font-family:ui-sans-serif,system-ui;}"
        "table{border-collapse:collapse;width:100%;font-size:12px;background:#fff;}th,td{border:1px solid #d0d6d9;padding:6px;text-align:left;}"
        "th{background:#ecf0f2;}code{background:#eef3f5;padding:1px 4px;border-radius:4px;}</style></head><body>"
    )
    in_table = False
    table_header_done = False
    for line in lines:
        if line.startswith("# "):
            out.append(f"<h1>{html.escape(line[2:])}</h1>")
            continue
        if line.startswith("## "):
            out.append(f"<h2>{html.escape(line[3:])}</h2>")
            continue
        if line.startswith("- "):
            out.append(f"<p>{html.escape(line)}</p>")
            continue
        if line.startswith("| ") and line.endswith(" |"):
            cells = [c.strip() for c in line[2:-2].split("|")]
            if not in_table:
                out.append("<table>")
                in_table = True
                table_header_done = False
            if all(c == "---" for c in cells):
                continue
            if not table_header_done:
                out.append("<tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in cells) + "</tr>")
                table_header_done = True
            else:
                out.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</table>")
            in_table = False
            table_header_done = False
        if line.strip():
            out.append(f"<p>{html.escape(line)}</p>")
        else:
            out.append("<br>")
    if in_table:
        out.append("</table>")
    out.append("</body></html>")
    return "\n".join(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Academic-style rigor experiment harness for ShareWith.")
    parser.add_argument("--sizes-structured", default="600,1200,1800,2400", help="Comma-separated cohort sizes for structured regime.")
    parser.add_argument("--sizes-messy", default="400,800,1200", help="Comma-separated cohort sizes for messy regime.")
    parser.add_argument("--repetitions", type=int, default=5, help="Repetitions per condition.")
    parser.add_argument("--partition-cell-size", type=int, default=250, help="Cell size for partitioned architecture.")
    parser.add_argument("--bootstrap-samples", type=int, default=2000, help="Bootstrap samples for CI.")
    parser.add_argument("--permutations", type=int, default=5000, help="Permutation samples for tests.")
    parser.add_argument("--base-seed", type=int, default=20260216, help="Base seed for run scheduling.")
    parser.add_argument("--outdir", type=Path, default=ROOT / "artifacts" / "academic")
    return parser.parse_args()


def _parse_sizes(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    args = parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    sizes_structured = _parse_sizes(args.sizes_structured)
    sizes_messy = _parse_sizes(args.sizes_messy)

    specs: List[RunSpec] = []
    seed_counter = 0
    for regime, sizes in (("structured", sizes_structured), ("messy", sizes_messy)):
        for cohort_size in sizes:
            for architecture in ("global", "partitioned"):
                for repetition in range(args.repetitions):
                    specs.append(
                        RunSpec(
                            regime=regime,
                            architecture=architecture,
                            cohort_size=cohort_size,
                            repetition=repetition,
                            seed=args.base_seed + seed_counter,
                        )
                    )
                    seed_counter += 1

    print(f"Planned runs: {len(specs)}")
    rows: List[Dict[str, Any]] = []
    for idx, spec in enumerate(specs, start=1):
        print(
            f"[{idx}/{len(specs)}] regime={spec.regime} arch={spec.architecture} "
            f"cohort={spec.cohort_size} rep={spec.repetition} seed={spec.seed}"
        )
        row = _run_spec(spec, cell_size=args.partition_cell_size)
        rows.append(row)
        print(
            f"  runtime={row['runtime_s']:.3f}s completed_per_1000={row['completed_per_1000']:.3f} "
            f"unmet_ratio={row['unmet_ratio']:.4f}"
        )

    rng = random.Random(args.base_seed + 99)
    summary_rows = _summarize_groups(rows, bootstrap_samples=args.bootstrap_samples, rng=rng)
    test_rows = _pairwise_tests(rows, permutations=args.permutations, rng=rng)

    _write_csv(outdir / "raw_runs.csv", rows)
    _write_csv(outdir / "group_summary.csv", summary_rows)
    _write_csv(outdir / "pairwise_tests.csv", test_rows)

    summary_payload = {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "args": {
            "sizes_structured": sizes_structured,
            "sizes_messy": sizes_messy,
            "repetitions": args.repetitions,
            "partition_cell_size": args.partition_cell_size,
            "bootstrap_samples": args.bootstrap_samples,
            "permutations": args.permutations,
            "base_seed": args.base_seed,
        },
        "rawRunsPath": str(outdir / "raw_runs.csv"),
        "groupSummaryPath": str(outdir / "group_summary.csv"),
        "pairwiseTestsPath": str(outdir / "pairwise_tests.csv"),
        "groupSummary": summary_rows,
        "pairwiseTests": test_rows,
    }
    (outdir / "rigor_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    report_md = _build_markdown_report(outdir=outdir, summary_rows=summary_rows, test_rows=test_rows, args=args)
    (outdir / "rigor_report.md").write_text(report_md, encoding="utf-8")
    (outdir / "rigor_report.html").write_text(_build_html_report(report_md), encoding="utf-8")

    print("Academic rigor experiment complete.")
    print(f"Output directory: {outdir}")
    for name in ("raw_runs.csv", "group_summary.csv", "pairwise_tests.csv", "rigor_summary.json", "rigor_report.md", "rigor_report.html"):
        print(f" - {outdir / name}")


if __name__ == "__main__":
    main()
