#!/usr/bin/env python3
"""
Kontur-backed benchmark for partition-first + bridge matching.

Builds an agent cohort from contiguous H3 cells in the Kontur population GPKG,
then runs weekly wave simulations using either:
- global matching
- partition_bridge matching (local pass + bridge pass + fallback rounds)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import h3

import server
import wave_tessellation_fulltilt as wave


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class KonturConfig:
    gpkg_path: Path
    outdir: Path
    seed: int
    weeks: int
    active_rate: float
    panel_mode: str
    variant: str
    max_hop: int
    patience_per_hop: int
    matching_mode: str
    bridge_hop: int
    fallback_rounds: int
    anchor_radius: int
    max_cells: int
    min_population: int
    agents_per_1000_pop: float
    min_agents_per_cell: int
    max_agents_per_cell: int
    max_total_agents: int
    params_overrides_file: Path | None


SKILL_POOLS: Dict[server.Context, Tuple[str, ...]] = wave.SKILL_POOLS


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _chunked(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _fetch_population_by_h3(conn: sqlite3.Connection, h3_cells: Sequence[str]) -> Dict[str, int]:
    if not h3_cells:
        return {}

    out: Dict[str, int] = {}
    for chunk in _chunked(list(h3_cells), 800):
        placeholders = ",".join("?" for _ in chunk)
        sql = f"SELECT h3, population FROM population WHERE h3 IN ({placeholders})"
        for h3_index, pop in conn.execute(sql, tuple(chunk)):
            out[str(h3_index)] = int(round(float(pop)))
    return out


def _load_contiguous_cells(cfg: KonturConfig) -> Dict[str, int]:
    conn = sqlite3.connect(str(cfg.gpkg_path))
    try:
        row = conn.execute(
            "SELECT h3, population FROM population WHERE population >= ? ORDER BY population DESC LIMIT 1",
            (cfg.min_population,),
        ).fetchone()
        if row is None:
            raise RuntimeError("No anchor cell found above configured min_population.")
        anchor = str(row[0])

        ring = h3.grid_disk(anchor, cfg.anchor_radius)
        population_map = _fetch_population_by_h3(conn, list(ring))
        population_map = {h: p for h, p in population_map.items() if p >= cfg.min_population}

        if len(population_map) > cfg.max_cells:
            ranked = sorted(population_map.items(), key=lambda kv: kv[1], reverse=True)
            population_map = dict(ranked[: cfg.max_cells])

        if not population_map:
            raise RuntimeError("No contiguous cells met constraints. Increase anchor_radius or lower min_population.")

        return population_map
    finally:
        conn.close()


def _coords_from_h3_cells(h3_cells: Sequence[str]) -> Tuple[Dict[str, Tuple[int, int]], wave.CellConfig]:
    anchor = h3_cells[0]
    raw_coords: Dict[str, Tuple[int, int]] = {}
    for h3_index in h3_cells:
        try:
            ij = h3.cell_to_local_ij(anchor, h3_index)
            raw_coords[h3_index] = (int(ij[0]), int(ij[1]))
        except Exception:
            lat, lng = h3.cell_to_latlng(h3_index)
            raw_coords[h3_index] = (int(round(lng * 200.0)), int(round(lat * 200.0)))

    min_x = min(x for x, _ in raw_coords.values())
    min_y = min(y for _, y in raw_coords.values())
    normalized = {h: (x - min_x, y - min_y) for h, (x, y) in raw_coords.items()}

    max_x = max(x for x, _ in normalized.values())
    max_y = max(y for _, y in normalized.values())
    grid = wave.CellConfig(width=max_x + 1, height=max_y + 1)
    return normalized, grid


def _declaration_pair(rng: random.Random, context: server.Context) -> Tuple[List[server.Offer], List[server.Want]]:
    pool = SKILL_POOLS[context]
    pidx = rng.randrange(len(pool))
    widx = (pidx + rng.randint(1, len(pool) - 1)) % len(pool)
    offer_skill = pool[pidx]
    want_skill = pool[widx]
    return (
        [server.Offer(skill=offer_skill, context=context, tags=(offer_skill, context.value.lower()))],
        [server.Want(skill=want_skill, context=context, tags=(want_skill, context.value.lower()))],
    )


def _build_agents_from_kontur(
    cfg: KonturConfig,
    population_map: Dict[str, int],
) -> Tuple[List[server.Agent], Dict[str, Tuple[int, int]], wave.CellConfig]:
    rng = random.Random(cfg.seed)
    h3_cells = sorted(population_map.keys())
    cell_coords, grid = _coords_from_h3_cells(h3_cells)

    contexts = (
        server.Context.SAFETY,
        server.Context.SOCIAL,
        server.Context.GROWTH,
        server.Context.SURVIVAL,
        server.Context.LUXURY,
    )
    context_weights = (0.33, 0.24, 0.2, 0.16, 0.07)

    per_cell_counts: Dict[str, int] = {}
    for h3_index, pop in population_map.items():
        raw = int(round((pop / 1000.0) * cfg.agents_per_1000_pop))
        count = max(cfg.min_agents_per_cell, raw)
        count = min(count, cfg.max_agents_per_cell)
        per_cell_counts[h3_index] = count

    total_agents = sum(per_cell_counts.values())
    if cfg.max_total_agents > 0 and total_agents > cfg.max_total_agents:
        scale = cfg.max_total_agents / max(total_agents, 1)
        for h3_index, count in list(per_cell_counts.items()):
            per_cell_counts[h3_index] = max(cfg.min_agents_per_cell, int(round(count * scale)))

    agents: List[server.Agent] = []
    home_cell: Dict[str, Tuple[int, int]] = {}
    aid = 0

    for h3_index in h3_cells:
        cx, cy = cell_coords[h3_index]
        for _ in range(per_cell_counts[h3_index]):
            context = rng.choices(contexts, weights=context_weights)[0]
            offers, wants = _declaration_pair(rng, context=context)
            agent_id = f"k{aid}"
            agent = server.Agent(
                agent_id=agent_id,
                name=f"K{aid}",
                mode=server.Mode.CHOICE if rng.random() < 0.22 else server.Mode.AUTO,
                offers=offers,
                wants=wants,
                location=server.Location(x=cx + rng.uniform(0.1, 0.9), y=cy + rng.uniform(0.1, 0.9)),
                trust=server.TrustMetrics(
                    completion=max(0.35, min(0.99, rng.gauss(0.78, 0.1))),
                    quality=max(0.35, min(0.99, rng.gauss(0.8, 0.1))),
                ),
                patience=rng.randint(0, 2),
                gave=rng.randint(1, 20),
                received=rng.randint(1, 20),
            )
            agents.append(agent)
            home_cell[agent_id] = (cx, cy)
            aid += 1

    return agents, home_cell, grid


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(cfg: KonturConfig) -> Dict[str, Any]:
    print("Loading contiguous Kontur cells from GPKG...")
    population_map = _load_contiguous_cells(cfg)
    print(f"Loaded {len(population_map)} cells from {cfg.gpkg_path}.")

    print("Building agent cohort from population-weighted H3 cells...")
    agents, home_cell, grid = _build_agents_from_kontur(cfg, population_map)
    print(f"Built {len(agents)} agents on grid {grid.width}x{grid.height}.")

    params = server.DEFAULT_PARAMS
    if cfg.params_overrides_file is not None:
        payload = json.loads(cfg.params_overrides_file.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "params_overrides" in payload:
            overrides = payload.get("params_overrides")
        else:
            overrides = payload
        if not isinstance(overrides, dict):
            raise ValueError("params overrides file must contain object or {\"params_overrides\": {...}}")
        params = params.with_overrides(overrides)
    rng = random.Random(cfg.seed + 101)
    n_active = max(2, int(len(agents) * cfg.active_rate))
    fixed_indices: List[int] = rng.sample(range(len(agents)), n_active) if cfg.panel_mode == "fixed" else []
    trail_memory: Dict[wave.TrailKey, float] = {}

    weekly_rows: List[Dict[str, Any]] = []
    started = time.perf_counter()

    for week in range(1, cfg.weeks + 1):
        week_seed = cfg.seed + week * 1009
        if cfg.panel_mode == "fixed":
            active_indices = fixed_indices
        else:
            active_indices = rng.sample(range(len(agents)), n_active)
        active_agents = [agents[idx] for idx in active_indices]

        t0 = time.perf_counter()
        metrics, projection, matching, cycles, reach_by_agent = wave._run_wave_week(
            active_agents=active_agents,
            params=params,
            home_cell=home_cell,
            grid=grid,
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
            wave._update_trail_memory(trail_memory=trail_memory, cycles=cycles, home_cell=home_cell)
        wave._update_agents_from_projection(active_agents, projection)
        wave_stats = wave._cross_cell_stats(matching=matching, cycles=cycles, home_cell=home_cell)
        reaches = [reach_by_agent[a.agent_id] for a in active_agents]
        week_elapsed = time.perf_counter() - t0

        row = {
            "week": week,
            "week_runtime_s": week_elapsed,
            "active_agents": len(active_agents),
            "feasible_edges": metrics["feasibleEdges"],
            "cycle_count": metrics["cycleCount"],
            "completed_cycles": metrics["completedCycles"],
            "completed_per_1000_active": metrics["completedCycles"] * 1000.0 / len(active_agents),
            "unmet_ratio": metrics["unmetWantsAfterExecution"] / max(metrics["totalWants"], 1),
            "cycle_survival": metrics["cycleSurvival"],
            "next_avg_trust": metrics["nextAvgTrust"],
            "next_avg_patience": metrics["nextAvgPatience"],
            "high_trust_edge_share": metrics["highTrustEdgeShare"],
            "high_trust_cycle_share": metrics["highTrustCycleShare"],
            "matching_cross_share": wave_stats["matching_cross_share"],
            "completed_cross_share": wave_stats["completed_cross_share"],
            "avg_reach_hops": _mean(float(v) for v in reaches),
            "share_reach_ge1": sum(1 for v in reaches if v >= 1) / len(reaches),
            "local_cycle_count": metrics.get("localCycleCount", metrics["cycleCount"]),
            "bridge_cycle_count": metrics.get("bridgeCycleCount", 0),
            "fallback_cycle_count": metrics.get("fallbackCycleCount", 0),
            "fallback_rounds_used": metrics.get("fallbackRoundsUsed", 0),
        }
        weekly_rows.append(row)
        print(
            f"week={week:02d} runtime={week_elapsed:.2f}s active={len(active_agents)} "
            f"cycles={row['cycle_count']} completed/1000={row['completed_per_1000_active']:.2f} "
            f"unmet={row['unmet_ratio']:.3f}"
        )

    total_runtime_s = time.perf_counter() - started
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "gpkg_path": str(cfg.gpkg_path),
        "matching_mode": cfg.matching_mode,
        "variant": cfg.variant,
        "params_overrides_file": str(cfg.params_overrides_file) if cfg.params_overrides_file else "",
        "anchor_radius": cfg.anchor_radius,
        "cells_loaded": len(population_map),
        "total_agents": len(agents),
        "active_agents_per_week": n_active,
        "weeks": cfg.weeks,
        "mean_week_runtime_s": _mean(float(r["week_runtime_s"]) for r in weekly_rows),
        "total_runtime_s": total_runtime_s,
        "mean_completed_per_1000_active": _mean(float(r["completed_per_1000_active"]) for r in weekly_rows),
        "mean_unmet_ratio": _mean(float(r["unmet_ratio"]) for r in weekly_rows),
        "mean_cycle_survival": _mean(float(r["cycle_survival"]) for r in weekly_rows),
        "mean_matching_cross_share": _mean(float(r["matching_cross_share"]) for r in weekly_rows),
        "mean_completed_cross_share": _mean(float(r["completed_cross_share"]) for r in weekly_rows),
        "mean_local_cycle_count": _mean(float(r["local_cycle_count"]) for r in weekly_rows),
        "mean_bridge_cycle_count": _mean(float(r["bridge_cycle_count"]) for r in weekly_rows),
        "mean_fallback_cycle_count": _mean(float(r["fallback_cycle_count"]) for r in weekly_rows),
    }

    cfg.outdir.mkdir(parents=True, exist_ok=True)
    (cfg.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(cfg.outdir / "weekly_metrics.csv", weekly_rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Kontur-backed partition/bridge benchmark.")
    parser.add_argument(
        "--gpkg-path",
        type=Path,
        default=Path("/Users/drewprescott/Desktop/old_but_good/sharewith.ai/CLEAN_PACKAGE/data/kontur_population/kontur_population_US_20231101.gpkg"),
    )
    parser.add_argument("--outdir", type=Path, default=ROOT / "artifacts" / "kontur_partition_bridge")
    parser.add_argument("--seed", type=int, default=20260219)
    parser.add_argument("--weeks", type=int, default=4)
    parser.add_argument("--active-rate", type=float, default=0.03)
    parser.add_argument("--panel-mode", choices=("fixed", "resample"), default="fixed")
    parser.add_argument("--variant", choices=("C", "D", "E", "F"), default="D")
    parser.add_argument("--max-hop", type=int, default=4)
    parser.add_argument("--patience-per-hop", type=int, default=1)
    parser.add_argument("--matching-mode", choices=("global", "partition_bridge"), default="partition_bridge")
    parser.add_argument("--bridge-hop", type=int, default=1)
    parser.add_argument("--fallback-rounds", type=int, default=1)
    parser.add_argument("--anchor-radius", type=int, default=45)
    parser.add_argument("--max-cells", type=int, default=6000)
    parser.add_argument("--min-population", type=int, default=20)
    parser.add_argument("--agents-per-1000-pop", type=float, default=3.0)
    parser.add_argument("--min-agents-per-cell", type=int, default=1)
    parser.add_argument("--max-agents-per-cell", type=int, default=18)
    parser.add_argument("--max-total-agents", type=int, default=180000)
    parser.add_argument("--params-overrides-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = KonturConfig(
        gpkg_path=args.gpkg_path.resolve(),
        outdir=args.outdir.resolve(),
        seed=int(args.seed),
        weeks=int(args.weeks),
        active_rate=float(args.active_rate),
        panel_mode=args.panel_mode,
        variant=args.variant,
        max_hop=int(args.max_hop),
        patience_per_hop=int(args.patience_per_hop),
        matching_mode=args.matching_mode,
        bridge_hop=max(1, int(args.bridge_hop)),
        fallback_rounds=max(0, int(args.fallback_rounds)),
        anchor_radius=max(1, int(args.anchor_radius)),
        max_cells=max(10, int(args.max_cells)),
        min_population=max(1, int(args.min_population)),
        agents_per_1000_pop=max(0.05, float(args.agents_per_1000_pop)),
        min_agents_per_cell=max(1, int(args.min_agents_per_cell)),
        max_agents_per_cell=max(1, int(args.max_agents_per_cell)),
        max_total_agents=max(1000, int(args.max_total_agents)),
        params_overrides_file=args.params_overrides_file.resolve() if args.params_overrides_file else None,
    )
    summary = run(cfg)
    print("\nKontur benchmark complete.")
    print(f"Output: {cfg.outdir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
