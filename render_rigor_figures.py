#!/usr/bin/env python3
"""
Render academic-style figures from artifacts/academic CSV outputs.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DEFAULT_ART = ROOT / "artifacts" / "academic"


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def _f(value: str) -> float:
    return float(value)


def _i(value: str) -> int:
    return int(float(value))


def _group_summary(rows: List[Dict[str, Any]], regime: str, architecture: str) -> List[Dict[str, Any]]:
    filtered = [r for r in rows if r["regime"] == regime and r["architecture"] == architecture]
    filtered.sort(key=lambda r: _i(r["cohort_size"]))
    return filtered


def _plot_runtime(summary_rows: List[Dict[str, Any]], outdir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
    fig.suptitle("ShareWith Runtime Scaling (95% Bootstrap CI)", fontsize=13, fontweight="bold")

    for ax, regime in zip(axes, ("structured", "messy")):
        for arch, color in (("global", "#c44e52"), ("partitioned", "#4c72b0")):
            rows = _group_summary(summary_rows, regime, arch)
            x = [_i(r["cohort_size"]) for r in rows]
            y = [_f(r["runtime_s_mean"]) for r in rows]
            ylo = [_f(r["runtime_s_ci95_lo"]) for r in rows]
            yhi = [_f(r["runtime_s_ci95_hi"]) for r in rows]
            err = [[yy - lo for yy, lo in zip(y, ylo)], [hi - yy for yy, hi in zip(y, yhi)]]
            ax.errorbar(x, y, yerr=err, marker="o", linewidth=2.0, capsize=4, label=arch, color=color)
        ax.set_title(f"{regime.capitalize()} regime")
        ax.set_xlabel("Cohort size")
        ax.set_ylabel("Runtime (s)")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = outdir / "runtime_scaling.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _plot_outcomes(summary_rows: List[Dict[str, Any]], outdir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    fig.suptitle("ShareWith Outcome Quality by Architecture", fontsize=13, fontweight="bold")

    metric_specs = (
        ("completed_per_1000", "Completed cycles per 1000", "#55a868"),
        ("unmet_ratio", "Unmet demand ratio", "#8172b2"),
    )
    regimes = ("structured", "messy")

    for row_idx, (metric, ylabel, color_base) in enumerate(metric_specs):
        for col_idx, regime in enumerate(regimes):
            ax = axes[row_idx][col_idx]
            for arch, color in (("global", color_base), ("partitioned", "#dd8452" if color_base == "#55a868" else "#937860")):
                rows = _group_summary(summary_rows, regime, arch)
                x = [_i(r["cohort_size"]) for r in rows]
                y = [_f(r[f"{metric}_mean"]) for r in rows]
                lo = [_f(r[f"{metric}_ci95_lo"]) for r in rows]
                hi = [_f(r[f"{metric}_ci95_hi"]) for r in rows]
                err = [[yy - ll for yy, ll in zip(y, lo)], [hh - yy for yy, hh in zip(y, hi)]]
                ax.errorbar(x, y, yerr=err, marker="o", linewidth=2.0, capsize=4, label=arch, color=color)
            ax.set_title(f"{regime.capitalize()} regime")
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.35)
            if row_idx == len(metric_specs) - 1:
                ax.set_xlabel("Cohort size")
            if row_idx == 0 and col_idx == 1:
                ax.legend(frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = outdir / "outcome_quality.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _plot_effect_sizes(pair_rows: List[Dict[str, Any]], outdir: Path) -> Path:
    rows = [r for r in pair_rows if r["metric"] in ("runtime_s", "completed_per_1000", "unmet_ratio")]
    rows.sort(key=lambda r: (r["regime"], _i(r["cohort_size"]), r["metric"]))

    labels = [f"{r['regime']} n={_i(r['cohort_size'])}\n{r['metric']}" for r in rows]
    deltas = [_f(r["delta_partition_minus_global"]) for r in rows]
    pvals = [_f(r["permutation_pvalue"]) for r in rows]

    colors = []
    for r in rows:
        metric = r["metric"]
        if metric == "runtime_s":
            colors.append("#4c72b0")
        elif metric == "completed_per_1000":
            colors.append("#55a868")
        else:
            colors.append("#c44e52")

    fig, ax = plt.subplots(figsize=(14, 5.6))
    bars = ax.bar(range(len(labels)), deltas, color=colors, alpha=0.9)
    ax.axhline(0.0, color="#444", linewidth=1.0)
    ax.set_title("Partitioned minus Global: Effect Direction and Magnitude", fontsize=13, fontweight="bold")
    ax.set_ylabel("Delta (partitioned - global)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    for bar, p in zip(bars, pvals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"p={p:.4f}",
            ha="center",
            va="bottom" if bar.get_height() >= 0 else "top",
            fontsize=7,
            rotation=90,
        )

    fig.tight_layout()
    out = outdir / "pairwise_effects.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render rigor figures from CSV outputs.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_ART, help="Directory containing group_summary.csv and pairwise_tests.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    summary_rows = _load_csv(outdir / "group_summary.csv")
    pair_rows = _load_csv(outdir / "pairwise_tests.csv")

    generated = [
        _plot_runtime(summary_rows, outdir=outdir),
        _plot_outcomes(summary_rows, outdir=outdir),
        _plot_effect_sizes(pair_rows, outdir=outdir),
    ]
    print("Generated figures:")
    for path in generated:
        print(f" - {path}")


if __name__ == "__main__":
    main()
