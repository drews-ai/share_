# ShareWith Academic Rigor Synthesis (2026-02-17)

## 1) Scope
This document consolidates three reproducible experiment suites:

1. `artifacts/academic` (70 runs): structured and messy cohorts up to 2400 agents.
2. `artifacts/academic_large` (42 runs): large-cohort sweep up to 10000 structured and 2000 messy agents.
3. `artifacts/academic_trust_patient` (18 runs): messy 1200/1600/2000 with trust/patience metrics.

All runs used fixed seeds, bootstrap confidence intervals, and permutation tests.

## 2) Thesis-Intent Alignment Check
Primary thesis intent is preserved where it matters computationally:

- Tessellation/partitioning as core architecture: `THESIS.md` states complexity reduction via tessellation and explicitly says nodes should not merge into a global graph (`THESIS.md:249`, `THESIS.md:296`, `THESIS.md:299`).
- Trust + patience weighting: equations and gating in thesis are represented by current edge admissibility and weighting (`THESIS.md:220`, `THESIS.md:360`).
- Context-aware operation is present (`SURVIVAL`, `SAFETY`, `SOCIAL`, `GROWTH`, `LUXURY`).

Known deviation and caveat:

- Thesis-era code still used explicit cycle enumeration limits (3-cycles, 4/5 disabled for performance), so historical implementation was not purely idealized at scale (`backend/simulation/matching.py:420`, `backend/simulation/matching.py:459`).

## 3) Key Results (Large Cohorts)
From `artifacts/academic_large/group_summary.csv`:

### Structured regime (3000-10000)
- Partitioning runtime speedup: `11.6x` (3000) to `39.8x` (10000).
- Outcome tradeoff: small throughput reduction (`-1.833` to `-4.667` completed cycles per 1000) and small unmet increase (`+0.0055` to `+0.0140`).

Interpretation: in clean, reciprocal markets, global matching slightly improves exchange quality, but partitioning gives major tractability gains.

### Messy regime (1200-2000)
- Partitioning runtime speedup: `14.7x` (1200) to `28.1x` (2000).
- Outcome gain: completion improves by `+15.0` to `+22.083` per 1000; unmet drops by `-0.0389` to `-0.0615`.

Interpretation: in noisy markets, partitioning is not only faster; it improves substantive outcomes.

## 4) Trust/Patient-First Results (Not Speed-First)
From `artifacts/academic_trust_patient/pairwise_tests.csv`:

### n=1200 (messy)
- completed/1000: `6.667 -> 23.889` (`+17.222`)
- unmet ratio: `0.9819 -> 0.9386` (`-0.0433`)
- next avg patience: `2.9442 -> 2.8750` (`-0.0692`)
- next avg trust: `0.7696 -> 0.7721` (`+0.0025`)
- high-trust cycle share: `0.6444 -> 0.5349` (`-0.1095`)

### n=1600 (messy)
- completed/1000: `5.625 -> 24.375` (`+18.750`)
- unmet ratio: `0.9854 -> 0.9354` (`-0.0500`)
- next avg patience: `2.9423 -> 2.8283` (`-0.1140`)
- next avg trust: `0.7709 -> 0.7700` (`-0.0009`)
- high-trust cycle share: `0.6884 -> 0.5661` (`-0.1223`)

### n=2000 (messy)
- completed/1000: `9.000 -> 22.000` (`+13.000`)
- unmet ratio: `0.9738 -> 0.9372` (`-0.0367`)
- next avg patience: `2.9222 -> 2.8277` (`-0.0945`)
- next avg trust: `0.7686 -> 0.7685` (`-0.0001`)
- high-trust cycle share: `0.5980 -> 0.5286` (`-0.0694`)

Interpretation:
- Patient-first objective improved: less waiting and fewer unmet wants.
- Trust level remained approximately flat (no trust collapse).
- But high-trust cycle share dropped, suggesting wider inclusion at the cost of concentration in top-trust cliques.

## 5) What Is Good vs Bad
Good:
- Reproducible behavior across seeds.
- Strong evidence that tessellation architecture is necessary in large noisy cohorts.
- Patient-first outcomes improve materially in messy conditions with partitioning.

Bad / unresolved:
- In structured clean markets, partitioning gives a small outcome penalty.
- High-trust cycle share decreases under partitioning in messy markets.
- Large-cohort p-values are coarse because the 10000/2000 sweeps used only 3 repetitions (effect sizes remain strong).

## 6) Concrete Next Optimization Targets
1. Trust-preserving partition boundary policy:
- Add cross-cell candidate lane for high-trust edges only (no global merge), preserving thesis principle.
- Promote boundary edges when both endpoints exceed trust thresholds and patience is above context-specific cutoffs.

2. Two-stage objective:
- Stage A: minimize unmet and patience drift.
- Stage B: maximize high-trust cycle share subject to Stage A constraints.

3. Adaptive cell-size policy:
- Use smaller cells in noisy/high-friction regions, larger cells in structured/high-reciprocity regions.

4. Statistical hardening:
- Repeat high-cost large-cohort sweeps at `n_runs >= 5` for tighter inference.

## 7) Artifact Index
- Core script: `academic_rigor_lab.py`
- Figure script: `render_rigor_figures.py`
- Reports:
  - `artifacts/academic/rigor_report.md`
  - `artifacts/academic_large/rigor_report.md`
  - `artifacts/academic_trust_patient/rigor_report.md`
- Figures:
  - `artifacts/academic_large/runtime_scaling.png`
  - `artifacts/academic_large/outcome_quality.png`
  - `artifacts/academic_large/pairwise_effects.png`
  - `artifacts/academic_trust_patient/runtime_scaling.png`
  - `artifacts/academic_trust_patient/outcome_quality.png`
  - `artifacts/academic_trust_patient/pairwise_effects.png`
