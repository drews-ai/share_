# Spice Run: Thesis Need/Want Ideation (2026-02-19)

## What changed

Implemented thesis-aligned declaration ideation in the wave simulator:

- Multi-contract declarations per agent (up to configurable cap)
- Need decomposition templates (bathroom repair, caregiving stack, career transition, vacation deconstruction)
- Cross-context want sampling for adjacent contexts
- Profiled declaration modes (`none`, `thesis_ideation`, `thesis_spice`)

Key code paths:

- `/Users/drewprescott/Desktop/share_labor_food/sharing/wave_tessellation_fulltilt.py`
- `/Users/drewprescott/Desktop/share_labor_food/sharing/wave_cde_benchmark.py`
- `/Users/drewprescott/Desktop/share_labor_food/sharing/test_wave_thesis_tdd.py`

## Experimental setup

Controlled A/B benchmarks on same cohort and seeds:

- Grid: 3x3
- Agents per tessellation: 1500 (13,500 total)
- Weeks: 4
- Active rate: 0.02
- Panel mode: fixed
- Seeds: 6
- Variants: C and D

Runs:

1. Baseline declarations (`spice_profile=none`)
2. Full thesis ideation (`spice_profile=thesis_ideation`)
3. Tuned ideation (`max_offers=3`, `max_wants=2`, `decomposition=0.24`, `cross_context=0.09`)

## Results summary

### Baseline (`spice_profile=none`)

- C: completed/1000 = 178.24, unmet = 0.5560, runtime/week = 0.313s, objective = 153.45
- D: completed/1000 = 200.15, unmet = 0.5100, runtime/week = 0.277s, objective = 172.59

### Full thesis ideation (`spice_profile=thesis_ideation`)

- C: completed/1000 = 77.78, unmet = 0.8837, runtime/week = 2.529s, objective = 63.62
- D: completed/1000 = 99.85, unmet = 0.8483, runtime/week = 2.320s, objective = 82.75

### Tuned ideation (`thesis_ideation` + overrides)

- C: completed/1000 = 100.15, unmet = 0.8177, runtime/week = 1.450s, objective = 83.98
- D: completed/1000 = 115.74, unmet = 0.7925, runtime/week = 1.360s, objective = 97.51

## Interpretation

1. D still beats C inside every declaration regime.
2. Directly turning on thesis-style decomposition increases demand graph complexity and raises unmet demand substantially.
3. Runtime increases are expected: more declarations produce larger node sets and assignment matrices.
4. Tuned ideation improves over full ideation, but remains materially below baseline because generated demand complexity outruns available compatible supply under current matching assumptions.

## Why this happens

The ideation layer is doing what it should (more complex needs), but two additional thesis components are still missing in the active code path:

- Predictive staging / supply seeding for decomposed bundles
- Explicit bundle-level orchestration policy (prioritization rules for multi-cycle need packs)

Without these, decomposition mostly inflates wants faster than feasible cycle closure.

## Artifact paths

- Baseline benchmark:
  - `/Users/drewprescott/Desktop/share_labor_food/sharing/artifacts/wave_spice_compare_baseline`
- Full thesis ideation benchmark:
  - `/Users/drewprescott/Desktop/share_labor_food/sharing/artifacts/wave_spice_compare_thesis_ideation`
- Tuned ideation benchmark:
  - `/Users/drewprescott/Desktop/share_labor_food/sharing/artifacts/wave_spice_compare_tuned_v2`

## Recommendation

Use this as current operating sequence:

1. Keep `variant=D` for core matching quality.
2. Keep ideation in `tuned` mode for research (not default production).
3. Next implementation should be bundle-aware supply staging before increasing decomposition rate further.
