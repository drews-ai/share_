# ShareWith Arena Benchmark Report

Generated from deterministic simulation runs in:
`/Users/drewprescott/Desktop/share_labor_food/sharing/artifacts`

## What Was Changed

- Tuned default social distance gate in `/Users/drewprescott/Desktop/share_labor_food/sharing/server.py`:
  - `context_base_distance.SOCIAL` changed from `3.0` to `10.0`.
- Hardened ranking API input handling:
  - `/api/rank` now accepts string-style pain payloads without crashing.
  - `rank_solutions()` now handles both dict pains and string pains.
- Expanded optimizer search space:
  - Added sampling for `context_base_distance`, `context_upsilon`, and `min_receiver_completion`.
- Added UI compatibility metric alias:
  - `metrics.unmatchedWants` mirrors `metrics.unmatchedWantsAfterMatch`.
- Added reproducible benchmark harness:
  - `/Users/drewprescott/Desktop/share_labor_food/sharing/arena_benchmark.py`
- Added smoke/regression tests:
  - `/Users/drewprescott/Desktop/share_labor_food/sharing/test_sharewith_engine.py`

## Proof Runs Executed

### Test run

Command:
```bash
python3 -m unittest -v /Users/drewprescott/Desktop/share_labor_food/sharing/test_sharewith_engine.py
```

Result: `Ran 4 tests ... OK`

### Heavy benchmark run

Command:
```bash
python3 /Users/drewprescott/Desktop/share_labor_food/sharing/arena_benchmark.py --seeds 320 --opt-budget 60 --opt-seeds 10
```

Run volume:
- 3,200 scenario runs across config variants
- 5 optimizer sweeps (one per scenario)

## Core Results

Global objective mean (higher is better):
- `legacy_social_3`: `-7.279`
- `tuned_default`: `-1.845`
- Net improvement: `+5.434`

Scenario-level objective mean (`legacy -> tuned`):
- `balanced_local`: `-7.100 -> 4.147`
- `choice_friction`: `-7.300 -> -1.393`
- `atomic_dependency`: `-11.600 -> -1.586`
- `trust_coldstart`: `-9.900 -> -9.900` (unchanged bottleneck)
- `mesh_pressure`: `-0.495 -> -0.495` (no net change under this specific tuning)

Scenario-level completed cycles mean (`legacy -> tuned`):
- `balanced_local`: `0.000 -> 0.706`
- `choice_friction`: `0.000 -> 0.350`
- `atomic_dependency`: `0.000 -> 0.703`
- `trust_coldstart`: `0.000 -> 0.000`
- `mesh_pressure`: `1.219 -> 1.219`

## Optimizer Outcome (Best - Baseline Objective)

- `balanced_local`: `+1.444` (candidate `7`)
- `choice_friction`: `+5.168` (candidate `37`)
- `atomic_dependency`: `+8.822` (candidate `8`)
- `trust_coldstart`: `+0.000` (candidate `0`)
- `mesh_pressure`: `+17.444` (candidate `8`)

## Artifacts

- Dashboard image:
  - `/Users/drewprescott/Desktop/share_labor_food/sharing/artifacts/benchmark_dashboard.png`
- Full run CSV:
  - `/Users/drewprescott/Desktop/share_labor_food/sharing/artifacts/benchmark_runs.csv`
- Aggregated scorecard CSV:
  - `/Users/drewprescott/Desktop/share_labor_food/sharing/artifacts/benchmark_scorecard.csv`
- Benchmark summary JSON:
  - `/Users/drewprescott/Desktop/share_labor_food/sharing/artifacts/benchmark_summary.json`
- Optimizer summaries:
  - `/Users/drewprescott/Desktop/share_labor_food/sharing/artifacts/optimizer_summary.csv`
  - `/Users/drewprescott/Desktop/share_labor_food/sharing/artifacts/optimizer_summary.json`

## Remaining Hard Pain Point

- `trust_coldstart` remains unresolved by parameter search.
- This indicates a structural data/geometry bottleneck (distance + trust cold-start), not just local parameter tuning.
- Next model iteration should add one or more of:
  - staged trust bootstrapping,
  - relay/bridge cycles,
  - context-aware long-distance exceptions for SURVIVAL/SAFETY with escrow-style safeguards.
