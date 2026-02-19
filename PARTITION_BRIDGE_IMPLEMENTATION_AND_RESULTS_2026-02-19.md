# Partition-Bridge + Orchestration Implementation (2026-02-19)

## What was built

Implemented in `/Users/drewprescott/Desktop/share_labor_food/sharing/wave_tessellation_fulltilt.py`:

1. New matching mode: `partition_bridge`
- Stage 1: local-only matching (`hop == 0`)
- Stage 2: adjacent bridge matching (`0 < hop <= bridge_hop`)
- Stage 3: fallback rounds on remaining declarations (`fallback_rounds`)

2. Execution orchestration simulation
- Per-stage confirmation, atomic commit, and execution simulation
- Committed declarations are consumed before next stage
- Fallback stage retries unresolved declarations in the same weekly cycle

3. Stage-level metrics now emitted
- `localCycleCount`
- `bridgeCycleCount`
- `fallbackCycleCount`
- `fallbackRoundsUsed`
- `matchingMode`

4. New Kontur-backed benchmark runner
- `/Users/drewprescott/Desktop/share_labor_food/sharing/kontur_partition_bridge_benchmark.py`
- Reads contiguous H3 cell sets from:
  - `/Users/drewprescott/Desktop/old_but_good/sharewith.ai/CLEAN_PACKAGE/data/kontur_population/kontur_population_US_20231101.gpkg`
- Builds population-weighted cohorts and runs wave simulation in `global` or `partition_bridge` mode

## TDD validation

Executed:
- `python -m unittest -v test_wave_thesis_tdd.py` -> **12/12 pass**
- `python -m unittest -v test_sharewith_engine.py` -> **6/6 pass**

## Benchmark results

### A/B on synthetic 27k cohort (same seed/config)

Commands:
- global: `wave_tessellation_fulltilt.py ... --matching-mode global`
- partition: `wave_tessellation_fulltilt.py ... --matching-mode partition_bridge --bridge-hop 1 --fallback-rounds 1`

From:
- `/Users/drewprescott/Desktop/share_labor_food/sharing/artifacts/wave_partition_ablate/global_27k/summary.json`
- `/Users/drewprescott/Desktop/share_labor_food/sharing/artifacts/wave_partition_ablate/partition_bridge_27k/summary.json`

| Metric | Global | Partition-Bridge | Delta (PB - Global) |
|---|---:|---:|---:|
| mean_week_runtime_s | 1.4645 | 0.9832 | -0.4813 |
| mean_completed_per_1000_active | 158.3333 | 224.5370 | +66.2037 |
| mean_unmet_ratio | 0.5870 | 0.4111 | -0.1759 |
| mean_local_cycle_count | 101.75 | 128.50 | +26.75 |
| mean_bridge_cycle_count | 0.00 | 3.50 | +3.50 |
| mean_fallback_cycle_count | 0.00 | 16.00 | +16.00 |

### Kontur-backed large run (36,267 total agents, 2,901 active/week, 3 weeks)

From:
- `/Users/drewprescott/Desktop/share_labor_food/sharing/artifacts/kontur_partition_bridge/large_pb/summary.json`

| Metric | Value |
|---|---:|
| matching_mode | partition_bridge |
| mean_week_runtime_s | 28.5846 |
| mean_completed_per_1000_active | 160.5194 |
| mean_unmet_ratio | 0.6652 |
| mean_cycle_survival | 0.8234 |
| mean_local_cycle_count | 256.33 |
| mean_bridge_cycle_count | 243.00 |
| mean_fallback_cycle_count | 66.33 |

### Kontur matched one-week A/B (36,267 total agents, 1,088 active)

From:
- PB: `/Users/drewprescott/Desktop/share_labor_food/sharing/artifacts/kontur_partition_bridge/large_pb_w1_ar003/summary.json`
- Global: `/Users/drewprescott/Desktop/share_labor_food/sharing/artifacts/kontur_partition_bridge/large_global_w1_ar003/summary.json`

| Metric | Global | Partition-Bridge | Delta (PB - Global) |
|---|---:|---:|---:|
| mean_week_runtime_s | 92.7755 | 0.1142 | -92.6613 |
| mean_completed_per_1000_active | 80.8824 | 93.7500 | +12.8676 |
| mean_unmet_ratio | 0.8336 | 0.8097 | -0.0239 |
| mean_local_cycle_count | 100.0 | 55.0 | -45.0 |
| mean_bridge_cycle_count | 0.0 | 56.0 | +56.0 |
| mean_fallback_cycle_count | 0.0 | 8.0 | +8.0 |

### Full-tilt Kontur run (76,057 total agents, 3,802 active, 1 week)

From:
- `/Users/drewprescott/Desktop/share_labor_food/sharing/artifacts/kontur_partition_bridge/fulltilt_pb_w1/summary.json`

| Metric | Value |
|---|---:|
| matching_mode | partition_bridge |
| mean_week_runtime_s | 65.9213 |
| mean_completed_per_1000_active | 157.8117 |
| mean_unmet_ratio | 0.6681 |
| mean_cycle_survival | 0.8439 |
| mean_local_cycle_count | 377.0 |
| mean_bridge_cycle_count | 287.0 |
| mean_fallback_cycle_count | 47.0 |

## Operational note

A 3-week global run on the 36,267-agent Kontur cohort at 8% active slice was terminated after prolonged CPU saturation; this is consistent with global solve complexity pressure versus staged partitioning.
