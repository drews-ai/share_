# Thesis Claims End-to-End Evaluation (2026-02-19)

## Scope
Claim audit across thesis statements, thesis-era implementation, and active implementation.

Primary sources:
- /Users/drewprescott/Desktop/thesis-prod/share-prod/THESIS.md
- /Users/drewprescott/Desktop/thesis-prod/share-prod/SHARING_ALGORITHM.md
- /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/matching.py
- /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/trust_network.py
- /Users/drewprescott/Desktop/share_labor_food/sharing/server.py
- /Users/drewprescott/Desktop/share_labor_food/sharing/wave_tessellation_fulltilt.py
- /Users/drewprescott/Desktop/share_labor_food/sharing/test_sharewith_engine.py
- /Users/drewprescott/Desktop/share_labor_food/sharing/test_wave_thesis_tdd.py

## Validation Runs Executed
- Unit/invariant tests:
  - `python3 -m unittest discover -v` in /Users/drewprescott/Desktop/share_labor_food/sharing
  - Result: 18/18 passing.
- Fresh wave revalidation (27k cohort, 6 weeks, variant D):
  - Global mode output:
    - /Users/drewprescott/Desktop/share_labor_food/sharing/artifacts/thesis_claims_eval_2026_02_19/reval_wave_global_27k_w6/summary.json
    - /Users/drewprescott/Desktop/share_labor_food/sharing/artifacts/thesis_claims_eval_2026_02_19/reval_wave_global_27k_w6/weekly_metrics.csv
  - Partition-bridge mode output:
    - /Users/drewprescott/Desktop/share_labor_food/sharing/artifacts/thesis_claims_eval_2026_02_19/reval_wave_pb_27k_w6/summary.json
    - /Users/drewprescott/Desktop/share_labor_food/sharing/artifacts/thesis_claims_eval_2026_02_19/reval_wave_pb_27k_w6/weekly_metrics.csv
- Existing large-cohort evidence used:
  - /Users/drewprescott/Desktop/share_labor_food/sharing/artifacts/wave_fulltilt_fixed/summary.json
  - /Users/drewprescott/Desktop/share_labor_food/sharing/artifacts/wave_fulltilt_fixed/weekly_metrics.csv
  - /Users/drewprescott/Desktop/share_labor_food/sharing/artifacts/kontur_partition_bridge/comparison_snapshot_2026-02-19.json

## Claim Verdicts

### 1) Zero-money cycle coordination works (no token/debt accounting)
Verdict: **SUPPORTED (active code)**
Evidence:
- Domain objects and pipeline are offer/want/cycle based with no internal currency ledger:
  - /Users/drewprescott/Desktop/share_labor_food/sharing/server.py:57
  - /Users/drewprescott/Desktop/share_labor_food/sharing/server.py:83
  - /Users/drewprescott/Desktop/share_labor_food/sharing/server.py:1733

### 2) Hungarian optimal assignment is implemented
Verdict: **SUPPORTED (active + thesis-era)**
Evidence:
- Active Hungarian implementation:
  - /Users/drewprescott/Desktop/share_labor_food/sharing/server.py:982
  - /Users/drewprescott/Desktop/share_labor_food/sharing/server.py:1041
- Thesis-era SciPy Hungarian:
  - /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/matching.py:7
  - /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/matching.py:284

### 3) Weighted edge function includes fit/proximity/trust/patience/generosity
Verdict: **SUPPORTED (active + thesis-era)**
Evidence:
- Active edge scoring:
  - /Users/drewprescott/Desktop/share_labor_food/sharing/server.py:891
  - /Users/drewprescott/Desktop/share_labor_food/sharing/server.py:897
  - /Users/drewprescott/Desktop/share_labor_food/sharing/server.py:910
  - /Users/drewprescott/Desktop/share_labor_food/sharing/server.py:913
  - /Users/drewprescott/Desktop/share_labor_food/sharing/server.py:923
- Thesis formula statement:
  - /Users/drewprescott/Desktop/thesis-prod/share-prod/THESIS.md:220

### 4) Cycle detection + atomic all-or-nothing execution
Verdict: **SUPPORTED (active)**
Evidence:
- Cycle detection and selection:
  - /Users/drewprescott/Desktop/share_labor_food/sharing/server.py:1138
  - /Users/drewprescott/Desktop/share_labor_food/sharing/server.py:1204
- Atomic commit cascade:
  - /Users/drewprescott/Desktop/share_labor_food/sharing/server.py:1271
- Atomic execution rollback cascade:
  - /Users/drewprescott/Desktop/share_labor_food/sharing/server.py:1336

### 5) Patience-based wave expansion across touching tessellations
Verdict: **SUPPORTED (active, tested, empirically observed)**
Evidence:
- Reach function + touching-cell visibility:
  - /Users/drewprescott/Desktop/share_labor_food/sharing/wave_tessellation_fulltilt.py:325
  - /Users/drewprescott/Desktop/share_labor_food/sharing/wave_tessellation_fulltilt.py:331
  - /Users/drewprescott/Desktop/share_labor_food/sharing/wave_tessellation_fulltilt.py:367
- Invariant tests:
  - /Users/drewprescott/Desktop/share_labor_food/sharing/test_wave_thesis_tdd.py:117
  - /Users/drewprescott/Desktop/share_labor_food/sharing/test_wave_thesis_tdd.py:131
  - /Users/drewprescott/Desktop/share_labor_food/sharing/test_wave_thesis_tdd.py:152
- Large run signal (153k total):
  - `share_reach_ge1` week1->week8: `0.000 -> 0.909`
  - `completed_cross_share` week1->week8: `0.000 -> 0.158`
  - /Users/drewprescott/Desktop/share_labor_food/sharing/artifacts/wave_fulltilt_fixed/weekly_metrics.csv:2

### 6) Adjacent tessellations co-solve via partition-first + bridge matching
Verdict: **SUPPORTED (active)**
Evidence:
- Local + bridge + fallback staged solve:
  - /Users/drewprescott/Desktop/share_labor_food/sharing/wave_tessellation_fulltilt.py:578
  - /Users/drewprescott/Desktop/share_labor_food/sharing/wave_tessellation_fulltilt.py:629
  - /Users/drewprescott/Desktop/share_labor_food/sharing/wave_tessellation_fulltilt.py:710
- Test verifies bridge stage activity:
  - /Users/drewprescott/Desktop/share_labor_food/sharing/test_wave_thesis_tdd.py:268
- Fresh 27k revalidation: partition-bridge improved throughput and unmet ratio vs global.

### 7) 40M+ computational tractability claim is proven end-to-end
Verdict: **NOT PROVEN BY CURRENT EMPIRICAL EVIDENCE**
Evidence:
- Thesis theoretical statement exists:
  - /Users/drewprescott/Desktop/thesis-prod/share-prod/THESIS.md:245
  - /Users/drewprescott/Desktop/thesis-prod/share-prod/THESIS.md:253
- Empirical runs are far below 40M active participants:
  - 153k total synthetic benchmark summary:
    - /Users/drewprescott/Desktop/share_labor_food/sharing/artifacts/wave_fulltilt_fixed/summary.json:2
  - 76,057 total Kontur benchmark snapshot:
    - /Users/drewprescott/Desktop/share_labor_food/sharing/artifacts/kontur_partition_bridge/comparison_snapshot_2026-02-19.json
- Conclusion: directionally supportive, not proof at thesis scale.

### 8) Complexity reduction O(N^3/m^2) is validated in practice
Verdict: **PARTIAL (theory implemented, empirical trend supportive, full proof missing)**
Evidence:
- Theoretical derivation in thesis:
  - /Users/drewprescott/Desktop/thesis-prod/share-prod/THESIS.md:739
  - /Users/drewprescott/Desktop/thesis-prod/share-prod/THESIS.md:753
- Small active sweep shows superlinear growth with increasing active cohort:
  - /Users/drewprescott/Desktop/share_labor_food/sharing/artifacts/thesis_claims_eval_2026_02_19/scaling_sweep.csv
- But no formal asymptotic fit at production-scale N.

### 9) Guild trust bootstrapping solves cold start in current active engine
Verdict: **PARTIAL / LEGACY-ONLY CORE IMPLEMENTATION**
Evidence:
- Thesis-era matching uses guild quality integration:
  - /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/matching.py:89
  - /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/matching.py:126
- Guild catalog in thesis-era config:
  - /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/config.py:244
- Active `sharing` engine does not model guild membership directly; “guild floor” appears as a recommended intervention, not an enforced trust inheritance mechanism:
  - /Users/drewprescott/Desktop/share_labor_food/sharing/server.py:345

### 10) Multi-need decomposition (complex contracts) is modeled
Verdict: **SUPPORTED (active generator + tests)**
Evidence:
- Need bundle templates and decomposition controls:
  - /Users/drewprescott/Desktop/share_labor_food/sharing/wave_tessellation_fulltilt.py:92
  - /Users/drewprescott/Desktop/share_labor_food/sharing/wave_tessellation_fulltilt.py:178
  - /Users/drewprescott/Desktop/share_labor_food/sharing/wave_tessellation_fulltilt.py:1009
- Tests validate multi-offer/multi-want generation:
  - /Users/drewprescott/Desktop/share_labor_food/sharing/test_wave_thesis_tdd.py:92

### 11) Strict “never global merge” axiom is enforced in current engine
Verdict: **PARTIAL (configurable, not strict)**
Evidence:
- Wave script supports both modes:
  - /Users/drewprescott/Desktop/share_labor_food/sharing/wave_tessellation_fulltilt.py:729
  - /Users/drewprescott/Desktop/share_labor_food/sharing/wave_tessellation_fulltilt.py:733
- `partition_bridge` aligns with thesis locality principle.
- `global` mode remains available, so strict prohibition is not enforced as an invariant.

### 12) Satellite-weighted tessellation + full food/fuel stack are complete in production path
Verdict: **PARTIAL / MIXED**
Evidence:
- Thesis itself marks GEE agricultural integration pending:
  - /Users/drewprescott/Desktop/thesis-prod/share-prod/THESIS.md:780
- Strategy doc reports critical GEE fetch breakage and missing dual-weight in that stage:
  - /Users/drewprescott/Desktop/thesis-prod/share-prod/tessellation-pipeline/TESSELLATION_STRATEGY_v2.md:294
  - /Users/drewprescott/Desktop/thesis-prod/share-prod/tessellation-pipeline/TESSELLATION_STRATEGY_v2.md:297
- There is substantial tessellation pipeline code and outputs, but claim-level completeness is not uniformly true across versions.

## Fresh Revalidation Snapshot (27k, 6 weeks, variant D)
- Global mode:
  - mean week runtime: `1.856s`
  - completed per 1000 active: `144.444`
  - mean unmet ratio: `0.629`
  - /Users/drewprescott/Desktop/share_labor_food/sharing/artifacts/thesis_claims_eval_2026_02_19/reval_wave_global_27k_w6/summary.json
- Partition-bridge mode:
  - mean week runtime: `1.164s`
  - completed per 1000 active: `224.074`
  - mean unmet ratio: `0.416`
  - /Users/drewprescott/Desktop/share_labor_food/sharing/artifacts/thesis_claims_eval_2026_02_19/reval_wave_pb_27k_w6/summary.json
- Delta (partition-bridge minus global):
  - runtime: `-0.692s/week`
  - completed per 1000: `+79.630`
  - unmet ratio: `-0.213`

## Bottom Line
The core cycle-trading engine is real, implemented, and test-backed. Your thesis intent is strongly present in the matching/trust/patience/atomic core. What is not yet claim-locked is the full end-to-end proof for 40M scale and some tessellation-side innovations (especially guild trust inheritance in active engine and full satellite/dual-weight productionization consistency).
