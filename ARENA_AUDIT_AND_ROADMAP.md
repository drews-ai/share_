# ShareWith Modeling Arena: Forensic Audit + Build Roadmap

Date: 2026-02-16
Scope:
- Legacy thesis simulation stack
- Archived v4.2.2 artifacts
- Current POC server/UI

Primary code reviewed:
- /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/models.py
- /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/matching.py
- /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/simulation.py
- /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/trust_network.py
- /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/config.py
- /Users/drewprescott/Desktop/thesis-prod/share-prod/THESIS.md
- /Users/drewprescott/Desktop/old_but_good/sharewith.ai/SHAREWITH_V4.2.2_SPECIFICATION.md
- /Users/drewprescott/Desktop/share_labor_food/sharing/server.py

Empirical artifact reviewed:
- /Users/drewprescott/Desktop/old_but_good/sharewith.ai/integration/simulation_results_20251014_205735.json

Note:
- The thesis simulation code and archived export code are byte-identical for core modules.

## 1) What Is Strong (Keep)

1. Clear domain modeling primitives in the thesis engine (Agent/Edge/Cycle/TrustMetrics) that map to the spec:
   - /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/models.py
2. Useful Monte Carlo loop and metrics capture (weekly participation, cycle pipeline, trust, patience):
   - /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/simulation.py
3. Explicit trust-network formalization via personalized PageRank (good scientific direction even if currently rough):
   - /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/trust_network.py
4. Strong tessellation and geography groundwork in thesis + archived work, including champion placement and multi-scale structures:
   - /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/voronoi_territories.py
   - /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/territory_generator.py
5. Current POC UI/server is a solid interaction shell for stakeholder communication and rapid scenario critique:
   - /Users/drewprescott/Desktop/share_labor_food/sharing/index.html
   - /Users/drewprescott/Desktop/share_labor_food/sharing/server.py

## 2) Critical Gaps / Regressions (Findings)

### P0: Core algorithm mismatch vs thesis/spec claims

1. The thesis code path does not use Hungarian in production matching path; it uses explicit cycle enumeration + greedy selection.
   - /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/matching.py:519
   - /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/matching.py:534
2. 4- and 5-cycles are hard-disabled (`if False`), meaning stated multi-cycle capacity is not represented in running code.
   - /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/matching.py:421
   - /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/matching.py:459
3. 3-cycle search is randomly sampled to at most 10,000 combos, making outcomes non-exhaustive and parameter-sensitive at scale.
   - /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/matching.py:389

Impact:
- Theoretical optimality claims and practical runtime behavior diverge.
- Scientific claims need ablation-backed alignment to actual solver behavior.

### P0: Atomicity principle not fully implemented in execution layer

1. Cycles are confirmed and committed independently; no global per-agent cycle-set atomic gate before commit.
   - /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/simulation.py:244
   - /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/simulation.py:248
2. Failure handling is effectively a no-op for extra penalties/causal accounting.
   - /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/simulation.py:433
   - /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/simulation.py:445

Impact:
- Spec-level “all or none cycle-set” behavior is not guaranteed in simulation.

### P0: Relationship memory bypasses admissibility math

1. Reactivated cycles are injected with fixed edge weight `100.0`, bypassing trust/distance/gates.
   - /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/simulation.py:528

Impact:
- Can dominate matching and distort outcome interpretation.
- Invalidates comparative experiments against base scoring rules.

### P1: Context handling inconsistency in cycle-level updates

1. Several trust/patience updates assume one context per cycle via first edge context.
   - /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/simulation.py:258
   - /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/simulation.py:398

Impact:
- If mixed-context cycles occur, trust and patience updates can be misattributed.

### P1: Config/spec inconsistencies

1. Spec appendix suggests `min_trust_for_matching: 0.40` and context bases like SURVIVAL=2, SAFETY=5.
   - /Users/drewprescott/Desktop/old_but_good/sharewith.ai/SHAREWITH_V4.2.2_SPECIFICATION.md:2920
   - /Users/drewprescott/Desktop/old_but_good/sharewith.ai/SHAREWITH_V4.2.2_SPECIFICATION.md:2958
2. Thesis code uses stricter context thresholds and different distance defaults.
   - /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/config.py:33
   - /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/config.py:41

Impact:
- Calibration uncertainty; hard to claim empirical consistency with written framework.

### P1: Trust-network implementation quality issues

1. `idx_to_agent` is built from id strings rather than Agent objects (not currently critical, but structurally wrong).
   - /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/trust_network.py:37
2. PPR vector is normalized by max value, not sum; this loses probabilistic interpretation.
   - /Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/trust_network.py:152

Impact:
- Interpretability and reproducibility of trust-derived distance multipliers are weakened.

### P2: Current POC server has expected prototype limits

1. Static scenarios only; no persistence, no historical events, no real declaration lifecycle.
   - /Users/drewprescott/Desktop/share_labor_food/sharing/server.py:267
   - /Users/drewprescott/Desktop/share_labor_food/sharing/server.py:869
2. Matching is brute-force DFS combinatorial search (fine for tiny scenarios, not arena-scale).
   - /Users/drewprescott/Desktop/share_labor_food/sharing/server.py:446
3. No multi-week state transitions, trust updates, or execution windows.
   - /Users/drewprescott/Desktop/share_labor_food/sharing/server.py:803

## 3) Empirical Reality Check (from archived run)

From:
- /Users/drewprescott/Desktop/old_but_good/sharewith.ai/integration/simulation_results_20251014_205735.json

Observed:
1. 4-week run, 750 agents total (10 per 75 neighborhoods)
2. Average match rate: 18.27%
3. Average confirmation rate: 99.42%
4. Average completion rate (reported weekly): 36.94%
5. All observed cycles were length 2 (avg=2.0, max=2)

Interpretation:
- System currently behaves as a bilateral exchange engine in practice.
- 3+ cycle emergence is not materially present in this run profile.
- Completion-rate reading is also affected by 2-week execution lag and short horizon.

## 4) New Lessons (Scientific, Not Cosmetic)

1. Solver truth over narrative: claims must map to active code path, not dormant functions.
2. Atomicity is a protocol concern, not just a post-hoc flag.
3. Relationship priors need calibrated Bayesian weighting, not hard overrides.
4. Multi-week metrics require cohort accounting (avoid censoring bias from unresolved in-flight cycles).
5. Any “trust multiplier” must preserve interpretability (probability-like semantics or explicit scale contracts).
6. Parameter provenance is mandatory (spec values, code values, fitted values need explicit versioning).

## 5) Data We Need to Collect or Create for a Real Arena

### A) Core event data (must-have)

1. Declaration events: create/update/cancel/expiry with timestamp + context + tags + hours + materials
2. Matching events: candidate edges, admissibility reasons, selected cycles, rejected alternatives
3. Confirmation events: who confirmed/declined/timeout + response latency
4. Execution events: edge-level completion outcomes, timestamps, no-shows, partials
5. Trust events: completion updates, ratings, disputes, appeals

### B) Spatial/operational data

1. Tessellation cell membership per agent over time
2. Cell adjacency graph and route/travel-time matrix
3. Optional service-time constraints (e.g., 15-minute L1 constraints)

### C) Supply chain / materials (if enabled)

1. Materials request and fulfillment events by cycle edge
2. Inventory snapshots by guild/node
3. Substitution and shortage outcomes

### D) Evaluation labels (for rigorous model improvement)

1. Failure reason taxonomy (coordination, skill mismatch, trust, distance, materials)
2. Counterfactual labels: “would this have succeeded under backup edge X?”
3. Human QA labels for fit quality and fairness outcomes

## 6) Arena Application Architecture (vNext)

### Layer 1: Domain + Event Store

1. Event-sourced core entities: Agent, Declaration, Edge, Cycle, Execution, TrustState, GuildState
2. SQLite/Postgres backend with immutable event tables + derived materialized views

### Layer 2: Matching + Execution Engine

1. Deterministic solver interface with interchangeable strategies:
   - exact 2/3-cycle ILP
   - heuristic search
   - Hungarian-derived assignment baseline
2. Explicit atomic cycle-set coordinator
3. Execution simulator with configurable delays, failures, and behavior models

### Layer 3: Experiment Harness

1. Monte Carlo runner with seeds and parameter sweeps
2. Ablation framework (toggle trust network, patience, guild floor, relationship memory)
3. Statistical outputs: confidence intervals, sensitivity curves, Pareto fronts

### Layer 4: Modeling Arena UI

1. Scenario builder (population, contexts, trust priors, geography)
2. Run controls (weeks, seeds, solver mode, policy toggles)
3. Diagnostics views:
   - edge filter waterfall
   - cycle length distribution
   - atomic rollback map
   - trust drift by cohort
   - fairness and backlog metrics

## 7) 30/60/90 Build Plan

### Phase 1 (0-30 days): Scientific baseline hardening

1. Extract core engine into modular package in `/sharing`
2. Implement explicit protocol tests:
   - atomicity invariants
   - trust update invariants
   - declaration balance constraints
3. Add run artifacts + reproducibility bundle (`seed`, config hash, solver hash)

Deliverable:
- Reproducible baseline simulator with deterministic outputs

### Phase 2 (31-60 days): Arena API + experiment engine

1. Add API endpoints for scenario CRUD, run, replay, and ablation sweeps
2. Persist all run events + aggregate metrics
3. Add confidence interval computation and sensitivity reports

Deliverable:
- Back-end ready for heavy experimentation

### Phase 3 (61-90 days): Full interaction arena UI

1. Build multi-panel modeling UI over real backend data
2. Add visual diffs across policy variants
3. Add “pain point to intervention” recommendation engine driven by measured causal impact

Deliverable:
- Full modeling arena app (not just wireframe)

## 8) Immediate Actions Recommended

1. Reconcile spec vs code parameter table into one source of truth (versioned YAML)
2. Remove hardcoded relationship weight override (`weight=100`) and replace with calibrated bounded bonus
3. Implement real atomic cycle-set gating before commitment
4. Add cohort-based completion metric to remove 2-week censoring bias
5. Make solver mode explicit and logged in every run output

## 9) Environment Note

Running thesis simulation directly in current shell failed due missing dependency:
- `ModuleNotFoundError: No module named 'scipy'`

This needs a dedicated environment for reproducible arena work.
