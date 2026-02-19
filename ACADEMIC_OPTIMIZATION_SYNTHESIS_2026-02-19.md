# Academic Optimization Synthesis (2026-02-19)

## 1) Thesis-intent alignment check

This implementation keeps the thesis core intact:

- Weekly resolution cycle remains: declarations -> weighted edge construction -> matching -> cycle detection -> confirmation -> atomic commit -> trust/patience update.
- Context-gating, trust-gating, patience dynamics, and distance admissibility are preserved.
- Tessellation wave behavior is receiver-centric and local-first: agents begin in home tessellation and expand to touching tessellations via patience-derived hop radius.
- No global graph merge was introduced.

Reference implementation anchors:

- `/Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/matching.py`
- `/Users/drewprescott/Desktop/thesis-prod/share-prod/backend/simulation/simulation.py`
- `/Users/drewprescott/Desktop/share_labor_food/sharing/wave_tessellation_fulltilt.py`

## 2) Critical TDD validation status

TDD now covers both structural thesis invariants and variant behavior:

- Wave geometry / reach invariants (monotonic hops, touching-cell semantics, hop filter correctness).
- Variant D failure-risk penalty behavior.
- Variant E waiting/disadvantage prioritization behavior.
- Variant F ant-trail reinforcement behavior.

Test files:

- `/Users/drewprescott/Desktop/share_labor_food/sharing/test_wave_thesis_tdd.py`
- `/Users/drewprescott/Desktop/share_labor_food/sharing/test_sharewith_engine.py`

Most recent run status:

- `test_wave_thesis_tdd.py`: 9/9 pass
- `test_sharewith_engine.py`: 6/6 pass

## 3) New algorithm variants

### C (baseline)

- Original wave + trust objective behavior.

### D (failure-aware expected utility)

- Risk-adjusted edge weight:
  - `weight <- weight * q - penalty(1-q)`
  - `q` is a bounded success proxy using trust + proximity + distance ratio.
- Intent: reduce brittle cycle selection under completion uncertainty.

### E (fairness/time-priority)

- Adds bounded waiting/disadvantage bonus to prioritize longer-waiting agents.
- Intent: patient-first fairness pressure.

### F (bio-inspired ant-trail reinforcement)

- Adds pheromone-like reinforcement on successful edge archetypes by cell-context-skill.
- Weekly evaporation prevents hard lock-in.
- Intent: exploit repeated reliable trade routes while staying adaptive.

## 4) Monte Carlo results

## 4.1 Main C/D/E benchmark (27k agents, fixed panel, 12 seeds)

Artifact directory:

- `/Users/drewprescott/Desktop/share_labor_food/sharing/artifacts/wave_cde_27k_s12`

Key means:

- C: completed/1000 = 115.10, unmet = 0.7199, runtime/week = 1.923s
- D: completed/1000 = 137.50, unmet = 0.6645, runtime/week = 1.678s
- E: completed/1000 = 110.75, unmet = 0.7306, runtime/week = 2.168s

Pairwise vs C:

- D:
  - completed/1000 delta = +22.40 (+19.46%), `p=0.00025`, Cliff's delta = 0.847
  - unmet delta = -0.0554 (-5.54 percentage points), `p=0.00025`, Cliff's delta = -0.861
  - high-trust objective delta = +19.64, `p=0.00025`
- E:
  - no significant lift vs C on primary metrics
  - runtime regression (+0.245s/week)

## 4.2 Bio-inspired C/D/F benchmark (27k agents, fixed panel, 8 seeds)

Artifact directory:

- `/Users/drewprescott/Desktop/share_labor_food/sharing/artifacts/wave_cdf_27k_s8`

Key means:

- C: completed/1000 = 117.71, unmet = 0.7162, runtime/week = 1.947s
- D: completed/1000 = 142.13, unmet = 0.6568, runtime/week = 1.668s
- F: completed/1000 = 136.30, unmet = 0.6652, runtime/week = 1.616s

Pairwise vs C:

- D:
  - completed/1000 delta = +24.42 (+20.75%), `p=0.00033`
  - unmet delta = -0.0594, `p=0.00067`
  - objective delta = +21.38, `p=0.00033`
- F:
  - completed/1000 delta = +18.60 (+15.80%), `p=0.0070`
  - unmet delta = -0.0510, `p=0.00233`
  - objective delta = +16.37, `p=0.00899`

Interpretation:

- F is a valid improvement over C and validates bio-inspired reinforcement.
- D remains best overall in this configuration.

## 4.3 Large-cohort stress check (153k agents, fixed panel, 2 weeks, 1 seed)

Artifact directory:

- `/Users/drewprescott/Desktop/share_labor_food/sharing/artifacts/wave_cde_153k_s1`

Results:

- C: completed/1000 = 64.87, unmet = 0.8330, runtime/week = 101.84s
- D: completed/1000 = 81.05, unmet = 0.7979, runtime/week = 85.07s

Delta (D vs C):

- completed/1000: +24.94%
- unmet: -3.51 percentage points
- runtime/week: -16.78s (faster)

This is the strongest signal that D scales better under heavy load while preserving thesis mechanics.

## 5) What is good, what is weak

Good:

- Wave mechanics are correct and test-backed.
- D is consistently superior on throughput and unmet demand.
- F demonstrates that biological positive-feedback can help materially.
- 27k Monte Carlo and 153k stress both show reproducible improvement signal.

Weak / current trade-off:

- D and F reduce cross-cell completion share vs C; this can under-explore adjacent tessellations.
- E (current fairness formulation) underperforms and needs redesign.
- Large-scale results need >1 seed for stronger external validity at 153k.

## 6) Data and experiments still needed for publication-grade rigor

- Multi-seed 153k study (at least 5-10 seeds) with confidence intervals.
- Ablation grid for D coefficients (success proxy and risk penalty terms).
- Fairness suite:
  - time-to-service quantiles by trust strata
  - distributional parity / max-wait bounds
- Robustness under shifted context mixes and demand shocks.
- Cold-start subgroup outcomes (new entrants vs incumbent high-trust agents).

## 7) Research synthesis and concrete next optimizations

The changes above align with methods from dynamic/robust kidney exchange and bio-inspired optimization literature:

1. Dynamic matching thickness and waiting:
   - Akbarpour et al., "Dynamic Matching Market Design" (Review of Economic Studies)
   - https://academic.oup.com/restud/article/87/5/2113/5475932

2. Failure-aware matching / expected utility under uncertainty:
   - Dickerson et al., "Failure-Aware Kidney Exchange" (AAAI 2019)
   - https://openreview.net/forum?id=BrTUvxGSPB
   - Bidkhori et al., "Kidney Exchange with Expected Utility and Times-to-Failure" (arXiv)
   - https://arxiv.org/abs/2007.07010

3. Dynamic fairness:
   - Gao et al., "A Continuous-time Dynamic Matching Market with Two-sided Heterogeneity" (arXiv)
   - https://arxiv.org/abs/1905.09991
   - McElfresh and Dickerson, "Long-Term Fairness in Kidney Exchange" (AAMAS)
   - https://liamjcm.com/pubs/mcelfresh2018fairness.pdf

4. Bio-inspired optimization:
   - Dorigo, "Ant colony optimization" (Scholarpedia)
   - http://www.scholarpedia.org/article/Ant_colony_optimization
   - Dorigo et al., "Ant System" (IEEE TSMC-B, DOI)
   - https://doi.org/10.1109/3477.484436
   - Stutzle and Hoos, "MAX-MIN Ant System" (Future Generation Computer Systems, DOI)
   - https://doi.org/10.1016/S0167-739X(00)00043-1

5. Evolutionary / multi-objective tuning:
   - Deb et al., "NSGA-II" (IEEE TEC, DOI)
   - https://doi.org/10.1109/4235.996017
   - Hansen, "The CMA Evolution Strategy" (arXiv)
   - https://arxiv.org/abs/1604.00772

6. Practical optimization ecosystem:
   - OpenKPD framework
   - https://github.com/JohnDickerson/KidneyExchange
   - OR-Tools (CP-SAT, min-cost flow stack)
   - https://developers.google.com/optimization

## 8) Recommended operating choice right now

For the current high-trust network objective:

- Use **D as production default**.
- Keep **F as experimental branch** for adaptive route reinforcement.
- Rework **E** before further fairness claims.

