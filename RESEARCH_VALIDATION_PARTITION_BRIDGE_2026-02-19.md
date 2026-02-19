# Research Validation Notes (2026-02-19)

This file links each implemented addition to an external research source.

## A) Partition-first + bridge matching

Implemented behavior:
- Local-first pass, then adjacent bridge pass, instead of single global solve.

Why this is academically grounded:
1. Dynamic matching literature shows welfare/throughput depends on controlled pooling thickness and timing, not only one-shot global matching.
   - Akbarpour, Li, Oveis Gharan, "Dynamic Matching Market Design" (Review of Economic Studies, 2020)
   - https://academic.oup.com/restud/article/87/5/2113/5475932

2. Kidney-exchange optimization literature formalizes cycle-limited matching formulations and decomposition-friendly structure at scale.
   - Dickerson et al., "Position-Indexed Formulations for Kidney Exchange" (arXiv)
   - https://arxiv.org/abs/1606.01623

## B) Fallback orchestration / recourse stage

Implemented behavior:
- After local + bridge passes, fallback rounds rematch remaining declarations in-week.

Why this is academically grounded:
1. Recourse under uncertainty (post-failure correction) is explicitly studied as a core extension of static matching.
   - Ghasemi et al., "Kidney exchange with recourse based on local search" (arXiv 2024)
   - https://arxiv.org/abs/2410.20708

2. Failure-aware and expected-utility matching under uncertain edge viability supports risk-aware retry logic.
   - Bidkhori et al., "Kidney Exchange with Expected Utility and Times-to-Failure" (arXiv)
   - https://arxiv.org/abs/2007.07010

## C) Risk-aware variant D remains default in staged mode

Implemented behavior:
- Existing variant D (failure-aware expected utility penalty) retained under both global and partition_bridge modes.

Why this is academically grounded:
- Same expected-utility + uncertainty rationale as above (Bidkhori et al.).

## D) Kontur + H3 population grounding

Implemented behavior:
- Contiguous H3 cohorts sampled from Kontur `.gpkg` to test at larger population scale.

Why this is academically grounded:
1. H3 is a hierarchical geospatial index designed for scalable hex-grid spatial partitioning.
   - H3 docs
   - https://h3geo.org/docs/

2. Kontur provides gridded population distribution data used for population-weighted spatial simulation.
   - Kontur Population product page
   - https://data.humdata.org/organization/kontur
   - https://www.kontur.io/portfolio/population-dataset/

## E) Why this matters for production reliability

These papers collectively support moving from:
- one-shot global matching

toward:
- staged decomposition + recourse + uncertainty-aware scoring,

which is exactly the direction implemented in:
- `/Users/drewprescott/Desktop/share_labor_food/sharing/wave_tessellation_fulltilt.py`
- `/Users/drewprescott/Desktop/share_labor_food/sharing/kontur_partition_bridge_benchmark.py`
