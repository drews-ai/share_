# Large-Cohort Cycle Trading Scaling Note (2026-02-16)

## What was tested

Engine: `/Users/drewprescott/Desktop/share_labor_food/sharing/server.py`

Two synthetic cohort regimes:

1. **Cycle-structured ring market (best-case liquidity)**
   - Agents are arranged in explicit 3-cycles.
   - One offer + one want per agent.
   - High trust, short distances, strong reciprocal structure.

2. **Messy mixed market (more realistic heterogeneity)**
   - Mixed contexts/skills with uneven local balance.
   - One offer + one want per agent.
   - Mixed trust, mixed AUTO/CHOICE.

## Measured results

### A) Global solver scaling (best-case cycle-structured ring)

| Agents | Runtime (s) | Matched | Cycles | Completed | Unmet wants |
|---:|---:|---:|---:|---:|---:|
| 1200 | 2.665 | 1200 | 400 | 387 | (not captured in this run) |
| 1800 | 5.849 | 1800 | 600 | 584 | (not captured in this run) |
| 2400 | 10.573 | 2400 | 800 | 777 | (not captured in this run) |
| 3000 | 15.921 | 3000 | 1000 | 971 | (not captured in this run) |
| 3600 | 23.535 | 3600 | 1200 | 1169 | (not captured in this run) |
| 5000 | 51.626 | 4998 | 1666 | 1626 | 120 |
| 7000 | 89.289 | 6999 | 2333 | 2274 | 177 |
| 10000 | 181.051 | 9999 | 3333 | 3247 | 258 |

### B) Partitioned 10k run (40 cells x 250 agents, sequential)

| Agents total | Cells | Runtime (s) | Matched | Cycles | Completed | Unmet wants |
|---:|---:|---:|---:|---:|---:|---:|
| 10000 | 40 | 4.455 | 9947 | 3307 | 3218 | 345 |

Notes:
- This was run **sequentially**; true parallel execution across cells would reduce wall-clock further.
- Throughput is very close to global result, with massive runtime reduction.

### C) Global solver scaling (messy mixed market)

| Agents | Runtime (s) | Feasible edges | Matched | Cycles | Completed | Unmet wants |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 19.275 | 293241 | 1000 | 10 | 5 | 982 |
| 1500 | 46.903 | 651800 | 1500 | 18 | 15 | 1462 |
| 2000 | 101.979 | 1170801 | 2000 | 9 | 8 | 1982 |

Notes:
- In messy markets, edge volume explodes and cycle yield collapses without stronger cycle-seeding/balancing logic.
- Extrapolation from these points suggests a naive global 10k messy run is operationally impractical.

## Interpretation

1. **Can the current cycle trading engine run at 10,000 agents globally?**
   - Yes, in a highly cycle-structured regime (measured: ~181s for one resolution).

2. **Is global matching at 10k practical for weekly production cadence?**
   - Not ideal. Runtime is too high for interactive or high-frequency operation.

3. **Does tessellation/partitioning help?**
   - Yes, decisively. A 10k partitioned run was ~4.5s sequential with comparable completion totals.

4. **What fails first at scale?**
   - In heterogeneous markets, not just runtime: **cycle formation quality** drops sharply unless supply-demand is actively balanced.

## Practical conclusion

For 10k+ populations, cycle trading should run as:

- **Local cell matching first** (tessellated partitions),
- **Periodic bridge/relay matching** between cells,
- **Cycle-seeding policies** for sparse skills/contexts,
- Optional global optimization only for targeted subgraphs (not full cohort each pass).
