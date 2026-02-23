# ShareWith Next TODOs

Date: 2026-02-23  
Owner: Drew  
Purpose: Single handoff list to give Codex for implementation.

## 1) Canonical Honest Test Input Shape (Freeze Shape, Scale Size)

- [ ] Define the first-experience declaration shape (offers, wants, urgency, context, trust, patience, mode).
- [ ] Lock one canonical schema/version for tests (`canonical_input_shape_v1`).
- [ ] Define archetypes for realistic early experiences:
  - [ ] urgent need
  - [ ] planned need
  - [ ] discretionary want
  - [ ] decomposed multi-need bundle
- [ ] Define scaling rule: increase population/volume only, do not change declaration shape logic.
- [ ] Define participation rule so external people can contribute test data without changing schema.

### Deliverable
- [ ] One fixed input contract + seed generator that works unchanged at 10k, 100k, 1M+ population tests.

---

## 2) L1 Tessellation Model Finalization (US-Scale Simulation)

- [ ] Finalize L1 tessellation geometry and size assumptions.
- [ ] Confirm hop math for ring-wave expansion (week-by-week outward expansion).
- [ ] Map population seeding to tessellations for realistic US distribution.
- [ ] Run controlled test matrix:
  - [ ] small cohort sanity
  - [ ] mid-scale stress
  - [ ] full US-scale simulation
- [ ] Track stability metrics across scale: fill rate, unmet ratio, trust drift, wait growth, cross-cell fairness.

### Deliverable
- [ ] One L1 tessellation config + reproducible US-scale run profile.

---

## 3) Cold-Start Augmentation Idea (Part A + Part B)

- [ ] Provide Part A concept (what it does, when it triggers, expected effect).
- [ ] Provide Part B concept (what it does, when it triggers, expected effect).
- [ ] Define interaction between Part A and Part B (order, dependencies, failure cases).
- [ ] Define measurable success criteria for cold start.
- [ ] Define what data is required vs optional for first rollout.

### What to Send Codex

- [ ] Part A summary (3-8 sentences)
- [ ] Part B summary (3-8 sentences)
- [ ] Example user journey (first 2-3 cycles)
- [ ] Constraints (what must never be changed)
- [ ] Preferred knobs to tune (what can change safely)

### Deliverable
- [ ] Implementable cold-start spec that can be simulated and A/B tested.

