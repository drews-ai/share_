# The ShareWith Sharing Algorithm

**Cycle-Based Zero-Money Exchange**

---

## Core Concept

No money. No tokens. Just cycles.

```
Alice offers: Haircut
Alice wants:  Plumbing repair

Bob offers:   Plumbing
Bob wants:    Yard work

Carol offers: Yard work
Carol wants:  Haircut

CYCLE: Alice → cuts Bob's hair → Bob plumbs Alice's sink
       Bob → plumbs Carol → Carol does Alice's yard
       Carol → does Alice's yard → Alice cuts Carol's hair

RESULT: Everyone gets what they want. No money exchanged.
```

---

## Weekly Resolution

Every week (or appropriate period per context), the system runs a **Resolution Day**:

```
┌─────────────────────────────────────────────────────────────────┐
│ RESOLUTION DAY WORKFLOW                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. DECLARATION PHASE                                           │
│     └─ Agents submit offers and wants                           │
│                                                                 │
│  2. EDGE CONSTRUCTION                                           │
│     └─ Build weighted bipartite graph                           │
│                                                                 │
│  3. CYCLE DETECTION                                             │
│     └─ Hungarian algorithm finds optimal cycles                 │
│                                                                 │
│  4. CONFIRMATION                                                │
│     └─ AUTO agents pre-confirmed                                │
│     └─ CHOICE agents have 24 hours                              │
│                                                                 │
│  5. ATOMIC EXECUTION                                            │
│     └─ All cycles complete or all fail                          │
│                                                                 │
│  6. TRUST UPDATE                                                │
│     └─ Completion and quality scores updated                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Declaration Model

Each agent can declare:
- Up to **5 offers** (things they can do)
- Up to **5 wants** (things they need)

```python
Declaration = (
    agent,
    type: OFFER or WANT,
    skill: "plumbing" | "haircut" | "cooking" | ...,
    context: SURVIVAL | SAFETY | SOCIAL | GROWTH | LUXURY,
    estimated_hours: float,
    materials_required: [list],
    location: (lat, lng),
    expiry: date
)
```

---

## Edge Weight Calculation

For each potential provider→receiver pair, compute edge weight:

```
W_ij = F_ij × Π_ij × Q_ij + B_gen + C_ij + ω × log(1 + patience)

Where:
  F_ij = Fit score (skill match quality)
  Π_ij = Proximity score (distance penalty)
  Q_ij = Quality baseline (trust)
  B_gen = Generosity bonus
  C_ij = Choice mode bonus (if applicable)
  ω = Patience weight
```

### Fit Score (F_ij)

How well does provider's offer match receiver's want?

```
F_ij = (tag_overlap) × (context_match) × (hours_compatibility)
```

### Proximity Score (Π_ij)

Distance penalty with patience expansion:

```
D* = D_base + η × log(1 + patience) × Υ(context)

Π_ij = exp(-distance / D*)
```

Where:
- `D_base` = base distance threshold (context-dependent)
- `η` = patience scaling factor
- `Υ(context)` = context multiplier

**Distance thresholds:**

| Context | D_base | Max (52 weeks) |
|---------|--------|----------------|
| SURVIVAL | 2 mi | ~32 mi |
| SAFETY | 5 mi | ~45 mi |
| SOCIAL | 3 mi | ~38 mi |
| GROWTH | 10 mi | ~60 mi |
| LUXURY | 15 mi | ~75 mi |

### Quality Baseline (Q_ij)

```
Q_ij = T_completion(provider) × T_quality(provider)

T_completion = cycles_completed / cycles_committed  (Did they show up?)
T_quality = normalized_rating                       (Did they do it well?)
```

### Generosity Bonus (B_gen)

Reward agents who give more than they receive:

```
Γ_i = cycles_gave / (cycles_received + 1)

B_gen = β × log(1 + Γ_i)
```

---

## The Matching Algorithm

We use the **Hungarian Algorithm** (Kuhn-Munkres) to find optimal assignments:

```python
def match_within_tessellation(node, week):
    # Get active agents in this cell
    agents = node.get_active_agents(week)
    
    # Build weighted bipartite graph
    edges = []
    for provider in agents:
        if not provider.current_offer:
            continue
        for receiver in agents:
            if not receiver.current_want:
                continue
            if provider == receiver:
                continue
            
            # Compute edge weight
            weight = compute_edge_weight(provider, receiver)
            
            # Check admissibility (distance, trust thresholds)
            if is_admissible(provider, receiver):
                edges.append(Edge(provider, receiver, weight))
    
    # Find optimal matching
    matching = hungarian_algorithm(edges)
    
    # Extract cycles from matching
    cycles = detect_cycles(matching)
    
    return cycles
```

### Cycle Detection

From the matching, extract cycles:

```python
def detect_cycles(matching):
    """
    Find all cycles in the assignment.
    
    A cycle exists when: A→B→C→...→A
    """
    cycles = []
    visited = set()
    
    for start in matching.providers:
        if start in visited:
            continue
        
        # Follow the chain
        path = [start]
        current = start
        
        while True:
            receiver = matching.get_receiver(current)
            if receiver is None:
                break
            
            next_provider = matching.get_provider_for(receiver)
            if next_provider is None:
                break
            
            if next_provider == start:
                # Found a cycle!
                cycles.append(Cycle(path))
                visited.update(path)
                break
            
            if next_provider in visited:
                break
            
            path.append(next_provider)
            current = next_provider
    
    return cycles
```

---

## Multi-Cycle Atomicity

**Key Property:** If an agent participates in multiple cycles, ALL must complete or NONE do.

Example:
- Alice needs bathroom fixed → decomposes into 5 cycles
- Alice offers 5 haircuts in return
- All 10 edges (5 giving, 5 receiving) must execute together

```python
def check_atomicity(agent, cycles):
    """
    All cycles involving this agent must succeed together.
    """
    agent_cycles = [c for c in cycles if agent in c.agents]
    
    if all(c.confirmed for c in agent_cycles):
        # All confirmed - proceed
        for c in agent_cycles:
            c.commit()
    else:
        # Some not confirmed - abort all
        for c in agent_cycles:
            c.abort()
```

---

## Confirmation Phase

Two modes:

### AUTO Mode (Default)
- Agent pre-commits to any eligible match
- No confirmation needed
- System automatically assigns

### CHOICE Mode
- Agent sees proposed match
- 24 hours to confirm or decline
- If declined, match fails and patience increases

```python
def confirm_cycle(cycle):
    for agent in cycle.agents:
        if agent.mode == Mode.AUTO:
            continue  # Auto-confirm
        
        # CHOICE mode: probabilistic confirmation
        trust_in_counterparties = mean([
            counterparty.get_completion_trust(cycle.context)
            for counterparty in cycle.agents if counterparty != agent
        ])
        
        confirm_probability = 0.7 + 0.3 * trust_in_counterparties
        
        if random() > confirm_probability:
            return False  # Cycle not confirmed
    
    return True  # All confirmed
```

---

## Trust Updates

After cycle completion:

```python
def update_trust(cycle, ratings):
    for edge in cycle.edges:
        provider = edge.provider
        receiver = edge.receiver
        context = edge.context
        
        # Completion trust: binary (did they show up?)
        provider.trust_metrics[context].record_completion()
        
        # Quality trust: 1-5 star rating from receiver
        rating = ratings[(provider.id, receiver.id)]
        provider.trust_metrics[context].record_rating(rating)
```

Completion trust formula (exponential moving average):
```
T_completion(t+1) = λ × T_completion(t) + (1-λ) × outcome

Where:
  λ = 0.85 (decay factor)
  outcome = 1.0 if completed, 0.0 if no-show
```

---

## Guild Trust

Guild members get a **trust floor**:

```
T_effective = max(T_earned, T_guild_floor)
```

Benefits:
- New employees start at 0.80 instead of 0.50
- Guild reputation protects individuals
- Incentive to join/form guilds

---

## Patience Expansion

Unmatched wants accumulate patience:

```python
def update_patience(agents, matched_ids, week):
    for agent in agents:
        if agent.id in matched_ids:
            agent.reset_patience()
        else:
            # Unmatched - increment patience
            agent.patience += 1
```

Patience expands admissible distance:
```
D*(patience) = D_base × (1 + α × log(1 + patience))
```

After many weeks, agents can match across tessellation boundaries.

---

## Cross-Tessellation Matching

High-patience agents can match with adjacent tessellations:

```python
def cross_node_matching(agent, tessellation_map, week):
    if agent.patience < CROSS_NODE_THRESHOLD:
        return []  # Not enough patience yet
    
    # Expand search to adjacent nodes
    my_node = agent.tessellation
    adjacent_nodes = get_adjacent(my_node)
    
    potential_matches = []
    for node in adjacent_nodes:
        for candidate in node.agents:
            if compatible(agent, candidate):
                potential_matches.append(candidate)
    
    return potential_matches
```

---

## Computational Complexity

Per tessellation cell (n = ~240 active agents):

| Operation | Complexity |
|-----------|------------|
| Edge construction | O(n²) |
| Hungarian algorithm | O(n³) |
| Cycle detection | O(n) |
| **Total per cell** | **O(n³) ≈ 14M ops** |

Full US (137,500 cells, parallel):
- Total: 137,500 × 14M = 1.9 × 10¹² ops
- With 1,000 cores: 1.9 × 10⁹ ops/core
- At 1B ops/sec: **~2 seconds**

---

## Algorithm Properties

✓ **Optimal within cell:** Hungarian guarantees maximum weight matching  
✓ **Atomic execution:** No partial fulfillment  
✓ **Trust-aware:** Higher trust = higher matching probability  
✓ **Distance-aware:** Local matches preferred  
✓ **Patience-fair:** Long-waiting wants get expanded radius  
✓ **Scalable:** Parallel tessellation processing  

---

## Example Matching

Tessellation T_75 (2,287 people, 229 active agents):

```
Declarations:
  Offers: 412 (avg 1.8 per agent)
  Wants:  398 (avg 1.74 per agent)

Edge Construction:
  Potential edges: 412 × 398 = 163,976
  Feasible edges (after filters): 8,234

Hungarian Algorithm:
  2-cycles found: 87
  3-cycles found: 12
  Total cycles: 99

Matched declarations: 87×2 + 12×3 = 210
Match rate: 210 / 412 = 51%

Execution:
  Cycles completed: 74
  Cycles failed: 25
  Completion rate: 74 / 99 = 75%
```

---

*"The algorithm is the heart. Cycles are the blood. Trust is the pulse."*
