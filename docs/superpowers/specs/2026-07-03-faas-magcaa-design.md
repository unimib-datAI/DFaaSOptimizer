# FaaS-MAGCAA: Greedy Coalition Auction Algorithm adapted to FRALB

## Context

The repo implements several decentralized algorithms for the FRALB (Function
Replica Allocation and Load Balancing) problem, all following the same
per-timestep pattern: solve a local MILP subproblem per node
(`solve_subproblem`), then run an inner iterative negotiation loop between
nodes until convergence, then combine and save the solution
(`combine_solutions` / `decode_solutions`). Existing decentralized methods:
FaaS-MADeA (`run_faasmadea.py` / `decentralized_auction.py`, price-based
double auction), FaaS-MADiG (diffusion), FaaS-MAPoD (power-of-d),
FaaS-MABR (best response), FaaS-MAPG (potential game).

This spec adds **FaaS-MAGCAA**, an adaptation of the Greedy Coalition
Auction Algorithm (Braquet & Bakolas, 2021, IFAC) as an additional
decentralized baseline for experimental comparison against the existing
family. Goal: pure experimental comparison, not a theoretical contribution
— so the implementation should stay close to the original algorithm's
mechanics rather than being tuned for best performance.

## Non-goals

- No dynamic price adjustment (unlike MADeA's `p[j,f]` update) — GCAA bids
  are pure utility, faithful to the paper.
- No replica-bidding / dynamic replica creation (MADeA's `memory_bids` /
  `start_additional_replicas`) — the original GCAA algorithm has a fixed
  task set and a null assignment (∅, utility 0) for agents with no
  convenient task; FRALB's "no capacity anywhere" case maps directly to
  that null assignment, so no new mechanism is needed.
- No change to the outer per-timestep loop, MILP subproblem, or solution
  encoding/saving — those are reused unmodified.

## Conceptual mapping (GCAA paper -> FRALB)

| GCAA (paper)                        | FaaS-MAGCAA (this repo)                                   |
|--------------------------------------|-------------------------------------------------------------|
| Agent *i*                            | (buyer node *i*, function *f*) pair with residual load `omega[i,f] > 0` |
| Task *T_j*                           | (seller node *j*, function *f*) pair with residual capacity `blackboard[j,f] > 0` |
| Bid / utility `U_i`                  | `beta[i,j,f] - latency_weight * latency[i,j] - fairness_weight * fairness[i,f]` (no price term) |
| Null assignment `a_i = ∅`            | No seller with positive utility available -> agent does not bid this round, its load stays unassigned/local |
| Coalition (multiple agents on same task) | Multiple buyer (i,f) agents contest the same seller (j,f) across successive rounds; capacity is filled incrementally, one winner per round |
| Convergence bound (≤ n steps, n = #agents) | ≤ number of (buyer, function) agent-task pairs at that timestep |

## Algorithm

Per timestep `t` (inside the existing `for t in range(min_run_time, ub,
run_time_step)` loop, after `solve_subproblem` produces the initial `sp_x`,
`sp_r`, `sp_omega` as today):

```
y = 0                          # allocated load (buyer -> seller)
while exists an agent (i,f) with residual omega[i,f] > 0 and at least one
      seller with positive utility for it:
  # 1. SelectBestTask
  for each agent (i,f) with omega[i,f] > 0:
    candidates = sellers j in neighborhood[i] with blackboard[j,f] > 0
    bid[i,f] = max utility over candidates (skip if no candidate: null task)
    proposal[i,f] = argmax seller j

  # 2. Consensus (single winner per task per round, faithful to Algorithm 3)
  for each contested task (j,f) [i.e. proposal[i,f] == j for >=1 agent]:
    winner = argmax_i bid[i,f] among agents proposing j for f
    q = min(omega[winner,f], blackboard[j,f])
    y[winner,j,f] += q
    omega[winner,f] -= q
    blackboard[j,f] -= q
    # losers: proposal reset to "none this round", they retry next round
```

Stopping condition (mirrors `check_stopping_criteria` from
`run_faasmadea.py`, simplified — no price/memory-bid branches):
- no agent has residual `omega[i,f] > tolerance`, OR
- no seller has residual `blackboard[j,f] > tolerance`, OR
- every agent with residual omega has no positive-utility seller left
  (all remaining proposals are null), OR
- max iterations / time limit reached (same knobs as MADeA:
  `max_iterations`, `time_limit`).

After the inner loop converges, the outer flow is unchanged: solve the
"restricted problem" / recompute `sp_x`, `sp_r`, `sp_rho` from `y` (reuse
`compute_social_welfare` as MADeA does), combine solutions, compute
centralized objective, track best-so-far, checkpoint/save — all via the
existing `run_faasmacro.py` / `run_centralized_model.py` helpers.

## Components

New file: `decentralized_gcaa.py`, structured like `decentralized_auction.py`:

- `compute_bid(i, f, candidates, sp_data, latency, fairness) -> dict[j] = utility`
  — pure-utility bid, no price term. New code (small, ~15 lines).
- `select_best_tasks(omega, blackboard, neighborhood, sp_data, latency, fairness) -> proposals: pd.DataFrame`
  — one row per agent with a proposal (i, f, j, bid). Agents with no
  positive-utility candidate are omitted (null assignment). New code,
  analogous shape to MADeA's `define_bids` but without price/memory_bids.
- `resolve_consensus(proposals, blackboard, omega) -> (y_round, blackboard, omega)`
  — groups proposals by (j, f), picks the single highest-bid winner per
  group, transfers `min(omega[winner,f], blackboard[j,f])`. New code
  (~Algorithm 3 from the paper).
- `check_stopping_criteria(...)` — trimmed copy of MADeA's version without
  the `rmp_omega` / `memory_bids` branches.
- `run(config, parallelism, log_on_file=False, disable_plotting=False)` —
  same outer per-timestep loop as `decentralized_auction.py`'s `run()`,
  swapping `define_bids` + `evaluate_bids` (+ price update) for
  `select_best_tasks` + `resolve_consensus` in an inner `while` loop.

Reused unmodified (imported, not reimplemented):
- `run_faasmadea.compute_residual_capacity`, `neigh_dict_to_matrix`
- `run_faasmacro.solve_subproblem`, `combine_solutions`, `decode_solutions`,
  `compute_social_welfare`, `compute_centralized_objective`
- `run_centralized_model.init_problem`, `get_current_load`,
  `init_complete_solution`, `join_complete_solution`, `save_checkpoint`,
  `save_solution`, `plot_history`, `update_data`
- `models.sp.LSP`, `LSPr`

## Integration

`run.py`:
- `METHOD_RESULT_MODELS["faas-gcaa"] = ("LSPc", "FaaS-MAGCAA")`
- add `"faas-gcaa"` to the `--methods` CLI choices list
- import `from decentralized_gcaa import run as run_gcaa`
- add a dispatch block mirroring the existing `if run_a: ... run_auction(...)`
  block, gated by a new `run_gcaa_flag`/similar boolean derived from
  `"faas-gcaa" in methods` (following the exact naming/branching pattern
  already used for `run_a`, `run_diffuse`, etc. in `run.py`)

## Testing

Following repo convention (no existing test suite dedicated to individual
decentralized algorithms was found beyond running them end-to-end via
`run.py`/manual configs) — verification is a runnable self-check: a small
`if __name__ == "__main__"` block already exists in sibling files
(`decentralized_auction.py`) for standalone invocation via
`parse_arguments()` + `load_configuration`. `decentralized_gcaa.py` will
follow the same pattern, plus one small `assert`-based smoke test
(e.g. a 2-node/1-function synthetic `sp_data` fixture) verifying:
- a single winner is picked per contested task per round
- losers retry and eventually get allocated once capacity or rounds allow
- the loop terminates within `Nn * Nf` rounds worst case (mirrors the
  paper's ≤ n convergence bound, adapted to the number of agent-task pairs)

## Open questions / risks

- The paper's ≤ n round bound assumes exactly one task finalized per
  round system-wide (single global consensus). Here, multiple **different**
  (j,f) tasks can each finalize a winner in the same round (independent
  seller queues), so the practical round count is likely lower than
  `Nn * Nf`, not higher — worth confirming empirically rather than
  re-deriving the bound formally, since this is a comparison baseline, not
  a proof.
- Fairness weighting (`fairness[i,f]`) accumulates only within a single
  timestep's inner loop today (matches MADeA's existing behavior) — no
  change proposed here, flagged only so it isn't mistaken for a bug during
  review.
