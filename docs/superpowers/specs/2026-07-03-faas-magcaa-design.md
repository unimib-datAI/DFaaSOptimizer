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

**Correction after re-reading the codebase** (the version below supersedes
the first draft): the canonical, actively-maintained MADeA implementation
is `run_faasmadea.py` (imported by `run.py` as `run_auction`), not the
older `decentralized_auction.py` (dead code, unreferenced by `run.py`,
lacks the `unit_bids` mode). `run_faasmadea.py`'s `define_bids` already
has a `unit_bids: true` mode that generates one bid row per integer unit
of load, each carrying both a ranking price `b` (VCG-style, includes
`epsilon`/`delta`) *and* the raw `utility` value in a separate column.
Since GCAA's bid is pure utility with no price adaptation, calling
`define_bids` with `p` pinned at an all-zero array that is **never
updated** (no `evaluate_bids` price-update step) makes its `ut`
computation exactly `beta - latency_weight*latency - fairness_weight*fairness`
— precisely the GCAA bid formula, with no fork needed. Passing
`rho = np.zeros(Nn)` (instead of the real `sp_rho`) makes
`potential_memory_sellers` always empty, so `memory_bids` stays empty and
the replica-bidding path is inert without needing to strip any code.
`check_stopping_criteria` in `run_faasmadea.py` is already fully
parameterized (all its branches degrade gracefully when `memory_bids` is
always empty and `a`/`additional_replicas` is always zero), so it is
reused unmodified too.

This means only the **consensus/winner-selection step** is new code —
everything else is direct reuse. New file: `decentralized_gcaa.py`.

- `resolve_gcaa_round(bids: pd.DataFrame, blackboard: np.array) -> Tuple[np.array, np.array]`
  — implements Algorithm 1 + Algorithm 3 from the paper in one pass:
  first keeps only the highest-`utility` row per agent `(i, f)` (one
  proposal per agent per round, i.e. `SelectBestTask`), then, per
  contested task `(j, f)`, picks the single highest-`utility` row as
  winner and transfers its `d` (always `1` under `unit_bids`) from
  `blackboard[j,f]`. Returns `(y_round, blackboard)` where `y_round` is
  the `(Nn, Nn, Nf)` allocation delta for this round. Losers are simply
  absent from `y_round`; they reappear in `bids` next round via the
  outer loop's call to `define_bids` (their `omega` is unchanged). New
  code, ~20-25 lines.
- `run(config, parallelism, log_on_file=False, disable_plotting=False)` —
  copy of `run_faasmadea.py`'s `run()` outer per-timestep loop and inner
  `while not stop_searching` loop, with the price/`evaluate_bids` block
  replaced by a call to `resolve_gcaa_round`, `p` never updated (stays
  zero), and no replica-bidding branch (dead under `rho=zeros`, so
  omitted rather than left as inert code — YAGNI). Everything else
  (subproblem solve, restricted-problem solve, best-solution tracking,
  checkpointing, saving) is an unmodified copy of the existing loop
  shape, per repo convention (each `decentralized_*.py` file owns a full
  `run()`, not a shared parameterized one).

Reused unmodified (imported, not reimplemented):
- `run_faasmadea.define_bids`, `check_stopping_criteria`,
  `compute_residual_capacity`, `neigh_dict_to_matrix`
- `run_faasmacro.solve_subproblem`, `combine_solutions`, `decode_solutions`,
  `compute_social_welfare`
- `utils.faasmacro.compute_centralized_objective`
- `utils.centralized.check_feasibility`
- `run_centralized_model.init_problem`, `get_current_load`,
  `init_complete_solution`, `join_complete_solution`, `save_checkpoint`,
  `save_solution`, `plot_history`, `update_data`
- `models.sp.LSP`, `LSPr`

## Config

New `solver_options.gcaa` block (mirrors `solver_options.auction` but only
the keys `define_bids` actually reads — no `eta`/`zeta`, those are
`evaluate_bids`-only and GCAA never calls it):
```json
"gcaa": {
  "unit_bids": true,
  "epsilon": 0.01,
  "latency_weight": 0.0,
  "fairness_weight": 0.0
}
```

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

**Correction:** the repo does have a real pytest suite
(`tests/test_potentialgame_*.py`, `tests/test_diffusion_*.py`, etc.), one
per algorithm, following a consistent three-layer pattern that
`decentralized_gcaa.py` will follow:

1. **Helper unit tests** (`tests/test_gcaa_helpers.py`, mirrors
   `test_potentialgame_helpers.py`) — pure-function tests of
   `resolve_gcaa_round` against small synthetic `bids` DataFrames and
   `blackboard` arrays: single winner picked per contested task, losers
   absent from `y_round`, ties broken deterministically, empty `bids`
   returns an all-zero `y_round`.
2. **Wiring tests** (`tests/test_gcaa_wiring.py`, mirrors
   `test_diffusion_wiring.py`) — `--methods faas-gcaa` accepted by
   `run.parse_arguments()`, `run.run_gcaa` exists and is callable,
   `run.set_solution_folder` tolerates a missing `"faas-gcaa"` key, and a
   fully monkeypatched `run()` smoke test (all I/O and solver calls
   stubbed) verifying the orchestration wiring without solving any MILP.
3. **End-to-end test** (`tests/test_gcaa_e2e.py`, mirrors
   `test_potentialgame_e2e.py`) — skipped if Gurobi is unavailable, runs
   `decentralized_gcaa.run()` on a small planar instance (same shape as
   `_e2e_config` in `test_potentialgame_e2e.py`) and asserts `obj.csv`,
   `runtime.csv`, `termination_condition.csv` are produced and well-formed.

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
