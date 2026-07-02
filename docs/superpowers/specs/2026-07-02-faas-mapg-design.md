# FaaS-MAPG — Potential-game coordination for DiFRALB

Design spec, 2026-07-02. Approved by Michele in brainstorming session.

## Goal

A new decentralized algorithm, **FaaS-MAPG** (Multi-Agent Potential Game),
for the DiFRALB problem: joint replica allocation and horizontal offloading
formulated as an **exact potential game**, with better-response dynamics that
provably terminate at an ε-Nash equilibrium while the social welfare
(centralized objective) increases monotonically.

Hard constraint: **no modification to any existing function** of other
algorithms. Reuse is allowed and encouraged; only new functions, methods,
classes, and files may be added.

## Game formulation

- **Players**: the nodes `i ∈ N`.
- **Strategy of node i**: the tuple `(r_i, x_i, z_i, y_i··)` — replicas,
  locally served load, Cloud-forwarded load, per-neighbour routing.
- **Utility**: node *i*'s own terms of the centralized objective
  (`utils.faasmacro.compute_centralized_objective`):

  ```
  u_i = Σ_f [ α_if·x_if + Σ_j β_ijf·y_ijf − γ_if·z_if ] / λ_if
  ```

  No latency/fairness terms: the potential coincides exactly with the
  centralized objective, enabling direct comparison with LMM/MABR/MADeA.
- **Unilateral-move rules**:
  1. node *i* must keep serving committed inbound flows `y_·i·`
     (LSPr-style utilization constraints);
  2. node *i* may only claim advertised residual capacity of its
     neighbours (shared ledger, decremented in place, Gauss-Seidel).

  Under these rules a move by *i* leaves every other node's utility terms
  untouched, hence **Φ = Σ_i u_i = centralized objective is an exact
  potential**.
- **Acceptance (better-response)**: a proposed move is committed iff it
  improves *i*'s true utility (evaluated after the β-aware split) by more
  than `ε`. Every accepted move raises Φ by ≥ ε; Φ is bounded above ⇒
  finitely many accepted moves ⇒ termination at an ε-Nash equilibrium.
  Welfare monotonicity is asserted at runtime.

## Node move (proposal + exact evaluation)

1. Release own row `y_i··` back to the ledger (same bookkeeping as
   `best_response_sweep` in `decentralized_bestresponse.py`).
2. Solve a local MILP with the **new model `LSP_pg`** (see below) via the
   reused `run_faasmacro.solve_subproblem`: decision `(r_i, x_i, z_i, ω_i)`
   with inbound commitments `y_bar` in the utilization constraints and
   per-function cap `omega_ub_f = Σ_{j ∈ eligible} ledger[j,f]`.
   Eligible sellers: neighbours with `β_ijf > −γ_if` (Cloud-relative
   advantage) and positive ledger.
3. Split `ω_if` across eligible neighbours by descending `β_ijf`
   (ties: ascending j), capped by ledger — a fractional knapsack, exact for
   the linear utility. Unplaced residual goes to `z_if`.
4. Compute the true `u_i(new)` with per-pair β. Accept iff
   `u_i(new) − u_i(old) > ε`; on acceptance commit the new row to the
   ledger and update `x_i, r_i, z_i`; otherwise restore the previous row.

## Seller replica expansion (memory market)

A selfish seller never opens replicas for others (β rewards the buyer).
Reuse the existing MADeA mechanism verbatim: unplaced demand emits
`memory_bids`; `run_faasmadea.start_additional_replicas` (reused, not
modified) converts seller memory slack into capacity. Opening replicas does
not change Φ (no r-term in the objective) and is bounded by finite memory,
so monotonicity and finite termination are preserved.

## Dynamics and stopping

- Gauss-Seidel sweeps. Variants: **MAPG-S** (fixed ascending order) and
  **MAPG-R** (per-sweep random permutation, seeded rng) — mirrors MABR-S/R.
- Stop when a full sweep produces zero accepted moves and zero replica
  additions ⇒ certified ε-Nash. Standard guards (max_iterations,
  time_limit) as in MABR.
- Per control period `t`: initial local solve P2 (`LSP` via
  `solve_subproblem`, reused) → sweeps until stop →
  `compute_social_welfare`, `combine_solutions`,
  `compute_centralized_objective`, `check_feasibility`,
  `decode_solutions`, `save_solution` — all reused, matching MABR's outer
  loop structure.

## Files and integration

| File | Change |
|------|--------|
| `decentralized_potentialgame.py` | NEW: `potential_game_sweep()`, `node_move()`, `compute_node_utility()`, `_run()` mirroring MABR, runners `run_pg_s` / `run_pg_r`, CLI `--variant {s,r}` |
| `models/sp.py` | ADD class `LSP_pg` (LSP + inbound `y_bar` param in utilization constraints + `omega_ub` cap) and `LSP_pg_fixedr` (for the fixed-optimal-replicas comparison mode, mirroring the others). No existing class touched. |
| `run.py` | ADD imports + map entries `faas-pg-s` / `faas-pg-r` → ("LSPc", "FaaS-MAPG-S"/"-R"). Additive lines only. |
| `config_files/*.json` (smoke/eval) | ADD `solver_options.pg_s` / `pg_r` with `epsilon` (default 1e-6) |
| `tests/test_potentialgame_*.py` | NEW: unit (knapsack split, ε-acceptance, `LSP_pg` constraints), property (welfare monotone non-decreasing across iterations; feasibility at every commit), e2e smoke vs centralized on a small instance |
| `faas-mapg-note/` | NEW: LaTeX note (`faas-mapg.tex` + `main.tex` wrapper + `references.bib` + `README.md`) following the `faas-bestresponse-note` pattern: game formulation, exact-potential proof, better-response algorithm, ε-Nash termination proof, positioning vs MABR/MADeA/MALD |

## `LSP_pg` model detail

Extends `LSP` (Pyomo): adds param `y_bar[(m,i,f)]` (committed inbound) and
`omega_ub[f]`. Constraints:

- flow: `x_f + ω_f + z_f == λ_if` (inherited `no_traffic_loss`)
- utilization: `D_if·(x_f + Σ_m y_bar[m,i,f]) ≤ r_f·U_max` (and the
  matching `≥ (r_f − 1)·U_max` lower bound, LSPr-style)
- memory budget (inherited), `ω_f ≤ omega_ub_f`

Objective: inherited LSP objective (α·x + δ·ω − γ·z, normalized). δ is a
proxy for β in the proposal; correctness comes from the exact acceptance
test in step 4, not from the proposal being optimal.

## Implementation delegation

- **Sonnet** (hard): `LSP_pg`/`LSP_pg_fixedr` models; `node_move` +
  `potential_game_sweep` + acceptance logic; monotonicity/e2e tests;
  **the LaTeX note** (formal proofs — quality critical).
- **Haiku** (easy): `run.py` registration, config keys, CLI/argparse and
  `_run` boilerplate mirroring MABR, README/docstring touch-ups.

## Correctness invariants (enforced in code/tests)

1. Φ (centralized objective on the combined solution) non-decreasing
   across accepted moves within a control period.
2. `check_feasibility` passes after every sweep commit.
3. Ledger conservation: `ledger[j,f] + Σ_i y[i,j,f] == C_j^f` after every
   move.
4. Termination: every run ends with a recorded reason
   (ε-Nash / max_iterations / time_limit).
