# FaaS-MALD — Multi-Agent Lagrangian Dual coordination (design)

Date: 2026-07-02
Status: draft, awaiting user review

## Context

DFaaSOptimizer solves DiFRALB with a family of decentralized coordinators that
share one skeleton (per-timestep local LSP solves, an inner coordination loop
that fills a shared residual-capacity ledger with offloading assignments
`y[i,j,f]`, replica top-ups via memory bids, `compute_social_welfare` re-solve,
`check_stopping_criteria`, standard solution saving):

- FaaS-MADEA — Bertsekas-style ε-increment auction (prices as bids)
- FaaS-MADiG — price-free greedy diffusion
- FaaS-MABR (S/R/O) — Gauss-Seidel best-response sweeps
- FaaS-MAPoD — power-of-d-choices sampling

None of them can say how far the coordination outcome is from the optimum of
the per-timestep coordination problem without running the centralized LMM.

## Goal

A new coordinator, **FaaS-MALD**, that is (a) scientifically grounded —
projected dual subgradient with known convergence guarantees, (b) original in
this suite — continuous marginal-value prices plus a per-timestep
**duality-gap certificate**, and (c) empirically competitive — the buyer step
is closed-form (no MILP), so inner iterations are cheap.

Constraint: existing code is reused, never modified. FaaS-MALD is one new
module `decentralized_dual.py` (plus its test), mirroring how
`decentralized_powerd.py` was added.

## The coordination problem and the algorithm

After the local LSP solves at time t, coordination reduces to the
transportation LP (same score and filter the other coordinators use):

```
max  Σ_{i,f} Σ_{j∈N(i)} s_{ijf} · y_{ijf}
s.t. Σ_j y_{ijf} ≤ ω_{if}          (buyer demand, local to i)
     Σ_i y_{ijf} ≤ C_{jf}          (seller capacity, coupling)
     y ≥ 0,   only pairs with s_{ijf} > −γ_{if}
```

with `s_{ijf} = β_{ijf} − w_lat·L_{ij} − w_fair·φ_{if}` (φ is the fairness
counter, as in MADiG/MABR).

Relax the coupling constraints with multipliers λ_{jf} ≥ 0 (the "shadow price"
of seller capacity):

```
L(λ) = Σ_{jf} λ_{jf} C_{jf} + Σ_{i,f} ω_{if} · max(0, max_{j∈N(i)} (s_{ijf} − λ_{jf}))
```

**Inner loop** (k = 0, 1, …, per outer coordination iteration):

1. **Buyer response (closed form).** Each buyer i, for each f with ω_{if}>0,
   ranks accessible sellers by price-adjusted score `s̃ = s_{ijf} − λ_{jf}` and
   requests its demand greedily from positive-`s̃` sellers, capping each request
   at the advertised capacity C_{jf} (so requests stay realistic and the
   waterfilling produces useful bids). The uncapped argmax response is what
   defines the subgradient; the capped ranked list doubles as the bid list for
   primal recovery.
2. **Seller price update (projected subgradient).** Each seller aggregates the
   demand D_{jf} directed at it and updates
   `λ_{jf} ← max(0, λ_{jf} + α_k (D_{jf} − C_{jf}))`.
   Step rule: `α_k = α0/√(k+1)` (default, guaranteed convergence of the dual
   to the LP dual optimum — Nedić & Ozdaglar 2009 / Bertsekas), or Polyak
   `α_k = θ (L(λ_k) − best_LB) / ‖g_k‖²` as an option.
3. **Primal recovery (reuse).** The ranked, price-ordered bids are fed to the
   existing `decentralized_diffusion.evaluate_assignments` against the true
   residual capacity, yielding a feasible y. Keep the best feasible y found
   across inner iterations (LB); L(λ_k) gives a monotone-tracked dual UB.
4. **Stop** when `(UB − LB)/max(1,|UB|) ≤ gap_tolerance`, or after
   `max_inner_iterations`, or when demand prices out (no positive s̃ anywhere).

**Certificate.** Per outer iteration the run logs `best_LB`, `best_UB`, and
the relative gap in the termination-condition CSV — a bound on suboptimality
of the coordination step that no other method in the suite provides.

**Outer loop, replicas, and everything else** are identical in structure to
`decentralized_diffusion.run`: commit the best y increment, re-run
`compute_social_welfare` (LSPr) to re-optimize local variables given y, use
memory bids + `start_additional_replicas` for replica top-ups when unmet
demand persists, stop via the shared `check_stopping_criteria` plus a
"no dual progress" guard, track best decentralized and best centralized
solutions, save with `save_solution`/`save_checkpoint`.

Distribution model: one inner iteration costs one 1-hop price broadcast per
seller and one 1-hop demand message per buyer — O(K·|E|·Nf) messages, no
global state beyond the ledger already assumed by every other coordinator.
Runtime is amortized per active node, matching `compute_sweep_runtime`-style
accounting used by the siblings.

## Approaches considered

1. **Lagrangian dual subgradient with greedy primal recovery (chosen).**
   Closed-form buyer step, tiny per-iteration cost, dual certificate,
   textbook convergence theory, maximal reuse of `evaluate_assignments`.
2. **Consensus-ADMM on y.** Stronger (O(1/k)) convergence and no step-size
   tuning, but needs per-node QP solves and variable duplication — much more
   new code, no certificate advantage over the dual bound. Rejected.
3. **Lyapunov drift-plus-penalty.** Provable stability for time-varying load,
   but changes the per-timestep framing (queues carried across t) and would
   not be comparable to the existing per-timestep suite. Rejected.

## Architecture and reuse map

New file `decentralized_dual.py` (structure mirrors
`decentralized_diffusion.py`):

| Piece | Source |
|---|---|
| timestep loop, IO, checkpoints, plots | `run_centralized_model` helpers (init_problem, get_current_load, update_data, save_*) |
| local solves LSP/LSPr, social welfare | `run_faasmacro.solve_subproblem`, `compute_social_welfare`, `combine_solutions`, `decode_solutions` |
| ledger, stopping, replicas | `run_faasmadea.compute_residual_capacity`, `check_stopping_criteria`, `start_additional_replicas`, `neigh_dict_to_matrix`, `check_ls_pr_feasibility_from_fixed_y` |
| feasible allocation from priced bids | `decentralized_diffusion.evaluate_assignments` |
| feasibility / objective checks | `utils.centralized.check_feasibility`, `utils.faasmacro.compute_centralized_objective` |
| **new**: `dual_coordination_round(...)` — inner subgradient loop returning (best_y_increment, memory_bids, gap_info, n_active) | `decentralized_dual.py` |

New functions are pure on arrays (like `best_response_sweep`) for testability.

## Configuration

Options under `solver_options["dual"]`, all read with defaults so existing
config files run unchanged:

```json
"dual": {
  "alpha0": 1.0,
  "step_rule": "sqrt",          // "sqrt" | "polyak"
  "max_inner_iterations": 50,
  "gap_tolerance": 0.01,
  "latency_weight": 0.0,
  "fairness_weight": 0.0
}
```

CLI: `uv run decentralized_dual.py -c <config> [-j N] [--disable_plotting]`,
same flags as the siblings. Method name in outputs: `FaaS-MALD`.

## Outputs

Same artifacts as MADiG (LSP/LSPc solutions, obj.csv, runtime.csv,
termination_condition.csv) with the gap certificate embedded in the
termination-condition string: `gap: <rel_gap> (LB: …, UB: …)`.

## Testing

`tests/test_decentralized_dual.py`:

- unit: on a hand-built 3-node/2-function instance, the inner loop's dual
  values are valid UBs (≥ LP optimum computed by brute force), best-LB is
  monotone non-decreasing, recovered y is feasible, and the gap closes below
  1% within the iteration budget;
- price behavior: an oversubscribed seller's λ rises until demand shifts to
  the cheaper neighbor;
- e2e smoke: full `run()` on the small manual-config-style instance produces
  feasible solutions (`check_feasibility` already asserts inside the loop) and
  the standard output files.

Benchmark (follow-up, not in scope here): add FaaS-MALD to the comparison
experiments against MADiG/MABR/MADEA on the paper instances.

## LaTeX technical note (deliverable)

A `faas-mald-note/` directory following the exact pattern of
`faas-madig-note/`:

- `faas-mald.tex` — the deliverable: a self-contained `\section` in the
  notation and style of `Decentralized_FaaS_coordination.pdf`, with a
  fenced `BEGIN/END self-contained notation recap` subsection (relevant
  rows of the paper's Tables 2–3 and capacity equations), the coordination
  LP and its Lagrangian relaxation, the algorithm (buyer response, projected
  subgradient price update, primal recovery, gap certificate), the
  convergence and weak-duality statements, complexity/message analysis, and
  a *Positioning with respect to the literature* subsection (dual
  decomposition / subgradient methods, auction baseline, price-based
  offloading).
- `references.bib` — entries cited by the positioning subsection.
- `main.tex` — minimal standalone wrapper to compile a preview PDF
  (`latexmk -pdf main.tex`).
- `README.md` — same structure as the sibling notes' READMEs.

The note is written after the implementation is validated, so the pseudocode
and the stated properties match the shipped code.

## Implementation execution policy

All implementation tasks from the plan (code, tests, LaTeX note) MUST be
executed with **Sonnet 5 at most** (subagents launched with `model: sonnet`,
or a lighter model where adequate). The plan document must carry this
constraint explicitly on each task.

## Non-goals

- No modification of any existing module, model, or config file.
- No integration into `remote_experiments/definitions/paper.py` (would
  require editing existing code); standalone entrypoint only.
- No asynchronous/message-passing runtime — same synchronous simulation model
  as the rest of the suite.

## Theoretical properties (to state in the note)

- With α_k = α0/√(k+1), λ_k converges to the dual optimum of the coordination
  LP; since the LP has zero duality gap, UB → LP optimum.
- Every reported gap is a valid per-timestep suboptimality certificate for the
  coordination step (weak duality), regardless of when the loop is stopped.
- Per-iteration complexity: O(Σ_i |N(i)|·Nf) buyer-side, O(Nn·Nf) seller-side;
  message complexity O(|E|·Nf) per iteration, 1-hop only.
