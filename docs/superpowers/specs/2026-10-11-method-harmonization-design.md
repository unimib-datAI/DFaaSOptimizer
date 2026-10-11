# Harmonization of the decentralized methods — design

Date: 2026-10-11. Status: agreed in conversation, pending review of this document.

## Goal

Every decentralized method solves its local problems the same way and reports
the same, comparable algorithm time:

- local problems solved by DP;
- phases in which several nodes compute at once run on one persistent worker
  pool (all cores by default), to use the multi-core machine;
- the algorithm time of a step is a **modelled distributed time**: as if each
  node were its own machine. A parallel phase counts its longest task; a
  sequential part counts its wall time. Loading, pool start-up, output
  writing and solution validation do not count;
- every time budget inside a method uses this modelled time, never wall time;
- every method writes `runtime.csv` in the same format.

## Scope

Harmonized and registered for campaigns (methods whose design is parallel):

| Method | Runner |
|---|---|
| MADEA | `run_faasmadea.run` |
| MADEA-one-shot | `decentralized_auction.run` |
| MADEA2, MADEA2-b | `madea2.run`, `madea2.run_b` |
| MADEA2-one-shot, MADEA2-b-one-shot | `madea2.run_one_shot`, `madea2.run_one_shot_b` |
| FaaS-MAFW, FaaS-MAFW-b | `decentralized_frankwolfe.run`, `hierarchical_frankwolfe.run_mafw_b` |
| hierarchical-MAFW | `hierarchical_frankwolfe.run` (`hierarchy=True, refine_welfare=False`) |
| hierarchical-madea | `hierarchical_auction.madea_level_cycles_runner.run` |
| hierarchical-madea-cycles | `hierarchical_auction.madea_cycles_runner.run` |
| hierarchical-one-shot | `hierarchical_auction.runner.run` (`IterativeHierarchicalAuctionEngine`) |
| hierarchical (legacy) | `hierarchical_auction.runner.run` |
| PLASMA, PLASMA-Welfare | `plasma.cli`, `plasma.welfare.run` |
| faas-macro, faas-macro-v0 | `run_faasmacro.run` |
| faas-diffuse | `decentralized_diffusion.run` |
| faas-powd | `decentralized_powerd.run` |
| faas-gcaa | `decentralized_gcaa.run` |
| faas-dual | `decentralized_dual.run` |

The implementation plan verifies the exact runner of each name and adds any
parallel-by-design method found in the repository but missing here.

Excluded (kept in the code base, not harmonized, not offered to campaigns),
because they need nodes to act one at a time in a network-wide order:

- sequential better-response dynamics: FaaS-MAPG-S, FaaS-MAPG-R, faas-br-s,
  faas-br-r, faas-br-o (Gauss-Seidel sweeps; the convergence argument needs a
  single mover);
- hybrids with a potential-game refinement: MADEA-PG, one-shot-pg,
  one-shot2-pg, one-shot2-b-pg, MAFW-PG, MAFW-b-PG, hierarchical-MAFW-PG,
  hierarchical-madea-pg, hierarchical-one-shot-pg, hierarchical-madea-level-cycles-pg
  (the refinement passes a turn through all N nodes in a fixed order and
  needs a network-wide termination check).

## Modelled time

One clock per step, `utils/step_time.py`, reachable from any function of the
step through a context variable (no signature changes along call chains):

```
modelled = (wall since step start) − Σ untimed_wall − Σ phase_wall + Σ longest task of each phase
```

- `step_clock()`: context manager opened by each runner around the decision
  of one step (from the step's loads being available to the decision being
  ready); yields the clock.
- A **parallel phase** is any set of per-node computations that the algorithm
  performs simultaneously: local subproblems of all agents, bid construction,
  winner determination, local price or consensus updates, PLASMA windows.
  It is recorded with its wall time and the duration of each task, whether
  the tasks run on the pool or in a loop of the main process. Task duration =
  the node's own computation (for pool tasks measured inside the worker, so
  it excludes transfer and queueing).
- Everything else in the step is sequential and counts with its wall time:
  coordinators, master problems, global aggregation steps that the algorithm
  defines as such.
- Pool start-up happens inside the first phase's wall time, so it is excluded
  by construction; the pool persists for the whole run (`parallel_solver_session`).
- Nested phases are not allowed (a phase inside a phase raises): each per-node
  computation belongs to exactly one phase.
- **Vectorized phases** (one numpy computation over all nodes, e.g. a
  Frank-Wolfe iteration): no per-node duration exists, the work per node row
  is uniform, so the longest task is the phase wall time divided by the
  number of nodes that take part.
- **Observer code** inside a step (feasibility checks such as
  `check_ls_pr_feasibility_from_fixed_y` and `check_feasibility`, decoding and
  accumulating solutions for export, verbose logging, checkpoints) is not part
  of the algorithm: it runs inside `step_time.untimed()` and its wall time is
  removed from the modelled time (it still counts in `wall_s`).
- Today each runner approximates this differently (mean agent runtime for
  subproblems, phase time divided by the number of auctions, sellers or
  nodes, full wall time elsewhere). All these accumulations are replaced by
  the clock.

## Time budgets

- New configuration key `solver_options.general.algorithm_time_limit`
  (seconds, per step, modelled time). Every method that stops on time reads
  this key and compares it with `clock.modelled()`; default: no limit.
- `TimeLimit` is again only a parameter of the external solver (used by the
  centralized model), no longer an algorithm budget.
- The plan lists every time-based stopping rule found in the harmonized
  runners and converts each one.

## Worker pool

- One persistent pool per run, created lazily by the first parallel phase and
  closed at the end of the run.
- `parallelism`: -1 = all available cores (new default of every harmonized
  runner and of the campaign harness), 0 = in-process (tasks still timed and
  modelled the same way), k = k workers.
- Results do not depend on the number of workers (same decisions, same
  welfare); only the simulation's wall time changes.

## DP

All local problems of the harmonized methods are solved by DP (`try_solve_local`).
A harmonized run with `solver_name = "missing_solver"` must succeed; any
local problem that would fall back to a solver is a defect to fix (extend DP
or route the problem to an existing DP model), not to work around.

## Output contract

`runtime.csv` in each run folder, one row per step:

| column | meaning |
|---|---|
| `step` | step index |
| `algorithm_s` | modelled algorithm time of the step (the limit applies to it) |
| `wall_s` | wall time of the step's decision (simulation cost) |
| `parallel_phases` | number of parallel phases in the step |
| `workers` | pool size (0 = in-process) |

Readers of the old one-column `tot` file (`benchmark_summary.py`,
`compare_results.py`, `families_extended.py`, `extend_frankwolfe.py`,
reports) are updated to `algorithm_s`.

## Registry

`methods.py`: `HARMONIZED: dict[str, Callable]` (name → runner with the
common signature `run(config, parallelism, log_on_file=False, disable_plotting=False)`)
and `EXCLUDED: dict[str, str]` (name → reason). `catalog_campaign.py` and the
campaign harness of `dfaas-edge-experiments` take the method list from
`HARMONIZED`.

## Testing

- Unit: the clock formula (sequential only; one phase; several phases; pool
  and in-process phases give the same modelled time for the same task
  durations; nested phase raises).
- Contract, every method in `HARMONIZED`, small instance, 2 steps,
  `solver_name = "missing_solver"`, with `parallelism` 0 and 2:
  `runtime.csv` has 2 rows and the five columns; `0 < algorithm_s ≤ wall_s`
  (+ small tolerance); at least one parallel phase per step; same welfare
  with 0 and 2 workers.
- Budget: with a tiny `algorithm_time_limit` every iterative method stops on
  the time rule and reports it; with none, results equal the unbounded run.
- Regression: welfare of each harmonized method on the 2-step instance equals
  the pre-change value (timing changes must not change decisions, except
  where a time budget was hit before).

## Out of scope

Message counting and simulated network latency (a separate metric);
parallel variants of the excluded sequential methods.
