# Two-phase auto-chained experimental campaign

**Date:** 2026-07-06
**Status:** Design approved, pending spec review
**Scope:** `remote_experiments/` — new `campaign` orchestrator, screening suite, survivor-driven confirmatory suites.

## Problem

The `paper.py` campaign is **12,630 experiments** across 9 suites, run on **3 VMs × 2 slots = 6 workers**. The confirmatory suites use 30 seeds and the quality/scalability suites run all 10–13 algorithms (e1: all algos; e2: up to n=200), producing the slowest, most numerous cells. On 6 workers this is weeks of wall-clock. There is no cross-suite orchestration: `define`/`materialize`/`run` are separate per-suite commands and `run` blocks on an interactive `input()` prompt (`cli.py:48`).

## Goal

Cut the campaign to the minimum runs that still answer the research questions, and launch it with a **single command**. Achieve this with a **screen-then-confirm** design: cheaply rank all candidate algorithms by optimality gap vs the centralized optimum, promote the best 4, then run the confirmatory suites only on those 4 plus the two anchors.

Locked decisions (from brainstorming):
- Screening metric: **relative-to-best objective** on a mid-scale grid, runtime as tie-break. Centralized is *not* run in screening (it does not scale past n≈20 within the 120s Gurobi limit, so it cannot provide an optimal baseline at screening sizes).
- Promote **top 4** decentralized algorithms.
- Anchors always present: `centralized`, `hierarchical-madea`.
- Confirmatory seeds: **5** (parametrized, so cells can be topped up later).
- Ablation (e6): **n=50 only**.
- Scalability (e2): **add n=500**.
- Automation level: **full auto-chain**, including automatic survivor selection.

## Screening metric: relative-to-best (settled)

The model maximizes (`models/rmp.py:119`, `models/model.py` — `sense = pyo.maximize`), so higher objective is better. Centralized would be the true upper bound but does not scale to screening sizes (excluded past n≤20 in `build_e2`, `TimeLimit=120s`), so screening ranks by **relative deficit to the best algorithm on each instance**. For each instance `i = (cell, seed)`, let `obj_best(i) = max_a obj_a(i)` over the algorithms present:

```
reldef(a) = mean over i of  (obj_best(i) − obj_a(i)) / obj_best(i) · 100    (≥ 0; smaller = better)
```

`hierarchical-madea` participates in screening so `obj_best` is anchored to the strongest known method even when candidates are weak; it is excluded from the promotion pool (it is an anchor and always advances). The computation reuses the deviation logic in `compare_results.py:279` (`(baseline − obj)/obj·100`, joined on shared keys), with `obj_best` per instance as the reference instead of a fixed baseline column.

## Architecture

Three-stage pipeline behind one command, each stage independently resumable.

```
campaign run
  │
  ├─ Stage 1  SCREENING   define+materialize+run  paper-a-screening   (260 runs)
  │
  ├─ Stage 2  SELECT      postprocess → relative-to-best deficit per algo
  │                       → rank → batches/survivors.json (top 4)
  │                       guard: abort if too many runs failed / no objective
  │
  └─ Stage 3  CONFIRM     for suite in e1..e8:
                            define(survivors) + materialize + run     (~1640 runs)
```

**Resumability.** Per-batch manifests (`*.manifest.json`) already give run-level resume. A new `batches/campaign-state.json` records the last completed stage and, within Stage 3, the last completed suite. Re-running `campaign run` continues from there. Ctrl-C uses the existing dispatcher clean-stop.

### New component: `remote_experiments/campaign.py`

- `run_campaign(inventory, gurobi_license, ...)` — drives the three stages, reading/writing `campaign-state.json`.
- Stage helpers reuse existing functions: `get_suite`, `materialize_batch`, `run_batch`, `Manifest`. No new run machinery.
- Selection helper `select_survivors(screening_results_dir) -> list[str]`:
  1. Postprocess each screening run (existing `postprocessing.py` pipeline) to obtain `obj.csv`, `runtime.csv`, `termination_condition.csv`.
  2. Drop runs that failed / produced no valid objective (via `termination_condition.csv` and a missing/NaN obj check). Note: decentralized methods do not report `optimal`; only failed/no-objective runs are dropped, not converged-but-suboptimal ones.
  3. Per instance `(cell, seed)`, compute `obj_best` across all algorithms present (including `hierarchical-madea`), then each candidate's `reldef` = mean over instances of `(obj_best − obj_a)/obj_best·100` (reuse `compare_results` deviation logic).
  4. Rank ascending by mean `reldef`; break ties by median runtime.
  5. Return the top 4 **candidate** algorithms (anchors excluded from promotion).
- Guard: if fewer than `MIN_VALID_FRACTION` (default 0.8) of screening runs are valid, or fewer than 4 candidates produce any valid run, abort Stage 2 with a clear message instead of promoting.

### CLI change: `cli.py`

- Add `campaign` subcommand → `remote_experiments/campaign.py:run_campaign`, taking the same `--inventory`, `--gurobi-license`, `--instances`, `--results-dir`, project args as `run`.
- Add `--yes` (alias `--select all`) to **`run`** so it skips the `input()` prompt. `campaign` invokes the run stage non-interactively via the same code path. Interactive behavior unchanged when the flag is absent.

### Survivor injection: `definitions/paper.py`

- `ANCHORS = ("centralized", "hierarchical-madea")`.
- `DEFAULT_SURVIVORS` = current `("faas-madea", "faas-diffuse", "faas-powd", "faas-br-o")` (fallback for standalone `define` / tests).
- `_survivors()` reads `batches/survivors.json` (`{"survivors": [...]}`) if present, else `DEFAULT_SURVIVORS`.
- `FINAL = ANCHORS + _survivors()` (6 algorithms).
- Confirmatory builders use `FINAL` (or the survivors-only / centralized-gated variants) instead of hardcoded `ALL/REPRESENTATIVE/TRADEOFF` tuples.
- Because Stage 2 writes `survivors.json` before Stage 3 calls `define`, the builders pick up the real survivors at define time.

## Suites

### New: `paper-a-screening`
Algorithms: `hierarchical-madea` (reference for `obj_best`, not promotable) + 12 candidates (`faas-macro`, `faas-macro-v0`, `faas-madea`, `faas-diffuse`, `faas-powd`, `faas-br-s`, `faas-br-r`, `faas-br-o`, `faas-pg-s`, `faas-pg-r`, `faas-gcaa`, `plasma`) = 13. **No `centralized`** (does not scale to these sizes).
Grid: nodes ∈ {50, 100}, functions ∈ {2, 4}, planar-3, 5 seeds.
**13 × 4 × 5 = 260 runs.** Promotion pool = the 12 candidates (anchors auto-advance regardless of their deficit).

### Confirmatory suites (5 seeds, algorithm set = `FINAL` unless noted)

| Suite | Grid | Algorithms | Runs |
|---|---|---|---:|
| e1-quality-runtime | nodes{10,20,30} × func{2,4} × planar3 | FINAL (6) | 180 |
| e2-scalability | nodes{10,20,50,100,200,**500**} × func{2,4,8} × reg3 | survivors+hier (5); +centralized for n≤20 (6) | 480 |
| e3-topology | 6 topologies × n50 f4 | FINAL (6) | 180 |
| e4-robustness | 7 conditions × n50 f4 planar3 | FINAL (6) | 210 |
| e5-dynamics | 3 traces × n50 f4 (100 steps) | FINAL (6) | 90 |
| e6-ablation | 10 variants × **n{50}** × 2 topologies | hierarchical-madea only | 100 |
| e7-tradeoffs | 7 weights × 2 topologies | weight-tunable subset of FINAL (≤4) | ≤280 |
| e8-spatial-latency | nodes{20,50,100} × 2 modes | weight-tunable subset of FINAL (≤4) | ≤120 |

**Weight-tunable subset (e7/e8):** only algorithms with a per-section weight (`latency_weight`/`fairness_weight`) can vary it. Section map: `hierarchical-madea`→auction, `faas-madea`→auction, `faas-diffuse`→diffusion, `faas-powd`→powerd. Survivors outside this map are excluded from e7/e8 only (they remain in e1–e5). If none of the 4 survivors is tunable, e7/e8 fall back to `hierarchical-madea` alone (a warning is logged).

**Total: 260 (screening) + ~1640 (confirmatory) ≈ 1,900 runs** vs 12,630 (**≈6.6× fewer**, and the slowest cells removed by construction). Screening runs are at n∈{50,100} so each is heavier than a small-grid run, but 260 runs are still negligible against the confirmatory total.

## Data flow

```
screening solutions/<id>/  ──postprocess──▶  obj.csv, runtime.csv, termination_condition.csv
        │
        ▼  obj_best per (cell,seed); reldef per algo; drop failed/no-obj runs
   rank + tie-break ──▶ batches/survivors.json {"survivors":[a1,a2,a3,a4]}
        │
        ▼  read by paper.py _survivors()
   confirmatory define(FINAL) ──▶ materialize ──▶ run
```

## Error handling

- **Screening degenerate** → Stage 2 aborts before promoting; user re-runs screening or overrides `survivors.json` by hand.
- **Individual run failure** → tracked in the suite manifest as not-`SUCCEEDED`; `campaign run` re-selects only pending on resume (existing behavior).
- **Ctrl-C** → dispatcher stops in-flight jobs cleanly; `campaign-state.json` preserves stage progress.
- **`survivors.json` missing at Stage 3 define** (e.g., manual run out of order) → builder falls back to `DEFAULT_SURVIVORS` and logs a warning; no crash.

## Statistical note (non-blocking)

5 seeds is thin for confidence intervals. `CONFIRMATORY_SEEDS` stays a single-source constant so a specific noisy cell can be topped up to 15/30 by editing one tuple and re-running (`materialize`/`run` reuse existing instances and resume). Report with non-parametric summaries / CIs given n=5.

## Testing

- `test_select_survivors`: synthetic screening results (known objs, `obj_best` per instance) → asserts the 4 lowest-`reldef` candidates are chosen and runtime breaks ties; asserts failed/no-objective runs are dropped and the guard trips below the valid-fraction threshold; asserts `hierarchical-madea` is never promoted even if it has the best objective.
- `test_paper_suites_counts`: each confirmatory suite defines the exact run count in the table above given a fixed `survivors.json`; screening = 260.
- `test_survivors_fallback`: `define` with no `survivors.json` uses `DEFAULT_SURVIVORS` and stays runnable.
- `test_run_yes_flag`: `run --yes` skips the prompt and selects all pending (no stdin).
- `test_e7e8_tunable_filter`: a non-tunable survivor is excluded from e7/e8 but present in e1.

## Out of scope

- No changes to the algorithms themselves or their runners.
- No new plotting; analysis reuses `compare_results.py`.
- No automatic top-up of seeds (manual one-tuple edit is enough).
