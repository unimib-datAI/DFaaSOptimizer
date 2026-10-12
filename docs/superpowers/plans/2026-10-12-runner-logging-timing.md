# Runner Logging, Timing and Shared Frame Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify runner logging, clocks and lifecycle without changing deterministic decisions, existing artifact contracts or the ability to read historical results.

**Architecture:** First capture a trustworthy artifact baseline, then fix the shared clock and introduce logging. Update readers before changing remaining runtime writers; extract the common frame through one pilot runner, then migrate each runner separately. Keep algorithm decisions and persistent algorithm state in the existing modules; reuse existing serialization, plotting and pool helpers.

**Tech Stack:** Python 3.10, standard-library logging/contextvars/dataclasses, existing NumPy/pandas/Pyomo/multiprocessing, pytest, Ruff and mypy. No new runtime dependencies.

**Spec:** [2026-10-12-runner-logging-timing-design.md](../specs/2026-10-12-runner-logging-timing-design.md)

## Global Constraints

- Execute in `DFaaSOptimizer/.worktrees/feat-frank-wolfe`, not `dfaas-edge-experiments`. Preserve the already modified design document. This plan does not authorize publishing or changing experiment results in place.
- Read `AGENTS.md`. Before editing existing symbols, run GitNexus context/impact, inspect direct callers and report HIGH/CRITICAL risks; before each commit run change detection and inspect the diff. Reuse this existing worktree.
- Use two-space indentation and existing dependencies. Keep public runner signatures, keyword variants, CLI defaults, registry keys and output labels; these labels are not necessarily the registry names.
- `COLUMNS = ("step", "algorithm_s", "wall_s", "parallel_phases", "workers")` for newly written **runner** runtime files. Historical and aggregate runtime formats remain readable.
- Preserve parallel phase accounting and sequential PG refinement; observers count zero, including nested logging. `step_time.now()` is the sole elapsed-time source in core modules. One helper may use calendar time for folder names.
- Logging: `verbose` 0 → WARNING, 1 → INFO, ≥ 2 → DEBUG; format `%(message)s`. Preserve parser-visible text, spaces and newlines. Keep explicit `verbose > 2` guards for existing high-volume bid dumps to preserve verbose-2 parser inputs.
- Preserve solver validation, feasibility checks, objective selection, tie rules, random-generator lifetime, warm pools, checkpoints and diagnostics. Do not weaken tests to accommodate changed decisions.
- Regression artifacts are immutable raw captures. Normalize only the comparison fields listed below. Real-time-budget runs use separate stopping tests.
- No structural rewrite of batch orchestration, postprocessing, RL or log parsers. No new method behavior. Every migrated public `run()` must fit in 80 lines; do not merely move a duplicated lifecycle into `_run()`.

## Review Focus

1. Logging inside an observer inside a node task, including an exception, must subtract elapsed time once and leave the next step clean (Tasks 2–3).
2. An old campaign resumed with new runner files must still aggregate runtime and objective columns correctly (Task 4).
3. PG refinement entered after parallel coordination, or called directly with no step clock, must honor its own cap and the remaining step budget (Task 5).
4. A multi-step PLASMA run and fixed-replica run must preserve engine/RNG state and reference-solution indexing (Tasks 6, 9).
5. A runner failure must close only its own log, reset clock state and release the pool; nested batch/run logs must not duplicate or leak messages (Tasks 3, 6, 10).

## Delivery order and file map

Three sequential milestones form one plan because later migrations depend on the baseline and timing contracts:

| Milestone | Tasks | Deliverable |
| --- | --- | --- |
| Evidence and shared infrastructure | 1–4 | Artifact oracle, safe clock/logging, compatible readers |
| Clock completion and shared lifecycle | 5–7 | PG/centralized timing, tested frame, explicit MADEA rules |
| Incremental migrations and final checks | 8–11 | All runners migrated with matching artifacts |

New production modules: `utils/run_log.py` (logging context), `utils/run_frame.py` (context, lifecycle and small iteration helpers). Extend `utils/step_time.py`; keep existing serializers in `run_centralized_model.py` and `utils/centralized.py`.

New verification modules: `experiments/harmonization/compare_artifacts.py`, `tests/test_runner_artifacts.py`, `tests/test_run_log.py`, `tests/test_run_frame.py`, `tests/test_runner_logging_timing_contract.py`. Extend `experiments/harmonization/capture_baseline.py`, existing clock, reader and family tests.

Core migration files: `run_centralized_model.py`, `run_faasmacro.py`, `run_faasmadea.py`, `decentralized_auction.py`, `decentralized_diffusion.py`, `decentralized_powerd.py`, `decentralized_dual.py`, `decentralized_frankwolfe.py`, `decentralized_gcaa.py`, `decentralized_bestresponse.py`, `decentralized_potentialgame.py`, `madea2.py`, `madea_pg.py`, `one_shot_pg.py`, `hierarchical_frankwolfe.py`, `hierarchical_auction/{runner,madea_runner,madea_cycles_runner,madea_level_cycles_runner}.py`, `plasma/{runner,welfare}.py`, `models/{model,dp_solver}.py`.

Commands below run from the worktree with `uv run --locked`. Set a writable `MPLCONFIGDIR` outside the checkout if needed. Baselines go under `/tmp/runner-logging-timing/`; do not add generated run folders to Git. Copy them to durable local storage if execution spans a machine restart.

## Task 1: Capture and compare artifacts before modifying runners

**Files:** Modify `experiments/harmonization/capture_baseline.py`; create `experiments/harmonization/compare_artifacts.py`, `tests/test_runner_artifacts.py`.

**Interfaces:** Keep `runners()`, `config_for(instance, out, steps=2)`, `welfare(runner, instance, parallelism=0)` and the positional welfare-JSON CLI working. Add `capture_artifacts(destination: Path, names: list[str] | None = None, solver: str = "glpk") -> None`; add `compare_artifacts(before: Path, after: Path, names: list[str] | None = None) -> list[str]` returning human-readable mismatches. CLI capture uses `--artifacts DIR [--method NAME ...] [--solver NAME]`; comparison takes `BEFORE AFTER [--method NAME ...]` and exits nonzero on differences or missing coverage.

- [ ] Write oracle tests with copied tiny CSV fixtures: timing-only changes compare equal; changing an objective, flow, iteration, reason or removing a diagnostic fails. Assert the original files are untouched, and old welfare callers still return the same structure. Run `uv run --locked pytest tests/test_runner_artifacts.py -q`; expect the new API tests to fail before implementation.
- [ ] Implement explicit baseline runner mappings: union of `methods.HARMONIZED`, all `methods.EXCLUDED` keys and `centralized`. Use current entry points below; do not infer callable names from labels or add excluded methods to campaign registries.

| Excluded labels | Existing callable |
| --- | --- |
| FaaS-MAPG-S / FaaS-MAPG-R | `decentralized_potentialgame.run_pg_s` / `run_pg_r` |
| faas-br-s / faas-br-r / faas-br-o | `decentralized_bestresponse.run_br_s` / `run_br_r` / `run_br_o` |
| MADEA-PG | `madea_pg.run` |
| one-shot-pg | `one_shot_pg.run` |
| one-shot2-pg / one-shot2-b-pg | `madea2.run_one_shot_pg` / `run_one_shot_b_pg` |
| MAFW-PG / MAFW-b-PG | `hierarchical_frankwolfe.run_flat_pg` / `run_flat_pg_b` |
| hierarchical-MAFW-PG | `hierarchical_frankwolfe.run_hierarchical_pg` |
| hierarchical-madea-pg / hierarchical-madea-level-cycles-pg | `madea_pg.run_hierarchical` (record these aliases explicitly) |
| hierarchical-one-shot-pg | `one_shot_pg.run_hierarchical` |

- [ ] Capture raw folders under `<destination>/<method>/<instance>/`, plus a manifest of source revision, configuration, solver, seed, workers, paths and coverage. Exercise all regression instances, two steps, fixed seeds, `parallelism=0`, bounded iterations/sweeps and nonbinding algorithm/refinement limits. Preserve normal configuration validation: use a large finite PG cap, not infinity. Use `missing_solver` only for known DP paths; probe an actual solve for centralized coverage and record unavailable solvers as incomplete coverage, not success.
- [ ] Implement comparison: solutions, checkpoints, objectives and materialized inputs byte-identical; runtime data compared by contract rather than values; termination strings normalize only `total runtime` and numeric elapsed/limit fields of time-limit reasons. Diagnostic CSV timing columns use an explicit filename/column allowlist. Compare MACrO/MADEA logs through existing parsers, excluding parsed times and the known output-folder path. Normalize only output-root paths in `config.json`; keep algorithm/solver settings intact. Compare file inventories; plots retain expected filenames and are checked by plotting smoke tests rather than encoded image bytes. Preserve the legacy welfare JSON baseline.
- [ ] Run oracle tests again, then capture twice before the refactor:

```bash
uv run --locked python experiments/harmonization/capture_baseline.py --artifacts /tmp/runner-logging-timing/before --solver glpk
uv run --locked python experiments/harmonization/capture_baseline.py --artifacts /tmp/runner-logging-timing/repeat --solver glpk
uv run --locked python experiments/harmonization/compare_artifacts.py /tmp/runner-logging-timing/before /tmp/runner-logging-timing/repeat
```

Expected: identical normalized decisions/artifacts and complete explicit method coverage. Resolve nondeterminism or inadequate caps before using this oracle; never regenerate the baseline to hide a later regression. Record the initial full-suite result with `uv run --locked pytest -q`.
- [ ] Inspect and commit only the capture/comparison code and fixture tests: `test: capture reproducible runner artifacts`.

## Task 2: Make observer intervals nest safely and expose one clock

**Files:** Modify `utils/step_time.py`, `utils/common.py`, `tests/test_step_time.py`.

**Interfaces:** Preserve existing timing APIs; add `now() -> float` wrapping the module's `perf_counter`, and `active() -> bool` for standalone refinement. Add `cancel_step(clock: StepClock) -> None` to clear an abandoned current step without publishing a runtime row. Keep `end_step(clock) -> dict` and `COLUMNS` unchanged.

- [ ] Add fake-clock tests named `test_nested_untimed_counts_union`, `test_nested_untimed_in_task`, `test_untimed_exception_resets_state`, `test_cancelled_step_does_not_leak`. Pin exact values: 1 s work plus an outer 5 s observer containing an inner 2 s observer yields `algorithm_s == 1`, `wall_s == 6`; repeat inside a node task. An exception followed by a new 3 s step must yield 3 s. Run `uv run --locked pytest tests/test_step_time.py -q` and confirm the new cases fail.
- [ ] Track nesting per context and account each elapsed observer interval once. Preserve task/phase accounting, including observer-only work and phase overhead; reset context tokens in `finally`. Test an observer spanning a phase as well as observers inside a phase, so neither work nor phase savings is subtracted twice. Do not mask accounting errors with a blanket `max(0, ...)`.
- [ ] Route the module's own elapsed readings through `now()` while preserving the existing `perf_counter` monkeypatch seam. `active()` reports the current clock without creating one; `cancel_step()` must not clear an unrelated clock. Add the existing-format `utils.common.run_timestamp() -> str` helper here, before clock migrations use it; include its deterministic formatting test.
- [ ] Run `uv run --locked pytest tests/test_step_time.py tests/test_harmonized_methods.py -q`; expect all new invariants and existing phase/worker tests to pass. Commit: `fix: exclude nested observer intervals once`.

## Task 3: Introduce logging with complete observer coverage

**Files:** Create `utils/run_log.py`, `tests/test_run_log.py`; extend `tests/test_step_time.py` only if a shared fixture is needed.

**Interfaces:** `logger(name: str) -> logging.Logger`; `open_run_log(folder: str | None, verbose: int)` context manager. `None` selects stdout; callers pass the run folder only for `log_on_file=True`.

- [ ] Add tests for levels 0/1/2/3, exact message formatting, file/stdout selection, exception cleanup, sequential runs and nested batch/run contexts. Assert stdout stays open, the outer handler/level is restored, and an inner run's messages do not also enter the outer log. Run `uv run --locked pytest tests/test_run_log.py -q`; expect missing-module failures initially.
- [ ] Implement `dfaas.<name>` loggers and one owned handler on `dfaas` with propagation disabled. Temporarily replace the owned outer handler for nested contexts and restore it in `finally`; close only the created handler. Wrap handler emission in `step_time.untimed()`.
- [ ] Add a fake-clock test around the entire logging call site: preparation adds 3 s and emission adds 2 s; both are inside an outer untimed block, so 1 s surrounding algorithm work remains 1 s while wall time becomes 6 s. At a disabled level the guarded preparation callable is never invoked. Repeat with nested phase/task and a raising preparation callable.
- [ ] Run `uv run --locked pytest tests/test_run_log.py tests/test_step_time.py -q`; expect no leaked handlers or timing state. Commit: `feat: add scoped untimed runner logging`.

## Task 4: Make readers accept new and historical formats before writer migration

**Files:** Modify `run.py`, `postprocessing.py`, `compare_results.py`, `benchmark_summary.py`, `remote_experiments/results_reader.py` only where needed; extend `tests/test_harmonized_methods.py`, `tests/test_remote_experiments_results_reader.py`, `tests/test_review_orchestration_regressions.py`, `tests/test_coverage_expansion.py`.

**Interfaces:** Preserve `run.runtime_series(frame, method_name)`, `benchmark_summary._single_column_sum(path)` and remote reader signatures. In raw-run runtime readers, select `algorithm_s` first; fall back to the historical model column or `tot` as appropriate. Do not change aggregate CSV readers or the objective branch of shared helpers.

- [ ] Add parameterized reader tests for five columns, `tot`, `LoadManagementModel`, `TightLoadManagementModel`, `SelfishLoadManagementModel`, empty/missing files, and a mixed-format campaign. Specifically assert a new centralized file uses `algorithm_s`, not `step`; preserve each reader's existing sum/mean semantics. Run the four listed test files and confirm new centralized/postprocessing cases fail.
- [ ] Apply minimal reader fixes. In `postprocessing.py`, derive the model label from existing artifact/config metadata rather than the first runtime column; preserve termination coloring. Keep `compare_results.py`'s aggregate formats intact if it already delegates raw reading.
- [ ] Run `uv run --locked pytest tests/test_harmonized_methods.py tests/test_remote_experiments_results_reader.py tests/test_review_orchestration_regressions.py tests/test_coverage_expansion.py -q`; expect old and new fixtures to pass. Commit: `fix: read mixed historical and unified runtimes`.

## Task 5: Complete PG, centralized and worker clock accounting

**Files:** Modify `madea_pg.py`, `decentralized_bestresponse.py`, `decentralized_potentialgame.py`, `run_centralized_model.py`, `models/model.py`, `models/dp_solver.py`, `run_faasmacro.py`, `plasma/welfare.py`; extend `tests/test_step_time.py`, `tests/test_madea_pg.py`, `tests/test_hierarchical_madea_pg_controls.py`, `tests/test_bestresponse_e2e.py`, `tests/test_potentialgame_e2e.py`, `tests/test_run_centralized_extended.py`.

**Interfaces:** Keep `refine_solution(data, solution, config, remaining_time=inf)` and its statistics; use Task 2's clock. Existing solver-return dictionaries keep their `runtime` key. New runtime writers use `step_time.write_runtime(folder, rows)`.

- [ ] Add fake-clock hybrid test: tasks of 2 s and 3 s in one phase followed by 4 s sequential work yield `algorithm_s == 7`, `wall_s == 9`, `parallel_phases == 1`; with a 10 s total budget, refinement enters with 7 s remaining. Test PG with an active step and standalone, zero budget, fractional GLPK limits and observer delay. Run the listed tests; newly specified clock/API and CSV cases must fail before edits.
- [ ] Replace direct elapsed clocks in shared solver/worker paths with `step_time.now()`; do not replace worker task durations with `current_modelled()` because workers have no parent step clock. Preserve external solver runtime semantics in `models/model.py`: measure the solver call, not instance generation or result serialization. Its failure-diagnostic filenames also use the shared `run_timestamp()` helper; retain `.ilp`/`.err` creation and original errors.
- [ ] In refinement, snapshot `active()` at entry and measure elapsed from a modelled-time start when active, otherwise from `now()`. Subtract this local elapsed from `min(configured_pg_budget, remaining_time)`; retain solver-specific integer limits and no-clock direct callers. Keep parallel initialization and coordination phases in all hybrids.
- [ ] Migrate best-response/PG runtime writers: sequential sweeps use the step clock, observers are untimed, existing parallel initialization stays instrumented. Keep existing budget configuration keys/defaults; this task changes the clock, not the user configuration schema. Update old writer assertions for `tot`, but retain historical reader fixtures.
- [ ] Centralized runtime rows use the reported solver-call duration as `algorithm_s`, full decision wall time as `wall_s`, `parallel_phases=0`. `workers` reports effective solver threads from backend metadata when available; use an explicit configured positive thread count otherwise, 1 for a known single-thread backend, and 0 when automatic/unknown (never pretend the decentralized `parallelism` is the thread count). Test each case without depending on a commercial license.
- [ ] Run `uv run --locked pytest tests/test_step_time.py tests/test_madea_pg.py tests/test_hierarchical_madea_pg_controls.py tests/test_bestresponse_e2e.py tests/test_potentialgame_e2e.py tests/test_run_centralized_extended.py -q`, plus artifact comparisons for the changed families using Task 1. Expect unchanged deterministic outputs and updated runtime contracts. Commit: `refactor: unify solver and refinement clocks`.

## Task 6: Extract the shared frame through the diffusion runner

**Files:** Create `utils/run_frame.py`, `tests/test_run_frame.py`; modify `decentralized_diffusion.py`, `utils/common.py`, `tests/test_diffusion_e2e.py`, `tests/test_zero_load_runners.py`.

**Interfaces:**

- Consume `utils.common.run_timestamp() -> str` from Task 2: the only core calendar-clock call, retaining `%Y-%m-%d_%H-%M-%S.%f`.
- `RunContext`: config, parallelism, folder, logger, solver_name/options/general_options, time_limit, tolerance, max_iterations, patience, checkpoint_interval, steps, base_data, traces, agents, graph, Nn/Nf, neighborhood, latency, optional reference solution. Reuse existing value types and defaults; algorithm-specific options remain in their modules. Keep complete-solution accumulators and per-output objective history here for existing final plot adapters.
- `open_context(config: dict, parallelism: int, *, log_on_file: bool)`: context manager yielding `RunContext`, owning folder creation, config save, logger and problem/reference setup. The frame consumes it now; centralized generation-only consumes it in Task 10 without starting a step or writing solver artifacts.
- `StepResult(solutions: dict[str, dict], objective: float, termination: str, runtime: dict | None = None)`: each named solution contains existing `x,y,z,r,xi,rho,U` arrays. Select these dictionaries from existing combined solutions; do not introduce another array model. `runtime` is used only for the centralized `algorithm_s`/`workers` override; the frame owns step/wall/phase fields.
- `run(config, parallelism, *, log_on_file, disable_plotting, method, solve_step, outputs) -> str`, where `solve_step(ctx, t, data) -> StepResult`. `outputs` maps each existing output name to `checkpoint: bool` and `plot_filename: str | None`; this captures actual current differences rather than imposing checkpoint/plot output on every solution.
- `Incumbent(objective: float, solution: dict | None = None, iteration: int = -1)` with `consider(solution: dict, objective: float, iteration: int) -> bool`: strictly greater welfare replaces and deep-copies under `untimed`; ties keep the existing incumbent. Use only for matching centralized-welfare selection; keep local minimization/potential rules separate.
- `snapshot(data: dict, *args, **kwargs) -> tuple[dict, float]`: inside `untimed`, forward the existing `combine_solutions` arguments, check feasibility and compute centralized welfare; preserve the original feasibility exception. Use existing helpers through function-local imports where needed to avoid runner/frame import cycles.

- [ ] Add frame tests with a fake `solve_step`: verify exact `obj.csv` label, legacy termination CSV index/header, configured checkpoint selection, five runtime columns, final saves, plotting enabled/disabled, non-unit `run_steps`, config immutability and fixed-reference indexing. Assert returned array dictionaries reach the existing decoder unchanged. Add incumbent tests for first negative objective, strict ties, deep-copy independence and infeasible snapshots. Run `uv run --locked pytest tests/test_run_frame.py -q`; expect failure before the new module exists.
- [ ] Implement setup and loop using existing `init_problem`, `run_steps`, `checkpoint_due`, decoders and serializers. Start the clock after input-load preparation; finish before decode/checkpoints/plots. On solve failure call `cancel_step` in `finally`, propagate the error and close the log. The existing public runner's `parallel_solver_run` decorator retains pool ownership; do not create a second pool in the frame.
- [ ] Reuse centralized setup helpers without moving their public definitions. Import the frame lazily from runner entry points where top-level imports would create cycles. Do not eagerly require auction options for PLASMA or a decentralized reference for centralized generation.
- [ ] Migrate diffusion alone: preserve distinct local and centralized incumbents, `LSP`/`LSPc`, `sp.png`, exact method label and stopping text. Extract its decision into `_solve_step(ctx, t, data) -> StepResult`; preserve all information used by the objective and solution validators. Change runner-owned test monkeypatches to the actual shared helper owner when the extraction moves a call.
- [ ] Add and pass `test_frame_failure_then_success_is_clean`: inject a failure, then run again and assert isolated logs, clock and pool cleanup. Run `uv run --locked pytest tests/test_run_frame.py tests/test_diffusion_e2e.py tests/test_zero_load_runners.py tests/test_warm_pool_runners.py -q`, then capture/compare `--method faas-diffuse`. Commit: `refactor: share runner lifecycle through diffusion`.

## Task 7: Pass MADEA rules explicitly

**Files:** Modify `run_faasmadea.py`, `decentralized_auction.py`, `madea2.py` and their direct cycle callers; extend `tests/test_madea_value_replicas.py`, `tests/test_madea_cycle_order.py`, `tests/test_madea_reoptimization.py`.

**Interfaces:** Add keyword-only `rules: dict | None = None` to MADEA's run/cycle and one-shot run. Required keys are `define_bids`, `evaluate_bids`, `start_additional_replicas`, using the existing callable signatures. Resolve `None` to current module functions at call time, preserving test monkeypatches. Reuse `madea2._RULES`.

- [ ] Test a MADEA2 run followed by MADEA, exception recovery and interleaved rule sets: the original module functions retain identity and each cycle invokes only its supplied rules. Run the listed files; the no-mutation case should fail under the current swapping context.
- [ ] Thread the rules through the shared cycle and one-shot helpers. Remove `_madea2_rules` and its `setattr` mutation; MADEA2 wrappers pass `_RULES`. Share identical portions of copied bid/evaluation logic while keeping the differing bid-value and replica-allocation rules explicit. Preserve decorators and assign each parallel phase exactly once.
- [ ] Run the three listed files and `tests/test_one_shot_pg.py`, then compare all MADEA2 and one-shot2 artifacts, including `-b` and PG variants. Commit: `refactor: inject MADEA rules without global mutation`.

## Task 8: Migrate iterative and auction runners one at a time

**Files:** The individual runner and its associated tests in the table; `utils/run_frame.py` only for a difference proven by a real second caller.

**Interfaces:** Task 6's `RunContext`, `StepResult`, `run`, `snapshot`, `Incumbent`; Task 7's rules. Preserve current public signatures including `v0`, PG and engine variants.

For **each row**, complete its test/capture/commit cycle before starting the next:

| Order | Runner | Main existing regression tests / artifact requirements |
| --- | --- | --- |
| 1 | `decentralized_powerd.py` | `tests/test_powerd_e2e.py`; preserve seeded sampling |
| 2 | `decentralized_dual.py` | `tests/test_dual_e2e.py`; preserve price/state updates |
| 3 | `decentralized_frankwolfe.py` | `tests/test_frankwolfe_wiring.py`, `tests/test_frankwolfe_helpers.py`; preserve fixed replicas and `frankwolfe_stats.csv` |
| 4 | `decentralized_gcaa.py` | `tests/test_gcaa_e2e.py`; preserve bid dump levels and winner ordering |
| 5 | `decentralized_bestresponse.py` | `tests/test_bestresponse_e2e.py`; all S/R/O variants |
| 6 | `decentralized_potentialgame.py` | `tests/test_potentialgame_e2e.py`; all S/R variants and potential objective |
| 7 | `run_faasmadea.py` | `tests/test_madea_cycle_order.py`, `tests/test_madea_pg.py`; baseline/MADEA2/PG |
| 8 | `decentralized_auction.py` | `tests/test_one_shot_pg.py`, `tests/test_madea_value_replicas.py`; one-shot/one-shot2/-b/PG |
| 9 | `run_faasmacro.py` | `tests/test_run_faasmacro_helpers.py`, `tests/test_logs_postprocessing.py`; normal/v0, LSP/LRMP/LSPc/LRMPc and both plots |

- [ ] Add a parameter for this runner to `tests/test_runner_logging_timing_contract.py`: assert its exact method label, output inventory, normalized baseline match and no active clock after return/failure. Preserve per-family assertions; run that case before extraction to establish the baseline, and add the new structural assertion (shared frame and no print/log_stream) that fails.
- [ ] Move only the lifecycle into the frame. Keep decision code in `_solve_step`, including method-specific validation and tie rules. Use shared snapshot/incumbent only where semantics match. Convert prints to lazy logger calls inside full untimed blocks; remove log_stream arguments and update every internal caller/test. Replace live phase clocks with `now()`, delete genuinely unused measurements.
- [ ] Keep diagnostic rows in a closure shared by this run's step callback; write their existing CSVs after the frame returns. A persistent object must be initialized once per run inside an untimed setup block, not recreated each step. Preserve MACrO's extra RMP objective series and `rmp.png` with a small runner-specific final plot helper using the retained context; do not add a generic hook registry.
- [ ] Run this row's tests, clock/log tests and its artifact comparison. Parameterize matching PG/wrapper variants too. Commit each row independently: `refactor: migrate <runner> to shared frame`.

## Task 9: Migrate hierarchical and stateful PLASMA runners

**Files:** `hierarchical_frankwolfe.py`, `hierarchical_auction/{runner,madea_runner,madea_cycles_runner,madea_level_cycles_runner}.py`, `plasma/{runner,welfare}.py`; existing hierarchical/PLASMA tests and `tests/test_runner_logging_timing_contract.py`.

**Interfaces:** Same frame and result as Task 8. Existing engine classes, partial entry points and refinement wrappers retain their signatures. The thin `madea_level_cycles_runner` delegates to the migrated cycle runner.

- [ ] Migrate in this order, separately: hierarchical Frank-Wolfe, hierarchical runner, legacy MADEA runner, cycle runner, PLASMA. For each, add a failing frame/logging contract assertion then extract the lifecycle as in Task 8.
- [ ] Preserve hierarchical coordination reserves, cycle stopping history and `refinement.csv`/`frankwolfe_stats.csv`. Rerun `tests/test_hierarchical_frankwolfe.py`, `tests/test_hierarchical_runner.py`, `tests/test_hierarchical_madea_runner.py`, `tests/test_hierarchical_madea_pg_controls.py`, `tests/test_hierarchical_one_shot_pg.py` against their respective migrations.
- [ ] Add `test_plasma_frame_preserves_engine_across_steps`: two different nonzero loads, fixed seed, W other than 1; assert one engine initialization, continuing RNG/node/message state, correct arrival scaling, `plasma_messages.csv` and welfare diagnostics. Keep load conversion outside the decision timer and feasibility/objective observers untimed. Run `tests/test_plasma_e2e.py`, `tests/test_step_time.py`, `tests/test_warm_pool_runners.py` and the new case.
- [ ] Capture/compare every registry alias implemented by the migrated file, including PG variants; commit each migrated runner independently. Do not recreate reference data merely because aggregate welfare still matches.

## Task 10: Centralized lifecycle and batch/experiment logging

**Files:** Modify `run_centralized_model.py`, `run.py`; active Python entry points under `experiments/madea_pg_compact/` and `experiments/harmonization/`; extend `tests/test_run_centralized_extended.py`, `tests/test_review_orchestration_regressions.py`, `tests/test_run_log.py`.

**Interfaces:** Preserve centralized `run(config, log_on_file=False, disable_plotting=False, generate_only=False, filter_traces=False)`. Use the frame for solve runs, preserving model name columns in objective/termination outputs; centralized termination uses `index=False` and model-name header, unlike decentralized termination. Select that known serialization difference from the centralized result/output identity, not a new general-purpose CSV configuration system.

- [ ] Add tests for default/tight/selfish models, generate-only with zero solver calls, generate-only plus filter-traces (still solves), and log cleanup after initialization/solver failure. Verify periodic and final plot paths and no-solution handling. Run these new cases before extraction to establish old behavior; add shared-frame/clock assertions that initially fail.
- [ ] Extract centralized solve into a callback producing named array solutions and Task 5's runtime override. Reuse `open_context` for generation-only; skip solve/artifact emission exactly as today. Preserve `filter_traces` after completed saves. In the frame, after decoding the current step, honor centralized periodic plots at `0 < t < steps.stop - 1 and t % plot_interval == 0`, using the existing `<model>/<t>_plot/<model>.png` path; plots remain outside the clock. Extend frame tests for this actual second serialization shape rather than changing legacy output formats.
- [ ] Convert batch and experiment status messages through `run_log`, with an outer stdout context; each nested run restores it. Keep `tqdm`, structured JSON/report stdout and process exit codes compatible; command-line report tools use INFO with an enabled stdout context so their output is not suppressed by a runner's verbose=0 setting. List the active `.py` entry points that emit messages with `rg -l 'print\(|logging\.' run.py experiments --glob '*.py'` and migrate each match; exclude archived result snapshots. Campaign UTC deadlines/timestamps are orchestration time, outside the core elapsed-clock ban.
- [ ] Run `uv run --locked pytest tests/test_run_centralized_extended.py tests/test_centralized_feasibility.py tests/test_review_orchestration_regressions.py tests/test_run_log.py tests/test_runner_artifacts.py -q` and centralized artifact comparison with the same solver/config as the baseline. Smoke the three documented CLI `--help` commands and one small plotting-enabled run. Commit centralized and orchestration changes separately.

## Task 11: Enforce contracts and finish verification

**Files:** Modify `tests/test_runner_logging_timing_contract.py`, `tests/test_harmonized_methods.py`, `tests/test_step_time.py`, `pyproject.toml`, `README.md`, `tests/REGRESSIONS.md`; conditionally delete `decentralized_auction_modelbased.py` after the usage check below.

**Interfaces:** One explicit `CORE_FILES` list in the contract test covers the file map above plus `utils/run_frame.py`/`utils/run_log.py` and any algorithm helpers found to own clocks. Exclude tests, RL, campaign/report tooling and archived snapshots. This list also defines the runner-source lint/duplication measurement scope; do not add a production registry merely for tests.

- [ ] Add AST tests that resolve import aliases and reject direct `datetime.now`, `time.time`, `time.monotonic`, `perf_counter` in core except `utils.step_time` and `utils.common.run_timestamp`. Reject print calls and log_stream names in migrated runners; require full-call untimed coverage. Replace the old print-specific verbose assertion only after behavioral tests pass. Enable Ruff `G004` for the concrete core file scope, without suppressing real logging calls elsewhere.
- [ ] Search imports, dynamic string references, CLI docs, shell commands and experiment configs for `decentralized_auction_modelbased`; verify GitNexus callers too. It has a standalone CLI, so absence of imports alone is insufficient. If no supported invocation remains, remove it and obsolete references; otherwise record the concrete caller and migrate its logging/clock/frame using Task 8's cycle. Do not delete based only on the original tentative assertion.
- [ ] Extend the runtime schema contract to excluded and centralized runners, with family-specific phase/worker expectations (sequential phases may be zero). Keep the existing harmonized worker-independence tests and PG tests; do not require every excluded method to acquire a worker pool.
- [ ] Run the final capture and compare against `before` for all methods, and rerun worker independence with `parallelism=0` and `2`. Add a verbose-0/2 pair with simulated slow observers; deterministic decisions and fake-clock budget stops must match. Verify final MACrO/MADEA parser results explicitly.
- [ ] Run `uv run --locked ruff check .`, `uv run --locked mypy`, `uv run --locked pytest -q`, and `git diff --check`. Expect no new failures; report unavailable real solvers and pre-existing failures by name rather than claiming complete coverage.
- [ ] Measure `run()` line spans via AST and inspect `_run`/step helpers for relocated lifecycle duplication. Rerun `symilar -d 10` with the same explicit file set/options used for the baseline; record before/after counts and command in the implementation report. If the original 8.3%/2026-line scope cannot be recovered, measure both revisions with the identical reconstructed core list and report that comparison, not a misleading percentage. Do not add a runtime dependency for this measurement.
- [ ] Document five-column runtime semantics, historical compatibility, verbosity and the regression capture commands in README/regression map. Inspect GitNexus change detection and `git diff --stat`; commit tests/docs/remaining cleanup as `test: enforce shared runner logging and timing contracts`.

## Completion evidence

The implementation report links the preserved baseline and comparison manifests, lists migrated runners/aliases, records test outcomes and skipped solver coverage, and gives actual duplication/line-count measurements. Completion requires matching deterministic artifacts, correct budget/observer accounting and readable mixed-format campaigns; smaller wrapper functions alone are insufficient.

This document is the implementation plan only. The current planning task does not run the refactor or recapture experiment baselines.
