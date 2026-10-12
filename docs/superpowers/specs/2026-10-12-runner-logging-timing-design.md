# One logging, one clock, one runner skeleton — design

Date: 2026-10-12. Status: proposed after the code review of `feat/frank-wolfe-relaxed-replicas`
(at `1ea47f8`), pending review of this document.

## Goal

- One way to log: every runner writes through `logging`, to `out.log` or stdout, with the verbosity
  of `config["verbose"]`; no `log_stream` passed around, no bare `print` in runners.
- One way to measure time: `utils/step_time.py` is the only clock; every runner (harmonized,
  excluded, centralized) writes `runtime.csv` in the same five columns.
- One runner skeleton: the set-up, the step loop and the output writing that 14-18 runners repeat
  live in one module; a runner supplies the decision of one step.

Success: the welfare of every method on the regression instances is unchanged, every output file
except timings is byte-identical, the duplication measured by `symilar -d 10` on the core modules
drops from 8.3% (2026 lines), and no `run()` of a migrated runner exceeds 80 lines.

## Facts from the review

- Logging: `print(..., file=log_stream)` (~110 calls in 17 runners), bare `print` (168 calls in 32
  files), `logging` only in `run.py`, `tqdm` in `run.py`; 117 `with step_time.untimed():` blocks,
  mostly around `if verbose > n:` prints; a test (`test_verbose_output_of_harmonized_runners_is_untimed`)
  enforces that every verbose block is untimed. `logs_postprocessing.py` parses the text of
  `out.log` of FaaS-MACrO and MADEA: those messages must keep their text.
- Time: `datetime.now()` (122 calls in 20 files, mostly dead `s`/`e`/`rt` from before the
  harmonization or wall-clock prints), `time.monotonic()` (26: verbose phase times in the
  hierarchical runners, the `madea_pg` deadline), `perf_counter` (`step_time`, `dp_solver`, pool
  tasks of MACrO, PLASMA welfare). Budgets of the harmonized methods already use the modelled time.
- `runtime.csv`: five columns (harmonized), one `tot` column of wall time (better-response, PG),
  one column per model (centralized); readers fall back between formats.
- Skeleton: the same prologue (configuration keys, run folder, `config.json`, log stream,
  `init_problem`, optional reference solution, neighbourhood and latency matrices) and epilogue
  (`join_complete_solution`, `save_solution`, `obj.csv`, `termination_condition.csv`, plot) in
  14-18 runners; `run()` is 200-540 lines in ~15 runners.
- MADEA2 swaps `define_bids`, `evaluate_bids`, `start_additional_replicas` into the modules of
  MADEA and one-shot with `setattr` during the run, and copies large parts of them.

## Logging

New module `utils/run_log.py`:

- `logger(name) -> logging.Logger`: the logger `dfaas.<name>`; modules call
  `log = run_log.logger(__name__)` once at import.
- `open_run_log(folder: str | None, verbose: int)`: context manager attaching one handler to the
  `dfaas` logger (propagation off): a file handler on `<folder>/out.log` when `log_on_file`,
  otherwise stdout; level `verbose` 0 → WARNING, 1 → INFO, ≥ 2 → DEBUG; format `%(message)s`
  (the text of `out.log` stays what the parsers expect). The handler is closed on exit, also on
  an exception.
- The handler's `emit` runs inside `step_time.untimed()`: logging is observer code by
  construction, so the `with step_time.untimed(): if verbose > n:` wrappers go away.
- Arguments are formatted lazily (`log.debug("it = %d", it)`, ruff rule `G004` on the core
  modules), so a disabled level costs nothing in the modelled time; a message whose arguments
  need computation is guarded by `if log.isEnabledFor(logging.DEBUG):`.
- Pool workers do not log; they return data (true today).
- `run.py` and the experiment scripts use the same module for their messages; `tqdm` progress bars
  stay (they are not logs).
- The test on verbose blocks is replaced by one asserting that the core runners contain no
  `print(` and no `log_stream`.

## Time

- `step_time.now()` (`perf_counter`) is the only time source of the core modules; an AST test
  forbids `datetime.now`, `time.time`, `time.monotonic` and `perf_counter` outside `step_time`,
  except the timestamp that names the run folder (one helper).
- The dead `s`/`e`/`rt` measurements are deleted; the per-phase durations shown at DEBUG come
  from `step_time` (wall time of the phase).
- Excluded methods (FaaS-MAPG-S/-R, better-response, the PG hybrids, `madea_pg`) run on the step
  clock too: a whole step is sequential, so `algorithm_s` is its wall time minus observer code; their
  time budgets (`madea_pg` deadline, `TimeLimit` passed to the refinement) read
  `step_time.current_modelled()`.
- The centralized model writes `runtime.csv` with the same columns (`algorithm_s` = solver time
  of the step, `parallel_phases` 0, `workers` = solver threads).
- One format for every `runtime.csv`: the five columns of `step_time.COLUMNS`; the readers
  (`run.py`, `postprocessing.py`, `compare_results.py`, `benchmark_summary.py`,
  `remote_experiments/results_reader.py`) drop their fallbacks.

## Runner skeleton

New module `utils/run_frame.py`:

- `RunContext`: configuration and its parsed keys (solver name and options, time limit,
  tolerance, iterations, patience, checkpoint interval, steps), run folder, logger, problem data
  (base data, traces, agents, graph, `Nn`, `Nf`, neighbourhood, latency) and the optional reference
  solution.
- `run(config, parallelism, *, log_on_file, disable_plotting, method, solve_step, outputs)`:
  creates the run folder and `config.json`, opens the log, builds the context, then for each step
  of `run_steps(config)`: current load and data, `step_time.start_step()`,
  `solve_step(ctx, t, data) -> StepResult`, `end_step`, decoding into the named solutions,
  checkpoints (`checkpoint_due`); at the end writes the solutions of `outputs`, `obj.csv` (column
  `method`), `termination_condition.csv`, `runtime.csv`, the plot; returns the run folder.
- `StepResult`: the combined solution per output name (e.g. `LSP`, `LSPc`), the objective, the
  termination string.
- Shared helpers for the iteration bodies: `Incumbent` (best solution by objective, with its
  iteration) and `snapshot` (combine, check feasibility, centralized objective; untimed), used by
  diffusion, PoD, dual, Frank-Wolfe, best-response, PG, GCAA.
- MADEA2 rules: the MADEA cycle and the one-shot runner take a `rules` argument (bid definition,
  bid evaluation, replica start; default MADEA's); MADEA2 passes its rules; no `setattr`, and the
  parts MADEA2 copied are shared.
- Each runner keeps its public `run(config, parallelism, log_on_file=False, disable_plotting=False)`
  (the registry and `run.py` do not change) and becomes the skeleton call plus its `solve_step`.
- Migration one runner at a time, each with the output comparison below.

## Verification

- Before the refactor, capture for every method in `methods.HARMONIZED` and the excluded runners
  the outputs on the regression instances (`experiments/harmonization/capture_baseline.py`):
  `obj.csv`, `termination_condition.csv` and the solution files. After each migrated runner, the
  same files must be byte-identical (timings excluded) and the existing harmonization tests
  (welfare regression, runtime contract, worker independence, time budget) must pass.
- `out.log` of MACrO and MADEA at verbose 2 is parsed by `logs_postprocessing` before and after
  with the same result (timings excluded).
- New unit tests: log level mapping, `out.log` written and closed on an exception, handler time
  removed from the modelled time; the AST tests on clocks and prints; the skeleton with a fake
  `solve_step` (folder, files, checkpoints, runtime rows).

## Out of scope

`run.py`'s `results_postprocessing` and `run` (636 and 541 lines), `compare_results.py`, the RL
environments, `logs_postprocessing.py` (except keeping its inputs), and new behaviour of any method.

## To confirm

`decentralized_auction_modelbased.py` is imported by nothing (no registry entry, no import): delete
it rather than migrate it.
