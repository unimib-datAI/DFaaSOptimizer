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

Success: on deterministic regression runs, welfare and solution files are unchanged; other
artifacts match under the explicit timing normalization below. Budget-limited runs satisfy the
separate clock and stopping tests below. The duplication measured by `symilar -d 10` on the core
modules drops from 8.3% (2026 lines), and no `run()` of a migrated runner exceeds 80 lines.

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
- The handler's `emit` runs inside `step_time.untimed()`, but this alone does not exclude
  argument evaluation or logger bookkeeping. Keep `step_time.untimed()` around the complete
  logging call sites, including level guards and argument preparation, when a step is active.
  Replace the prints inside those blocks; do not remove the blocks wholesale.
- Arguments are formatted lazily (`log.debug("it = %d", it)`, ruff rule `G004` on the core
  modules). Guard expensive arguments, such as `state.y.tolist()`, with
  `if log.isEnabledFor(logging.DEBUG):` inside the untimed block. Lazy formatting avoids
  formatting disabled messages; it does not make a logger call or argument evaluation free.
- Before introducing the handler, make `step_time.untimed()` safe to nest: overlapping observer
  intervals count only once, including inside parallel phases and tasks and on exceptions.
  A handler called from an untimed observer must not subtract its duration a second time.
- Pool workers do not log; they return data (true today).
- `run.py` and the experiment scripts use the same module for their messages; `tqdm` progress bars
  stay (they are not logs).
- Replace the print-specific test with checks that the core runners contain no `print(` and no
  `log_stream`, and retain coverage of complete logging call sites being untimed. Behavioral
  tests must cover argument preparation and nested observers, not just handler emission.

## Time

- `step_time.now()` (`perf_counter`) is the only time source of the core modules; an AST test
  forbids `datetime.now`, `time.time`, `time.monotonic` and `perf_counter` outside `step_time`,
  except the timestamp that names the run folder (one helper).
- The dead `s`/`e`/`rt` measurements are deleted; the per-phase durations shown at DEBUG come
  from `step_time` (wall time of the phase).
- Excluded methods run on the step clock too. Sequential node sweeps in FaaS-MAPG-S/-R,
  better-response and `madea_pg` count their wall time minus observer code; any explicitly
  parallel initialization retains its existing phase accounting.
- PG hybrids retain the parallel phases of their base method and add the sequential refinement
  time. Exclusion from `methods.HARMONIZED` does not make a whole step sequential. For example,
  `hierarchical_frankwolfe.run(..., refine_welfare=True)` keeps the parallel local solves before
  refinement; do not replace their modelled duration with the sum of node task durations.
- The total step budget reads `step_time.current_modelled()`. Refinement receives the remaining
  modelled step budget and measures its own elapsed time relative to refinement entry, capped
  by its configured PG budget. Preserve coordination reserves and per-solver `TimeLimit` rules.
  Direct calls to `madea_pg.refine_solution` must still enforce their budget without an active
  step clock (use `step_time.now()` for that standalone sequential interval).
- The centralized model writes `runtime.csv` with the same columns (`algorithm_s` = solver time
  of the step, `parallel_phases` 0, `workers` = solver threads).
- One format for every newly written runner `runtime.csv`: the five columns of
  `step_time.COLUMNS`. Readers (`run.py`, `postprocessing.py`, `compare_results.py`,
  `benchmark_summary.py`, `remote_experiments/results_reader.py`) prefer `algorithm_s` and
  continue accepting historical `tot` and centralized model-name columns. Preserve separate
  aggregate postprocessing formats and objective-reading behavior in shared reader helpers.
  Existing result directories need no conversion; mixed old/new campaigns remain readable.

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
- Each decentralized runner keeps its public
  `run(config, parallelism, log_on_file=False, disable_plotting=False)` and becomes the skeleton
  call plus its `solve_step`. Registry entries and caller interfaces stay unchanged; `run.py`
  receives only the logging and runtime-reader changes described above.
- Migration one runner at a time, each with the output comparison below.

## Verification

- Before changing runners, extend `experiments/harmonization/capture_baseline.py`: it currently
  captures only welfare for `methods.HARMONIZED` and discards temporary run directories. Keep
  its existing welfare baseline interface, and add artifact capture for all harmonized methods,
  explicit callable mappings for excluded runners, and the centralized runner. Persist the
  artifacts before temporary directories are removed, with method, instance and configuration
  recorded. The centralized baseline needs an available real solver, not `missing_solver`;
  missing solver coverage must be reported explicitly.
- Use fixed seeds and iteration/sweep limits with nonbinding time budgets for deterministic
  before/after runs. Compare `obj.csv` and solution/checkpoint CSVs byte-for-byte, and compare
  the artifact inventory so method-specific diagnostic files cannot silently disappear.
- Normalize timing values in `termination_condition.csv` before comparison: replace only the
  numeric duration in `total runtime: ...` and elapsed/limit values in time-limit reasons with
  fixed markers. Preserve the reason, iteration counts, objective deviations and all other
  fields. Define timing-column exclusions per diagnostic file (e.g. `seconds` in
  `refinement.csv`); do not drop entire rows or arbitrary numeric fields. Compare `out.log`
  through the MACrO/MADEA parsers at verbose 2, excluding parsed timing fields and run-folder
  paths. Keep raw artifacts; normalization applies only to the comparison.
- Prove the comparison itself: two unchanged deterministic runs must compare equal despite
  different timings; a changed objective, solution, iteration count or stop reason must fail.
  Budget-limited runs are tested separately with a fake clock for stop reasons, refinement
  reserves and remaining `TimeLimit`; do not demand byte-identical decisions from real-time
  cutoffs. Existing welfare, runtime-contract, worker-independence and time-budget tests pass.
- New logging tests: level mapping; `out.log` written and handler detached/closed on an
  exception; expensive argument preparation skipped at disabled levels and excluded at enabled
  levels; emission and nested observers excluded exactly once, both sequentially and inside
  parallel tasks, including exceptions. With a fake clock, adding observer delay must change
  only `wall_s`, leaving modelled time, budget decisions and solutions unchanged.
- New hybrid timing test: two node tasks of 2 s and 3 s in one parallel phase followed by a 4 s
  sequential refinement contribute 7 s of algorithm time, not 9 s, before any measured overhead.
  Check the remaining refinement budget and retain standalone refinement budget coverage.
- Reader regressions cover new five-column files, historical `tot` and centralized model-name
  files, and a campaign mixing these formats. Existing objective and aggregate readers keep
  their contracts.
- Retain AST checks on clocks and prints; test the skeleton with a fake `solve_step` for folders,
  output files, checkpoints and runtime rows.

## Out of scope

Structural refactoring of `run.py`'s `results_postprocessing` and `run` (636 and 541 lines) and
`compare_results.py` is out of scope; the logging and compatible runtime-reader changes above
are in scope. The RL environments, `logs_postprocessing.py` (except keeping its inputs), and
new algorithm behavior are out of scope.

## To confirm

`decentralized_auction_modelbased.py` is imported by nothing (no registry entry, no import): delete
it rather than migrate it.
