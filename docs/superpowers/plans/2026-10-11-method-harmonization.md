# Method Harmonization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every parallel-by-design decentralized method solves local problems by DP on one persistent worker pool and reports, per step, a modelled distributed algorithm time (parallel phase = longest task, sequential part = wall time, observer code excluded) through one shared clock; time budgets use that clock; `runtime.csv` has one format; one registry lists harmonized and excluded methods.

**Architecture:** A small module `utils/step_time.py` holds a per-step clock in a context variable. Parallel phases are opened where per-node work happens (shared helpers: `solve_subproblem`, `compute_social_welfare`, bid/assignment functions, Frank-Wolfe, PLASMA batches) and record each node's task duration (measured in the worker for pool tasks). Runners only start/end the clock per step, read `clock.modelled()` for time-based stopping, mark observer code `untimed()`, and write `runtime.csv` through one helper. Every existing ad hoc `total_runtime` accumulation is removed.

**Tech Stack:** Python 3.10, numpy, pandas, multiprocessing (existing `parallel_solver_session` pool), pytest.

**Spec:** `docs/superpowers/specs/2026-10-11-method-harmonization-design.md`

## Global Constraints

- Work in `~/Downloads/DFaaSOptimizer/.worktrees/feat-frank-wolfe` (branch `feat/frank-wolfe-relaxed-replicas`).
- This repository's `CLAUDE.md` applies: before editing a function run GitNexus impact (`npx gitnexus impact <symbol>`, or the MCP tool) and report direct callers and risk; stop and tell the user on HIGH/CRITICAL; run `npx gitnexus detect_changes` (or `git diff --stat` if unavailable) before each commit.
- Indentation 2 spaces (repository style). No new dependencies.
- Decisions must not change: same welfare per step as before for every harmonized method on the regression instances (Task 0 baseline), with any number of workers. Only timing outputs and the time-budget key change.
- `algorithm_s` = modelled time; `wall_s` = wall time of the step decision. Formula: `modelled = wall − Σ untimed − Σ phase_wall + Σ longest_task`.
- Time budget key: `solver_options.general.algorithm_time_limit` (seconds per step, modelled; default unlimited). `TimeLimit` is no longer read by harmonized runners as an algorithm budget.
- `parallelism`: −1 = all cores (CLI default of every harmonized runner), 0 = in-process, k = k workers.
- Excluded methods (FaaS-MAPG-S/R, faas-br-s/r/o, every `*-PG` hybrid) are not edited, except where they share a helper (they then simply get the new timing for free).
- No push without the user's permission.

## Review Focus

- A parallel phase opened inside another one (double instrumentation of a helper called from an instrumented helper) must raise, not silently mis-time (Task 1 test `test_nested_phase_raises`; Task 9 contract run catches real nestings).
- Observer code that stays timed inflates `algorithm_s` without any test failing; the contract test checks `algorithm_s ≤ wall_s` but not this. Task 9 adds `test_observer_time_is_excluded` on MADEA with an artificially slow `check_feasibility`.
- A method whose time-based stop still reads `TimeLimit` or wall time keeps machine-dependent results (Task 9 `test_algorithm_time_limit_stops_every_iterative_method`).
- Results depending on the number of workers (PLASMA in-process path uses a different turn order than the pool path today) (Task 9 contract compares welfare for 0 and 2 workers).
- `runtime.csv` readers still expecting `tot` break silently in reports (Task 8 tests `benchmark_summary` and `families_extended` on a new-format folder).

---

## File Structure

- Create `utils/step_time.py`: clock, phases, tasks, untimed blocks, budget key, workers, `runtime.csv` writer.
- Create `methods.py`: `HARMONIZED`, `EXCLUDED`.
- Modify `run_faasmacro.py`: pool batch timing, phases in `solve_subproblem` and `compute_social_welfare`, runner loop.
- Modify `run_faasmadea.py`: phases in `compute_residual_capacity`, `define_bids`, `evaluate_bids`, `start_additional_replicas`; `run_madea_cycle` and `run` loops.
- Modify `madea2.py` (`define_bids`, any own per-node helper), `decentralized_auction.py`, `decentralized_gcaa.py` (`resolve_gcaa_round`), `decentralized_diffusion.py` (`define_assignments`, `evaluate_assignments`), `decentralized_powerd.py` (`sample_assignments`), `decentralized_dual.py` (`dual_coordination_round`), `decentralized_frankwolfe.py` (`frankwolfe_assignments`, `round_flows`, repair), `hierarchical_frankwolfe.py`, `hierarchical_auction/runner.py`, `hierarchical_auction/madea_runner.py`, `hierarchical_auction/madea_cycles_runner.py`, `plasma/welfare.py`, `plasma/runner.py`, `plasma/core` engine (negotiation loop).
- Modify readers: `benchmark_summary.py`, `compare_results.py`, `experiments/madea_pg_compact/families_extended.py`, `experiments/madea_pg_compact/extend_frankwolfe.py`, `experiments/madea_pg_compact/catalog_campaign.py`.
- Create tests: `tests/test_step_time.py`, `tests/test_harmonized_methods.py`; data `tests/data/harmonization/` (two small instances, baseline JSON); script `experiments/harmonization/capture_baseline.py`.

---

### Task 0: Regression baseline (before any change)

**Files:**
- Create: `experiments/harmonization/capture_baseline.py`
- Create: `tests/data/harmonization/<instance>/...` (2 instances), `tests/data/harmonization/baseline.json`

**Interfaces:**
- Produces: `baseline.json` = `{method: {instance: [welfare_step0, welfare_step1]}}` for every method of the spec's harmonized table, `parallelism=0`, `solver_name="missing_solver"`, 2 steps. Used by Task 9.

- [ ] **Step 1: Copy two small instances**

```bash
mkdir -p tests/data/harmonization
cp -r "$(ls -d ~/Downloads/dfaas-edge-experiments/catalog/cilic-europe30/instances/*v1*huawei-d1-f5*-s1-s1 | head -1)" tests/data/harmonization/
cp -r "$(ls -d ~/Downloads/dfaas-edge-experiments/catalog/ether-urban/instances/ether-urban-n40-s1-knn__huawei-d1-f5*-s1-s1 | head -1)" tests/data/harmonization/
du -sh tests/data/harmonization
```

Expected: two folders, a few MB in total. (n30 measured edge network; n40 heterogeneous devices.)

- [ ] **Step 2: Write the capture script**

Create `experiments/harmonization/capture_baseline.py`:

```python
"""Welfare per step of every harmonized method on the regression instances.

Run before and after the harmonization: decisions must not change.
  uv run python experiments/harmonization/capture_baseline.py tests/data/harmonization/baseline.json
"""
import json
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "madea_pg_compact"))

import pandas as pd  # noqa: E402

DATA = ROOT / "tests" / "data" / "harmonization"
CONFIG = ROOT / "experiments" / "madea_pg_compact" / "catalog_config_overnight.json"


def runners():
  try:
    from methods import HARMONIZED
    return HARMONIZED
  except ImportError:  # before Task 7: the same names, resolved by hand
    from functools import partial
    import decentralized_auction, decentralized_diffusion, decentralized_dual, decentralized_frankwolfe
    import decentralized_gcaa, decentralized_powerd, hierarchical_frankwolfe, madea2, run_faasmacro, run_faasmadea
    from hierarchical_auction import madea_cycles_runner, madea_level_cycles_runner, madea_runner, runner
    from hierarchical_auction.iterative_engine import IterativeHierarchicalAuctionEngine
    from plasma import welfare
    from plasma.runner import run as plasma_run
    return {
      "MADEA": run_faasmadea.run, "MADEA-one-shot": decentralized_auction.run,
      "MADEA2": madea2.run, "MADEA2-b": madea2.run_b,
      "MADEA2-one-shot": madea2.run_one_shot, "MADEA2-b-one-shot": madea2.run_one_shot_b,
      "FaaS-MAFW": decentralized_frankwolfe.run, "FaaS-MAFW-b": hierarchical_frankwolfe.run_mafw_b,
      "hierarchical-MAFW": partial(hierarchical_frankwolfe.run, hierarchy=True, refine_welfare=False),
      "hierarchical-madea": madea_level_cycles_runner.run,
      "hierarchical-madea-cycles": madea_cycles_runner.run,
      "hierarchical-madea-legacy": madea_runner.run,
      "hierarchical-one-shot": partial(runner.run, engine_class=IterativeHierarchicalAuctionEngine),
      "hierarchical": runner.run,
      "PLASMA": plasma_run, "PLASMA-Welfare": welfare.run,
      "faas-macro": run_faasmacro.run, "faas-macro-v0": partial(run_faasmacro.run, v0=True),
      "faas-diffuse": decentralized_diffusion.run, "faas-powd": decentralized_powerd.run,
      "faas-gcaa": decentralized_gcaa.run, "faas-dual": decentralized_dual.run,
    }


def config_for(instance: Path, out: Path, steps: int = 2) -> dict:
  config = json.loads(CONFIG.read_text())
  full = json.loads((ROOT / "config_files" / "eval_full.json").read_text())["solver_options"]
  config["solver_options"] = {**{k: v for k, v in full.items() if k != "general"}, **config["solver_options"]}
  config["solver_options"]["frankwolfe"] = dict(iterations=400, kappa=10.0, smoothing=1.0, max_rounds=20)
  config["solver_options"]["plasma_welfare"] = dict(W=1., rounds_per_step=20, epsilon=1e-6,
                                                    hb_latency_rounds=1, hb_loss=0., staleness_rounds=3)
  config.update(solver_name="missing_solver", seed=1, max_steps=steps, min_run_time=0,
                max_run_time=steps, run_time_step=1, checkpoint_interval=1,
                base_solution_folder=str(out))
  config["limits"].update(instance_type="materialized", path=str(instance.resolve()))
  return config


def welfare(runner, instance: Path, parallelism: int = 0) -> list[float]:
  with tempfile.TemporaryDirectory() as tmp:
    folder = Path(runner(config_for(instance, Path(tmp)), parallelism, log_on_file=True, disable_plotting=True))
    return [float(v) for v in pd.read_csv(folder / "obj.csv").iloc[:, -1]]


def main(output: Path) -> None:
  instances = sorted(p for p in DATA.iterdir() if (p / "metadata.json").exists())
  result = {name: {i.name: welfare(run, i) for i in instances} for name, run in runners().items()}
  output.write_text(json.dumps(result, indent=1))
  print(json.dumps(result, indent=1))


if __name__ == "__main__":
  main(Path(sys.argv[1]))
```

- [ ] **Step 3: Capture**

Run: `uv run python experiments/harmonization/capture_baseline.py tests/data/harmonization/baseline.json`
Expected: 22 methods × 2 instances, two welfare values each, no exception. If a method fails, stop and report it to the user (method, error): the baseline must cover every harmonized method.

- [ ] **Step 4: Record the current test-suite state**

Run: `uv run pytest -q 2>&1 | tail -30 > tests/data/harmonization/pytest-before.txt; tail -3 tests/data/harmonization/pytest-before.txt`
Expected: the summary line of the suite before any change (pre-existing failures, if any, are listed and compared in Task 9).

- [ ] **Step 5: Commit**

```bash
git add experiments/harmonization/capture_baseline.py tests/data/harmonization
git commit -m "harmonization: regression baseline of the harmonized methods

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 1: The step clock

**Files:**
- Create: `utils/step_time.py`
- Test: `tests/test_step_time.py`

**Interfaces:**
- Produces (used by every later task):
  - `start_step() -> StepClock`; `end_step(clock: StepClock, workers: int) -> dict` with keys `algorithm_s, wall_s, parallel_phases, workers`
  - `StepClock.modelled() -> float`, `StepClock.wall() -> float`
  - `parallel_phase()` context manager → `Phase` (`add(key, seconds)`, `uniform(nodes: int)`); raises `RuntimeError` when nested
  - `parallel(fn)` decorator: the whole call is one phase
  - `task(key)` context manager (times one node's work in the current phase; no-op outside a phase)
  - `record(key, seconds)` (adds an externally measured task, e.g. from a pool worker)
  - `untimed()` context manager (observer code, removed from modelled time)
  - `algorithm_time_limit(config: dict) -> float`
  - `workers(parallelism: int) -> int`
  - `COLUMNS = ("step", "algorithm_s", "wall_s", "parallel_phases", "workers")`; `write_runtime(folder: str, rows: list[dict]) -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_step_time.py`:

```python
import pandas as pd
import pytest

from utils import step_time


class FakeClock:
  def __init__(self):
    self.t = 0.0

  def __call__(self):
    return self.t


@pytest.fixture(autouse=True)
def no_clock():
  step_time._clock.set(None)  # context variables survive between tests in one thread
  yield
  step_time._clock.set(None)


@pytest.fixture
def now(monkeypatch):
  fake = FakeClock()
  monkeypatch.setattr(step_time, "perf_counter", fake)
  return fake


def test_sequential_time_is_wall_time(now):
  clock = step_time.start_step()
  now.t = 5.0
  assert clock.modelled() == 5.0 and clock.wall() == 5.0


def test_phase_counts_its_longest_task(now):
  clock = step_time.start_step()
  now.t = 1.0                       # 1 s sequential
  with step_time.parallel_phase():
    with step_time.task(0):
      now.t = 3.0                   # node 0: 2 s
    with step_time.task(1):
      now.t = 4.0                   # node 1: 1 s
    now.t = 4.5                     # 0.5 s of phase overhead (not a task)
  now.t = 5.0                       # 0.5 s sequential
  assert clock.modelled() == pytest.approx(1.0 + 2.0 + 0.5)
  assert step_time.end_step(clock, 0) == {"algorithm_s": pytest.approx(3.5), "wall_s": 5.0,
                                          "parallel_phases": 1, "workers": 0}


def test_tasks_of_one_node_add_up_and_pool_records_match_in_process(now):
  clock = step_time.start_step()
  with step_time.parallel_phase() as phase:
    phase.add(7, 1.0)
    step_time.record(7, 2.0)        # same node twice in the phase: 3 s
    step_time.record(8, 2.5)
    now.t = 4.0                     # pool wall time, transfers included
  assert clock.modelled() == pytest.approx(3.0)


def test_uniform_phase_divides_wall_by_nodes(now):
  clock = step_time.start_step()
  with step_time.parallel_phase() as phase:
    phase.uniform(4)
    now.t = 8.0
  assert clock.modelled() == pytest.approx(2.0)


def test_untimed_is_removed_outside_phases(now):
  clock = step_time.start_step()
  now.t = 1.0
  with step_time.untimed():
    now.t = 3.0
  assert clock.modelled() == pytest.approx(1.0) and clock.wall() == 3.0


def test_nested_phase_raises():
  step_time.start_step()
  with step_time.parallel_phase():
    with pytest.raises(RuntimeError, match="nested"):
      with step_time.parallel_phase():
        pass


def test_decorator_and_no_clock_are_safe(now):
  @step_time.parallel
  def work():
    with step_time.task(0):
      now.t += 2.0
    return "done"

  assert work() == "done"           # no clock: runs, records nothing
  clock = step_time.start_step()
  work()
  assert clock.modelled() == pytest.approx(2.0)


def test_budget_workers_and_csv(tmp_path):
  assert step_time.algorithm_time_limit({}) == float("inf")
  assert step_time.algorithm_time_limit({"solver_options": {"general": {"algorithm_time_limit": 300}}}) == 300.0
  assert step_time.workers(3) == 3 and step_time.workers(-1) >= 1
  step_time.write_runtime(str(tmp_path), [dict(step=0, algorithm_s=1.0, wall_s=2.0, parallel_phases=3, workers=2)])
  frame = pd.read_csv(tmp_path / "runtime.csv")
  assert tuple(frame.columns) == step_time.COLUMNS and frame.iloc[0].algorithm_s == 1.0
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_step_time.py -v`
Expected: FAIL with `ImportError: cannot import name 'step_time'`.

- [ ] **Step 3: Implement `utils/step_time.py`**

```python
"""Modelled distributed time of one algorithm step.

Each node is modelled as its own machine: a parallel phase (per-node work done
simultaneously) counts its longest task, sequential code counts its wall time,
observer code (checks, export, logging) does not count.

  modelled = wall - sum(untimed) - sum(phase wall) + sum(longest task per phase)
"""
from __future__ import annotations

import contextvars
import functools
import os
from contextlib import contextmanager
from time import perf_counter

import pandas as pd

COLUMNS = ("step", "algorithm_s", "wall_s", "parallel_phases", "workers")
_clock = contextvars.ContextVar("step_clock", default=None)
_phase = contextvars.ContextVar("step_phase", default=None)


class Phase:
  def __init__(self) -> None:
    self.tasks: dict = {}
    self.nodes = 0

  def add(self, key, seconds: float) -> None:
    self.tasks[key] = self.tasks.get(key, 0.0) + seconds

  def uniform(self, nodes: int) -> None:
    """Vectorized phase: the same work per node, longest task = wall / nodes."""
    self.nodes = max(self.nodes, int(nodes))

  def longest(self, wall: float) -> float:
    if self.nodes:
      return wall / self.nodes
    return min(wall, max(self.tasks.values(), default=0.0))


class StepClock:
  def __init__(self) -> None:
    self.started = perf_counter()
    self.removed = 0.0
    self.phases = 0

  def wall(self) -> float:
    return perf_counter() - self.started

  def modelled(self) -> float:
    return self.wall() - self.removed


def start_step() -> StepClock:
  clock = StepClock()
  _clock.set(clock)
  return clock


def end_step(clock: StepClock, workers: int) -> dict:
  row = dict(algorithm_s=clock.modelled(), wall_s=clock.wall(), parallel_phases=clock.phases,
             workers=workers)
  _clock.set(None)
  return row


@contextmanager
def parallel_phase():
  if _phase.get() is not None:
    raise RuntimeError("nested parallel phase: each per-node computation belongs to one phase")
  phase = Phase()
  token = _phase.set(phase)
  started = perf_counter()
  try:
    yield phase
  finally:
    _phase.reset(token)
    wall = perf_counter() - started
    clock = _clock.get()
    if clock is not None:
      clock.removed += wall - phase.longest(wall)
      clock.phases += 1


def parallel(fn):
  """Decorator: one call of ``fn`` is one parallel phase."""
  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    with parallel_phase():
      return fn(*args, **kwargs)
  return wrapper


@contextmanager
def task(key):
  started = perf_counter()
  try:
    yield
  finally:
    phase = _phase.get()
    if phase is not None:
      phase.add(key, perf_counter() - started)


def record(key, seconds: float) -> None:
  phase = _phase.get()
  if phase is not None:
    phase.add(key, seconds)


@contextmanager
def untimed():
  started = perf_counter()
  try:
    yield
  finally:
    clock = _clock.get()
    if clock is not None and _phase.get() is None:  # inside a phase only tasks count anyway
      clock.removed += perf_counter() - started


def algorithm_time_limit(config: dict) -> float:
  general = config.get("solver_options", {}).get("general", {})
  return float(general.get("algorithm_time_limit", float("inf")))


def workers(parallelism: int) -> int:
  if parallelism >= 0:
    return parallelism
  try:
    return len(os.sched_getaffinity(0))
  except AttributeError:  # macOS
    return os.cpu_count() or 1


def write_runtime(folder: str, rows: list[dict]) -> None:
  pd.DataFrame(rows, columns=list(COLUMNS)).to_csv(os.path.join(folder, "runtime.csv"), index=False)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_step_time.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add utils/step_time.py tests/test_step_time.py
git commit -m "step_time: modelled distributed time of an algorithm step

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Subproblem phases in the shared solver helpers (`run_faasmacro.py`)

Every harmonized method solves agent subproblems through `solve_subproblem` or `compute_social_welfare`; this task makes both one parallel phase each, with per-agent task times measured where the agent is solved.

**Files:**
- Modify: `run_faasmacro.py` (`_solve_parallel_batch` ~l.102, `_solve_agents_parallel` ~l.108-126, `compute_social_welfare` agent block ~l.327-337, `solve_subproblem` agent block ~l.636-651)
- Test: `tests/test_step_time.py` (append)

**Interfaces:**
- Consumes: `step_time.parallel_phase`, `task`, `record` (Task 1).
- Produces: `_solve_parallel_batch(task) -> list[tuple[int, dict, float]]` (agent, result, seconds); `_solve_agents_parallel(...)` unchanged return type (dict agent → result) and records task times into the open phase.

- [ ] **Step 1: Impact analysis**

Run: `npx gitnexus impact _solve_parallel_batch; npx gitnexus impact _solve_agents_parallel; npx gitnexus impact solve_subproblem; npx gitnexus impact compute_social_welfare`
Expected: report direct callers (all runners, `experiments/madea_pg_compact/warm_pool.py`). `warm_pool.py` unpacks `_solve_parallel_batch` results: update it in Step 4 too. Stop on HIGH/CRITICAL and tell the user.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_step_time.py`:

```python
def test_solve_subproblem_is_one_phase_with_one_task_per_agent(monkeypatch):
  import run_faasmacro

  durations = {0: 0.02, 1: 0.05, 2: 0.01}

  def fake_solve(model, data, options, solver):
    import time
    agent = data[None]["whoami"][None] - 1
    time.sleep(durations[agent])
    return {"agent": agent}

  monkeypatch.setattr(run_faasmacro, "solve_agent_problem", fake_solve)
  monkeypatch.setattr(run_faasmacro, "merge_agents_solutions",
                      lambda *a, **k: (None,) * 6 + ({"tot": 0.0}, None, None))
  monkeypatch.setattr(run_faasmacro, "compute_utilization", lambda *a, **k: None)
  data = {None: {"Nn": {None: 3}, "Nf": {None: 1}}}
  clock = step_time.start_step()
  run_faasmacro.solve_subproblem(data, [0, 1, 2], None, "missing_solver", {}, 0)
  row = step_time.end_step(clock, 0)
  assert row["parallel_phases"] == 1
  assert row["algorithm_s"] == pytest.approx(0.05, abs=0.02) and row["wall_s"] >= 0.08
```

- [ ] **Step 3: Run it to verify it fails**

Run: `uv run pytest tests/test_step_time.py::test_solve_subproblem_is_one_phase_with_one_task_per_agent -v`
Expected: FAIL (`parallel_phases == 0`).

- [ ] **Step 4: Implement**

At the top of `run_faasmacro.py` add `import time` (if absent) and `from utils import step_time`.

Replace `_solve_parallel_batch`:

```python
def _solve_parallel_batch(task):
  data, options, solver, model, agents, prices = task
  # Each task carries the current snapshot, even if this worker ran another model.
  init_parallel_worker(data, options, solver, model)
  out = []
  for agent in agents:
    started = time.perf_counter()  # the node's own computation, without transfer
    agent_id, result = solve_single_agent(agent, prices)
    out.append((agent_id, result, time.perf_counter() - started))
  return out
```

In `_solve_agents_parallel` replace the two lines after `results = state['pool'].map(...)`:

```python
  by_agent = {}
  for batch in results:
    for agent, result, seconds in batch:
      by_agent[agent] = result
      step_time.record(agent, seconds)
  return {agent: by_agent[agent] for agent in agents}
```

In `solve_subproblem` wrap the agent block (from `agents_sol = {}` through the end of the sequential `for agent in agents:` loop) in one phase and time each in-process agent:

```python
  agents_sol = {}
  with step_time.parallel_phase():
    if parallelism != 0:
      agents_sol = _solve_agents_parallel(
        sp_data, agents, sp, solver_name, solver_options, parallelism, detailed_pi,
      )
    else:
      for agent in agents:
        with step_time.task(agent):
          agent_data = sp_data.copy()
          agent_data[None] = sp_data[None].copy()
          agent_data[None]["whoami"] = {None: agent + 1}
          if detailed_pi is not None:
            agent_data[None]["pi"] = {f+1: detailed_pi[agent,f] for f in range(Nf)}
          agents_sol[agent] = solve_agent_problem(
            sp, agent_data, solver_options, solver_name
          )
```

Apply the same transformation to the agent block of `compute_social_welfare` (the `if parallelism != 0: agents_sol = _solve_agents_parallel(...) else: for agent in agents: ... solve_agent_problem(...)` block around line 327): wrap it in `with step_time.parallel_phase():` and put the body of its sequential loop inside `with step_time.task(agent):`.

In `experiments/madea_pg_compact/warm_pool.py`, wherever results of `_solve_parallel_batch` are unpacked as `(agent, result)` pairs, unpack `(agent, result, _seconds)`.

- [ ] **Step 5: Run the test**

Run: `uv run pytest tests/test_step_time.py -v`
Expected: 9 passed.

- [ ] **Step 6: Commit**

```bash
npx gitnexus detect_changes 2>/dev/null || git diff --stat
git add run_faasmacro.py experiments/madea_pg_compact/warm_pool.py tests/test_step_time.py
git commit -m "subproblem solves are timed parallel phases (task time measured in the worker)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Auction family — MADEA, MADEA-one-shot, MADEA2 (+ -b, one-shot), faas-gcaa

**Files:**
- Modify: `run_faasmadea.py` (`compute_residual_capacity` l.183, `define_bids` l.242, `evaluate_bids` l.364, `start_additional_replicas` l.514, `check_stopping_criteria` call sites, `run_madea_cycle` l.586, `run` l.880-1090)
- Modify: `decentralized_auction.py` (`run`, l.76-470)
- Modify: `madea2.py` (`define_bids` l.35 and any other per-node helper it defines)
- Modify: `decentralized_gcaa.py` (`resolve_gcaa_round` l.42, `run` l.108-330)

**Interfaces:**
- Consumes: Task 1 API; Task 2 phases (subproblem and social-welfare solves are already timed).
- Produces: these runners write `runtime.csv` with `step_time.COLUMNS`; `run_madea_cycle(...)` keeps its signature; `MadeaState.total_runtime` now holds `clock.modelled()`; `auction_time_limit` callers keep working (value in modelled seconds).

- [ ] **Step 1: Impact analysis**

Run: `npx gitnexus impact compute_residual_capacity; npx gitnexus impact define_bids; npx gitnexus impact evaluate_bids; npx gitnexus impact start_additional_replicas; npx gitnexus impact run_madea_cycle`
Report callers (diffusion, powerd, dual, gcaa, hierarchical runners, PG code). Stop on HIGH/CRITICAL.

- [ ] **Step 2: Per-node helpers become phases**

In `run_faasmadea.py` add `from utils import step_time`. Then:

- `compute_residual_capacity`: decorate with `@step_time.parallel`; wrap the body of `for n in range(Nn):` (l.192) in `with step_time.task(n):`.
- `define_bids`: decorate with `@step_time.parallel`; wrap the body of `for i,f in zip(potential_buyers, functions_to_share):` (l.263) in `with step_time.task(i):` (a buyer's functions add up to the buyer's task).
- `evaluate_bids`: decorate with `@step_time.parallel`; wrap the body of `for j in unique_potential_sellers:` (l.422) in `with step_time.task(j):`. The loops at l.451 and l.497-500 that run after the per-seller loop: if they iterate per buyer (`for i, f, remaining_d, b in pending_bids`), wrap their body in `with step_time.task(("buyer", i)):`; the price update loop over functions (l.497-500) is per seller state already computed — leave it (phase overhead, not a task).
- `start_additional_replicas`: decorate with `@step_time.parallel`; wrap the body of `for j, bids_for_j in memory_bids.groupby("j"):` (l.523) in `with step_time.task(j):`.

In `madea2.py` do the same for its own `define_bids` (decorate; wrap the per-buyer loop body in `with step_time.task(i):`) and for any other function in the file that loops over nodes to build bids or evaluate them (search `for i` / `for j` loops inside functions called from the auction loop). In `decentralized_gcaa.py` decorate `resolve_gcaa_round` and wrap its per-node (per-seller) loop body in `with step_time.task(j):`.

- [ ] **Step 3: `run_madea_cycle` uses the clock**

In `run_madea_cycle`:

- Replace `time_limit = general_solver_options.get("TimeLimit", np.inf)` with `time_limit = step_time.algorithm_time_limit(config)`.
- Delete every `total_runtime += ...` line (l.646, 676, 708, 740, 780) and the `total_runtime = time.monotonic() - wall_started_at` lines (l.831, 865); before each call to `check_stopping_criteria(...)` and before `state.total_runtime = total_runtime` (l.863) insert `total_runtime = step_time.current_modelled()`.
- Wrap observer blocks in `with step_time.untimed():` — the `check_ls_pr_feasibility_from_fixed_y` call and its `raise`, every `combine_solutions` / `compute_centralized_objective` / `check_feasibility` / `decode_solutions` block that only tracks or exports the best solution, and verbose `print` blocks are already cheap (leave them).

Add to `utils/step_time.py` (and a one-line test in `tests/test_step_time.py`: `assert step_time.current_modelled() == 0.0` with no clock):

```python
def current_modelled() -> float:
  clock = _clock.get()
  return 0.0 if clock is None else clock.modelled()
```

- [ ] **Step 4: `run` (MADEA) step loop**

In `run_faasmadea.run`, inside `for t in range(min_run_time, ub, run_time_step):`:

- right after the step's loads are computed (`data = update_data(...)` / `loadt = ...`), insert `clock = step_time.start_step()`;
- remove `total_runtime` accumulations of the step (the subproblem `sp_runtime["tot"]` addition and any other `+=`);
- replace `runtime_list.append(total_runtime)` (l.1034) with `runtime_rows.append(dict(step=t, **step_time.end_step(clock, step_time.workers(parallelism))))`;
- initialize `runtime_rows = []` where `runtime_list = []` was (l.944);
- replace the `pd.DataFrame({"tot": runtime_list}).to_csv(...)` block (l.1084-1086) with `step_time.write_runtime(solution_folder, runtime_rows)`;
- wrap the per-step export block (`decode_solutions`, `save_checkpoint`, the best-solution bookkeeping after the cycle) in `with step_time.untimed():`;
- change the `-j/--parallelism` default in `parse_arguments` to `-1`.

- [ ] **Step 5: `decentralized_auction.run` and `decentralized_gcaa.run`**

Apply to each the same edits as Step 3 and Step 4: `time_limit = step_time.algorithm_time_limit(config)`; `clock = step_time.start_step()` after the step's loads; delete all `total_runtime +=` lines (auction l.167, 203, 229, 254, 275, 312, 391; gcaa l.187, 232); `total_runtime = step_time.current_modelled()` before each stopping check that receives `total_runtime=` (auction l.360, gcaa l.262); `runtime_rows` / `end_step` / `write_runtime` in place of `runtime_list` and the `tot` CSV (auction l.134, 464; gcaa l.158, 293, 322); observer blocks in `untimed()`; CLI `--parallelism` default `-1`. In `decentralized_auction.run` the refinement branch (`refine_welfare`, l.391) belongs to excluded PG hybrids: leave its logic, but do not add `refinement["seconds"]` (the clock already counts it as sequential wall time).

- [ ] **Step 6: Regression and contract check for this family**

Run:

```bash
uv run python - <<'EOF'
import json, sys
sys.path[:0] = ["experiments/harmonization", "."]
from capture_baseline import runners, welfare, DATA
import pandas as pd, tempfile
base = json.load(open("tests/data/harmonization/baseline.json"))
for name in ["MADEA", "MADEA-one-shot", "MADEA2", "MADEA2-b", "MADEA2-one-shot", "MADEA2-b-one-shot", "faas-gcaa"]:
  for inst in sorted(p for p in DATA.iterdir() if (p / "metadata.json").exists()):
    for par in (0, 2):
      got = welfare(runners()[name], inst, par)
      assert all(abs(a - b) < 1e-6 for a, b in zip(got, base[name][inst.name])), (name, inst.name, par, got)
  print("ok", name)
EOF
```

Expected: `ok` for the 7 methods (same welfare with 0 and 2 workers as the baseline).

- [ ] **Step 7: Commit**

```bash
npx gitnexus detect_changes 2>/dev/null || git diff --stat
git add run_faasmadea.py decentralized_auction.py madea2.py decentralized_gcaa.py utils/step_time.py tests/test_step_time.py
git commit -m "auction family on the step clock: per-node bid phases, modelled budget, runtime.csv

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Hierarchical auction runners

**Files:**
- Modify: `hierarchical_auction/runner.py` (`run` l.113-415), `hierarchical_auction/madea_runner.py` (`run` l.99-450), `hierarchical_auction/madea_cycles_runner.py` (`_run` l.64-335)

**Interfaces:**
- Consumes: Task 1, 2, 3 (`define_bids`, `evaluate_bids`, `compute_residual_capacity`, `start_additional_replicas`, `run_madea_cycle` are already phases / on the clock).

- [ ] **Step 1: Impact analysis**

Run: `npx gitnexus impact run --file hierarchical_auction/runner.py; npx gitnexus impact run --file hierarchical_auction/madea_runner.py; npx gitnexus impact _run --file hierarchical_auction/madea_cycles_runner.py`
Report callers (registry wrappers, PG hybrids). Stop on HIGH/CRITICAL.

- [ ] **Step 2: Engine coordination phases**

The hierarchy engine (`HierarchicalAuctionEngine`, `IterativeHierarchicalAuctionEngine`) aggregates and coordinates per level. Read `hierarchical_auction/engine*.py` / `iterative_engine.py`: per-node or per-cluster loops that each cluster computes for itself (cluster bids, cluster clearing) are parallel phases — decorate the function with `@step_time.parallel` and wrap the per-cluster loop body in `with step_time.task(cluster_id):`. A step that a single coordinator performs over all clusters stays sequential. List each function you instrument and each you leave sequential in the commit message.

- [ ] **Step 3: Runner loops**

In each of the three runners:

- `time_limit = step_time.algorithm_time_limit(config)` in place of `general_solver_options.get("TimeLimit", ...)`; the `pg_reserve` arithmetic (refinement hybrids) keeps working on that value;
- `clock = step_time.start_step()` after the step's loads;
- delete the `total_runtime +=` lines (runner l.204, 263, 296; madea_runner l.152-328; cycles l.135-277) and the `time.monotonic() - started` assignments (runner l.309, 358; cycles l.170, 273); set `total_runtime = step_time.current_modelled()` (cycles: `state.total_runtime = step_time.current_modelled()`) right before each stopping check and before passing a remaining-time to `refine_solution`;
- the `refinement["seconds"]` addition (cycles l.277) is removed (the clock counts it);
- `runtime_rows` / `end_step` / `write_runtime` instead of `runtime_list`/`runtimes` and the `tot` CSV (runner l.179, 370, 410; madea_runner l.~146, 443; cycles l.~128, 329);
- observer blocks (`combine_solutions` + `compute_centralized_objective` used only for bookkeeping/export, `decode_solutions`, `check_ls_pr_feasibility_from_fixed_y`) in `step_time.untimed()`;
- CLI `--parallelism` default `-1` (already −1 in runner signatures: keep).

- [ ] **Step 4: Regression**

Run the Task 3 Step 6 script with the list `["hierarchical-madea", "hierarchical-madea-cycles", "hierarchical-madea-legacy", "hierarchical-one-shot", "hierarchical"]`.
Expected: `ok` for the 5 methods.

- [ ] **Step 5: Commit**

```bash
npx gitnexus detect_changes 2>/dev/null || git diff --stat
git add hierarchical_auction
git commit -m "hierarchical auction runners on the step clock (cluster phases, modelled budget)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Frank-Wolfe family — FaaS-MAFW, FaaS-MAFW-b, hierarchical-MAFW

**Files:**
- Modify: `decentralized_frankwolfe.py` (`frankwolfe_assignments` ~l.170-230, `round_flows` ~l.233, repair helper with the per-node loop at ~l.350, `run` l.524-760)
- Modify: `hierarchical_frankwolfe.py` (`run` l.89-275)

**Interfaces:**
- Consumes: Task 1, 2.

- [ ] **Step 1: Impact analysis**

Run: `npx gitnexus impact frankwolfe_assignments; npx gitnexus impact round_flows; npx gitnexus impact run --file decentralized_frankwolfe.py; npx gitnexus impact run --file hierarchical_frankwolfe.py`
Stop on HIGH/CRITICAL.

- [ ] **Step 2: Phases**

- `frankwolfe_assignments`: each Frank-Wolfe iteration is one vectorized parallel phase over the nodes (every node computes its row of prices and best responses; iterations are sequential). Inside `for t in range(int(options["iterations"])):` wrap the whole iteration body in

```python
    with step_time.parallel_phase() as phase:
      phase.uniform(Nn)
      ...  # the existing iteration body, unchanged
```

  and move the `if gap <= tol * scale and t > 0: break` test after the `with` block (a `break` inside the `with` is fine in Python; keep it where it is if it reads more simply).
- `round_flows`: decorate with `@step_time.parallel`; wrap the body of `for i in range(Nn):` in `with step_time.task(i):`; the `limit` capping loop afterwards iterates per sender `(i, f)`: wrap its body in `with step_time.task(i):` too (same phase, tasks add up per node).
- The repair helper whose loop is `for n in range(Nn):` (~l.350) and the receiver-side loop `for n in range(Nn):` at ~l.104: decorate the containing function with `@step_time.parallel` and wrap each loop body in `with step_time.task(n):`.
- In `run`, delete `total_runtime += (datetime.now() - s).total_seconds() / Nn` (l.646) and its comment block (the per-node average is replaced by the phases above).

- [ ] **Step 3: Runner loops**

In `decentralized_frankwolfe.run` and `hierarchical_frankwolfe.run`: `time_limit = step_time.algorithm_time_limit(config)`; `clock = step_time.start_step()` after the step's loads; delete `total_runtime +=` lines (FW l.602, 646, 663; HFW the `elapsed`/`total` arithmetic at l.153, 227, 254) and use `step_time.current_modelled()` where a stop rule or a remaining time is computed (HFW l.227, 247); `runtime_rows` / `end_step` / `write_runtime` (FW l.578, 720, 755; HFW l.150, 259, 272); observer blocks in `untimed()`; CLI `--parallelism` default `-1`.

- [ ] **Step 4: Regression**

Run the Task 3 Step 6 script with `["FaaS-MAFW", "FaaS-MAFW-b", "hierarchical-MAFW"]`.
Expected: `ok` for the 3 methods.

- [ ] **Step 5: Commit**

```bash
npx gitnexus detect_changes 2>/dev/null || git diff --stat
git add decentralized_frankwolfe.py hierarchical_frankwolfe.py
git commit -m "Frank-Wolfe family on the step clock (vectorized iteration phases, per-node rounding and repair)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Diffusion, power-of-d, dual, MACrO

**Files:**
- Modify: `decentralized_diffusion.py` (`define_assignments` l.49, `evaluate_assignments` l.143, `run` l.277-520)
- Modify: `decentralized_powerd.py` (`sample_assignments` l.50, `run` l.189-440)
- Modify: `decentralized_dual.py` (`dual_coordination_round` l.131, `run` l.292-560)
- Modify: `run_faasmacro.py` (`run` l.724-1240)

**Interfaces:**
- Consumes: Task 1, 2, 3 (`compute_residual_capacity`, `start_additional_replicas` already phases).

- [ ] **Step 1: Impact analysis**

Run: `npx gitnexus impact define_assignments; npx gitnexus impact evaluate_assignments; npx gitnexus impact sample_assignments; npx gitnexus impact dual_coordination_round; npx gitnexus impact run --file run_faasmacro.py`
Stop on HIGH/CRITICAL.

- [ ] **Step 2: Phases**

- `define_assignments` (buyers propose) and `sample_assignments` (buyers sample d sellers): decorate with `@step_time.parallel`; wrap the per-buyer loop body in `with step_time.task(i):`.
- `evaluate_assignments` (sellers accept): decorate with `@step_time.parallel`; wrap the per-seller loop body in `with step_time.task(j):`. It is called both from runners and from inside `dual_coordination_round` (l.204): to avoid a nested phase, split `dual_coordination_round` into its phases explicitly instead of decorating it — open `with step_time.parallel_phase():` around the buyers' `buyer_price_response` loop (task key = buyer) and let the `evaluate_assignments` call that follows be its own (decorated) phase; the price update per seller after it, if it is a loop over sellers, is a third phase with `task(j)`.
- In each runner delete the averaged accumulations (`rt / n_auctions`, `rt / n_sellers`, `rt / n_active`) — the phases replace them.
- `run_faasmacro.run`: the master problem (`solve_master_problem`, l.~925) is solved by one coordinator: sequential, counts its wall time (delete `total_runtime += runtime`, l.941); the subproblems are timed by Task 2.

- [ ] **Step 3: Runner loops**

For `decentralized_diffusion.run`, `decentralized_powerd.run`, `decentralized_dual.run`, `run_faasmacro.run`: `time_limit = step_time.algorithm_time_limit(config)` (macro: wherever `TimeLimit` feeds `check_stopping_criteria`'s `time_limit`); `clock = step_time.start_step()` after the step's loads; delete all `total_runtime +=` lines (diffusion l.357-430; powerd l.274-348; dual l.376-437; macro l.907, 941, 987); `total_runtime = step_time.current_modelled()` before each `check_stopping_criteria(...)`; `runtime_rows` / `end_step` / `write_runtime` (diffusion l.333, 479, 514; powerd l.250, 397, 432; dual l.352, 506, 556; macro l.798, 1145, 1232); observer blocks (`check_ls_pr_feasibility_from_fixed_y`, `combine_solutions`/`compute_centralized_objective`/`check_feasibility` bookkeeping, `decode_solutions`) in `untimed()`; CLI `--parallelism` default `-1`.

- [ ] **Step 4: Regression**

Run the Task 3 Step 6 script with `["faas-diffuse", "faas-powd", "faas-dual", "faas-macro", "faas-macro-v0"]`.
Expected: `ok` for the 5 methods.

- [ ] **Step 5: Commit**

```bash
npx gitnexus detect_changes 2>/dev/null || git diff --stat
git add decentralized_diffusion.py decentralized_powerd.py decentralized_dual.py run_faasmacro.py
git commit -m "diffusion, power-of-d, dual and MACrO on the step clock

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: PLASMA and PLASMA-Welfare

**Files:**
- Modify: `plasma/welfare.py` (`WelfareEngine._parallel_round` l.~306, `run_rounds` l.~327-372)
- Modify: the base engine in `plasma/core/` (class `PlasmaEngine`: its per-round negotiation loop and heartbeat sending)
- Modify: `plasma/runner.py` (`run` l.55-170)

**Interfaces:**
- Consumes: Task 1.

- [ ] **Step 1: Impact analysis**

Run: `npx gitnexus impact run_rounds; npx gitnexus impact _parallel_round; npx gitnexus impact PlasmaEngine; npx gitnexus impact run --file plasma/runner.py`
Stop on HIGH/CRITICAL.

- [ ] **Step 2: One turn order for both execution modes**

Today PLASMA-Welfare negotiates in conflict-free batches on the pool (`_negotiation_batches`, which keeps the original order for every pair sharing a participant) but in strict sequential turns in-process, and computes initial plans in `start_window(initialize=True)` in-process but on the pool otherwise. Make both modes use the same structure, so decisions and modelled time do not depend on `parallelism`:

- initial plans: always computed per alive node as one parallel phase — on the pool (`pool.map(_solve_plan, ...)`) when a pool exists, otherwise by calling `_solve_plan(_plan_job(node, ()))` per node inside `with step_time.task(i):`; then `start_window(load, round, initialize=False)` in both modes;
- negotiation: `on_round` always iterates `self._negotiation_batches(round_)`; each batch is one parallel phase; for each node of the batch the offers + plan + accept work is the node's task (`with step_time.task(i):`; pool plans are recorded with `step_time.record(i, seconds)` measured inside a wrapper of `_solve_plan` that returns `(plan, seconds)`);
- heartbeats (`_send_heartbeats`): one parallel phase, task per sending node.

Verify on the regression instances that PLASMA-Welfare's welfare with `parallelism=0` equals the baseline captured in Task 0 (which used `parallelism=0`, i.e. the old sequential turns). If it differs, the batch order is not equivalent to the sequential order on these instances: stop and report to the user with both values before going on (deciding which order is the algorithm is the user's call).

- [ ] **Step 3: Plain PLASMA**

In the base `PlasmaEngine` round loop (sequential turns, single process by design: "The original PLASMA stays single-process"): per round, the nodes' local work that the protocol performs simultaneously (each node's turn computations that do not depend on another node's turn in the same round, and heartbeats) becomes a phase with one task per node, using the same `_negotiation_batches` rule (pairs sharing a participant keep their order: separate batches). No pool is added to plain PLASMA (its local work is tiny); only its time is modelled.

- [ ] **Step 4: Runner loop**

In `plasma/runner.run`: `clock = step_time.start_step()` right before `engine.run_rounds(...)`; `runtime_rows.append(dict(step=t, **step_time.end_step(clock, step_time.workers(parallelism))))` right after it (instead of `elapsed`/`runtime_list`, l.120-140); the feasibility check, `decode_solution`, objective and message rows after it are observer code (they run after `end_step`, so outside the clock); `step_time.write_runtime(solution_folder, runtime_rows)` instead of the `tot` CSV (l.163); CLI `--parallelism` default `-1` in `plasma/cli.py`.

- [ ] **Step 5: Regression**

Run the Task 3 Step 6 script with `["PLASMA", "PLASMA-Welfare"]`.
Expected: `ok` for both (equal welfare with 0 and 2 workers).

- [ ] **Step 6: Commit**

```bash
npx gitnexus detect_changes 2>/dev/null || git diff --stat
git add plasma
git commit -m "PLASMA on the step clock; one negotiation order for pool and in-process runs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Registry and `runtime.csv` readers

**Files:**
- Create: `methods.py`
- Modify: `experiments/madea_pg_compact/catalog_campaign.py`, `experiments/madea_pg_compact/families_extended.py` (l.148 and runtime use), `experiments/madea_pg_compact/extend_frankwolfe.py` (l.16-30), `benchmark_summary.py` (l.31, 76), `compare_results.py` (runtime loading)
- Test: `tests/test_harmonized_methods.py` (registry part)

**Interfaces:**
- Produces: `methods.HARMONIZED: dict[str, Callable]` (22 names of Task 0), `methods.EXCLUDED: dict[str, str]`; `catalog_campaign.METHODS = methods.HARMONIZED`; readers use column `algorithm_s`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_harmonized_methods.py`:

```python
"""Contract of the harmonized methods (registry, runtime.csv, budget, regression)."""
import inspect
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments" / "harmonization"), str(ROOT)]

import methods  # noqa: E402
from utils import step_time  # noqa: E402

BASELINE = json.loads((ROOT / "tests/data/harmonization/baseline.json").read_text())


def test_registry_covers_the_baseline_and_excludes_sequential_methods():
  assert set(methods.HARMONIZED) == set(BASELINE)
  assert {"FaaS-MAPG-S", "FaaS-MAPG-R", "faas-br-s", "MADEA-PG", "MAFW-PG"} <= set(methods.EXCLUDED)
  assert not set(methods.HARMONIZED) & set(methods.EXCLUDED)
  for name, runner in methods.HARMONIZED.items():
    params = list(inspect.signature(runner).parameters)
    assert params[:2] == ["config", "parallelism"], name
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_harmonized_methods.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'methods'`.

- [ ] **Step 3: Implement `methods.py`**

Move the dictionary of `runners()` from `experiments/harmonization/capture_baseline.py` (Task 0) into `methods.py` as `HARMONIZED` (same names and runners, same imports at module level), and add:

```python
SEQUENTIAL = "nodes act one at a time in a network-wide order (Gauss-Seidel better-response sweeps)"
PG_REFINED = "parallel phase followed by a potential-game refinement that passes a turn through all nodes"
EXCLUDED = {
  "FaaS-MAPG-S": SEQUENTIAL, "FaaS-MAPG-R": SEQUENTIAL,
  "faas-br-s": SEQUENTIAL, "faas-br-r": SEQUENTIAL, "faas-br-o": SEQUENTIAL,
  "MADEA-PG": PG_REFINED, "one-shot-pg": PG_REFINED, "one-shot2-pg": PG_REFINED,
  "one-shot2-b-pg": PG_REFINED, "MAFW-PG": PG_REFINED, "MAFW-b-PG": PG_REFINED,
  "hierarchical-MAFW-PG": PG_REFINED, "hierarchical-madea-pg": PG_REFINED,
  "hierarchical-one-shot-pg": PG_REFINED, "hierarchical-madea-level-cycles-pg": PG_REFINED,
}
```

`madea2.run_one_shot` has a keyword-only `refine_welfare`: register it as is (the first two parameters are `config, parallelism`). In `capture_baseline.runners()` keep only `from methods import HARMONIZED; return HARMONIZED`.

- [ ] **Step 4: Point `catalog_campaign` at the registry**

In `experiments/madea_pg_compact/catalog_campaign.py` replace the `METHODS = {...}` definition and `_extra_methods()` with:

```python
import methods as _methods
METHODS = dict(_methods.HARMONIZED)
```

(`sys.path` already contains the repository root through `families_extended`.) Keep `ALL_METHODS` imports only if still used elsewhere in the file; otherwise remove them.

- [ ] **Step 5: Readers use `algorithm_s`**

- `families_extended.py` l.148-~170: `runtime = pd.read_csv(folder / 'runtime.csv')` for every method (no `MADEA-one-shot` special case: every harmonized method writes it now); `native_seconds=float(runtime['algorithm_s'].sum())`; add `wall_seconds_native=float(runtime['wall_s'].sum())` to the row.
- `extend_frankwolfe.py` l.16-30: remove the code that synthesizes a missing `runtime.csv` for one-shot (now always written).
- `benchmark_summary.py`: `runtime_s` = sum of column `algorithm_s` (fallback to the single column for folders written before this change: `frame["algorithm_s"] if "algorithm_s" in frame else frame.iloc[:, 0]`).
- `compare_results.py`: where per-model `runtime.csv` is read, use `algorithm_s` with the same fallback.

Add to `tests/test_harmonized_methods.py`:

```python
def test_benchmark_summary_reads_the_new_format(tmp_path):
  import benchmark_summary
  step_time.write_runtime(str(tmp_path), [dict(step=0, algorithm_s=1.5, wall_s=9.0, parallel_phases=2, workers=4),
                                          dict(step=1, algorithm_s=2.5, wall_s=9.0, parallel_phases=2, workers=4)])
  assert benchmark_summary._single_column_sum(str(tmp_path / "runtime.csv")) == pytest.approx(4.0)
```

(`_single_column_sum` is the helper at `benchmark_summary.py:31`; make it pick `algorithm_s` when present.)

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_harmonized_methods.py tests/test_step_time.py -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
npx gitnexus detect_changes 2>/dev/null || git diff --stat
git add methods.py experiments/harmonization/capture_baseline.py experiments/madea_pg_compact benchmark_summary.py compare_results.py tests/test_harmonized_methods.py
git commit -m "method registry (harmonized/excluded) and runtime.csv readers on algorithm_s

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Contract, budget and regression tests over the whole registry

**Files:**
- Modify: `tests/test_harmonized_methods.py`

- [ ] **Step 1: Write the tests**

Append:

```python
from capture_baseline import DATA, config_for, welfare  # noqa: E402

INSTANCES = sorted(p for p in DATA.iterdir() if (p / "metadata.json").exists())
NAMES = sorted(methods.HARMONIZED)


def run_folder(name, instance, parallelism, tmp, **general):
  config = config_for(instance, Path(tmp))
  config["solver_options"]["general"].update(general)
  return Path(methods.HARMONIZED[name](config, parallelism, log_on_file=True, disable_plotting=True))


@pytest.mark.parametrize("name", NAMES)
def test_runtime_contract_and_worker_independence(name, tmp_path):
  instance = INSTANCES[0]
  for parallelism in (0, 2):
    folder = run_folder(name, instance, parallelism, tmp_path / str(parallelism))
    frame = pd.read_csv(folder / "runtime.csv")
    assert tuple(frame.columns) == step_time.COLUMNS and len(frame) == 2
    assert (frame.algorithm_s > 0).all() and (frame.algorithm_s <= frame.wall_s + 1e-3).all()
    assert (frame.parallel_phases >= 1).all() and (frame.workers == parallelism).all()
    got = [float(v) for v in pd.read_csv(folder / "obj.csv").iloc[:, -1]]
    assert got == pytest.approx(BASELINE[name][instance.name], abs=1e-6), (name, parallelism)


@pytest.mark.parametrize("name", NAMES)
def test_regression_on_every_instance(name):
  for instance in INSTANCES[1:]:
    assert welfare(methods.HARMONIZED[name], instance, 2) == pytest.approx(
      BASELINE[name][instance.name], abs=1e-6)


ITERATIVE = [n for n in NAMES if not n.startswith("PLASMA") and "one-shot" not in n]


@pytest.mark.parametrize("name", ITERATIVE)
def test_algorithm_time_limit_stops_every_iterative_method(name, tmp_path):
  folder = run_folder(name, INSTANCES[0], 0, tmp_path, algorithm_time_limit=1e-9, TimeLimit=1e9)
  log = (folder / "out.log").read_text()
  assert "time limit" in log.lower(), name
  # a tiny budget allows at most the first round: welfare is defined and not above the unbounded run
  got = [float(v) for v in pd.read_csv(folder / "obj.csv").iloc[:, -1]]
  assert all(g <= b + 1e-6 for g, b in zip(got, BASELINE[name][INSTANCES[0].name]))


def test_observer_time_is_excluded(tmp_path, monkeypatch):
  import time
  import run_faasmadea
  original = run_faasmadea.check_feasibility

  def slow(*args, **kwargs):
    time.sleep(0.2)
    return original(*args, **kwargs)

  monkeypatch.setattr(run_faasmadea, "check_feasibility", slow)
  folder = run_folder("MADEA", INSTANCES[0], 0, tmp_path)
  frame = pd.read_csv(folder / "runtime.csv")
  assert (frame.wall_s - frame.algorithm_s >= 0.2).all()
```

If a method does not log the reason of stopping in `out.log`, make its stopping message (the `why_stopping` string, e.g. `reached time limit: ...`) be written to the log stream it already uses for `verbose` output, unconditionally on a time-limit stop.

- [ ] **Step 2: Run them**

Run: `uv run pytest tests/test_harmonized_methods.py -v -x`
Expected: all pass. Typical failures and their fixes: a nested phase `RuntimeError` (a decorated helper called inside another phase: remove the inner decorator and keep its loop tasks); `parallel_phases == 0` (a runner's subproblem path bypasses `solve_subproblem`: route it there); welfare differs between 0 and 2 workers (non-deterministic merge order: sort by agent before merging).

- [ ] **Step 3: DP-only check on a larger instance**

Run:

```bash
uv run python - <<'EOF'
import sys, tempfile
from pathlib import Path
sys.path[:0] = ["experiments/harmonization", "."]
import methods
from capture_baseline import config_for
inst = Path(sorted(Path.home().glob("Downloads/dfaas-edge-experiments/catalog/zhang-national/instances/*n160*huawei-d1-f5*-s1-s1"))[0])
for name, run in sorted(methods.HARMONIZED.items()):
  with tempfile.TemporaryDirectory() as tmp:
    try:
      run(config_for(inst, Path(tmp)), -1, log_on_file=True, disable_plotting=True)
      print("ok ", name)
    except Exception as e:
      print("ERR", name, type(e).__name__, str(e)[:120])
EOF
```

Expected: `ok` for all 22 (no local problem falls back to a solver at 160 nodes). An `ERR` naming a solver is a DP coverage defect: report it to the user with the model name before fixing it.

- [ ] **Step 4: Full test suite**

Run: `uv run pytest -q`
Expected: no failure that is not already listed in `tests/data/harmonization/pytest-before.txt` (Task 0).

- [ ] **Step 5: Commit**

```bash
npx gitnexus detect_changes 2>/dev/null || git diff --stat
git add tests/test_harmonized_methods.py
git commit -m "contract, budget, observer and regression tests over every harmonized method

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Ask the user to push** (or for permission): the campaign repository's submodule needs this branch on GitHub.
