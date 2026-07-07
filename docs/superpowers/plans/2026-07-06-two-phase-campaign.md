# Two-Phase Auto-Chained Campaign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 12,630-run `paper.py` campaign with a screen-then-confirm design (~1,900 runs) launched by one `campaign` command: a cheap screening suite ranks candidate algorithms by relative-to-best objective, the top 4 are auto-selected as survivors, then the confirmatory suites run only on survivors + the two anchors.

**Architecture:** Three resumable stages behind `remote_experiments campaign run`: (1) run `paper-a-screening`; (2) parse its results and write `batches/survivors.json`; (3) `define`+`materialize`+`run` each confirmatory suite non-interactively, with the suite builders reading survivors from that file. New modules `results_reader.py`, `survivors.py`, `campaign.py`; edits to `definitions/paper.py` and `cli.py`.

**Tech Stack:** Python 3.10, `uv`, `pytest`, `pandas` (already used by `compare_results.py`), `ray_dispatcher`.

## Global Constraints

- Two-space indentation, `from __future__ import annotations`, module docstring first line — match every file in `remote_experiments/`.
- Optimization is **maximize** (`models/rmp.py:119`): higher objective is better; screening ranks by `reldef = (obj_best − obj)/obj_best·100`, smaller is better.
- Anchors always in the confirmatory set and never promotable: `("centralized", "hierarchical-madea")`.
- Confirmatory seed count: **5**. Screening seed count: **5**.
- Weight-tunable algorithms (eligible for e7/e8): `{"hierarchical-madea": "auction", "faas-madea": "auction", "faas-diffuse": "diffusion", "faas-powd": "powerd"}` — any other survivor is excluded from e7/e8 only.
- Tests live in `tests/test_*.py`, flat, run with `uv run pytest`.
- Do **not** modify `build_e0` / `paper-e0-pilot`; it is superseded by screening but left intact.

---

### Task 1: `--yes` flag makes `run` non-interactive

**Files:**
- Modify: `remote_experiments/cli.py:38-53` (`cmd_run`), `remote_experiments/cli.py:110-119` (`run` parser)
- Test: `tests/test_remote_experiments_cli.py`

**Interfaces:**
- Produces: `run` subcommand accepts `--yes`; when set, `cmd_run` selects all pending (no `input()`).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_remote_experiments_cli.py` (reuse the `_FakeDispatcher` pattern already in that file; do not add an `input` monkeypatch — its absence is the point):

```python
def test_cmd_run_yes_skips_prompt(tmp_path, monkeypatch):
  batch = _write_smoke_batch(tmp_path)  # same helper the other run tests use
  monkeypatch.setattr("remote_experiments.cli.Dispatcher", _FakeDispatcher)
  monkeypatch.setattr(
    "builtins.input",
    lambda prompt: (_ for _ in ()).throw(AssertionError("prompt must not be called")),
  )
  args = build_parser().parse_args([
    "run", str(batch), "--inventory", str(_write_inventory(tmp_path)), "--yes",
  ])
  cmd_run(args)  # must not raise
```

If `_write_smoke_batch` / `_write_inventory` helpers do not already exist in the test file, lift the inline setup from `test_cmd_run_wires_dispatcher_into_manifest` into small local helpers first.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_remote_experiments_cli.py::test_cmd_run_yes_skips_prompt -v`
Expected: FAIL — `--yes` is an unrecognized argument (SystemExit from argparse).

- [ ] **Step 3: Implement**

In `build_parser`, under the `run_p` block:

```python
  run_p.add_argument("--yes", action="store_true", help="Run all pending without prompting")
```

In `cmd_run`, replace the prompt block:

```python
  default_idx = default_selection(experiment_ids, manifest)
  print(f"{len(batch.experiments)} experiments in batch, {len(default_idx)} pending")
  for i, e in enumerate(batch.experiments):
    print(f"  [{i}] {e.id} ({manifest.status(e.id)})")
  if args.yes:
    selected_idx = default_idx
  else:
    raw = input(f"Select to run [default: {len(default_idx)} pending] (indices/ranges/'all'): ")
    selected_idx = parse_selection(raw, len(batch.experiments)) if raw.strip() else default_idx
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_remote_experiments_cli.py -v`
Expected: PASS (all cli tests).

- [ ] **Step 5: Commit**

```bash
git add remote_experiments/cli.py tests/test_remote_experiments_cli.py
git commit -m "feat(remote): --yes runs all pending non-interactively"
```

---

### Task 2: Survivors plumbing and shared constants in `paper.py`

**Files:**
- Modify: `remote_experiments/definitions/paper.py` (top-of-module constants + helpers)
- Test: `tests/test_paper_experiment_suites.py`

**Interfaces:**
- Produces: `ANCHORS: tuple[str, ...]`, `DEFAULT_SURVIVORS: tuple[str, ...]`, `WEIGHT_TUNABLE: dict[str, str]`, `SURVIVORS_PATH: Path`, `_survivors() -> tuple[str, ...]`, `_final_algorithms() -> tuple[str, ...]`, `_tunable(algos) -> tuple[str, ...]`. `CONFIRMATORY_SEEDS = tuple(range(1001, 1006))`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_paper_experiment_suites.py`:

```python
import json
from remote_experiments.definitions import paper as paper_mod


def test_survivors_fallback_when_file_absent(tmp_path, monkeypatch):
  monkeypatch.setattr(paper_mod, "SURVIVORS_PATH", tmp_path / "missing.json")
  assert paper_mod._survivors() == paper_mod.DEFAULT_SURVIVORS
  assert paper_mod._final_algorithms() == paper_mod.ANCHORS + paper_mod.DEFAULT_SURVIVORS


def test_survivors_read_from_file(tmp_path, monkeypatch):
  path = tmp_path / "survivors.json"
  path.write_text(json.dumps({"survivors": ["faas-gcaa", "faas-pg-s", "faas-powd", "faas-diffuse"]}))
  monkeypatch.setattr(paper_mod, "SURVIVORS_PATH", path)
  assert paper_mod._survivors() == ("faas-gcaa", "faas-pg-s", "faas-powd", "faas-diffuse")


def test_confirmatory_seeds_are_five():
  assert len(paper_mod.CONFIRMATORY_SEEDS) == 5


def test_tunable_filters_non_weight_algorithms():
  assert paper_mod._tunable(("hierarchical-madea", "faas-powd", "faas-br-o")) == (
    "hierarchical-madea", "faas-powd",
  )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_paper_experiment_suites.py::test_survivors_fallback_when_file_absent -v`
Expected: FAIL — `paper` has no attribute `SURVIVORS_PATH`.

- [ ] **Step 3: Implement**

In `remote_experiments/definitions/paper.py`, replace the `CONFIRMATORY_SEEDS`/`ALL_ALGORITHMS`/`REPRESENTATIVE_ALGORITHMS`/`TRADEOFF_ALGORITHMS` block near the top with:

```python
PILOT_SEEDS = tuple(range(1, 11))
CONFIRMATORY_SEEDS = tuple(range(1001, 1006))

ALL_ALGORITHMS = (
  "centralized", "faas-macro", "faas-macro-v0", "faas-madea", "hierarchical-madea",
  "faas-diffuse", "faas-powd", "faas-br-s", "faas-br-r", "faas-br-o",
)
ANCHORS = ("centralized", "hierarchical-madea")
DEFAULT_SURVIVORS = ("faas-madea", "faas-diffuse", "faas-powd", "faas-br-o")
WEIGHT_TUNABLE = {
  "hierarchical-madea": "auction", "faas-madea": "auction",
  "faas-diffuse": "diffusion", "faas-powd": "powerd",
}
SURVIVORS_PATH = Path(__file__).resolve().parents[2] / "batches" / "survivors.json"

SCREENING_CANDIDATES = (
  "faas-macro", "faas-macro-v0", "faas-madea", "faas-diffuse", "faas-powd",
  "faas-br-s", "faas-br-r", "faas-br-o", "faas-pg-s", "faas-pg-r", "faas-gcaa", "plasma",
)


def _survivors() -> tuple[str, ...]:
  if SURVIVORS_PATH.exists():
    return tuple(json.loads(SURVIVORS_PATH.read_text())["survivors"])
  return DEFAULT_SURVIVORS


def _final_algorithms() -> tuple[str, ...]:
  return ANCHORS + _survivors()


def _tunable(algorithms: tuple[str, ...]) -> tuple[str, ...]:
  return tuple(a for a in algorithms if a in WEIGHT_TUNABLE)
```

Keep the existing `NON_CENTRALIZED_ALGORITHMS = tuple(a for a in ALL_ALGORITHMS if a != "centralized")` line. `json` and `Path` are already imported at the top of the file.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_paper_experiment_suites.py -k "survivors or seeds or tunable" -v`
Expected: PASS (4 new tests). Existing count tests still fail — fixed in Task 4.

- [ ] **Step 5: Commit**

```bash
git add remote_experiments/definitions/paper.py tests/test_paper_experiment_suites.py
git commit -m "feat(paper): survivors plumbing, anchors, tunable filter, 5 seeds"
```

---

### Task 3: `paper-a-screening` suite

**Files:**
- Modify: `remote_experiments/definitions/paper.py` (new `build_screening`)
- Test: `tests/test_paper_experiment_suites.py`

**Interfaces:**
- Consumes: `SCREENING_CANDIDATES`, `_new_config`, `_euclidean_planar`, `_experiment` (all in `paper.py`).
- Produces: `build_screening(seeds=..., ) -> list[Experiment]` registered as `paper-a-screening`.

- [ ] **Step 1: Write the failing test**

```python
from remote_experiments.definitions.paper import build_screening

def test_screening_count_algorithms_and_no_centralized():
  experiments = build_screening()
  assert len(experiments) == 260                      # 13 algos x {50,100}x{2,4} x 5 seeds
  algos = {e.algorithm for e in experiments}
  assert "centralized" not in algos
  assert "hierarchical-madea" in algos                # reference for obj_best
  assert algos == set(paper_mod.SCREENING_CANDIDATES) | {"hierarchical-madea"}
  assert {e.config["limits"]["Nn"]["min"] for e in experiments} == {50, 100}
  assert len({e.id for e in experiments}) == 260
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_paper_experiment_suites.py::test_screening_count_algorithms_and_no_centralized -v`
Expected: FAIL — cannot import `build_screening`.

- [ ] **Step 3: Implement**

Add to `paper.py` (after `build_e0`):

```python
@register_suite("paper-a-screening")
def build_screening(
    seeds: tuple[int, ...] = CONFIRMATORY_SEEDS,
    algorithms: tuple[str, ...] = SCREENING_CANDIDATES + ("hierarchical-madea",),
  ) -> list[Experiment]:
  suite = "paper-a-screening"
  return [
    _experiment(
      suite, f"n{nodes}-f{functions}-planar3", algorithm, seed,
      _new_config(nodes, functions, _euclidean_planar()),
    )
    for nodes in (50, 100)
    for functions in (2, 4)
    for algorithm in algorithms
    for seed in seeds
  ]
```

Add `"paper-a-screening"` to the expected set in `test_paper_suites_are_registered`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_paper_experiment_suites.py -k "screening or registered" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add remote_experiments/definitions/paper.py tests/test_paper_experiment_suites.py
git commit -m "feat(paper): paper-a-screening suite (260 runs, relative-to-best)"
```

---

### Task 4: Rewire confirmatory suites e1–e8 to survivors

**Files:**
- Modify: `remote_experiments/definitions/paper.py` (`build_e1`..`build_e8`)
- Test: `tests/test_paper_experiment_suites.py`

**Interfaces:**
- Consumes: `_final_algorithms`, `_survivors`, `_tunable`, `ANCHORS`, `CONFIRMATORY_SEEDS`.

- [ ] **Step 1: Update the count tests to the new expected values**

Replace the existing count assertions in `tests/test_paper_experiment_suites.py`:

```python
def test_e1_default_count_and_unique_ids():
  experiments = build_e1()
  assert len(experiments) == 180                    # 3 nodes x 2 funcs x 6 algos x 5 seeds
  assert len({e.id for e in experiments}) == 180
  assert set(e.algorithm for e in experiments) == set(paper_mod._final_algorithms())

def test_e2_count_and_centralized_size_limit():
  experiments = build_e2()
  assert len(experiments) == 480
  centralized_sizes = {
    e.config["limits"]["Nn"]["min"]
    for e in experiments if e.algorithm == "centralized"
  }
  assert centralized_sizes == {10, 20}
  assert {e.config["limits"]["Nn"]["min"] for e in experiments} == {10, 20, 50, 100, 200, 500}

def test_e3_count_and_topology_coverage():
  assert len(build_e3()) == 180

def test_e4_count_and_conditions():
  experiments = build_e4()
  assert len(experiments) == 7 * 6 * 5

def test_e5_count_trace_coverage_and_steps():
  experiments = build_e5()
  assert len(experiments) == 3 * 6 * 5
  assert {e.config["max_steps"] for e in experiments} == {100}

def test_e6_default_count_and_hierarchical_only():
  experiments = build_e6()
  assert len(experiments) == 100                    # 10 variants x 1 node x 2 topo x 5
  assert {e.algorithm for e in experiments} == {"hierarchical-madea"}
  assert {e.config["limits"]["Nn"]["min"] for e in experiments} == {50}

def test_e7_default_count_and_weight_pairs():
  experiments = build_e7()
  assert len(experiments) == 7 * 2 * 4 * 5          # 4 = tunable subset of FINAL
  assert {e.algorithm for e in experiments} == {
    "hierarchical-madea", "faas-madea", "faas-diffuse", "faas-powd",
  }

def test_e8_default_count_and_spatial_latency_coverage():
  experiments = paper.build_e8()
  assert len(experiments) == 3 * 2 * 4 * 5
  assert {e.config["limits"]["Nn"]["min"] for e in experiments} == {20, 50, 100}
```

Delete `test_e1_expands_function_vectors_and_output_folder`'s dependence on `algorithms=("hierarchical",)`? No — leave it; `build_e1` keeps its `algorithms` keyword. Keep `test_generated_config_builds_a_real_instance` and `test_paper_experiments_round_trip_as_batch` (they pass explicit `algorithms=`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_paper_experiment_suites.py -k "e1 or e2 or e3 or e4 or e5 or e6 or e7 or e8" -v`
Expected: FAIL with old counts (1800/4230/…).

- [ ] **Step 3: Implement**

In `paper.py`, change the builder signatures/defaults:

`build_e1`: default `algorithms: tuple[str, ...] = ()` and, at call, `algorithms = algorithms or _final_algorithms()`. Concretely:

```python
@register_suite("paper-e1-quality-runtime")
def build_e1(
    seeds: tuple[int, ...] = CONFIRMATORY_SEEDS,
    algorithms: tuple[str, ...] = (),
  ) -> list[Experiment]:
  suite = "paper-e1-quality-runtime"
  algorithms = algorithms or _final_algorithms()
  return [
    _experiment(
      suite, f"n{nodes}-f{functions}-planar3", algorithm, seed,
      _new_config(nodes, functions, _euclidean_planar()),
    )
    for nodes in (10, 20, 30)
    for functions in (2, 4)
    for algorithm in algorithms
    for seed in seeds
  ]
```

`build_e2`: add 500 to the node loop and switch the non-centralized set to survivors + `hierarchical-madea`:

```python
@register_suite("paper-e2-scalability")
def build_e2(
    seeds: tuple[int, ...] = CONFIRMATORY_SEEDS,
  ) -> list[Experiment]:
  suite = "paper-e2-scalability"
  scalable = ("hierarchical-madea",) + _survivors()
  experiments = []
  for nodes in (10, 20, 50, 100, 200, 500):
    algorithms = scalable + (("centralized",) if nodes <= 20 else ())
    for functions in (2, 4, 8):
      for algorithm in algorithms:
        for seed in seeds:
          experiments.append(_experiment(
            suite, f"n{nodes}-f{functions}-reg3", algorithm, seed,
            _new_config(nodes, functions, {"k": 3}),
          ))
  return experiments
```

`build_e3`, `build_e4`, `build_e5`: change their `algorithms` default from `REPRESENTATIVE_ALGORITHMS` to `()` and add `algorithms = algorithms or _final_algorithms()` as the first line of the body (same pattern as e1).

`build_e6`: change the node loop from `for nodes in (20, 50):` to `for nodes in (50,):`. No other change.

`build_e7`: default `algorithms: tuple[str, ...] = ()`; first body line `algorithms = algorithms or _tunable(_final_algorithms())`. The existing `section = {...}[algorithm]` map must be replaced with `section = WEIGHT_TUNABLE[algorithm]` (guaranteed present because `_tunable` filtered).

`build_e8`: same treatment as e7 — default `()`, `algorithms = algorithms or _tunable(_final_algorithms())`, and `section = WEIGHT_TUNABLE[algorithm]`.

Delete the now-unused `REPRESENTATIVE_ALGORITHMS` and `TRADEOFF_ALGORITHMS` constants if nothing else references them (grep first).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_paper_experiment_suites.py -v`
Expected: PASS (all).

- [ ] **Step 5: Verify total campaign size**

Run:
```bash
uv run python -c "
from remote_experiments.definitions import paper as p
suites=['build_screening','build_e1','build_e2','build_e3','build_e4','build_e5','build_e6','build_e7','build_e8']
tot=sum(len(getattr(p,s)()) for s in suites)
print('total', tot)
assert tot == 1900, tot
"
```
Expected: `total 1900`.

- [ ] **Step 6: Commit**

```bash
git add remote_experiments/definitions/paper.py tests/test_paper_experiment_suites.py
git commit -m "feat(paper): confirmatory suites on survivors+anchors, e2 n=500, e6 n=50"
```

---

### Task 5: `results_reader.py` — per-run objective/runtime/failure

**Files:**
- Create: `remote_experiments/results_reader.py`
- Test: `tests/test_remote_experiments_results_reader.py`

**Interfaces:**
- Produces:
  - `read_run_objective(run_dir: Path) -> float | None` — mean of the single data column of the first `obj.csv` found under `run_dir`; `None` if no `obj.csv` or it is empty.
  - `read_run_runtime(run_dir: Path) -> float | None` — mean of the single data column of the first `runtime.csv`; `None` if absent.
  - `run_failed(run_dir: Path) -> bool` — `read_run_objective(run_dir) is None`.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
from remote_experiments.results_reader import (
  read_run_objective, read_run_runtime, run_failed,
)


def _make_run(tmp_path: Path, obj_rows, runtime_rows=None) -> Path:
  run = tmp_path / "exp-id" / "2026-07-06_00-00-00.000000"
  run.mkdir(parents=True)
  if obj_rows is not None:
    (run / "obj.csv").write_text("Model\n" + "\n".join(str(v) for v in obj_rows) + "\n")
  if runtime_rows is not None:
    (run / "runtime.csv").write_text("tot\n" + "\n".join(str(v) for v in runtime_rows) + "\n")
  return tmp_path / "exp-id"


def test_read_objective_is_column_mean(tmp_path):
  run = _make_run(tmp_path, [10.0, 20.0, 30.0])
  assert read_run_objective(run) == 20.0


def test_read_runtime_is_column_mean(tmp_path):
  run = _make_run(tmp_path, [1.0], runtime_rows=[0.2, 0.4])
  assert read_run_runtime(run) == 0.3


def test_missing_obj_is_failure(tmp_path):
  run = _make_run(tmp_path, None)
  assert read_run_objective(run) is None
  assert run_failed(run) is True


def test_present_obj_is_not_failure(tmp_path):
  run = _make_run(tmp_path, [5.0])
  assert run_failed(run) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_remote_experiments_results_reader.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement**

Create `remote_experiments/results_reader.py`:

```python
"""Read a single experiment's objective/runtime from its output CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _first_column_mean(run_dir: Path, filename: str) -> float | None:
  matches = sorted(Path(run_dir).rglob(filename))
  if not matches:
    return None
  frame = pd.read_csv(matches[0])
  if frame.empty or frame.shape[1] == 0:
    return None
  return float(frame.iloc[:, 0].mean())


def read_run_objective(run_dir: Path) -> float | None:
  return _first_column_mean(run_dir, "obj.csv")


def read_run_runtime(run_dir: Path) -> float | None:
  return _first_column_mean(run_dir, "runtime.csv")


def run_failed(run_dir: Path) -> bool:
  return read_run_objective(run_dir) is None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_remote_experiments_results_reader.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add remote_experiments/results_reader.py tests/test_remote_experiments_results_reader.py
git commit -m "feat(remote): results_reader for per-run objective/runtime"
```

---

### Task 6: `survivors.py` — relative-to-best selection

**Files:**
- Create: `remote_experiments/survivors.py`
- Test: `tests/test_remote_experiments_survivors.py`

**Interfaces:**
- Consumes: `Batch`/`Experiment` (`batch.py`), `read_run_objective`/`read_run_runtime`/`run_failed` (`results_reader.py`).
- Produces: `select_survivors(batch: Batch, results_dir: Path, *, anchors: tuple[str, ...] = ("centralized", "hierarchical-madea"), n: int = 4, min_valid_fraction: float = 0.8) -> list[str]`. Raises `SurvivorSelectionError` when the guard trips.

- [ ] **Step 1: Write the failing test**

```python
import pytest
from remote_experiments.batch import Batch, Experiment
from remote_experiments.survivors import select_survivors, SurvivorSelectionError


def _exp(algo, nodes, seed):
  return Experiment(
    id=f"paper-a-screening-n{nodes}-f2-planar3-{algo}-s{seed}",
    suite="paper-a-screening", algorithm=algo, seed=seed,
    graph_params={"Nn": {"min": nodes, "max": nodes}, "Nf": {"min": 2, "max": 2}},
    load_params={}, config={},
  )


def _write_result(results_dir, exp, obj, runtime=0.1):
  run = results_dir / exp.id / "2026-07-06_00-00-00.000000"
  run.mkdir(parents=True)
  if obj is not None:
    (run / "obj.csv").write_text(f"Model\n{obj}\n")
  (run / "runtime.csv").write_text(f"tot\n{runtime}\n")


def test_selects_lowest_relative_deficit(tmp_path):
  # per instance obj_best is the max; rank candidates by mean deficit to it
  objs = {  # algo -> obj on the single instance (n50, seed1)
    "hierarchical-madea": 100.0,  # anchor, best, but not promotable
    "faas-madea": 99.0, "faas-diffuse": 98.0, "faas-powd": 97.0,
    "faas-br-o": 96.0, "faas-gcaa": 50.0,
  }
  exps = [_exp(a, 50, 1) for a in objs]
  batch = Batch(suite="paper-a-screening", experiments=tuple(exps))
  for e in exps:
    _write_result(tmp_path, e, objs[e.algorithm])
  survivors = select_survivors(batch, tmp_path, n=4)
  assert survivors == ["faas-madea", "faas-diffuse", "faas-powd", "faas-br-o"]
  assert "hierarchical-madea" not in survivors


def test_runtime_breaks_ties(tmp_path):
  exps = [_exp("faas-madea", 50, 1), _exp("faas-powd", 50, 1),
          _exp("hierarchical-madea", 50, 1)]
  batch = Batch(suite="paper-a-screening", experiments=tuple(exps))
  _write_result(tmp_path, exps[0], 90.0, runtime=0.5)
  _write_result(tmp_path, exps[1], 90.0, runtime=0.1)   # same obj, faster
  _write_result(tmp_path, exps[2], 100.0, runtime=0.9)
  assert select_survivors(batch, tmp_path, n=1) == ["faas-powd"]


def test_guard_trips_on_too_many_failures(tmp_path):
  exps = [_exp("faas-madea", 50, 1), _exp("faas-powd", 50, 1)]
  batch = Batch(suite="paper-a-screening", experiments=tuple(exps))
  _write_result(tmp_path, exps[0], 90.0)
  _write_result(tmp_path, exps[1], None)   # failed
  with pytest.raises(SurvivorSelectionError):
    select_survivors(batch, tmp_path, n=1, min_valid_fraction=0.8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_remote_experiments_survivors.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement**

Create `remote_experiments/survivors.py`:

```python
"""Select survivor algorithms from screening results by relative-to-best objective."""

from __future__ import annotations

import statistics
from pathlib import Path

from .batch import Batch
from .results_reader import read_run_objective, read_run_runtime


class SurvivorSelectionError(RuntimeError):
  """Raised when screening results are too degenerate to promote survivors."""


def _instance_key(experiment) -> tuple[int, int, int]:
  limits = experiment.graph_params
  return (limits["Nn"]["min"], limits["Nf"]["min"], experiment.seed)


def select_survivors(
    batch: Batch,
    results_dir: Path,
    *,
    anchors: tuple[str, ...] = ("centralized", "hierarchical-madea"),
    n: int = 4,
    min_valid_fraction: float = 0.8,
  ) -> list[str]:
  results_dir = Path(results_dir)
  obj_by_algo: dict[str, dict[tuple, float]] = {}
  runtime_by_algo: dict[str, list[float]] = {}
  valid = 0
  for e in batch.experiments:
    objective = read_run_objective(results_dir / e.id)
    if objective is None:
      continue
    valid += 1
    obj_by_algo.setdefault(e.algorithm, {})[_instance_key(e)] = objective
    runtime = read_run_runtime(results_dir / e.id)
    if runtime is not None:
      runtime_by_algo.setdefault(e.algorithm, []).append(runtime)

  if valid < min_valid_fraction * len(batch.experiments):
    raise SurvivorSelectionError(
      f"only {valid}/{len(batch.experiments)} screening runs valid "
      f"(< {min_valid_fraction:.0%}); refusing to promote"
    )

  # obj_best per instance across every algorithm that ran it
  best: dict[tuple, float] = {}
  for per_instance in obj_by_algo.values():
    for key, value in per_instance.items():
      best[key] = max(value, best.get(key, float("-inf")))

  candidates = [a for a in obj_by_algo if a not in anchors]
  scored = []
  for algo in candidates:
    deficits = [
      (best[key] - obj) / best[key] * 100.0
      for key, obj in obj_by_algo[algo].items()
    ]
    reldef = statistics.fmean(deficits)
    runtime = statistics.median(runtime_by_algo.get(algo, [float("inf")]))
    scored.append((reldef, runtime, algo))

  if len(scored) < n:
    raise SurvivorSelectionError(
      f"only {len(scored)} candidate algorithms produced valid runs; need {n}"
    )
  scored.sort(key=lambda t: (t[0], t[1]))
  return [algo for _, _, algo in scored[:n]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_remote_experiments_survivors.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add remote_experiments/survivors.py tests/test_remote_experiments_survivors.py
git commit -m "feat(remote): relative-to-best survivor selection with guard"
```

---

### Task 7: `campaign.py` orchestrator + `campaign` subcommand

**Files:**
- Create: `remote_experiments/campaign.py`
- Modify: `remote_experiments/cli.py` (extract `execute_batch`, add `campaign` subcommand)
- Test: `tests/test_remote_experiments_campaign.py`

**Interfaces:**
- Consumes: `get_suite`, `Batch`, `materialize_batch`, `Manifest`, `select_survivors`, and a batch-execution callable.
- Produces:
  - `execute_batch(batch_path, args) -> None` in `cli.py` — the Dispatcher/Project/run_batch wiring currently inside `cmd_run`, callable non-interactively (`select all pending`).
  - `run_campaign(args) -> None` in `campaign.py` — drives stages via `campaign-state.json`.
  - `CONFIRMATORY_SUITES: tuple[str, ...]`, `SCREENING_SUITE = "paper-a-screening"`, `write_survivors(path, survivors)`.

- [ ] **Step 1: Write the failing test (stage sequencing with fakes)**

```python
import json
from pathlib import Path
import remote_experiments.campaign as campaign


def test_campaign_runs_screening_then_selects_then_confirms(tmp_path, monkeypatch):
  calls = []

  def fake_run_suite(suite, args, state):
    calls.append(suite)

  def fake_select(batch, results_dir, **kw):
    return ["faas-madea", "faas-diffuse", "faas-powd", "faas-br-o"]

  monkeypatch.setattr(campaign, "_run_suite", fake_run_suite)
  monkeypatch.setattr(campaign, "select_survivors", fake_select)
  monkeypatch.setattr(campaign, "SURVIVORS_PATH", tmp_path / "survivors.json")
  monkeypatch.setattr(campaign, "STATE_PATH", tmp_path / "campaign-state.json")

  args = _fake_args(tmp_path)   # namespace with inventory/results-dir/etc.
  campaign.run_campaign(args)

  assert calls[0] == "paper-a-screening"
  assert calls[1:] == list(campaign.CONFIRMATORY_SUITES)
  survivors = json.loads((tmp_path / "survivors.json").read_text())["survivors"]
  assert survivors == ["faas-madea", "faas-diffuse", "faas-powd", "faas-br-o"]


def test_campaign_resumes_from_state(tmp_path, monkeypatch):
  calls = []
  monkeypatch.setattr(campaign, "_run_suite", lambda s, a, st: calls.append(s))
  monkeypatch.setattr(campaign, "select_survivors", lambda *a, **k: ["x", "y", "z", "w"])
  monkeypatch.setattr(campaign, "SURVIVORS_PATH", tmp_path / "survivors.json")
  state = tmp_path / "campaign-state.json"
  monkeypatch.setattr(campaign, "STATE_PATH", state)
  (tmp_path / "survivors.json").write_text(json.dumps({"survivors": ["x", "y", "z", "w"]}))
  state.write_text(json.dumps({"stage": "confirm", "done_suites": ["paper-e1-quality-runtime"]}))

  campaign.run_campaign(_fake_args(tmp_path))
  assert "paper-a-screening" not in calls
  assert "paper-e1-quality-runtime" not in calls
  assert calls[0] == "paper-e2-scalability"
```

Add a `_fake_args(tmp_path)` helper returning an `argparse.Namespace` with the fields `execute_batch`/`_run_suite` read (inventory, results_dir, instances, gurobi_license, project_path, python_version, uv_version).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_remote_experiments_campaign.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Extract `execute_batch` in `cli.py`**

Move the body of `cmd_run` after selection (the `Inventory.from_yaml(...)` through the completion print) into:

```python
def execute_batch(batch, manifest, manifest_path, selected, args) -> bool:
  inventory = Inventory.from_yaml(args.inventory)
  secrets = ()
  if args.gurobi_license:
    secrets = (
      SecretFile(source=args.gurobi_license, remote_name="gurobi.lic", env_var="GRB_LICENSE_FILE"),
    )
  project = Project(
    path=str(Path(args.project_path).resolve()),
    project_id="dfaas-optimizer",
    python=args.python_version,
    uv_version=args.uv_version,
    secrets=secrets,
    exclude=(
      ".venv/", ".git/", "solutions/", "results/", "batches/",
      "remote_experiments/instances/",
    ),
  )
  config_dir = manifest_path.parent / f"{manifest_path.stem}-configs"
  jobs = [experiment_to_job(e, config_dir, Path(args.instances)) for e in selected]
  with Dispatcher(inventory, project, results_dir=args.results_dir) as dispatcher:
    start = time.monotonic()
    with live_view(batch, manifest, inventory, start_time=start) as on_tick:
      return run_batch(dispatcher, jobs, manifest, on_tick)
```

`cmd_run` keeps its selection logic (incl. `--yes` from Task 1) and calls `execute_batch(...)`, then prints the completion/failure summary as before.

- [ ] **Step 4: Implement `campaign.py`**

```python
"""Orchestrate the two-phase campaign: screening -> select survivors -> confirm."""

from __future__ import annotations

import json
from pathlib import Path

from .batch import Batch
from .definitions import get_suite
from .definitions.paper import SURVIVORS_PATH
from .instances import materialize_batch
from .manifest import Manifest
from .selection import default_selection
from .survivors import select_survivors

SCREENING_SUITE = "paper-a-screening"
CONFIRMATORY_SUITES = (
  "paper-e1-quality-runtime", "paper-e2-scalability", "paper-e3-topology",
  "paper-e4-robustness", "paper-e5-dynamics", "paper-e6-ablation",
  "paper-e7-tradeoffs", "paper-e8-spatial-latency",
)
STATE_PATH = Path("batches") / "campaign-state.json"


def _load_state() -> dict:
  if STATE_PATH.exists():
    return json.loads(STATE_PATH.read_text())
  return {"stage": "screening", "done_suites": []}


def _save_state(state: dict) -> None:
  STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
  STATE_PATH.write_text(json.dumps(state, indent=2))


def write_survivors(path: Path, survivors: list[str]) -> None:
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  Path(path).write_text(json.dumps({"survivors": survivors}, indent=2))


def _batch_path(suite: str) -> Path:
  return Path("batches") / f"{suite}.json"


def _define_and_materialize(suite: str, instances_root: str) -> Batch:
  batch = Batch(suite=suite, experiments=tuple(get_suite(suite)()))
  path = _batch_path(suite)
  batch.save(path)
  materialize_batch(batch, instances_root)
  return batch


def _run_suite(suite: str, args, state: dict) -> None:
  # imported lazily to avoid a cli<->campaign import cycle
  from .cli import execute_batch
  batch = _define_and_materialize(suite, args.instances)
  manifest_path = _batch_path(suite).with_suffix(".manifest.json")
  manifest = Manifest(manifest_path)
  selected_idx = default_selection([e.id for e in batch.experiments], manifest)
  selected = [batch.experiments[i] for i in selected_idx]
  if selected:
    execute_batch(batch, manifest, manifest_path, selected, args)


def run_campaign(args) -> None:
  state = _load_state()

  if state["stage"] == "screening":
    _run_suite(SCREENING_SUITE, args, state)
    state["stage"] = "select"
    _save_state(state)

  if state["stage"] == "select":
    batch = Batch.load(_batch_path(SCREENING_SUITE))
    survivors = select_survivors(batch, Path(args.results_dir))
    write_survivors(SURVIVORS_PATH, survivors)
    print(f"survivors: {survivors}")
    state["stage"] = "confirm"
    _save_state(state)

  if state["stage"] == "confirm":
    for suite in CONFIRMATORY_SUITES:
      if suite in state["done_suites"]:
        continue
      _run_suite(suite, args, state)
      state["done_suites"].append(suite)
      _save_state(state)
  print("campaign complete")
```

- [ ] **Step 5: Wire the `campaign` subcommand in `cli.py`**

```python
def cmd_campaign(args: argparse.Namespace) -> None:
  from .campaign import run_campaign
  run_campaign(args)
```

In `build_parser`, add (share the same optional args as `run`):

```python
  campaign_p = sub.add_parser("campaign", help="Run the full two-phase campaign (screen->select->confirm)")
  campaign_p.add_argument("--inventory", required=True)
  campaign_p.add_argument("--project-path", default=".")
  campaign_p.add_argument("--results-dir", default="./results")
  campaign_p.add_argument("--instances", default="remote_experiments/instances")
  campaign_p.add_argument("--gurobi-license", default=None)
  campaign_p.add_argument("--python-version", default="3.10.19")
  campaign_p.add_argument("--uv-version", default="0.11.25")
  campaign_p.set_defaults(func=cmd_campaign)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_remote_experiments_campaign.py tests/test_remote_experiments_cli.py -v`
Expected: PASS (campaign sequencing + resume, and cli still green after the `execute_batch` extraction).

- [ ] **Step 7: Full suite regression**

Run: `uv run pytest tests/ -k "remote or paper or campaign or survivors or results_reader" -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add remote_experiments/campaign.py remote_experiments/cli.py tests/test_remote_experiments_campaign.py
git commit -m "feat(remote): campaign orchestrator with resumable screen/select/confirm"
```

---

### Task 8: Docs — README campaign flow

**Files:**
- Modify: `remote_experiments/README.md` (replace the "Paper experiment batches" section)

- [ ] **Step 1: Replace the manual per-suite loop instructions**

Under "Paper experiment batches", document the one-command flow:

```markdown
## Run the two-phase paper campaign

One command runs screening, auto-selects the 4 survivor algorithms, then runs
the confirmatory suites on survivors + anchors (centralized, hierarchical-madea):

    uv run -m remote_experiments campaign --inventory my-inventory.yaml \
      --gurobi-license ~/gurobi.lic

Progress is checkpointed in `batches/campaign-state.json` and each suite has its
own `batches/<suite>.manifest.json`, so re-running `campaign` resumes where it
stopped. Survivors are written to `batches/survivors.json`; delete it (and reset
the state file to `{"stage":"screening","done_suites":[]}`) to re-screen.

To inspect a single suite without the orchestrator, the old
`define`/`materialize`/`run` commands still work (add `--yes` to `run` to skip
the prompt).
```

- [ ] **Step 2: Commit**

```bash
git add remote_experiments/README.md
git commit -m "docs(remote): document one-command two-phase campaign"
```

---

## Self-Review notes

- **Spec coverage:** screening suite (T3), relative-to-best metric incl. anchor-as-reference and non-promotable (T6), 5 seeds (T2), e2 n=500 / e6 n=50 (T4), tunable e7/e8 subset (T2+T4), `--yes` (T1), survivors.json injection (T2), orchestrator + campaign-state resume + guard (T6+T7), fallback when survivors.json absent (T2), README (T8). Total-count check (1900) in T4.
- **Not automated (by design):** none — full auto-chain per approved spec. The only stop is the guard raising `SurvivorSelectionError`.
- **Type consistency:** `select_survivors(batch, results_dir, *, anchors, n, min_valid_fraction)` used identically in T6 and T7; `execute_batch(batch, manifest, manifest_path, selected, args)` defined in T7 Step 3 and called in `cmd_run` and `_run_suite`; `SURVIVORS_PATH` is the single source in `paper.py`, imported by `campaign.py`.
- **Statistical note (spec):** 5-seed thinness is a reporting caveat, not code; `CONFIRMATORY_SEEDS` is one editable tuple (T2) — no task needed.
