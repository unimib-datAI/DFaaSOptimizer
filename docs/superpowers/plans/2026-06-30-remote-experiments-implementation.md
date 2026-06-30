# Remote Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `remote_experiments/`, a two-phase (define / run) tool that dispatches DFaaSOptimizer algorithm-comparison experiments to Gurobi-equipped VMs via `ray-dispatcher`, with a pluggable experiment-suite registry and a `rich` TUI showing live per-experiment and per-VM progress, supporting stop/resume across process restarts.

**Architecture:** `definitions/` builds `Experiment` lists from a registry of pluggable suite functions; `batch.py` serializes them to a static JSON file; `jobs.py` maps each `Experiment` to a `ray_dispatcher.Job`; `runner.py` drives `Dispatcher.submit/status/cancel/running_hosts` in a poll loop, persisting state via `manifest.py`; `tui.py` renders that state with `rich`; `cli.py` wires it all behind `define`/`run` subcommands.

**Tech Stack:** Python 3.10 (`uv`), `ray-dispatcher` (local sibling checkout, path dependency), `rich`, `pytest` — no new test framework.

## Global Constraints

- Indentation: 2 spaces (`pyproject.toml` `[tool.ruff] indent-width = 2`) — this repo's style, distinct from `ray-dispatcher`'s 4-space.
- Line length: 100.
- Python: `>=3.10,<3.11`; this machine's interpreter is `3.10.19`, `uv` is `0.11.25`.
- New tests go in `tests/` (flat, matches existing convention — no `tests/remote_experiments/` subdirectory).
- Prerequisite: `ray-dispatcher`'s `Dispatcher.running_hosts()` must exist — implemented by
  `~/Downloads/ray-dispatcher`'s `docs/superpowers/plans/2026-06-30-ray-dispatcher-phase-9-live-host-visibility.md`.
  **Run that plan first if it hasn't been executed yet** — Task 9 of this plan fails without it.
- Reference design: [docs/superpowers/specs/2026-06-30-remote-experiments-design.md](../specs/2026-06-30-remote-experiments-design.md).
- `Dispatcher` has no public `resolve()` — never call it. Job duration is computed locally
  (wall-clock from submit to terminal status), not read from `JobResult`.

---

### Task 1: Add `ray-dispatcher` and `rich` dependencies

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `import ray_dispatcher`, `import rich` available in the project's `uv` environment.

- [ ] **Step 1: Add the dependencies**

In `pyproject.toml`, add to the `dependencies` list (after `"pyyaml>=6.0.2",`):

```toml
  "pyyaml>=6.0.2",
  "ray-dispatcher",
  "rich>=13,<15",
```

Add a new section after `[tool.uv]`:

```toml
[tool.uv]
package = false

[tool.uv.sources]
ray-dispatcher = { path = "../ray-dispatcher", editable = true }
```

- [ ] **Step 2: Sync and verify**

Run: `cd /Users/micheleciavotta/Downloads/DFaaSOptimizer && uv sync`
Expected: resolves and installs `ray-dispatcher` (editable, from `../ray-dispatcher`), `ray`, `fabric`, `rich`, and their transitive deps without conflicts.

Run: `uv run python -c "import ray_dispatcher, rich; print(ray_dispatcher.__version__)"`
Expected: prints `0.1.0`, no import errors.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add ray-dispatcher and rich for remote_experiments"
```

---

### Task 2: Expose `faas-macro-v0` via a CLI flag

**Files:**
- Modify: `run_faasmacro.py` (the `parse_arguments` function and the `__main__` block, lines ~50-78 and ~1142-1155)
- Test: `tests/test_run_faasmacro_v0_flag.py`

**Interfaces:**
- Produces: `run_faasmacro.py --v0` sets `args.v0 = True`, passed through to `run(..., v0=True)`. `run()` already accepts `v0: bool = False` (line 625) — only the CLI wiring is missing.

**Why:** `remote_experiments` dispatches algorithms as subprocess CLI calls (`uv run run_faasmacro.py -c config.json ...`). `run.py`'s in-process orchestrator currently reaches `v0=True` by calling `run_iterations(..., v0=True)` directly in Python — there is no CLI path to it. `SCRIPT_BY_ALGORITHM` (Task 6) needs one for the `faas-macro-v0` algorithm.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_faasmacro_v0_flag.py
import run_faasmacro


def test_parse_arguments_default_v0_is_false(monkeypatch):
  monkeypatch.setattr("sys.argv", ["run_faasmacro.py"])
  args = run_faasmacro.parse_arguments()
  assert args.v0 is False


def test_parse_arguments_accepts_v0_flag(monkeypatch):
  monkeypatch.setattr("sys.argv", ["run_faasmacro.py", "--v0"])
  args = run_faasmacro.parse_arguments()
  assert args.v0 is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_run_faasmacro_v0_flag.py -v`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'v0'`

- [ ] **Step 3: Implement**

In `run_faasmacro.py`, inside `parse_arguments()`, add after the `--disable_plotting` argument:

```python
  parser.add_argument(
    "--v0",
    help = "Use the v0 (non-accelerated) FaaS-MACrO iteration variant",
    default = False,
    action = "store_true"
  )
```

In the `if __name__ == "__main__":` block, change:

```python
  run(
    config, 
    parallelism, 
    log_on_file = False, 
    disable_plotting = disable_plotting
  )
```

to:

```python
  run(
    config, 
    parallelism, 
    log_on_file = False, 
    disable_plotting = disable_plotting,
    v0 = args.v0
  )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_run_faasmacro_v0_flag.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add run_faasmacro.py tests/test_run_faasmacro_v0_flag.py
git commit -m "feat: expose faas-macro v0 variant via --v0 CLI flag"
```

---

### Task 3: `remote_experiments` package + `Experiment`/`Batch`

**Files:**
- Create: `remote_experiments/__init__.py`
- Create: `remote_experiments/batch.py`
- Test: `tests/test_remote_experiments_batch.py`

**Interfaces:**
- Produces:
  - `Experiment(id, suite, algorithm, seed, graph_params, load_params, config)` — frozen dataclass, `.to_dict()`/`.from_dict()`
  - `Batch(suite, experiments)` — frozen dataclass, `.to_dict()`/`.from_dict()`/`.save(path)`/`.load(path)` (classmethod)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_remote_experiments_batch.py
import json

from remote_experiments.batch import Batch, Experiment


def _experiment(id="e1", seed=1):
  return Experiment(
    id=id, suite="smoke", algorithm="centralized", seed=seed,
    graph_params={"Nn": 10}, load_params={"trace_type": "sinusoidal"},
    config={"seed": seed, "base_solution_folder": f"solutions/{id}"},
  )


def test_experiment_round_trips_through_dict():
  e = _experiment()
  assert Experiment.from_dict(e.to_dict()) == e


def test_batch_save_and_load_round_trips(tmp_path):
  batch = Batch(suite="smoke", experiments=(_experiment("e1", 1), _experiment("e2", 2)))
  path = tmp_path / "batch.json"
  batch.save(path)
  loaded = Batch.load(path)
  assert loaded == batch


def test_batch_save_writes_readable_json(tmp_path):
  batch = Batch(suite="smoke", experiments=(_experiment(),))
  path = tmp_path / "batch.json"
  batch.save(path)
  raw = json.loads(path.read_text())
  assert raw["suite"] == "smoke"
  assert raw["experiments"][0]["id"] == "e1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_remote_experiments_batch.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'remote_experiments'`

- [ ] **Step 3: Implement**

```python
# remote_experiments/__init__.py
"""Define-then-run batches of DFaaSOptimizer experiments on remote Gurobi VMs via ray-dispatcher."""
```

```python
# remote_experiments/batch.py
"""Experiment and Batch value objects: build once, serialize, run later."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Experiment:
  id: str
  suite: str
  algorithm: str
  seed: int
  graph_params: dict
  load_params: dict
  config: dict

  def to_dict(self) -> dict:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: dict) -> Experiment:
    return cls(**data)


@dataclass(frozen=True)
class Batch:
  suite: str
  experiments: tuple[Experiment, ...]

  def to_dict(self) -> dict:
    return {"suite": self.suite, "experiments": [e.to_dict() for e in self.experiments]}

  @classmethod
  def from_dict(cls, data: dict) -> Batch:
    return cls(
      suite=data["suite"],
      experiments=tuple(Experiment.from_dict(e) for e in data["experiments"]),
    )

  def save(self, path: str | Path) -> None:
    Path(path).write_text(json.dumps(self.to_dict(), indent=2))

  @classmethod
  def load(cls, path: str | Path) -> Batch:
    return cls.from_dict(json.loads(Path(path).read_text()))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_remote_experiments_batch.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add remote_experiments/__init__.py remote_experiments/batch.py tests/test_remote_experiments_batch.py
git commit -m "feat: add Experiment/Batch value objects for remote_experiments"
```

---

### Task 4: Suite registry

**Files:**
- Create: `remote_experiments/definitions/__init__.py`
- Test: `tests/test_remote_experiments_registry.py`

**Interfaces:**
- Consumes: `Experiment` (Task 3)
- Produces: `register_suite(name)` (decorator), `get_suite(name) -> Callable[..., list[Experiment]]`, `list_suites() -> list[str]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_remote_experiments_registry.py
import pytest

from remote_experiments.definitions import get_suite, list_suites, register_suite


def test_get_suite_unknown_raises_key_error():
  with pytest.raises(KeyError):
    get_suite("does-not-exist")


def test_register_suite_makes_it_listable():
  @register_suite("__test_registry_suite__")
  def build():
    return []

  assert "__test_registry_suite__" in list_suites()
  assert get_suite("__test_registry_suite__") is build


def test_register_suite_rejects_duplicate_name():
  @register_suite("__test_dup__")
  def build_a():
    return []

  with pytest.raises(ValueError):
    @register_suite("__test_dup__")
    def build_b():
      return []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_remote_experiments_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'remote_experiments.definitions'`

- [ ] **Step 3: Implement**

```python
# remote_experiments/definitions/__init__.py
"""Registry of pluggable experiment suites: @register_suite("name") -> builder fn."""

from collections.abc import Callable

from ..batch import Experiment

_REGISTRY: dict[str, Callable[..., list[Experiment]]] = {}


def register_suite(name: str):
  def decorator(fn: Callable[..., list[Experiment]]) -> Callable[..., list[Experiment]]:
    if name in _REGISTRY:
      raise ValueError(f"suite already registered: {name}")
    _REGISTRY[name] = fn
    return fn
  return decorator


def get_suite(name: str) -> Callable[..., list[Experiment]]:
  if name not in _REGISTRY:
    raise KeyError(f"unknown suite: {name!r}; available: {sorted(_REGISTRY)}")
  return _REGISTRY[name]


def list_suites() -> list[str]:
  return sorted(_REGISTRY)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_remote_experiments_registry.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add remote_experiments/definitions/__init__.py tests/test_remote_experiments_registry.py
git commit -m "feat: add experiment suite registry"
```

---

### Task 5: `smoke` suite (concrete pluggable example)

**Files:**
- Create: `remote_experiments/definitions/smoke.py`
- Modify: `remote_experiments/definitions/__init__.py` (add the registration-side-effect import)
- Test: `tests/test_remote_experiments_smoke_suite.py`

**Interfaces:**
- Consumes: `register_suite` (Task 4), `Experiment` (Task 3), `config_files/eval_smoke.json` (existing file)
- Produces: `ALGORITHMS` (tuple of 10 algorithm names), `build(seeds=(42, 43), algorithms=ALGORITHMS) -> list[Experiment]`, registered under the name `"smoke"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_remote_experiments_smoke_suite.py
from remote_experiments.definitions import get_suite
from remote_experiments.definitions.smoke import ALGORITHMS, build


def test_smoke_suite_is_registered():
  assert get_suite("smoke") is build


def test_build_produces_one_experiment_per_seed_and_algorithm():
  experiments = build(seeds=(1, 2), algorithms=("centralized", "faas-macro"))
  assert len(experiments) == 4


def test_build_produces_unique_ids():
  experiments = build(seeds=(1, 2), algorithms=("centralized", "faas-macro"))
  ids = [e.id for e in experiments]
  assert len(ids) == len(set(ids))


def test_build_sets_seed_in_config_and_base_solution_folder():
  experiments = build(seeds=(7,), algorithms=("centralized",))
  e = experiments[0]
  assert e.config["seed"] == 7
  assert e.config["base_solution_folder"] == f"solutions/{e.id}"


def test_build_default_covers_all_ten_algorithms():
  experiments = build()
  assert {e.algorithm for e in experiments} == set(ALGORITHMS)
  assert len(ALGORITHMS) == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_remote_experiments_smoke_suite.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'remote_experiments.definitions.smoke'`

- [ ] **Step 3: Implement**

```python
# remote_experiments/definitions/smoke.py
"""Example pluggable suite: every algorithm across a couple of seeds, from eval_smoke.json."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from ..batch import Experiment
from . import register_suite

_BASE_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config_files" / "eval_smoke.json"

ALGORITHMS = (
  "centralized", "faas-macro", "faas-macro-v0", "faas-madea", "hierarchical",
  "faas-diffuse", "faas-powd", "faas-br-s", "faas-br-r", "faas-br-o",
)


@register_suite("smoke")
def build(
    seeds: tuple[int, ...] = (42, 43),
    algorithms: tuple[str, ...] = ALGORITHMS,
  ) -> list[Experiment]:
  base_config = json.loads(_BASE_CONFIG_PATH.read_text())
  limits = base_config["limits"]
  graph_params = {k: v for k, v in limits.items() if k != "load"}
  load_params = limits["load"]
  experiments = []
  for seed in seeds:
    for algorithm in algorithms:
      config = copy.deepcopy(base_config)
      config["seed"] = seed
      experiment_id = f"smoke-{algorithm}-seed{seed}"
      config["base_solution_folder"] = f"solutions/{experiment_id}"
      experiments.append(Experiment(
        id=experiment_id, suite="smoke", algorithm=algorithm, seed=seed,
        graph_params=graph_params, load_params=load_params, config=config,
      ))
  return experiments
```

Add the import so the registration side effect runs whenever the registry is imported, at the bottom of `remote_experiments/definitions/__init__.py`:

```python
from . import smoke  # imported for its @register_suite("smoke") side effect
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_remote_experiments_smoke_suite.py tests/test_remote_experiments_registry.py -v`
Expected: PASS (8 tests total)

- [ ] **Step 5: Commit**

```bash
git add remote_experiments/definitions/smoke.py remote_experiments/definitions/__init__.py tests/test_remote_experiments_smoke_suite.py
git commit -m "feat: add smoke suite covering all 10 algorithms"
```

---

### Task 6: Experiment → Job mapping

**Files:**
- Create: `remote_experiments/jobs.py`
- Test: `tests/test_remote_experiments_jobs.py`

**Interfaces:**
- Consumes: `Experiment` (Task 3), `ray_dispatcher.{Job, InputSpec, OutputSpec}`
- Produces: `SCRIPT_BY_ALGORITHM: dict[str, tuple[str, tuple[str, ...]]]`, `experiment_to_job(experiment, config_dir) -> Job`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_remote_experiments_jobs.py
import json

import pytest

from remote_experiments.batch import Experiment
from remote_experiments.jobs import SCRIPT_BY_ALGORITHM, experiment_to_job


def _experiment(algorithm="centralized"):
  return Experiment(
    id="e1", suite="smoke", algorithm=algorithm, seed=1,
    graph_params={}, load_params={}, config={"seed": 1, "base_solution_folder": "solutions/e1"},
  )


def test_all_ten_algorithms_are_mapped():
  expected = {
    "centralized", "faas-macro", "faas-macro-v0", "faas-madea", "hierarchical",
    "faas-diffuse", "faas-powd", "faas-br-s", "faas-br-r", "faas-br-o",
  }
  assert set(SCRIPT_BY_ALGORITHM) == expected


def test_experiment_to_job_builds_expected_command_for_variant_algorithm(tmp_path):
  job = experiment_to_job(_experiment("faas-br-r"), tmp_path)
  assert job.command == (
    "uv", "run", "decentralized_bestresponse.py", "-c", "config.json",
    "--disable_plotting", "--variant", "r",
  )


def test_experiment_to_job_builds_expected_command_for_plain_algorithm(tmp_path):
  job = experiment_to_job(_experiment("centralized"), tmp_path)
  assert job.command == (
    "uv", "run", "run_centralized_model.py", "-c", "config.json", "--disable_plotting",
  )


def test_experiment_to_job_writes_config_file(tmp_path):
  job = experiment_to_job(_experiment(), tmp_path)
  config_path = tmp_path / "e1.json"
  assert config_path.exists()
  assert json.loads(config_path.read_text())["seed"] == 1
  assert job.inputs[0].destination == "config.json"


def test_experiment_to_job_output_matches_experiment_id(tmp_path):
  job = experiment_to_job(_experiment(), tmp_path)
  assert job.outputs[0].source == "solutions/e1"
  assert job.outputs[0].destination == "e1"


def test_experiment_to_job_unknown_algorithm_raises(tmp_path):
  with pytest.raises(KeyError):
    experiment_to_job(_experiment("nope"), tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_remote_experiments_jobs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'remote_experiments.jobs'`

- [ ] **Step 3: Implement**

```python
# remote_experiments/jobs.py
"""Maps an Experiment to a ray_dispatcher.Job that runs it on a VM."""

from __future__ import annotations

import json
from pathlib import Path

from ray_dispatcher import InputSpec, Job, OutputSpec

from .batch import Experiment

SCRIPT_BY_ALGORITHM: dict[str, tuple[str, tuple[str, ...]]] = {
  "centralized":   ("run_centralized_model.py", ()),
  "faas-macro":    ("run_faasmacro.py", ()),
  "faas-macro-v0": ("run_faasmacro.py", ("--v0",)),
  "faas-madea":    ("run_faasmadea.py", ()),
  "hierarchical":  ("hierarchical_auction/runner.py", ()),
  "faas-diffuse":  ("decentralized_diffusion.py", ()),
  "faas-powd":     ("decentralized_powerd.py", ()),
  "faas-br-s":     ("decentralized_bestresponse.py", ("--variant", "s")),
  "faas-br-r":     ("decentralized_bestresponse.py", ("--variant", "r")),
  "faas-br-o":     ("decentralized_bestresponse.py", ("--variant", "o")),
}


def experiment_to_job(experiment: Experiment, config_dir: Path) -> Job:
  if experiment.algorithm not in SCRIPT_BY_ALGORITHM:
    raise KeyError(f"unknown algorithm: {experiment.algorithm!r}")
  script, extra_args = SCRIPT_BY_ALGORITHM[experiment.algorithm]
  config_dir.mkdir(parents=True, exist_ok=True)
  config_path = config_dir / f"{experiment.id}.json"
  config_path.write_text(json.dumps(experiment.config, indent=2))
  return Job(
    id=experiment.id,
    command=("uv", "run", script, "-c", "config.json", "--disable_plotting", *extra_args),
    inputs=(InputSpec(source=str(config_path), destination="config.json"),),
    outputs=(OutputSpec(source=f"solutions/{experiment.id}", destination=experiment.id, required=True),),
  )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_remote_experiments_jobs.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add remote_experiments/jobs.py tests/test_remote_experiments_jobs.py
git commit -m "feat: map Experiment to ray_dispatcher.Job"
```

---

### Task 7: Manifest (persisted per-experiment status)

**Files:**
- Create: `remote_experiments/manifest.py`
- Test: `tests/test_remote_experiments_manifest.py`

**Interfaces:**
- Produces: `Manifest(path)` with `.status(id) -> str`, `.host(id) -> str | None`, `.duration(id) -> float | None`, `.record(id, **fields)`, `.pending_ids(ids) -> list[str]`; module constants `NEVER_RUN = "never_run"`, `SUCCEEDED = "succeeded"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_remote_experiments_manifest.py
from remote_experiments.manifest import Manifest


def test_unknown_experiment_status_is_never_run(tmp_path):
  manifest = Manifest(tmp_path / "m.json")
  assert manifest.status("e1") == "never_run"


def test_record_updates_status_and_persists(tmp_path):
  path = tmp_path / "m.json"
  manifest = Manifest(path)
  manifest.record("e1", status="running", host="vm1")
  assert manifest.status("e1") == "running"
  assert manifest.host("e1") == "vm1"
  assert path.exists()


def test_record_partial_update_preserves_other_fields(tmp_path):
  manifest = Manifest(tmp_path / "m.json")
  manifest.record("e1", status="running", host="vm1")
  manifest.record("e1", status="succeeded", duration_s=12.5)
  assert manifest.host("e1") == "vm1"
  assert manifest.duration("e1") == 12.5


def test_manifest_reloads_from_disk(tmp_path):
  path = tmp_path / "m.json"
  Manifest(path).record("e1", status="succeeded")
  reloaded = Manifest(path)
  assert reloaded.status("e1") == "succeeded"


def test_pending_ids_excludes_succeeded(tmp_path):
  manifest = Manifest(tmp_path / "m.json")
  manifest.record("e1", status="succeeded")
  manifest.record("e2", status="failed")
  assert manifest.pending_ids(["e1", "e2", "e3"]) == ["e2", "e3"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_remote_experiments_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'remote_experiments.manifest'`

- [ ] **Step 3: Implement**

```python
# remote_experiments/manifest.py
"""Per-experiment status persisted to disk, for cross-process stop/resume."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

NEVER_RUN = "never_run"
SUCCEEDED = "succeeded"


@dataclass
class ManifestEntry:
  status: str = NEVER_RUN
  host: str | None = None
  duration_s: float | None = None


class Manifest:
  def __init__(self, path: str | Path) -> None:
    self._path = Path(path)
    self._entries: dict[str, ManifestEntry] = {}
    if self._path.exists():
      raw = json.loads(self._path.read_text())
      self._entries = {k: ManifestEntry(**v) for k, v in raw.items()}

  def status(self, experiment_id: str) -> str:
    entry = self._entries.get(experiment_id)
    return entry.status if entry else NEVER_RUN

  def host(self, experiment_id: str) -> str | None:
    entry = self._entries.get(experiment_id)
    return entry.host if entry else None

  def duration(self, experiment_id: str) -> float | None:
    entry = self._entries.get(experiment_id)
    return entry.duration_s if entry else None

  def record(self, experiment_id: str, **fields) -> None:
    current = self._entries.get(experiment_id, ManifestEntry())
    updated = ManifestEntry(**{**asdict(current), **fields})
    self._entries[experiment_id] = updated
    self._save()

  def pending_ids(self, experiment_ids: list[str]) -> list[str]:
    return [eid for eid in experiment_ids if self.status(eid) != SUCCEEDED]

  def _save(self) -> None:
    self._path.parent.mkdir(parents=True, exist_ok=True)
    self._path.write_text(
      json.dumps({k: asdict(v) for k, v in self._entries.items()}, indent=2)
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_remote_experiments_manifest.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add remote_experiments/manifest.py tests/test_remote_experiments_manifest.py
git commit -m "feat: add Manifest for persisted per-experiment status"
```

---

### Task 8: Selection parsing and resume defaults

**Files:**
- Create: `remote_experiments/selection.py`
- Test: `tests/test_remote_experiments_selection.py`

**Interfaces:**
- Consumes: `Manifest`, `SUCCEEDED` (Task 7)
- Produces: `parse_selection(text, n) -> list[int]`, `default_selection(experiment_ids, manifest) -> list[int]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_remote_experiments_selection.py
import pytest

from remote_experiments.manifest import Manifest
from remote_experiments.selection import default_selection, parse_selection


def test_parse_selection_all_returns_every_index():
  assert parse_selection("all", 5) == [0, 1, 2, 3, 4]


def test_parse_selection_empty_returns_every_index():
  assert parse_selection("", 5) == [0, 1, 2, 3, 4]


def test_parse_selection_comma_list():
  assert parse_selection("0,2,4", 5) == [0, 2, 4]


def test_parse_selection_range():
  assert parse_selection("1-3", 5) == [1, 2, 3]


def test_parse_selection_mixed():
  assert parse_selection("0, 2-3", 5) == [0, 2, 3]


def test_parse_selection_out_of_range_raises():
  with pytest.raises(ValueError):
    parse_selection("9", 5)


def test_default_selection_skips_succeeded(tmp_path):
  manifest = Manifest(tmp_path / "m.json")
  manifest.record("e1", status="succeeded")
  assert default_selection(["e1", "e2", "e3"], manifest) == [1, 2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_remote_experiments_selection.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'remote_experiments.selection'`

- [ ] **Step 3: Implement**

```python
# remote_experiments/selection.py
"""Parse free-text experiment selection ("1,2,5-7" / "all") and compute resume defaults."""

from __future__ import annotations

from .manifest import Manifest, SUCCEEDED


def parse_selection(text: str, n: int) -> list[int]:
  text = text.strip()
  if text == "" or text.lower() == "all":
    return list(range(n))
  indices: set[int] = set()
  for part in text.split(","):
    part = part.strip()
    if not part:
      continue
    if "-" in part:
      start_s, end_s = part.split("-", 1)
      indices.update(range(int(start_s), int(end_s) + 1))
    else:
      indices.add(int(part))
  for i in indices:
    if not (0 <= i < n):
      raise ValueError(f"selection index out of range: {i} (valid: 0..{n - 1})")
  return sorted(indices)


def default_selection(experiment_ids: list[str], manifest: Manifest) -> list[int]:
  return [i for i, eid in enumerate(experiment_ids) if manifest.status(eid) != SUCCEEDED]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_remote_experiments_selection.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add remote_experiments/selection.py tests/test_remote_experiments_selection.py
git commit -m "feat: add selection parsing and resume defaults"
```

---

### Task 9: Batch/VM stats and ETA heuristic

**Files:**
- Create: `remote_experiments/stats.py`
- Test: `tests/test_remote_experiments_stats.py`

**Interfaces:**
- Consumes: `Manifest`, `SUCCEEDED` (Task 7); `ray_dispatcher.Inventory`
- Produces: `HostStats(host, slots_total, slots_busy, jobs_completed, current_jobs)`, `BatchStats(total, succeeded, failed, running, pending, elapsed_s, throughput_per_min, eta_s, hosts)`, `estimate_eta(avg_duration_s, remaining, total_slots) -> float`, `summarize(experiment_ids, manifest, inventory, running_hosts, elapsed_s) -> BatchStats`

**Requires:** `Dispatcher.running_hosts()` from the ray-dispatcher phase-9 plan (see Global Constraints).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_remote_experiments_stats.py
import pytest
from ray_dispatcher import Inventory, RemoteHost

from remote_experiments.manifest import Manifest
from remote_experiments.stats import estimate_eta, summarize


def test_estimate_eta_zero_remaining_is_zero():
  assert estimate_eta(avg_duration_s=10.0, remaining=0, total_slots=2) == 0.0


def test_estimate_eta_divides_by_total_slots():
  assert estimate_eta(avg_duration_s=10.0, remaining=4, total_slots=2) == 20.0


def test_estimate_eta_rejects_zero_slots():
  with pytest.raises(ValueError):
    estimate_eta(avg_duration_s=10.0, remaining=1, total_slots=0)


def _inventory():
  return Inventory((
    RemoteHost("vm1", user="ubuntu", slots=2),
    RemoteHost("vm2", user="ubuntu", slots=1),
  ))


def test_summarize_counts_by_status(tmp_path):
  manifest = Manifest(tmp_path / "m.json")
  manifest.record("e1", status="succeeded", host="vm1", duration_s=10.0)
  manifest.record("e2", status="running", host="vm1")
  manifest.record("e3", status="never_run")
  stats = summarize(["e1", "e2", "e3"], manifest, _inventory(), {"e2": "vm1"}, elapsed_s=60.0)
  assert stats.total == 3
  assert stats.succeeded == 1
  assert stats.running == 1
  assert stats.pending == 1
  assert stats.failed == 0


def test_summarize_eta_uses_succeeded_average_duration(tmp_path):
  manifest = Manifest(tmp_path / "m.json")
  manifest.record("e1", status="succeeded", host="vm1", duration_s=10.0)
  manifest.record("e2", status="running", host="vm1")
  stats = summarize(["e1", "e2"], manifest, _inventory(), {"e2": "vm1"}, elapsed_s=60.0)
  assert stats.eta_s == 10.0 / 3  # avg_duration=10, remaining=1, total_slots=3


def test_summarize_eta_none_without_any_succeeded(tmp_path):
  manifest = Manifest(tmp_path / "m.json")
  manifest.record("e1", status="running", host="vm1")
  stats = summarize(["e1"], manifest, _inventory(), {"e1": "vm1"}, elapsed_s=60.0)
  assert stats.eta_s is None


def test_summarize_attributes_current_jobs_per_host(tmp_path):
  manifest = Manifest(tmp_path / "m.json")
  manifest.record("e1", status="running")
  manifest.record("e2", status="running")
  stats = summarize(
    ["e1", "e2"], manifest, _inventory(), {"e1": "vm1", "e2": "vm2"}, elapsed_s=10.0
  )
  vm1 = next(h for h in stats.hosts if h.host == "vm1")
  vm2 = next(h for h in stats.hosts if h.host == "vm2")
  assert vm1.current_jobs == ("e1",)
  assert vm1.slots_busy == 1
  assert vm2.current_jobs == ("e2",)


def test_summarize_throughput_per_minute(tmp_path):
  manifest = Manifest(tmp_path / "m.json")
  manifest.record("e1", status="succeeded", duration_s=5.0)
  stats = summarize(["e1"], manifest, _inventory(), {}, elapsed_s=30.0)
  assert stats.throughput_per_min == 2.0  # 1 succeeded in 30s -> 2/min
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_remote_experiments_stats.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'remote_experiments.stats'`

- [ ] **Step 3: Implement**

```python
# remote_experiments/stats.py
"""Pure computation of batch/VM progress stats for the TUI — no rendering here."""

from __future__ import annotations

from dataclasses import dataclass

from ray_dispatcher import Inventory

from .manifest import Manifest, SUCCEEDED

_RUNNING = "running"
_NEVER_RUN = "never_run"
_PENDING = "pending"


@dataclass(frozen=True)
class HostStats:
  host: str
  slots_total: int
  slots_busy: int
  jobs_completed: int
  current_jobs: tuple[str, ...]


@dataclass(frozen=True)
class BatchStats:
  total: int
  succeeded: int
  failed: int
  running: int
  pending: int
  elapsed_s: float
  throughput_per_min: float
  eta_s: float | None
  hosts: tuple[HostStats, ...]


def estimate_eta(avg_duration_s: float, remaining: int, total_slots: int) -> float:
  if total_slots <= 0:
    raise ValueError("total_slots must be > 0")
  if remaining <= 0:
    return 0.0
  return avg_duration_s * remaining / total_slots


def summarize(
    experiment_ids: list[str],
    manifest: Manifest,
    inventory: Inventory,
    running_hosts: dict[str, str],
    elapsed_s: float,
  ) -> BatchStats:
  statuses = {eid: manifest.status(eid) for eid in experiment_ids}
  succeeded = sum(1 for s in statuses.values() if s == SUCCEEDED)
  running = sum(1 for s in statuses.values() if s == _RUNNING)
  pending = sum(1 for s in statuses.values() if s in (_NEVER_RUN, _PENDING))
  failed = len(statuses) - succeeded - running - pending

  durations = [
    manifest.duration(eid) for eid in experiment_ids
    if statuses[eid] == SUCCEEDED and manifest.duration(eid) is not None
  ]
  avg_duration = sum(durations) / len(durations) if durations else 0.0
  total_slots = sum(h.slots for h in inventory.hosts)
  remaining = running + pending
  eta_s = estimate_eta(avg_duration, remaining, total_slots) if durations else None
  throughput_per_min = (succeeded / elapsed_s * 60) if elapsed_s > 0 else 0.0

  host_stats = []
  for h in inventory.hosts:
    current_jobs = tuple(
      eid for eid, host in running_hosts.items() if host == h.host and eid in statuses
    )
    completed = sum(
      1 for eid in experiment_ids
      if statuses[eid] == SUCCEEDED and manifest.host(eid) == h.host
    )
    host_stats.append(HostStats(
      host=h.host, slots_total=h.slots, slots_busy=len(current_jobs),
      jobs_completed=completed, current_jobs=current_jobs,
    ))

  return BatchStats(
    total=len(statuses), succeeded=succeeded, failed=failed, running=running,
    pending=pending, elapsed_s=elapsed_s, throughput_per_min=throughput_per_min,
    eta_s=eta_s, hosts=tuple(host_stats),
  )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_remote_experiments_stats.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add remote_experiments/stats.py tests/test_remote_experiments_stats.py
git commit -m "feat: add batch/VM progress stats and ETA heuristic"
```

---

### Task 10: Poll loop (submit, status, stop/resume)

**Files:**
- Create: `remote_experiments/runner.py`
- Test: `tests/test_remote_experiments_runner.py`

**Interfaces:**
- Consumes: `Manifest` (Task 7), `ray_dispatcher.{Job, JobHandle, JobStatus}`
- Produces: `run_batch(dispatcher, jobs, manifest, on_tick, *, poll_interval_s=1.0, sleep=time.sleep, now=time.monotonic) -> bool`. `dispatcher` is duck-typed: `.submit(jobs) -> list[JobHandle]`, `.status(handle) -> JobStatus`, `.cancel(handle) -> None`, `.running_hosts() -> dict[str, str]`. `on_tick: Callable[[dict[str, str]], None]` is called once per poll iteration with the current `running_hosts()` snapshot.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_remote_experiments_runner.py
from ray_dispatcher import Job, JobHandle, JobStatus

from remote_experiments.manifest import Manifest
from remote_experiments.runner import run_batch


class FakeDispatcher:
  def __init__(self, status_sequences, hosts=None):
    self._sequences = {k: list(v) for k, v in status_sequences.items()}
    self._hosts = hosts or {}
    self.cancelled: list[str] = []

  def submit(self, jobs):
    return [JobHandle(batch_id="b1", job_id=j.id, token=j.id) for j in jobs]

  def status(self, handle):
    seq = self._sequences[handle.job_id]
    return seq.pop(0) if len(seq) > 1 else seq[0]

  def cancel(self, handle):
    self.cancelled.append(handle.job_id)

  def running_hosts(self):
    return dict(self._hosts)


def _jobs(*ids):
  return [Job(id=i, command=("echo",)) for i in ids]


def test_run_batch_records_succeeded_status_and_host(tmp_path):
  dispatcher = FakeDispatcher(
    {"e1": [JobStatus.RUNNING, JobStatus.SUCCEEDED]}, hosts={"e1": "vm1"}
  )
  manifest = Manifest(tmp_path / "m.json")
  ticks = []
  completed = run_batch(
    dispatcher, _jobs("e1"), manifest, on_tick=ticks.append, sleep=lambda s: None
  )
  assert completed is True
  assert manifest.status("e1") == "succeeded"
  assert manifest.host("e1") == "vm1"
  assert len(ticks) >= 1


def test_run_batch_records_duration_from_submit_to_terminal(tmp_path):
  times = iter([100.0, 105.0])
  dispatcher = FakeDispatcher({"e1": [JobStatus.SUCCEEDED]}, hosts={})
  manifest = Manifest(tmp_path / "m.json")
  completed = run_batch(
    dispatcher, _jobs("e1"), manifest, on_tick=lambda h: None,
    sleep=lambda s: None, now=lambda: next(times),
  )
  assert completed is True
  assert manifest.duration("e1") == 5.0


def test_run_batch_stops_and_cancels_on_keyboard_interrupt(tmp_path):
  dispatcher = FakeDispatcher({"e1": [JobStatus.RUNNING] * 5}, hosts={"e1": "vm1"})
  manifest = Manifest(tmp_path / "m.json")

  def _raising_sleep(_seconds):
    raise KeyboardInterrupt

  completed = run_batch(
    dispatcher, _jobs("e1"), manifest, on_tick=lambda h: None, sleep=_raising_sleep
  )
  assert completed is False
  assert "e1" in dispatcher.cancelled
  assert manifest.status("e1") == "cancelled"


def test_run_batch_preserves_host_after_lease_released_before_terminal_is_observed(tmp_path):
  hosts_per_tick = [{"e1": "vm1"}, {}]

  class _Dispatcher(FakeDispatcher):
    def running_hosts(self):
      return hosts_per_tick.pop(0) if hosts_per_tick else {}

  dispatcher = _Dispatcher({"e1": [JobStatus.RUNNING, JobStatus.SUCCEEDED]})
  manifest = Manifest(tmp_path / "m.json")
  run_batch(dispatcher, _jobs("e1"), manifest, on_tick=lambda h: None, sleep=lambda s: None)
  assert manifest.host("e1") == "vm1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_remote_experiments_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'remote_experiments.runner'`

- [ ] **Step 3: Implement**

```python
# remote_experiments/runner.py
"""Submit/poll loop driving a batch through ray_dispatcher, with stop-on-interrupt."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from ray_dispatcher import Job, JobHandle, JobStatus

from .manifest import Manifest

_TERMINAL = frozenset({
  JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.TIMED_OUT,
})


def run_batch(
    dispatcher,
    jobs: Sequence[Job],
    manifest: Manifest,
    on_tick: Callable[[dict[str, str]], None],
    *,
    poll_interval_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
  ) -> bool:
  """Submit jobs, poll until all terminal or interrupted.

  Returns True if the batch ran to completion, False if stopped (Ctrl-C):
  in-flight jobs are cancelled and the manifest reflects their last known
  state, so a later run_batch() call on the same experiment ids resumes
  from there.
  """
  handles: list[JobHandle] = dispatcher.submit(jobs)
  started_at = {h.job_id: now() for h in handles}
  last_known_host: dict[str, str] = {}
  for handle in handles:
    manifest.record(handle.job_id, status="pending")

  remaining = list(handles)
  try:
    while remaining:
      running_hosts = dispatcher.running_hosts()
      last_known_host.update(running_hosts)
      next_remaining = []
      for handle in remaining:
        status = dispatcher.status(handle)
        host = last_known_host.get(handle.job_id)
        if status in _TERMINAL:
          duration = now() - started_at[handle.job_id]
          manifest.record(handle.job_id, status=status.value, host=host, duration_s=duration)
        else:
          manifest.record(handle.job_id, status=status.value, host=host)
          next_remaining.append(handle)
      remaining = next_remaining
      on_tick(running_hosts)
      if remaining:
        sleep(poll_interval_s)
    return True
  except KeyboardInterrupt:
    for handle in remaining:
      dispatcher.cancel(handle)
      manifest.record(handle.job_id, status=JobStatus.CANCELLED.value)
    return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_remote_experiments_runner.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add remote_experiments/runner.py tests/test_remote_experiments_runner.py
git commit -m "feat: add submit/poll loop with stop-on-interrupt"
```

---

### Task 11: TUI rendering

**Files:**
- Create: `remote_experiments/tui.py`
- Test: `tests/test_remote_experiments_tui.py`

**Interfaces:**
- Consumes: `Batch` (Task 3), `Manifest` (Task 7), `BatchStats`/`summarize` (Task 9), `ray_dispatcher.Inventory`
- Produces: `render(batch, manifest, stats) -> rich.layout.Layout`; `live_view(batch, manifest, inventory, *, start_time) -> contextmanager yielding Callable[[dict[str, str]], None]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_remote_experiments_tui.py
from rich.console import Console
from ray_dispatcher import Inventory, RemoteHost

from remote_experiments.batch import Batch, Experiment
from remote_experiments.manifest import Manifest
from remote_experiments.stats import summarize
from remote_experiments.tui import render


def _batch():
  return Batch(suite="smoke", experiments=(
    Experiment(id="e1", suite="smoke", algorithm="centralized", seed=1,
               graph_params={}, load_params={}, config={}),
  ))


def test_render_includes_experiment_status_and_host(tmp_path):
  manifest = Manifest(tmp_path / "m.json")
  manifest.record("e1", status="running", host="vm1")
  inventory = Inventory((RemoteHost("vm1", user="ubuntu", slots=1),))
  stats = summarize(["e1"], manifest, inventory, {"e1": "vm1"}, elapsed_s=10.0)
  layout = render(_batch(), manifest, stats)

  console = Console(record=True, width=120)
  console.print(layout)
  output = console.export_text()
  assert "e1" in output
  assert "running" in output
  assert "vm1" in output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_remote_experiments_tui.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'remote_experiments.tui'`

- [ ] **Step 3: Implement**

```python
# remote_experiments/tui.py
"""rich-based live progress view: renders BatchStats/Manifest, no business logic."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from ray_dispatcher import Inventory

from .batch import Batch
from .manifest import Manifest
from .stats import BatchStats, summarize


def render(batch: Batch, manifest: Manifest, stats: BatchStats) -> Layout:
  summary = Table.grid(padding=(0, 2))
  summary.add_row(
    f"total: {stats.total}", f"succeeded: {stats.succeeded}", f"failed: {stats.failed}",
    f"running: {stats.running}", f"pending: {stats.pending}",
    f"throughput: {stats.throughput_per_min:.1f}/min",
    f"eta: {_format_seconds(stats.eta_s)}",
  )

  vm_table = Table(title="VMs")
  vm_table.add_column("host")
  vm_table.add_column("slots")
  vm_table.add_column("current jobs")
  vm_table.add_column("completed")
  for h in stats.hosts:
    vm_table.add_row(
      h.host, f"{h.slots_busy}/{h.slots_total}", ", ".join(h.current_jobs) or "-",
      str(h.jobs_completed),
    )

  exp_table = Table(title="Experiments")
  exp_table.add_column("id")
  exp_table.add_column("algorithm")
  exp_table.add_column("seed")
  exp_table.add_column("status")
  exp_table.add_column("host")
  for e in batch.experiments:
    exp_table.add_row(
      e.id, e.algorithm, str(e.seed), manifest.status(e.id), manifest.host(e.id) or "-",
    )

  layout = Layout()
  layout.split_column(
    Layout(Panel(summary, title="Batch"), size=3),
    Layout(vm_table, size=len(stats.hosts) + 4),
    Layout(exp_table),
  )
  return layout


def _format_seconds(seconds: float | None) -> str:
  if seconds is None:
    return "-"
  minutes, secs = divmod(int(seconds), 60)
  return f"{minutes}m{secs:02d}s"


@contextmanager
def live_view(
    batch: Batch, manifest: Manifest, inventory: Inventory, *, start_time: float,
  ) -> Iterator[Callable[[dict[str, str]], None]]:
  console = Console()
  experiment_ids = [e.id for e in batch.experiments]
  with Live(console=console, refresh_per_second=4) as live:
    def on_tick(running_hosts: dict[str, str]) -> None:
      elapsed = time.monotonic() - start_time
      stats = summarize(experiment_ids, manifest, inventory, running_hosts, elapsed)
      live.update(render(batch, manifest, stats))
    yield on_tick
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_remote_experiments_tui.py -v`
Expected: PASS (1 test)

`live_view` itself (the `Live`-driven context manager) is not unit tested here — it needs a real/interactive console session. Task 12's manual verification step covers it end-to-end.

- [ ] **Step 5: Commit**

```bash
git add remote_experiments/tui.py tests/test_remote_experiments_tui.py
git commit -m "feat: add rich TUI rendering for batch/VM/experiment progress"
```

---

### Task 12: CLI (`define` / `run`)

**Files:**
- Create: `remote_experiments/cli.py`
- Create: `remote_experiments/__main__.py`
- Test: `tests/test_remote_experiments_cli.py`

**Interfaces:**
- Consumes: everything from Tasks 3-11
- Produces: `build_parser() -> argparse.ArgumentParser`, `cmd_define(args)`, `cmd_run(args)`, `main(argv=None)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_remote_experiments_cli.py
import json

import pytest

from remote_experiments.batch import Batch
from remote_experiments.cli import build_parser, cmd_define


def test_define_accepts_registered_suite_name():
  parser = build_parser()
  args = parser.parse_args(["define", "smoke", "-o", "/tmp/x.json"])
  assert args.suite == "smoke"


def test_define_rejects_unregistered_suite_name():
  parser = build_parser()
  with pytest.raises(SystemExit):
    parser.parse_args(["define", "does-not-exist", "-o", "/tmp/x.json"])


def test_cmd_define_writes_batch_file(tmp_path):
  out_path = tmp_path / "out" / "batch.json"
  args = build_parser().parse_args(["define", "smoke", "-o", str(out_path)])
  cmd_define(args)
  loaded = Batch.load(out_path)
  assert loaded.suite == "smoke"
  assert len(loaded.experiments) == 20  # smoke suite default: 2 seeds x 10 algorithms


def test_cmd_define_output_is_valid_json(tmp_path):
  out_path = tmp_path / "batch.json"
  cmd_define(build_parser().parse_args(["define", "smoke", "-o", str(out_path)]))
  raw = json.loads(out_path.read_text())
  assert raw["suite"] == "smoke"


def test_run_subcommand_parses_required_arguments():
  parser = build_parser()
  args = parser.parse_args(["run", "batches/foo.json", "--inventory", "inv.yaml"])
  assert args.batch_file == "batches/foo.json"
  assert args.inventory == "inv.yaml"
  assert args.results_dir == "./results"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_remote_experiments_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'remote_experiments.cli'`

- [ ] **Step 3: Implement**

```python
# remote_experiments/cli.py
"""CLI entry point: `define` builds a batch file from a suite, `run` executes/resumes it."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ray_dispatcher import Dispatcher, Inventory, Project, SecretFile

from . import definitions  # imports register all suites as a side effect
from .batch import Batch
from .definitions import get_suite, list_suites
from .jobs import experiment_to_job
from .manifest import Manifest
from .runner import run_batch
from .selection import default_selection, parse_selection
from .tui import live_view


def cmd_define(args: argparse.Namespace) -> None:
  build = get_suite(args.suite)
  experiments = build()
  batch = Batch(suite=args.suite, experiments=tuple(experiments))
  out_path = Path(args.output)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  batch.save(out_path)
  print(f"wrote {len(experiments)} experiments to {out_path}")


def cmd_run(args: argparse.Namespace) -> None:
  batch = Batch.load(args.batch_file)
  manifest_path = Path(args.batch_file).with_suffix(".manifest.json")
  manifest = Manifest(manifest_path)
  experiment_ids = [e.id for e in batch.experiments]

  default_idx = default_selection(experiment_ids, manifest)
  print(f"{len(batch.experiments)} experiments in batch, {len(default_idx)} pending")
  for i, e in enumerate(batch.experiments):
    print(f"  [{i}] {e.id} ({manifest.status(e.id)})")
  raw = input(f"Select to run [default: {len(default_idx)} pending] (indices/ranges/'all'): ")
  selected_idx = parse_selection(raw, len(batch.experiments)) if raw.strip() else default_idx
  selected = [batch.experiments[i] for i in selected_idx]
  if not selected:
    print("nothing selected, exiting")
    return

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
    exclude=(".venv/", ".git/", "solutions/", "results/", "batches/"),
  )
  config_dir = manifest_path.parent / f"{manifest_path.stem}-configs"
  jobs = [experiment_to_job(e, config_dir) for e in selected]

  with Dispatcher(inventory, project, results_dir=args.results_dir) as dispatcher:
    start = time.monotonic()
    with live_view(batch, manifest, inventory, start_time=start) as on_tick:
      completed = run_batch(dispatcher, jobs, manifest, on_tick)
  print("batch complete" if completed else "stopped — rerun the same command to resume")


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(prog="remote_experiments")
  sub = parser.add_subparsers(dest="command", required=True)

  define_p = sub.add_parser("define", help="Build a batch file from a registered suite")
  define_p.add_argument("suite", choices=list_suites())
  define_p.add_argument("-o", "--output", required=True)
  define_p.set_defaults(func=cmd_define)

  run_p = sub.add_parser("run", help="Run (or resume) a batch file through the TUI")
  run_p.add_argument("batch_file")
  run_p.add_argument("--inventory", required=True)
  run_p.add_argument("--project-path", default=".")
  run_p.add_argument("--results-dir", default="./results")
  run_p.add_argument("--gurobi-license", default=None)
  run_p.add_argument("--python-version", default="3.10.19")
  run_p.add_argument("--uv-version", default="0.11.25")
  run_p.set_defaults(func=cmd_run)

  return parser


def main(argv: list[str] | None = None) -> None:
  parser = build_parser()
  args = parser.parse_args(argv)
  args.func(args)
```

```python
# remote_experiments/__main__.py
from .cli import main

if __name__ == "__main__":
  main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_remote_experiments_cli.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Manual verification of `cmd_run`/`main`**

`cmd_run` needs a real `Inventory` (SSH-reachable VMs) and is not covered by an automated test. Verify the wiring manually:

Run: `uv run -m remote_experiments --help`
Expected: prints usage with `define` and `run` subcommands, no traceback.

Run: `uv run -m remote_experiments define smoke -o /tmp/smoke-batch.json && cat /tmp/smoke-batch.json | python3 -m json.tool | head -20`
Expected: a real batch file with 20 experiments, valid JSON.

Run: `uv run -m remote_experiments run --help`
Expected: prints `run` subcommand usage (`--inventory` required, etc.), no traceback — confirms the module imports cleanly (pulls in `tui.py`/`runner.py`/`ray_dispatcher` without error) even though no VMs are configured to actually execute against.

- [ ] **Step 6: Commit**

```bash
git add remote_experiments/cli.py remote_experiments/__main__.py tests/test_remote_experiments_cli.py
git commit -m "feat: add define/run CLI for remote_experiments"
```

---

### Task 13: Inventory example, README, full-suite verification

**Files:**
- Create: `remote_experiments/inventory.yaml.example`
- Create: `remote_experiments/README.md`
- No test (documentation + verification only)

**Interfaces:** None — closing task.

- [ ] **Step 1: Add an example inventory file**

```yaml
# remote_experiments/inventory.yaml.example
hosts:
  - host: 10.0.0.10
    user: ubuntu
    slots: 2
    identity_file: ~/.ssh/id_ed25519
  - host: 10.0.0.11
    user: ubuntu
    slots: 2
    identity_file: ~/.ssh/id_ed25519
```

- [ ] **Step 2: Write the README**

```markdown
# remote_experiments/README.md
# remote_experiments

Define batches of DFaaSOptimizer experiments, then run them on Gurobi-equipped
VMs via `ray-dispatcher` with a live TUI.

## Define a batch

\`\`\`
uv run -m remote_experiments define smoke -o batches/smoke.json
\`\`\`

Builds every `Experiment` for the named suite (a registered function under
`remote_experiments/definitions/`) and writes them to a static JSON file.

## Run (or resume) a batch

\`\`\`
cp remote_experiments/inventory.yaml.example my-inventory.yaml  # edit hosts
uv run -m remote_experiments run batches/smoke.json --inventory my-inventory.yaml \
  --gurobi-license ~/gurobi.lic
\`\`\`

Shows which experiments are pending (everything not yet `succeeded`,
tracked in `batches/smoke.manifest.json`), prompts for a selection
(indices, ranges, or `all` — default is every pending one), then submits
and shows a live progress view.

Ctrl-C cancels in-flight jobs and stops cleanly. Re-running the same `run`
command resumes — it defaults to selecting only what isn't `succeeded` yet.

## Adding a suite

Add a file to `remote_experiments/definitions/` with a function decorated
`@register_suite("name")` returning a `list[Experiment]` (see `smoke.py`).
```

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest tests/ -v -k remote_experiments`
Expected: PASS, all `remote_experiments`-related tests green (batch, registry, smoke suite, jobs, manifest, selection, stats, runner, tui, cli — Tasks 3-12).

Run: `uv run pytest tests/ -v` (full suite, including pre-existing tests)
Expected: PASS — no regressions in the rest of the repo (including `tests/test_run_faasmacro_v0_flag.py` from Task 2).

- [ ] **Step 4: Run ruff**

Run: `uv run ruff check remote_experiments tests/test_remote_experiments_*.py run_faasmacro.py`
Expected: no errors (this repo's ruff `select = ["E9", "F"]` with `F401/F541/F841` ignored — mostly a syntax-error check).

- [ ] **Step 5: Commit**

```bash
git add remote_experiments/inventory.yaml.example remote_experiments/README.md
git commit -m "docs: add remote_experiments usage README and example inventory"
```
