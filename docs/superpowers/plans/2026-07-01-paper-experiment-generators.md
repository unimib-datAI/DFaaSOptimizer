# Paper Experiment Generators Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate all paper experiment matrices as registered `remote_experiments` suites compatible with the existing `Batch` and job workflow.

**Architecture:** Add one `paper.py` suite module with small configuration helpers and eight explicit builders, preserving one batch per experiment family. Add a bounded connected-graph retry for probability-based topologies because E3 requires connected instances.

**Tech Stack:** Python 3.10, pytest, NetworkX, existing `remote_experiments` registry and batch dataclasses.

---

## File map

- Create `remote_experiments/definitions/paper.py`: constants, configuration helpers, and E0–E7 suite builders.
- Modify `remote_experiments/definitions/__init__.py`: import `paper` for registration.
- Modify `generators/generate_data.py`: bounded deterministic resampling for connected probability graphs.
- Create `tests/test_paper_experiment_suites.py`: suite counts, factors, IDs, and batch compatibility.
- Modify `tests/test_generate_data_extended.py`: connected probability-topology regression tests.
- Modify `remote_experiments/README.md`: list commands for defining the paper batches.

### Task 1: Connected probability topologies

**Files:**
- Modify: `tests/test_generate_data_extended.py`
- Modify: `generators/generate_data.py:185-223`

- [ ] **Step 1: Add failing connectivity tests**

```python
def test_probability_neighborhood_is_connected():
  limits = {"neighborhood": {"p": 1.0}}
  matrix, graph = generate_neighborhood(4, limits, np.random.default_rng(1))
  assert nx.is_connected(graph)
  assert matrix.shape == (4, 4)


def test_probability_neighborhood_rejects_impossible_connectivity():
  limits = {"neighborhood": {"p": 0.0}}
  with pytest.raises(ValueError, match="connected random neighborhood"):
    generate_neighborhood(4, limits, np.random.default_rng(1))
```

- [ ] **Step 2: Run RED**

Run: `.venv/bin/pytest -q tests/test_generate_data_extended.py -k probability_neighborhood`

Expected: the impossible-connectivity test fails because the current generator returns a disconnected graph.

- [ ] **Step 3: Run impact analysis**

Run: `npx gitnexus impact -r DFaaSOptimizer --direction upstream generate_neighborhood`

Expected: inspect all direct generation flows and stop for user confirmation only if risk is HIGH or CRITICAL.

- [ ] **Step 4: Implement bounded resampling**

Replace the `p` branch with a 1,000-attempt loop that rebuilds the symmetric
adjacency matrix using the existing RNG and exits when `nx.is_connected(graph)`.
After the loop, raise:

```python
raise ValueError("could not generate a connected random neighborhood in 1000 attempts")
```

- [ ] **Step 5: Run GREEN**

Run: `.venv/bin/pytest -q tests/test_generate_data_extended.py -k probability_neighborhood`

Expected: both tests pass.

### Task 2: Common paper-suite helpers and E0/E1

**Files:**
- Create: `tests/test_paper_experiment_suites.py`
- Create: `remote_experiments/definitions/paper.py`

- [ ] **Step 1: Add failing registration and E0/E1 tests**

```python
from remote_experiments.definitions import get_suite, list_suites
from remote_experiments.definitions.paper import build_e0, build_e1


def test_paper_suites_are_registered():
  expected = {
    "paper-e0-pilot", "paper-e1-quality-runtime", "paper-e2-scalability",
    "paper-e3-topology", "paper-e4-robustness", "paper-e5-dynamics",
    "paper-e6-ablation", "paper-e7-tradeoffs",
  }
  assert expected <= set(list_suites())


def test_e0_default_count():
  assert len(build_e0()) == 120


def test_e1_default_count_and_unique_ids():
  experiments = build_e1()
  assert len(experiments) == 1800
  assert len({e.id for e in experiments}) == 1800
```

- [ ] **Step 2: Run RED**

Run: `.venv/bin/pytest -q tests/test_paper_experiment_suites.py -k 'registered or e0 or e1'`

Expected: collection fails because `paper.py` does not exist.

- [ ] **Step 3: Run impact analysis**

Run: `npx gitnexus impact -r DFaaSOptimizer --direction upstream register_suite`

Expected: registry consumers and CLI are reported.

- [ ] **Step 4: Implement constants and helpers**

Create `paper.py` with fixed `PILOT_SEEDS = tuple(range(1, 11))`,
`CONFIRMATORY_SEEDS = tuple(range(1001, 1031))`, method sets, and these helpers:

```python
def _base_config() -> dict:
  return json.loads(_BASE_CONFIG_PATH.read_text())


def _set_dimensions(config: dict, nodes: int, functions: int) -> None:
  limits = config["limits"]
  limits["Nn"] = {"values": [nodes]}
  limits["Nf"] = {"min": functions, "max": functions}
  limits["demand"]["values"] = [1.0, 1.2] * ((functions + 1) // 2)
  limits["demand"]["values"] = limits["demand"]["values"][:functions]
  limits["memory_requirement"]["values"] = [2, 3] * ((functions + 1) // 2)
  limits["memory_requirement"]["values"] = limits["memory_requirement"]["values"][:functions]


def _experiment(suite: str, cell: str, algorithm: str, seed: int, config: dict) -> Experiment:
  experiment_id = f"{suite}-{cell}-{algorithm}-s{seed}"
  config["seed"] = seed
  config["base_solution_folder"] = f"solutions/{experiment_id}"
  limits = config["limits"]
  return Experiment(
    id=experiment_id, suite=suite, algorithm=algorithm, seed=seed,
    graph_params={k: copy.deepcopy(v) for k, v in limits.items() if k != "load"},
    load_params=copy.deepcopy(limits["load"]), config=config,
  )
```

Implement `build_e0` and `build_e1` as explicit products over their documented
factors, copying the base config for every run.

- [ ] **Step 5: Run GREEN**

Run: `.venv/bin/pytest -q tests/test_paper_experiment_suites.py -k 'registered or e0 or e1'`

Expected: selected tests pass.

### Task 3: E2 scalability and E3 topology

**Files:**
- Modify: `tests/test_paper_experiment_suites.py`
- Modify: `remote_experiments/definitions/paper.py`

- [ ] **Step 1: Add failing E2/E3 tests**

```python
def test_e2_count_and_centralized_size_limit():
  experiments = build_e2()
  assert len(experiments) == 4230
  centralized_sizes = {
    e.config["limits"]["Nn"]["values"][0]
    for e in experiments if e.algorithm == "centralized"
  }
  assert centralized_sizes == {10, 20}


def test_e3_count_and_topology_cells():
  experiments = build_e3()
  assert len(experiments) == 1260
  cells = {e.id.split("-hierarchical-")[0] for e in experiments if e.algorithm == "hierarchical"}
  assert len(cells) == 7 * 30
```

- [ ] **Step 2: Run RED**

Run: `.venv/bin/pytest -q tests/test_paper_experiment_suites.py -k 'e2 or e3'`

Expected: imports or assertions fail because E2/E3 builders are absent.

- [ ] **Step 3: Run impact analysis**

Run: `npx gitnexus impact -r DFaaSOptimizer --direction upstream _experiment`

Expected: only paper builders and tests depend directly on the helper.

- [ ] **Step 4: Implement E2/E3**

Implement the exact factors from the design. E2 emits the nine non-centralized
methods for all `(nodes, functions, seed)` cells and emits centralized only when
`nodes <= 20`. E3 emits seven topology dictionaries: planar `k=3`, regular
`k=3/5/7`, and probability `p=0.1/0.2/0.3`.

- [ ] **Step 5: Run GREEN**

Run: `.venv/bin/pytest -q tests/test_paper_experiment_suites.py -k 'e2 or e3'`

Expected: selected tests pass.

### Task 4: E4 robustness and E5 dynamics

**Files:**
- Modify: `tests/test_paper_experiment_suites.py`
- Modify: `remote_experiments/definitions/paper.py`

- [ ] **Step 1: Add failing E4/E5 tests**

```python
def test_e4_count_and_conditions():
  experiments = build_e4(seeds=(1001,))
  assert len(experiments) == 7 * 6
  assert {e.id.split("-hierarchical-")[0].split("e4-")[-1] for e in experiments if e.algorithm == "hierarchical"} == {
    "baseline", "load-low", "load-high", "memory-scarce", "memory-ample",
    "nodes-homogeneous", "nodes-heterogeneous",
  }


def test_e5_count_trace_coverage_and_steps():
  experiments = build_e5(seeds=(1001,))
  assert len(experiments) == 3 * 6
  assert {e.config["limits"]["load"]["trace_type"] for e in experiments} == {
    "sinusoidal", "clipped", "fixed_sum_minmax",
  }
  assert {e.config["max_steps"] for e in experiments} == {100}
```

- [ ] **Step 2: Run RED**

Run: `.venv/bin/pytest -q tests/test_paper_experiment_suites.py -k 'e4 or e5'`

Expected: builders are missing.

- [ ] **Step 3: Implement E4/E5**

Use seven named mutators for E4 and three trace names for E5. Static conditions
use 50 nodes, four functions, and planar `k=3`. E5 sets `max_steps=100`,
`min_run_time=0`, `max_run_time=100`, and `run_time_step=1`.

- [ ] **Step 4: Run GREEN**

Run: `.venv/bin/pytest -q tests/test_paper_experiment_suites.py -k 'e4 or e5'`

Expected: selected tests pass.

### Task 5: E6 ablations and E7 trade-offs

**Files:**
- Modify: `tests/test_paper_experiment_suites.py`
- Modify: `remote_experiments/definitions/paper.py`

- [ ] **Step 1: Add failing E6/E7 tests**

```python
def test_e6_default_count_and_hierarchical_only():
  experiments = build_e6()
  assert len(experiments) == 1200
  assert {e.algorithm for e in experiments} == {"hierarchical"}


def test_e7_default_count_and_weight_pairs():
  experiments = build_e7()
  assert len(experiments) == 1680
  assert {
    (e.config["solver_options"]["auction"]["latency_weight"],
     e.config["solver_options"]["auction"]["fairness_weight"])
    for e in experiments
  } == {(0, 0), (0.25, 0), (1, 0), (0, 0.25), (0, 1), (0.25, 0.25), (1, 1)}
```

- [ ] **Step 2: Run RED**

Run: `.venv/bin/pytest -q tests/test_paper_experiment_suites.py -k 'e6 or e7'`

Expected: builders are missing.

- [ ] **Step 3: Implement E6/E7**

E6 applies ten named config mutations to the hierarchical method and crosses
them with two sizes, two topologies, and 30 seeds. E7 crosses seven weight pairs,
two topologies, four fixed methods, and 30 seeds.

- [ ] **Step 4: Run GREEN**

Run: `.venv/bin/pytest -q tests/test_paper_experiment_suites.py -k 'e6 or e7'`

Expected: selected tests pass.

### Task 6: Registration, serialization, docs, and full verification

**Files:**
- Modify: `remote_experiments/definitions/__init__.py`
- Modify: `remote_experiments/README.md`
- Modify: `tests/test_paper_experiment_suites.py`

- [ ] **Step 1: Add the reduced batch round-trip test**

```python
def test_paper_experiments_round_trip_as_batch(tmp_path):
  experiments = build_e1(seeds=(1001,), algorithms=("hierarchical",))
  path = tmp_path / "batch.json"
  expected = Batch(suite="paper-e1-quality-runtime", experiments=tuple(experiments))
  expected.save(path)
  assert Batch.load(path) == expected
  assert all(e.config["base_solution_folder"] == f"solutions/{e.id}" for e in experiments)
```

- [ ] **Step 2: Import paper suites and document commands**

Add `from . import paper` beside the smoke import. Add eight `uv run -m
remote_experiments define ...` examples to the README, one output batch per suite.

- [ ] **Step 3: Run focused tests**

Run: `.venv/bin/pytest -q tests/test_paper_experiment_suites.py tests/test_generate_data_extended.py`

Expected: all selected tests pass.

- [ ] **Step 4: Verify actual CLI generation**

Run: `uv run -m remote_experiments define paper-e0-pilot -o /tmp/paper-e0.json`

Run: `.venv/bin/python -c 'from remote_experiments.batch import Batch; assert len(Batch.load("/tmp/paper-e0.json").experiments) == 120'`

Expected: both commands exit 0.

- [ ] **Step 5: Run the complete regression suite**

Run: `MPLCONFIGDIR=/tmp/matplotlib .venv/bin/pytest -q tests/test_remote_experiments_*.py tests/test_generate_data_extended.py tests/test_paper_experiment_suites.py`

Expected: all tests pass.

- [ ] **Step 6: Run GitNexus scope verification**

Run `gitnexus_detect_changes(scope="all")`. Confirm changed symbols are limited
to neighborhood generation, suite registration/builders, tests, and README.

- [ ] **Step 7: Check diff hygiene**

Run: `git diff --check && git status --short`

Expected: no whitespace errors; preserve the untracked
`test_instances/2026-06-30_13-54-13.841040/` directory.
