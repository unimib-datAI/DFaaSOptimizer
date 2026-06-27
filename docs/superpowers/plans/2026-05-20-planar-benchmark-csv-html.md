# Planar Benchmark CSV+HTML Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone benchmark script that runs centralized, distributed, and hierarchical models on planar degree-3 graphs for 20, 40, and 50 nodes across 5 seeds, then writes raw CSV results plus an HTML report with mean and standard deviation for objective and runtime.

**Architecture:** Keep the benchmark isolated from the existing runners. The new script will build one config per `(Nn, seed)`, invoke the three existing run functions directly, collect `objective`, `wallclock_s`, and `termination` into a flat results table, then aggregate by `Nn` and model for a summary CSV and HTML report. No runner internals change; the script only composes existing entry points and the already-fixed planar degree-3 graph generation path.

**Tech Stack:** Python 3.10, pandas, numpy, networkx, existing `run_centralized_model.run`, `run_faasmacro.run`, `hierarchical_auction.runner.run`, pytest, ruff, mypy, standard-library HTML generation.

---

## File Structure

Create:

- `benchmark_planar_3reg.py`: CLI entry point, per-run execution, result extraction, CSV/HTML writing.
- `tests/test_benchmark_planar_3reg.py`: unit tests for config building, aggregation, HTML rendering, and the orchestration helper.

Do not modify the existing runners unless a test exposes a real bug in shared code. The benchmark should call the already-present run functions and use their generated artifacts.

---

## Task 1: Define the Benchmark Data Model and Summary Helpers

**Files:**
- Create: `benchmark_planar_3reg.py`
- Create: `tests/test_benchmark_planar_3reg.py`

- [ ] **Step 1: Write failing tests for summary aggregation and HTML rendering**

Add tests that exercise pure helpers only, using synthetic data:

```python
import pandas as pd

from benchmark_planar_3reg import aggregate_results, render_html_report


def test_aggregate_results_computes_mean_and_std():
  raw = pd.DataFrame([
    {"Nn": 20, "seed": 0, "model": "centralized", "objective": 10.0, "wallclock_s": 1.0},
    {"Nn": 20, "seed": 1, "model": "centralized", "objective": 14.0, "wallclock_s": 3.0},
    {"Nn": 20, "seed": 0, "model": "hierarchical", "objective": 8.0, "wallclock_s": 2.0},
    {"Nn": 20, "seed": 1, "model": "hierarchical", "objective": 12.0, "wallclock_s": 4.0},
  ])

  summary = aggregate_results(raw)
  row = summary[(summary["Nn"] == 20) & (summary["model"] == "centralized")].iloc[0]

  assert row["objective_mean"] == 12.0
  assert row["objective_std"] == pytest.approx(2.8284271247)
  assert row["wallclock_mean_s"] == 2.0
  assert row["wallclock_std_s"] == pytest.approx(1.4142135623)
  assert row["runs"] == 2


def test_render_html_report_contains_summary_and_raw_tables(tmp_path):
  raw = pd.DataFrame([
    {"Nn": 20, "seed": 0, "model": "centralized", "objective": 10.0, "wallclock_s": 1.0},
  ])
  summary = pd.DataFrame([
    {"Nn": 20, "model": "centralized", "objective_mean": 10.0, "objective_std": 0.0,
     "wallclock_mean_s": 1.0, "wallclock_std_s": 0.0, "runs": 1},
  ])

  html_path = tmp_path / "report.html"
  render_html_report(summary, raw, html_path, meta={"sizes": [20, 40, 50], "seeds": [0, 1, 2, 3, 4]})

  text = html_path.read_text()
  assert "<html" in text.lower()
  assert "Summary" in text
  assert "Raw Results" in text
  assert "centralized" in text
```

- [ ] **Step 2: Run the tests to verify they fail for missing helpers**

Run: `uv run pytest tests/test_benchmark_planar_3reg.py -q`

Expected: fail because `benchmark_planar_3reg.py` does not exist yet.

- [ ] **Step 3: Implement the pure helpers**

In `benchmark_planar_3reg.py`, add:

```python
def aggregate_results(raw: pd.DataFrame) -> pd.DataFrame:
  grouped = raw.groupby(["Nn", "model"], as_index=False).agg(
    objective_mean=("objective", "mean"),
    objective_std=("objective", lambda s: s.std(ddof=1)),
    wallclock_mean_s=("wallclock_s", "mean"),
    wallclock_std_s=("wallclock_s", lambda s: s.std(ddof=1)),
    runs=("seed", "count"),
  )
  return grouped.sort_values(["Nn", "model"]).reset_index(drop=True)


def render_html_report(
  summary: pd.DataFrame,
  raw: pd.DataFrame,
  html_path: Path,
  meta: dict[str, object],
) -> None:
  # Write one HTML file with:
  # - title and benchmark metadata
  # - summary table
  # - raw results table
  # - minimal inline CSS for readability
```

Use `pandas.DataFrame.to_html(index=False)` and keep the HTML generation dependency-free.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_benchmark_planar_3reg.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmark_planar_3reg.py tests/test_benchmark_planar_3reg.py
git commit -m "feat: add planar benchmark summary helpers"
```

---

## Task 2: Build the Benchmark Orchestrator

**Files:**
- Modify: `benchmark_planar_3reg.py`
- Modify: `tests/test_benchmark_planar_3reg.py`

The script should run the same benchmark for `Nn = 20, 40, 50` and five seeds by default. Use a fixed two-function workload so the comparison remains meaningful across graph sizes. The three model runners should be executed sequentially for each `(Nn, seed)` pair, with `parallelism=0` and `disable_plotting=True` for reproducibility.

- [ ] **Step 1: Write failing tests for config building and orchestration**

Add tests that monkeypatch the three runner functions and the metric reader so no solver is needed:

```python
from pathlib import Path

import pandas as pd

from benchmark_planar_3reg import build_base_config, run_benchmark


def test_build_base_config_uses_requested_sizes_and_seed(tmp_path):
  config = build_base_config(tmp_path, nn=20, seed=3)
  assert config["limits"]["Nn"]["min"] == 20
  assert config["limits"]["Nn"]["max"] == 20
  assert config["seed"] == 3
  assert config["solver_name"] == "gurobi"
  assert config["limits"]["neighborhood"] == {"type": "planar", "degree": 3}


def test_run_benchmark_collects_three_models(monkeypatch, tmp_path):
  def fake_run(_config, **_kwargs):
    folder = tmp_path / "fake"
    folder.mkdir(exist_ok=True)
    pd.DataFrame({"LoadManagementModel": [11.0]}).to_csv(folder / "obj.csv", index=False)
    pd.DataFrame({"runtime": [1.0]}).to_csv(folder / "runtime.csv", index=False)
    pd.DataFrame({"0": ["optimal"]}).to_csv(folder / "termination_condition.csv", index=False)
    return str(folder)

  monkeypatch.setattr("benchmark_planar_3reg.run_centralized", fake_run)
  monkeypatch.setattr("benchmark_planar_3reg.run_distributed", fake_run)
  monkeypatch.setattr("benchmark_planar_3reg.run_hierarchical", fake_run)

  raw = run_benchmark(output_root=tmp_path, sizes=[20], seeds=[0], solver_name="gurobi")
  assert set(raw["model"]) == {"centralized", "distributed", "hierarchical"}
  assert len(raw) == 3
```

- [ ] **Step 2: Run the tests to verify they fail against the empty script**

Run: `uv run pytest tests/test_benchmark_planar_3reg.py -q`

Expected: fail because `build_base_config` and `run_benchmark` are not implemented yet.

- [ ] **Step 3: Implement the orchestrator**

In `benchmark_planar_3reg.py`, add:

```python
def build_base_config(output_root: Path, nn: int, seed: int) -> dict:
  return {
    "base_solution_folder": str(output_root / f"n{nn}" / f"seed{seed}"),
    "seed": seed,
    "limits": {
      "Nn": {"min": nn, "max": nn},
      "Nf": {"min": 2, "max": 2},
      "neighborhood": {"type": "planar", "degree": 3},
      "weights": {
        "alpha": {"min": 1.0, "max": 1.5},
        "beta_multiplier": {"min": 1.5, "max": 2.5},
        "gamma": {"min": 0.05, "max": 0.15},
        "delta_multiplier": {"min": 0.1, "max": 0.2},
      },
      "demand": {"values": [1.0, 1.2]},
      "memory_capacity": {"values": [12] * nn},
      "memory_requirement": {"values": [2, 3]},
      "max_utilization": {"min": 0.65, "max": 0.75},
      "load": {"trace_type": "fixed_sum", "values": [2.0, 3.0]},
    },
    "solver_name": "gurobi",
    "solver_options": {
      "general": {"TimeLimit": 60, "OutputFlag": 0},
      "auction": {
        "epsilon": 0.01,
        "eta": [0.5, 0.3, 0.15],
        "zeta": 0.1,
        "latency_weight": 0.0,
        "fairness_weight": 0.0,
      },
    },
    "max_iterations": 2,
    "patience": 1,
    "sw_patience": 1,
    "max_steps": 8,
    "min_run_time": 1,
    "max_run_time": 1,
    "run_time_step": 1,
    "checkpoint_interval": 1,
    "max_hierarchy_depth": 3,
    "tolerance": 1e-6,
    "verbose": 0,
  }


def run_benchmark(
  output_root: Path,
  sizes: list[int],
  seeds: list[int],
  solver_name: str = "gurobi",
) -> pd.DataFrame:
  # For each (nn, seed):
  # 1. Build config
  # 2. Call centralized / distributed / hierarchical runners
  # 3. Measure wallclock with time.perf_counter()
  # 4. Read objective from the run folder
  # 5. Append one row per model to a raw results DataFrame
```

Use explicit runner aliases at the top of the file:

```python
from run_centralized_model import run as run_centralized
from run_faasmacro import run as run_distributed
from hierarchical_auction.runner import run as run_hierarchical
```

Store per-run outputs under `solutions/planar_benchmark/<timestamp>/n{nn}/seed{seed}/{model}/` and write `raw_results.csv` after all runs complete.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_benchmark_planar_3reg.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmark_planar_3reg.py tests/test_benchmark_planar_3reg.py
git commit -m "feat: orchestrate planar benchmark runs"
```

---

## Task 3: Write the CSV and HTML Report Artifacts

**Files:**
- Modify: `benchmark_planar_3reg.py`
- Modify: `tests/test_benchmark_planar_3reg.py`

The final report should be easy to read and easy to reuse. Write:

- `raw_results.csv`: one row per `(Nn, seed, model)`
- `summary.csv`: grouped by `(Nn, model)` with mean and standard deviation
- `summary.html`: a single HTML document containing metadata, the summary table, and the raw table

- [ ] **Step 1: Write failing tests for artifact writing**

Add a test that creates a small raw results frame, calls the report writer, and asserts the files exist and contain the expected headings:

```python
def test_report_writer_creates_csv_and_html(tmp_path):
  raw = pd.DataFrame([
    {"Nn": 20, "seed": 0, "model": "centralized", "objective": 10.0, "wallclock_s": 1.0},
    {"Nn": 20, "seed": 1, "model": "centralized", "objective": 14.0, "wallclock_s": 3.0},
  ])
  summary = aggregate_results(raw)

  write_report(raw, summary, tmp_path, meta={"sizes": [20], "seeds": [0, 1]})

  assert (tmp_path / "raw_results.csv").exists()
  assert (tmp_path / "summary.csv").exists()
  html = (tmp_path / "summary.html").read_text()
  assert "Planar Benchmark" in html
  assert "Objective mean" in html
  assert "Wallclock mean" in html
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_benchmark_planar_3reg.py::test_report_writer_creates_csv_and_html -v`

Expected: FAIL before the implementation exists.

- [ ] **Step 3: Implement the report writer**

In `benchmark_planar_3reg.py`, add:

```python
def write_report(
  raw: pd.DataFrame,
  summary: pd.DataFrame,
  output_root: Path,
  meta: dict[str, object],
) -> None:
  raw.to_csv(output_root / "raw_results.csv", index=False)
  summary.to_csv(output_root / "summary.csv", index=False)
  render_html_report(summary, raw, output_root / "summary.html", meta)
```

The HTML should use a compact inline stylesheet, a top metadata block, then the summary table, then the raw table. Keep the model order stable as `centralized`, `distributed`, `hierarchical`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_benchmark_planar_3reg.py -q`

Expected: PASS.

- [ ] **Step 5: Manual benchmark smoke**

Run:

```bash
uv run python benchmark_planar_3reg.py \
  --sizes 20 40 50 \
  --seeds 0 1 2 3 4 \
  --solver-name gurobi \
  --output-root solutions/planar_benchmark
```

Expected:
- one benchmark folder with `raw_results.csv`, `summary.csv`, and `summary.html`
- 45 raw rows total
- summary rows for each `(Nn, model)` pair
- no Pyomo objective replacement warnings

---

## Self-Review Checklist

- The script is standalone and does not disturb the existing run pipeline.
- The benchmark uses the requested node sizes and exactly 5 seeds by default.
- The report exposes both mean and standard deviation for objective and runtime.
- The HTML report is generated without adding new dependencies.
- The benchmark remains reproducible and can be extended later to more sizes/seeds.
