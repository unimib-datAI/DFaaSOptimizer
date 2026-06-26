# FaaS-MADiG Distributed-Heuristic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add FaaS-MADiG, a distributed greedy-diffusion heuristic that ablates the price/bidding signal of the production FaaS-MADeA auction, as a fourth comparable method.

**Architecture:** A new standalone runner `decentralized_diffusion.py` mirrors `run_faasmadea.py`. It reuses the local MILP planning (`LSP`/`LSPr`) and every shared helper unchanged (imported from `run_faasmadea`), and replaces only the two market functions with price-free equivalents: `define_assignments` (greedy by utility, no bid price) and `evaluate_assignments` (greedy capacity fill by utility, no price update, score-based replacement of lower-utility incumbents). Wiring into `run.py` and `compare_results.py` is purely additive.

**Tech Stack:** Python 3.10, NumPy, pandas, Pyomo (Gurobi/GLPK), pytest. 2-space indentation throughout (project style).

## Global Constraints

- **DO NOT modify functions that support other methods.** Files `run_faasmadea.py`, `run_faasmacro.py`, `run_centralized_model.py`, `models/`, `utils/`, `hierarchical_auction/`, `postprocessing.py`, `logs_postprocessing.py` are **read-only**. Reuse their functions by `import`.
- **New behavior → new function.** Where FaaS-MADiG needs slightly different behavior than the auction, write a NEW function in `decentralized_diffusion.py` (`define_assignments`, `evaluate_assignments`, `run`). Never edit `define_bids`/`evaluate_bids`/`run` in `run_faasmadea.py`.
- **`run.py` and `compare_results.py` changes must be ADDITIVE only** — new `--methods` choice, new dispatch block, new dict key, new ternary branch. Existing methods' code paths must remain byte-for-byte unchanged. The wiring test (Task 4/5) doubles as a regression guard for the existing methods' labels.
- **Reuse, don't duplicate.** Import `compute_residual_capacity`, `check_stopping_criteria`, `neigh_dict_to_matrix`, `start_additional_replicas`, `ensure_memory_sellers`, `check_ls_pr_feasibility_from_fixed_y`, `VAR_TYPE` from `run_faasmadea`.
- **Ablation rule:** no prices anywhere — no `epsilon`/`eta`/`zeta`/`u0`/`p`, no `min_b` tracking, no price update. Keep the price-free `tentatively_start_replicas` branch and use current utility scores, not prices, for any `last_y` replacement.
- **Method name:** `FaaS-MADiG`. CLI key: `faas-diffuse`. `obj.csv` column: `FaaS-MADiG`. Artifact family (`mkey`): `LSPc`.
- **Out of scope (v1):** `--fix_r` / `opt_solution` support in the new runner (use plain `LSP()`), and Options B/C from the spec.
- Spec: `docs/superpowers/specs/2026-06-25-faas-madig-distributed-heuristic-design.md`.

---

## File Structure

- **Create `decentralized_diffusion.py`** (top level) — `parse_arguments`, `define_assignments`, `evaluate_assignments`, `run`, `__main__`.
- **Create `tests/test_diffusion_helpers.py`** — unit tests for the two helpers (no solver).
- **Create `tests/test_diffusion_e2e.py`** — solver-gated end-to-end smoke test.
- **Create `tests/test_diffusion_wiring.py`** — `run.py` CLI/wiring tests (no solver).
- **Modify `run.py`** — additive wiring (import, choices, `solution_folders`, `run_d`, dispatch, `mkey`/`mname`, colors).
- **Modify `config_files/planar_comparison.json`** — add `solver_options.diffusion`.
- **Modify `compare_results.py`** — add `FaaS-MADiG` to the box/violin `colors` dict.

---

## Task 1: `define_assignments` (price-free buyer side)

**Files:**
- Create: `decentralized_diffusion.py`
- Test: `tests/test_diffusion_helpers.py`

**Interfaces:**
- Consumes: `run_faasmadea.VAR_TYPE` (int/float scalar type).
- Produces:
  `define_assignments(omega, blackboard, data, neighborhood, rho, diffusion_options, latency, fairness, force_memory_bids) -> (pd.DataFrame, pd.DataFrame, int)`.
  The first DataFrame has columns `["i","j","f","d","utility"]` (no `b`); the second has `["i","j","f"]`; the int is the number of potential buyers.

- [ ] **Step 1: Write the failing test**

Create `tests/test_diffusion_helpers.py`:

```python
import numpy as np
import pandas as pd
import pytest

from decentralized_diffusion import define_assignments, evaluate_assignments


def _base_data(Nn=3, Nf=1):
  data = {None: {
    "Nn": {None: Nn},
    "Nf": {None: Nf},
    "beta": {},
    "gamma": {},
    "demand": {},
    "memory_requirement": {f + 1: 2 for f in range(Nf)},
    "max_utilization": {f + 1: 0.8 for f in range(Nf)},
  }}
  for i in range(Nn):
    for f in range(Nf):
      data[None]["gamma"][(i + 1, f + 1)] = 0.05
      data[None]["demand"][(i + 1, f + 1)] = 1.0
      for j in range(Nn):
        data[None]["beta"][(i + 1, j + 1, f + 1)] = 1.0
  return data


def _ring_neighborhood(Nn=3):
  neighborhood = np.zeros((Nn, Nn))
  for n in range(Nn):
    neighborhood[n, (n + 1) % Nn] = 1
    neighborhood[n, (n - 1) % Nn] = 1
  return neighborhood


def test_define_assignments_greedy_by_utility_no_price_column():
  data = _base_data(Nn=3, Nf=1)
  # buyer 0 prefers seller 1 (beta 2.0) over seller 2 (beta 1.0)
  data[None]["beta"][(1, 2, 1)] = 2.0
  data[None]["beta"][(1, 3, 1)] = 1.0
  omega = np.zeros((3, 1)); omega[0, 0] = 2.0
  blackboard = np.zeros((3, 1)); blackboard[1, 0] = 5.0; blackboard[2, 0] = 5.0
  rho = np.zeros((3,))
  options = {"latency_weight": 0.0, "fairness_weight": 0.0, "unit_bids": False}
  latency = np.zeros((3, 3))
  fairness = np.zeros((3, 1))

  bids, memory_bids, n_buyers = define_assignments(
    omega, blackboard, data, _ring_neighborhood(3), rho,
    options, latency, fairness, force_memory_bids=False,
  )

  assert n_buyers == 1
  assert "b" not in bids.columns
  assert list(bids[["i", "j", "f"]].iloc[0]) == [0, 1, 0]
  assert bids.iloc[0]["d"] == 2.0
  assert bids.iloc[0]["utility"] == 2.0
  assert len(memory_bids) == 0


def test_define_assignments_requests_replicas_when_no_capacity_seller():
  data = _base_data(Nn=2, Nf=1)
  omega = np.zeros((2, 1)); omega[0, 0] = 2.0
  blackboard = np.zeros((2, 1))           # no capacity sellers
  rho = np.zeros((2,)); rho[1] = 4.0       # neighbor 1 has spare memory
  options = {"latency_weight": 0.0, "fairness_weight": 0.0, "unit_bids": False}
  neighborhood = np.array([[0.0, 1.0], [1.0, 0.0]])

  bids, memory_bids, _ = define_assignments(
    omega, blackboard, data, neighborhood, rho,
    options, np.zeros((2, 2)), np.zeros((2, 1)), force_memory_bids=False,
  )

  assert len(bids) == 0
  assert list(memory_bids[["i", "j", "f"]].iloc[0]) == [0, 1, 0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_diffusion_helpers.py::test_define_assignments_greedy_by_utility_no_price_column -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'decentralized_diffusion'`.

- [ ] **Step 3: Write minimal implementation**

Create `decentralized_diffusion.py` with the imports and `define_assignments`:

```python
from run_centralized_model import (
  get_current_load,
  init_complete_solution,
  init_problem,
  join_complete_solution,
  plot_history,
  save_checkpoint,
  save_solution,
  update_data,
)
from run_faasmacro import (
  combine_solutions,
  compute_social_welfare,
  decode_solutions,
  solve_subproblem,
)
from run_faasmadea import (
  VAR_TYPE,
  check_ls_pr_feasibility_from_fixed_y,
  check_stopping_criteria,
  compute_residual_capacity,
  ensure_memory_sellers,
  neigh_dict_to_matrix,
  start_additional_replicas,
)
from utils.centralized import check_feasibility
from utils.faasmacro import compute_centralized_objective
from utils.common import load_configuration
from models.sp import LSP, LSPr

from networkx import adjacency_matrix
from collections import deque
from datetime import datetime
from copy import deepcopy
from typing import Tuple
import pandas as pd
import numpy as np
import argparse
import json
import sys
import os


def define_assignments(
    omega: np.array,
    blackboard: np.array,
    data: dict,
    neighborhood: np.array,
    rho: np.array,
    diffusion_options: dict,
    latency: np.array,
    fairness: np.array,
    force_memory_bids: bool,
  ) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
  """Price-free counterpart of run_faasmadea.define_bids.

  Greedy-by-utility assignment of residual load to neighbours with spare
  capacity. No bid price is computed (no epsilon/delta); the per-pair score is
  stored in the ``utility`` column, on which evaluate_assignments later sorts.
  """
  potential_buyers, functions_to_share = np.nonzero(omega)
  bids = {"i": [], "j": [], "f": [], "d": [], "utility": []}
  memory_bids = {"i": [], "j": [], "f": []}
  for i, f in zip(potential_buyers, functions_to_share):
    potential_sellers = set(np.nonzero(neighborhood[i, :])[0])
    potential_capacity_sellers = potential_sellers.intersection(
      set(np.where(blackboard[:, f] >= 1)[0])
    )
    potential_memory_sellers = potential_sellers.intersection(
      set(np.nonzero(rho)[0])
    )
    utility = []
    candidate_sellers = []
    for j in potential_capacity_sellers:
      ut = (
        data[None]["beta"][(i + 1, j + 1, f + 1)]
        - diffusion_options["latency_weight"] * latency[i, j]
        - diffusion_options["fairness_weight"] * fairness[i, f]
      )
      if ut > - data[None]["gamma"][(i + 1, f + 1)]:
        utility.append(ut)
        candidate_sellers.append(j)
    assigned = 0
    if len(utility) > 0:
      utility = np.array(utility)
      sellers_order = np.argsort(utility)[::-1]
      idx = 0
      while idx < len(sellers_order) and assigned < omega[i, f]:
        j = candidate_sellers[sellers_order[idx]]
        if diffusion_options.get("unit_bids", False):
          d = 1
          while (d < int(min(blackboard[j, f], omega[i, f])) + 1) and (
              assigned < omega[i, f]
            ):
            bids["i"].append(i)
            bids["f"].append(f)
            bids["j"].append(j)
            bids["d"].append(1)
            bids["utility"].append(utility[sellers_order[idx]])
            assigned += 1
            d += 1
        else:
          d = VAR_TYPE(min(blackboard[j, f], (omega[i, f] - assigned)))
          bids["i"].append(i)
          bids["f"].append(f)
          bids["j"].append(j)
          bids["d"].append(d)
          bids["utility"].append(utility[sellers_order[idx]])
          assigned += d
        idx += 1
      if assigned < omega[i, f]:
        for idx in sellers_order:
          j = candidate_sellers[idx]
          if j in potential_memory_sellers:
            memory_bids["i"].append(i)
            memory_bids["j"].append(j)
            memory_bids["f"].append(f)
    if assigned < omega[i, f] or force_memory_bids:
      for j in potential_memory_sellers - potential_capacity_sellers:
        memory_bids["i"].append(i)
        memory_bids["j"].append(j)
        memory_bids["f"].append(f)
  return pd.DataFrame(bids), pd.DataFrame(memory_bids), len(potential_buyers)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_diffusion_helpers.py -k define_assignments -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add decentralized_diffusion.py tests/test_diffusion_helpers.py
git commit -m "feat(madig): add price-free define_assignments helper"
```

---

## Task 2: `evaluate_assignments` (price-free seller side)

**Files:**
- Modify: `decentralized_diffusion.py`
- Test: `tests/test_diffusion_helpers.py`

**Interfaces:**
- Consumes: `run_faasmadea.ensure_memory_sellers`; the `bids` DataFrame from Task 1.
- Produces:
  `evaluate_assignments(bids, residual_capacity, data, ell, r, initial_rho, tentatively_start_replicas) -> (np.array, np.array, int)`
  returning `(y, additional_replicas, n_sellers)` — **no price array**.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_diffusion_helpers.py`:

```python
def test_evaluate_assignments_respects_capacity_and_tiebreaks_by_buyer():
  data = _base_data(Nn=3, Nf=1)
  # two buyers tie on utility for seller 1; capacity only fits one fully
  bids = pd.DataFrame({
    "i": [2, 0],
    "j": [1, 1],
    "f": [0, 0],
    "d": [2.0, 2.0],
    "utility": [5.0, 5.0],
  })
  residual_capacity = np.zeros((3, 1)); residual_capacity[1, 0] = 3.0
  ell = np.zeros((3, 1))
  r = np.zeros((3, 1))
  rho = np.zeros((3,))

  y, additional_replicas, n_sellers = evaluate_assignments(
    bids, residual_capacity, data, ell, r, rho,
    tentatively_start_replicas=False,
  )

  assert y[:, 1, 0].sum() == 3.0            # never exceeds capacity
  assert y[0, 1, 0] == 2.0                  # lower buyer index served first
  assert y[2, 1, 0] == 1.0
  assert (additional_replicas == 0).all()
  assert n_sellers == 1


def test_evaluate_assignments_is_deterministic_and_returns_no_price():
  data = _base_data(Nn=3, Nf=1)
  bids = pd.DataFrame({
    "i": [2, 0], "j": [1, 1], "f": [0, 0],
    "d": [2.0, 2.0], "utility": [5.0, 5.0],
  })
  residual_capacity = np.zeros((3, 1)); residual_capacity[1, 0] = 3.0
  args = (bids, residual_capacity, data, np.zeros((3, 1)),
          np.zeros((3, 1)), np.zeros((3,)))

  first = evaluate_assignments(*args, tentatively_start_replicas=False)
  second = evaluate_assignments(*args, tentatively_start_replicas=False)

  assert len(first) == 3                    # (y, additional_replicas, n_sellers)
  assert np.array_equal(first[0], second[0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_diffusion_helpers.py -k evaluate_assignments -v`
Expected: FAIL — `ImportError: cannot import name 'evaluate_assignments'`.

- [ ] **Step 3: Write minimal implementation**

Add to `decentralized_diffusion.py` (after `define_assignments`):

```python
def evaluate_assignments(
    bids: pd.DataFrame,
    residual_capacity: np.array,
    data: dict,
    ell: np.array,
    r: np.array,
    initial_rho: np.array,
    tentatively_start_replicas: bool,
  ) -> Tuple[np.array, np.array, int]:
  """Price-free counterpart of run_faasmadea.evaluate_bids.

  Pure greedy capacity fill: sellers serve buyers by descending ``utility``
  (tie-break on buyer index ``i`` for reproducibility). No min_b tracking, no
  price update; score-based `last_y` replacement may re-award lower-utility
  incumbent load. The price-free
  tentatively_start_replicas branch is kept for parity with the baseline.
  """
  Nn = data[None]["Nn"][None]
  Nf = data[None]["Nf"][None]
  potential_sellers, functions_to_share = np.nonzero(residual_capacity)
  if tentatively_start_replicas:
    potential_sellers, functions_to_share = ensure_memory_sellers(
      potential_sellers, functions_to_share, np.nonzero(initial_rho)[0], Nf
    )
  y = np.zeros((Nn, Nn, Nf))
  additional_replicas = np.zeros((Nn, Nf))
  rho = deepcopy(initial_rho)
  for j, f in zip(potential_sellers, functions_to_share):
    j = int(j)
    f = int(f)
    bids_for_j = bids[(bids["j"] == j) & (bids["f"] == f)].sort_values(
      by=["utility", "i"], ascending=[False, True]
    )
    remaining_capacity = int(residual_capacity[j, f])
    next_bid_idx = 0
    while next_bid_idx < len(bids_for_j) and remaining_capacity > 0:
      q = min(remaining_capacity, bids_for_j.iloc[next_bid_idx]["d"])
      y[int(bids_for_j.iloc[next_bid_idx]["i"]), j, f] += q
      remaining_capacity -= q
      next_bid_idx += 1
    # price-free tentative replica start (decides on utilization, not price)
    if (
        remaining_capacity == 0
        and (next_bid_idx > 0 or len(bids_for_j) > 0)
        and tentatively_start_replicas
      ):
      max_a = int(rho[j] / data[None]["memory_requirement"][f + 1])
      if max_a > 0:
        a = 1
        while next_bid_idx < len(bids_for_j) and a <= max_a:
          q = bids_for_j.iloc[next_bid_idx]["d"]
          u = data[None]["demand"][(j + 1, f + 1)] * (
            ell[j, f] + y[:, j, f].sum() + q
          ) / (r[j, f] + a)
          if u <= data[None]["max_utilization"][f + 1]:
            y[int(bids_for_j.iloc[next_bid_idx]["i"]), j, f] += q
            next_bid_idx += 1
            additional_replicas[j, f] = a
            rho[j] -= (a * data[None]["memory_requirement"][f + 1])
          else:
            a += 1
  return y, additional_replicas, len(potential_sellers)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_diffusion_helpers.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add decentralized_diffusion.py tests/test_diffusion_helpers.py
git commit -m "feat(madig): add price-free evaluate_assignments helper"
```

---

## Task 3: `run()` runner + `parse_arguments` + e2e smoke

**Files:**
- Modify: `decentralized_diffusion.py`
- Test: `tests/test_diffusion_e2e.py`

**Interfaces:**
- Consumes: `define_assignments`, `evaluate_assignments` (Tasks 1–2) and the reused `run_faasmadea`/`run_faasmacro`/`run_centralized_model` helpers.
- Produces: `run(config, parallelism, log_on_file=False, disable_plotting=False) -> str` (solution folder path). Saves `LSP*`/`LSPc*` solutions, `obj.csv` (column `FaaS-MADiG`), `termination_condition.csv`, and `runtime.csv` (column `tot`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_diffusion_e2e.py`:

```python
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest

from decentralized_diffusion import run as run_diffusion


def _require_gurobi() -> None:
  solver = pyo.SolverFactory("gurobi")
  if not solver.available(exception_flag=False):
    pytest.skip("Gurobi solver is not available")


def _diffusion_e2e_config(base_solution_folder: Path) -> dict:
  return {
    "base_solution_folder": str(base_solution_folder),
    "seed": 21,
    "limits": {
      "Nn": {"min": 10, "max": 10},
      "Nf": {"min": 1, "max": 1},
      "neighborhood": {"type": "planar", "degree": 3},
      "weights": {
        "alpha": {"min": 1.0, "max": 1.0},
        "beta_multiplier": {"min": 1.5, "max": 2.0},
        "gamma": {"min": 0.05, "max": 0.1},
        "delta_multiplier": {"min": 0.1, "max": 0.2},
      },
      "demand": {"values": [1.0]},
      "memory_capacity": {"values": [12] * 10},
      "memory_requirement": {"values": [2]},
      "max_utilization": {"min": 0.7, "max": 0.7},
      "load": {
        "trace_type": "clipped",
        "min": {"min": 2.0, "max": 2.0},
        "max": {"min": 3.0, "max": 3.0},
      },
    },
    "solver_name": "gurobi",
    "solver_options": {
      "general": {"TimeLimit": 60, "OutputFlag": 0},
      "diffusion": {"latency_weight": 0.0, "fairness_weight": 0.0, "unit_bids": False},
    },
    "max_iterations": 2,
    "patience": 1,
    "max_steps": 8,
    "min_run_time": 1,
    "max_run_time": 1,
    "run_time_step": 1,
    "checkpoint_interval": 1,
    "tolerance": 1e-6,
    "verbose": 0,
  }


def test_diffusion_runner_produces_expected_artifacts(tmp_path):
  _require_gurobi()
  folder = run_diffusion(
    _diffusion_e2e_config(tmp_path), parallelism=0, disable_plotting=True
  )

  obj = pd.read_csv(Path(folder, "obj.csv"))
  assert "FaaS-MADiG" in obj.columns
  assert np.isfinite(pd.to_numeric(obj["FaaS-MADiG"], errors="coerce")).all()

  runtime = pd.read_csv(Path(folder, "runtime.csv"))
  assert "tot" in runtime.columns
  assert (runtime["tot"] >= 0).all()

  assert Path(folder, "termination_condition.csv").exists()
  assert Path(folder, "LSPc_solution.csv").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_diffusion_e2e.py -v`
Expected: FAIL — `ImportError: cannot import name 'run'` (or `AttributeError`), or skip if Gurobi missing.

- [ ] **Step 3: Write minimal implementation**

Add `parse_arguments` and `run` to `decentralized_diffusion.py`:

```python
def parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Run FaaS-MADiG",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
    "-c", "--config", help="Configuration file", type=str,
    default="config_files/manual_config.json",
  )
  parser.add_argument(
    "-j", "--parallelism",
    help="Number of parallel processes (-1: auto, 0: sequential)",
    type=int, default=-1,
  )
  parser.add_argument(
    "--disable_plotting",
    help="True to disable automatic plot generation for each experiment",
    default=False, action="store_true",
  )
  return parser.parse_known_args()[0]


def run(
    config: dict,
    parallelism: int,
    log_on_file: bool = False,
    disable_plotting: bool = False,
  ):
  base_solution_folder = config["base_solution_folder"]
  seed = config["seed"]
  limits = config["limits"]
  trace_type = config["limits"]["load"].get("trace_type", "fixed_sum")
  verbose = config.get("verbose", 0)
  solver_name = config["solver_name"]
  solver_options = config["solver_options"]
  general_solver_options = solver_options.get("general", {})
  diffusion_options = solver_options["diffusion"]
  diffusion_options.setdefault(
    "unit_bids", solver_options.get("auction", {}).get("unit_bids", False)
  )
  time_limit = general_solver_options.get("TimeLimit", np.inf)
  tolerance = config.get("tolerance", 1e-6)
  max_iterations = config["max_iterations"]
  max_steps = config["max_steps"]
  min_run_time = config.get("min_run_time", 0)
  max_run_time = config.get("max_run_time", max_steps)
  run_time_step = config.get("run_time_step", 1)
  checkpoint_interval = config["checkpoint_interval"]
  patience = config["patience"]
  now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
  solution_folder = f"{base_solution_folder}/{now}"
  os.makedirs(solution_folder, exist_ok=True)
  with open(os.path.join(solution_folder, "config.json"), "w") as ostream:
    ostream.write(json.dumps(config, indent=2))
  log_stream = sys.stdout
  if log_on_file:
    log_stream = open(os.path.join(solution_folder, "out.log"), "w")
  base_instance_data, input_requests_traces, agents, graph = init_problem(
    limits, trace_type, max_steps, seed, solution_folder
  )
  Nn = base_instance_data[None]["Nn"][None]
  Nf = base_instance_data[None]["Nf"][None]
  neighborhood = neigh_dict_to_matrix(
    base_instance_data[None]["neighborhood"], Nn
  )
  latency = adjacency_matrix(graph, weight="network_latency")
  ub = (
    max_run_time + run_time_step
  ) if max_run_time == min_run_time else max_run_time
  sp_complete_solution = init_complete_solution()
  spc_complete_solution = init_complete_solution()
  obj_dict = {"LSPr_final": []}
  tc_dict = {"LSPr": []}
  runtime_list = []
  for t in range(min_run_time, ub, run_time_step):
    if verbose > 0:
      print(f"t = {t}", file=log_stream, flush=True)
    loadt = get_current_load(input_requests_traces, agents, t)
    data = update_data(base_instance_data, {"incoming_load": loadt})
    total_runtime = 0
    ss = datetime.now()
    sp_data = deepcopy(data)
    sp = LSP()
    spr = LSPr()
    (
      sp_data, sp_x, _, _, sp_omega, sp_r, sp_rho, sp_U, obj, tc, sp_runtime
    ) = solve_subproblem(
      sp_data, agents, sp, solver_name, general_solver_options, parallelism
    )
    total_runtime += sp_runtime["tot"]
    it = 0
    stop_searching = False
    best_solution_so_far = None
    best_centralized_solution = None
    best_cost_so_far = np.inf
    spr_obj = np.inf
    best_centralized_cost = 0.0
    best_it_so_far = -1
    best_centralized_it = -1
    y = np.zeros((Nn, Nn, Nf))
    omega = deepcopy(sp_omega)
    fairness = np.zeros((Nn, Nf))
    n_accepted_queue = deque(maxlen=patience)
    while not stop_searching:
      s = datetime.now()
      capacity, residual_capacity, ell = compute_residual_capacity(
        sp_x, y, sp_r, sp_data
      )
      blackboard = np.maximum(0.0, capacity - sp_x)
      total_runtime += (datetime.now() - s).total_seconds()
      s = datetime.now()
      bids, memory_bids, n_auctions = define_assignments(
        omega, blackboard, sp_data, neighborhood, sp_rho,
        diffusion_options, latency, fairness,
        force_memory_bids=(
          (sp_rho > 0).any()
          and len(n_accepted_queue) >= n_accepted_queue.maxlen
          and all(x == n_accepted_queue[0] for x in n_accepted_queue)
        ),
      )
      rt = (datetime.now() - s).total_seconds()
      total_runtime += (rt / n_auctions) if n_auctions else rt
      rmp_omega = np.zeros((Nn, Nf))
      additional_replicas = np.zeros((Nn, Nf))
      if len(bids) > 0:
        s = datetime.now()
        diffusion_y, additional_replicas, n_sellers = evaluate_assignments(
          bids, residual_capacity, sp_data, ell, sp_r, sp_rho,
          tentatively_start_replicas=(len(memory_bids) == 0),
        )
        rt = (datetime.now() - s).total_seconds()
        total_runtime += (rt / n_sellers) if n_sellers else rt
        y += diffusion_y
        for n in range(Nn):
          for f in range(Nf):
            rmp_omega[n, f] = y[n, :, f].sum()
            if rmp_omega[n, f] > 0:
              fairness[n, f] += 1
        n_accepted_queue.append(rmp_omega.sum())
        bad_nodes = check_ls_pr_feasibility_from_fixed_y(sp_data, y)
        if bad_nodes:
          raise RuntimeError(f"LSPr infeasible from fixed y assignments: {bad_nodes}")
        spr_sol, spr_obj, spr_tc, spr_runtime = compute_social_welfare(
          spr, sp_data, agents, solver_name, general_solver_options,
          y, rmp_omega, parallelism
        )
        total_runtime += spr_runtime
        sp_x, _, _, _, sp_r, sp_rho = spr_sol
        for i in range(Nn):
          for f in range(Nf):
            omega[i, f] = sp_omega[i, f] - rmp_omega[i, f]
            if abs(omega[i, f]) < tolerance:
              omega[i, f] = 0.0
      if len(memory_bids) > 0 and not (additional_replicas > 0).any():
        s = datetime.now()
        additional_replicas, sp_rho = start_additional_replicas(
          memory_bids, sp_r, sp_data, sp_rho
        )
        sp_r += additional_replicas
        total_runtime += (datetime.now() - s).total_seconds()
      csol = combine_solutions(
        Nn, Nf, sp_data, loadt, sp_x, sp_r, sp_rho,
        None, y, None, None, None, None
      )
      cobj = compute_centralized_objective(
        sp_data, csol["sp"]["x"], csol["sp"]["y"], csol["sp"]["z"]
      )
      feas = check_feasibility(
        csol["sp"]["x"], csol["sp"]["y"].sum(axis=1), csol["sp"]["z"],
        csol["sp"]["r"], csol["sp"]["U"], sp_data
      )
      assert feas[0], feas[1]
      if spr_obj < best_cost_so_far or it == 0:
        best_cost_so_far = spr_obj
        best_solution_so_far = deepcopy(csol)
        best_it_so_far = it
      if cobj > best_centralized_cost:
        best_centralized_cost = cobj
        best_centralized_solution = deepcopy(csol)
        best_centralized_it = it
      stop_searching, why_stop_searching = check_stopping_criteria(
        it, max_iterations, blackboard, omega, rmp_omega,
        additional_replicas, bids, memory_bids,
        tolerance, total_runtime, time_limit
      )
      if not stop_searching:
        it += 1
      else:
        sp_complete_solution, _, objf = decode_solutions(
          sp_data, best_solution_so_far, sp_complete_solution, None
        )
        spc_complete_solution, _, _ = decode_solutions(
          sp_data, best_centralized_solution, spc_complete_solution, None
        )
        obj_dict["LSPr_final"].append(objf)
        tc_dict["LSPr"].append(
          f"{why_stop_searching} "
          f"(it: {it}; obj. deviation: {None}; best it: {best_it_so_far}; "
          f"best centralized it: {best_centralized_it}; "
          f"total runtime: {total_runtime})"
        )
        if t % checkpoint_interval == 0 or t == max_steps - 1:
          save_checkpoint(
            sp_complete_solution, os.path.join(solution_folder, "LSP"), t
          )
          save_checkpoint(
            spc_complete_solution, os.path.join(solution_folder, "LSPc"), t
          )
    runtime_list.append(total_runtime)
    if verbose > 0:
      print(
        f"    TOTAL RUNTIME [s] = {total_runtime} "
        f"(wallclock: {(datetime.now() - ss).total_seconds()})",
        file=log_stream, flush=True
      )
  sp_solution, sp_offloaded, sp_detailed_fwd_solution = join_complete_solution(
    sp_complete_solution
  )
  spc_solution, spc_offloaded, spc_detailed_fwd_solution = join_complete_solution(
    spc_complete_solution
  )
  if not disable_plotting and Nf <= 10 and Nn <= 10:
    plot_history(
      input_requests_traces, min_run_time, max_run_time, run_time_step,
      sp_solution, sp_complete_solution["utilization"],
      sp_complete_solution["replicas"], sp_offloaded,
      obj_dict["LSPr_final"], os.path.join(solution_folder, "sp.png")
    )
  save_solution(
    sp_solution, sp_offloaded, sp_complete_solution,
    sp_detailed_fwd_solution, "LSP", solution_folder
  )
  save_solution(
    spc_solution, spc_offloaded, spc_complete_solution,
    spc_detailed_fwd_solution, "LSPc", solution_folder
  )
  pd.DataFrame(obj_dict["LSPr_final"], columns=["FaaS-MADiG"]).to_csv(
    os.path.join(solution_folder, "obj.csv"), index=False
  )
  pd.DataFrame(tc_dict["LSPr"]).to_csv(
    os.path.join(solution_folder, "termination_condition.csv")
  )
  pd.DataFrame({"tot": runtime_list}).to_csv(
    os.path.join(solution_folder, "runtime.csv"), index=False
  )
  if verbose > 0:
    print(f"All solutions saved in: {solution_folder}", file=log_stream, flush=True)
  if log_on_file:
    log_stream.close()
  return solution_folder


if __name__ == "__main__":
  args = parse_arguments()
  config = load_configuration(args.config)
  run(config, args.parallelism, disable_plotting=args.disable_plotting)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_diffusion_e2e.py -v`
Expected: PASS (or SKIP if Gurobi unavailable). Also run `uv run python decentralized_diffusion.py --help` → prints the FaaS-MADiG usage.

- [ ] **Step 5: Commit**

```bash
git add decentralized_diffusion.py tests/test_diffusion_e2e.py
git commit -m "feat(madig): add price-free run() runner with runtime.csv"
```

---

## Task 4: Wire `faas-diffuse` into `run.py` (additive)

**Files:**
- Modify: `run.py` (additive edits only)
- Test: `tests/test_diffusion_wiring.py`

**Interfaces:**
- Consumes: `decentralized_diffusion.run`.
- Produces: `run.py` accepts `--methods faas-diffuse`, dispatches to `run_diffusion`, and maps it to `mkey="LSPc"`, `mname="FaaS-MADiG"`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_diffusion_wiring.py`:

```python
import run


def test_methods_choice_accepts_faas_diffuse(monkeypatch):
  argv = ["run.py", "-c", "config_files/planar_comparison.json",
          "--methods", "faas-diffuse"]
  monkeypatch.setattr("sys.argv", argv)
  args = run.parse_arguments()
  assert "faas-diffuse" in args.methods


def test_run_module_exposes_diffusion_runner():
  assert hasattr(run, "run_diffusion")
  assert callable(run.run_diffusion)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_diffusion_wiring.py -v`
Expected: FAIL — `faas-diffuse` not in choices (argparse SystemExit) and `run` has no `run_diffusion`.

- [ ] **Step 3: Make the additive edits**

In `run.py`, add the import after line 6 (`from hierarchical_auction.runner import run as run_hierarchical`):

```python
from decentralized_diffusion import run as run_diffusion
```

Add `"faas-diffuse"` to the `--methods` choices list (after `"hierarchical",` near line 54):

```python
      "hierarchical",
      "faas-diffuse",
      "generate_only"
```

Extend `method_colors` (line 259-265) with a sixth color so all methods are distinct:

```python
    method_colors = [
      mcolors.TABLEAU_COLORS["tab:blue"],
      mcolors.TABLEAU_COLORS["tab:orange"],
      mcolors.TABLEAU_COLORS["tab:red"],
      mcolors.TABLEAU_COLORS["tab:green"],
      mcolors.TABLEAU_COLORS["tab:pink"],
      mcolors.TABLEAU_COLORS["tab:purple"]
    ]
```

Extend the `mname` ternary (line 280-286) — add the `faas-diffuse` branch (additive; existing methods unchanged):

```python
        mname = "LoadManagementModel" if method == "centralized" else (
          "FaaS-MACrO" if method == "faas-macro" else (
            "FaaS-MACrO(v0)" if method == "faas-macro-v0" else (
              "FaaS-MADeA" if method == "faas-madea" else (
                "FaaS-MADiG" if method == "faas-diffuse" else "HierarchicalAuction"
              )
            )
          )
        )
```

(`mkey` at line 277-279 already resolves to `"LSPc"` for any non-centralized, non-`faas-macro*` method, so `faas-diffuse` correctly maps to `LSPc` with no change.)

Add `"faas-diffuse": []` to the `solution_folders` init (line 863-870):

```python
  solution_folders = {
    "experiments_list": [],
    "centralized": [],
    "faas-macro": [],
    "faas-macro-v0": [],
    "faas-madea": [],
    "hierarchical": [],
    "faas-diffuse": []
  }
```

Add the `run_d` flag init after line 888 (`run_h = False # -- hierarchical`):

```python
    run_h = False # -- hierarchical
    run_d = False # -- faas-diffuse (FaaS-MADiG)
```

Add the resume check after the hierarchical block (after line 923):

```python
      if (not generate_only and "hierarchical" in methods) and ((
          len(solution_folders["hierarchical"]) <= experiment_idx
        ) or (
          solution_folders["hierarchical"][experiment_idx] is None
        )):
        run_h = True
      if (not generate_only and "faas-diffuse" in methods) and ((
          len(solution_folders.get("faas-diffuse", [])) <= experiment_idx
        ) or (
          solution_folders["faas-diffuse"][experiment_idx] is None
        )):
        run_d = True
```

Add to the `except ValueError` simple-check block (after line 929 `run_h = "hierarchical" in methods`):

```python
      run_h = "hierarchical" in methods
      run_d = "faas-diffuse" in methods
```

Extend the run-guard condition (line 931):

```python
    if run_c or run_i or run_i_v0 or run_a or run_h or run_d or generate_only:
```

Add the dispatch block after the hierarchical dispatch (after line 1031, before `# -- save info`):

```python
        set_solution_folder(
          solution_folders, "hierarchical", experiment_idx, h_folder
        )
      # -- solve diffusion (FaaS-MADiG)
      if run_d:
        d_folder = run_diffusion(
          config,
          sp_parallelism,
          log_on_file = log_on_file,
          disable_plotting = disable_plotting
        )
        set_solution_folder(
          solution_folders, "faas-diffuse", experiment_idx, d_folder
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_diffusion_wiring.py -v`
Expected: 2 passed.
Then a regression check that other methods' wiring is intact:
Run: `uv run pytest tests/test_parse_args_extended.py tests/test_run_helpers.py -v`
Expected: all previously-passing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add run.py tests/test_diffusion_wiring.py
git commit -m "feat(madig): wire faas-diffuse method into run.py orchestrator"
```

---

## Task 5: Config + `compare_results.py` palette (additive)

**Files:**
- Modify: `config_files/planar_comparison.json`
- Modify: `compare_results.py` (additive — one dict key)
- Test: `tests/test_diffusion_wiring.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: a `solver_options.diffusion` block in the planar config; a `FaaS-MADiG` entry in the `compare_results.py` box/violin `colors` dict.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_diffusion_wiring.py`:

```python
import json
from pathlib import Path


def test_planar_config_has_diffusion_section():
  config = json.loads(Path("config_files/planar_comparison.json").read_text())
  diffusion = config["solver_options"]["diffusion"]
  assert "latency_weight" in diffusion
  assert "fairness_weight" in diffusion


def test_compare_results_palette_includes_madig():
  import inspect

  import compare_results

  source = inspect.getsource(compare_results)
  assert '"FaaS-MADiG"' in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_diffusion_wiring.py -k "diffusion_section or palette" -v`
Expected: FAIL — no `diffusion` key in config; no `FaaS-MADiG` in `compare_results.py`.

- [ ] **Step 3: Make the additive edits**

In `config_files/planar_comparison.json`, add the `diffusion` block inside `solver_options` (after the `auction` block, line 15-21):

```json
    "auction": {
      "epsilon": 0.01,
      "eta": [0.5, 0.3, 0.15],
      "zeta": 0.1,
      "latency_weight": 0.0,
      "fairness_weight": 0.0
    },
    "diffusion": {
      "latency_weight": 0.0,
      "fairness_weight": 0.0,
      "unit_bids": false
    }
```

In `compare_results.py`, add the `FaaS-MADiG` key to the `colors` dict (line 667-672):

```python
  colors = {
    "LoadManagementModel": mcolors.CSS4_COLORS["lightgreen"],
    "FaaS-MACrO": mcolors.CSS4_COLORS["lightpink"],
    "FaaS-MACrO(v0)": mcolors.CSS4_COLORS["lightcoral"],
    "FaaS-MADeA": mcolors.CSS4_COLORS["lightskyblue"],
    "FaaS-MADiG": mcolors.CSS4_COLORS["plum"]
  }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_diffusion_wiring.py -v`
Expected: 4 passed.
Then validate the config parses and is still a valid planar config:
Run: `uv run python -c "import json; json.load(open('config_files/planar_comparison.json'))"`
Expected: no error.

- [ ] **Step 5: Commit**

```bash
git add config_files/planar_comparison.json compare_results.py tests/test_diffusion_wiring.py
git commit -m "feat(madig): add diffusion config block and compare_results palette"
```

---

## Final verification (after all tasks)

- [ ] Run the full unit suite: `uv run pytest tests/test_diffusion_helpers.py tests/test_diffusion_wiring.py -v` → all pass.
- [ ] Run the e2e (needs Gurobi): `uv run pytest tests/test_diffusion_e2e.py -v` → pass or skip.
- [ ] Run the existing suite to confirm no regressions: `uv run pytest -q`.
- [ ] Optional manual four-way run (needs SageMath + Gurobi):
  ```
  uv run python run.py -c config_files/planar_comparison.json \
    --methods centralized faas-macro faas-madea hierarchical faas-diffuse \
    --n_experiments 1 --loop_over Nn
  ```
  Confirm `solutions/planar_comparison/` contains a FaaS-MADiG folder with `obj.csv` (column `FaaS-MADiG`) and `runtime.csv`.

## Self-review notes

- **Spec coverage:** §3.1 → Task 1; §3.2 → Task 2; §4 module + §6 artifacts (obj.csv column, runtime.csv, runtime accounting + `n_auctions` guard) → Task 3; §5 wiring + labels → Task 4; §4 config + §6 compare_results palette → Task 5; §7 tests → Tasks 1–5.
- **Caveat compliance:** no shared/algorithmic function is edited; new behavior is isolated in `decentralized_diffusion.py`; `run.py`/`compare_results.py` edits are additive; reused helpers are imported from `run_faasmadea`.
- **Type consistency:** `define_assignments` returns `(DataFrame[i,j,f,d,utility], DataFrame[i,j,f], int)`; `evaluate_assignments` consumes that first DataFrame and returns `(y, additional_replicas, int)`; `run()` consumes both. `mkey="LSPc"`, `mname="FaaS-MADiG"`, obj column `FaaS-MADiG`, runtime column `tot` are consistent across Tasks 3–5.
