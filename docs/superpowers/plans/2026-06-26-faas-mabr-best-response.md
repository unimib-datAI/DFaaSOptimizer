# FaaS-MABR Distributed Best-Response (Gauss-Seidel) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three solver-light Gauss-Seidel distributed heuristics — FaaS-MABR-S (sequential greedy, fixed order), FaaS-MABR-R (sequential greedy, randomized order), FaaS-MABR-O (capped local best response) — as the sequential counterpart to FaaS-MADiG/MAPoD's simultaneous coordination.

**Architecture:** One new module `decentralized_bestresponse.py` holds a shared `best_response_sweep` (sequential placement on a shared, in-place-decremented residual ledger), a solver-backed `reoptimize_node` (per-node `LSP_capped` solve for the `-O` variant), a parameterized internal `_run` (the per-runner control-period loop, copied from `decentralized_diffusion.run` with the coordinate step swapped for one sweep), and three thin entry points `run_br_s/r/o`. Two additive model classes `LSP_capped`/`LSP_capped_fixedr` add an `omega` upper bound. All `run.py`/`compare_results.py`/planar-config changes are additive.

**Tech Stack:** Python 3.10, NumPy, pandas, Pyomo (Gurobi/GLPK). 2-space indentation. Run/test via `uv run` / `uv run pytest`. LaTeX via `latexmk`.

## Global Constraints

- **2-space indentation** throughout (match existing files).
- **Do NOT modify** functions/classes that support other methods: `run_faasmadea.py`, `run_faasmacro.py`, `run_centralized_model.py`, existing classes in `models/sp.py`, `utils/`, `hierarchical_auction/`, `decentralized_diffusion.py`, `decentralized_powerd.py`. Reuse by import; new behaviour → new function/class.
- **`run.py` / `compare_results.py` / `planar_comparison.json` changes are ADDITIVE only** — existing methods' code paths stay byte-for-byte unchanged.
- **Method names** `FaaS-MABR-S` / `FaaS-MABR-R` / `FaaS-MABR-O`; CLI keys `faas-br-s` / `faas-br-r` / `faas-br-o`; `obj.csv` column = the method name; artifact key `LSPc`; runner symbols `run_br_s` / `run_br_r` / `run_br_o`.
- **Coordination = one sequential sweep over a shared ledger** (true residual capacity `compute_residual_capacity`, decremented in place). Admission uses score `s_{ij}^f = beta − latency_weight·L − fairness_weight·phi` and threshold `s > −gamma_i^f`; placement is descending score, tie-break lower `j`. No seller-clearing phase, no reuse of `evaluate_assignments`/`define_assignments`.
- **`-O` reopt:** per node, cap `omega` at accessible neighbour residual capacity and re-solve via `solve_subproblem([node], LSP_capped()/LSP_capped_fixedr())`; only `omega[node,:]` is committed; final `x/z/r/rho` come from the subsequent `LSPr` solve. The 4th combination (randomized-order reopt) is out of scope.
- **`-R` stochasticity:** node order permutation from `np.random.default_rng(config["seed"])` (created once); variance via `--n_experiments`. `_run` copies its options block with `dict(...)` so the input config is never mutated.
- **Stopping:** reuse `check_stopping_criteria` (passing `bids=None`) plus an explicit MABR guard: stop with `"no best-response progress"` when a sweep places no load and emits no memory bids.
- **Replica expansion** is sweep-level via `start_additional_replicas`, at parity with FaaS-MADiG's two-block memory-bid emission.

---

### Task 1: `LSP_capped` / `LSP_capped_fixedr` model classes

**Files:**
- Modify: `models/sp.py` (append two classes)
- Test: `tests/test_lsp_capped.py`

**Interfaces:**
- Consumes: existing `LSP`, `LSP_fixedr` (in `models/sp.py`).
- Produces: `LSP_capped()` (subclass of `LSP`) and `LSP_capped_fixedr()` (subclass of `LSP_fixedr`), each with a `Param omega_ub` (indexed by the function set `F`, default `1e9`) and a `Constraint cap_offloading` enforcing `omega[f] <= omega_ub[f]`. Data key: `data[None]["omega_ub"] = {f+1: value}`.

- [ ] **Step 1: Write the failing structural test**

Create `tests/test_lsp_capped.py`:

```python
from models.sp import LSP, LSP_fixedr, LSP_capped, LSP_capped_fixedr


def test_lsp_capped_subclasses_lsp_and_adds_cap():
  m = LSP_capped()
  assert isinstance(m, LSP)
  assert m.model.component("omega_ub") is not None
  assert m.model.component("cap_offloading") is not None


def test_lsp_capped_fixedr_subclasses_fixedr_and_adds_cap():
  m = LSP_capped_fixedr()
  assert isinstance(m, LSP_fixedr)
  assert m.model.component("omega_ub") is not None
  assert m.model.component("cap_offloading") is not None
  assert m.model.component("fix_r") is not None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_lsp_capped.py -v`
Expected: FAIL — `ImportError: cannot import name 'LSP_capped' from 'models.sp'`.

- [ ] **Step 3: Append the two classes to `models/sp.py`**

Append at the end of `models/sp.py`:

```python
##############################################################################
# OFFLOADING CAP (for FaaS-MABR-O capped local best response)
##############################################################################

class LSP_capped(LSP):
  def __init__(self):
    super().__init__()
    self.name = "LSP_capped"
    ###########################################################################
    # Problem parameters
    ###########################################################################
    # per-function upper bound on horizontal offloading
    self.model.omega_ub = pyo.Param(
      self.model.F, within = pyo.NonNegativeReals, default = 1e9
    )
    ###########################################################################
    # Constraints
    ###########################################################################
    self.model.cap_offloading = pyo.Constraint(
      self.model.F, rule = self.cap_offloading
    )

  @staticmethod
  def cap_offloading(model, f):
    return model.omega[f] <= model.omega_ub[f]


class LSP_capped_fixedr(LSP_fixedr):
  def __init__(self):
    super().__init__()
    self.name = "LSP_capped_fixedr"
    ###########################################################################
    # Problem parameters
    ###########################################################################
    # per-function upper bound on horizontal offloading
    self.model.omega_ub = pyo.Param(
      self.model.F, within = pyo.NonNegativeReals, default = 1e9
    )
    ###########################################################################
    # Constraints
    ###########################################################################
    self.model.cap_offloading = pyo.Constraint(
      self.model.F, rule = self.cap_offloading
    )

  @staticmethod
  def cap_offloading(model, f):
    return model.omega[f] <= model.omega_ub[f]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_lsp_capped.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add models/sp.py tests/test_lsp_capped.py
git commit -m "feat: add LSP_capped / LSP_capped_fixedr offloading-cap models"
```

---

### Task 2: `best_response_sweep` (sequential greedy core)

**Files:**
- Create: `decentralized_bestresponse.py`
- Test: `tests/test_bestresponse_helpers.py`

**Interfaces:**
- Consumes: `data[None]["beta"][(i+1,j+1,f+1)]`, `data[None]["gamma"][(i+1,f+1)]` (same layout as `decentralized_diffusion.define_assignments`).
- Produces: `best_response_sweep(omega, residual_capacity, data, neighborhood, rho, br_options, latency, fairness, force_memory_bids, *, order, response, rng=None, reopt_fn=None) -> (y_increment: np.ndarray(Nn,Nn,Nf), memory_bids: pd.DataFrame[i,j,f], n_active: int, placed_total: float, reopt_runtime: float)`. When `response=="reopt"`, `reopt_fn(node:int, omega_ub_row:np.ndarray(Nf)) -> (capped_omega_row:np.ndarray(Nf), runtime:float)` is called per node before placement.

- [ ] **Step 1: Create the module header + the failing test file**

Create `decentralized_bestresponse.py` with the full header (everything `_run` will need in Task 4) followed by nothing else yet:

```python
from run_centralized_model import (
  encode_solution,
  get_current_load,
  init_complete_solution,
  init_problem,
  join_complete_solution,
  plot_history,
  save_checkpoint,
  save_solution,
  update_data,
)
from postprocessing import load_solution
from run_faasmacro import (
  combine_solutions,
  compute_social_welfare,
  decode_solutions,
  solve_subproblem,
)
from run_faasmadea import (
  check_ls_pr_feasibility_from_fixed_y,
  check_stopping_criteria,
  compute_residual_capacity,
  neigh_dict_to_matrix,
  start_additional_replicas,
)
from utils.centralized import check_feasibility
from utils.faasmacro import compute_centralized_objective
from utils.common import load_configuration
from models.sp import LSP, LSP_fixedr, LSPr, LSP_capped, LSP_capped_fixedr

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
```

Then create `tests/test_bestresponse_helpers.py`:

```python
import numpy as np
import pandas as pd

from decentralized_bestresponse import best_response_sweep


def _base_data(Nn=4, Nf=1):
  data = {None: {
    "Nn": {None: Nn},
    "Nf": {None: Nf},
    "beta": {},
    "gamma": {},
  }}
  for i in range(Nn):
    for f in range(Nf):
      data[None]["gamma"][(i + 1, f + 1)] = 0.05
      for j in range(Nn):
        data[None]["beta"][(i + 1, j + 1, f + 1)] = 1.0
  return data


def _full_neighborhood(Nn=4):
  return np.ones((Nn, Nn)) - np.eye(Nn)


def _opts():
  return {"latency_weight": 0.0, "fairness_weight": 0.0}


def test_sweep_ledger_is_order_dependent():
  # buyers 0 and 1 both want 2 from the only capacity seller 2 (cap 2);
  # in a fixed sequential sweep, node 0 (first) takes it all, node 1 gets none.
  data = _base_data(Nn=3, Nf=1)
  omega = np.zeros((3, 1)); omega[0, 0] = 2.0; omega[1, 0] = 2.0
  residual = np.zeros((3, 1)); residual[2, 0] = 2.0
  neighborhood = np.zeros((3, 3))
  neighborhood[0, 2] = 1; neighborhood[1, 2] = 1
  rho = np.zeros((3,))

  y, mem, n_active, placed, rt = best_response_sweep(
    omega, residual, data, neighborhood, rho, _opts(),
    np.zeros((3, 3)), np.zeros((3, 1)), force_memory_bids=False,
    order="fixed", response="greedy")

  assert y[0, 2, 0] == 2.0      # first node claimed the shared capacity
  assert y[1, 2, 0] == 0.0      # later node saw the decremented ledger
  assert placed == 2.0
  assert rt == 0.0


def test_sweep_fixed_order_is_deterministic():
  data = _base_data()
  data[None]["beta"][(1, 2, 1)] = 3.0
  data[None]["beta"][(1, 3, 1)] = 2.0
  omega = np.zeros((4, 1)); omega[0, 0] = 3.0
  residual = np.zeros((4, 1)); residual[1, 0] = 2; residual[2, 0] = 2
  args = (omega, residual, data, _full_neighborhood(), np.zeros((4,)),
          _opts(), np.zeros((4, 4)), np.zeros((4, 1)))

  y1, *_ = best_response_sweep(*args, force_memory_bids=False,
                               order="fixed", response="greedy")
  y2, *_ = best_response_sweep(*args, force_memory_bids=False,
                               order="fixed", response="greedy")
  assert np.array_equal(y1, y2)


def test_sweep_random_order_reproducible_with_same_seed():
  data = _base_data()
  for j in [1, 2, 3]:
    data[None]["beta"][(1, j + 1, 1)] = float(j)
  omega = np.zeros((4, 1)); omega[0, 0] = 2.0
  residual = np.zeros((4, 1)); residual[1, 0] = 1; residual[2, 0] = 1; residual[3, 0] = 1
  args = (omega, residual, data, _full_neighborhood(), np.zeros((4,)),
          _opts(), np.zeros((4, 4)), np.zeros((4, 1)))

  y1, *_ = best_response_sweep(*args, force_memory_bids=False, order="random",
                               response="greedy", rng=np.random.default_rng(7))
  y2, *_ = best_response_sweep(*args, force_memory_bids=False, order="random",
                               response="greedy", rng=np.random.default_rng(7))
  assert np.array_equal(y1, y2)


def test_sweep_threshold_excludes_unconvenient_seller():
  data = _base_data(Nn=2, Nf=1)
  data[None]["beta"][(1, 2, 1)] = -1.0   # score -1.0 <= -0.05 => excluded
  omega = np.zeros((2, 1)); omega[0, 0] = 1.0
  residual = np.zeros((2, 1)); residual[1, 0] = 5
  neighborhood = np.array([[0.0, 1.0], [1.0, 0.0]])

  y, mem, n_active, placed, rt = best_response_sweep(
    omega, residual, data, neighborhood, np.zeros((2,)), _opts(),
    np.zeros((2, 2)), np.zeros((2, 1)), force_memory_bids=False,
    order="fixed", response="greedy")

  assert placed == 0.0
  assert len(mem) == 0


def test_sweep_emits_memory_bids_when_no_capacity():
  data = _base_data(Nn=2, Nf=1)
  omega = np.zeros((2, 1)); omega[0, 0] = 2.0
  residual = np.zeros((2, 1))                 # no capacity sellers
  rho = np.zeros((2,)); rho[1] = 4.0           # neighbour 1 has spare memory
  neighborhood = np.array([[0.0, 1.0], [1.0, 0.0]])

  y, mem, n_active, placed, rt = best_response_sweep(
    omega, residual, data, neighborhood, rho, _opts(),
    np.zeros((2, 2)), np.zeros((2, 1)), force_memory_bids=False,
    order="fixed", response="greedy")

  assert placed == 0.0
  assert list(mem[["i", "j", "f"]].iloc[0]) == [0, 1, 0]


def test_sweep_reopt_branch_caps_omega_via_reopt_fn():
  # response="reopt" calls reopt_fn before placement; a fake fn halves omega.
  data = _base_data(Nn=2, Nf=1)
  omega = np.zeros((2, 1)); omega[0, 0] = 4.0
  residual = np.zeros((2, 1)); residual[1, 0] = 10
  neighborhood = np.array([[0.0, 1.0], [1.0, 0.0]])
  calls = {}

  def fake_reopt(node, omega_ub_row):
    calls["node"] = node
    calls["ub"] = list(omega_ub_row)
    return np.array([2.0]), 0.5     # capped to 2.0, 0.5s runtime

  y, mem, n_active, placed, rt = best_response_sweep(
    omega, residual, data, neighborhood, np.zeros((2,)), _opts(),
    np.zeros((2, 2)), np.zeros((2, 1)), force_memory_bids=False,
    order="fixed", response="reopt", reopt_fn=fake_reopt)

  assert calls["node"] == 0
  assert calls["ub"] == [10.0]      # accessible neighbour capacity
  assert placed == 2.0              # placed the capped amount, not 4.0
  assert rt == 0.5
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_bestresponse_helpers.py -v`
Expected: FAIL — `ImportError: cannot import name 'best_response_sweep'`.

- [ ] **Step 3: Implement `best_response_sweep`**

Append to `decentralized_bestresponse.py`:

```python
def best_response_sweep(
    omega: np.array,
    residual_capacity: np.array,
    data: dict,
    neighborhood: np.array,
    rho: np.array,
    br_options: dict,
    latency: np.array,
    fairness: np.array,
    force_memory_bids: bool,
    *,
    order: str,
    response: str,
    rng: np.random.Generator = None,
    reopt_fn=None,
  ) -> Tuple[np.array, pd.DataFrame, int, float, float]:
  """Sequential (Gauss-Seidel) best-response sweep.

  Nodes act in `order` ("fixed" = ascending index, "random" = rng permutation);
  each places its residual demand greedily by descending score onto the shared
  `ledger` (a copy of residual_capacity), decremented in place so later nodes
  see what earlier nodes took. For response=="reopt", each node first caps its
  omega row at the accessible neighbour capacity via reopt_fn before placing.
  Replica-expansion memory bids use the FaaS-MADiG two-block emission.
  """
  Nn = data[None]["Nn"][None]
  Nf = data[None]["Nf"][None]
  ledger = np.array(residual_capacity, dtype=float)
  omega = np.array(omega, dtype=float)  # working copy; never mutate the caller's
  y_increment = np.zeros((Nn, Nn, Nf))
  memory_bids = {"i": [], "j": [], "f": []}
  reopt_runtime = 0.0
  active = set()
  memory_seller_nodes = set(int(j) for j in np.nonzero(rho)[0])
  if order == "random":
    node_order = [int(i) for i in rng.permutation(Nn)]
  else:
    node_order = list(range(Nn))
  for i in node_order:
    neighbours = set(int(j) for j in np.nonzero(neighborhood[i, :])[0])
    if response == "reopt" and reopt_fn is not None:
      omega_ub_row = np.array(
        [sum(ledger[j, f] for j in neighbours) for f in range(Nf)]
      )
      capped_row, rt = reopt_fn(i, omega_ub_row)
      reopt_runtime += rt
      omega[i, :] = capped_row
    potential_memory = neighbours & memory_seller_nodes
    for f in range(Nf):
      if omega[i, f] <= 0:
        continue
      score = {}
      candidates = []
      for j in neighbours:
        if ledger[j, f] >= 1:
          s = (
            data[None]["beta"][(i + 1, j + 1, f + 1)]
            - br_options["latency_weight"] * latency[i, j]
            - br_options["fairness_weight"] * fairness[i, f]
          )
          if s > - data[None]["gamma"][(i + 1, f + 1)]:
            score[j] = s
            candidates.append(j)
      placed = 0.0
      for j in sorted(candidates, key=lambda k: (-score[k], k)):
        if placed >= omega[i, f]:
          break
        q = min(ledger[j, f], omega[i, f] - placed)
        if q <= 0:
          continue
        y_increment[i, j, f] += q
        ledger[j, f] -= q
        placed += q
        active.add(i)
      potential_capacity = set(
        j for j in neighbours if residual_capacity[j, f] >= 1
      )
      if placed < omega[i, f]:
        for j in sorted(score, key=lambda k: (-score[k], k)):
          if j in potential_memory:
            memory_bids["i"].append(i)
            memory_bids["j"].append(j)
            memory_bids["f"].append(f)
      if placed < omega[i, f] or force_memory_bids:
        for j in sorted(potential_memory - potential_capacity):
          memory_bids["i"].append(i)
          memory_bids["j"].append(j)
          memory_bids["f"].append(f)
  placed_total = float(y_increment.sum())
  return (
    y_increment, pd.DataFrame(memory_bids), len(active),
    placed_total, reopt_runtime,
  )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_bestresponse_helpers.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add decentralized_bestresponse.py tests/test_bestresponse_helpers.py
git commit -m "feat: add sequential best-response sweep (FaaS-MABR core)"
```

---

### Task 3: `reoptimize_node` (capped per-node solve)

**Files:**
- Modify: `decentralized_bestresponse.py` (append `reoptimize_node`)
- Test: `tests/test_bestresponse_reopt.py`

**Interfaces:**
- Consumes: `solve_subproblem` (imported), `LSP_capped`/`LSP_capped_fixedr` (Task 1).
- Produces: `reoptimize_node(node, omega_ub_row, sp_data, solver_name, general_solver_options, parallelism, use_fixed_r) -> (omega_row: np.ndarray(Nf), runtime: float)`. Deep-copies `sp_data`, sets `omega_ub`, solves only `node`, returns its re-optimized `omega` row and the solve runtime. This is the `reopt_fn` the `-O` runner passes to `best_response_sweep`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_bestresponse_reopt.py`:

```python
from pathlib import Path

import numpy as np
import pyomo.environ as pyo
import pytest

from run_centralized_model import init_problem, update_data, get_current_load
from decentralized_bestresponse import reoptimize_node


def _require_gurobi() -> None:
  solver = pyo.SolverFactory("gurobi")
  if not solver.available(exception_flag=False):
    pytest.skip("Gurobi solver is not available")


def _tiny_instance(tmp_path: Path):
  limits = {
    "Nn": {"min": 3, "max": 3},
    "Nf": {"min": 1, "max": 1},
    "neighborhood": {"type": "planar", "degree": 2},
    "weights": {
      "alpha": {"min": 1.0, "max": 1.0},
      "beta_multiplier": {"min": 1.5, "max": 2.0},
      "gamma": {"min": 0.05, "max": 0.1},
      "delta_multiplier": {"min": 0.1, "max": 0.2},
    },
    "demand": {"values": [1.0]},
    "memory_capacity": {"values": [12, 12, 12]},
    "memory_requirement": {"values": [2]},
    "max_utilization": {"min": 0.7, "max": 0.7},
    "load": {"trace_type": "clipped",
             "min": {"min": 2.0, "max": 2.0},
             "max": {"min": 3.0, "max": 3.0}},
  }
  base, traces, agents, graph = init_problem(
    limits, "clipped", 4, 21, str(tmp_path))
  loadt = get_current_load(traces, agents, 1)
  data = update_data(base, {"incoming_load": loadt})
  return data


def test_reoptimize_node_respects_cap_and_reduces_offload(tmp_path):
  _require_gurobi()
  data = _tiny_instance(tmp_path)
  Nf = data[None]["Nf"][None]

  loose = np.full(Nf, 1e9)
  tight = np.zeros(Nf)        # no neighbour capacity at all

  omega_loose, rt1 = reoptimize_node(
    0, loose, data, "gurobi", {"OutputFlag": 0}, 0, use_fixed_r=False)
  omega_tight, rt2 = reoptimize_node(
    0, tight, data, "gurobi", {"OutputFlag": 0}, 0, use_fixed_r=False)

  assert (omega_tight <= tight + 1e-6).all()    # cap respected (omega ~ 0)
  assert omega_tight.sum() <= omega_loose.sum() + 1e-6  # capping cannot raise offload
  assert rt1 >= 0 and rt2 >= 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_bestresponse_reopt.py -v`
Expected: FAIL — `ImportError: cannot import name 'reoptimize_node'`.

- [ ] **Step 3: Implement `reoptimize_node`**

Append to `decentralized_bestresponse.py`:

```python
def reoptimize_node(
    node: int,
    omega_ub_row: np.array,
    sp_data: dict,
    solver_name: str,
    general_solver_options: dict,
    parallelism: int,
    use_fixed_r: bool,
  ) -> Tuple[np.array, float]:
  """Capped local best response for one node.

  Deep-copies sp_data, sets the per-function offloading cap omega_ub to the
  accessible neighbour residual capacity, and re-solves ONLY ``node`` with
  LSP_capped (or LSP_capped_fixedr when fixed replicas are active). Returns the
  node's re-optimized omega row and the solve runtime. Only this omega row is
  consumed by the caller; the capped solve's x/z/r are diagnostic.
  """
  Nf = sp_data[None]["Nf"][None]
  node_data = deepcopy(sp_data)
  node_data[None]["omega_ub"] = {
    (f + 1): float(omega_ub_row[f]) for f in range(Nf)
  }
  model = LSP_capped_fixedr() if use_fixed_r else LSP_capped()
  result = solve_subproblem(
    node_data, [node], model, solver_name, general_solver_options, parallelism
  )
  sp_omega = result[4]
  runtime = result[10]["tot"]
  return np.array(sp_omega[node, :], dtype=float), float(runtime)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_bestresponse_reopt.py -v`
Expected: PASS if Gurobi is available, else SKIP.

- [ ] **Step 5: Commit**

```bash
git add decentralized_bestresponse.py tests/test_bestresponse_reopt.py
git commit -m "feat: add reoptimize_node capped per-node solve (FaaS-MABR-O)"
```

---

### Task 4: `_run` + entry points `run_br_s/r/o`

**Files:**
- Modify: `decentralized_bestresponse.py` (append `parse_arguments`, `_run`, `run_br_s/r/o`, `__main__`)
- Test: `tests/test_bestresponse_run.py`

**Interfaces:**
- Consumes: `best_response_sweep` (Task 2), `reoptimize_node` (Task 3), all imported helpers.
- Produces: `_run(config, parallelism, *, order, response, method_name, options_key, log_on_file=False, disable_plotting=False) -> str`; `run_br_s/run_br_r/run_br_o(config, parallelism, log_on_file=False, disable_plotting=False) -> str`. Each writes `obj.csv` (column = its method name), `runtime.csv`, `termination_condition.csv`, `LSP_solution.csv`, `LSPc_solution.csv`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_bestresponse_run.py`:

```python
import numpy as np
import networkx as nx

import decentralized_bestresponse as mabr


def test_run_br_o_uses_fixed_replicas_and_does_not_mutate_options(tmp_path, monkeypatch):
  seen = {}
  base_data = {None: {"Nn": {None: 1}, "Nf": {None: 1}, "neighborhood": {(1, 1): 0}}}

  monkeypatch.setattr(mabr, "init_problem",
    lambda *a, **k: (base_data, {}, [], nx.empty_graph(1)))
  monkeypatch.setattr(mabr, "load_solution",
    lambda folder, model: ("s", "rep", "fwd", None, None), raising=False)
  monkeypatch.setattr(mabr, "encode_solution",
    lambda Nn, Nf, s, d, r, t: (None, None, None, np.array([[3]]), None), raising=False)
  monkeypatch.setattr(mabr, "LSP", lambda: "LSP")
  monkeypatch.setattr(mabr, "LSP_fixedr", lambda: "LSP_fixedr", raising=False)

  def _solve_subproblem(sp_data, agents, sp, *a):
    seen["sp"] = sp
    seen["r_bar"] = dict(sp_data[None].get("r_bar", {}))
    return (sp_data, np.zeros((1, 1)), None, None, np.zeros((1, 1)),
            np.ones((1, 1)), np.zeros((1,)), np.zeros((1, 1)),
            {"tot": 0.0}, {"tot": "ok"}, {"tot": 0.0})

  monkeypatch.setattr(mabr, "solve_subproblem", _solve_subproblem)
  monkeypatch.setattr(mabr, "get_current_load", lambda *a: {})
  monkeypatch.setattr(mabr, "update_data", lambda data, u: data)
  monkeypatch.setattr(mabr, "compute_residual_capacity",
    lambda *a: (np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))))
  monkeypatch.setattr(mabr, "best_response_sweep",
    lambda *a, **k: (np.zeros((1, 1, 1)),
                     __import__("pandas").DataFrame({"i": [], "j": [], "f": []}),
                     0, 0.0, 0.0))
  monkeypatch.setattr(mabr, "combine_solutions",
    lambda *a: {"sp": {"x": np.zeros((1, 1)), "y": np.zeros((1, 1, 1)),
                       "z": np.zeros((1, 1)), "r": np.ones((1, 1)), "U": np.zeros((1, 1))}})
  monkeypatch.setattr(mabr, "compute_centralized_objective", lambda *a: 1.0)
  monkeypatch.setattr(mabr, "check_feasibility", lambda *a: (True, "ok"))
  monkeypatch.setattr(mabr, "decode_solutions",
    lambda sp_data, sol, comp, arg: (comp, None, 1.0))
  monkeypatch.setattr(mabr, "join_complete_solution", lambda comp: ({}, {}, {}))
  monkeypatch.setattr(mabr, "save_checkpoint", lambda *a: None)
  monkeypatch.setattr(mabr, "save_solution", lambda *a: None)

  config = {
    "base_solution_folder": str(tmp_path),
    "seed": 1,
    "limits": {"load": {"trace_type": "fixed_sum"}},
    "solver_name": "mock",
    "solver_options": {"general": {"TimeLimit": 10},
                       "br_o": {}},
    "max_iterations": 1, "patience": 1, "max_steps": 1,
    "min_run_time": 0, "max_run_time": 0, "run_time_step": 1,
    "checkpoint_interval": 1, "verbose": 0,
    "opt_solution_folder": "centralized-folder",
  }

  mabr.run_br_o(config, parallelism=0, disable_plotting=True)

  assert seen["sp"] == "LSP_fixedr"
  assert seen["r_bar"] == {(1, 1): 3}
  assert "latency_weight" not in config["solver_options"]["br_o"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_bestresponse_run.py -v`
Expected: FAIL — `AttributeError: module 'decentralized_bestresponse' has no attribute 'run_br_o'`.

- [ ] **Step 3: Append `parse_arguments`, `_run`, entry points, `__main__`**

First copy `parse_arguments` (lines 234–253), `run` (lines 256–500), and `__main__` (lines 503–506) **verbatim** from `decentralized_diffusion.py` into the end of `decentralized_bestresponse.py`. Then apply exactly these edits:

**Edit 3a** — in `parse_arguments`, change the description and add a `--variant` argument (insert the new argument just before `return parser.parse_known_args()[0]`):

```python
  parser = argparse.ArgumentParser(
    description="Run FaaS-MABR",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
```
```python
  parser.add_argument(
    "--variant", choices=["s", "r", "o"], default="s",
    help="FaaS-MABR variant: s (sequential), r (randomized), o (re-optimization)",
  )
  return parser.parse_known_args()[0]
```

**Edit 3b** — rename `def run(` to `def _run(` and replace its signature with:

```python
def _run(
    config: dict,
    parallelism: int,
    *,
    order: str,
    response: str,
    method_name: str,
    options_key: str,
    log_on_file: bool = False,
    disable_plotting: bool = False,
  ) -> str:
```

**Edit 3c** — replace the diffusion options block:

```python
  diffusion_options = dict(solver_options["diffusion"])
  diffusion_options.setdefault(
    "unit_bids", solver_options.get("auction", {}).get("unit_bids", False)
  )
```
with the MABR options block + RNG:

```python
  br_options = dict(solver_options[options_key])
  br_options.setdefault("latency_weight", 0.0)
  br_options.setdefault("fairness_weight", 0.0)
  rng = np.random.default_rng(seed)
```

**Edit 3d** — replace the whole coordinate block (from `s = datetime.now()` immediately above `bids, memory_bids, n_auctions = define_assignments(` down to the end of the `start_additional_replicas` block, i.e. `decentralized_diffusion.py` lines 357–410):

```python
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
          last_y=y,
          diffusion_options=diffusion_options,
          latency=latency,
          fairness=fairness,
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
```

with the MABR sweep block:

```python
      reopt_fn = None
      if response == "reopt":
        def reopt_fn(node, omega_ub_row):
          return reoptimize_node(
            node, omega_ub_row, sp_data, solver_name,
            general_solver_options, parallelism,
            use_fixed_r=(opt_solution is not None),
          )
      s = datetime.now()
      sweep_y, memory_bids, n_active, placed_total, reopt_runtime = best_response_sweep(
        omega, residual_capacity, sp_data, neighborhood, sp_rho,
        br_options, latency, fairness,
        force_memory_bids=(
          (sp_rho > 0).any()
          and len(n_accepted_queue) >= n_accepted_queue.maxlen
          and all(x == n_accepted_queue[0] for x in n_accepted_queue)
        ),
        order=order, response=response, rng=rng, reopt_fn=reopt_fn,
      )
      rt = (datetime.now() - s).total_seconds()
      total_runtime += (rt / n_active) if n_active else rt
      total_runtime += reopt_runtime
      rmp_omega = np.zeros((Nn, Nf))
      additional_replicas = np.zeros((Nn, Nf))
      if placed_total > 0:
        y += sweep_y
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
      mabr_no_progress = (placed_total == 0 and len(memory_bids) == 0)
```

**Edit 3e** — replace the `check_stopping_criteria` call (pass `None` for `bids`) and add the MABR guard. Change:

```python
      stop_searching, why_stop_searching = check_stopping_criteria(
        it, max_iterations, blackboard, omega, rmp_omega,
        additional_replicas, bids, memory_bids,
        tolerance, total_runtime, time_limit
      )
```
to:
```python
      stop_searching, why_stop_searching = check_stopping_criteria(
        it, max_iterations, blackboard, omega, rmp_omega,
        additional_replicas, None, memory_bids,
        tolerance, total_runtime, time_limit
      )
      if not stop_searching and mabr_no_progress:
        stop_searching = True
        why_stop_searching = "no best-response progress"
```

**Edit 3f** — change the `obj.csv` column header to the method name:

```python
  pd.DataFrame(obj_dict["LSPr_final"], columns=[method_name]).to_csv(
    os.path.join(solution_folder, "obj.csv"), index=False
  )
```

**Edit 3g** — append the three entry points and replace the `__main__` block:

```python
def run_br_s(config, parallelism, log_on_file=False, disable_plotting=False):
  return _run(config, parallelism, order="fixed", response="greedy",
              method_name="FaaS-MABR-S", options_key="br_s",
              log_on_file=log_on_file, disable_plotting=disable_plotting)


def run_br_r(config, parallelism, log_on_file=False, disable_plotting=False):
  return _run(config, parallelism, order="random", response="greedy",
              method_name="FaaS-MABR-R", options_key="br_r",
              log_on_file=log_on_file, disable_plotting=disable_plotting)


def run_br_o(config, parallelism, log_on_file=False, disable_plotting=False):
  return _run(config, parallelism, order="fixed", response="reopt",
              method_name="FaaS-MABR-O", options_key="br_o",
              log_on_file=log_on_file, disable_plotting=disable_plotting)


if __name__ == "__main__":
  args = parse_arguments()
  config = load_configuration(args.config)
  runner = {"s": run_br_s, "r": run_br_r, "o": run_br_o}[args.variant]
  runner(config, args.parallelism, disable_plotting=args.disable_plotting)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_bestresponse_run.py -v`
Expected: PASS.

- [ ] **Step 5: Confirm earlier tests still pass and the module imports cleanly**

Run: `uv run pytest tests/test_bestresponse_helpers.py tests/test_bestresponse_run.py -v && uv run python -c "import decentralized_bestresponse as m; print('ok', callable(m.run_br_s), callable(m.run_br_r), callable(m.run_br_o))"`
Expected: tests PASS; prints `ok True True True`.

- [ ] **Step 6: Commit**

```bash
git add decentralized_bestresponse.py tests/test_bestresponse_run.py
git commit -m "feat: add FaaS-MABR _run loop and run_br_s/r/o entry points"
```

---

### Task 5: Additive wiring (run.py, compare_results.py, planar config)

**Files:**
- Modify: `run.py`
- Modify: `compare_results.py`
- Modify: `config_files/planar_comparison.json`
- Test: `tests/test_bestresponse_wiring.py`

**Interfaces:**
- Consumes: `decentralized_bestresponse.run_br_s/r/o` (Task 4).
- Produces: `run.run_br_s/r/o` symbols; `faas-br-s/-r/-o` accepted in `--methods`; `FaaS-MABR-S/-R/-O` in `compare_results` default set and palettes; `solver_options.{br_s,br_r,br_o}` in the planar config.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bestresponse_wiring.py`:

```python
import json
from pathlib import Path

import run


def test_methods_choice_accepts_faas_br(monkeypatch):
  argv = ["run.py", "-c", "config_files/planar_comparison.json",
          "--methods", "faas-br-s", "faas-br-r", "faas-br-o"]
  monkeypatch.setattr("sys.argv", argv)
  args = run.parse_arguments()
  assert {"faas-br-s", "faas-br-r", "faas-br-o"}.issubset(set(args.methods))


def test_run_module_exposes_br_runners():
  for name in ("run_br_s", "run_br_r", "run_br_o"):
    assert callable(getattr(run, name))


def test_planar_config_has_br_blocks():
  config = json.loads(Path("config_files/planar_comparison.json").read_text())
  so = config["solver_options"]
  assert "br_s" in so and "br_r" in so and "br_o" in so


def test_compare_results_palette_includes_mabr():
  import inspect
  import compare_results
  src = inspect.getsource(compare_results)
  assert '"FaaS-MABR-S"' in src and '"FaaS-MABR-R"' in src and '"FaaS-MABR-O"' in src


def test_compare_results_defaults_include_mabr(monkeypatch):
  monkeypatch.setattr("sys.argv", ["compare_results.py", "-i", "solutions/demo"])
  import compare_results
  args = compare_results.parse_arguments()
  assert {"FaaS-MABR-S", "FaaS-MABR-R", "FaaS-MABR-O"}.issubset(set(args.models))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_bestresponse_wiring.py -v`
Expected: FAIL — `faas-br-s` not in argparse choices / `run` has no `run_br_s` / config has no `br_s` / palette missing.

- [ ] **Step 3a: `run.py` import** — after `from decentralized_powerd import run as run_powerd`:

```python
from decentralized_bestresponse import (
  run_br_s as run_br_s,
  run_br_r as run_br_r,
  run_br_o as run_br_o,
)
```

- [ ] **Step 3b: `run.py` `--methods` choices** — add the three keys after `"faas-powd",`:

```python
      "faas-powd",
      "faas-br-s",
      "faas-br-r",
      "faas-br-o",
      "generate_only"
```

- [ ] **Step 3c: `run.py` `solution_folders` template** — add three keys (note the comma after `"faas-powd": []`):

```python
    "faas-diffuse": [],
    "faas-powd": [],
    "faas-br-s": [],
    "faas-br-r": [],
    "faas-br-o": []
  }
```

- [ ] **Step 3d: `run.py` flag init** — after the `run_p = False` line:

```python
    run_brs = False # -- faas-br-s (FaaS-MABR-S)
    run_brr = False # -- faas-br-r (FaaS-MABR-R)
    run_bro = False # -- faas-br-o (FaaS-MABR-O)
```

- [ ] **Step 3e: `run.py` resume-time checks** — after the `faas-powd` resume block (ending `run_p = True`):

```python
      if (not generate_only and "faas-br-s" in methods) and ((
          len(solution_folders.get("faas-br-s", [])) <= experiment_idx
        ) or (
          solution_folders["faas-br-s"][experiment_idx] is None
        )):
        run_brs = True
      if (not generate_only and "faas-br-r" in methods) and ((
          len(solution_folders.get("faas-br-r", [])) <= experiment_idx
        ) or (
          solution_folders["faas-br-r"][experiment_idx] is None
        )):
        run_brr = True
      if (not generate_only and "faas-br-o" in methods) and ((
          len(solution_folders.get("faas-br-o", [])) <= experiment_idx
        ) or (
          solution_folders["faas-br-o"][experiment_idx] is None
        )):
        run_bro = True
```

- [ ] **Step 3f: `run.py` `except ValueError` branch** — after `run_p = "faas-powd" in methods`:

```python
      run_brs = "faas-br-s" in methods
      run_brr = "faas-br-r" in methods
      run_bro = "faas-br-o" in methods
```

- [ ] **Step 3g: `run.py` run-guard** — change:

```python
    if run_c or run_i or run_i_v0 or run_a or run_h or run_d or run_p or generate_only:
```
to:
```python
    if run_c or run_i or run_i_v0 or run_a or run_h or run_d or run_p or run_brs or run_brr or run_bro or generate_only:
```

- [ ] **Step 3h: `run.py` dispatch blocks** — after the `faas-powd` dispatch block (ending its `set_solution_folder(..., "faas-powd", ...)`):

```python
      # -- solve best-response sequential (FaaS-MABR-S)
      if run_brs:
        brs_folder = run_br_s(
          config, sp_parallelism,
          log_on_file = log_on_file, disable_plotting = disable_plotting
        )
        set_solution_folder(
          solution_folders, "faas-br-s", experiment_idx, brs_folder
        )
      # -- solve best-response randomized (FaaS-MABR-R)
      if run_brr:
        brr_folder = run_br_r(
          config, sp_parallelism,
          log_on_file = log_on_file, disable_plotting = disable_plotting
        )
        set_solution_folder(
          solution_folders, "faas-br-r", experiment_idx, brr_folder
        )
      # -- solve best-response re-optimization (FaaS-MABR-O)
      if run_bro:
        bro_folder = run_br_o(
          config, sp_parallelism,
          log_on_file = log_on_file, disable_plotting = disable_plotting
        )
        set_solution_folder(
          solution_folders, "faas-br-o", experiment_idx, bro_folder
        )
```

- [ ] **Step 3i: `run.py` `mname` ternary** — replace the innermost branch:

```python
                "FaaS-MADiG" if method == "faas-diffuse" else (
                  "FaaS-MAPoD" if method == "faas-powd" else "HierarchicalAuction"
                )
```
with:
```python
                "FaaS-MADiG" if method == "faas-diffuse" else (
                  "FaaS-MAPoD" if method == "faas-powd" else (
                    "FaaS-MABR-S" if method == "faas-br-s" else (
                      "FaaS-MABR-R" if method == "faas-br-r" else (
                        "FaaS-MABR-O" if method == "faas-br-o" else "HierarchicalAuction"
                      )
                    )
                  )
                )
```
(`mkey` needs no change: none of `faas-br-*` start with `faas-macro`, so they resolve to `"LSPc"`.)

- [ ] **Step 3j: `run.py` `method_colors`** — extend the list (add a comma after `tab:brown` and three new entries):

```python
      mcolors.TABLEAU_COLORS["tab:brown"],
      mcolors.TABLEAU_COLORS["tab:olive"],
      mcolors.TABLEAU_COLORS["tab:cyan"],
      mcolors.TABLEAU_COLORS["tab:gray"]
    ]
```

- [ ] **Step 3k: `compare_results.py` default model set** — change line ~55:

```python
    default = ["LoadManagementModel", "FaaS-MACrO", "FaaS-MADeA", "FaaS-MADiG", "FaaS-MAPoD"]
```
to:
```python
    default = ["LoadManagementModel", "FaaS-MACrO", "FaaS-MADeA", "FaaS-MADiG", "FaaS-MAPoD", "FaaS-MABR-S", "FaaS-MABR-R", "FaaS-MABR-O"]
```

- [ ] **Step 3l: `compare_results.py` both palettes** — in `plot_by_key` and `violinplot_by_key`, add three entries after the `"FaaS-MAPoD"` line (add a comma after it):

```python
    "FaaS-MAPoD": mcolors.CSS4_COLORS["khaki"],
    "FaaS-MABR-S": mcolors.CSS4_COLORS["mediumaquamarine"],
    "FaaS-MABR-R": mcolors.CSS4_COLORS["sandybrown"],
    "FaaS-MABR-O": mcolors.CSS4_COLORS["mediumpurple"]
```

- [ ] **Step 3m: `config_files/planar_comparison.json`** — after the `powerd` block, add a comma and the three blocks:

```json
    "powerd": {
      "d": 2,
      "criterion": "score",
      "latency_weight": 0.0,
      "fairness_weight": 0.0,
      "unit_bids": false
    },
    "br_s": { "latency_weight": 0.0, "fairness_weight": 0.0 },
    "br_r": { "latency_weight": 0.0, "fairness_weight": 0.0 },
    "br_o": { "latency_weight": 0.0, "fairness_weight": 0.0 }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_bestresponse_wiring.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Verify JSON + imports + no regression of existing wiring**

Run: `uv run python -c "import json; json.load(open('config_files/planar_comparison.json')); import run, compare_results; print('imports ok')"`
Expected: prints `imports ok`.

- [ ] **Step 6: Commit**

```bash
git add run.py compare_results.py config_files/planar_comparison.json tests/test_bestresponse_wiring.py
git commit -m "feat: wire FaaS-MABR-S/R/O into run.py, compare_results, planar config"
```

---

### Task 6: End-to-end smoke + reproducibility (Gurobi-gated)

**Files:**
- Test: `tests/test_bestresponse_e2e.py`

**Interfaces:**
- Consumes: `decentralized_bestresponse.run_br_s/r/o` (Task 4).
- Produces: none (test-only verification gate).

- [ ] **Step 1: Write the e2e test**

Create `tests/test_bestresponse_e2e.py`:

```python
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest

from decentralized_bestresponse import run_br_s, run_br_r, run_br_o


def _require_gurobi() -> None:
  solver = pyo.SolverFactory("gurobi")
  if not solver.available(exception_flag=False):
    pytest.skip("Gurobi solver is not available")


def _e2e_config(base_solution_folder: Path) -> dict:
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
      "load": {"trace_type": "clipped",
               "min": {"min": 2.0, "max": 2.0},
               "max": {"min": 3.0, "max": 3.0}},
    },
    "solver_name": "gurobi",
    "solver_options": {
      "general": {"TimeLimit": 60, "OutputFlag": 0},
      "br_s": {"latency_weight": 0.0, "fairness_weight": 0.0},
      "br_r": {"latency_weight": 0.0, "fairness_weight": 0.0},
      "br_o": {"latency_weight": 0.0, "fairness_weight": 0.0},
    },
    "max_iterations": 2, "patience": 1, "max_steps": 8,
    "min_run_time": 1, "max_run_time": 1, "run_time_step": 1,
    "checkpoint_interval": 1, "tolerance": 1e-6, "verbose": 0,
  }


def _assert_artifacts(folder, column):
  obj = pd.read_csv(Path(folder, "obj.csv"))
  assert column in obj.columns
  assert len(obj) >= 1
  assert np.isfinite(pd.to_numeric(obj[column], errors="coerce")).all()
  runtime = pd.read_csv(Path(folder, "runtime.csv"))
  assert "tot" in runtime.columns and (runtime["tot"] >= 0).all()
  assert Path(folder, "termination_condition.csv").exists()
  assert Path(folder, "LSPc_solution.csv").exists()


def test_br_s_artifacts(tmp_path):
  _require_gurobi()
  _assert_artifacts(
    run_br_s(_e2e_config(tmp_path), parallelism=0, disable_plotting=True),
    "FaaS-MABR-S")


def test_br_r_artifacts(tmp_path):
  _require_gurobi()
  _assert_artifacts(
    run_br_r(_e2e_config(tmp_path), parallelism=0, disable_plotting=True),
    "FaaS-MABR-R")


def test_br_o_artifacts(tmp_path):
  _require_gurobi()
  _assert_artifacts(
    run_br_o(_e2e_config(tmp_path), parallelism=0, disable_plotting=True),
    "FaaS-MABR-O")


def test_br_s_reproducible(tmp_path):
  _require_gurobi()
  fa = run_br_s(_e2e_config(tmp_path / "a"), parallelism=0, disable_plotting=True)
  fb = run_br_s(_e2e_config(tmp_path / "b"), parallelism=0, disable_plotting=True)
  oa = pd.read_csv(Path(fa, "obj.csv"))["FaaS-MABR-S"].to_numpy()
  ob = pd.read_csv(Path(fb, "obj.csv"))["FaaS-MABR-S"].to_numpy()
  assert oa.shape == ob.shape and oa.shape[0] >= 1
  assert np.allclose(oa, ob)
```

- [ ] **Step 2: Run the e2e test**

Run: `uv run pytest tests/test_bestresponse_e2e.py -v`
Expected: PASS if Gurobi available, else SKIP.

- [ ] **Step 3: Commit**

```bash
git add tests/test_bestresponse_e2e.py
git commit -m "test: add FaaS-MABR e2e smoke + reproducibility (Gurobi-gated)"
```

---

### Task 7: LaTeX note `faas-bestresponse-note/`

**Files:**
- Create: `faas-bestresponse-note/faas-mabr.tex`
- Create: `faas-bestresponse-note/main.tex`
- Create: `faas-bestresponse-note/references.bib`
- Create: `faas-bestresponse-note/README.md`
- Create: `faas-bestresponse-note/.gitignore`

**Interfaces:**
- Consumes: the committed `faas-mapod-note/` files as templates (notation recap, preview wrapper, gitignore).
- Produces: a standalone, compilable LaTeX note for the FaaS-MABR family.

- [ ] **Step 1: Scaffold from the sibling note**

```bash
mkdir -p faas-bestresponse-note
cp faas-mapod-note/.gitignore faas-bestresponse-note/.gitignore
```

- [ ] **Step 2: Create `faas-bestresponse-note/references.bib`**

```bibtex
% References for the FaaS-MABR positioning subsection (shared, verified set
% plus the Gauss-Seidel anchor).

@book{bertsekastsitsiklis1989,
  author    = {Dimitri P. Bertsekas and John N. Tsitsiklis},
  title     = {Parallel and Distributed Computation: Numerical Methods},
  publisher = {Prentice-Hall},
  year      = {1989},
  address   = {Englewood Cliffs, NJ}
}

@article{bertsekas1988,
  author  = {Dimitri P. Bertsekas},
  title   = {The auction algorithm: A distributed relaxation method for the assignment problem},
  journal = {Annals of Operations Research},
  volume  = {14},
  number  = {1},
  pages   = {105--123},
  year    = {1988},
  doi     = {10.1007/BF02186476}
}

@article{cybenko1989,
  author  = {George Cybenko},
  title   = {Dynamic load balancing for distributed memory multiprocessors},
  journal = {Journal of Parallel and Distributed Computing},
  volume  = {7},
  number  = {2},
  pages   = {279--301},
  year    = {1989},
  doi     = {10.1016/0743-7315(89)90021-X}
}
```

- [ ] **Step 3: Create `faas-bestresponse-note/main.tex`**

```latex
% Standalone preview wrapper for the FaaS-MABR note.
\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath, amssymb}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage[numbers,square]{natbib}
\usepackage[margin=1in]{geometry}

\begin{document}
\input{faas-mabr}
\bibliographystyle{plainnat}
\bibliography{references}
\end{document}
```

- [ ] **Step 4: Create the notation recap block**

Copy the self-contained "Notation and capacity model" subsection from `faas-mapod-note/faas-mapod.tex` (the block between `% BEGIN self-contained notation recap` and `% END self-contained notation recap`) into a scratch buffer, changing only: relabel `sec:mapod-notation` → `sec:mabr-notation` and `tab:mapod-notation` → `tab:mabr-notation`; in the table's "Parameters and weights" group, replace the two FaaS-MAPoD rows (`Sample size` and `Selection criterion`) with one row:
```
Offloading cap & $\omega^{\mathrm{ub}}_i$ & per-function cap on $\omega_i$ in the capped local best response (FaaS-MABR-O).\\
```

- [ ] **Step 5: Create `faas-bestresponse-note/faas-mabr.tex`**

```latex
% =====================================================================
% FaaS-MABR: distributed best-response (Gauss-Seidel) heuristics.
% Meant to be \input{} (or pasted) into the paper. Assumes the paper's
% notation (N, F, C_i^f, rho_i, omega_i^f, beta_{ij}^f, gamma_i^f, ...).
% Cross-references are plain text; convert to \ref{} in the host paper.
% =====================================================================

\section{FaaS-MABR: sequential best-response (Gauss-Seidel) variants}
\label{sec:mabr}

\textsc{FaaS-MABR} (Multi-Agent Best-Response) is the \emph{sequential}
counterpart of the \emph{simultaneous} coordination used by \textsc{FaaS-MADiG}
and \textsc{FaaS-MAPoD}. The auction and its price-free diffusion ablations
compute every overloaded node's request against a single snapshot of advertised
capacity and then resolve the resulting cross-node conflicts (a Jacobi-style
round followed by seller-side clearing). \textsc{FaaS-MABR} instead visits nodes
in an order and lets each node claim capacity on a \emph{shared residual-capacity
ledger that is decremented in place}, so a node responds to what earlier nodes
have already taken. This is the Gauss-Seidel analogue of the same coordination
problem, and because updates are sequential there are no cross-node conflicts and
\emph{no seller-clearing phase}. Everything else is inherited from
\textsc{FaaS-MADiG}: the local problem~(P2) that fixes $x_i^f$, $r_i^f$, and the
residual demand $\omega_i^f$; the price-free score~$s_{ij}^f$ and the convenience
threshold $\gamma_i^f$; the price-free replica expansion of Algorithm~4; and the
restricted re-solve after each round.

% PASTE HERE the notation recap subsection prepared in Step 4
% (\subsection{Notation and capacity model} with table tab:mabr-notation,
%  including the omega^ub row).

\subsection{Sequential best-response sweep}
\label{sec:mabr-sweep}

At iteration $h$ the coordination is a single \emph{sweep} over the nodes. Let
$\tilde C_j^f$ be the shared ledger, initialized to the true residual capacity
$C_j^f(h)$ and decremented in place during the sweep. Visiting the nodes in the
order $\sigma$ (a fixed index order or a per-sweep random permutation), each node
$i$, for every $f$ with $\omega_i^f(h)>0$, admits the candidate sellers
\begin{equation}
  A_i^f(h)=\{\,j\in N_i : \tilde C_j^f \ge 1 \ \text{and}\ s_{ij}^f(h)>-\gamma_i^f\,\},
  \qquad
  s_{ij}^f(h)=\beta_{ij}^f-w_{\mathrm{lat}}L_{ij}-w_{\mathrm{fair}}\phi_i^f(h),
  \label{eq:mabr-score}
\end{equation}
and claims their capacity greedily in descending score order (tie-break: lower
node id), decrementing $\tilde C_{j}^f$ as it claims, until $\omega_i^f(h)$ is met
or $A_i^f(h)$ is exhausted. The immediate decrement is what distinguishes
Gauss-Seidel from the Jacobi snapshot of \textsc{FaaS-MADiG}: in a fixed-order
sweep a node that acts earlier can consume capacity that a later node would
otherwise have seen. Unplaced demand triggers the same price-free
replica-expansion requests to neighbours with memory slack $\rho_j(h)>0$ as in
\textsc{FaaS-MADiG} (the two-block emission), collected over the whole sweep and
served by Algorithm~4 afterwards. The inter-sweep restricted re-solve makes the
overall scheme a Gauss-Seidel relaxation: each sweep best-responds to the state
left by the previous sweep's re-solve, and the loop stops at a fixed point (a
sweep that places no new load and emits no expansion requests).

\subsection{The three variants}
\label{sec:mabr-variants}

\textsc{FaaS-MABR} spans two axes.

\paragraph{Node order.} \textsc{FaaS-MABR-S} sweeps in a fixed ascending node
order (deterministic, reproducible). \textsc{FaaS-MABR-R} draws a fresh random
node permutation each sweep from a per-run generator seeded once; statistical
variance is obtained by repeating experiments, as for \textsc{FaaS-MAPoD}.

\paragraph{Response type.} \textsc{FaaS-MABR-S}/\textsc{-R} use a \emph{sequential
greedy} response: the node places its residual demand on the ledger as above.
These are Gauss-Seidel greedy responses, not optimal best responses.
\textsc{FaaS-MABR-O} uses a \emph{capped local best response}: before placing,
node $i$ re-solves its local subproblem with the horizontal offloading capped at
the capacity its neighbours currently still advertise,
\begin{equation}
  \omega_i^f \le \omega^{\mathrm{ub}}_{i,f}, \qquad
  \omega^{\mathrm{ub}}_{i,f}=\sum_{j\in N_i}\tilde C_j^f ,
  \label{eq:mabr-cap}
\end{equation}
so it can reduce its horizontal demand and leave the rest to local execution or
Cloud forwarding in the subsequent restricted solve. Only the re-optimized
$\omega_i$ is committed to the sweep before placement; the capped probe's
$x/z/r$ are diagnostic, and the final feasible state is produced by the
restricted re-solve with fixed $y$. (The fourth combination, randomized-order
re-optimization, is not used.)

\begin{algorithm}
\caption{\textsc{FaaS-MABR} --- one sequential best-response sweep}
\label{alg:mabr-sweep}
\begin{algorithmic}[1]
\Require ledger $\tilde C\gets C(h)$, residual demand $\omega(h)$, order $\sigma$, response $\in\{\text{greedy},\text{reopt}\}$
\For{each node $i$ in order $\sigma$}
  \If{response $=$ reopt}
    \State $\omega^{\mathrm{ub}}_{i,f}\gets\sum_{j\in N_i}\tilde C_j^f$;\quad re-solve node $i$'s capped local problem; commit only $\omega_i$
  \EndIf
  \For{each $f$ with $\omega_i^f(h)>0$}
    \State $A_i^f(h)\gets\{\,j\in N_i:\tilde C_j^f\ge1,\ s_{ij}^f(h)>-\gamma_i^f\,\}$
    \State place $\omega_i^f(h)$ greedily by descending $s_{ij}^f(h)$ (tie-break lower id), decrementing $\tilde C_j^f$ in place
    \If{demand remains} emit price-free replica-expansion requests to neighbours with $\rho_j(h)>0$ \EndIf
  \EndFor
\EndFor
\State after the sweep: start additional replicas (Alg.~4); re-solve the restricted problem with fixed $y$
\end{algorithmic}
\end{algorithm}

\subsection{What is kept and what changes}
\label{sec:mabr-ablation}

\noindent\textbf{Kept} (identical to \textsc{FaaS-MADiG}): the local
problem~(P2); the advertised residual capacity and memory slack
(Eqs.~(12)--(13)); the price-free score~\eqref{eq:mabr-score} and convenience
threshold $\gamma_i^f$; the fairness penalty $\phi_i^f$; the price-free replica
expansion of Algorithm~4; and the restricted re-solve.

\noindent\textbf{Changed}: simultaneous (Jacobi) coordination becomes sequential
(Gauss-Seidel) with an in-place-decremented ledger; the seller-clearing phase is
removed (sequential updates produce no conflicts); and \textsc{FaaS-MABR-O} adds
the anticipatory capped local re-optimization of Eq.~\eqref{eq:mabr-cap}.

\subsection{Positioning with respect to the literature}
\label{sec:mabr-related}

We state plainly that \textsc{FaaS-MABR} instantiates a classical
distributed-optimization pattern and is not claimed as novel. The sequential,
in-place-update sweep is the \emph{Gauss-Seidel} relaxation, and the simultaneous
snapshot of \textsc{FaaS-MADiG}/\textsc{FaaS-MAPoD} is its \emph{Jacobi}
counterpart; both are textbook iterative schemes for distributed fixed-point
computation~\citep{bertsekastsitsiklis1989}. \textsc{FaaS-MABR-S}/\textsc{-R} are
sequential greedy responses (better-response, not exact best-response) dynamics,
while \textsc{FaaS-MABR-O} computes a per-node capped best response. The
underlying push of excess load to less-loaded neighbours remains the diffusion
paradigm of Cybenko~\citep{cybenko1989}, and the relation to the auction is the
familiar one between Gauss-Seidel relaxation and Bertsekas-style coordinate
methods~\citep{bertsekas1988}. The value of the family is therefore the
\emph{sequential-vs-simultaneous} contrast with \textsc{FaaS-MADiG} and the
re-optimization ablation of \textsc{FaaS-MABR-O}, not a new heuristic.
```

(After pasting, replace the Step-4 comment line with the actual notation recap subsection.)

- [ ] **Step 6: Create `faas-bestresponse-note/README.md`**

```markdown
# FaaS-MABR note

A paper-ready LaTeX section explaining the **FaaS-MABR** family — three
sequential (Gauss-Seidel) best-response heuristics (S: fixed-order greedy, R:
randomized-order greedy, O: capped local best response) — in the notation of
`Decentralized_FaaS_coordination.pdf`.

## Files
- `faas-mabr.tex` — the `\section{}` to `\input{}` (or paste) into the paper.
  Remove the self-contained "Notation and capacity model" subsection on
  insertion.
- `main.tex` — standalone preview wrapper.
- `references.bib` — cited works (Bertsekas & Tsitsiklis 1989 Gauss-Seidel
  anchor; Cybenko diffusion; Bertsekas auction).
- `.gitignore` — LaTeX build artifacts.

## Build a preview
```bash
cd faas-bestresponse-note
latexmk -pdf main.tex
```

## Insert into the paper
1. `\input{faas-mabr}` (or paste the section).
2. Delete the "Notation and capacity model" subsection.
3. Convert plain-text cross-references to the host paper's `\ref{}` labels.
4. Merge `references.bib` into the paper's bibliography.
```

- [ ] **Step 7: Compile the preview to verify it builds**

Run:
```bash
cd faas-bestresponse-note && latexmk -pdf -interaction=nonstopmode main.tex
```
Expected: `main.pdf` is produced; no LaTeX errors (undefined-citation warnings on the first pass are fine — `latexmk` reruns BibTeX). Then clean: `latexmk -c main.tex`.

- [ ] **Step 8: Commit**

```bash
git add faas-bestresponse-note/faas-mabr.tex faas-bestresponse-note/main.tex \
        faas-bestresponse-note/references.bib faas-bestresponse-note/README.md \
        faas-bestresponse-note/.gitignore
git commit -m "docs: add FaaS-MABR LaTeX note (Gauss-Seidel positioning)"
```

---

## Self-Review

**1. Spec coverage:**
- Part I §3 architecture (module, shared sweep, three entry points, reuse, no `evaluate_assignments`) → Tasks 2, 4. ✅
- §3.1 `best_response_sweep` (ledger, order, greedy, threshold, memory parity, reopt plumbing, return tuple incl. `placed_total`/`reopt_runtime`) → Task 2. ✅
- §3.2 reopt context (single-node `solve_subproblem([i])`, only `omega[i,:]` committed) → Task 3. ✅
- §4 `LSP_capped`/`LSP_capped_fixedr` additive classes → Task 1. ✅
- §3 fixed-point guard ("no best-response progress") + sweep-level replica expansion + `check_stopping_criteria` with `bids=None` → Task 4 (Edits 3d, 3e). ✅
- §5 config blocks + no-mutation copy → Tasks 4, 5. ✅
- §6 additive wiring ×3 (run.py, compare_results both palettes + default set, planar) → Task 5. ✅
- §8 testing (sweep unit incl. order-dependence + stopping signal; `LSP_capped` structural + behavioral via reopt; wiring; e2e ×3 + reproducibility + `-O` fix_r) → Tasks 1,2,3,5,6. ✅
- §2 names/CLI/columns `FaaS-MABR-S/-R/-O`, `faas-br-s/-r/-o`, `LSPc` → Tasks 4,5. ✅
- Part II LaTeX note → Task 7. ✅
- Out of scope (randomized-order reopt; other methods untouched) → respected. ✅

**2. Placeholder scan:** No "TBD"/"TODO"/"handle edge cases". The notation-recap block is copied verbatim from a committed sibling file with one explicitly listed row change (Task 7 Steps 4–5) — concrete, not a placeholder.

**3. Type consistency:** `best_response_sweep(...) -> (y_increment, memory_bids, n_active, placed_total, reopt_runtime)` is produced in Task 2 and unpacked identically in Task 4 (Edit 3d). `reopt_fn(node, omega_ub_row) -> (capped_row, runtime)` is the contract in both Task 2 (consumer) and Task 3 (`reoptimize_node`, the producer the `-O` runner wraps). `LSP_capped`/`LSP_capped_fixedr` (Task 1) are imported and used in Task 3. CLI keys `faas-br-{s,r,o}`, method names `FaaS-MABR-{S,R,O}`, runner symbols `run_br_{s,r,o}`, `mkey="LSPc"`, and `obj.csv` columns are consistent across Tasks 4–6.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-26-faas-mabr-best-response.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session with checkpoints.

Which approach?
