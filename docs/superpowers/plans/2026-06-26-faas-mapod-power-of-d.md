# FaaS-MAPoD Power-of-d Heuristic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add **FaaS-MAPoD**, a randomized power-of-d-choices distributed heuristic that probes only `d` randomly sampled neighbours per offloading step, as a sixth comparable method alongside LMM, FaaS-MACrO, FaaS-MADeA, HierarchicalAuction, and FaaS-MADiG.

**Architecture:** A new standalone runner `decentralized_powerd.py` mirrors `decentralized_diffusion.py` (the FaaS-MADiG runner). It **reuses** `evaluate_assignments` from `decentralized_diffusion` and every helper from `run_faasmadea` **unchanged by import**. It replaces only the buyer side: a new `sample_assignments` function implements iterated power-of-d sampling with a config-selectable `criterion` (`score`|`capacity`). The sampling RNG is seeded once per run from `config["seed"]`; variance is captured by the existing `--n_experiments` aggregation. All wiring into `run.py`, `compare_results.py`, and `config_files/planar_comparison.json` is **additive only**.

**Tech Stack:** Python 3.10, NumPy, pandas, Pyomo (Gurobi/GLPK). 2-space indentation. Run via `uv run`. Tests via `uv run pytest`. LaTeX via `latexmk` (Task 5).

## Global Constraints

- **2-space indentation** throughout (match the existing files).
- **Do NOT modify** functions that support other methods: `run_faasmadea.py`, `run_faasmacro.py`, `run_centralized_model.py`, `models/`, `utils/`, `hierarchical_auction/`, and the existing `define_assignments`/`evaluate_assignments`/`run` in `decentralized_diffusion.py`. Reuse by import. New behaviour → new function in the new file.
- **`run.py` / `compare_results.py` / `planar_comparison.json` changes are ADDITIVE only** — existing methods' code paths must stay byte-for-byte unchanged.
- **Method name** `FaaS-MAPoD`; CLI key `faas-powd`; `obj.csv` column header `FaaS-MAPoD`; artifact family key `LSPc`.
- **`evaluate_assignments` is reused unchanged** by import from `decentralized_diffusion`; the seller clearing stays score-ordered even when `criterion="capacity"`.
- **Config block:** `solver_options.powerd = {"d":2, "criterion":"score", "latency_weight":0.0, "fairness_weight":0.0, "unit_bids":false}`. `unit_bids` falls back to `solver_options["auction"]["unit_bids"]` when absent. `run()` copies the options dict (`dict(...)`) before applying defaults so the input config is never mutated.
- **Reproducibility:** the sampling RNG is `np.random.default_rng(config["seed"])`, created **once** before the time loop. Determinism is verified against two independent generators with the same seed (not two calls on one advanced generator).
- **Out of scope (v1):** Option C (distributed best-response); a fully capacity-ordered seller clearing; per-instance K-seed averaging.

---

### Task 1: `sample_assignments` (randomized power-of-d buyer side)

**Files:**
- Create: `decentralized_powerd.py`
- Test: `tests/test_powerd_helpers.py`

**Interfaces:**
- Consumes: `VAR_TYPE` from `run_faasmadea`; `data[None]["beta"][(i+1,j+1,f+1)]`, `data[None]["gamma"][(i+1,f+1)]` (same dict layout used by `decentralized_diffusion.define_assignments`).
- Produces: `sample_assignments(omega, blackboard, data, neighborhood, rho, powerd_options, latency, fairness, force_memory_bids, rng) -> (pd.DataFrame[i,j,f,d,utility], pd.DataFrame[i,j,f], int)`. The `utility` column always holds the score `s_{ij}^f` (regardless of `criterion`) so the reused `evaluate_assignments` can sort sellers by `(utility, i)`.

- [ ] **Step 1: Write the new module header + the failing test file**

Create `decentralized_powerd.py` with the full import header (everything `run()` will need in Task 2, plus the reused `evaluate_assignments`):

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
  VAR_TYPE,
  check_ls_pr_feasibility_from_fixed_y,
  check_stopping_criteria,
  compute_residual_capacity,
  ensure_memory_sellers,
  neigh_dict_to_matrix,
  start_additional_replicas,
)
from decentralized_diffusion import evaluate_assignments
from utils.centralized import check_feasibility
from utils.faasmacro import compute_centralized_objective
from utils.common import load_configuration
from models.sp import LSP, LSP_fixedr, LSPr

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

Then create `tests/test_powerd_helpers.py`:

```python
import numpy as np
import pandas as pd

from decentralized_powerd import sample_assignments
from decentralized_diffusion import define_assignments


def _base_data(Nn=4, Nf=1):
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


def _full_neighborhood(Nn=4):
  neighborhood = np.ones((Nn, Nn)) - np.eye(Nn)
  return neighborhood


def _opts(d=2, criterion="score", unit_bids=False):
  return {
    "d": d, "criterion": criterion, "unit_bids": unit_bids,
    "latency_weight": 0.0, "fairness_weight": 0.0,
  }


def test_sample_assignments_is_deterministic_given_seed():
  data = _base_data()
  # distinct betas so the random sampling has something to choose between
  data[None]["beta"][(1, 2, 1)] = 3.0
  data[None]["beta"][(1, 3, 1)] = 2.0
  data[None]["beta"][(1, 4, 1)] = 1.5
  omega = np.zeros((4, 1)); omega[0, 0] = 6.0
  blackboard = np.zeros((4, 1)); blackboard[1, 0] = 2; blackboard[2, 0] = 2; blackboard[3, 0] = 2
  rho = np.zeros((4,))
  args = (omega, blackboard, data, _full_neighborhood(), rho,
          _opts(d=2), np.zeros((4, 4)), np.zeros((4, 1)))

  bids1, _, _ = sample_assignments(*args, force_memory_bids=False,
                                   rng=np.random.default_rng(123))
  bids2, _, _ = sample_assignments(*args, force_memory_bids=False,
                                   rng=np.random.default_rng(123))

  pd.testing.assert_frame_equal(bids1, bids2)


def test_sample_assignments_degenerates_to_madig_when_d_covers_candidates():
  # d >= |candidates|, criterion="score", unique scores => same as FaaS-MADiG
  data = _base_data()
  data[None]["beta"][(1, 2, 1)] = 3.0
  data[None]["beta"][(1, 3, 1)] = 2.0
  data[None]["beta"][(1, 4, 1)] = 1.0
  omega = np.zeros((4, 1)); omega[0, 0] = 4.0
  blackboard = np.zeros((4, 1)); blackboard[1, 0] = 2; blackboard[2, 0] = 2; blackboard[3, 0] = 2
  rho = np.zeros((4,))
  common = (omega, blackboard, data, _full_neighborhood(), rho)
  geo = (np.zeros((4, 4)), np.zeros((4, 1)))

  sampled, _, _ = sample_assignments(
    *common, _opts(d=99, criterion="score"), *geo,
    force_memory_bids=False, rng=np.random.default_rng(0))
  greedy, _, _ = define_assignments(
    *common, _opts(d=99, criterion="score"), *geo, force_memory_bids=False)

  pd.testing.assert_frame_equal(
    sampled.reset_index(drop=True), greedy.reset_index(drop=True))


def test_sample_assignments_capacity_criterion_prefers_largest_capacity():
  # score ranks seller 1 best, but seller 2 advertises far more capacity;
  # with d covering both, criterion="capacity" must pick seller 2 first.
  data = _base_data(Nn=3, Nf=1)
  data[None]["beta"][(1, 2, 1)] = 9.0   # seller 1: best score, small capacity
  data[None]["beta"][(1, 3, 1)] = 1.0   # seller 2: worse score, big capacity
  omega = np.zeros((3, 1)); omega[0, 0] = 3.0
  blackboard = np.zeros((3, 1)); blackboard[1, 0] = 1; blackboard[2, 0] = 10
  rho = np.zeros((3,))

  bids, _, _ = sample_assignments(
    omega, blackboard, data, _full_neighborhood(3), rho,
    _opts(d=99, criterion="capacity"), np.zeros((3, 3)), np.zeros((3, 1)),
    force_memory_bids=False, rng=np.random.default_rng(0))

  first = bids.iloc[0]
  assert first["j"] == 2           # largest-capacity seller picked first
  assert first["d"] == 3.0


def test_sample_assignments_threshold_excludes_unconvenient_seller():
  data = _base_data(Nn=2, Nf=1)
  data[None]["beta"][(1, 2, 1)] = 0.0
  data[None]["gamma"][(1, 1)] = 0.05    # score 0.0 > -0.05 holds; flip sign below
  data[None]["beta"][(1, 2, 1)] = -1.0  # score -1.0 <= -0.05 => excluded
  omega = np.zeros((2, 1)); omega[0, 0] = 1.0
  blackboard = np.zeros((2, 1)); blackboard[1, 0] = 5
  rho = np.zeros((2,))

  bids, memory_bids, _ = sample_assignments(
    omega, blackboard, data, np.array([[0.0, 1.0], [1.0, 0.0]]), rho,
    _opts(d=2), np.zeros((2, 2)), np.zeros((2, 1)),
    force_memory_bids=False, rng=np.random.default_rng(0))

  assert len(bids) == 0
  assert len(memory_bids) == 0


def test_sample_assignments_requests_replicas_when_no_capacity_seller():
  data = _base_data(Nn=2, Nf=1)
  omega = np.zeros((2, 1)); omega[0, 0] = 2.0
  blackboard = np.zeros((2, 1))           # no capacity sellers
  rho = np.zeros((2,)); rho[1] = 4.0       # neighbour 1 has spare memory

  bids, memory_bids, _ = sample_assignments(
    omega, blackboard, data, np.array([[0.0, 1.0], [1.0, 0.0]]), rho,
    _opts(d=2), np.zeros((2, 2)), np.zeros((2, 1)),
    force_memory_bids=False, rng=np.random.default_rng(0))

  assert len(bids) == 0
  assert list(memory_bids[["i", "j", "f"]].iloc[0]) == [0, 1, 0]


def test_sample_assignments_unit_bids_emits_one_unit_per_request():
  data = _base_data(Nn=2, Nf=1)
  omega = np.zeros((2, 1)); omega[0, 0] = 3.0
  blackboard = np.zeros((2, 1)); blackboard[1, 0] = 3
  rho = np.zeros((2,))

  bids, _, _ = sample_assignments(
    omega, blackboard, data, np.array([[0.0, 1.0], [1.0, 0.0]]), rho,
    _opts(d=2, unit_bids=True), np.zeros((2, 2)), np.zeros((2, 1)),
    force_memory_bids=False, rng=np.random.default_rng(0))

  assert len(bids) == 3
  assert (bids["d"] == 1).all()
  assert (bids["j"] == 1).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_powerd_helpers.py -v`
Expected: FAIL — `ImportError: cannot import name 'sample_assignments' from 'decentralized_powerd'` (the function does not exist yet).

- [ ] **Step 3: Implement `sample_assignments`**

Append this function to `decentralized_powerd.py` (after the import header):

```python
def sample_assignments(
    omega: np.array,
    blackboard: np.array,
    data: dict,
    neighborhood: np.array,
    rho: np.array,
    powerd_options: dict,
    latency: np.array,
    fairness: np.array,
    force_memory_bids: bool,
    rng: np.random.Generator,
  ) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
  """Power-of-d-choices counterpart of decentralized_diffusion.define_assignments.

  For each overloaded (i, f), instead of scanning the whole candidate
  neighbourhood greedily (FaaS-MADiG), repeatedly probe a random sample of ``d``
  candidate sellers (uniform, without replacement) and pick the best by
  ``criterion``: "score" -> max s_{ij}^f = beta - w_lat*L - w_fair*phi;
  "capacity" -> max advertised residual capacity. Ties break on the lower node
  id for reproducibility. The score is always stored in the ``utility`` column,
  on which the reused evaluate_assignments later sorts the seller side.
  """
  d = int(powerd_options["d"])
  criterion = powerd_options.get("criterion", "score")
  unit_bids = powerd_options.get("unit_bids", False)
  potential_buyers, functions_to_share = np.nonzero(omega)
  bids = {"i": [], "j": [], "f": [], "d": [], "utility": []}
  memory_bids = {"i": [], "j": [], "f": []}
  for i, f in zip(potential_buyers, functions_to_share):
    i = int(i)
    f = int(f)
    potential_sellers = set(np.nonzero(neighborhood[i, :])[0])
    potential_capacity_sellers = potential_sellers.intersection(
      set(np.where(blackboard[:, f] >= 1)[0])
    )
    potential_memory_sellers = potential_sellers.intersection(
      set(np.nonzero(rho)[0])
    )
    score = {}
    candidates = []
    for j in potential_capacity_sellers:
      j = int(j)
      s = (
        data[None]["beta"][(i + 1, j + 1, f + 1)]
        - powerd_options["latency_weight"] * latency[i, j]
        - powerd_options["fairness_weight"] * fairness[i, f]
      )
      if s > - data[None]["gamma"][(i + 1, f + 1)]:
        score[j] = s
        candidates.append(j)
    # buyer-local view of advertised capacity, decremented as we bid, so we
    # never request more from a seller than the buyer has observed locally
    remaining = {j: int(blackboard[j, f]) for j in candidates}
    assigned = 0
    while assigned < omega[i, f] and len(candidates) > 0:
      sample_size = min(d, len(candidates))
      sample = rng.choice(candidates, size=sample_size, replace=False)
      if criterion == "capacity":
        j_star = int(max(sample, key=lambda j: (remaining[int(j)], -int(j))))
      else:
        j_star = int(max(sample, key=lambda j: (score[int(j)], -int(j))))
      if unit_bids:
        q = 1
      else:
        q = VAR_TYPE(min(remaining[j_star], omega[i, f] - assigned))
      bids["i"].append(i)
      bids["f"].append(f)
      bids["j"].append(j_star)
      bids["d"].append(q)
      bids["utility"].append(score[j_star])
      assigned += q
      remaining[j_star] -= q
      if remaining[j_star] < 1:
        candidates.remove(j_star)
    if assigned < omega[i, f] or force_memory_bids:
      for j in potential_memory_sellers - potential_capacity_sellers:
        memory_bids["i"].append(i)
        memory_bids["j"].append(int(j))
        memory_bids["f"].append(f)
  return pd.DataFrame(bids), pd.DataFrame(memory_bids), len(potential_buyers)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_powerd_helpers.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add decentralized_powerd.py tests/test_powerd_helpers.py
git commit -m "feat: add sample_assignments power-of-d buyer rule (FaaS-MAPoD)"
```

---

### Task 2: `run()` for FaaS-MAPoD

**Files:**
- Modify: `decentralized_powerd.py` (append `parse_arguments`, `run`, `__main__`)
- Test: `tests/test_powerd_wiring.py`

**Interfaces:**
- Consumes: `sample_assignments` (Task 1, same file); `evaluate_assignments` (imported from `decentralized_diffusion`); all `run_faasmadea`/`run_faasmacro`/`run_centralized_model` helpers (imported in Task 1's header).
- Produces: `run(config, parallelism, log_on_file=False, disable_plotting=False) -> str`. Writes `obj.csv` (column `FaaS-MAPoD`), `runtime.csv` (column `tot`), `termination_condition.csv`, `LSP_solution.csv`, `LSPc_solution.csv` into a timestamped folder under `config["base_solution_folder"]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_powerd_wiring.py`:

```python
import json
from pathlib import Path

import numpy as np
import networkx as nx

import decentralized_powerd


def test_powerd_run_uses_fixed_replicas_without_mutating_options(tmp_path, monkeypatch):
  seen = {}
  base_data = {
    None: {
      "Nn": {None: 1},
      "Nf": {None: 1},
      "neighborhood": {(1, 1): 0},
    }
  }

  monkeypatch.setattr(
    decentralized_powerd, "init_problem",
    lambda *args, **kwargs: (base_data, {}, [], nx.empty_graph(1)),
  )
  monkeypatch.setattr(
    decentralized_powerd, "load_solution",
    lambda folder, model: ("opt_solution", "opt_replicas", "opt_detailed", None, None),
    raising=False,
  )
  monkeypatch.setattr(
    decentralized_powerd, "encode_solution",
    lambda Nn, Nf, solution, detailed, replicas, t: (None, None, None, np.array([[3]]), None),
    raising=False,
  )
  monkeypatch.setattr(decentralized_powerd, "LSP", lambda: "LSP")
  monkeypatch.setattr(decentralized_powerd, "LSP_fixedr", lambda: "LSP_fixedr", raising=False)

  def _solve_subproblem(sp_data, agents, sp, *args):
    seen["sp"] = sp
    seen["r_bar"] = dict(sp_data[None].get("r_bar", {}))
    return (
      sp_data, np.zeros((1, 1)), None, None, np.zeros((1, 1)),
      np.ones((1, 1)), np.zeros((1,)), np.zeros((1, 1)),
      {"tot": 0.0}, {"tot": "ok"}, {"tot": 0.0},
    )

  monkeypatch.setattr(decentralized_powerd, "solve_subproblem", _solve_subproblem)
  monkeypatch.setattr(decentralized_powerd, "get_current_load", lambda *args: {})
  monkeypatch.setattr(decentralized_powerd, "update_data", lambda data, update: data)
  monkeypatch.setattr(
    decentralized_powerd, "compute_residual_capacity",
    lambda *args: (np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))),
  )
  monkeypatch.setattr(
    decentralized_powerd, "sample_assignments",
    lambda *args, **kwargs: (
      __import__("pandas").DataFrame({"i": [], "j": [], "f": [], "d": [], "utility": []}),
      __import__("pandas").DataFrame({"i": [], "j": [], "f": []}),
      0,
    ),
  )
  monkeypatch.setattr(
    decentralized_powerd, "combine_solutions",
    lambda *args: {"sp": {"x": np.zeros((1, 1)), "y": np.zeros((1, 1, 1)), "z": np.zeros((1, 1)), "r": np.ones((1, 1)), "U": np.zeros((1, 1))}},
  )
  monkeypatch.setattr(decentralized_powerd, "compute_centralized_objective", lambda *args: 1.0)
  monkeypatch.setattr(decentralized_powerd, "check_feasibility", lambda *args: (True, "ok"))
  monkeypatch.setattr(
    decentralized_powerd, "decode_solutions",
    lambda sp_data, solution, complete, arg: (complete, None, 1.0),
  )
  monkeypatch.setattr(
    decentralized_powerd, "join_complete_solution",
    lambda complete: ({}, {}, {}),
  )
  monkeypatch.setattr(decentralized_powerd, "save_checkpoint", lambda *args: None)
  monkeypatch.setattr(decentralized_powerd, "save_solution", lambda *args: None)

  config = {
    "base_solution_folder": str(tmp_path),
    "seed": 1,
    "limits": {"load": {"trace_type": "fixed_sum"}},
    "solver_name": "mock",
    "solver_options": {
      "general": {"TimeLimit": 10},
      "auction": {"unit_bids": True},
      "powerd": {"latency_weight": 0.0, "fairness_weight": 0.0},
    },
    "max_iterations": 1,
    "patience": 1,
    "max_steps": 1,
    "min_run_time": 0,
    "max_run_time": 0,
    "run_time_step": 1,
    "checkpoint_interval": 1,
    "verbose": 0,
    "opt_solution_folder": "centralized-folder",
  }

  decentralized_powerd.run(config, parallelism=0, disable_plotting=True)

  assert seen["sp"] == "LSP_fixedr"
  assert seen["r_bar"] == {(1, 1): 3}
  # run() applied its defaults to a COPY, not the input config
  assert "d" not in config["solver_options"]["powerd"]
  assert "unit_bids" not in config["solver_options"]["powerd"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_powerd_wiring.py::test_powerd_run_uses_fixed_replicas_without_mutating_options -v`
Expected: FAIL — `AttributeError: module 'decentralized_powerd' has no attribute 'run'`.

- [ ] **Step 3: Append `parse_arguments`, `run`, and `__main__`**

Copy `parse_arguments` (lines 234–253), `run` (lines 256–500), and the `__main__` block (lines 503–506) **verbatim** from `decentralized_diffusion.py` into the end of `decentralized_powerd.py`, then apply exactly these five edits (and no others):

**Edit 3a** — in `parse_arguments`, change the description string:

```python
  parser = argparse.ArgumentParser(
    description="Run FaaS-MAPoD",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
```

**Edit 3b** — in `run`, replace the diffusion-options block:

```python
  diffusion_options = dict(solver_options["diffusion"])
  diffusion_options.setdefault(
    "unit_bids", solver_options.get("auction", {}).get("unit_bids", False)
  )
```

with the power-of-d options block plus the RNG:

```python
  powerd_options = dict(solver_options["powerd"])
  powerd_options.setdefault("d", 2)
  powerd_options.setdefault("criterion", "score")
  powerd_options.setdefault("latency_weight", 0.0)
  powerd_options.setdefault("fairness_weight", 0.0)
  powerd_options.setdefault(
    "unit_bids", solver_options.get("auction", {}).get("unit_bids", False)
  )
  rng = np.random.default_rng(seed)
```

**Edit 3c** — in the iteration loop, replace the `define_assignments(...)` call:

```python
      bids, memory_bids, n_auctions = define_assignments(
        omega, blackboard, sp_data, neighborhood, sp_rho,
        diffusion_options, latency, fairness,
        force_memory_bids=(
          (sp_rho > 0).any()
          and len(n_accepted_queue) >= n_accepted_queue.maxlen
          and all(x == n_accepted_queue[0] for x in n_accepted_queue)
        ),
      )
```

with the `sample_assignments(...)` call (note the trailing `rng=rng`):

```python
      bids, memory_bids, n_auctions = sample_assignments(
        omega, blackboard, sp_data, neighborhood, sp_rho,
        powerd_options, latency, fairness,
        force_memory_bids=(
          (sp_rho > 0).any()
          and len(n_accepted_queue) >= n_accepted_queue.maxlen
          and all(x == n_accepted_queue[0] for x in n_accepted_queue)
        ),
        rng=rng,
      )
```

**Edit 3d** — in the `evaluate_assignments(...)` call, pass `powerd_options` instead of `diffusion_options`:

```python
        diffusion_y, additional_replicas, n_sellers = evaluate_assignments(
          bids, residual_capacity, sp_data, ell, sp_r, sp_rho,
          tentatively_start_replicas=(len(memory_bids) == 0),
          last_y=y,
          diffusion_options=powerd_options,
          latency=latency,
          fairness=fairness,
        )
```

**Edit 3e** — change the `obj.csv` column header from `FaaS-MADiG` to `FaaS-MAPoD`:

```python
  pd.DataFrame(obj_dict["LSPr_final"], columns=["FaaS-MAPoD"]).to_csv(
    os.path.join(solution_folder, "obj.csv"), index=False
  )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_powerd_wiring.py::test_powerd_run_uses_fixed_replicas_without_mutating_options -v`
Expected: PASS.

- [ ] **Step 5: Confirm the helper tests still pass and the module imports cleanly**

Run: `uv run pytest tests/test_powerd_helpers.py -v && uv run python -c "import decentralized_powerd; print('ok', callable(decentralized_powerd.run))"`
Expected: helper tests PASS; prints `ok True`.

- [ ] **Step 6: Commit**

```bash
git add decentralized_powerd.py tests/test_powerd_wiring.py
git commit -m "feat: add FaaS-MAPoD run() reusing diffusion seller clearing"
```

---

### Task 3: Additive wiring (run.py, compare_results.py, planar config)

**Files:**
- Modify: `run.py`
- Modify: `compare_results.py`
- Modify: `config_files/planar_comparison.json`
- Test: `tests/test_powerd_wiring.py` (append tests)

**Interfaces:**
- Consumes: `decentralized_powerd.run` (Task 2).
- Produces: `run.run_powerd` symbol; `faas-powd` accepted in `--methods`; `FaaS-MAPoD` in `compare_results` default model set and palettes; `solver_options.powerd` block in the planar config.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_powerd_wiring.py`:

```python
import run


def test_methods_choice_accepts_faas_powd(monkeypatch):
  argv = ["run.py", "-c", "config_files/planar_comparison.json",
          "--methods", "faas-powd"]
  monkeypatch.setattr("sys.argv", argv)
  args = run.parse_arguments()
  assert "faas-powd" in args.methods


def test_run_module_exposes_powerd_runner():
  assert hasattr(run, "run_powerd")
  assert callable(run.run_powerd)


def test_planar_config_has_powerd_section():
  config = json.loads(Path("config_files/planar_comparison.json").read_text())
  powerd = config["solver_options"]["powerd"]
  assert powerd["d"] == 2
  assert powerd["criterion"] == "score"


def test_compare_results_palette_includes_mapod():
  import inspect
  import compare_results
  source = inspect.getsource(compare_results)
  assert '"FaaS-MAPoD"' in source


def test_compare_results_defaults_include_mapod(monkeypatch):
  monkeypatch.setattr("sys.argv", ["compare_results.py", "-i", "solutions/demo"])
  import compare_results
  args = compare_results.parse_arguments()
  assert "FaaS-MAPoD" in args.models
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_powerd_wiring.py -k "powd or mapod or powerd" -v`
Expected: FAIL — `faas-powd` not in argparse choices / `run` has no `run_powerd` / config has no `powerd` section / palette missing.

- [ ] **Step 3a: Add the import in `run.py`**

After `run.py:7` (`from decentralized_diffusion import run as run_diffusion`):

```python
from decentralized_powerd import run as run_powerd
```

- [ ] **Step 3b: Add `faas-powd` to the `--methods` choices**

In `run.py` `parse_arguments`, add `"faas-powd",` after `"faas-diffuse",`:

```python
    choices = [
      "centralized", 
      "faas-macro-v0", 
      "faas-macro", 
      "faas-madea",
      "hierarchical",
      "faas-diffuse",
      "faas-powd",
      "generate_only"
    ],
```

- [ ] **Step 3c: Add the `faas-powd` key to the `solution_folders` template**

In `run.py` (the `solution_folders = {...}` initializer), add the new key (note the added comma after the diffuse entry):

```python
  solution_folders = {
    "experiments_list": [],
    "centralized": [],
    "faas-macro": [],
    "faas-macro-v0": [],
    "faas-madea": [],
    "hierarchical": [],
    "faas-diffuse": [],
    "faas-powd": []
  }
```

- [ ] **Step 3d: Add the `run_p` flag initialization**

After the `run_d = False # -- faas-diffuse (FaaS-MADiG)` line:

```python
    run_p = False # -- faas-powd (FaaS-MAPoD)
```

- [ ] **Step 3e: Add the resume-time check for `faas-powd`**

After the `faas-diffuse` resume block (the one ending `run_d = True`), add:

```python
      if (not generate_only and "faas-powd" in methods) and ((
          len(solution_folders.get("faas-powd", [])) <= experiment_idx
        ) or (
          solution_folders["faas-powd"][experiment_idx] is None
        )):
        run_p = True
```

- [ ] **Step 3f: Add `run_p` to the `except ValueError` branch**

After `run_d = "faas-diffuse" in methods`:

```python
      run_p = "faas-powd" in methods
```

- [ ] **Step 3g: Add `run_p` to the run-guard**

Change the guard line:

```python
    if run_c or run_i or run_i_v0 or run_a or run_h or run_d or generate_only:
```

to:

```python
    if run_c or run_i or run_i_v0 or run_a or run_h or run_d or run_p or generate_only:
```

- [ ] **Step 3h: Add the dispatch block**

After the `faas-diffuse` dispatch block (the one calling `run_diffusion` and `set_solution_folder(..., "faas-diffuse", ...)`), add:

```python
      # -- solve power-of-d (FaaS-MAPoD)
      if run_p:
        p_folder = run_powerd(
          config,
          sp_parallelism,
          log_on_file = log_on_file,
          disable_plotting = disable_plotting
        )
        set_solution_folder(
          solution_folders, "faas-powd", experiment_idx, p_folder
        )
```

- [ ] **Step 3i: Extend the `mname` ternary**

In `run.py` `results_postprocessing`, replace the innermost `mname` branch:

```python
                "FaaS-MADiG" if method == "faas-diffuse" else "HierarchicalAuction"
```

with:

```python
                "FaaS-MADiG" if method == "faas-diffuse" else (
                  "FaaS-MAPoD" if method == "faas-powd" else "HierarchicalAuction"
                )
```

(The `mkey` ternary needs no change: `faas-powd` does not start with `faas-macro`, so it resolves to `"LSPc"`.)

- [ ] **Step 3j: Add a 7th method color**

In `run.py` `results_postprocessing`, extend `method_colors` (add a comma after `tab:purple` and a new entry):

```python
    method_colors = [
      mcolors.TABLEAU_COLORS["tab:blue"],
      mcolors.TABLEAU_COLORS["tab:orange"],
      mcolors.TABLEAU_COLORS["tab:red"],
      mcolors.TABLEAU_COLORS["tab:green"],
      mcolors.TABLEAU_COLORS["tab:pink"],
      mcolors.TABLEAU_COLORS["tab:purple"],
      mcolors.TABLEAU_COLORS["tab:brown"]
    ]
```

- [ ] **Step 3k: Add `FaaS-MAPoD` to the `compare_results.py` default model set**

Change `compare_results.py:55`:

```python
    default = ["LoadManagementModel", "FaaS-MACrO", "FaaS-MADeA", "FaaS-MADiG"]
```

to:

```python
    default = ["LoadManagementModel", "FaaS-MACrO", "FaaS-MADeA", "FaaS-MADiG", "FaaS-MAPoD"]
```

- [ ] **Step 3l: Add `FaaS-MAPoD` to both palettes in `compare_results.py`**

In `plot_by_key` (the dict near line 668) add the entry after the `FaaS-MADiG` line (add a comma after it):

```python
    "FaaS-MADiG": mcolors.CSS4_COLORS["plum"],
    "FaaS-MAPoD": mcolors.CSS4_COLORS["khaki"]
```

Apply the **identical** addition to the second palette in `violinplot_by_key` (the dict near line 829).

- [ ] **Step 3m: Add the `powerd` block to `config_files/planar_comparison.json`**

In `solver_options`, after the `diffusion` block, add a comma and the `powerd` block:

```json
    "diffusion": {
      "latency_weight": 0.0,
      "fairness_weight": 0.0,
      "unit_bids": false
    },
    "powerd": {
      "d": 2,
      "criterion": "score",
      "latency_weight": 0.0,
      "fairness_weight": 0.0,
      "unit_bids": false
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_powerd_wiring.py -v`
Expected: PASS (all wiring tests + the Task 2 run mock test).

- [ ] **Step 5: Verify no existing method wiring regressed**

Run: `uv run python -c "import json; json.load(open('config_files/planar_comparison.json')); import run, compare_results; print('imports ok')"`
Expected: prints `imports ok` (valid JSON, both modules import).

- [ ] **Step 6: Commit**

```bash
git add run.py compare_results.py config_files/planar_comparison.json tests/test_powerd_wiring.py
git commit -m "feat: wire FaaS-MAPoD into run.py, compare_results, planar config"
```

---

### Task 4: End-to-end smoke + reproducibility (Gurobi-gated)

**Files:**
- Test: `tests/test_powerd_e2e.py`

**Interfaces:**
- Consumes: `decentralized_powerd.run` (Task 2).
- Produces: none (test-only verification gate).

- [ ] **Step 1: Write the e2e test**

Create `tests/test_powerd_e2e.py`:

```python
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest

from decentralized_powerd import run as run_powerd


def _require_gurobi() -> None:
  solver = pyo.SolverFactory("gurobi")
  if not solver.available(exception_flag=False):
    pytest.skip("Gurobi solver is not available")


def _powerd_e2e_config(base_solution_folder: Path) -> dict:
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
      "powerd": {"d": 2, "criterion": "score",
                 "latency_weight": 0.0, "fairness_weight": 0.0, "unit_bids": False},
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


def test_powerd_runner_produces_expected_artifacts(tmp_path):
  _require_gurobi()
  folder = run_powerd(
    _powerd_e2e_config(tmp_path), parallelism=0, disable_plotting=True
  )

  obj = pd.read_csv(Path(folder, "obj.csv"))
  assert "FaaS-MAPoD" in obj.columns
  assert np.isfinite(pd.to_numeric(obj["FaaS-MAPoD"], errors="coerce")).all()

  runtime = pd.read_csv(Path(folder, "runtime.csv"))
  assert "tot" in runtime.columns
  assert (runtime["tot"] >= 0).all()

  assert Path(folder, "termination_condition.csv").exists()
  assert Path(folder, "LSPc_solution.csv").exists()


def test_powerd_runner_is_reproducible_for_same_seed(tmp_path):
  _require_gurobi()
  folder_a = run_powerd(
    _powerd_e2e_config(tmp_path / "a"), parallelism=0, disable_plotting=True
  )
  folder_b = run_powerd(
    _powerd_e2e_config(tmp_path / "b"), parallelism=0, disable_plotting=True
  )

  obj_a = pd.read_csv(Path(folder_a, "obj.csv"))["FaaS-MAPoD"].to_numpy()
  obj_b = pd.read_csv(Path(folder_b, "obj.csv"))["FaaS-MAPoD"].to_numpy()

  assert obj_a.shape == obj_b.shape
  assert np.allclose(obj_a, obj_b)
```

- [ ] **Step 2: Run the e2e test**

Run: `uv run pytest tests/test_powerd_e2e.py -v`
Expected: PASS if Gurobi is available; otherwise both tests SKIP with "Gurobi solver is not available".

- [ ] **Step 3: Commit**

```bash
git add tests/test_powerd_e2e.py
git commit -m "test: add FaaS-MAPoD e2e smoke and reproducibility (Gurobi-gated)"
```

---

### Task 5: LaTeX note `faas-mapod-note/`

**Files:**
- Create: `faas-mapod-note/faas-mapod.tex`
- Create: `faas-mapod-note/main.tex`
- Create: `faas-mapod-note/references.bib`
- Create: `faas-mapod-note/README.md`
- Create: `faas-mapod-note/.gitignore`

**Interfaces:**
- Consumes: the committed `faas-madig-note/` files as templates (notation recap, preview wrapper, bibliography, gitignore).
- Produces: a standalone, compilable LaTeX note for FaaS-MAPoD.

- [ ] **Step 1: Scaffold by copying the shared sibling files**

```bash
mkdir -p faas-mapod-note
cp faas-madig-note/references.bib faas-mapod-note/references.bib
cp faas-madig-note/.gitignore faas-mapod-note/.gitignore
```

(The reference set is the already-verified shared set; reuse it as-is. `nezami2021` and `bertsekas1988` stay available even though the note leans on Mitzenmacher as the basis.)

- [ ] **Step 2: Create `faas-mapod-note/main.tex` (preview wrapper)**

```latex
% Standalone preview wrapper for the FaaS-MAPoD note.
% The deliverable section lives in faas-mapod.tex and is meant to be
% \input{} (or pasted) into the host paper; this wrapper only renders it
% on its own for review.
\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath, amssymb}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage[numbers,square]{natbib}
\usepackage[margin=1in]{geometry}

\begin{document}
\input{faas-mapod}
\bibliographystyle{plainnat}
\bibliography{references}
\end{document}
```

- [ ] **Step 3: Create the notation recap block**

Copy the self-contained notation recap subsection from `faas-madig-note/faas-madig.tex` (the block delimited by `% BEGIN self-contained notation recap` … `% END self-contained notation recap`, i.e. the `\subsection{Notation and capacity model}` with its table) into a scratch buffer; it will be pasted into `faas-mapod.tex` in Step 4 with these two changes:
- relabel `sec:madig-notation` → `sec:mapod-notation` and `tab:madig-notation` → `tab:mapod-notation`;
- add two rows to the table's "Parameters and weights" group:
  - `Sample size & $d$ & number of neighbours probed per offloading step (power-of-$d$).\\`
  - `Selection criterion & --- & \texttt{score} (max $s_{ij}^f$) or \texttt{capacity} (max $C_j^f$).\\`

- [ ] **Step 4: Create `faas-mapod-note/faas-mapod.tex`**

```latex
% =====================================================================
% FaaS-MAPoD: randomized power-of-d-choices variant of FaaS-MADiG.
% This file is meant to be \input{} (or pasted) into the paper.
% It assumes the paper's notation (N, F, C_i^f, rho_i, omega_i^f,
% beta_{ij}^f, gamma_i^f, ...) is already defined.
% Cross-references are written as plain text "Eq.~(7)", "Alg.~4", etc.;
% convert them to \ref{} once the labels of the host paper are known.
% =====================================================================

\section{FaaS-MAPoD: a randomized power-of-$d$-choices variant}
\label{sec:mapod}

\textsc{FaaS-MAPoD} (Multi-Agent Power-of-$d$) is the randomized,
\emph{partial-visibility} sibling of \textsc{FaaS-MADiG}. It keeps every
price-free ingredient of \textsc{FaaS-MADiG}---the local problem~(P2), the
one-hop blackboard advertising residual execution capacity
$C_i^f(h)=\max\{0,\mathrm{Cap}_i^f(h)-x_i^f(h)\}$~(Eq.~(12)) and memory slack
$\rho_i(h)$~(Eq.~(13)), the convenience threshold $\gamma_i^f$, the fairness
penalty $\phi_i^f$, the price-free replica expansion of Algorithm~4, and the
restricted re-solve after each round. It changes \emph{only} how an overloaded
buyer chooses where to offload: instead of scanning its entire one-hop
candidate set $A_i^f(h)$ (as \textsc{FaaS-MADiG} does), it probes a random
sample of just $d$ candidates per offloading step and serves the best of that
sample---the classical \emph{power-of-$d$-choices} rule. Its purpose is to
isolate the value of full neighbourhood visibility: \emph{how much does scanning
the whole neighbourhood (FaaS-MADiG) buy over probing a random sample of $d$?}

% PASTE HERE the notation recap subsection prepared in Step 3
% (\subsection{Notation and capacity model} with table tab:mapod-notation,
%  including the two added rows for d and the selection criterion).

\subsection{Power-of-$d$ coordination}
\label{sec:mapod-coordination}

At iteration $h$, an overloaded buyer $i$ (with $\omega_i^f(h)>0$) admits the
same candidate sellers as \textsc{FaaS-MADiG}: those advertising residual
capacity $C_j^f(h)\ge 1$ and passing the convenience threshold,
\begin{equation}
  A_i^f(h)=\{\,j\in N_i : C_j^f(h)\ge 1 \ \text{and}\ s_{ij}^f(h)>-\gamma_i^f\,\},
  \qquad
  s_{ij}^f(h)=\beta_{ij}^f-w_{\mathrm{lat}}L_{ij}-w_{\mathrm{fair}}\phi_i^f(h),
  \label{eq:mapod-score}
\end{equation}
where $s_{ij}^f(h)$ is exactly the \textsc{FaaS-MADiG} score (the
\textsc{FaaS-MADeA} utility with the price term removed). The difference is the
\emph{visibility}: rather than ranking all of $A_i^f(h)$, the buyer repeatedly
draws a uniform random sample $S\subseteq A_i^f(h)$ of size $\min(d,|A_i^f(h)|)$,
\emph{without replacement}, and selects from $S$ the seller
\begin{equation}
  j^\star=
  \begin{cases}
    \arg\max_{j\in S}\ s_{ij}^f(h), & \text{criterion}=\texttt{score},\\[2pt]
    \arg\max_{j\in S}\ C_j^f(h),     & \text{criterion}=\texttt{capacity},
  \end{cases}
  \label{eq:mapod-criterion}
\end{equation}
with ties broken by the lower node id for reproducibility. The \texttt{score}
criterion ablates only the buyer's \emph{visibility} relative to
\textsc{FaaS-MADiG}; the \texttt{capacity} criterion is the textbook
shortest-queue (largest-spare-capacity) power-of-$d$ rule. In the default
\emph{batched} form the buyer claims $\min(C_{j^\star}^f(h),\ \omega_i^f(h)-o)$
from $j^\star$, removes it from the pool, and repeats until its demand is met or
the candidate set is empty; in the \emph{per-unit} form (\texttt{unit\_bids})
each unit of $\omega_i^f(h)$ draws a fresh sample of $d$, recovering the classic
per-arrival power-of-$d$-choices. Any residual demand triggers the same
price-free replica-expansion requests to neighbours with memory slack
$\rho_j(h)>0$ as in \textsc{FaaS-MADiG}.

When $d\ge|A_i^f(h)|$ the sample is the whole candidate set, and (for
distinct scores) the \texttt{score} criterion reduces exactly to
\textsc{FaaS-MADiG}'s greedy buyer rule---so \textsc{FaaS-MAPoD} interpolates
between full-visibility greedy diffusion ($d=|A_i^f(h)|$) and blind random
placement ($d=1$).

\begin{algorithm}
\caption{\textsc{FaaS-MAPoD} --- buyer power-of-$d$ sampling (replaces the FaaS-MADiG buyer rule)}
\label{alg:mapod-buyer}
\begin{algorithmic}[1]
\Require residual demand $\omega_i^f(h)$, blackboard residuals $C_j^f(h)$ and memory slacks $\rho_j(h)$, sample size $d$, criterion, RNG
\For{each function $f$ with $\omega_i^f(h)>0$}
  \State $A_i^f(h)\gets\{\,j\in N_i: C_j^f(h)\ge 1 \text{ and } s_{ij}^f(h)>-\gamma_i^f\,\}$
  \State $o\gets 0$;\quad maintain a local view $\tilde C_j\gets C_j^f(h)$ for $j\in A_i^f(h)$
  \While{$o<\omega_i^f(h)$ \textbf{and} $A_i^f(h)\ne\emptyset$}
     \State draw a uniform sample $S\subseteq A_i^f(h)$, $|S|=\min(d,|A_i^f(h)|)$, without replacement
     \State $j^\star\gets\arg\max_{j\in S} s_{ij}^f(h)$ \textbf{if} criterion${}=\texttt{score}$ \textbf{else} $\arg\max_{j\in S}\tilde C_j$ \Comment{tie-break: lower id}
     \State $q\gets 1$ \textbf{if} \texttt{unit\_bids} \textbf{else} $\min(\tilde C_{j^\star},\,\omega_i^f(h)-o)$
     \State emit assignment request $(i\!\to\! j^\star,f,q)$ with score $s_{ij^\star}^f(h)$;\quad $o\gets o+q$;\quad $\tilde C_{j^\star}\gets\tilde C_{j^\star}-q$
     \If{$\tilde C_{j^\star}<1$} remove $j^\star$ from $A_i^f(h)$ \EndIf
  \EndWhile
  \If{$o<\omega_i^f(h)$}
     \For{$j\in N_i$ with $\rho_j(h)>0$ and $j$ not already a capacity seller}
        \State emit \emph{price-free} replica-expansion request $(i\!\to\! j,f)$
     \EndFor
  \EndIf
\EndFor
\end{algorithmic}
\end{algorithm}

The seller side is \emph{unchanged}: requests received by $j$ are cleared by
\textsc{FaaS-MADiG}'s greedy fill (the \texttt{evaluate\_assignments} routine of
Section~\ref{sec:madig-greedy}), which serves buyers in descending-score order
up to the true residual capacity $C_j^f(h)$, tentatively starts replicas under
the utilization test $\kappa_j^f\le 1$, and re-awards incumbent load only to a
higher-score buyer. Consequently cross-buyer conflicts---when several buyers
sample the same seller---are resolved identically to \textsc{FaaS-MADiG}; only
the buyer's \emph{probe} is randomized.

\subsection{What is removed and what is kept}
\label{sec:mapod-ablation}

\noindent\textbf{Removed} (relative to \textsc{FaaS-MADeA}): the entire price
machinery, exactly as in \textsc{FaaS-MADiG}. \textbf{Removed} (relative to
\textsc{FaaS-MADiG}): \emph{full neighbourhood visibility}---the buyer now sees
only a random size-$d$ sample per step rather than the whole candidate set.

\noindent\textbf{Kept} (identical to \textsc{FaaS-MADiG}): the local
problem~(P2); the advertising of $C_i^f(h)$ and $\rho_i(h)$ (Eqs.~(12)--(13));
the score~\eqref{eq:mapod-score} and the convenience threshold $\gamma_i^f$; the
fairness penalty $\phi_i^f$; the price-free replica expansion of Algorithm~4;
the score-ordered seller clearing and reassignment; and the restricted
re-solve after each coordination round.

\subsection{Rationale and properties}
\label{sec:mapod-rationale}

\paragraph{Visibility ablation.} Because \textsc{FaaS-MAPoD} shares its score,
threshold, seller clearing, and provisioning with \textsc{FaaS-MADiG}, the
\emph{only} behavioural difference is how many neighbours the buyer inspects per
step. Any performance gap is therefore attributable to partial visibility
alone, placing \textsc{FaaS-MAPoD} at the random/partial-visibility end of the
\emph{market $\to$ greedy $\to$ random} spectrum
(\textsc{FaaS-MADeA} $\to$ \textsc{FaaS-MADiG} $\to$ \textsc{FaaS-MAPoD}).

\paragraph{Reproducibility.} The sampling generator is seeded once per run from
the experiment seed, so a run is fully reproducible; statistical variance across
the random draws is obtained by repeating experiments (the campaign's
$n_{\text{experiments}}$ replicates, each with its own seed and instance) rather
than by in-run averaging.

\paragraph{Communication.} Each offloading step inspects at most $d$ neighbours,
so \textsc{FaaS-MAPoD} has the smallest probe footprint of the three methods:
it reads $O(d)$ blackboard entries per step instead of the full one-hop
neighbourhood, while exchanging no prices.

\subsection{Positioning with respect to the literature}
\label{sec:mapod-related}

We state plainly that the buyer rule of \textsc{FaaS-MAPoD} \emph{is} the
power-of-$d$-choices load-balancing rule of
Mitzenmacher~\citep{mitzenmacher2001}, applied to one-hop FRALB offloading; it
is, by design, \emph{less} novel than \textsc{FaaS-MADiG} and is included as a
controlled baseline, not as a new algorithm.

\paragraph{Basis.} Probing $d$ uniformly random candidates and dispatching to
the best is exactly the power-of-$d$-choices paradigm
of~\citep{mitzenmacher2001}; with the \texttt{capacity} criterion and $d=2$ it
is the canonical ``power of two choices'' (shortest of two sampled queues). The
underlying push of excess load to less-loaded neighbours is the diffusion
paradigm of Cybenko~\citep{cybenko1989} and the sender-initiated strategies
catalogued by Willebeek-LeMair and Reeves~\citep{willebeek1993}; greedy,
locality-driven offloading in FaaS/edge dispatchers~\citep{leechoi2021} is a
full-visibility relative. \textsc{FaaS-MAPoD} and \textsc{FaaS-MADiG} are
siblings: \textsc{FaaS-MADiG} is the $d\!=\!|A_i^f(h)|$ (full-visibility) limit
of \textsc{FaaS-MAPoD}, and \textsc{FaaS-MADiG} is itself the price-frozen limit
of the \textsc{FaaS-MADeA} auction~\citep{bertsekas1988}.

\paragraph{Differences.} Two elements separate \textsc{FaaS-MAPoD} from the
textbook power-of-$d$ rule. \emph{(i) Objective-aligned selection.} With the
\texttt{score} criterion the buyer ranks the sample by the model's economic
weights ($\beta_{ij}^f$, $\gamma_i^f$, latency $L_{ij}$, fairness $\phi_i^f$)
rather than by raw queue length, and every round is re-validated by the local
problem~(P2) and the restricted re-solve. \emph{(ii) Joint balancing and
provisioning.} Beyond migrating load over a fixed graph, \textsc{FaaS-MAPoD}
triggers memory-slack-based replica expansion (Algorithm~4), inherited from the
\textsc{FaaS-MADeA}/\textsc{FaaS-MADiG} framework.

\paragraph{Takeaway.} As an isolated coordination rule, \textsc{FaaS-MAPoD}
instantiates power-of-$d$-choices balancing and is not claimed as novel; its
value is as a partial-visibility ablation that attributes any gap with respect
to \textsc{FaaS-MADiG} to full neighbourhood visibility, and as the random
endpoint of the market$\to$greedy$\to$random spectrum.
```

(After pasting, replace the Step-3 comment line with the actual notation recap subsection.)

- [ ] **Step 5: Create `faas-mapod-note/README.md`**

```markdown
# FaaS-MAPoD note

A paper-ready LaTeX section explaining **FaaS-MAPoD**, the randomized
power-of-d-choices sibling of FaaS-MADiG, in the notation of
`Decentralized_FaaS_coordination.pdf`.

## Files
- `faas-mapod.tex` — the `\section{}` to `\input{}` (or paste) into the paper.
  Remove the self-contained "Notation and capacity model" subsection on
  insertion (the host paper already defines that notation and those equations).
- `main.tex` — standalone preview wrapper (compile this to review the note).
- `references.bib` — cited works (shared, already-verified set; Mitzenmacher is
  the direct basis).
- `.gitignore` — LaTeX build artifacts.

## Build a preview
```bash
cd faas-mapod-note
latexmk -pdf main.tex
```

## Insert into the paper
1. `\input{faas-mapod}` (or paste the section).
2. Delete the "Notation and capacity model" subsection.
3. Convert the plain-text cross-references ("Eq.~(7)", "Alg.~4",
   "Section~\ref{sec:madig-greedy}") to the host paper's `\ref{}` labels.
4. Merge `references.bib` into the paper's bibliography (or re-key the
   `\citep{}` commands).
```

- [ ] **Step 6: Compile the preview to verify it builds**

Run:
```bash
cd faas-mapod-note && latexmk -pdf -interaction=nonstopmode main.tex
```
Expected: `main.pdf` is produced; no LaTeX errors (undefined-citation warnings on first pass are fine — `latexmk` reruns BibTeX). Then clean intermediates: `latexmk -c main.tex`.

- [ ] **Step 7: Commit**

```bash
git add faas-mapod-note/faas-mapod.tex faas-mapod-note/main.tex \
        faas-mapod-note/references.bib faas-mapod-note/README.md \
        faas-mapod-note/.gitignore
git commit -m "docs: add FaaS-MAPoD LaTeX note (power-of-d positioning)"
```

---

## Self-Review

**1. Spec coverage:**
- Part I §3.1 `sample_assignments` (criteria, batched/per-unit, threshold, memory fallback, remaining_blackboard, tie-break, degeneracy) → Task 1. ✅
- Part I §3.2 `run()` (RNG once, powerd options w/ defaults + no mutation, sample swap, evaluate reuse, fix_r/LSP_fixedr, obj column) → Task 2. ✅
- Part I §5 additive `run.py` wiring (import, choices, solution_folders, run_p flag/resume/except/guard, dispatch, mname, color) → Task 3. ✅
- Part I §4 config block + §6 compare_results palette/defaults + planar config → Task 3. ✅
- Part I §7 testing (unit, wiring, e2e + reproducibility) → Tasks 1, 2, 3, 4. ✅
- Part II §9–§12 LaTeX note → Task 5. ✅
- Out-of-scope items (Option C, capacity-ordered clearing, K-seed averaging) → not implemented, as required. ✅

**2. Placeholder scan:** No "TBD"/"TODO"/"handle edge cases". The only deferred content is the notation-recap block, which is copied verbatim from a committed file with two explicitly listed row additions (Task 5 Steps 3–4) — concrete, not a placeholder.

**3. Type consistency:** `sample_assignments(... , rng)` returns `(DataFrame[i,j,f,d,utility], DataFrame[i,j,f], int)`, matching the `bids, memory_bids, n_auctions` unpack in `run()` (Task 2 Edit 3c) and the `evaluate_assignments` consumer. `powerd_options` keys (`d`, `criterion`, `unit_bids`, `latency_weight`, `fairness_weight`) are produced in Task 2 Edit 3b and consumed by `sample_assignments` (Task 1). CLI key `faas-powd`, method name `FaaS-MAPoD`, mkey `LSPc`, obj column `FaaS-MAPoD`, runner symbol `run_powerd` are consistent across Tasks 2–4.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-26-faas-mapod-power-of-d.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session with checkpoints.

Which approach?
