# FaaS-MAGCAA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add FaaS-MAGCAA, a decentralized load-allocation algorithm adapted
from the Greedy Coalition Auction Algorithm (Braquet & Bakolas, 2021), as a
new experimental baseline alongside the existing FaaS-MA* family in this
repo.

**Architecture:** One new module, `decentralized_gcaa.py`, following the
exact per-timestep / inner-negotiation-loop shape every sibling
`decentralized_*.py` file already uses. It reuses `run_faasmadea.py`'s
`define_bids` unmodified (pinning its price vector at zero and its replica
vector at zero neutralizes the price-adaptation and replica-bidding paths
for free) and `check_stopping_criteria` unmodified. The only new logic is
a single consensus function, `resolve_gcaa_round`, implementing the
paper's single-winner-per-contested-task-per-round rule (Algorithms 1+3).
`run.py` gets a new registration entry following the exact pattern already
used for `faas-pg-s`/`faas-pg-r`.

**Tech Stack:** Python, NumPy, pandas, Pyomo/Gurobi (existing repo stack;
no new dependencies).

## Global Constraints

- No price adaptation: GCAA bids are pure utility (spec decision, see
  `docs/superpowers/specs/2026-07-03-faas-magcaa-design.md`).
- No replica-bidding / dynamic replica creation (spec decision — faithful
  to the paper's fixed task set with null assignment).
- Single winner per contested task per round; losers retry next round
  (spec decision — faithful to Algorithm 3, not a MADeA-style
  fill-capacity-in-one-round).
- `gcaa.unit_bids` must be `true` in config (the per-round consensus
  assumes each bid row is exactly one discrete unit of load).
- Follow existing repo conventions: each `decentralized_*.py` file owns a
  complete, self-contained `run()` (not a shared parameterized one); 2-space
  indentation (matches existing `.py` and `.json` files in this repo).

---

## Task 1: GCAA consensus function (`resolve_gcaa_round`)

**Files:**
- Create: `decentralized_gcaa.py` (this task adds only the consensus
  function and its imports; Task 2 extends the same file)
- Test: `tests/test_gcaa_helpers.py`

**Interfaces:**
- Produces: `resolve_gcaa_round(bids: pd.DataFrame, residual_capacity: np.array) -> np.array`
  — `bids` has columns `["i", "j", "f", "d", "b", "utility"]` (the exact
  shape `run_faasmadea.define_bids` returns). `residual_capacity` is a
  `(Nn, Nf)` array. Returns a `(Nn, Nn, Nf)` allocation delta for one
  round, where `y_round[i, j, f]` is the load agent `i` sends to seller
  `j` for function `f` this round.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gcaa_helpers.py`:

```python
import numpy as np
import pandas as pd

from decentralized_gcaa import resolve_gcaa_round


def _bids(rows):
  return pd.DataFrame(rows, columns=["i", "j", "f", "d", "b", "utility"])


def test_empty_bids_returns_zero_allocation():
  residual_capacity = np.ones((2, 1))
  y_round = resolve_gcaa_round(_bids([]), residual_capacity)
  assert y_round.shape == (2, 2, 1)
  assert (y_round == 0).all()


def test_single_winner_per_contested_task():
  # agents 0 and 1 both target seller 1 / function 0; agent 1 has the
  # higher utility and must be the sole winner this round
  bids = _bids([
    {"i": 0, "j": 1, "f": 0, "d": 1, "b": 0.5, "utility": 1.0},
    {"i": 1, "j": 1, "f": 0, "d": 1, "b": 0.7, "utility": 2.0},
  ])
  residual_capacity = np.array([[0.0], [5.0]])
  y_round = resolve_gcaa_round(bids, residual_capacity)
  assert y_round[1, 1, 0] == 1.0
  assert y_round[0, 1, 0] == 0.0
  assert y_round.sum() == 1.0


def test_agent_proposes_only_its_best_task():
  # agent 0 has bid rows for two different sellers; only the
  # higher-utility one (seller 2) should be proposed this round
  bids = _bids([
    {"i": 0, "j": 1, "f": 0, "d": 1, "b": 0.1, "utility": 0.5},
    {"i": 0, "j": 2, "f": 0, "d": 1, "b": 0.9, "utility": 3.0},
  ])
  residual_capacity = np.array([[5.0], [5.0], [5.0]])
  y_round = resolve_gcaa_round(bids, residual_capacity)
  assert y_round[0, 2, 0] == 1.0
  assert y_round[0, 1, 0] == 0.0


def test_seller_with_no_residual_capacity_is_skipped():
  bids = _bids([
    {"i": 0, "j": 1, "f": 0, "d": 1, "b": 0.5, "utility": 1.0},
  ])
  residual_capacity = np.array([[5.0], [0.0]])
  y_round = resolve_gcaa_round(bids, residual_capacity)
  assert y_round.sum() == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gcaa_helpers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'decentralized_gcaa'`

- [ ] **Step 3: Implement `resolve_gcaa_round`**

Create `decentralized_gcaa.py`:

```python
import numpy as np
import pandas as pd


def resolve_gcaa_round(
    bids: pd.DataFrame, residual_capacity: np.array
  ) -> np.array:
  """One GCAA consensus round (Braquet & Bakolas, 2021 - Algorithms 1+3):
  each buyer agent (i, f) proposes only its highest-utility task (j, f);
  for each contested task, the single highest-utility proposal wins and
  consumes one unit of residual_capacity[j, f]. Losers are absent from
  the returned allocation and simply re-propose next round via a fresh
  call to define_bids (their omega is left untouched by this function)."""
  Nn, Nf = residual_capacity.shape
  y_round = np.zeros((Nn, Nn, Nf))
  if len(bids) == 0:
    return y_round
  best_per_agent = bids.loc[bids.groupby(["i", "f"])["utility"].idxmax()]
  for (j, f), group in best_per_agent.groupby(["j", "f"]):
    j, f = int(j), int(f)
    if residual_capacity[j, f] <= 0:
      continue
    winner = group.loc[group["utility"].idxmax()]
    y_round[int(winner["i"]), j, f] += winner["d"]
  return y_round
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gcaa_helpers.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add decentralized_gcaa.py tests/test_gcaa_helpers.py
git commit -m "$(cat <<'EOF'
Add GCAA consensus function (resolve_gcaa_round)

Implements the single-winner-per-contested-task-per-round rule from
Braquet & Bakolas (2021) Algorithms 1+3, as the core new logic for the
FaaS-MAGCAA baseline.
EOF
)"
```

---

## Task 2: GCAA `run()` orchestration and CLI entrypoint

**Files:**
- Modify: `decentralized_gcaa.py` (append to the file created in Task 1)

**Interfaces:**
- Consumes: `resolve_gcaa_round(bids, residual_capacity) -> np.array` (Task 1)
- Consumes (reused unmodified, all pre-existing in this repo):
  - `run_faasmadea.define_bids(omega, blackboard, p, data, neighborhood, rho, auction_options, latency, fairness, force_memory_bids) -> Tuple[pd.DataFrame, pd.DataFrame, int]`
  - `run_faasmadea.check_stopping_criteria(it, max_iterations, blackboard, omega, rmp_omega, a, bids, memory_bids, tolerance, total_runtime, time_limit) -> Tuple[bool, str]`
  - `run_faasmadea.compute_residual_capacity(x, y, r, data) -> Tuple[np.array, np.array, np.array]` (returns `capacity, residual_capacity, ell`)
  - `run_faasmadea.neigh_dict_to_matrix(neighborhood_dict, Nn) -> np.array`
  - `run_faasmadea.check_ls_pr_feasibility_from_fixed_y(sp_data, y, tol=1e-9) -> list`
  - `run_faasmacro.solve_subproblem(sp_data, agents, sp, solver_name, general_solver_options, parallelism) -> 11-tuple` (`sp_data, sp_x, _, _, sp_omega, sp_r, sp_rho, sp_U, obj, tc, sp_runtime`)
  - `run_faasmacro.compute_social_welfare(spr, sp_data, agents, solver_name, general_solver_options, y, rmp_omega, parallelism) -> Tuple[tuple, float, str, float]` (solution unpacks as `sp_x, _, _, _, sp_r, sp_rho`)
  - `run_faasmacro.combine_solutions(Nn, Nf, sp_data, loadt, sp_x, sp_r, sp_rho, None, y, None, None, None, None) -> dict` (has `["sp"]["x"|"y"|"z"|"r"|"U"]`)
  - `run_faasmacro.decode_solutions(sp_data, solution, complete, None) -> Tuple[dict, Any, float]`
  - `utils.faasmacro.compute_centralized_objective(sp_data, x, y, z) -> float`
  - `utils.centralized.check_feasibility(x, y_sum, z, r, U, sp_data) -> Tuple[bool, str]`
  - `run_centralized_model.init_problem(limits, trace_type, max_steps, seed, solution_folder) -> Tuple[dict, dict, list, Graph]`
  - `run_centralized_model.get_current_load(input_requests_traces, agents, t) -> dict`
  - `run_centralized_model.update_data(base_instance_data, {"incoming_load": loadt}) -> dict`
  - `run_centralized_model.init_complete_solution() -> dict`
  - `run_centralized_model.join_complete_solution(complete) -> Tuple[dict, dict, dict]`
  - `run_centralized_model.save_checkpoint(complete_solution, path_prefix, t)`
  - `run_centralized_model.save_solution(solution, offloaded, complete_solution, detailed_fwd, name, folder)`
  - `run_centralized_model.plot_history(...)`
  - `utils.common.load_configuration(path) -> dict`
  - `models.sp.LSP()`, `models.sp.LSPr()`
- Produces: `run(config: dict, parallelism: int, log_on_file: bool = False, disable_plotting: bool = False) -> str` (returns solution folder path)
- Produces: `parse_arguments() -> argparse.Namespace`

- [ ] **Step 1: Add imports to the top of `decentralized_gcaa.py`**

Prepend to `decentralized_gcaa.py` (above the `resolve_gcaa_round`
function added in Task 1):

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
  check_ls_pr_feasibility_from_fixed_y,
  check_stopping_criteria,
  compute_residual_capacity,
  define_bids,
  neigh_dict_to_matrix,
)
from utils.centralized import check_feasibility
from utils.faasmacro import compute_centralized_objective
from utils.common import load_configuration
from models.sp import LSP, LSPr

from networkx import adjacency_matrix
from datetime import datetime
from copy import deepcopy
import argparse
import json
import sys
import os
```

(the `numpy as np` / `pandas as pd` imports already exist from Task 1;
do not duplicate them)

- [ ] **Step 2: Append `parse_arguments` and `run` to `decentralized_gcaa.py`**

```python
def parse_arguments() -> argparse.Namespace:
  """
  Parse input arguments
  """
  parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description = "Run FaaS-MAGCAA",
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument(
    "-c", "--config",
    help = "Configuration file",
    type = str,
    default = "manual_config.json"
  )
  parser.add_argument(
    "-j", "--parallelism",
    help = "Number of parallel processes to start (-1: auto, 0: sequential)",
    type = int,
    default = -1
  )
  parser.add_argument(
    "--disable_plotting",
    help = "True to disable automatic plot generation for each experiment",
    default = False,
    action = "store_true"
  )
  args: argparse.Namespace = parser.parse_known_args()[0]
  return args


def run(
    config: dict,
    parallelism: int,
    log_on_file: bool = False,
    disable_plotting: bool = False
  ):
  base_solution_folder = config["base_solution_folder"]
  seed = config["seed"]
  limits = config["limits"]
  trace_type = config["limits"]["load"].get("trace_type", "fixed_sum")
  verbose = config.get("verbose", 0)
  solver_name = config["solver_name"]
  solver_options = config["solver_options"]
  general_solver_options = solver_options.get("general", {})
  gcaa_options = solver_options["gcaa"]
  time_limit = general_solver_options.get("TimeLimit", np.inf)
  tolerance = config.get("tolerance", 1e-6)
  max_iterations = config["max_iterations"]
  max_steps = config["max_steps"]
  min_run_time = config.get("min_run_time", 0)
  max_run_time = config.get("max_run_time", max_steps)
  run_time_step = config.get("run_time_step", 1)
  checkpoint_interval = config["checkpoint_interval"]
  # generate solution folder
  now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
  solution_folder = f"{base_solution_folder}/{now}"
  os.makedirs(solution_folder, exist_ok = True)
  with open(os.path.join(solution_folder, "config.json"), "w") as ostream:
    ostream.write(json.dumps(config, indent = 2))
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
  latency = adjacency_matrix(graph, weight = "network_latency")
  ub = (
    max_run_time + run_time_step
  ) if max_run_time == min_run_time else max_run_time
  sp_complete_solution = init_complete_solution()
  spc_complete_solution = init_complete_solution()
  obj_dict = {"LSPr_final": []}
  tc_dict = {"LSPr": []}
  runtime_list = []
  # GCAA bids are pure utility: price stays at zero forever (no
  # evaluate_bids-style price adaptation)
  no_price = np.zeros((Nn, Nf))
  # zero replica capacity disables define_bids' memory-bid path: GCAA
  # has no replica-bidding, a buyer with no convenient seller just gets
  # the null assignment (utility 0) for this round
  no_replica_sellers = np.zeros((Nn,))
  for t in range(min_run_time, ub, run_time_step):
    if verbose > 0:
      print(f"t = {t}", file = log_stream, flush = True)
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
    if verbose > 1:
      print(
        f"    sp: DONE ({tc['tot']}; obj = {obj['tot']}; "
        f"runtime = {sp_runtime['tot']})",
        file = log_stream, flush = True
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
    while not stop_searching:
      if verbose > 0:
        print(f"    it = {it}", file = log_stream, flush = True)
      capacity, residual_capacity, ell = compute_residual_capacity(
        sp_x, y, sp_r, sp_data
      )
      blackboard = np.maximum(0.0, capacity - sp_x)
      bids, memory_bids, n_auctions = define_bids(
        omega, blackboard, no_price, sp_data, neighborhood,
        no_replica_sellers, gcaa_options, latency, fairness,
        force_memory_bids = False
      )
      if verbose > 2:
        print(bids, file = log_stream, flush = True)
      rmp_omega = np.zeros((Nn, Nf))
      if len(bids) > 0:
        auction_y = resolve_gcaa_round(bids, residual_capacity)
        y += auction_y
        for n in range(Nn):
          for f in range(Nf):
            rmp_omega[n,f] = y[n,:,f].sum()
            if rmp_omega[n,f] > 0:
              fairness[n,f] += 1
        bad_nodes = check_ls_pr_feasibility_from_fixed_y(sp_data, y)
        if bad_nodes:
          raise RuntimeError(
            f"LSPr infeasible from fixed y assignments: {bad_nodes}"
          )
        spr_sol, spr_obj, spr_tc, spr_runtime = compute_social_welfare(
          spr, sp_data, agents, solver_name, general_solver_options,
          y, rmp_omega, parallelism
        )
        total_runtime += spr_runtime
        sp_x, _, _, _, sp_r, sp_rho = spr_sol
        for i in range(Nn):
          for f in range(Nf):
            omega[i,f] = sp_omega[i,f] - rmp_omega[i,f]
            if abs(omega[i,f]) < tolerance:
              omega[i,f] = 0.0
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
        it, max_iterations, blackboard, omega, rmp_omega, None,
        bids, memory_bids, tolerance, total_runtime, time_limit
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
          f"total runtime: {total_runtime})"
        )
        if t % checkpoint_interval == 0 or t == max_steps - 1:
          save_checkpoint(
            sp_complete_solution, os.path.join(solution_folder, "LSP"), t
          )
          save_checkpoint(
            spc_complete_solution, os.path.join(solution_folder, "LSPc"), t
          )
    ee = datetime.now()
    if verbose > 0:
      print(
        f"    TOTAL RUNTIME [s] = {total_runtime} "
        f"(wallclock: {(ee-ss).total_seconds()})",
        file = log_stream, flush = True
      )
    runtime_list.append(total_runtime)
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
  pd.DataFrame(obj_dict["LSPr_final"], columns = ["FaaS-MAGCAA"]).to_csv(
    os.path.join(solution_folder, "obj.csv"), index = False
  )
  pd.DataFrame(tc_dict["LSPr"]).to_csv(
    os.path.join(solution_folder, "termination_condition.csv")
  )
  pd.DataFrame({"tot": runtime_list}).to_csv(
    os.path.join(solution_folder, "runtime.csv"), index = False
  )
  if verbose > 0:
    print(
      f"All solutions saved in: {solution_folder}",
      file = log_stream, flush = True
    )
  if log_on_file:
    log_stream.close()
  return solution_folder


if __name__ == "__main__":
  args = parse_arguments()
  config = load_configuration(args.config)
  run(
    config, args.parallelism, log_on_file = False,
    disable_plotting = args.disable_plotting
  )
```

- [ ] **Step 3: Verify the module imports cleanly**

Run: `python -c "import decentralized_gcaa"`
Expected: no output, exit code 0 (catches typos/import errors before the
behavioral wiring test in Task 3 exercises the full `run()` path)

- [ ] **Step 4: Re-run Task 1's tests to confirm nothing broke**

Run: `pytest tests/test_gcaa_helpers.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add decentralized_gcaa.py
git commit -m "$(cat <<'EOF'
Add FaaS-MAGCAA run() orchestration

Per-timestep loop matches run_faasmadea.py's shape exactly, reusing
define_bids/check_stopping_criteria/compute_residual_capacity
unmodified; only evaluate_bids is replaced, by resolve_gcaa_round.
Behavioral verification (wiring smoke test, e2e test) follows in the
next two tasks.
EOF
)"
```

---

## Task 3: Register FaaS-MAGCAA in `run.py`

**Files:**
- Modify: `run.py:11` (imports), `run.py:30-44` (`METHOD_RESULT_MODELS`),
  `run.py:71-87` (`--methods` choices), `run.py:921-934` (flag
  declarations), `run.py:~1013-1018` (try-block read-back),
  `run.py:~1020-1029` (except-block fallback + combined condition),
  `run.py:~1199-1207` (dispatch block)
- Test: `tests/test_gcaa_wiring.py`

**Interfaces:**
- Consumes: `decentralized_gcaa.run` (Task 2), `decentralized_gcaa.resolve_gcaa_round`/module attributes for monkeypatching
- Produces: `run.run_gcaa` (module-level name usable by other code/tests), `run.METHOD_RESULT_MODELS["faas-gcaa"] == ("LSPc", "FaaS-MAGCAA")`, `"faas-gcaa"` accepted by `run.parse_arguments()`'s `--methods`

- [ ] **Step 1: Write the failing wiring tests**

Create `tests/test_gcaa_wiring.py`:

```python
import json
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd

import decentralized_gcaa
import run


def test_methods_choice_accepts_faas_gcaa(monkeypatch):
  argv = ["run.py", "-c", "config_files/planar_comparison.json",
          "--methods", "faas-gcaa"]
  monkeypatch.setattr("sys.argv", argv)
  args = run.parse_arguments()
  assert "faas-gcaa" in args.methods


def test_run_module_exposes_gcaa_runner():
  assert hasattr(run, "run_gcaa")
  assert callable(run.run_gcaa)


def test_method_result_models_has_gcaa_entry():
  assert run.METHOD_RESULT_MODELS["faas-gcaa"] == ("LSPc", "FaaS-MAGCAA")


def test_planar_config_has_gcaa_section():
  config = json.loads(Path("config_files/planar_comparison.json").read_text())
  gcaa = config["solver_options"]["gcaa"]
  assert gcaa["unit_bids"] is True
  assert "latency_weight" in gcaa
  assert "fairness_weight" in gcaa


def test_set_solution_folder_tolerates_missing_method_key():
  solution_folders = {"experiments_list": []}
  run.set_solution_folder(solution_folders, "faas-gcaa", 0, "/some/folder")
  assert solution_folders["faas-gcaa"][0] == "/some/folder"


def test_gcaa_run_stops_when_no_bids_available(tmp_path, monkeypatch):
  base_data = {
    None: {
      "Nn": {None: 1},
      "Nf": {None: 1},
      "neighborhood": {(1, 1): 0},
    }
  }
  monkeypatch.setattr(
    decentralized_gcaa, "init_problem",
    lambda *args, **kwargs: (base_data, {}, [], nx.empty_graph(1)),
  )
  monkeypatch.setattr(decentralized_gcaa, "get_current_load", lambda *args: {})
  monkeypatch.setattr(decentralized_gcaa, "update_data", lambda data, update: data)
  monkeypatch.setattr(decentralized_gcaa, "LSP", lambda: "LSP")
  monkeypatch.setattr(decentralized_gcaa, "LSPr", lambda: "LSPr")

  def _solve_subproblem(sp_data, agents, sp, *args):
    return (
      sp_data,
      np.zeros((1, 1)),
      None,
      None,
      np.zeros((1, 1)),   # sp_omega: no residual load -> no bids
      np.ones((1, 1)),
      np.array([0.0]),
      np.zeros((1, 1)),
      {"tot": 0.0},
      {"tot": "ok"},
      {"tot": 0.0},
    )

  monkeypatch.setattr(decentralized_gcaa, "solve_subproblem", _solve_subproblem)
  monkeypatch.setattr(
    decentralized_gcaa, "compute_residual_capacity",
    lambda *args: (np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))),
  )
  monkeypatch.setattr(
    decentralized_gcaa, "define_bids",
    lambda *args, **kwargs: (
      pd.DataFrame({"i": [], "j": [], "f": [], "d": [], "b": [], "utility": []}),
      pd.DataFrame({"i": [], "j": [], "f": []}),
      1,
    ),
  )
  monkeypatch.setattr(
    decentralized_gcaa, "combine_solutions",
    lambda *args: {"sp": {
      "x": np.zeros((1, 1)), "y": np.zeros((1, 1, 1)),
      "z": np.zeros((1, 1)), "r": np.ones((1, 1)), "U": np.zeros((1, 1)),
    }},
  )
  monkeypatch.setattr(decentralized_gcaa, "compute_centralized_objective", lambda *args: -1.0)
  monkeypatch.setattr(decentralized_gcaa, "check_feasibility", lambda *args: (True, "ok"))

  decoded = []

  def _decode(sp_data, solution, complete, arg):
    decoded.append(solution)
    return complete, None, 1.0

  monkeypatch.setattr(decentralized_gcaa, "decode_solutions", _decode)
  monkeypatch.setattr(
    decentralized_gcaa, "join_complete_solution", lambda complete: ({}, {}, {})
  )
  monkeypatch.setattr(decentralized_gcaa, "save_checkpoint", lambda *args: None)
  monkeypatch.setattr(decentralized_gcaa, "save_solution", lambda *args: None)

  config = {
    "base_solution_folder": str(tmp_path),
    "seed": 1,
    "limits": {"load": {"trace_type": "fixed_sum"}},
    "solver_name": "mock",
    "solver_options": {
      "general": {"TimeLimit": 10},
      "gcaa": {
        "unit_bids": True, "epsilon": 0.01,
        "latency_weight": 0.0, "fairness_weight": 0.0,
      },
    },
    "max_iterations": 5,
    "max_steps": 1,
    "min_run_time": 0,
    "max_run_time": 0,
    "run_time_step": 1,
    "checkpoint_interval": 1,
    "verbose": 0,
  }

  decentralized_gcaa.run(config, parallelism=0, disable_plotting=True)

  assert len(decoded) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gcaa_wiring.py -v`
Expected: FAIL — `ModuleNotFoundError` / `AttributeError: module 'run' has
no attribute 'run_gcaa'` / `KeyError: 'faas-gcaa'`

- [ ] **Step 3: Add the import**

In `run.py`, modify line 11 (the line reading
`from decentralized_potentialgame import run_pg_s, run_pg_r`):

```python
from decentralized_potentialgame import run_pg_s, run_pg_r
from decentralized_gcaa import run as run_gcaa
```

- [ ] **Step 4: Register in `METHOD_RESULT_MODELS`**

In `run.py`, modify the `METHOD_RESULT_MODELS` dict (currently ends with
`"faas-pg-r": ("LSPc", "FaaS-MAPG-R"),` followed by the closing `}`):

```python
  "faas-pg-r": ("LSPc", "FaaS-MAPG-R"),
  "faas-gcaa": ("LSPc", "FaaS-MAGCAA"),
}
```

- [ ] **Step 5: Add to `--methods` CLI choices**

In `run.py`, modify the `choices` list in `parse_arguments`:

```python
      "faas-pg-s",
      "faas-pg-r",
      "faas-gcaa",
      "generate_only"
    ],
```

- [ ] **Step 6: Add the per-experiment flag declaration**

In `run.py`, modify the flag block inside the experiments loop (currently
ends with `run_pgr = False # -- faas-pg-r (FaaS-MAPG-R)` followed by
`experiment_idx = None`):

```python
    run_pgr = False # -- faas-pg-r (FaaS-MAPG-R)
    run_g = False # -- faas-gcaa (FaaS-MAGCAA)
    experiment_idx = None
```

- [ ] **Step 7: Add the try-block read-back check**

In `run.py`, modify the `try:` block, immediately after the existing
`faas-pg-r` check (which ends with `run_pgr = True`) and before
`except ValueError:`:

```python
      if (not generate_only and "faas-pg-r" in methods) and ((
          len(solution_folders.get("faas-pg-r", [])) <= experiment_idx
        ) or (
          solution_folders["faas-pg-r"][experiment_idx] is None
        )):
        run_pgr = True
      if (not generate_only and "faas-gcaa" in methods) and ((
          len(solution_folders.get("faas-gcaa", [])) <= experiment_idx
        ) or (
          solution_folders["faas-gcaa"][experiment_idx] is None
        )):
        run_g = True
    except ValueError:
```

- [ ] **Step 8: Add the except-block fallback and combined condition**

In `run.py`, modify the `except ValueError:` block (currently ends with
`run_pgr = "faas-pg-r" in methods` followed by the `# if the experiment
is still to run...` comment):

```python
      run_pgr = "faas-pg-r" in methods
      run_g = "faas-gcaa" in methods
    # if the experiment is still to run...
```

Then modify the combined condition line immediately after:

```python
    if run_c or run_i or run_i_v0 or run_a or run_h or run_hm or run_d or run_p or run_brs or run_brr or run_bro or run_pgs or run_pgr or run_g or generate_only:
```

- [ ] **Step 9: Add the dispatch block**

In `run.py`, modify the dispatch section, immediately after the existing
`run_pgr` dispatch block (which ends with the `set_solution_folder(...,
"faas-pg-r", ...)` call) and before the `# -- save info` comment:

```python
      # -- solve potential game randomized (FaaS-MAPG-R)
      if run_pgr:
        pgr_folder = run_pg_r(
          config, sp_parallelism,
          log_on_file = log_on_file, disable_plotting = disable_plotting
        )
        set_solution_folder(
          solution_folders, "faas-pg-r", experiment_idx, pgr_folder
        )
      # -- solve greedy coalition auction (FaaS-MAGCAA)
      if run_g:
        g_folder = run_gcaa(
          config, sp_parallelism,
          log_on_file = log_on_file, disable_plotting = disable_plotting
        )
        set_solution_folder(
          solution_folders, "faas-gcaa", experiment_idx, g_folder
        )
      # -- save info
```

- [ ] **Step 10: Add the `gcaa` section to `config_files/planar_comparison.json`**

(Needed now because `test_planar_config_has_gcaa_section` reads this
file; the remaining config files are updated in Task 4.)

Modify `config_files/planar_comparison.json`:

```json
      "latency_weight": 0.0,
      "fairness_weight": 0.0
    },
    "gcaa": {
      "unit_bids": true,
      "epsilon": 0.01,
      "latency_weight": 0.0,
      "fairness_weight": 0.0
    },
    "diffusion": {
```

- [ ] **Step 11: Run tests to verify they pass**

Run: `pytest tests/test_gcaa_wiring.py -v`
Expected: 6 passed

- [ ] **Step 12: Run the full existing test suite to check for regressions**

Run: `pytest tests/ -x -q`
Expected: all tests pass (or skip, for Gurobi-dependent tests if Gurobi
is unavailable in this environment) — no new failures introduced by the
`run.py` edits

- [ ] **Step 13: Commit**

```bash
git add run.py config_files/planar_comparison.json tests/test_gcaa_wiring.py
git commit -m "$(cat <<'EOF'
Register FaaS-MAGCAA in run.py

Wires up --methods faas-gcaa, METHOD_RESULT_MODELS, and the
per-experiment run/skip/dispatch flags, following the exact pattern
already used for faas-pg-s/faas-pg-r.
EOF
)"
```

---

## Task 4: Config files and end-to-end test

**Files:**
- Modify: `config_files/eval_full.json`, `config_files/eval_smoke.json`,
  `config_files/eval_tuned_smoke.json`
- Test: `tests/test_gcaa_e2e.py`

**Interfaces:**
- Consumes: `decentralized_gcaa.run` (Task 2)

- [ ] **Step 1: Add the `gcaa` section to the remaining config files**

Modify `config_files/eval_full.json` and `config_files/eval_smoke.json`
(both share the same auction-block tail):

```json
      "latency_weight": 0.0,
      "fairness_weight": 0.0
    },
    "gcaa": {
      "unit_bids": true,
      "epsilon": 0.01,
      "latency_weight": 0.0,
      "fairness_weight": 0.0
    },
    "diffusion": {
```

Modify `config_files/eval_tuned_smoke.json` (its auction block ends with
an explicit `"unit_bids": false` line):

```json
      "fairness_weight": 0.0,
      "unit_bids": false
    },
    "gcaa": {
      "unit_bids": true,
      "epsilon": 0.01,
      "latency_weight": 0.0,
      "fairness_weight": 0.0
    },
    "diffusion": {
```

- [ ] **Step 2: Verify all four config files still parse as valid JSON**

Run:
```bash
python3 -c "
import json
for f in ['config_files/planar_comparison.json', 'config_files/eval_full.json', 'config_files/eval_smoke.json', 'config_files/eval_tuned_smoke.json']:
    c = json.load(open(f))
    assert c['solver_options']['gcaa']['unit_bids'] is True, f
    print(f, 'OK')
"
```
Expected: all four files print `OK`, no exceptions

- [ ] **Step 3: Write the failing end-to-end test**

Create `tests/test_gcaa_e2e.py`:

```python
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest
from parse import parse

from decentralized_gcaa import run as run_gcaa


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
      "load": {
        "trace_type": "clipped",
        "min": {"min": 2.0, "max": 2.0},
        "max": {"min": 3.0, "max": 3.0},
      },
    },
    "solver_name": "gurobi",
    "solver_options": {
      "general": {"TimeLimit": 60, "OutputFlag": 0},
      "gcaa": {
        "unit_bids": True, "epsilon": 0.01,
        "latency_weight": 0.0, "fairness_weight": 0.0,
      },
    },
    "max_iterations": 50,
    "max_steps": 8,
    "min_run_time": 1,
    "max_run_time": 1,
    "run_time_step": 1,
    "checkpoint_interval": 1,
    "tolerance": 1e-6,
    "verbose": 0,
  }


def test_run_gcaa_produces_artifacts(tmp_path):
  _require_gurobi()
  folder = run_gcaa(
    _e2e_config(tmp_path), parallelism=0, disable_plotting=True
  )
  obj = pd.read_csv(Path(folder, "obj.csv"))
  assert "FaaS-MAGCAA" in obj.columns
  assert len(obj) >= 1
  assert np.isfinite(pd.to_numeric(obj["FaaS-MAGCAA"], errors="coerce")).all()
  runtime = pd.read_csv(Path(folder, "runtime.csv"))
  assert "tot" in runtime.columns
  assert (runtime["tot"] >= 0).all()
  tc = pd.read_csv(Path(folder, "termination_condition.csv"))
  assert len(tc) >= 1
  for s in tc["0"]:
    assert parse(
      "{} (it: {}; obj. deviation: {}; best it: {}; total runtime: {})", s
    ) is not None
```

- [ ] **Step 4: Run the test**

Run: `pytest tests/test_gcaa_e2e.py -v`
Expected: PASS if Gurobi is available, otherwise SKIPPED with reason
"Gurobi solver is not available" (do not treat a skip as a failure — it
mirrors the exact same skip behavior as `test_potentialgame_e2e.py`)

- [ ] **Step 5: If Gurobi is available and the test fails, diagnose before
  changing the consensus logic**

The most likely failure mode is `max_iterations reached` before all load
is assigned, given the single-winner-per-round rule needs one round per
finalized unit in the worst case. If this happens, raise
`max_iterations` in `_e2e_config` (not the production defaults) — this
is a test-tuning concern, not a correctness bug, unless `obj.csv` /
`runtime.csv` / `termination_condition.csv` are malformed.

- [ ] **Step 6: Run the full test suite one more time**

Run: `pytest tests/ -q`
Expected: all tests pass or skip (Gurobi-dependent), zero failures

- [ ] **Step 7: Commit**

```bash
git add config_files/eval_full.json config_files/eval_smoke.json \
  config_files/eval_tuned_smoke.json tests/test_gcaa_e2e.py
git commit -m "$(cat <<'EOF'
Add FaaS-MAGCAA config sections and end-to-end test

Completes the FaaS-MAGCAA baseline: all four existing config files now
carry a gcaa solver_options section, and an end-to-end test (skipped
without Gurobi) exercises the full run() against a 10-node planar
instance.
EOF
)"
```
