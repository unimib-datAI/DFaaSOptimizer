"""The hierarchy runs only at completed MADEA cycle boundaries."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest

import run_faasmadea as madea
from hierarchical_auction import madea_cycles_runner as madea_runner
from hierarchical_auction.engine import HierarchicalAuctionEngine, LevelResult
from hierarchical_auction.types import AcceptedAllocation
from test_review_distributed_regressions import _materialized_config, _two_node_data


@pytest.fixture
def cycle_run(tmp_path, monkeypatch):
  data = _two_node_data()
  data[None].update({
    "incoming_load": {(1, 1): 100, (2, 1): 1},
    "demand": {(1, 1): 0.4, (2, 1): 0.4},
    "gamma": {(1, 1): 10.0, (2, 1): 10.0},
    "memory_capacity": {1: 1, 2: 2},
    "beta": {(1, 1, 1): 0.0, (1, 2, 1): 1.0,
             (2, 1, 1): 1.0, (2, 2, 1): 0.0},
  })
  config = _materialized_config(tmp_path, data)
  config["solver_options"] = {"general": {}, "auction": {
    "eta": 0.0, "epsilon": 0.001, "zeta": 0.01, "unit_bids": True,
  }}
  # Only replace solver calls. Bidding, feasibility, scoring and the cycle
  # itself stay real, so ordering tests also run without an external solver.
  def solve(sp_data, *args, **kwargs):
    return (
      sp_data, np.array([[2.], [1.]]), np.zeros((2, 2, 1)), np.zeros((2, 1)),
      np.array([[98.], [0.]]), np.array([[1.], [2.]]), np.zeros(2),
      np.array([[0.8], [0.2]]), {"tot": 0.0}, {"tot": "optimal"}, {"tot": 0.25},
    )

  def welfare(spr, sp_data, agents, solver_name, options, y, rmp_omega, parallelism, x):
    r = np.ceil((x + y.sum(axis=0)) * 0.4)
    rho = np.array([1., 2.]) - r[:, 0]
    return (x, y, None, None, r, rho), 0.0, "optimal", 0.1

  monkeypatch.setattr(madea_runner, "solve_subproblem", solve)
  monkeypatch.setattr(madea_runner, "compute_social_welfare", welfare)
  monkeypatch.setattr(madea, "compute_social_welfare", welfare)
  events = []
  real_define = madea.define_bids

  def define(*args, **kwargs):
    events.append("madea")
    bids, memory_bids, count = real_define(*args, **kwargs)
    # One bid per iteration makes the cycle boundary observable.
    return bids.head(1), memory_bids, count

  # Observe the real shared cycle, not a replacement implementation.
  monkeypatch.setattr(madea, "define_bids", define)

  def run(stops, hierarchy_quantity=0, eta=0.0):
    config["solver_options"]["auction"]["eta"] = eta
    outcomes = iter(stops)

    def check(*args, **kwargs):
      result = next(outcomes)
      events.append(("check", *result))
      return result

    def higher(self, **kwargs):
      events.append("hierarchy")
      y, omega = kwargs["y"].copy(), kwargs["omega"].copy()
      allocations = []
      if hierarchy_quantity:
        y[0, 1, 0] += hierarchy_quantity
        omega[0, 0] -= hierarchy_quantity
        allocations.append(AcceptedAllocation(2, 0, 0, 1, 0, 1, hierarchy_quantity, 1.0))
      return LevelResult(y, omega, allocations)

    monkeypatch.setattr(madea, "check_stopping_criteria", check)
    monkeypatch.setattr(HierarchicalAuctionEngine, "run_higher_levels", higher)
    folder = madea_runner.run(config, parallelism=0, disable_plotting=True)
    return events, folder

  return run


def test_complete_cycle_finishes_before_hierarchy_and_restarts(cycle_run):
  events, _ = cycle_run([
    (False, None), (False, None), (True, "UB/LB diff < tol"),
    (True, "all load assigned"),
  ])
  assert events == [
    "madea", ("check", False, None),
    "madea", ("check", False, None),
    "madea", ("check", True, "UB/LB diff < tol"),
    "hierarchy", "madea", ("check", True, "all load assigned"),
  ]


def test_all_load_assigned_never_invokes_hierarchy(cycle_run):
  events, _ = cycle_run([(True, "all load assigned")])
  assert events == ["madea", ("check", True, "all load assigned")]


@pytest.mark.parametrize("reason", [
  "no capacity left", "max iterations reached", "load cannot be assigned",
  "no available or convenient sellers", "reached time limit: 120 >= 120",
])
def test_other_stopping_reasons_invoke_hierarchy(cycle_run, reason):
  events, _ = cycle_run([(True, reason), (True, "all load assigned")])
  assert events == [
    "madea", ("check", True, reason), "hierarchy",
    "madea", ("check", True, "all load assigned"),
  ]


def test_real_no_progress_runs_one_followup_cycle(tmp_path, monkeypatch):
  if not pyo.SolverFactory("glpk").available(exception_flag=False):
    pytest.skip("GLPK needed for the real MADEA cycle")
  data = _two_node_data()
  data[None]["memory_capacity"] = {1: 0, 2: 0}
  config = _materialized_config(tmp_path, data)
  events = []
  real_check = madea.check_stopping_criteria

  def check(*args, **kwargs):
    result = real_check(*args, **kwargs)
    events.append(("check", result[1]))
    assert len(events) < 6, "Unchanged state must not loop forever"
    return result

  def higher(self, **kwargs):
    events.append("hierarchy")
    return LevelResult(kwargs["y"].copy(), kwargs["omega"].copy())

  monkeypatch.setattr(madea, "check_stopping_criteria", check)
  monkeypatch.setattr(HierarchicalAuctionEngine, "run_higher_levels", higher)
  folder = madea_runner.run(config, parallelism=0, disable_plotting=True)
  assert events == [
    ("check", "no capacity left"), "hierarchy", ("check", "no capacity left"),
  ]
  termination = pd.read_csv(f"{folder}/termination_condition.csv").iloc[0, -1]
  assert "no progress" in termination


@pytest.mark.parametrize("resume", [False, True])
def test_both_algorithms_are_selectable_and_resume_independently(tmp_path, monkeypatch, resume):
  import json
  import run
  from hierarchical_auction import madea_runner as original

  monkeypatch.setattr("sys.argv", [
    "run.py", "--methods", "hierarchical-madea", "hierarchical-madea-cycles",
  ])
  methods = run.parse_arguments().methods
  assert run.run_hierarchical_madea is original.run
  assert run.run_hierarchical_madea_cycles is madea_runner.run
  assert run.METHOD_RESULT_MODELS["hierarchical-madea-cycles"] == (
    "LSPc", "HierarchicalMADeACycles",
  )
  if resume:
    (tmp_path / "experiments.json").write_text(json.dumps({
      "experiments_list": [[2, 123]], "hierarchical-madea": ["old-result"],
    }))
  calls = []

  def old(*args, **kwargs):
    calls.append("old")
    return "old-result"

  def new(*args, **kwargs):
    calls.append("cycles")
    return "cycles-result"

  monkeypatch.setattr(run, "run_hierarchical_madea", old)
  monkeypatch.setattr(run, "run_hierarchical_madea_cycles", new)
  monkeypatch.setattr(run, "results_postprocessing", lambda *a, **kw: None)
  run.run(
    {"seed": 123, "verbose": 0, "limits": {"Nn": {"values": [2]}}},
    str(tmp_path), n_experiments=1, methods=methods,
    reference_method="hierarchical-madea", fix_r=False, sp_parallelism=0,
    enable_plotting=False, loop_over="Nn",
  )
  assert calls == (["cycles"] if resume else ["old", "cycles"])
  results = json.loads((tmp_path / "experiments.json").read_text())
  assert results["hierarchical-madea"] == ["old-result"]
  assert results["hierarchical-madea-cycles"] == ["cycles-result"]


def test_next_cycle_preserves_hierarchy_state_and_resets_iteration_history(cycle_run, monkeypatch):
  snapshots = []
  checks = []
  real_cycle = madea_runner.run_madea_cycle

  def cycle(state, **kwargs):
    snapshots.append(deepcopy(state))
    result = real_cycle(state, **kwargs)
    snapshots.append(deepcopy(result))
    return result

  real_evaluate = madea.evaluate_bids

  def evaluate(*args, **kwargs):
    checks.append((deepcopy(args[8]["eta"]), kwargs["it"]))
    return real_evaluate(*args, **kwargs)

  monkeypatch.setattr(madea_runner, "run_madea_cycle", cycle)
  monkeypatch.setattr(madea, "evaluate_bids", evaluate)
  cycle_run([
    (False, None), (True, "UB/LB diff < tol"),
    (True, "all load assigned"),
  ], hierarchy_quantity=1, eta=[0.5, 0.3, 0.1])

  initial, end_first, start_second, end_second = snapshots
  assert checks == [([0.5, 0.3, 0.1], 0), ([0.5, 0.3, 0.1], 1), ([0.5, 0.3, 0.1], 0)]
  assert start_second.y[0, 1, 0] == end_first.y[0, 1, 0] + 1
  assert start_second.omega[0, 0] == end_first.omega[0, 0] - 1
  np.testing.assert_array_equal(start_second.p, end_first.p)
  assert start_second.fairness[0, 0] == end_first.fairness[0, 0] + 1
  np.testing.assert_array_equal(start_second.sp_r, [[1.], [2.]])
  np.testing.assert_array_equal(start_second.sp_rho, [0., 0.])
  assert initial.total_runtime < end_first.total_runtime < start_second.total_runtime < end_second.total_runtime
  assert start_second.best_centralized_cost > end_first.best_centralized_cost
  assert end_second.best_centralized_cost >= start_second.best_centralized_cost
  assert end_second.it == 0
  assert end_second.iterations == 3


def test_real_all_load_assigned_skips_hierarchy(tmp_path, monkeypatch):
  if not pyo.SolverFactory("glpk").available(exception_flag=False):
    pytest.skip("GLPK needed for the real MADEA cycle")
  data = _two_node_data()
  data[None]["incoming_load"] = {(1, 1): 1, (2, 1): 1}
  config = _materialized_config(tmp_path, data)

  def unexpected(*args, **kwargs):
    pytest.fail("all load assigned must terminate before the hierarchy")

  monkeypatch.setattr(HierarchicalAuctionEngine, "run_higher_levels", unexpected)
  folder = madea_runner.run(config, parallelism=0, disable_plotting=True)
  criterion = pd.read_csv(f"{folder}/termination_condition.csv").iloc[0, -1]
  assert criterion.startswith("all load assigned (")


def test_rerouting_without_served_load_or_welfare_gain_stops(tmp_path, monkeypatch):
  """Changed routes alone must not keep alternating complete cycles forever."""
  import networkx as nx

  nodes = range(1, 4)
  data = {None: {
    "Nn": {None: 3}, "Nf": {None: 1},
    "incoming_load": {(n, 1): 2 if n == 1 else 0 for n in nodes},
    "neighborhood": {(i, j): int(i != j) for i in nodes for j in nodes},
    "demand": {(n, 1): 1.0 for n in nodes},
    "memory_capacity": {n: 1 for n in nodes}, "memory_requirement": {1: 1},
    "max_utilization": {1: 1.0},
    "alpha": {(n, 1): 1.0 for n in nodes},
    "gamma": {(n, 1): 1.0 for n in nodes},
    "delta": {(n, 1): 0.0 for n in nodes},
    "beta": {(i, j, 1): 1.0 for i in nodes for j in nodes},
  }}
  graph = nx.complete_graph(3)
  monkeypatch.setattr(madea_runner, "init_problem", lambda *a: (data, {}, [0, 1, 2], graph))
  monkeypatch.setattr(madea_runner, "get_current_load", lambda *a: data[None]["incoming_load"])
  monkeypatch.setattr(madea_runner, "solve_subproblem", lambda *a: (
    data, np.zeros((3, 1)), None, None, np.array([[2.], [0.], [0.]]),
    np.ones((3, 1)), np.zeros(3), None,
    {"tot": 0}, {"tot": "optimal"}, {"tot": 0},
  ))
  events = []

  def cycle(state, **kwargs):
    events.append("madea")
    count = events.count("madea")
    assert count <= 2, "A plateau with different routes must still terminate"
    state.y.fill(0)
    state.y[0, count, 0] = 1
    state.omega = np.array([[1.], [0.], [0.]])
    state.best_centralized_solution = madea.combine_solutions(
      3, 1, data, data[None]["incoming_load"], np.zeros((3, 1)),
      state.sp_r, state.sp_rho, None, state.y, None, None, None, None,
    )
    state.best_centralized_cost = 0.0  # one forwarded, one rejected
    state.reason = "UB/LB diff < tol"
    state.iterations += 1
    return state

  def higher(self, **kwargs):
    events.append("hierarchy")
    return LevelResult(kwargs["y"].copy(), kwargs["omega"].copy())

  monkeypatch.setattr(madea_runner, "run_madea_cycle", cycle)
  monkeypatch.setattr(HierarchicalAuctionEngine, "run_higher_levels", higher)
  madea_runner.run({
    "base_solution_folder": str(tmp_path), "seed": 4850,
    "limits": {"load": {}}, "max_steps": 1, "checkpoint_interval": 1,
    "solver_name": "glpk", "solver_options": {}, "max_iterations": 100,
  }, parallelism=0, disable_plotting=True)
  assert events == ["madea", "hierarchy", "madea"]


@pytest.mark.parametrize("runner,column", [
  (madea, "FaaS-MADeA"), (madea_runner, "HierarchicalMADeACycles"),
])
def test_supplied_configuration_preserves_first_snapshot_welfare(tmp_path, runner, column):
  """Use the user's seed/topology/load, with a pre-refactor numerical baseline."""
  import json
  from pathlib import Path

  if not pyo.SolverFactory("glpk").available(exception_flag=False):
    pytest.skip("GLPK needed for the supplied-configuration integration")
  config_path = Path(__file__).resolve().parents[1] / "config_files/hierarchical_madea_cycles.json"
  config = json.loads(config_path.read_text())
  config.update(base_solution_folder=str(tmp_path), solver_name="glpk", max_run_time=0)
  config["solver_options"]["general"] = {"TimeLimit": 120, "mipgap": 1e-5}
  folder = runner.run(config, parallelism=0, log_on_file=True, disable_plotting=True)
  objectives = pd.read_csv(f"{folder}/obj.csv")[column].tolist()
  # Measured using the unmodified standalone runner on the same configuration.
  baseline = 150.9883071774954
  assert len(objectives) == 1
  if runner is madea:
    assert objectives[0] == pytest.approx(baseline)
  else:
    assert objectives[0] >= baseline - 1e-6
    from run import load_termination_condition
    termination = load_termination_condition(folder)
    assert termination.loc[0, "criterion"].startswith("no progress after hierarchy and MADEA")
