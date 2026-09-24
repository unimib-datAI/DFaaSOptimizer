"""Correct-behavior regressions for branch review findings 1, 11–13, 18–21."""

import json
import importlib
import inspect
from pathlib import Path

import networkx as nx
import numpy as np
import pyomo.environ as pyo
import pytest

from decentralized_powerd import sample_assignments
from generators.generate_data import generate_neighborhood
from generators.generate_load import generate_load_traces
from models import model as model_module
from models.model import BaseAbstractModel, LoadManagementModel, TightLoadManagementModel
from models.sp import LSPr, LSPr_x
from remote_experiments.definitions.paper import build_e0


@pytest.fixture
def glpk():
  solver = pyo.SolverFactory("glpk")
  if not solver.available(exception_flag=False):
    pytest.skip("GLPK executable required for numerical model regressions")
  return solver


def test_paper_suite_topology_can_be_generated():
  """#1: suite definitions and the generator must agree on topology schema."""
  experiment = build_e0(seeds=(1,), algorithms=("centralized",))[0]
  limits = experiment.config["limits"]
  nn = limits["Nn"]["min"]

  matrix, graph = generate_neighborhood(nn, limits, np.random.default_rng(1))

  assert matrix.shape == (nn, nn)
  assert nx.is_connected(graph)
  assert nx.check_planarity(graph)[0]
  assert graph.number_of_edges() == 15  # first pilot cell: 10 nodes, mean degree 3


def test_solver_options_do_not_leak_into_next_solve(glpk, monkeypatch, tmp_path):
  """#11: a previous solve's log destination must not be reused implicitly."""
  monkeypatch.setattr(model_module, "_SOLVER_CACHE", {})
  instance = pyo.ConcreteModel()
  instance.x = pyo.Var(bounds=(0, 2))
  instance.OBJ = pyo.Objective(expr=instance.x, sense=pyo.maximize)
  model = BaseAbstractModel()
  first_log = tmp_path / "first-solve.log"

  first = model.solve(instance, {"log": str(first_log)}, solver_name="glpk")
  assert first["obj"] == pytest.approx(2)
  assert first_log.exists()
  first_log.unlink()
  second = model.solve(instance, {}, solver_name="glpk")

  assert second["obj"] == pytest.approx(2)
  assert not first_log.exists(), "The second solve inherited the first solve's log option"


def test_tight_model_preserves_primary_welfare_at_large_load(glpk):
  """#12: replica minimization must not replace serving traffic with rejection."""
  data = {None: {
    "Nn": {None: 1}, "Nf": {None: 1},
    "incoming_load": {(1, 1): 20000},
    "demand": {(1, 1): 1}, "max_utilization": {1: 1},
    "memory_requirement": {1: 1}, "memory_capacity": {1: 20000},
    "alpha": {(1, 1): 1}, "gamma": {(1, 1): 0.1},
  }}
  original = LoadManagementModel().generate_instance(data)
  tight = TightLoadManagementModel().generate_instance(data)
  for instance in (original, tight):
    result = glpk.solve(instance)
    assert result.solver.termination_condition == pyo.TerminationCondition.optimal

  assert pyo.value(original.x[1, 1]) == pytest.approx(20000)
  # Serving all 20000 requests is feasible and yields welfare 1; rejecting
  # even one request lowers the primary objective, independently of replicas.
  assert pyo.value(tight.x[1, 1]) == pytest.approx(20000)
  assert pyo.value(tight.z[1, 1]) == pytest.approx(0)


@pytest.mark.parametrize("model_class", [LSPr, LSPr_x], ids=["LSPr", "LSPr_x"])
def test_local_reoptimization_accounts_for_all_rejected_load(glpk, model_class):
  """#13: one replica cannot serve both functions; rejection must be recorded."""
  data = {None: {
    "Nn": {None: 1}, "Nf": {None: 2},
    "incoming_load": {(1, 1): 1, (1, 2): 1},
    "demand": {(1, 1): 1, (1, 2): 1},
    "max_utilization": {1: 1, 2: 1},
    "memory_requirement": {1: 1, 2: 1}, "memory_capacity": {1: 1},
    "alpha": {(1, 1): 1, (1, 2): 0.9},
    "gamma": {(1, 1): 0.1, (1, 2): 100},
    "omega_bar": {(1, 1): 0, (1, 2): 0},
    "y_bar": {(1, 1, 1): 0, (1, 1, 2): 0},
    "x_bar": {(1, 1): 0, (1, 2): 1},
  }}
  instance = model_class().generate_instance(data)
  result = glpk.solve(instance)
  assert result.solver.termination_condition == pyo.TerminationCondition.optimal

  for f in (1, 2):
    accounted = pyo.value(instance.x[f] + instance.z[f])
    assert accounted == pytest.approx(1), f"Function {f} lost traffic"
  # Rejecting function 1 costs .1; rejecting function 2 costs 100.
  assert pyo.value(instance.x[2]) == pytest.approx(1)


def test_powerd_unit_bids_never_exceed_fractional_demand():
  """#18: unit bidding must stop when the remaining appetite is below one."""
  data = {None: {
    "Nn": {None: 2}, "Nf": {None: 1},
    "beta": {(1, 2, 1): 1.0}, "gamma": {(1, 1): 0.05},
    "memory_requirement": {1: 1},
  }}
  bids, _, _ = sample_assignments(
    omega=np.array([[2.5], [0.0]]),
    blackboard=np.array([[0.0], [5.0]]),
    data=data, neighborhood=np.array([[0, 1], [1, 0]]), rho=np.zeros(2),
    powerd_options={
      "d": 1, "criterion": "score", "unit_bids": True,
      "latency_weight": 0.0, "fairness_weight": 0.0,
    },
    latency=np.zeros((2, 2)), fairness=np.zeros((2, 1)),
    force_memory_bids=False, rng=np.random.default_rng(0),
  )

  assert 0 < bids["d"].sum() <= 2.5


@pytest.mark.parametrize("filename", ["planar_comparison.json", "planar_hierarchical.json"])
def test_shipped_planar_configuration_generates_connected_planar_graph(filename):
  """#19: shipped topology configurations must remain executable."""
  path = Path(__file__).resolve().parents[1] / "config_files" / filename
  limits = json.loads(path.read_text())["limits"]
  nn = limits["Nn"]["values"][0]

  matrix, graph = generate_neighborhood(nn, limits, np.random.default_rng(1))

  assert matrix.shape == (nn, nn)
  assert nx.is_connected(graph)
  assert nx.check_planarity(graph)[0]
  assert set(dict(graph.degree()).values()) == {3}


def test_planar_degree_alias_is_reproducible():
  limits = {"neighborhood": {"type": "planar", "degree": 3}}
  first, graph = generate_neighborhood(20, limits, np.random.default_rng(17))
  second, _ = generate_neighborhood(20, limits, np.random.default_rng(17))
  other, _ = generate_neighborhood(20, limits, np.random.default_rng(18))
  assert nx.is_connected(graph) and nx.check_planarity(graph)[0]
  assert set(dict(graph.degree()).values()) == {3}
  np.testing.assert_array_equal(first, second)
  assert not np.array_equal(first, other)


@pytest.mark.parametrize("model_class", [LSPr, LSPr_x])
def test_reoptimization_conserves_fractional_committed_forwarding(glpk, model_class):
  data = {None: {
    "Nn": {None: 1}, "Nf": {None: 1},
    "incoming_load": {(1, 1): 2}, "demand": {(1, 1): 1},
    "memory_requirement": {1: 1}, "memory_capacity": {1: 0},
    "omega_bar": {(1, 1): 0.5}, "y_bar": {(1, 1, 1): 0},
    "x_bar": {(1, 1): 0},
  }}
  instance = model_class().generate_instance(data)
  result = glpk.solve(instance)
  assert result.solver.termination_condition == pyo.TerminationCondition.optimal
  assert pyo.value(instance.z[1]) == pytest.approx(1.5)


def test_integer_fixed_sum_traces_preserve_system_workload():
  """#20: integer allocation must distribute rounding remainders, not lose them."""
  traces = generate_load_traces(
    {0: {0: 10, 1: 10}}, max_steps=20, seed=4850,
    trace_type="fixed_sum", enable_plotting=False,
  )
  workload = np.stack(list(traces[0].values()))

  assert np.all(workload >= 0)
  assert np.all(workload == np.floor(workload))
  np.testing.assert_array_equal(workload.sum(axis=0), np.full(20, 10))


@pytest.mark.parametrize("steps", [1, 2])
def test_short_sinusoidal_traces_are_finite_and_within_bounds(steps):
  """#21: fewer timesteps than distinct periods must still produce usable load."""
  traces = generate_load_traces(
    {0: {0: {"min": 1, "max": 10}}}, max_steps=steps, seed=4850,
    trace_type="sinusoidal", enable_plotting=False,
  )
  workload = traces[0][0]

  assert workload.shape == (steps,)
  assert np.isfinite(workload).all()
  assert np.all((workload >= 1) & (workload <= 10))
  assert np.all(workload == np.floor(workload))


@pytest.mark.parametrize("module_name,entry", [
  ("decentralized_dual", "run"),
  ("decentralized_diffusion", "run"),
  ("decentralized_powerd", "run"),
  ("decentralized_bestresponse", "run_br_s"),
  ("decentralized_gcaa", "run"),
])
def test_overloaded_runners_reach_stopping_check_with_correct_argument_types(
    glpk, tmp_path, monkeypatch, module_name, entry,
  ):
  # Reuse the small integration fixture; this test runs actual GLPK solves,
  # requiring no Gurobi license and no mocked stopping decisions.
  from test_e2e_gurobi_planar import _planar_e2e_config
  import pandas as pd

  config = _planar_e2e_config(tmp_path)
  config["solver_name"] = "glpk"
  config["max_iterations"] = 3
  config["limits"]["memory_capacity"] = {"values": [2, 24] * 5}
  config["solver_options"]["general"] = {"TimeLimit": 15}
  config["solver_options"].update({
    "diffusion": {"latency_weight": 0.0, "fairness_weight": 0.0, "unit_bids": False},
    "powerd": {"d": 2, "criterion": "score", "unit_bids": True},
    "br_s": {"latency_weight": 0.0, "fairness_weight": 0.0},
    "gcaa": {
      "unit_bids": True, "epsilon": 0.01, "latency_weight": 0.0, "fairness_weight": 0.0,
    },
  })

  module = importlib.import_module(module_name)
  stopping_check = module.check_stopping_criteria

  def checked_stopping(*args, **kwargs):
    # Observe the actual API boundary, then run the real decision. Incorrect
    # tolerance wiring can terminate early without raising an exception.
    bound = inspect.signature(stopping_check).bind(*args, **kwargs)
    bound.apply_defaults()
    assert bound.arguments["tolerance"] == config["tolerance"]
    assert bound.arguments["time_limit"] == 15
    return stopping_check(*args, **kwargs)

  monkeypatch.setattr(module, "check_stopping_criteria", checked_stopping)
  folder = getattr(module, entry)(
    config, parallelism=0, disable_plotting=True,
  )

  objectives = pd.read_csv(Path(folder) / "obj.csv").to_numpy(dtype=float)
  assert objectives.size > 0
  assert np.isfinite(objectives).all()
