"""PLASMA scoring preserves true zero and positive fractional demand."""

import numpy as np
import pyomo.environ as pyo
import pytest

from plasma.baselines import milp_baseline
from plasma.core.types import PlasmaOptions
from plasma.engine import PlasmaEngine
from plasma.runner import build_nodes
from utils.faasmacro import compute_centralized_objective


def _one_node_data(load):
  return {None: {
    "Nn": {None: 1}, "Nf": {None: 1},
    "incoming_load": {(1, 1): load}, "neighborhood": {(1, 1): 0},
    "alpha": {(1, 1): 2.0}, "beta": {(1, 1, 1): 0.0},
    "gamma": {(1, 1): 0.1}, "demand": {(1, 1): 1.0},
    "max_utilization": {1: 1.0}, "memory_capacity": {1: 1},
    "memory_requirement": {1: 1},
  }}


def test_plasma_zero_arrivals_have_zero_finite_welfare():
  data = _one_node_data(0)
  opts = PlasmaOptions(k_sb=0)
  engine = PlasmaEngine(build_nodes(data, opts, seed=0), opts,
                        np.random.default_rng(0))

  result = engine.run_rounds(2, np.array([[0]]))
  objective = compute_centralized_objective(data, result.x, result.y, result.z)

  assert objective == 0.0
  np.testing.assert_array_equal(result.x + result.y.sum(axis=1) + result.z, [[0]])
  assert data[None]["incoming_load"][(1, 1)] == 0


def test_stale_baseline_preserves_fractional_load_normalization(monkeypatch):
  data = _one_node_data(0.5)

  def solve_fractional_snapshot(data, solver_name, solver_options):
    return np.array([[0.5]]), np.zeros((1, 1, 1)), np.zeros((1, 1)), \
      np.ones((1, 1)), 2.0

  monkeypatch.setattr(milp_baseline, "solve_snapshot", solve_fractional_snapshot)
  objectives = milp_baseline.stale_objectives(
    data, {0: {0: [0.5, 0.5]}}, [0], range(2), "unused", {}, resolve_every=2,
  )

  # Every arrival is served: alpha * 0.5 / 0.5 = 2, including the held step.
  assert objectives == [2.0, 2.0]


def test_milp_snapshot_with_zero_load_has_no_phantom_traffic(monkeypatch):
  if not pyo.SolverFactory("glpk").available(exception_flag=False):
    pytest.skip("GLPK is needed for the zero-load baseline integration")
  import models.model as model_module

  monkeypatch.setattr(model_module, "_SOLVER_CACHE", {})
  data = _one_node_data(0)

  x, y, z, _, objective = milp_baseline.solve_snapshot(data, "glpk", {})

  assert objective == 0.0
  assert (x.sum(), y.sum(), z.sum()) == (0.0, 0.0, 0.0)
  assert data[None]["incoming_load"][(1, 1)] == 0
