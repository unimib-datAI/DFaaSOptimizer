"""TightLoadManagementModel matches LoadManagementModel on random instances."""
import numpy as np
import pytest

from generators.generate_data import generate_data, update_data
from generators.generate_load import generate_load_traces
from models.model import LoadManagementModel, TightLoadManagementModel
from utils.centralized import get_current_load

pytest.importorskip("gurobipy")

LIMITS = {
  "Nn": {"min": 15, "max": 15},
  "Nf": {"min": 2, "max": 2},
  "neighborhood": {"p": 0.3},
  "demand": {"values": [1.0, 1.2]},
  "memory_capacity": {"min": 12, "max": 12},
  "memory_requirement": {"values": [2, 3]},
  "max_utilization": {"min": 0.65, "max": 0.75},
  "load": {
    "trace_type": "sinusoidal",
    "min": {"min": 5, "max": 10},
    "max": {"min": 50, "max": 100},
  },
  "weights": {
    "alpha": {"min": 1.0, "max": 1.5},
    "beta_multiplier": {"min": 1.5, "max": 2.5},
    "gamma": {"min": 0.05, "max": 0.15},
    "delta_multiplier": {"min": 0.1, "max": 0.2},
  },
}
SOLVER_OPTIONS = {"MIPGap": 1e-6, "TimeLimit": 120, "OutputFlag": 0}


def _solve(model_cls, data):
  M = model_cls()
  instance = M.generate_instance(data)
  solution = M.solve(instance, SOLVER_OPTIONS, "gurobi")
  assert solution["solution_exists"], solution["termination_condition"]
  return solution


@pytest.mark.parametrize("seed", [42, 7])
def test_tight_model_matches_original(seed):
  rng = np.random.default_rng(seed=seed)
  base_data, load_limits, _ = generate_data("random", rng=rng, limits=LIMITS)
  traces = generate_load_traces(
    load_limits, 3, seed, "sinusoidal", None, enable_plotting=False
  )
  load = get_current_load(traces, list(range(15)), 0)
  data = update_data(base_data, {"incoming_load": load})

  original = _solve(LoadManagementModel, data)
  tight = _solve(TightLoadManagementModel, data)

  # same objective up to the epsilon replica penalty and MIP gap
  penalty = 1e-4 * sum(tight["r"])
  assert tight["obj"] + penalty == pytest.approx(original["obj"], rel=1e-4)
  # replicas stay minimal even without utilization_equilibrium2
  assert sum(tight["r"]) <= sum(original["r"]) + 1e-6
