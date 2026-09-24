"""TightLoadManagementModel matches LoadManagementModel on random instances."""
import numpy as np
import pytest

from generators.generate_data import generate_data, update_data
from generators.generate_load import generate_load_traces
from models.model import LoadManagementModel, TightLoadManagementModel
from utils.centralized import get_current_load

from solver_support import gurobi_unavailable_reason

pytestmark = pytest.mark.skipif(
  gurobi_unavailable_reason() is not None,
  reason=gurobi_unavailable_reason() or "Gurobi license required",
)

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

  assert tight["obj"] == pytest.approx(original["obj"], rel=1e-4)
  x = np.array(tight["x"]).reshape(15, 2)
  y = np.array(tight["y"]).reshape(15, 15, 2)
  r = np.array(tight["r"]).reshape(15, 2)
  for n in range(15):
    for f in range(2):
      served = data[None]["demand"][(n + 1, f + 1)] * (x[n, f] + y[:, n, f].sum())
      utilization = data[None]["max_utilization"][f + 1]
      assert (r[n, f] - 1) * utilization <= served + 1e-6
      assert served <= r[n, f] * utilization + 1e-6
