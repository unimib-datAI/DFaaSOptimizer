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
    "neighborhood": {"p": 1.0},
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
