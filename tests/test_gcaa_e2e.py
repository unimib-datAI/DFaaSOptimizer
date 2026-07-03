from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest
from gurobipy import GurobiError
from parse import parse
from pyomo.common.errors import ApplicationError

from decentralized_gcaa import run as run_gcaa


def _require_gurobi() -> None:
  solver = pyo.SolverFactory("gurobi")
  if not solver.available(exception_flag=False):
    pytest.skip("Gurobi solver is not available")
  model = pyo.ConcreteModel()
  model.x = pyo.Var(bounds=(0, 1))
  model.objective = pyo.Objective(expr=model.x)
  try:
    solver.solve(model)
  except (ApplicationError, GurobiError) as exc:
    pytest.skip(f"Gurobi solver/license is not available: {exc}")


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
