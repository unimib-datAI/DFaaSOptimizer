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
