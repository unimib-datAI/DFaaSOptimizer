from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest

from decentralized_dual import run as run_dual


def _require_gurobi() -> None:
  solver = pyo.SolverFactory("gurobi")
  if not solver.available(exception_flag=False):
    pytest.skip("Gurobi solver is not available")


def _dual_e2e_config(base_solution_folder: Path) -> dict:
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
      "dual": {
        "alpha0": 1.0, "step_rule": "sqrt", "max_inner_iterations": 30,
        "gap_tolerance": 0.01, "latency_weight": 0.0,
        "fairness_weight": 0.0,
      },
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


def test_dual_runner_produces_expected_artifacts_with_gap(tmp_path):
  _require_gurobi()
  folder = run_dual(
    _dual_e2e_config(tmp_path), parallelism=0, disable_plotting=True
  )

  obj = pd.read_csv(Path(folder, "obj.csv"))
  assert len(obj) >= 1
  assert "FaaS-MALD" in obj.columns
  assert np.isfinite(pd.to_numeric(obj["FaaS-MALD"], errors="coerce")).all()

  runtime = pd.read_csv(Path(folder, "runtime.csv"))
  assert "tot" in runtime.columns
  assert (runtime["tot"] >= 0).all()

  tc = pd.read_csv(Path(folder, "termination_condition.csv"))
  assert tc.iloc[:, -1].astype(str).str.contains("gap:").all()
  assert Path(folder, "LSPc_solution.csv").exists()


def test_dual_runner_is_reproducible_for_same_seed(tmp_path):
  _require_gurobi()
  folder_a = run_dual(
    _dual_e2e_config(tmp_path / "a"), parallelism=0, disable_plotting=True
  )
  folder_b = run_dual(
    _dual_e2e_config(tmp_path / "b"), parallelism=0, disable_plotting=True
  )

  obj_a = pd.read_csv(Path(folder_a, "obj.csv"))["FaaS-MALD"].to_numpy()
  obj_b = pd.read_csv(Path(folder_b, "obj.csv"))["FaaS-MALD"].to_numpy()

  assert obj_a.shape[0] >= 1
  assert obj_a.shape == obj_b.shape
  assert np.allclose(obj_a, obj_b)


def test_dual_defaults_applied_without_dual_options_key(tmp_path):
  _require_gurobi()
  config = _dual_e2e_config(tmp_path)
  del config["solver_options"]["dual"]
  folder = run_dual(config, parallelism=0, disable_plotting=True)
  assert Path(folder, "obj.csv").exists()
