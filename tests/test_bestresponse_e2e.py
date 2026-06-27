from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest

from decentralized_bestresponse import run_br_s, run_br_r, run_br_o


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
      "br_s": {"latency_weight": 0.0, "fairness_weight": 0.0},
      "br_r": {"latency_weight": 0.0, "fairness_weight": 0.0},
      "br_o": {"latency_weight": 0.0, "fairness_weight": 0.0},
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


def _assert_artifacts(folder, column):
  obj = pd.read_csv(Path(folder, "obj.csv"))
  assert column in obj.columns
  assert len(obj) >= 1
  assert np.isfinite(pd.to_numeric(obj[column], errors="coerce")).all()

  runtime = pd.read_csv(Path(folder, "runtime.csv"))
  assert "tot" in runtime.columns
  assert (runtime["tot"] >= 0).all()

  assert Path(folder, "termination_condition.csv").exists()
  assert Path(folder, "LSPc_solution.csv").exists()


def test_br_s_artifacts(tmp_path):
  _require_gurobi()
  _assert_artifacts(
    run_br_s(_e2e_config(tmp_path), parallelism=0, disable_plotting=True),
    "FaaS-MABR-S",
  )


def test_br_r_artifacts(tmp_path):
  _require_gurobi()
  _assert_artifacts(
    run_br_r(_e2e_config(tmp_path), parallelism=0, disable_plotting=True),
    "FaaS-MABR-R",
  )


def test_br_o_artifacts(tmp_path):
  _require_gurobi()
  _assert_artifacts(
    run_br_o(_e2e_config(tmp_path), parallelism=0, disable_plotting=True),
    "FaaS-MABR-O",
  )


def test_br_s_reproducible(tmp_path):
  _require_gurobi()
  fa = run_br_s(
    _e2e_config(tmp_path / "a"), parallelism=0, disable_plotting=True
  )
  fb = run_br_s(
    _e2e_config(tmp_path / "b"), parallelism=0, disable_plotting=True
  )
  oa = pd.read_csv(Path(fa, "obj.csv"))["FaaS-MABR-S"].to_numpy()
  ob = pd.read_csv(Path(fb, "obj.csv"))["FaaS-MABR-S"].to_numpy()
  assert oa.shape == ob.shape and oa.shape[0] >= 1
  assert np.allclose(oa, ob)


def test_br_r_reproducible(tmp_path):
  _require_gurobi()
  fa = run_br_r(
    _e2e_config(tmp_path / "a"), parallelism=0, disable_plotting=True
  )
  fb = run_br_r(
    _e2e_config(tmp_path / "b"), parallelism=0, disable_plotting=True
  )
  oa = pd.read_csv(Path(fa, "obj.csv"))["FaaS-MABR-R"].to_numpy()
  ob = pd.read_csv(Path(fb, "obj.csv"))["FaaS-MABR-R"].to_numpy()
  assert oa.shape == ob.shape and oa.shape[0] >= 1
  assert np.allclose(oa, ob)
