from pathlib import Path

import numpy as np
import pandas as pd

from utils import step_time
import pyomo.environ as pyo
import pytest

from decentralized_bestresponse import run_br_s, run_br_r, run_br_o


from solver_support import require_gurobi as _require_gurobi


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
  assert tuple(runtime.columns) == step_time.COLUMNS
  assert (runtime["algorithm_s"] >= 0).all() and (runtime["wall_s"] >= runtime["algorithm_s"]).all()

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


@pytest.mark.parametrize("module_name, runner_name", [
  ("decentralized_bestresponse", "run_br_s"), ("decentralized_bestresponse", "run_br_o"),
  ("decentralized_potentialgame", "run_pg_s"), ("decentralized_potentialgame", "run_pg_r"),
])
def test_sequential_sweeps_run_on_the_step_clock_with_untimed_observers(tmp_path, monkeypatch,
                                                                        module_name, runner_name):
  import importlib
  import sys
  import time
  sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "harmonization"))
  from capture_baseline import capture_config, instances
  module = importlib.import_module(module_name)
  original, calls = module.compute_centralized_objective, []

  def slow_objective(*args, **kwargs):  # observer: global welfare of the incumbent
    calls.append(1)
    time.sleep(0.02)
    return original(*args, **kwargs)

  monkeypatch.setattr(module, "compute_centralized_objective", slow_objective)
  config = capture_config(instances()[0], tmp_path, runner_name)
  config["verbose"] = 0
  folder = getattr(module, runner_name)(config, 0, log_on_file=True, disable_plotting=True)
  runtime = pd.read_csv(Path(folder, "runtime.csv"))
  assert tuple(runtime.columns) == step_time.COLUMNS and runtime["step"].tolist() == [0, 1]
  assert calls and (runtime["wall_s"] - runtime["algorithm_s"]).sum() >= 0.02 * len(calls) * 0.9
  assert (runtime["parallel_phases"] >= 1).all()          # the parallel initialization stays a phase
  termination = pd.read_csv(Path(folder, "termination_condition.csv")).iloc[:, 1]
  totals = [float(t.rsplit("total runtime: ", 1)[1].rstrip(")")) for t in termination]
  assert all(total <= a + 1e-9 for total, a in zip(totals, runtime["algorithm_s"]))
  assert not step_time.active()
