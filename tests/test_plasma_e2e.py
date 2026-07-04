import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plasma.runner import run as run_plasma


def _config(tmp_path, Nn=3, max_steps=3):
  return {
    "base_solution_folder": str(tmp_path),
    "verbose": 0,
    "seed": 42,
    "max_steps": max_steps,
    "min_run_time": 0,
    # ub = max_run_time (unequal min/max branch, mirroring decentralized_gcaa's
    # range(min_run_time, ub, run_time_step)) -- must equal max_steps for the
    # loop to cover all max_steps timesteps
    "max_run_time": max_steps,
    "run_time_step": 1,
    "checkpoint_interval": 1,
    "solver_name": "none",
    "solver_options": {
      "plasma": {"rounds_per_step": 5, "k_sb": 2, "n_sb_steps": 100,
                 "n_hyst": 1}
    },
    "limits": {
      "Nn": {"min": Nn, "max": Nn},
      "Nf": {"min": 2, "max": 2},
      "neighborhood": {"m": Nn - 1},   # a line/tree on 3 nodes
      "demand": {"values": [1.0, 1.2]},
      "memory_capacity": {"min": 12, "max": 12},
      "memory_requirement": {"values": [2, 3]},
      "max_utilization": {"min": 0.65, "max": 0.75},
      "load": {"trace_type": "sinusoidal",
               "min": {"min": 5, "max": 10},
               "max": {"min": 20, "max": 30}},
      "weights": {"alpha": {"min": 1.0, "max": 1.5},
                  "beta_multiplier": {"min": 1.5, "max": 2.5},
                  "gamma": {"min": 0.05, "max": 0.15},
                  "delta_multiplier": {"min": 0.1, "max": 0.2}},
    },
  }


def test_runner_produces_lspc_artifacts(tmp_path):
  folder = run_plasma(_config(tmp_path), parallelism=0)
  for name in ("LSPc_solution.csv", "LSPc_offloaded.csv",
               "LSPc_utilization.csv", "LSPc_replicas.csv",
               "LSPc_detailed_fwd_solution.csv",
               "LSPc_residual_capacity.csv", "obj.csv", "runtime.csv",
               "termination_condition.csv", "plasma_messages.csv",
               "config.json"):
    assert os.path.exists(os.path.join(folder, name)), name


def test_runner_objective_column_is_plasma(tmp_path):
  folder = run_plasma(_config(tmp_path), parallelism=0)
  obj = pd.read_csv(os.path.join(folder, "obj.csv"))
  assert list(obj.columns) == ["Plasma"]
  assert len(obj) == 3


def test_runner_is_deterministic_given_seed(tmp_path):
  f1 = run_plasma(_config(tmp_path / "a"), parallelism=0)
  f2 = run_plasma(_config(tmp_path / "b"), parallelism=0)
  o1 = pd.read_csv(os.path.join(f1, "obj.csv"))
  o2 = pd.read_csv(os.path.join(f2, "obj.csv"))
  pd.testing.assert_frame_equal(o1, o2)


def test_runner_messages_bounded(tmp_path):
  folder = run_plasma(_config(tmp_path), parallelism=0)
  msgs = pd.read_csv(os.path.join(folder, "plasma_messages.csv"))
  assert (msgs["hb_per_node_s"] <= 2.0 + 1e-9).all()  # deg <= 2 on a tree of 3


import run as run_module


def test_methods_choice_accepts_plasma(monkeypatch):
  argv = ["run.py", "-c", "config_files/plasma_comparison.json",
          "--methods", "plasma"]
  monkeypatch.setattr("sys.argv", argv)
  args = run_module.parse_arguments()
  assert "plasma" in args.methods


def test_run_module_exposes_plasma_runner():
  assert callable(run_module.run_plasma)


def test_method_result_models_has_plasma_entry():
  assert run_module.METHOD_RESULT_MODELS["plasma"] == ("LSPc", "Plasma")


def test_plasma_comparison_config_exists_and_has_section():
  config = json.loads(
    Path("config_files/plasma_comparison.json").read_text()
  )
  assert "plasma" in config["solver_options"]


def test_set_solution_folder_tolerates_missing_plasma_key():
  solution_folders = {"experiments_list": []}
  run_module.set_solution_folder(solution_folders, "plasma", 0, "/x")
  assert solution_folders["plasma"][0] == "/x"


def test_runner_supports_w_not_one(tmp_path):
  cfg1 = _config(tmp_path / "w1")
  folder1 = run_plasma(cfg1, parallelism=0)
  cfg2 = _config(tmp_path / "w2")
  cfg2["solver_options"]["plasma"]["W"] = 2.0
  folder2 = run_plasma(cfg2, parallelism=0)  # must not trip check_feasibility
  obj = pd.read_csv(os.path.join(folder2, "obj.csv"))["Plasma"]
  assert np.isfinite(obj).all()
  # same seed => identical per-second load traces, so a W=2 window handles
  # ~2x the requests of a W=1 window (every LSPc_solution column is a
  # per-window request count)
  def _total(folder):
    sol = pd.read_csv(os.path.join(folder, "LSPc_solution.csv"))
    return sol.select_dtypes(include=[np.number]).to_numpy().sum()
  ratio = _total(folder2) / _total(folder1)
  assert 1.8 <= ratio <= 2.2, ratio
