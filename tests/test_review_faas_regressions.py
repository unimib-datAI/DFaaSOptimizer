"""Regression cases for review findings 2, 9, 10 and 25.

These assert the intended behavior and deliberately fail until the bugs are
fixed. GLPK cases need the optional system executable ``glpsol``, not Gurobi.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest

import models.model as model_module
import run_faasmacro as macro
import run_faasmadea as madea
from models.sp import LSP


@pytest.fixture
def glpk(monkeypatch):
  if not pyo.SolverFactory("glpk").available(exception_flag=False):
    pytest.skip("This numerical regression needs the GLPK executable glpsol")
  # Solver options are process-global in production; isolate every test.
  monkeypatch.setattr(model_module, "_SOLVER_CACHE", {})
  return "glpk"


@pytest.mark.parametrize("runner", [madea, macro], ids=["madea", "macro"])
@pytest.mark.parametrize("gamma", [0.0, 0.1], ids=["zero", "negative"])
def test_runner_preserves_nonpositive_feasible_incumbent(tmp_path, glpk, runner, gamma):
  """With no RAM all traffic must be rejected, with welfare exactly -2*gamma."""
  config = {
    "base_solution_folder": str(tmp_path),
    "seed": 21,
    "limits": {
      "Nn": {"min": 2, "max": 2},
      "Nf": {"min": 1, "max": 1},
      "neighborhood": {"p": 1.0},
      "weights": {
        "alpha": {"min": 1.0, "max": 1.0},
        "beta_multiplier": {"min": 1.5, "max": 1.5},
        "gamma": {"min": gamma, "max": gamma},
        "delta_multiplier": {"min": 0.1, "max": 0.1},
      },
      "demand": {"values": [1.0]},
      "memory_capacity": {"values": [0, 0]},
      "memory_requirement": {"values": [1]},
      "max_utilization": {"min": 0.7, "max": 0.7},
      "load": {
        "trace_type": "clipped",
        "min": {"min": 2.0, "max": 2.0},
        "max": {"min": 3.0, "max": 3.0},
      },
    },
    "solver_name": glpk,
    "solver_options": {
      "general": {},
      "auction": {"epsilon": 0.01, "eta": 0.5, "zeta": 0.1},
    },
    "max_iterations": 2,
    "patience": 1,
    "sw_patience": 1,
    "max_steps": 4,
    "min_run_time": 1,
    "max_run_time": 1,
    "run_time_step": 1,
    "checkpoint_interval": 1,
    "tolerance": 1e-6,
    "verbose": 0,
  }
  folder = runner.run(config, parallelism=0, disable_plotting=True)
  assert folder is not None, "The feasible all-rejected solution must be exported"
  objectives = pd.read_csv(Path(folder) / "obj.csv")
  column = "FaaS-MADeA" if runner is madea else "FaaS-MACrO"
  assert objectives[column].tolist() == pytest.approx([-2 * gamma])
  assert (Path(folder) / "LSPc_solution.csv").is_file()


@pytest.mark.parametrize("parallelism", [0, 2], ids=["sequential", "parallel"])
def test_detailed_prices_control_each_agents_offload_decision(glpk, parallelism):
  """Price 7 favors offloading; price 11 exceeds the rejection penalty 10."""
  data = {None: {
    "Nn": {None: 2},
    "Nf": {None: 1},
    "incoming_load": {(1, 1): 10, (2, 1): 10},
    "memory_capacity": {1: 0, 2: 0},
    "memory_requirement": {1: 1},
    "demand": {(1, 1): 1.0, (2, 1): 1.0},
    "max_utilization": {1: 1.0},
    "gamma": {(1, 1): 10.0, (2, 1): 10.0},
  }}
  result = macro.solve_subproblem(
    data, [0, 1], LSP(), glpk, {}, parallelism,
    pi={1: 3.0}, detailed_pi=np.array([[7.0], [11.0]]),
  )
  np.testing.assert_allclose(result[3], [[0.0], [10.0]])  # rejected
  np.testing.assert_allclose(result[4], [[10.0], [0.0]])  # offloaded
  assert result[0][None]["pi"] == {1: 3.0}, "An agent must not overwrite global prices"


def test_madea_uses_each_available_replica_slot_once():
  """Three unit bids fit in the seller's three free memory slots."""
  data = {None: {
    "Nn": {None: 4},
    "Nf": {None: 1},
    "memory_requirement": {1: 1},
    "demand": {(n + 1, 1): 1.0 for n in range(4)},
    "max_utilization": {1: 1.0},
  }}
  bids = pd.DataFrame([
    {"i": i, "j": 3, "f": 0, "b": 3 - i, "d": 1.0} for i in range(3)
  ])
  y, _, replicas, _ = madea.evaluate_bids(
    bids, np.zeros((4, 1)), data,
    initial_rho=np.array([0.0, 0.0, 0.0, 3.0]),
    tentatively_start_replicas=True,
  )
  np.testing.assert_allclose(y[:3, 3, 0], [1.0, 1.0, 1.0])
  assert replicas[3, 0] == 3


@pytest.mark.parametrize("module", ["rlagents.ma_environment", "rlagents.sa_environment"])
def test_uv_environment_can_import_training_environments(module):
  """Import only: never start train_rl_agent.py or a Ray training session."""
  result = subprocess.run(
    [sys.executable, "-c", f"import {module}"],
    cwd=Path(__file__).resolve().parents[1],
    capture_output=True, text=True, timeout=30, check=False,
  )
  assert result.returncode == 0, result.stderr


def test_madea_reuses_started_replica_after_free_memory_is_exhausted():
  data = {None: {
    "Nn": {None: 2}, "Nf": {None: 1},
    "memory_requirement": {1: 1},
    "demand": {(1, 1): 0.25, (2, 1): 0.25},
    "max_utilization": {1: 1.0},
  }}
  bids = pd.DataFrame([
    {"i": 0, "j": 1, "f": 0, "b": 3 - i, "d": 1.0} for i in range(3)
  ])
  y, _, replicas, _ = madea.evaluate_bids(
    bids, np.zeros((2, 1)), data,
    initial_rho=np.array([0.0, 1.0]), tentatively_start_replicas=True,
  )
  assert y[0, 1, 0] == 3.0
  assert replicas[1, 0] == 1.0
