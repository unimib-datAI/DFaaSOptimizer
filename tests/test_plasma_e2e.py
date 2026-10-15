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
  # The shared solution artifacts retain requests/second, independently of W.
  def _total(folder):
    sol = pd.read_csv(os.path.join(folder, "LSPc_solution.csv"))
    return sol.select_dtypes(include=[np.number]).to_numpy().sum()
  ratio = _total(folder2) / _total(folder1)
  assert ratio == pytest.approx(1.0)


from utils.faasmacro import compute_centralized_objective


def test_objective_preserves_zero_and_positive_fractional_loads():
  data = {None: {
    "Nn": {None: 1}, "Nf": {None: 2},
    "incoming_load": {(1, 1): 0, (1, 2): 0.5},
    "alpha": {(1, 1): 1, (1, 2): 2},
    "beta": {(1, 1, 1): 0, (1, 1, 2): 0},
    "gamma": {(1, 1): 1, (1, 2): 1},
  }}
  objective = compute_centralized_objective(
    data, np.array([[0, 0.5]]), np.zeros((1, 1, 2)), np.zeros((1, 2)),
  )
  assert objective == pytest.approx(2.0)
  assert data[None]["incoming_load"] == {(1, 1): 0, (1, 2): 0.5}


from solver_support import gurobi_unavailable_reason


def _solver_available():
  return gurobi_unavailable_reason() is None


@pytest.mark.skipif(not _solver_available(), reason="no MILP solver")
def test_plasma_gap_vs_milp_on_small_graph(tmp_path):
  from generators.generate_data import update_data
  from plasma.baselines.milp_baseline import solve_snapshot
  from run_centralized_model import init_problem
  from utils.centralized import get_current_load
  # PLASMA has no fixed turns (seeded arrival order), so one seed is one draw:
  # on 3 nodes the late-horizon gap to the oracle averages ~0.5 over seeds
  # 40-49 (it did with the former fixed index order too). Check the mean.
  gaps = []
  for seed in range(40, 45):
    config = _config(tmp_path / str(seed), Nn=3, max_steps=6)
    config["seed"] = seed
    config["solver_options"]["plasma"]["rounds_per_step"] = 30
    folder = run_plasma(config, parallelism=0)
    plasma_obj = pd.read_csv(os.path.join(folder, "obj.csv"))["Plasma"]
    oracle_folder = tmp_path / str(seed) / "oracle"
    oracle_folder.mkdir(parents=True)
    base, traces, agents, _ = init_problem(
      config["limits"], "sinusoidal", config["max_steps"], seed, str(oracle_folder),
    )
    # mirror plasma.runner.run's own t-range so oracle[t] and plasma_obj[t] match
    ub = config["max_run_time"]
    oracle = []
    for t in range(config["min_run_time"], ub, config["run_time_step"]):
      loadt = get_current_load(traces, agents, t)
      loadt = {k: int(round(v)) for k, v in loadt.items()}
      data = update_data(base, {"incoming_load": loadt})
      oracle.append(solve_snapshot(data, "gurobi", {"OutputFlag": 0})[4])
    assert len(oracle) == len(plasma_obj)
    late_p, late_o = plasma_obj.iloc[-2:].mean(), np.mean(oracle[-2:])
    gaps.append((late_o - late_p) / abs(late_o))
  assert np.mean(gaps) <= 0.6


@pytest.mark.parametrize("welfare", [False, True])
def test_plasma_frame_preserves_engine_across_steps(tmp_path, monkeypatch, welfare):
  from plasma.engine import PlasmaEngine
  from plasma.runner import run as run_plasma_runner
  from utils.common import load_requests_traces
  config = _config(tmp_path, max_steps=2)
  if welfare:
    config["solver_options"]["plasma_welfare"] = dict(
      W=2.0, rounds_per_step=5, epsilon=1e-6, hb_latency_rounds=1, hb_loss=0., staleness_rounds=3)
  else:
    config["solver_options"]["plasma"]["W"] = 2.0
  from plasma.welfare import WelfareEngine
  engines, calls = [], []
  init = PlasmaEngine.__init__

  def counting_init(self, *args, **kwargs):
    engines.append(self)
    init(self, *args, **kwargs)

  def recording(run_rounds):
    def recording_run_rounds(self, rounds, arrivals, **kwargs):
      calls.append((id(self), self.msg_count, arrivals.copy()))
      return run_rounds(self, rounds, arrivals, **kwargs)
    return recording_run_rounds

  monkeypatch.setattr(PlasmaEngine, "__init__", counting_init)
  engine_class = WelfareEngine if welfare else PlasmaEngine
  monkeypatch.setattr(engine_class, "run_rounds", recording(engine_class.run_rounds))
  folder = Path(run_plasma_runner(config, parallelism=0, welfare=welfare))
  assert len(engines) == 1 and len(calls) == 2                     # one engine for the whole run
  assert calls[0][0] == calls[1][0] and calls[1][1] > calls[0][1] > -1   # its state continues
  traces, *_ = load_requests_traces(str(folder))
  for t, (_, _, arrivals) in enumerate(calls):
    expected = np.array([[int(round(traces[f][n][t] * 2.0)) for f in sorted(traces)]
                         for n in sorted(traces[0])])
    np.testing.assert_array_equal(arrivals, expected)              # W scales the arrivals
  assert not np.array_equal(calls[0][2], calls[1][2])                # two different loads
  messages = pd.read_csv(folder / "plasma_messages.csv")
  assert messages["t"].tolist() == [0, 1] and (messages["msgs_per_node_s"] > 0).all()
  if welfare:
    assert pd.read_csv(folder / "plasma_welfare.csv")["time"].tolist() == [0, 1]
