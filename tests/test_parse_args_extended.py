"""Tests for parse_arguments across all modules and remaining edge cases."""

import sys

import numpy as np
import pandas as pd


def test_decentralized_auction_parse_args(monkeypatch):
  monkeypatch.setattr(
    "sys.argv",
    ["prog", "-c", "config_files/test.json", "-j", "2", "--disable_plotting"],
  )
  from decentralized_auction import parse_arguments
  args = parse_arguments()
  assert args.config == "config_files/test.json"
  assert args.parallelism == 2
  assert args.disable_plotting is True


def test_decentralized_auction_parse_args_defaults(monkeypatch):
  monkeypatch.setattr("sys.argv", ["prog"])
  from decentralized_auction import parse_arguments
  args = parse_arguments()
  assert args.config == "manual_config.json"
  assert args.parallelism == -1
  assert args.disable_plotting is False


def test_run_centralized_parse_args(monkeypatch):
  monkeypatch.setattr(
    "sys.argv", ["prog", "-c", "my_config.json", "--disable_plotting"],
  )
  from run_centralized_model import parse_arguments
  args = parse_arguments()
  assert args.config == "my_config.json"
  assert args.disable_plotting is True


def test_run_faasmacro_parse_args(monkeypatch):
  monkeypatch.setattr(
    "sys.argv", ["prog", "-c", "cfg.json", "-j", "4"],
  )
  from run_faasmacro import parse_arguments
  args = parse_arguments()
  assert args.config == "cfg.json"
  assert args.parallelism == 4


def test_run_faasmadea_parse_args(monkeypatch):
  monkeypatch.setattr(
    "sys.argv", ["prog", "--disable_plotting"],
  )
  from run_faasmadea import parse_arguments
  args = parse_arguments()
  assert args.disable_plotting is True


def test_what_if_parse_args(monkeypatch):
  monkeypatch.setattr(
    "sys.argv", ["prog", "-f", "/tmp/experiment", "-m", "0", "30", "60"],
  )
  from what_if_analysis import parse_arguments
  args = parse_arguments()
  assert args.base_folder == "/tmp/experiment"
  assert args.milestones == ["0", "30", "60"]


def test_postprocessing_parse_args(monkeypatch):
  monkeypatch.setattr(
    "sys.argv", ["prog", "-i", "/tmp/results", "--models", "A", "B"],
  )
  from postprocessing import parse_arguments
  args = parse_arguments()
  assert args.postprocessing_folder == "/tmp/results"
  assert args.models == ["A", "B"]


def test_hierarchical_runner_parse_args(monkeypatch):
  monkeypatch.setattr(
    "sys.argv", ["prog", "-c", "hier_config.json", "-j", "0"],
  )
  from hierarchical_auction.runner import parse_arguments
  args = parse_arguments()
  assert args.config == "hier_config.json"
  assert args.parallelism == 0


def test_run_py_parse_args_methods(monkeypatch):
  monkeypatch.setattr(
    "sys.argv", ["run.py", "-c", "config_files/config.json", "--methods", "hierarchical"],
  )
  from run import parse_arguments
  args = parse_arguments()
  assert args.methods == ["hierarchical"]


# --- Additional decentralized_auction edge cases ---

def test_check_stopping_criteria_no_capacity_left():
  from decentralized_auction import check_stopping_criteria
  stop, why = check_stopping_criteria(
    0, 5,
    blackboard=np.zeros((2, 2)),
    omega=np.ones((2, 2)),
    rmp_omega=np.ones((2, 2)),
    bids=pd.DataFrame({"a": [1]}),
    memory_bids=pd.DataFrame({"b": [1]}),
    tolerance=1e-6,
    total_runtime=1.0,
    time_limit=100.0,
  )
  assert stop is True
  assert why == "no capacity left"


def test_check_stopping_criteria_load_cannot_be_assigned():
  from decentralized_auction import check_stopping_criteria
  stop, why = check_stopping_criteria(
    0, 5,
    blackboard=np.ones((2, 2)),
    omega=np.ones((2, 2)),
    rmp_omega=np.zeros((2, 2)),
    bids=pd.DataFrame({"a": [1]}),
    memory_bids=pd.DataFrame({"b": [1]}),
    tolerance=1e-6,
    total_runtime=1.0,
    time_limit=100.0,
  )
  assert stop is True
  assert "cannot be assigned" in why


# --- Additional postprocessing test ---

def test_postprocessing_load_solution_formats_columns(tmp_path):
  from postprocessing import load_solution

  solution = pd.DataFrame({
    "n0_f0_loc": [1.0, 2.0],
    "n0_f0_fwd": [0.5, 0.5],
    "n0_f0": [0.0, 0.0],
    "n1_f0_loc": [3.0, 4.0],
    "n1_f0_fwd": [0.0, 0.5],
    "n1_f0": [0.0, 0.0],
  })
  replicas = pd.DataFrame({"n0_f0": [1.0, 2.0], "n1_f0": [2.0, 3.0]})
  detailed = pd.DataFrame({
    "n0_f0_n1_tot": [1.0, 0.5],
    "n1_f0_n0_tot": [0.0, 2.0],
  })
  utilization = pd.DataFrame({"n0_f0": [0.5, 0.6], "n1_f0": [0.3, 0.4]})
  pd.DataFrame({"SP/coord": [10.0, 12.0]}).to_csv(tmp_path / "obj.csv", index=False)
  solution.to_csv(tmp_path / "ModelX_solution.csv", index=False)
  replicas.to_csv(tmp_path / "ModelX_replicas.csv", index=False)
  detailed.to_csv(tmp_path / "ModelX_detailed_fwd_solution.csv", index=False)
  utilization.to_csv(tmp_path / "ModelX_utilization.csv", index=False)

  sol, rep, det, util, obj = load_solution(str(tmp_path), "ModelX")

  assert "n0_f0_loc" in sol.columns
  assert det.loc[0, "n0_f0_n1_tot"] == 1.0
  assert rep.loc[1, "n1_f0"] == 3.0


# --- Additional load_generator test ---

def test_load_generator_initialization_and_trace_limits():
  from load_generator import LoadGenerator

  lg = LoadGenerator(average_requests=50, amplitude_requests=25)
  assert lg.average_requests == 50
  assert lg.amplitude_requests == 25
