import json
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd

import decentralized_gcaa
import run


def test_methods_choice_accepts_faas_gcaa(monkeypatch):
  argv = ["run.py", "-c", "config_files/planar_comparison.json",
          "--methods", "faas-gcaa"]
  monkeypatch.setattr("sys.argv", argv)
  args = run.parse_arguments()
  assert "faas-gcaa" in args.methods


def test_run_module_exposes_gcaa_runner():
  assert hasattr(run, "run_gcaa")
  assert callable(run.run_gcaa)


def test_method_result_models_has_gcaa_entry():
  assert run.METHOD_RESULT_MODELS["faas-gcaa"] == ("LSPc", "FaaS-MAGCAA")


def test_planar_config_has_gcaa_section():
  config = json.loads(Path("config_files/planar_comparison.json").read_text())
  gcaa = config["solver_options"]["gcaa"]
  assert gcaa["unit_bids"] is True
  assert "latency_weight" in gcaa
  assert "fairness_weight" in gcaa


def test_set_solution_folder_tolerates_missing_method_key():
  solution_folders = {"experiments_list": []}
  run.set_solution_folder(solution_folders, "faas-gcaa", 0, "/some/folder")
  assert solution_folders["faas-gcaa"][0] == "/some/folder"


def test_gcaa_run_stops_when_no_bids_available(tmp_path, monkeypatch):
  base_data = {
    None: {
      "Nn": {None: 1},
      "Nf": {None: 1},
      "neighborhood": {(1, 1): 0},
    }
  }
  monkeypatch.setattr(
    decentralized_gcaa, "init_problem",
    lambda *args, **kwargs: (base_data, {}, [], nx.empty_graph(1)),
  )
  monkeypatch.setattr(decentralized_gcaa, "get_current_load", lambda *args: {})
  monkeypatch.setattr(decentralized_gcaa, "update_data", lambda data, update: data)
  monkeypatch.setattr(decentralized_gcaa, "LSP", lambda: "LSP")
  monkeypatch.setattr(decentralized_gcaa, "LSPr", lambda: "LSPr")

  def _solve_subproblem(sp_data, agents, sp, *args):
    return (
      sp_data,
      np.zeros((1, 1)),
      None,
      None,
      np.zeros((1, 1)),   # sp_omega: no residual load -> no bids
      np.ones((1, 1)),
      np.array([0.0]),
      np.zeros((1, 1)),
      {"tot": 0.0},
      {"tot": "ok"},
      {"tot": 0.0},
    )

  monkeypatch.setattr(decentralized_gcaa, "solve_subproblem", _solve_subproblem)
  monkeypatch.setattr(
    decentralized_gcaa, "compute_residual_capacity",
    lambda *args: (np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))),
  )
  monkeypatch.setattr(
    decentralized_gcaa, "define_bids",
    lambda *args, **kwargs: (
      pd.DataFrame({"i": [], "j": [], "f": [], "d": [], "b": [], "utility": []}),
      pd.DataFrame({"i": [], "j": [], "f": []}),
      1,
    ),
  )
  monkeypatch.setattr(
    decentralized_gcaa, "combine_solutions",
    lambda *args: {"sp": {
      "x": np.zeros((1, 1)), "y": np.zeros((1, 1, 1)),
      "z": np.zeros((1, 1)), "r": np.ones((1, 1)), "U": np.zeros((1, 1)),
    }},
  )
  monkeypatch.setattr(decentralized_gcaa, "compute_centralized_objective", lambda *args: -1.0)
  monkeypatch.setattr(decentralized_gcaa, "check_feasibility", lambda *args: (True, "ok"))

  decoded = []

  def _decode(sp_data, solution, complete, arg):
    decoded.append(solution)
    return complete, None, 1.0

  monkeypatch.setattr(decentralized_gcaa, "decode_solutions", _decode)
  monkeypatch.setattr(
    decentralized_gcaa, "join_complete_solution", lambda complete: ({}, {}, {})
  )
  monkeypatch.setattr(decentralized_gcaa, "save_checkpoint", lambda *args: None)
  monkeypatch.setattr(decentralized_gcaa, "save_solution", lambda *args: None)

  config = {
    "base_solution_folder": str(tmp_path),
    "seed": 1,
    "limits": {"load": {"trace_type": "fixed_sum"}},
    "solver_name": "mock",
    "solver_options": {
      "general": {"TimeLimit": 10},
      "gcaa": {
        "unit_bids": True, "epsilon": 0.01,
        "latency_weight": 0.0, "fairness_weight": 0.0,
      },
    },
    "max_iterations": 5,
    "max_steps": 1,
    "min_run_time": 0,
    "max_run_time": 0,
    "run_time_step": 1,
    "checkpoint_interval": 1,
    "verbose": 0,
  }

  decentralized_gcaa.run(config, parallelism=0, disable_plotting=True)

  assert len(decoded) == 2
