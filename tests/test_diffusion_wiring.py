import json
from pathlib import Path

import numpy as np
import networkx as nx

import decentralized_diffusion
import run


def test_methods_choice_accepts_faas_diffuse(monkeypatch):
  argv = ["run.py", "-c", "config_files/planar_comparison.json",
          "--methods", "faas-diffuse"]
  monkeypatch.setattr("sys.argv", argv)
  args = run.parse_arguments()
  assert "faas-diffuse" in args.methods


def test_run_module_exposes_diffusion_runner():
  assert hasattr(run, "run_diffusion")
  assert callable(run.run_diffusion)


def test_planar_config_has_diffusion_section():
  config = json.loads(Path("config_files/planar_comparison.json").read_text())
  diffusion = config["solver_options"]["diffusion"]
  assert "latency_weight" in diffusion
  assert "fairness_weight" in diffusion


def test_compare_results_palette_includes_madig():
  import inspect

  import compare_results

  source = inspect.getsource(compare_results)
  assert '"FaaS-MADiG"' in source


def test_compare_results_defaults_include_madig(monkeypatch):
  monkeypatch.setattr("sys.argv", ["compare_results.py", "-i", "solutions/demo"])
  import compare_results

  args = compare_results.parse_arguments()

  assert "FaaS-MADiG" in args.models


def test_set_solution_folder_tolerates_missing_method_key():
  solution_folders = {"experiments_list": []}  # simulates a pre-feature experiments.json
  run.set_solution_folder(solution_folders, "faas-diffuse", 0, "/some/folder")
  assert solution_folders["faas-diffuse"][0] == "/some/folder"


def test_diffusion_run_uses_fixed_replicas_without_mutating_options(tmp_path, monkeypatch):
  seen = {}
  base_data = {
    None: {
      "Nn": {None: 1},
      "Nf": {None: 1},
      "neighborhood": {(1, 1): 0},
    }
  }

  monkeypatch.setattr(
    decentralized_diffusion,
    "init_problem",
    lambda *args, **kwargs: (base_data, {}, [], nx.empty_graph(1)),
  )
  monkeypatch.setattr(
    decentralized_diffusion,
    "load_solution",
    lambda folder, model: ("opt_solution", "opt_replicas", "opt_detailed", None, None),
    raising=False,
  )
  monkeypatch.setattr(
    decentralized_diffusion,
    "encode_solution",
    lambda Nn, Nf, solution, detailed, replicas, t: (None, None, None, np.array([[3]]), None),
    raising=False,
  )
  monkeypatch.setattr(decentralized_diffusion, "LSP", lambda: "LSP")
  monkeypatch.setattr(decentralized_diffusion, "LSP_fixedr", lambda: "LSP_fixedr", raising=False)

  def _solve_subproblem(sp_data, agents, sp, *args):
    seen["sp"] = sp
    seen["r_bar"] = dict(sp_data[None].get("r_bar", {}))
    return (
      sp_data,
      np.zeros((1, 1)),
      None,
      None,
      np.zeros((1, 1)),
      np.ones((1, 1)),
      np.zeros((1,)),
      np.zeros((1, 1)),
      {"tot": 0.0},
      {"tot": "ok"},
      {"tot": 0.0},
    )

  monkeypatch.setattr(decentralized_diffusion, "solve_subproblem", _solve_subproblem)
  monkeypatch.setattr(decentralized_diffusion, "get_current_load", lambda *args: {})
  monkeypatch.setattr(decentralized_diffusion, "update_data", lambda data, update: data)
  monkeypatch.setattr(
    decentralized_diffusion,
    "compute_residual_capacity",
    lambda *args: (np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))),
  )
  monkeypatch.setattr(
    decentralized_diffusion,
    "define_assignments",
    lambda *args, **kwargs: (
      __import__("pandas").DataFrame({"i": [], "j": [], "f": [], "d": [], "utility": []}),
      __import__("pandas").DataFrame({"i": [], "j": [], "f": []}),
      0,
    ),
  )
  monkeypatch.setattr(
    decentralized_diffusion,
    "combine_solutions",
    lambda *args: {"sp": {"x": np.zeros((1, 1)), "y": np.zeros((1, 1, 1)), "z": np.zeros((1, 1)), "r": np.ones((1, 1)), "U": np.zeros((1, 1))}},
  )
  monkeypatch.setattr(decentralized_diffusion, "compute_centralized_objective", lambda *args: 1.0)
  monkeypatch.setattr(decentralized_diffusion, "check_feasibility", lambda *args: (True, "ok"))
  monkeypatch.setattr(
    decentralized_diffusion,
    "decode_solutions",
    lambda sp_data, solution, complete, arg: (complete, None, 1.0),
  )
  monkeypatch.setattr(
    decentralized_diffusion,
    "join_complete_solution",
    lambda complete: ({}, {}, {}),
  )
  monkeypatch.setattr(decentralized_diffusion, "save_checkpoint", lambda *args: None)
  monkeypatch.setattr(decentralized_diffusion, "save_solution", lambda *args: None)

  config = {
    "base_solution_folder": str(tmp_path),
    "seed": 1,
    "limits": {"load": {"trace_type": "fixed_sum"}},
    "solver_name": "mock",
    "solver_options": {
      "general": {"TimeLimit": 10},
      "auction": {"unit_bids": True},
      "diffusion": {"latency_weight": 0.0, "fairness_weight": 0.0},
    },
    "max_iterations": 1,
    "patience": 1,
    "max_steps": 1,
    "min_run_time": 0,
    "max_run_time": 0,
    "run_time_step": 1,
    "checkpoint_interval": 1,
    "verbose": 0,
    "opt_solution_folder": "centralized-folder",
  }

  decentralized_diffusion.run(config, parallelism=0, disable_plotting=True)

  assert seen["sp"] == "LSP_fixedr"
  assert seen["r_bar"] == {(1, 1): 3}
  assert "unit_bids" not in config["solver_options"]["diffusion"]
