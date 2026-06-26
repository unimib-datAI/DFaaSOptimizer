import json
from pathlib import Path

import numpy as np
import networkx as nx

import decentralized_powerd


def test_powerd_run_uses_fixed_replicas_without_mutating_options(tmp_path, monkeypatch):
  seen = {}
  base_data = {
    None: {
      "Nn": {None: 1},
      "Nf": {None: 1},
      "neighborhood": {(1, 1): 0},
    }
  }

  monkeypatch.setattr(
    decentralized_powerd, "init_problem",
    lambda *args, **kwargs: (base_data, {}, [], nx.empty_graph(1)),
  )
  monkeypatch.setattr(
    decentralized_powerd, "load_solution",
    lambda folder, model: ("opt_solution", "opt_replicas", "opt_detailed", None, None),
    raising=False,
  )
  monkeypatch.setattr(
    decentralized_powerd, "encode_solution",
    lambda Nn, Nf, solution, detailed, replicas, t: (None, None, None, np.array([[3]]), None),
    raising=False,
  )
  monkeypatch.setattr(decentralized_powerd, "LSP", lambda: "LSP")
  monkeypatch.setattr(decentralized_powerd, "LSP_fixedr", lambda: "LSP_fixedr", raising=False)

  def _solve_subproblem(sp_data, agents, sp, *args):
    seen["sp"] = sp
    seen["r_bar"] = dict(sp_data[None].get("r_bar", {}))
    return (
      sp_data, np.zeros((1, 1)), None, None, np.zeros((1, 1)),
      np.ones((1, 1)), np.zeros((1,)), np.zeros((1, 1)),
      {"tot": 0.0}, {"tot": "ok"}, {"tot": 0.0},
    )

  monkeypatch.setattr(decentralized_powerd, "solve_subproblem", _solve_subproblem)
  monkeypatch.setattr(decentralized_powerd, "get_current_load", lambda *args: {})
  monkeypatch.setattr(decentralized_powerd, "update_data", lambda data, update: data)
  monkeypatch.setattr(
    decentralized_powerd, "compute_residual_capacity",
    lambda *args: (np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))),
  )
  monkeypatch.setattr(
    decentralized_powerd, "sample_assignments",
    lambda *args, **kwargs: (
      __import__("pandas").DataFrame({"i": [], "j": [], "f": [], "d": [], "utility": []}),
      __import__("pandas").DataFrame({"i": [], "j": [], "f": []}),
      0,
    ),
  )
  monkeypatch.setattr(
    decentralized_powerd, "combine_solutions",
    lambda *args: {"sp": {"x": np.zeros((1, 1)), "y": np.zeros((1, 1, 1)), "z": np.zeros((1, 1)), "r": np.ones((1, 1)), "U": np.zeros((1, 1))}},
  )
  monkeypatch.setattr(decentralized_powerd, "compute_centralized_objective", lambda *args: 1.0)
  monkeypatch.setattr(decentralized_powerd, "check_feasibility", lambda *args: (True, "ok"))
  monkeypatch.setattr(
    decentralized_powerd, "decode_solutions",
    lambda sp_data, solution, complete, arg: (complete, None, 1.0),
  )
  monkeypatch.setattr(
    decentralized_powerd, "join_complete_solution",
    lambda complete: ({}, {}, {}),
  )
  monkeypatch.setattr(decentralized_powerd, "save_checkpoint", lambda *args: None)
  monkeypatch.setattr(decentralized_powerd, "save_solution", lambda *args: None)

  config = {
    "base_solution_folder": str(tmp_path),
    "seed": 1,
    "limits": {"load": {"trace_type": "fixed_sum"}},
    "solver_name": "mock",
    "solver_options": {
      "general": {"TimeLimit": 10},
      "auction": {"unit_bids": True},
      "powerd": {"latency_weight": 0.0, "fairness_weight": 0.0},
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

  decentralized_powerd.run(config, parallelism=0, disable_plotting=True)

  assert seen["sp"] == "LSP_fixedr"
  assert seen["r_bar"] == {(1, 1): 3}
  # run() applied its defaults to a COPY, not the input config
  assert "d" not in config["solver_options"]["powerd"]
  assert "unit_bids" not in config["solver_options"]["powerd"]


import run


def test_methods_choice_accepts_faas_powd(monkeypatch):
  argv = ["run.py", "-c", "config_files/planar_comparison.json",
          "--methods", "faas-powd"]
  monkeypatch.setattr("sys.argv", argv)
  args = run.parse_arguments()
  assert "faas-powd" in args.methods


def test_run_module_exposes_powerd_runner():
  assert hasattr(run, "run_powerd")
  assert callable(run.run_powerd)


def test_planar_config_has_powerd_section():
  config = json.loads(Path("config_files/planar_comparison.json").read_text())
  powerd = config["solver_options"]["powerd"]
  assert powerd["d"] == 2
  assert powerd["criterion"] == "score"


def test_compare_results_palette_includes_mapod():
  import inspect
  import compare_results
  source = inspect.getsource(compare_results)
  assert '"FaaS-MAPoD"' in source


def test_compare_results_defaults_include_mapod(monkeypatch):
  monkeypatch.setattr("sys.argv", ["compare_results.py", "-i", "solutions/demo"])
  import compare_results
  args = compare_results.parse_arguments()
  assert "FaaS-MAPoD" in args.models
