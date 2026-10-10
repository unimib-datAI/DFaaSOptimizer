from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import decentralized_frankwolfe
import run


def test_frankwolfe_run_uses_fixed_replicas_and_does_not_mutate_config(tmp_path, monkeypatch):
  seen = {}
  base_data = {
    None: {
      "Nn": {None: 1},
      "Nf": {None: 1},
      "neighborhood": {(1, 1): 0},
    }
  }
  m = decentralized_frankwolfe
  monkeypatch.setattr(
    m, "init_problem", lambda *args, **kwargs: (base_data, {}, [], nx.empty_graph(1)),
  )
  monkeypatch.setattr(
    m, "load_solution",
    lambda folder, model: ("opt_solution", "opt_replicas", "opt_detailed", None, None),
  )
  monkeypatch.setattr(
    m, "encode_solution",
    lambda Nn, Nf, solution, detailed, replicas, t: (None, None, None, np.array([[3]]), None),
  )
  monkeypatch.setattr(m, "LSP", lambda: "LSP")
  monkeypatch.setattr(m, "LSP_fixedr", lambda: "LSP_fixedr")
  monkeypatch.setattr(m, "LSPr", lambda: seen.setdefault("spr", "LSPr"))
  monkeypatch.setattr(m, "LSPr_fixedr", lambda: seen.setdefault("spr", "LSPr_fixedr"))

  def _solve_subproblem(sp_data, agents, sp, *args):
    seen["sp"] = sp
    seen["r_bar"] = dict(sp_data[None].get("r_bar", {}))
    return (
      sp_data, np.zeros((1, 1)), None, None, np.zeros((1, 1)),
      np.ones((1, 1)), np.array([5.0]), np.zeros((1, 1)),
      {"tot": 0.0}, {"tot": "ok"}, {"tot": 0.0},
    )

  def _assignments(omega, y, x, r, rho, *args, **kwargs):
    seen["coordination_rho"] = np.array(rho, copy=True)
    return np.zeros_like(y), {"accepted": 0.0}

  monkeypatch.setattr(m, "solve_subproblem", _solve_subproblem)
  monkeypatch.setattr(m, "get_current_load", lambda *args: {})
  monkeypatch.setattr(m, "update_data", lambda data, update: data)
  monkeypatch.setattr(m, "frankwolfe_assignments", _assignments)
  monkeypatch.setattr(m, "fw_parameters", lambda data: {})
  monkeypatch.setattr(
    m, "combine_solutions",
    lambda *args: {"sp": {
      "x": np.zeros((1, 1)), "y": np.zeros((1, 1, 1)), "z": np.zeros((1, 1)),
      "r": np.ones((1, 1)), "U": np.zeros((1, 1)),
    }},
  )
  monkeypatch.setattr(m, "compute_centralized_objective", lambda *args: -1.0)
  monkeypatch.setattr(m, "check_feasibility", lambda *args: (True, "ok"))
  decoded = []

  def _decode(sp_data, solution, complete, arg):
    decoded.append(solution)
    return complete, None, 1.0

  monkeypatch.setattr(m, "decode_solutions", _decode)
  monkeypatch.setattr(m, "join_complete_solution", lambda complete: ({}, {}, {}))
  monkeypatch.setattr(m, "save_checkpoint", lambda *args: None)
  monkeypatch.setattr(m, "save_solution", lambda *args: None)

  config = {
    "base_solution_folder": str(tmp_path),
    "seed": 1,
    "limits": {"load": {"trace_type": "fixed_sum"}},
    "solver_name": "mock",
    "solver_options": {
      "general": {"TimeLimit": 10},
      "frankwolfe": {"kappa": 2.0},
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

  folder = m.run(config, parallelism=0, disable_plotting=True)

  assert seen["sp"] == "LSP_fixedr"
  assert seen["spr"] == "LSPr_fixedr"
  assert seen["r_bar"] == {(1, 1): 3}
  assert (seen["coordination_rho"] == 0).all()
  assert len(decoded) == 2
  assert config["solver_options"]["frankwolfe"] == {"kappa": 2.0}
  assert list(pd.read_csv(Path(folder, "obj.csv")).columns) == ["FaaS-MAFW"]
  assert Path(folder, "frankwolfe_stats.csv").exists()


def test_methods_choice_accepts_faas_fw(monkeypatch):
  argv = ["run.py", "-c", "config_files/planar_comparison.json", "--methods", "faas-fw"]
  monkeypatch.setattr("sys.argv", argv)
  assert "faas-fw" in run.parse_arguments().methods


def test_run_module_exposes_frankwolfe_runner_and_label():
  assert callable(run.run_frankwolfe)
  assert run.METHOD_RESULT_MODELS["faas-fw"] == ("LSPc", "FaaS-MAFW")


def test_compare_results_knows_mafw(monkeypatch):
  monkeypatch.setattr("sys.argv", ["compare_results.py", "-i", "solutions/demo"])
  import compare_results
  assert "FaaS-MAFW" in compare_results.parse_arguments().models
  assert "FaaS-MAFW" in compare_results.colors
  assert compare_results.colors["FaaS-MAFW"] not in {
    v for k, v in compare_results.colors.items() if k != "FaaS-MAFW"
  }


def test_remote_job_table_has_frankwolfe():
  pytest.importorskip("ray_dispatcher")
  from remote_experiments import jobs
  assert jobs.SCRIPT_BY_ALGORITHM["faas-fw"][0] == ("decentralized_frankwolfe.py",)


def test_frankwolfe_options_reject_unknown_rounding():
  with pytest.raises(ValueError):
    decentralized_frankwolfe.frankwolfe_options({"frankwolfe": {"rounding": "bogus"}})
