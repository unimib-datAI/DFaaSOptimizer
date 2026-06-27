import numpy as np
import networkx as nx

import decentralized_bestresponse as mabr


def test_run_br_o_uses_fixed_replicas_and_does_not_mutate_options(tmp_path, monkeypatch):
  seen = {}
  base_data = {None: {"Nn": {None: 1}, "Nf": {None: 1}, "neighborhood": {(1, 1): 0}}}

  monkeypatch.setattr(mabr, "init_problem",
    lambda *a, **k: (base_data, {}, [], nx.empty_graph(1)))
  monkeypatch.setattr(mabr, "load_solution",
    lambda folder, model: ("s", "rep", "fwd", None, None), raising=False)
  monkeypatch.setattr(mabr, "encode_solution",
    lambda Nn, Nf, s, d, r, t: (None, None, None, np.array([[3]]), None), raising=False)
  monkeypatch.setattr(mabr, "LSP", lambda: "LSP")
  monkeypatch.setattr(mabr, "LSP_fixedr", lambda: "LSP_fixedr", raising=False)

  def _solve_subproblem(sp_data, agents, sp, *a):
    seen["sp"] = sp
    seen["r_bar"] = dict(sp_data[None].get("r_bar", {}))
    return (sp_data, np.zeros((1, 1)), None, None, np.zeros((1, 1)),
            np.ones((1, 1)), np.zeros((1,)), np.zeros((1, 1)),
            {"tot": 0.0}, {"tot": "ok"}, {"tot": 0.0})

  monkeypatch.setattr(mabr, "solve_subproblem", _solve_subproblem)
  monkeypatch.setattr(mabr, "get_current_load", lambda *a: {})
  monkeypatch.setattr(mabr, "update_data", lambda data, u: data)
  monkeypatch.setattr(mabr, "compute_residual_capacity",
    lambda *a: (np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))))
  monkeypatch.setattr(mabr, "best_response_sweep",
    lambda *a, **k: (np.zeros((1, 1, 1)),
                     __import__("pandas").DataFrame({"i": [], "j": [], "f": []}),
                     0, 0.0, 0.0))
  monkeypatch.setattr(mabr, "combine_solutions",
    lambda *a: {"sp": {"x": np.zeros((1, 1)), "y": np.zeros((1, 1, 1)),
                       "z": np.zeros((1, 1)), "r": np.ones((1, 1)), "U": np.zeros((1, 1))}})
  monkeypatch.setattr(mabr, "compute_centralized_objective", lambda *a: 1.0)
  monkeypatch.setattr(mabr, "check_feasibility", lambda *a: (True, "ok"))
  monkeypatch.setattr(mabr, "decode_solutions",
    lambda sp_data, sol, comp, arg: (comp, None, 1.0))
  monkeypatch.setattr(mabr, "join_complete_solution", lambda comp: ({}, {}, {}))
  monkeypatch.setattr(mabr, "save_checkpoint", lambda *a: None)
  monkeypatch.setattr(mabr, "save_solution", lambda *a: None)

  config = {
    "base_solution_folder": str(tmp_path),
    "seed": 1,
    "limits": {"load": {"trace_type": "fixed_sum"}},
    "solver_name": "mock",
    "solver_options": {"general": {"TimeLimit": 10},
                       "br_o": {}},
    "max_iterations": 1, "patience": 1, "max_steps": 1,
    "min_run_time": 0, "max_run_time": 0, "run_time_step": 1,
    "checkpoint_interval": 1, "verbose": 0,
    "opt_solution_folder": "centralized-folder",
  }

  mabr.run_br_o(config, parallelism=0, disable_plotting=True)

  assert seen["sp"] == "LSP_fixedr"
  assert seen["r_bar"] == {(1, 1): 3}
  assert "latency_weight" not in config["solver_options"]["br_o"]
