import numpy as np

import decentralized_bestresponse as mabr
from frame_stubs import stub_frame


def test_run_br_o_uses_fixed_replicas_and_does_not_mutate_options(tmp_path, monkeypatch):
  seen = {}
  base_data = {None: {"Nn": {None: 1}, "Nf": {None: 1}, "neighborhood": {(1, 1): 0}}}

  decoded = stub_frame(monkeypatch, base_data, reference_r=[[3]])
  monkeypatch.setattr(mabr, "LSP", lambda: "LSP")
  monkeypatch.setattr(mabr, "LSP_fixedr", lambda: "LSP_fixedr")
  monkeypatch.setattr(mabr, "LSPr", lambda: seen.setdefault("spr", "LSPr"))
  monkeypatch.setattr(mabr, "LSPr_fixedr", lambda: seen.setdefault("spr", "LSPr_fixedr"))

  def _solve_subproblem(sp_data, agents, sp, *a):
    seen["sp"] = sp
    seen["r_bar"] = dict(sp_data[None].get("r_bar", {}))
    return (sp_data, np.zeros((1, 1)), None, None, np.zeros((1, 1)),
            np.ones((1, 1)), np.array([5.0]), np.zeros((1, 1)),
            {"tot": 0.0}, {"tot": "ok"}, {"tot": 0.0})

  monkeypatch.setattr(mabr, "solve_subproblem", _solve_subproblem)
  monkeypatch.setattr(mabr, "compute_residual_capacity",
    lambda *a: (np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))))
  def _best_response_sweep(*args, **kwargs):
    seen["coordination_rho"] = np.array(args[4], copy=True)
    return (
      np.zeros((1, 1, 1)),
      __import__("pandas").DataFrame({"i": [], "j": [], "f": []}),
      0, 0.0, 0.0,
    )

  monkeypatch.setattr(mabr, "best_response_sweep", _best_response_sweep)
  monkeypatch.setattr(mabr, "compute_centralized_objective", lambda *a: -1.0)

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
  assert seen["spr"] == "LSPr_fixedr"
  assert seen["r_bar"] == {(1, 1): 3}
  assert (seen["coordination_rho"] == 0).all()
  assert len(decoded) == 2
  assert "latency_weight" not in config["solver_options"]["br_o"]
