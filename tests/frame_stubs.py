"""Stubs of the helpers the shared frame (utils/run_frame.py) calls, for runner wiring tests.

``stub_frame`` replaces problem loading, the fixed reference, the step data, the snapshot
helpers, decoding and every file write, so a runner can be driven on a one-node problem.
Runner-owned names (models, solve_subproblem, ...) are patched by the test on the runner.
"""
import networkx as nx
import numpy as np

ONE_NODE = {
  "x": np.zeros((1, 1)), "y": np.zeros((1, 1, 1)), "z": np.zeros((1, 1)), "r": np.ones((1, 1)),
  "xi": np.zeros((1, 1, 1)), "rho": np.zeros(1), "U": np.zeros((1, 1)),
}


def stub_frame(monkeypatch, base_data, reference_r=None) -> list:
  """Returns the list of decoded ``x`` arrays (one per output and step)."""
  import postprocessing
  import run_centralized_model as rcm
  import run_faasmacro
  import utils.centralized
  import utils.faasmacro
  nodes = base_data[None]["Nn"][None]
  monkeypatch.setattr(rcm, "init_problem", lambda *a, **k: (base_data, {}, [], nx.empty_graph(nodes)))
  monkeypatch.setattr(rcm, "get_current_load", lambda *a: {})
  monkeypatch.setattr(rcm, "update_data", lambda data, update: {None: {**data[None], **update}})
  monkeypatch.setattr(rcm, "join_complete_solution", lambda complete: ({}, {}, {}))
  monkeypatch.setattr(rcm, "save_checkpoint", lambda *a: None)
  monkeypatch.setattr(rcm, "save_solution", lambda *a: None)
  if reference_r is not None:
    monkeypatch.setattr(postprocessing, "load_solution",
                        lambda folder, model: ("opt_solution", "opt_replicas", "opt_detailed", None, None))
    monkeypatch.setattr(utils.centralized, "encode_solution",
                        lambda *a: (None, None, None, np.array(reference_r), None))
  monkeypatch.setattr(run_faasmacro, "combine_solutions", lambda *a: {"sp": dict(ONE_NODE)})
  monkeypatch.setattr(utils.faasmacro, "compute_centralized_objective", lambda *a: -1.0)
  monkeypatch.setattr(utils.centralized, "check_feasibility", lambda *a: (True, "ok"))
  decoded = []

  def decode(x, y, z, r, xi, rho, U, complete):
    decoded.append(x)
    return complete

  monkeypatch.setattr(utils.centralized, "decode_solution", decode)
  return decoded
