"""MADEA incumbents followed by bounded, welfare-improving potential-game moves."""

import argparse
from copy import deepcopy
from math import isfinite
import time

import numpy as np
import pyomo.environ as pyo

from decentralized_potentialgame import node_move, propose_node_move, compute_rho
from models.local_sp import dp_enabled
from models.model import PYO_VAR_TYPE
from models.sp import LSP_pg, LSP_pg_fixedr
from run_faasmacro import combine_solutions
from utils.faasmacro import compute_centralized_objective
from utils.common import load_configuration


def refinement_options(config):
  options = {"max_sweeps": 5, "epsilon": 1e-6, "time_limit": 5.0}
  options.update(config.get("solver_options", {}).get("madea_pg", {}))
  if type(options["max_sweeps"]) is not int or options["max_sweeps"] < 0:
    raise ValueError("madea_pg.max_sweeps must be a nonnegative integer")
  for key in ("epsilon", "time_limit", "time_limit_per_node"):
    if key not in options:
      continue
    value = options[key]
    if (isinstance(value, bool) or not isinstance(value, (int, float))
        or not isfinite(value) or value < 0 or (key == "epsilon" and value == 0)):
      raise ValueError(f"madea_pg.{key} must be finite and {'positive' if key == 'epsilon' else 'nonnegative'}")
  return options


class _InvalidLocalProposal(Exception):
  """The proposing node rejected its solver output using its own constraints."""


def refine_solution(data, solution, config, remaining_time=float("inf")):
  """Accept feasible improvements only; return the incumbent on a stalled search.

  Fixed-order node moves preserve other sources' commitments. The exact local solve
  proposes moves; the node's own normalized utility decides acceptance. No-improvement
  means the proposals stalled, not a certificate of Nash/global optimality.
  """
  options = refinement_options(config)
  started = time.monotonic()
  sp = solution["sp"]
  Nn, Nf = sp["x"].shape
  budget = (options["time_limit_per_node"] * Nn
            if "time_limit_per_node" in options else options["time_limit"])
  budget = max(0., min(budget, remaining_time))
  deadline = started + budget
  welfare = compute_centralized_objective(data, sp["x"], sp["y"], sp["z"])
  stats = dict(welfare_before=welfare, welfare_after=welfare, sweeps=0, moves=0,
               seconds=0., time_budget=budget, reason="max sweeps reached")
  x, y, r = (sp[k].copy() for k in ("x", "y", "r"))
  neighborhood = np.zeros((Nn, Nn))
  for (i, j), value in data[None]["neighborhood"].items():
    neighborhood[i - 1, j - 1] = value
  integer_flows = PYO_VAR_TYPE is pyo.NonNegativeIntegers
  model = LSP_pg_fixedr() if "r_bar" in data[None] else LSP_pg()
  solver = config["solver_name"]
  # Either the local DP or the configured solver proposes; glpk's whole-second
  # TimeLimit only constrains the budget when glpk is the one being called.
  integer_limit = (solver in {"glpk", "glpsol"}
                   and not dp_enabled(config.get("solver_options", {}).get("general", {})))
  exhausted = False
  for sweep in range(options["max_sweeps"]):
    accepted_in_sweep = 0
    for i in range(Nn):
      remaining = deadline - time.monotonic()
      if remaining <= 0 or (integer_limit and remaining < 1):
        exhausted = True
        break
      stats["sweeps"] = sweep + 1
      solver_options = deepcopy(config.get("solver_options", {}).get("general", {}))
      solver_options["TimeLimit"] = int(remaining) if integer_limit else remaining
      def propose(node, cap):
        proposal = propose_node_move(node, cap, y, data, model, solver, solver_options, 0)
        local_x, local_r, offload, _ = proposal
        values = data[None]
        load = np.array([values["incoming_load"][node + 1, f + 1] for f in range(Nf)])
        demand = np.array([values["demand"][node + 1, f + 1] for f in range(Nf)])
        max_utilization = np.array([values["max_utilization"][f + 1] for f in range(Nf)])
        memory = np.array([values["memory_requirement"][f + 1] for f in range(Nf)])
        arrays = (local_x, local_r, offload)
        if (any(not np.isfinite(a).all() or (a < -1e-6).any() for a in arrays)
            or (integer_flows and any(not np.allclose(a, np.rint(a), rtol=0, atol=1e-6)
                                      for a in arrays))
            or not np.allclose(local_r, np.rint(local_r), rtol=0, atol=1e-6)
            or (local_x + offload > load + 1e-6).any()
            or (offload > cap + 1e-6).any()
            or (demand * (local_x + y[:, node, :].sum(axis=0))
                > local_r * max_utilization + 1e-6).any()
            or local_r @ memory > values["memory_capacity"][node + 1] + 1e-6):
          raise _InvalidLocalProposal
        return proposal

      try:
        accepted, *_ = node_move(
          i, x, y, r, data, neighborhood, np.zeros(Nn),
          options["epsilon"], propose, 1e-6, integer_flows=integer_flows,
        )
      except _InvalidLocalProposal:
        continue
      if accepted:
        stats["moves"] += 1
        accepted_in_sweep += 1
    if exhausted:
      stats["reason"] = "time budget exhausted"
      break
    if not accepted_in_sweep:
      stats["reason"] = "no improving proposal"
      break
  # Observer/export only: no global feasibility or welfare oracle selects moves.
  if stats["moves"]:
    solution = combine_solutions(
      Nn, Nf, data, data[None]["incoming_load"], x, r, np.zeros(Nn),
      None, y, None, None, None, None,
    )
    solution["sp"]["rho"] = compute_rho(solution["sp"]["r"], data)
  sp = solution["sp"]
  stats["welfare_after"] = compute_centralized_objective(data, sp["x"], sp["y"], sp["z"])
  stats["seconds"] = time.monotonic() - started
  return solution, stats


def run(config, parallelism=-1, log_on_file=False, disable_plotting=False):
  from run_faasmadea import run as run_madea
  refinement_options(config)
  config = deepcopy(config)
  config["algorithm"] = "faas-madea-pg"
  return run_madea(config, parallelism, log_on_file, disable_plotting, refine_welfare=True)


def run_hierarchical(config, parallelism=-1, log_on_file=False, disable_plotting=False):
  from hierarchical_auction.madea_cycles_runner import _run
  from hierarchical_auction.iterative_engine import IterativeHierarchicalAuctionEngine
  refinement_options(config)
  config = deepcopy(config)
  config["algorithm"] = "hierarchical-madea-level-cycles-pg"
  return _run(
    config, parallelism, log_on_file, disable_plotting,
    engine_class=IterativeHierarchicalAuctionEngine,
    result_name="HierarchicalMADeALevelCyclesPG", refine_welfare=True,
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("-c", "--config", default="config_files/hierarchical_madea_cycles.json")
  parser.add_argument("-j", "--parallelism", type=int, default=-1)
  parser.add_argument("--disable_plotting", action="store_true")
  parser.add_argument("--variant", choices=("madea", "hierarchical"), default="madea")
  args = parser.parse_args()
  runner = run if args.variant == "madea" else run_hierarchical
  runner(load_configuration(args.config), args.parallelism, disable_plotting=args.disable_plotting)
