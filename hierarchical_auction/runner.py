"""Standalone runner for the hierarchical multi-structure auction.

Follows the same lifecycle as decentralized_auction.run():
init -> time loop -> iteration loop -> save.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from networkx import adjacency_matrix as nx_adjacency_matrix

from decentralized_auction import (
  check_stopping_criteria,
  compute_residual_capacity,
  define_bids,
  evaluate_bids,
  neigh_dict_to_matrix,
  start_additional_replicas,
)
from hierarchical_auction.engine import HierarchicalAuctionEngine
from hierarchical_auction.types import FloatArray
from models.sp import LSP, LSPr
from run_centralized_model import (
  get_current_load,
  init_complete_solution,
  init_problem,
  join_complete_solution,
  save_checkpoint,
  save_solution,
)
from run_faasmacro import (
  combine_solutions,
  compute_centralized_objective,
  compute_social_welfare,
  decode_solutions,
  solve_subproblem,
)
from utils.common import load_configuration


def parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Run Hierarchical Multi-Structure Auction",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
    "-c", "--config",
    help="Configuration file",
    type=str,
    default="config_files/manual_config.json",
  )
  parser.add_argument(
    "-j", "--parallelism",
    help="Number of parallel processes (-1: auto, 0: sequential)",
    type=int,
    default=-1,
  )
  parser.add_argument(
    "--disable_plotting",
    help="Disable automatic plot generation",
    default=False,
    action="store_true",
  )
  return parser.parse_known_args()[0]


def build_auction_options(config: Mapping[str, Any]) -> dict[str, object]:
  solver_options = config.get("solver_options", {})
  auction = solver_options.get("auction", {})
  return {
    "epsilon": auction.get("epsilon", 0.01),
    "eta": auction.get("eta", [0.5, 0.3, 0.1]),
    "zeta": auction.get("zeta", 0.1),
    "latency_weight": auction.get("latency_weight", 0.0),
    "fairness_weight": auction.get("fairness_weight", 0.0),
  }


def compute_offloaded_demand(y: FloatArray) -> FloatArray:
  """Return total offloaded demand per buyer node/function."""
  return y.sum(axis=1)


def bids_for_stopping(
  bids: pd.DataFrame,
  hierarchical_allocations_count: int,
) -> pd.DataFrame:
  """Avoid treating a productive hierarchical round as no-progress."""
  if len(bids) > 0 or hierarchical_allocations_count <= 0:
    return bids
  return pd.DataFrame({"hierarchical_progress": [hierarchical_allocations_count]})


def _extract_latency(graph: Any) -> FloatArray:
  return nx_adjacency_matrix(graph, weight="network_latency").toarray()


def run(
  config: dict[str, Any],
  parallelism: int = -1,
  log_on_file: bool = False,
  disable_plotting: bool = False,
) -> str:
  base_solution_folder = config["base_solution_folder"]
  seed = config["seed"]
  limits = config["limits"]
  trace_type = config["limits"]["load"].get("trace_type", "fixed_sum")
  verbose = config.get("verbose", 0)

  solver_name = config["solver_name"]
  solver_options = config.get("solver_options", {})
  general_solver_options = solver_options.get("general", {})
  auction_options = build_auction_options(config)
  time_limit = general_solver_options.get("TimeLimit", float("inf"))
  tolerance = config.get("tolerance", 1e-6)

  max_iterations = config["max_iterations"]
  max_steps = config["max_steps"]
  min_run_time = config.get("min_run_time", 0)
  max_run_time = config.get("max_run_time", max_steps)
  run_time_step = config.get("run_time_step", 1)
  checkpoint_interval = config["checkpoint_interval"]
  max_hierarchy_depth = config.get("max_hierarchy_depth", 3)

  now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
  solution_folder = f"{base_solution_folder}/{now}"
  os.makedirs(solution_folder, exist_ok=True)
  with open(os.path.join(solution_folder, "config.json"), "w") as ostream:
    ostream.write(json.dumps(config, indent=2))

  log_stream = sys.stdout
  if log_on_file:
    log_stream = open(os.path.join(solution_folder, "out.log"), "w")

  base_instance_data, input_requests_traces, agents, graph = init_problem(
    limits, trace_type, max_steps, seed, solution_folder,
  )
  Nn = base_instance_data[None]["Nn"][None]
  Nf = base_instance_data[None]["Nf"][None]

  neighborhood = neigh_dict_to_matrix(
    base_instance_data[None]["neighborhood"], Nn,
  )
  latency = _extract_latency(graph)

  ub = (max_run_time + run_time_step) if max_run_time == min_run_time else max_run_time
  spc_complete_solution = init_complete_solution()
  obj_dict: dict[str, list[Any]] = {"LSPr_final": []}
  tc_dict: dict[str, list[Any]] = {"LSPr": []}
  runtime_list: list[float] = []

  for t in range(min_run_time, ub, run_time_step):
    if verbose > 0:
      print(f"t = {t}", file=log_stream, flush=True)

    loadt = get_current_load(input_requests_traces, agents, t)
    data = deepcopy(base_instance_data)
    data[None]["incoming_load"] = loadt

    u0 = np.ones((Nn, Nf)) * 0.9
    p = np.zeros((Nn, Nf))
    fairness = np.zeros((Nn, Nf))

    sp_data = deepcopy(data)
    sp = LSP()
    spr = LSPr()
    (
      sp_data, sp_x, _, _sp_z, sp_omega, sp_r, sp_rho, _sp_u,
      _obj, _tc, sp_runtime,
    ) = solve_subproblem(
      sp_data, agents, sp, solver_name, general_solver_options, parallelism,
    )
    total_runtime = sp_runtime["tot"] if isinstance(sp_runtime, dict) else 0.0

    it = 0
    stop_searching = False
    y = np.zeros((Nn, Nn, Nf))
    omega = deepcopy(sp_omega)
    rmp_omega = np.zeros((Nn, Nf))
    best_centralized_solution: dict | None = None
    best_centralized_cost = 0.0
    best_centralized_it = -1
    engine = HierarchicalAuctionEngine(
      neighborhood=neighborhood,
      num_functions=Nf,
      service_quantum=np.ones(Nf),
      max_depth=max_hierarchy_depth,
      auction_options=auction_options,
    )

    while not stop_searching:
      if verbose > 0:
        print(f"    it = {it}", file=log_stream, flush=True)

      capacity, blackboard, ell = compute_residual_capacity(
        sp_x, y, sp_r, sp_data,
      )

      level1_options = dict(auction_options)
      if isinstance(level1_options.get("eta"), list):
        level1_options["eta"] = level1_options["eta"][0]

      bids, memory_bids = define_bids(
        omega, blackboard, p, sp_data, neighborhood, sp_rho,
        level1_options,
        latency=latency,
        fairness=fairness,
        delta=np.zeros((Nn, Nn)),
      )

      if len(bids) > 0:
        auction_y, p = evaluate_bids(
          bids, blackboard, sp_data, ell, p, capacity, u0,
          level1_options,
        )
        y += auction_y

        rmp_omega = compute_offloaded_demand(y)
        fairness += (rmp_omega > 0).astype(fairness.dtype)

        spr_sol, _spr_obj, _spr_tc, spr_runtime = compute_social_welfare(
          spr, sp_data, agents, solver_name, general_solver_options,
          y, rmp_omega, parallelism,
        )
        total_runtime += spr_runtime if isinstance(spr_runtime, (int, float)) else 0.0

        sp_x = spr_sol[0]
        sp_r = spr_sol[4]
        sp_rho = spr_sol[5]
        omega = sp_omega - rmp_omega
        omega[np.abs(omega) < tolerance] = 0.0
      else:
        a, sp_rho = start_additional_replicas(
          memory_bids, sp_r, sp_data, sp_rho,
        )
        sp_r += a

      capacity, blackboard, ell = compute_residual_capacity(
        sp_x, y, sp_r, sp_data,
      )

      result = engine.run_higher_levels(
        y=y,
        omega=omega,
        residual_capacity=blackboard,
        node_prices=p,
        latency=latency,
        fairness=fairness,
      )
      y = result.y
      rmp_omega = compute_offloaded_demand(y)

      if result.accepted_allocations:
        spr_sol, _spr_obj, _spr_tc, spr_runtime = compute_social_welfare(
          spr, sp_data, agents, solver_name, general_solver_options,
          y, rmp_omega, parallelism,
        )
        total_runtime += (
          spr_runtime if isinstance(spr_runtime, (int, float)) else 0.0
        )

        sp_x = spr_sol[0]
        sp_r = spr_sol[4]
        sp_rho = spr_sol[5]
        omega = sp_omega - rmp_omega
        omega[np.abs(omega) < tolerance] = 0.0

      stopping_bids = bids_for_stopping(
        bids, len(result.accepted_allocations),
      )
      stop_searching, why_stop_searching = check_stopping_criteria(
        it, max_iterations, blackboard, omega, rmp_omega,
        stopping_bids, memory_bids, tolerance, total_runtime, time_limit,
      )

      csol = combine_solutions(
        Nn, Nf, sp_data, loadt,
        sp_x, sp_r, sp_rho,
        np.zeros((Nn, Nf)), y, np.zeros((Nn, Nf)),
        np.zeros((Nn, Nf)), np.zeros((Nn, Nn, Nf)),
        np.zeros((Nn,)),
      )
      cobj = compute_centralized_objective(
        sp_data, csol["sp"]["x"], csol["sp"]["y"], csol["sp"]["z"],
      )

      if cobj > best_centralized_cost:
        best_centralized_cost = cobj
        best_centralized_solution = csol
        best_centralized_it = it

      if not stop_searching:
        it += 1
      else:
        spc_complete_solution, _, _ = decode_solutions(
          sp_data,
          best_centralized_solution if best_centralized_solution is not None else csol,
          spc_complete_solution, None,
        )
        obj_dict["LSPr_final"].append(best_centralized_cost)
        tc_dict["LSPr"].append(
          f"{why_stop_searching} (it: {it}; "
          f"best centralized it: {best_centralized_it}; "
          f"total runtime: {total_runtime})"
        )
        runtime_list.append(total_runtime)
        if t % checkpoint_interval == 0 or t == max_steps - 1:
          save_checkpoint(
            spc_complete_solution,
            os.path.join(solution_folder, "LSPc"), t,
          )

  spc_solution, spc_offloaded, spc_detailed_fwd_solution = join_complete_solution(
    spc_complete_solution,
  )
  save_solution(
    spc_solution, spc_offloaded, spc_complete_solution,
    spc_detailed_fwd_solution, "LSPc", solution_folder,
  )
  pd.DataFrame(obj_dict["LSPr_final"], columns=["HierarchicalAuction"]).to_csv(
    os.path.join(solution_folder, "obj.csv"), index=False,
  )
  pd.DataFrame(tc_dict["LSPr"]).to_csv(
    os.path.join(solution_folder, "termination_condition.csv"),
  )
  pd.DataFrame({"tot": runtime_list}).to_csv(
    os.path.join(solution_folder, "runtime.csv"), index=False,
  )

  if verbose > 0:
    print(f"All solutions saved in: {solution_folder}", file=log_stream, flush=True)
  if log_on_file and log_stream is not sys.stdout:
    log_stream.close()

  return solution_folder


if __name__ == "__main__":
  args = parse_arguments()
  config = load_configuration(args.config)
  run(
    config,
    parallelism=args.parallelism,
    log_on_file=False,
    disable_plotting=args.disable_plotting,
  )
