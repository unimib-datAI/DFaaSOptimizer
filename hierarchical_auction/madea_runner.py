"""Hierarchical auction with the production FaaS-MADeA auction at level one."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from networkx import adjacency_matrix

from hierarchical_auction.engine import HierarchicalAuctionEngine
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

from run_faasmadea import (
  check_ls_pr_feasibility_from_fixed_y,
  check_stopping_criteria,
  compute_residual_capacity,
  define_bids,
  evaluate_bids,
  neigh_dict_to_matrix,
  start_additional_replicas,
)
from utils.common import load_configuration


def build_auction_options(config: dict[str, Any]) -> dict[str, Any]:
  options = deepcopy(config.get("solver_options", {}).get("auction", {}))
  options.setdefault("epsilon", 0.01)
  options.setdefault("eta", [0.5, 0.3, 0.1])
  options.setdefault("zeta", 0.1)
  options.setdefault("latency_weight", 0.0)
  options.setdefault("fairness_weight", 0.0)
  options.setdefault("unit_bids", False)
  return options


def level1_options(options: dict[str, Any]) -> dict[str, Any]:
  result = deepcopy(options)
  if isinstance(result["eta"], list):
    result["eta"] = result["eta"][0]
  return result


def compute_offloaded_demand(y: np.ndarray) -> np.ndarray:
  return y.sum(axis=1)


def bids_for_stopping(
    bids: pd.DataFrame, hierarchical_allocations_count: int,
  ) -> pd.DataFrame:
  if len(bids) > 0 or hierarchical_allocations_count <= 0:
    return bids
  return pd.DataFrame({"hierarchical_progress": [hierarchical_allocations_count]})


def parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Hierarchical auction extending production FaaS-MADeA",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("-c", "--config", default="config_files/manual_config.json")
  parser.add_argument("-j", "--parallelism", type=int, default=-1)
  parser.add_argument("--disable_plotting", action="store_true")
  return parser.parse_known_args()[0]


def run(
    config: dict[str, Any],
    parallelism: int = -1,
    log_on_file: bool = False,
    disable_plotting: bool = False,
  ) -> str:
  del disable_plotting
  base_solution_folder = config["base_solution_folder"]
  seed = config["seed"]
  limits = config["limits"]
  trace_type = limits["load"].get("trace_type", "fixed_sum")
  verbose = config.get("verbose", 0)
  solver_name = config["solver_name"]
  solver_options = config.get("solver_options", {})
  general_solver_options = solver_options.get("general", {})
  auction_options = build_auction_options(config)
  first_level_options = level1_options(auction_options)
  time_limit = general_solver_options.get("TimeLimit", float("inf"))
  tolerance = config.get("tolerance", 1e-6)
  max_iterations = config["max_iterations"]
  max_steps = config["max_steps"]
  min_run_time = config.get("min_run_time", 0)
  max_run_time = config.get("max_run_time", max_steps)
  run_time_step = config.get("run_time_step", 1)
  checkpoint_interval = config["checkpoint_interval"]
  max_hierarchy_depth = config.get("max_hierarchy_depth", 3)
  patience = config.get("patience", 1)

  now = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
  solution_folder = f"{base_solution_folder}/{now}"
  os.makedirs(solution_folder, exist_ok=True)
  with open(os.path.join(solution_folder, "config.json"), "w") as stream:
    stream.write(json.dumps(config, indent=2))
  log_stream = open(os.path.join(solution_folder, "out.log"), "w") \
    if log_on_file else sys.stdout

  base_data, traces, agents, graph = init_problem(
    limits, trace_type, max_steps, seed, solution_folder,
  )
  Nn = base_data[None]["Nn"][None]
  Nf = base_data[None]["Nf"][None]
  neighborhood = neigh_dict_to_matrix(base_data[None]["neighborhood"], Nn)
  latency = adjacency_matrix(graph, weight="network_latency").toarray()
  ub = max_run_time + run_time_step if max_run_time == min_run_time else max_run_time

  complete_solution = init_complete_solution()
  objectives = []
  termination_conditions = []
  runtimes = []

  for t in range(min_run_time, ub, run_time_step):
    started_at = time.monotonic()
    loadt = get_current_load(traces, agents, t)
    sp_data = deepcopy(base_data)
    sp_data[None]["incoming_load"] = loadt
    sp = LSP()
    spr = LSPr()
    (
      sp_data, sp_x, _, _, sp_omega, sp_r, sp_rho, _, _, _, _,
    ) = solve_subproblem(
      sp_data, agents, sp, solver_name, general_solver_options, parallelism,
    )

    u0 = np.ones((Nn, Nf)) * 0.8
    p = np.zeros((Nn, Nf))
    y = np.zeros((Nn, Nn, Nf))
    omega = deepcopy(sp_omega)
    fairness = np.zeros((Nn, Nf))
    accepted_queue = deque(maxlen=patience)
    best_solution = None
    best_cost = -np.inf
    best_it = -1
    it = 0
    stop = False
    engine = HierarchicalAuctionEngine(
      neighborhood=neighborhood,
      num_functions=Nf,
      service_quantum=np.ones(Nf),
      max_depth=max_hierarchy_depth,
      auction_options=auction_options,
    )

    while not stop:
      if verbose > 0:
        print(f"t = {t}; it = {it}", file=log_stream, flush=True)
      capacity, residual_capacity, ell = compute_residual_capacity(
        sp_x, y, sp_r, sp_data,
      )
      blackboard = np.maximum(0.0, capacity - sp_x)
      stalled = (
        len(accepted_queue) >= accepted_queue.maxlen
        and all(value == accepted_queue[0] for value in accepted_queue)
      )
      bids, memory_bids, _ = define_bids(
        omega, blackboard, p, sp_data, neighborhood, sp_rho,
        first_level_options, latency, fairness,
        force_memory_bids=(sp_rho > 0).any() and stalled,
      )

      additional_replicas = np.zeros((Nn, Nf))
      if len(bids) > 0:
        auction_y, p, additional_replicas, _ = evaluate_bids(
          bids, residual_capacity, sp_data, y, ell, p, capacity, u0,
          first_level_options, sp_rho, sp_r,
          tentatively_start_replicas=(len(memory_bids) == 0),
        )
        y += auction_y
        rmp_omega = compute_offloaded_demand(y)
        bad_nodes = check_ls_pr_feasibility_from_fixed_y(sp_data, y)
        if bad_nodes:
          raise RuntimeError(f"LSPr infeasible from fixed y assignments: {bad_nodes}")
        spr_sol, _, _, _ = compute_social_welfare(
          spr, sp_data, agents, solver_name, general_solver_options,
          y, rmp_omega, parallelism,
        )
        sp_x, _, _, _, sp_r, sp_rho = spr_sol
        omega = sp_omega - rmp_omega
        omega[np.abs(omega) < tolerance] = 0.0

      if len(memory_bids) > 0 and not (additional_replicas > 0).any():
        additional_replicas, sp_rho = start_additional_replicas(
          memory_bids, sp_r, sp_data, sp_rho,
        )
        sp_r += additional_replicas

      _, residual_capacity, _ = compute_residual_capacity(
        sp_x, y, sp_r, sp_data,
      )
      result = engine.run_higher_levels(
        y=y,
        omega=omega,
        residual_capacity=residual_capacity,
        node_prices=p,
        latency=latency,
        fairness=fairness,
      )
      y = result.y
      rmp_omega = compute_offloaded_demand(y)
      if result.accepted_allocations:
        bad_nodes = check_ls_pr_feasibility_from_fixed_y(sp_data, y)
        if bad_nodes:
          raise RuntimeError(f"LSPr infeasible from fixed y assignments: {bad_nodes}")
        spr_sol, _, _, _ = compute_social_welfare(
          spr, sp_data, agents, solver_name, general_solver_options,
          y, rmp_omega, parallelism,
        )
        sp_x, _, _, _, sp_r, sp_rho = spr_sol
      omega = sp_omega - rmp_omega
      omega[np.abs(omega) < tolerance] = 0.0
      fairness += (rmp_omega > 0).astype(fairness.dtype)
      accepted_queue.append(float(rmp_omega.sum()))

      _, final_residual, _ = compute_residual_capacity(sp_x, y, sp_r, sp_data)
      combined = combine_solutions(
        Nn, Nf, sp_data, loadt,
        sp_x, sp_r, sp_rho,
        None, y, None, None, None, None,
      )
      cost = compute_centralized_objective(
        sp_data, combined["sp"]["x"], combined["sp"]["y"], combined["sp"]["z"],
      )
      if cost > best_cost:
        best_cost = cost
        best_solution = deepcopy(combined)
        best_it = it

      elapsed = time.monotonic() - started_at
      stop, reason = check_stopping_criteria(
        it=it,
        max_iterations=max_iterations,
        blackboard=final_residual,
        omega=omega,
        rmp_omega=rmp_omega,
        a=additional_replicas,
        bids=bids_for_stopping(bids, len(result.accepted_allocations)),
        memory_bids=memory_bids,
        tolerance=tolerance,
        total_runtime=elapsed,
        time_limit=time_limit,
      )
      if stop:
        complete_solution, _, objective = decode_solutions(
          sp_data, best_solution, complete_solution, None,
        )
        objectives.append(objective)
        termination_conditions.append(
          f"{reason} (it: {it}; obj. deviation: None; best it: {best_it}; "
          f"best centralized it: {best_it}; total runtime: {elapsed})"
        )
        runtimes.append(elapsed)
        if t % checkpoint_interval == 0 or t == max_steps - 1:
          save_checkpoint(complete_solution, os.path.join(solution_folder, "LSPc"), t)
      else:
        it += 1

  solution, offloaded, detailed = join_complete_solution(complete_solution)
  save_solution(
    solution, offloaded, complete_solution, detailed, "LSPc", solution_folder,
  )
  pd.DataFrame({"HierarchicalMADeA": objectives}).to_csv(
    os.path.join(solution_folder, "obj.csv"), index=False,
  )
  pd.DataFrame(termination_conditions).to_csv(
    os.path.join(solution_folder, "termination_condition.csv"),
  )
  pd.DataFrame({"tot": runtimes}).to_csv(
    os.path.join(solution_folder, "runtime.csv"), index=False,
  )
  if log_on_file:
    log_stream.close()
  return solution_folder


if __name__ == "__main__":
  args = parse_arguments()
  run(
    load_configuration(args.config),
    parallelism=args.parallelism,
    disable_plotting=args.disable_plotting,
  )
