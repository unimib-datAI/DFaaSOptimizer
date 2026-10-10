"""Hierarchical FaaS-MAFW: Frank-Wolfe at level one, structure auctions above.

Each coordination round has the shape of the hierarchical one-shot runner
(``hierarchical_auction.runner``), with the level-one auction replaced by one
FaaS-MAFW round:

1. level 1: distributed Frank-Wolfe on the relaxed one-hop problem, rounding
   and slack-preserving repair (``frankwolfe_assignments``), then LSPr with the
   forwarded load fixed;
2. levels 2..max_hierarchy_depth: the existing iterative structure auctions
   (``IterativeHierarchicalAuctionEngine``) sell the residual capacity of the
   replicas already running, as indivisible tokens, to the residual demand of
   the structures; flows stay between direct neighbours.  The node price of
   these auctions is the last memory price advertised by each receiver in the
   Frank-Wolfe iterations;
3. optional potential-game refinement of the incumbent (``madea_pg``), with
   the same time reservation as hierarchical-one-shot-pg.

``hierarchy=False`` skips step 2, so the same runner also gives the flat
MAFW + PG variant used as an ablation.  Like the hierarchical runner, the time
budget is wall-clock time per timestep.
"""

from run_centralized_model import (
  get_current_load,
  init_complete_solution,
  init_problem,
  join_complete_solution,
  save_checkpoint,
  save_solution,
  update_data,
)
from run_faasmacro import (
  combine_solutions,
  compute_social_welfare,
  decode_solutions,
  parallel_solver_run,
  solve_subproblem,
)
from run_faasmadea import (
  check_ls_pr_feasibility_from_fixed_y,
  compute_residual_capacity,
  neigh_dict_to_matrix,
)
from utils.centralized import check_feasibility
from utils.faasmacro import compute_centralized_objective
from models.sp import LSP, LSPr
from decentralized_frankwolfe import (
  frankwolfe_assignments,
  frankwolfe_options,
  fw_parameters,
)
from hierarchical_auction.iterative_engine import IterativeHierarchicalAuctionEngine
from hierarchical_auction.runner import build_auction_options

from networkx import adjacency_matrix
from datetime import datetime
from copy import deepcopy
from functools import partial
import pandas as pd
import numpy as np
import json
import time
import os
import sys


RESULT_NAMES = {
  (True, False): "Hierarchical-MAFW",
  (True, True): "Hierarchical-MAFW-PG",
  (False, True): "MAFW-PG",
  (False, False): "FaaS-MAFW-flat",
}


def _update_from_lspr(spr, sp_data, agents, solver_name, solver_options, y, parallelism):
  bad_nodes = check_ls_pr_feasibility_from_fixed_y(sp_data, y)
  if bad_nodes:
    raise RuntimeError(f"LSPr infeasible from fixed y assignments: {bad_nodes}")
  rmp_omega = y.sum(axis=1)
  spr_sol, spr_obj, _, spr_runtime = compute_social_welfare(
    spr, sp_data, agents, solver_name, solver_options, y, rmp_omega, parallelism
  )
  sp_x, _, _, _, sp_r, sp_rho = spr_sol
  return sp_x, sp_r, sp_rho, rmp_omega


@parallel_solver_run
def run(
    config: dict,
    parallelism: int,
    log_on_file: bool = False,
    disable_plotting: bool = False,
    *,
    hierarchy: bool = True,
    refine_welfare: bool = False,
  ) -> str:
  base_solution_folder = config["base_solution_folder"]
  seed = config["seed"]
  limits = config["limits"]
  trace_type = limits["load"].get("trace_type", "fixed_sum")
  verbose = config.get("verbose", 0)
  solver_name = config["solver_name"]
  solver_options = config["solver_options"]
  general_solver_options = solver_options.get("general", {})
  fw_options = frankwolfe_options(solver_options)
  auction_options = build_auction_options(config)
  rng = np.random.default_rng(seed)
  time_limit = general_solver_options.get("TimeLimit", np.inf)
  tolerance = config.get("tolerance", 1e-6)
  max_iterations = config["max_iterations"]
  max_steps = config["max_steps"]
  min_run_time = config.get("min_run_time", 0)
  max_run_time = config.get("max_run_time", max_steps)
  run_time_step = config.get("run_time_step", 1)
  checkpoint_interval = config["checkpoint_interval"]
  max_hierarchy_depth = config.get("max_hierarchy_depth", 3)
  if "opt_solution_folder" in config:
    raise ValueError("hierarchical FaaS-MAFW does not support fixed replicas")
  result_name = RESULT_NAMES[(hierarchy, refine_welfare)]

  now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
  solution_folder = f"{base_solution_folder}/{now}"
  os.makedirs(solution_folder, exist_ok=True)
  with open(os.path.join(solution_folder, "config.json"), "w") as ostream:
    ostream.write(json.dumps(config, indent=2))
  log_stream = open(os.path.join(solution_folder, "out.log"), "w") if log_on_file else sys.stdout

  base_instance_data, input_requests_traces, agents, graph = init_problem(
    limits, trace_type, max_steps, seed, solution_folder
  )
  Nn = base_instance_data[None]["Nn"][None]
  Nf = base_instance_data[None]["Nf"][None]
  neighborhood = neigh_dict_to_matrix(base_instance_data[None]["neighborhood"], Nn)
  latency = np.asarray(adjacency_matrix(graph, weight="network_latency").todense())

  pg_reserve = 0.0
  if refine_welfare:
    from madea_pg import refinement_options
    pg_options = refinement_options(config)
    if pg_options["max_sweeps"] > 0:
      requested = (pg_options["time_limit_per_node"] * Nn
                   if "time_limit_per_node" in pg_options else pg_options["time_limit"])
      pg_reserve = min(requested, time_limit / 2)
  coordination_limit = time_limit - pg_reserve

  ub = (max_run_time + run_time_step) if max_run_time == min_run_time else max_run_time
  spc_complete_solution = init_complete_solution()
  obj_dict, tc_dict = {"LSPr_final": []}, {"LSPr": []}
  stats_rows, runtime_list, refinement_history = [], [], []

  for t in range(min_run_time, ub, run_time_step):
    started = time.monotonic()
    loadt = get_current_load(input_requests_traces, agents, t)
    data = update_data(base_instance_data, {"incoming_load": loadt})
    sp_data = deepcopy(data)
    (
      sp_data, sp_x, _, _, sp_omega, sp_r, sp_rho, _, _, _, _
    ) = solve_subproblem(
      sp_data, agents, LSP(), solver_name, general_solver_options, parallelism
    )
    spr = LSPr()
    fw_params = fw_parameters(sp_data)
    y = np.zeros((Nn, Nn, Nf))
    omega = deepcopy(sp_omega)
    fairness = np.zeros((Nn, Nf))
    engine = IterativeHierarchicalAuctionEngine(
      neighborhood=neighborhood, num_functions=Nf, service_quantum=np.ones(Nf),
      max_depth=max_hierarchy_depth, auction_options=auction_options,
    ) if hierarchy else None

    def snapshot():
      csol = combine_solutions(
        Nn, Nf, sp_data, loadt, sp_x, sp_r, sp_rho,
        None, y, None, None, None, None
      )
      feas = check_feasibility(
        csol["sp"]["x"], csol["sp"]["y"].sum(axis=1), csol["sp"]["z"],
        csol["sp"]["r"], csol["sp"]["U"], sp_data
      )
      assert feas[0], feas[1]
      return csol, compute_centralized_objective(
        sp_data, csol["sp"]["x"], csol["sp"]["y"], csol["sp"]["z"]
      )

    # the local solution without forwarding is the first incumbent
    best_solution, best_cost = snapshot()
    best_it = -1
    it = 0
    while True:
      y_new, fw_stats = frankwolfe_assignments(
        omega, y, sp_x, sp_r, sp_rho, sp_data, neighborhood, latency, fairness,
        fw_options, rng, tolerance=tolerance, params=fw_params,
      )
      price = fw_stats.pop("price")
      if y_new.sum() > tolerance:
        y = y + y_new
        y[np.abs(y) < tolerance] = 0.0
        sp_x, sp_r, sp_rho, rmp_omega = _update_from_lspr(
          spr, sp_data, agents, solver_name, general_solver_options, y, parallelism
        )
        fairness += (rmp_omega > tolerance)
        omega = sp_omega - rmp_omega
        omega[np.abs(omega) < tolerance] = 0.0
      accepted_higher = 0.0
      if engine is not None and (omega > tolerance).any():
        _, blackboard, _ = compute_residual_capacity(sp_x, y, sp_r, sp_data)
        result = engine.run_higher_levels(
          y=y, omega=omega, residual_capacity=blackboard,
          node_prices=np.maximum(price, 0.0), latency=latency, fairness=fairness,
        )
        if result.accepted_allocations:
          accepted_higher = float((result.y - y).sum())
          y = result.y
          y[np.abs(y) < tolerance] = 0.0
          sp_x, sp_r, sp_rho, rmp_omega = _update_from_lspr(
            spr, sp_data, agents, solver_name, general_solver_options, y, parallelism
          )
          fairness += (rmp_omega > tolerance)
          omega = sp_omega - rmp_omega
          omega[np.abs(omega) < tolerance] = 0.0
      fw_stats.update({"t": t, "round": it, "accepted_higher_levels": accepted_higher})
      stats_rows.append(fw_stats)
      csol, cobj = snapshot()
      if cobj > best_cost:
        best_solution, best_cost, best_it = deepcopy(csol), cobj, it
      elapsed = time.monotonic() - started
      if it + 1 >= max_iterations:
        why = "max iterations"
      elif fw_options["max_rounds"] and it + 1 >= fw_options["max_rounds"]:
        why = "max rounds"
      elif not (omega > tolerance).any():
        why = "all load assigned"
      elif y_new.sum() <= tolerance and accepted_higher <= tolerance:
        why = "no more assignments"
      elif elapsed > coordination_limit:
        why = "time limit"
      else:
        it += 1
        continue
      break

    incumbent = best_solution
    if refine_welfare:
      from madea_pg import refine_solution
      incumbent, refinement = refine_solution(
        sp_data, incumbent, config, time_limit - (time.monotonic() - started),
      )
      refinement_history.append({"time": t, **refinement, "coordination_rounds": it + 1,
                                 "pg_reserve": pg_reserve})
    spc_complete_solution, _, final_objective = decode_solutions(
      sp_data, incumbent, spc_complete_solution, None
    )
    total = time.monotonic() - started
    obj_dict["LSPr_final"].append(final_objective)
    tc_dict["LSPr"].append(
      f"{why} (it: {it}; best centralized it: {best_it}; total runtime: {total})"
    )
    runtime_list.append(total)
    if verbose > 0:
      print(f"t = {t}: {why}, rounds {it + 1}, welfare {final_objective}",
            file=log_stream, flush=True)
    if t % checkpoint_interval == 0 or t == max_steps - 1:
      save_checkpoint(spc_complete_solution, os.path.join(solution_folder, "LSPc"), t)

  spc_solution, spc_offloaded, spc_detailed = join_complete_solution(spc_complete_solution)
  save_solution(spc_solution, spc_offloaded, spc_complete_solution, spc_detailed,
                "LSPc", solution_folder)
  pd.DataFrame(obj_dict["LSPr_final"], columns=[result_name]).to_csv(
    os.path.join(solution_folder, "obj.csv"), index=False)
  pd.DataFrame(tc_dict["LSPr"]).to_csv(os.path.join(solution_folder, "termination_condition.csv"))
  pd.DataFrame({"tot": runtime_list}).to_csv(os.path.join(solution_folder, "runtime.csv"), index=False)
  pd.DataFrame(stats_rows).to_csv(os.path.join(solution_folder, "frankwolfe_stats.csv"), index=False)
  if refinement_history:
    pd.DataFrame(refinement_history).to_csv(os.path.join(solution_folder, "refinement.csv"), index=False)
  if log_on_file:
    log_stream.close()
  return solution_folder


run_hierarchical = partial(run, hierarchy=True, refine_welfare=False)
run_hierarchical_pg = partial(run, hierarchy=True, refine_welfare=True)
run_flat_pg = partial(run, hierarchy=False, refine_welfare=True)
