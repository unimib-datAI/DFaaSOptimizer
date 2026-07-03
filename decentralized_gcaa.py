import numpy as np
import pandas as pd

from run_centralized_model import (
  get_current_load,
  init_complete_solution,
  init_problem,
  join_complete_solution,
  plot_history,
  save_checkpoint,
  save_solution,
  update_data,
)
from run_faasmacro import (
  combine_solutions,
  compute_social_welfare,
  decode_solutions,
  solve_subproblem,
)
from run_faasmadea import (
  check_ls_pr_feasibility_from_fixed_y,
  check_stopping_criteria,
  compute_residual_capacity,
  define_bids,
  neigh_dict_to_matrix,
)
from utils.centralized import check_feasibility
from utils.faasmacro import compute_centralized_objective
from utils.common import load_configuration
from models.sp import LSP, LSPr

from networkx import adjacency_matrix
from datetime import datetime
from copy import deepcopy
import argparse
import json
import sys
import os


def resolve_gcaa_round(
    bids: pd.DataFrame,
    residual_capacity: np.array,
    current_y: np.array = None,
  ) -> np.array:
  """One GCAA consensus round (Braquet & Bakolas, 2021 - Algorithms 1+3):
  each buyer agent (i, f) proposes only its highest-utility task (j, f);
  for each contested task, the single highest-utility proposal wins and
  consumes one unit of residual_capacity[j, f]. Losers are absent from
  the returned allocation and simply re-propose next round via a fresh
  call to define_bids (their omega is left untouched by this function)."""
  Nn, Nf = residual_capacity.shape
  y_round = np.zeros((Nn, Nn, Nf))
  if len(bids) == 0:
    return y_round
  if current_y is None:
    current_y = y_round
  sending = current_y.sum(axis=1) > 1e-10
  receiving = current_y.sum(axis=0) > 1e-10
  best_per_agent = bids.loc[bids.groupby(["i", "f"])["utility"].idxmax()]
  for (j, f), group in best_per_agent.groupby(["j", "f"]):
    j, f = int(j), int(f)
    if sending[j, f]:
      continue
    for _, winner in group.sort_values("utility", ascending=False).iterrows():
      i = int(winner["i"])
      if receiving[i, f] or residual_capacity[j, f] < winner["d"]:
        continue
      y_round[i, j, f] += winner["d"]
      sending[i, f] = True
      receiving[j, f] = True
      break
  return y_round


def parse_arguments() -> argparse.Namespace:
  """
  Parse input arguments
  """
  parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description = "Run FaaS-MAGCAA",
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument(
    "-c", "--config",
    help = "Configuration file",
    type = str,
    default = "manual_config.json"
  )
  parser.add_argument(
    "-j", "--parallelism",
    help = "Number of parallel processes to start (-1: auto, 0: sequential)",
    type = int,
    default = -1
  )
  parser.add_argument(
    "--disable_plotting",
    help = "True to disable automatic plot generation for each experiment",
    default = False,
    action = "store_true"
  )
  args: argparse.Namespace = parser.parse_known_args()[0]
  return args


def run(
    config: dict,
    parallelism: int,
    log_on_file: bool = False,
    disable_plotting: bool = False
  ):
  base_solution_folder = config["base_solution_folder"]
  seed = config["seed"]
  limits = config["limits"]
  trace_type = config["limits"]["load"].get("trace_type", "fixed_sum")
  verbose = config.get("verbose", 0)
  solver_name = config["solver_name"]
  solver_options = config["solver_options"]
  general_solver_options = solver_options.get("general", {})
  gcaa_options = solver_options["gcaa"]
  if gcaa_options.get("unit_bids") is not True:
    raise ValueError("solver_options.gcaa.unit_bids must be true")
  time_limit = general_solver_options.get("TimeLimit", np.inf)
  tolerance = config.get("tolerance", 1e-6)
  max_iterations = config["max_iterations"]
  max_steps = config["max_steps"]
  min_run_time = config.get("min_run_time", 0)
  max_run_time = config.get("max_run_time", max_steps)
  run_time_step = config.get("run_time_step", 1)
  checkpoint_interval = config["checkpoint_interval"]
  # generate solution folder
  now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
  solution_folder = f"{base_solution_folder}/{now}"
  os.makedirs(solution_folder, exist_ok = True)
  with open(os.path.join(solution_folder, "config.json"), "w") as ostream:
    ostream.write(json.dumps(config, indent = 2))
  log_stream = sys.stdout
  if log_on_file:
    log_stream = open(os.path.join(solution_folder, "out.log"), "w")
  base_instance_data, input_requests_traces, agents, graph = init_problem(
    limits, trace_type, max_steps, seed, solution_folder
  )
  Nn = base_instance_data[None]["Nn"][None]
  Nf = base_instance_data[None]["Nf"][None]
  neighborhood = neigh_dict_to_matrix(
    base_instance_data[None]["neighborhood"], Nn
  )
  latency = adjacency_matrix(graph, weight = "network_latency")
  ub = (
    max_run_time + run_time_step
  ) if max_run_time == min_run_time else max_run_time
  sp_complete_solution = init_complete_solution()
  spc_complete_solution = init_complete_solution()
  obj_dict = {"LSPr_final": []}
  tc_dict = {"LSPr": []}
  runtime_list = []
  # GCAA bids are pure utility: price stays at zero forever (no
  # evaluate_bids-style price adaptation)
  no_price = np.zeros((Nn, Nf))
  # zero replica capacity disables define_bids' memory-bid path: GCAA
  # has no replica-bidding, a buyer with no convenient seller just gets
  # the null assignment (utility 0) for this round
  no_replica_sellers = np.zeros((Nn,))
  for t in range(min_run_time, ub, run_time_step):
    if verbose > 0:
      print(f"t = {t}", file = log_stream, flush = True)
    loadt = get_current_load(input_requests_traces, agents, t)
    data = update_data(base_instance_data, {"incoming_load": loadt})
    total_runtime = 0
    ss = datetime.now()
    sp_data = deepcopy(data)
    sp = LSP()
    spr = LSPr()
    (
      sp_data, sp_x, _, _, sp_omega, sp_r, sp_rho, sp_U, obj, tc, sp_runtime
    ) = solve_subproblem(
      sp_data, agents, sp, solver_name, general_solver_options, parallelism
    )
    if verbose > 1:
      print(
        f"    sp: DONE ({tc['tot']}; obj = {obj['tot']}; "
        f"runtime = {sp_runtime['tot']})",
        file = log_stream, flush = True
      )
    total_runtime += sp_runtime["tot"]
    it = 0
    stop_searching = False
    best_solution_so_far = None
    best_centralized_solution = None
    best_cost_so_far = np.inf
    spr_obj = np.inf
    best_centralized_cost = -np.inf
    best_it_so_far = -1
    best_centralized_it = -1
    y = np.zeros((Nn, Nn, Nf))
    omega = deepcopy(sp_omega)
    fairness = np.zeros((Nn, Nf))
    while not stop_searching:
      if verbose > 0:
        print(f"    it = {it}", file = log_stream, flush = True)
      capacity, residual_capacity, ell = compute_residual_capacity(
        sp_x, y, sp_r, sp_data
      )
      blackboard = np.maximum(0.0, capacity - sp_x)
      bids, memory_bids, n_auctions = define_bids(
        omega, blackboard, no_price, sp_data, neighborhood,
        no_replica_sellers, gcaa_options, latency, fairness,
        force_memory_bids = False
      )
      if verbose > 2:
        print(bids, file = log_stream, flush = True)
      rmp_omega = np.zeros((Nn, Nf))
      if len(bids) > 0:
        auction_y = resolve_gcaa_round(bids, residual_capacity, y)
        y += auction_y
        for n in range(Nn):
          for f in range(Nf):
            rmp_omega[n,f] = y[n,:,f].sum()
            if rmp_omega[n,f] > 0:
              fairness[n,f] += 1
        bad_nodes = check_ls_pr_feasibility_from_fixed_y(sp_data, y)
        if bad_nodes:
          raise RuntimeError(
            f"LSPr infeasible from fixed y assignments: {bad_nodes}"
          )
        spr_sol, spr_obj, spr_tc, spr_runtime = compute_social_welfare(
          spr, sp_data, agents, solver_name, general_solver_options,
          y, rmp_omega, parallelism
        )
        total_runtime += spr_runtime
        sp_x, _, _, _, sp_r, sp_rho = spr_sol
        for i in range(Nn):
          for f in range(Nf):
            omega[i,f] = sp_omega[i,f] - rmp_omega[i,f]
            if abs(omega[i,f]) < tolerance:
              omega[i,f] = 0.0
      csol = combine_solutions(
        Nn, Nf, sp_data, loadt, sp_x, sp_r, sp_rho,
        None, y, None, None, None, None
      )
      cobj = compute_centralized_objective(
        sp_data, csol["sp"]["x"], csol["sp"]["y"], csol["sp"]["z"]
      )
      feas = check_feasibility(
        csol["sp"]["x"], csol["sp"]["y"].sum(axis=1), csol["sp"]["z"],
        csol["sp"]["r"], csol["sp"]["U"], sp_data
      )
      assert feas[0], feas[1]
      if spr_obj < best_cost_so_far or it == 0:
        best_cost_so_far = spr_obj
        best_solution_so_far = deepcopy(csol)
        best_it_so_far = it
      if best_centralized_solution is None or cobj > best_centralized_cost:
        best_centralized_cost = cobj
        best_centralized_solution = deepcopy(csol)
        best_centralized_it = it
      stop_searching, why_stop_searching = check_stopping_criteria(
        it, max_iterations, blackboard, omega, rmp_omega, None,
        bids, memory_bids, tolerance, total_runtime, time_limit
      )
      if not stop_searching:
        it += 1
      else:
        sp_complete_solution, _, objf = decode_solutions(
          sp_data, best_solution_so_far, sp_complete_solution, None
        )
        spc_complete_solution, _, _ = decode_solutions(
          sp_data, best_centralized_solution, spc_complete_solution, None
        )
        obj_dict["LSPr_final"].append(objf)
        tc_dict["LSPr"].append(
          f"{why_stop_searching} "
          f"(it: {it}; obj. deviation: {None}; best it: {best_it_so_far}; "
          f"total runtime: {total_runtime})"
        )
        if t % checkpoint_interval == 0 or t == max_steps - 1:
          save_checkpoint(
            sp_complete_solution, os.path.join(solution_folder, "LSP"), t
          )
          save_checkpoint(
            spc_complete_solution, os.path.join(solution_folder, "LSPc"), t
          )
    ee = datetime.now()
    if verbose > 0:
      print(
        f"    TOTAL RUNTIME [s] = {total_runtime} "
        f"(wallclock: {(ee-ss).total_seconds()})",
        file = log_stream, flush = True
      )
    runtime_list.append(total_runtime)
  sp_solution, sp_offloaded, sp_detailed_fwd_solution = join_complete_solution(
    sp_complete_solution
  )
  spc_solution, spc_offloaded, spc_detailed_fwd_solution = join_complete_solution(
    spc_complete_solution
  )
  if not disable_plotting and Nf <= 10 and Nn <= 10:
    plot_history(
      input_requests_traces, min_run_time, max_run_time, run_time_step,
      sp_solution, sp_complete_solution["utilization"],
      sp_complete_solution["replicas"], sp_offloaded,
      obj_dict["LSPr_final"], os.path.join(solution_folder, "sp.png")
    )
  save_solution(
    sp_solution, sp_offloaded, sp_complete_solution,
    sp_detailed_fwd_solution, "LSP", solution_folder
  )
  save_solution(
    spc_solution, spc_offloaded, spc_complete_solution,
    spc_detailed_fwd_solution, "LSPc", solution_folder
  )
  pd.DataFrame(obj_dict["LSPr_final"], columns = ["FaaS-MAGCAA"]).to_csv(
    os.path.join(solution_folder, "obj.csv"), index = False
  )
  pd.DataFrame(tc_dict["LSPr"]).to_csv(
    os.path.join(solution_folder, "termination_condition.csv")
  )
  pd.DataFrame({"tot": runtime_list}).to_csv(
    os.path.join(solution_folder, "runtime.csv"), index = False
  )
  if verbose > 0:
    print(
      f"All solutions saved in: {solution_folder}",
      file = log_stream, flush = True
    )
  if log_on_file:
    log_stream.close()
  return solution_folder


if __name__ == "__main__":
  args = parse_arguments()
  config = load_configuration(args.config)
  run(
    config, args.parallelism, log_on_file = False,
    disable_plotting = args.disable_plotting
  )
