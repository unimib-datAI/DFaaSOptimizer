from run_centralized_model import (
  init_complete_solution,
  join_complete_solution,
  get_current_load, 
  save_checkpoint,
  save_solution,
  plot_history,
  init_problem, 
  update_data
)
from run_faasmacro import (
  compute_centralized_objective,
  compute_social_welfare,
  combine_solutions, 
  decode_solutions,
  solve_subproblem
)
from run_faasmadea import (
  check_stopping_criteria,
  compute_residual_capacity,
  define_bids, 
  evaluate_bids,
  neigh_dict_to_matrix,
  relative_objective_gap,
  start_additional_replicas
)
from utils.common import load_configuration
from models.sp import LSP, LSPr_x

from networkx import adjacency_matrix
from collections import deque
from datetime import datetime
from copy import deepcopy
from typing import Tuple
import pandas as pd
import numpy as np
import argparse
import json
import sys
import os


def parse_arguments() -> argparse.Namespace:
  """
  Parse input arguments
  """
  parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description = "Run FaaS-MADeA", 
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument(
    "-c", "--config",
    help = "Configuration file",
    type = str,
    default = "config_files/manual_config.json"
  )
  parser.add_argument(
    "-j", "--parallelism",
    help = "Number of parallel processes to start (-1: auto, 0: sequential)",
    type = int,
    default = 0
  )
  parser.add_argument(
    "--disable_plotting",
    help = "True to disable automatic plot generation for each experiment",
    default = False,
    action = "store_true"
  )
  # Parse the arguments
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
  patience = config.get("patience", 1)
  verbose = config.get("verbose", 0)
  # -- solver name and options
  solver_name = config["solver_name"]
  solver_options = config["solver_options"]
  general_solver_options = solver_options.get("general", {})
  auction_options = solver_options["auction"]
  time_limit = general_solver_options.get("TimeLimit", np.inf)
  tolerance = config.get("tolerance", 1e-6)
  # -- maximum number of iterations and time limits
  max_iterations = config["max_iterations"]
  max_steps = config["max_steps"]
  min_run_time = config.get("min_run_time", 0)
  max_run_time = config.get("max_run_time", max_steps)
  run_time_step = config.get("run_time_step", 1)
  checkpoint_interval = config["checkpoint_interval"]
  plot_interval = config.get("plot_interval", max_iterations)
  # generate solution folder
  now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
  solution_folder = f"{base_solution_folder}/{now}"
  os.makedirs(solution_folder, exist_ok = True)
  with open(os.path.join(solution_folder, "config.json"), "w") as ostream:
    ostream.write(json.dumps(config, indent = 2))
  # initialize log stream (if required)
  log_stream = sys.stdout
  if log_on_file:
    log_stream = open(os.path.join(solution_folder, "out.log"), "w")
  # generate base instance data and load traces
  base_instance_data, input_requests_traces, agents, graph = init_problem(
    limits, trace_type, max_steps, seed, solution_folder
  )
  Nn = base_instance_data[None]["Nn"][None]
  Nf = base_instance_data[None]["Nf"][None]
  # -- save neighborhood matrix
  neighborhood = neigh_dict_to_matrix(
    base_instance_data[None]["neighborhood"], Nn
  )
  latency = adjacency_matrix(graph, weight = "network_latency")
  # loop over time
  ub = (
    max_run_time + run_time_step
  ) if max_run_time == min_run_time else max_run_time
  sp_complete_solution = init_complete_solution()
  spc_complete_solution = init_complete_solution()
  obj_dict = {"LSPr_final": []}
  tc_dict = {"LSPr": []}
  for t in range(min_run_time, ub, run_time_step):
    if verbose > 0:
      print(f"t = {t}", file = log_stream, flush = True)
    # get current load and generate data
    loadt = get_current_load(input_requests_traces, agents, t)
    data = update_data(base_instance_data, {"incoming_load": loadt})
    # local planning
    total_runtime = 0
    ss = datetime.now()
    # -- solve subproblem
    sp = LSP()
    spr = LSPr_x()
    sp_data = deepcopy(data)
    s = datetime.now()
    (
      sp_data, sp_x, _, _, sp_omega, sp_r, sp_rho, sp_U, obj, tc, sp_runtime
    ) = solve_subproblem(
      sp_data, 
      agents, 
      sp, 
      solver_name, 
      general_solver_options, 
      parallelism
    )
    e = datetime.now()
    if verbose > 1:
      print(
        f"    sp: DONE ",
        f"({tc['tot']}; obj = {obj['tot']}; runtime = {sp_runtime['tot']})", 
        file = log_stream, 
        flush = True
      )
    total_runtime += sp_runtime["tot"]
    # define target operating point and initial prices
    u0 = np.ones((Nn,Nf)) * 0.9
    p = np.zeros((Nn,Nf))
    # loop over iterations
    it = 0
    stop_searching = False
    best_solution_so_far = None
    best_centralized_solution = None
    best_cost_so_far = np.inf
    best_centralized_cost = 0.0
    best_it_so_far = -1
    best_centralized_it = -1
    y = np.zeros((Nn,Nn,Nf))
    omega = deepcopy(sp_omega)
    fairness = np.zeros((Nn,Nf))
    odev_queue = deque(maxlen=patience)
    while not stop_searching:
      if verbose > 0:
        print(f"    it = {it}", file = log_stream, flush = True)
      # compute residual computational capacity
      s = datetime.now()
      capacity, blackboard, ell = compute_residual_capacity(
        sp_x, y, sp_r, sp_data
      )
      e = datetime.now()
      if verbose > 1:
        print(
          f"        compute_residual_capacity: DONE ",
          f"({capacity.tolist()}; blackboard = {blackboard.tolist()}; "
          f"ell = {ell.tolist()}; runtime = {(e - s).total_seconds()})", 
          file = log_stream, 
          flush = True
        )
      total_runtime += (e - s).total_seconds()
      # buyers define their bids
      s = datetime.now()
      bids, memory_bids, n_auctions = define_bids(
        omega, 
        blackboard, 
        p, 
        sp_data, 
        neighborhood, 
        sp_rho,
        auction_options, 
        latency,
        fairness,
        False
      )
      e = datetime.now()
      rt = (e - s).total_seconds()
      if verbose > 1:
        print(
          f"        define_bids: DONE; runtime = {rt/max(n_auctions, 1)}; "
          f"n_auctions = {n_auctions}; tot runtime = {rt})",
          file = log_stream,
          flush = True
        )
        if verbose > 2:
          print(bids, file = log_stream, flush = True)
      total_runtime += (e - s).total_seconds()
      # sellers accept/reject bids
      if len(bids) > 0:
        s = datetime.now()
        auction_y, p, _, _ = evaluate_bids(
          bids, 
          blackboard, 
          data, 
          previous_y = y,
          ell = ell, 
          p = p, 
          capacity = capacity, 
          u0 = u0, 
          auction_options = auction_options,
          tentatively_start_replicas = False,
          may_replace_existing_assignments = False
        )
        e = datetime.now()
        if verbose > 1:
          print(
           f"        evaluate_bids: DONE; runtime = {(e - s).total_seconds()})", 
           file = log_stream, 
           flush = True
          )
        total_runtime += (e - s).total_seconds()
        # update effective load, number of replicas and fairness matrix
        y += auction_y
        rmp_omega = np.zeros((Nn,Nf))
        for n in range(Nn):
          for f in range(Nf):
            rmp_omega[n,f] = y[n,:,f].sum()
            if rmp_omega[n,f] > 0:
              fairness[n,f] += 1
        # -- solve "restricted problem"
        spr_sol, spr_obj, spr_tc, spr_runtime = compute_social_welfare(
          spr, 
          sp_data, 
          agents, 
          solver_name, 
          general_solver_options, 
          y, 
          rmp_omega,
          parallelism,
          sp_x
        )
        total_runtime += spr_runtime
        if verbose > 1:
          print(
            f"        solve 'restricted problem': DONE ({spr_tc}; "
            f"obj: {spr_obj}; runtime = {spr_runtime})", 
            file = log_stream, 
            flush = True
          )
        # -- update solution
        _, _, _, _, sp_r, sp_rho = spr_sol # x, y, z, omega, r, rho
        for i in range(Nn):
          for f in range(Nf):
            omega[i,f] = sp_omega[i,f] - rmp_omega[i,f]
            if abs(omega[i,f]) < tolerance:
              omega[i,f] = 0.0
        if verbose > 1:
          print(
            f"        solution updated: DONE (auct_y = {auction_y.tolist()}; "
            f"omega = {omega.tolist()}; x: {sp_x.tolist()}; "
            f"r = {sp_r.tolist()}; rho = {sp_rho.tolist()})", 
            file = log_stream, 
            flush = True
          )
      else:
        # tentatively start additional replicas
        s = datetime.now()
        a, sp_rho = start_additional_replicas(
          memory_bids, sp_r, sp_data, sp_rho
        )
        sp_r += a
        e = datetime.now()
        print(
          f"        additional replicas started: DONE (a = {a.tolist()}; "
          f"rho = {sp_rho.tolist()}; runtime = {(e - s).total_seconds()})", 
          file = log_stream, 
          flush = True
        )
        total_runtime += (e - s).total_seconds()
      # merge solutions and compute the centralized objective value
      csol = combine_solutions(
        Nn, Nf, sp_data, loadt, 
        sp_x, sp_r, sp_rho,
        None, y, None, None, None, None
      )
      cobj = compute_centralized_objective(
        sp_data, csol["sp"]["x"], csol["sp"]["y"], csol["sp"]["z"]
      )
      # update best solution so far
      if spr_obj < best_cost_so_far:
        best_cost_so_far = spr_obj
        best_solution_so_far = csol
        best_it_so_far = it
        if verbose > 0:
          print(
            f"        best solution updated; obj = {spr_obj}",
            file = log_stream,
            flush = True
          )
      prev_cobj = best_centralized_cost
      if cobj > best_centralized_cost:
        best_centralized_cost = cobj
        best_centralized_solution = csol
        best_centralized_it = it
        if verbose > 0:
          print(
            f"        best centralized solution updated; obj = {cobj}",
            file = log_stream,
            flush = True
          )
      odev_queue.append(
        relative_objective_gap(prev_cobj, best_centralized_cost)
      )
      # check termination criteria
      s = datetime.now()
      stop_searching, why_stop_searching = check_stopping_criteria(
        it = it,
        max_iterations = max_iterations,
        blackboard = blackboard,
        omega = omega,
        rmp_omega = rmp_omega,
        odev_queue = odev_queue,
        bids = bids,
        memory_bids = memory_bids,
        tolerance = tolerance,
        total_runtime = total_runtime,
        time_limit = time_limit
      )
      e = datetime.now()
      if verbose > 1:
        print(
          f"        check_stopping_criteria: DONE "
          f"(runtime = {(e - s).total_seconds()}; "
          f"total runtime = {total_runtime}; "
          f"wallclock: {(datetime.now() - ss).total_seconds()}) "
          f"--> stop? {stop_searching} ({why_stop_searching})", 
          file = log_stream, 
          flush = True
        )
      # -- move to next iteration, or...
      if not stop_searching:
        it += 1
      # -- ...save solution
      else:
        # save solutions
        sp_complete_solution, _, objf = decode_solutions(
          sp_data, 
          best_solution_so_far, 
          sp_complete_solution, 
          None
        )
        spc_complete_solution, _, _ = decode_solutions(
          sp_data, 
          best_centralized_solution, 
          spc_complete_solution, 
          None
        )
        obj_dict["LSPr_final"].append(objf)
        tc_dict["LSPr"].append(
          f"{why_stop_searching} "
          f"(it: {it}; obj. deviation: {None}; best it: {best_it_so_far}; "
          f"best centralized it: {best_centralized_it}; "
          f"total runtime: {total_runtime})"
        )
        # save checkpoint
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
        file = log_stream, 
        flush = True
      )
  # join
  sp_solution, sp_offloaded, sp_detailed_fwd_solution = join_complete_solution(
    sp_complete_solution
  )
  spc_solution, spc_offloaded, spc_detailed_fwd_solution = join_complete_solution(
    spc_complete_solution
  )
  if not disable_plotting and Nf <= 10 and Nn <= 10:
    plot_history(
      input_requests_traces, 
      min_run_time,
      max_run_time,
      run_time_step,
      sp_solution, 
      sp_complete_solution["utilization"], 
      sp_complete_solution["replicas"], 
      sp_offloaded,
      # obj_dict["LSP"][max_iterations-1],
      obj_dict["LSPr_final"],
      os.path.join(solution_folder, "sp.png")
    )
  save_solution(
    sp_solution,
    sp_offloaded,
    sp_complete_solution,
    sp_detailed_fwd_solution,
    "LSP",
    solution_folder
  )
  save_solution(
    spc_solution,
    spc_offloaded,
    spc_complete_solution,
    spc_detailed_fwd_solution,
    "LSPc",
    solution_folder
  )
  # save objective function values
  pd.DataFrame(obj_dict["LSPr_final"], columns = ["FaaS-MADeA"]).to_csv(
    os.path.join(solution_folder, "obj.csv"), index = False
  )
  # save models termination condition
  pd.DataFrame(tc_dict["LSPr"]).to_csv(
    os.path.join(solution_folder, "termination_condition.csv")
  )
  if verbose > 0:
    print(
      f"All solutions saved in: {solution_folder}", 
      file = log_stream, 
      flush = True
    )
  # close log stream if needed
  if log_on_file:
    log_stream.close()
  return solution_folder
      


if __name__ == "__main__":
  args = parse_arguments()
  config_file = args.config
  parallelism = args.parallelism
  disable_plotting = args.disable_plotting
  # load configuration file
  config = load_configuration(config_file)
  # run
  run(
    config, 
    parallelism, 
    log_on_file = False, 
    disable_plotting = disable_plotting
  )
