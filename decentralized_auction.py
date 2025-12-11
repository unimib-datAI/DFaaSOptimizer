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
from utilities import load_configuration
from sp import LSP, LSPr

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
  # Parse the arguments
  args: argparse.Namespace = parser.parse_known_args()[0]
  return args


def check_stopping_criteria(
    it: int,
    max_iterations: int,
    blackboard: np.array,
    sp_omega: np.array,
    rmp_omega: np.array,
    tolerance: float,
    total_runtime: float,
    time_limit: float
  ) -> Tuple[bool, str]:
  stop = False
  why_stopping = None
  if it >= max_iterations - 1:
    stop = True
    why_stopping = "max iterations reached"
  elif (blackboard <= tolerance).all():
    stop = True
    why_stopping = "no capacity left"
  elif (sp_omega <= tolerance).all():
    stop = True
    why_stopping = "all load assigned"
  elif (rmp_omega <= tolerance).all():
    stop = True
    why_stopping = "load cannot be assigned"
  elif total_runtime >= time_limit:
    stop = True
    why_stopping = f"reached time limit: {total_runtime} >= {time_limit}"
  return stop, why_stopping


def compute_residual_capacity(
    x: np.array, y: np.array, r: np.array, data: dict
  ) -> Tuple[np.array, np.array, np.array]:
  Nn = data[None]["Nn"][None]
  Nf = data[None]["Nf"][None]
  # loop over nodes and functions
  cap = np.zeros((Nn,Nf))
  c = np.zeros((Nn,Nf))
  ell = np.zeros((Nn,Nf))
  for n in range(Nn):
    for f in range(Nf):
      # number of enqueued requests
      ell[n,f] = x[n,f] + y[:,n,f].sum()
      # computational capacity
      cap[n,f] = r[n,f] * (
        data[None]["max_utilization"][f+1] / data[None]["demand"][(n+1,f+1)]
      )
      # residual capacity
      c[n,f] = max(0.0, cap[n,f] - ell[n,f])
  return cap, c, ell


def define_bids(
    omega: np.array,
    blackboard: np.array, 
    p: np.array, 
    data: dict,
    neighborhood: np.array,
    auction_options: dict,
    latency: np.array,
    fairness: np.array,
    delta: np.array
  ) -> pd.DataFrame:
  # loop over agents and functions
  potential_buyers, functions_to_share = np.nonzero(omega)
  bids = {
    "i": [], "j": [], "f": [], "d": [], "b": [], "utility": [], "weight": []
  }
  for i,f in zip(potential_buyers, functions_to_share):
    # identify potential sellers
    potential_sellers = np.nonzero(neighborhood[i,:])[0]
    utility = []
    candidate_sellers = []
    for j in potential_sellers:
      # -- check residual capacity
      if blackboard[j,f] > 0:
        # -- compute utility
        ut = (  
          data[None]["beta"][(i+1,j+1,f+1)] - 
          p[j,f] - 
          auction_options["latency_weight"] * latency[i,j] - 
          auction_options["fairness_weight"] * fairness[i,f]
        )
        if ut > 0:
          utility.append(ut)
          candidate_sellers.append(j)
    # compute weights and define bids
    if len(utility) > 0:
      utility = np.array(utility)
      weights = np.exp(utility) / np.sum(np.exp(utility))
      for idx,(j,w) in enumerate(zip(candidate_sellers,weights)):
        d = min(blackboard[j,f], w * omega[i,f])
        b = p[j,f] + auction_options["epsilon"] + delta[i,f]
        bids["i"].append(i)
        bids["f"].append(f)
        bids["j"].append(j)
        bids["d"].append(d)
        bids["b"].append(b)
        bids["utility"].append(utility[idx])
        bids["weight"].append(w)
  return pd.DataFrame(bids)


def evaluate_bids(
    bids: pd.DataFrame, 
    blackboard: np.array, 
    data: dict, 
    ell: np.array, 
    p: np.array, 
    capacity: np.array, 
    u0: np.array, 
    auction_options: dict
  ) -> np.array:
  Nn = data[None]["Nn"][None]
  Nf = data[None]["Nf"][None]
  # loop over agents and functions
  potential_sellers, functions_to_share = np.nonzero(blackboard)
  y = np.zeros((Nn,Nn,Nf))
  for j,f in zip(potential_sellers,functions_to_share):
    # extract bids for the current node
    bids_for_j = bids[(bids["j"] == j) & (bids["f"] == f)].sort_values(
      by = "b", ascending = False
    )
    remaining_capacity = blackboard[j,f]
    next_bid_idx = 0
    min_b = bids_for_j["b"].max()
    # loop over bids until there is remaining capacity
    while next_bid_idx < len(bids_for_j) and remaining_capacity > 0:
      q = min(remaining_capacity, bids_for_j.iloc[next_bid_idx]["d"])
      y[int(bids_for_j.iloc[next_bid_idx]["i"]),j,f] += q
      remaining_capacity -= q
      min_b = min(min_b, bids_for_j.iloc[next_bid_idx]["b"])
      next_bid_idx += 1
    # compute utilization and update prices
    if next_bid_idx > 0:
      u = (ell[j,f] + y[:,j,f].sum()) / capacity[j,f]
      p[j,f] = min_b + auction_options["eta"] * (u - u0[j,f])
    else:
      p[j,f] *= (1 - auction_options["zeta"])
  return y, p


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
  base_instance_data, input_requests_traces, agents = init_problem(
    limits, trace_type, max_steps, seed, solution_folder
  )
  Nn = base_instance_data[None]["Nn"][None]
  Nf = base_instance_data[None]["Nf"][None]
  # -- save neighborhood matrix
  neighborhood = np.zeros((Nn,Nn))
  for n1 in range(Nn):
    for n2 in range(Nn):
      if n1 != n2 and base_instance_data[None]["neighborhood"][(n1+1,n2+1)]:
        neighborhood[n1,n2] = 1
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
    spr = LSPr()
    sp_data = deepcopy(data)
    s = datetime.now()
    (
      sp_data, sp_x, sp_omega, sp_r, sp_rho, sp_U, obj, tc, sp_runtime
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
          f"    compute_residual_capacity: DONE ",
          f"({capacity.tolist()}; blackboard = {blackboard.tolist()}; "
          f"ell = {ell.tolist()}; runtime = {(e - s).total_seconds()})", 
          file = log_stream, 
          flush = True
        )
      total_runtime += (e - s).total_seconds()
      # # update neighborhood given the nodes residual memory capacity
      # neighborhood = update_neighborhood(
      #   data[None]["neighborhood"], sp_rho, sp_omega
      # )
      # buyers define their bids
      s = datetime.now()
      bids = define_bids(
        sp_omega, 
        blackboard, 
        p, 
        sp_data, 
        neighborhood, 
        auction_options, 
        np.zeros((Nn,Nn)), 
        np.zeros((Nn,Nf)),
        np.zeros((Nn,Nf))
      )
      e = datetime.now()
      if verbose > 1:
        print(
          f"    define_bids: DONE ",
          f"; runtime = {(e - s).total_seconds()})", 
          file = log_stream, 
          flush = True
        )
      total_runtime += (e - s).total_seconds()
      # sellers accept/reject bids
      if len(bids) > 0:
        s = datetime.now()
        auction_y, p = evaluate_bids(
          bids, blackboard, data, ell, p, capacity, u0, auction_options
        )
        e = datetime.now()
        if verbose > 1:
          print(
            f"    evaluate_bids: DONE ",
            f"; runtime = {(e - s).total_seconds()})", 
            file = log_stream, 
            flush = True
          )
        total_runtime += (e - s).total_seconds()
        # update effective load and number of replicas
        y += auction_y
        rmp_omega = np.zeros((Nn,Nf))
        for n in range(Nn):
          for f in range(Nf):
            rmp_omega[n,f] = y[n,:,f].sum()
        # -- solve "restricted problem"
        spr_sol, spr_obj, spr_tc, spr_runtime = compute_social_welfare(
          spr, 
          sp_data, 
          agents, 
          solver_name, 
          general_solver_options, 
          y, 
          rmp_omega,
          parallelism
        )
        total_runtime += spr_runtime
        if verbose > 1:
          print(
            f"        solve 'restricted problem': DONE ({spr_tc}; obj: {spr_obj}"
            f"; runtime = {spr_runtime})", 
            file = log_stream, 
            flush = True
          )
        # -- update solution
        sp_x, _, sp_r, sp_rho = spr_sol
        sp_omega -= rmp_omega
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
            f"        best solution updated; obj = {cobj}",
            file = log_stream,
            flush = True
          )
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
      # check termination criteria
      s = datetime.now()
      stop_searching, why_stop_searching = check_stopping_criteria(
        it,
        max_iterations,
        blackboard,
        sp_omega,
        rmp_omega,
        tolerance,
        total_runtime,
        time_limit
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
  pd.DataFrame(obj_dict["LSPr_final"], columns = ["FaaS-MACrO"]).to_csv(
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
  # args = parse_arguments()
  # config_file = args.config
  # parallelism = args.parallelism
  # disable_plotting = args.disable_plotting
  config_file = "manual_config.json"
  parallelism = 0
  disable_plotting = False
  # load configuration file
  config = load_configuration(config_file)
  # run
  run(
    config, 
    parallelism, 
    log_on_file = False, 
    disable_plotting = disable_plotting
  )
