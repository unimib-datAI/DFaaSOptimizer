from run_centralized_model import init_problem, get_current_load, update_data
from run_faasmacro import solve_subproblem, update_neighborhood
from utilities import load_configuration
from sp import LSP

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
    bids: pd.DataFrame, blackboard: np.array
  ):
  # loop over agents and functions
  potential_sellers, functions_to_share = np.nonzero(blackboard)
  for j,f in zip(potential_sellers,functions_to_share):
    # extract bids for the current node
    bids_for_j = bids[(bids["j"] == j) & (bids["f"] == f)].sort_values(
      by = "b", ascending = False
    )


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
      capacity, blackboard, ell = compute_residual_capacity(
        sp_x, y, sp_r, sp_data
      )
      # # update neighborhood given the nodes residual memory capacity
      # neighborhood = update_neighborhood(
      #   data[None]["neighborhood"], sp_rho, sp_omega
      # )
      # buyers define their bids
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
      # sellers accept/reject bids
      evaluate_bids(bids, blackboard)
      #
      print("here")
      u = ell / capacity
      p += auction_options["eta"] * (u - u0)


if __name__ == "__main__":
  # args = parse_arguments()
  # config_file = args.config
  # parallelism = args.parallelism
  # disable_plotting = args.disable_plotting
  config_file = "manual_config.json"
  parallelism = 0
  disable_plotting = True
  # load configuration file
  config = load_configuration(config_file)
  # run
  run(
    config, 
    parallelism, 
    log_on_file = False, 
    disable_plotting = disable_plotting
  )
