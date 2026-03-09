from postprocessing import load_solution
from run_centralized_model import (
  encode_solution,
  get_current_load, 
  init_complete_solution,
  init_problem, 
  join_complete_solution,
  plot_history,
  save_checkpoint,
  save_solution,
  update_data
)
from run_faasmacro import (
  combine_solutions, 
  compute_centralized_objective,
  compute_social_welfare,
  decode_solutions,
  solve_subproblem
)
from utilities import load_configuration
from models.sp import LSP, LSPr, LSP_fixedr

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
    omega: np.array,
    rmp_omega: np.array,
    a: np.array,
    bids: pd.DataFrame,
    memory_bids: pd.DataFrame,
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
  elif (omega <= tolerance).all():
    stop = True
    why_stopping = "all load assigned"
  elif (rmp_omega <= tolerance).all() and (a <= tolerance).all():
    stop = True
    why_stopping = "load cannot be assigned"
  elif len(bids) == 0 and len(memory_bids) == 0:
    stop = True
    why_stopping = "no available or convenient sellers"
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
  residual_capacity = np.zeros((Nn,Nf))
  ell = np.zeros((Nn,Nf))
  for n in range(Nn):
    for f in range(Nf):
      # number of enqueued requests
      ell[n,f] = x[n,f] + y[:,n,f].sum()
      # computational capacity
      cap[n,f] = r[n,f] * (
        data[None]["max_utilization"][f+1] / data[None]["demand"][(n+1,f+1)]
      )
      # residual capacity (the blackboard does not consider y)
      c[n,f] = max(0.0, cap[n,f] - x[n,f])
      residual_capacity[n,f] = max(0.0, cap[n,f] - ell[n,f])
  return cap, c, residual_capacity, ell


def define_bids(
    omega: np.array,
    blackboard: np.array, 
    p: np.array, 
    data: dict,
    neighborhood: np.array,
    rho: np.array,
    auction_options: dict,
    latency: np.array,
    fairness: np.array,
    force_memory_bids: bool
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
  # loop over agents and functions
  potential_buyers, functions_to_share = np.nonzero(omega)
  bids = {
    "i": [], "j": [], "f": [], "d": [], "b": [], "utility": []
  }
  memory_bids = {
    "i": [], "j": [], "f": []
  }
  for i,f in zip(potential_buyers, functions_to_share):
    # identify potential sellers
    potential_sellers = set(np.nonzero(neighborhood[i,:])[0])
    # -- capacity sellers are neighbors with residual computing capacity 
    # for function f
    potential_capacity_sellers = potential_sellers.intersection(
      set(np.where(blackboard[:,f]>=1)[0])
    )
    # -- memory sellers are neighbors with residual memory capacity to 
    # instantiate new replicas
    potential_memory_sellers = potential_sellers.intersection(
      set(np.nonzero(rho)[0])
    )
    # -- loop over potential sellers
    utility = []
    candidate_sellers = []
    for j in potential_capacity_sellers:
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
    assigned = 0
    if len(utility) > 0:
      utility = np.array(utility)
      sellers_order = np.argsort(utility)[::-1]
      idx = 0
      while idx < len(sellers_order) and assigned < omega[i,f]:
        j = candidate_sellers[sellers_order[idx]]
        delta = 0.0
        if idx < len(sellers_order) - 1:
          delta = utility[sellers_order[idx]] - utility[sellers_order[idx+1]]
        b = p[j,f] + auction_options["epsilon"] + delta
        d = 1
        while (d < int(min(blackboard[j,f], omega[i,f])) + 1) and (
            assigned < omega[i,f]
          ):
          bids["i"].append(i)
          bids["f"].append(f)
          bids["j"].append(j)
          bids["d"].append(1)
          bids["b"].append(b)
          bids["utility"].append(utility[idx])
          assigned += 1
          d += 1
        idx += 1
      # -- if you could not bid for everything, ask also for new replicas
      if assigned < omega[i,f]:
        for idx in sellers_order:
          j = candidate_sellers[idx]
          if j in potential_memory_sellers:
            memory_bids["i"].append(i)
            memory_bids["j"].append(j)
            memory_bids["f"].append(f)
    # if you could not bid for all, ask for new replicas
    if assigned < omega[i,f] or force_memory_bids:
      for j in potential_memory_sellers - potential_capacity_sellers:
        memory_bids["i"].append(i)
        memory_bids["j"].append(j)
        memory_bids["f"].append(f)
  return pd.DataFrame(bids), pd.DataFrame(memory_bids)


def ensure_memory_sellers(
    potential_sellers, functions_to_share,
    potential_memory_sellers, Nf
  ):
  extra_sellers = np.repeat(potential_memory_sellers, Nf)
  extra_funcs = np.tile(np.arange(Nf), len(potential_memory_sellers))
  sellers_all = np.concatenate((potential_sellers, extra_sellers))
  funcs_all = np.concatenate((functions_to_share, extra_funcs))
  pairs = np.unique(np.column_stack((sellers_all, funcs_all)), axis=0)
  return pairs[:,0], pairs[:,1]


def evaluate_bids(
    bids: pd.DataFrame, 
    blackboard: np.array, 
    data: dict, 
    last_y: np.array,
    ell: np.array, 
    p: np.array, 
    capacity: np.array, 
    u0: np.array, 
    auction_options: dict,
    rho: np.array,
    r: np.array,
    tentatively_start_replicas: bool
  ) -> np.array:
  Nn = data[None]["Nn"][None]
  Nf = data[None]["Nf"][None]
  # loop over agents and functions
  potential_sellers, functions_to_share = np.nonzero(blackboard)
  if tentatively_start_replicas:
    potential_sellers, functions_to_share = ensure_memory_sellers(
      potential_sellers,
      functions_to_share,
      np.nonzero(rho)[0],
      Nf
    )
  y = np.zeros((Nn,Nn,Nf))
  additional_replicas = np.zeros((Nn,Nf))
  for j,f in zip(potential_sellers,functions_to_share):
    # extract bids for the current node
    bids_for_j = bids[(bids["j"] == j) & (bids["f"] == f)].sort_values(
      by = "b", ascending = False
    )
    remaining_capacity = int(blackboard[j,f])
    next_bid_idx = 0
    min_b = bids_for_j["b"].max()
    # loop over bids until there is remaining capacity
    while next_bid_idx < len(bids_for_j) and remaining_capacity > 0:
      q = min(remaining_capacity, bids_for_j.iloc[next_bid_idx]["d"])
      y[int(bids_for_j.iloc[next_bid_idx]["i"]),j,f] += q
      remaining_capacity -= q
      min_b = min(min_b, bids_for_j.iloc[next_bid_idx]["b"])
      next_bid_idx += 1
    # if computational capacity is exhausted and there are still bids, 
    # consider starting new replicas
    if remaining_capacity == 0 and (
        next_bid_idx > 0 or (next_bid_idx == 0 and len(bids_for_j) > 0)
      ):
      max_a = 0
      if tentatively_start_replicas:
        max_a = int(rho[j] / data[None]["memory_requirement"][f+1])
        if max_a > 0:
          a = 1
          while next_bid_idx < len(bids_for_j) and a <= max_a:
            # -- check utilization with one more replica
            q = bids_for_j.iloc[next_bid_idx]["d"]
            u = data[None]["demand"][(j+1,f+1)] * (
              ell[j,f] + y[:,j,f].sum() + q
            ) / (r[j,f] + a)
            if u <= data[None]["max_utilization"][f+1]:
              # -- if possible, accomodate one more bid...
              y[int(bids_for_j.iloc[next_bid_idx]["i"]),j,f] += q
              min_b = min(min_b, bids_for_j.iloc[next_bid_idx]["b"])
              next_bid_idx += 1
              additional_replicas[j,f] = a
            else:
              # -- ...otherwhise, try to increase replicas
              a += 1
      if not tentatively_start_replicas or max_a == 0:
        # if no additional replicas can start, replace existing assignments
        # -- check who previously won the assignment to j
        previous_buyers = np.nonzero(last_y[:,j,f])[0]
        pbidx = 0
        while next_bid_idx < len(bids_for_j) and pbidx < len(previous_buyers):
          i = int(bids_for_j.iloc[next_bid_idx]["i"])
          if (
              previous_buyers[pbidx] != i and
                bids_for_j.iloc[next_bid_idx]["b"] > p[j,f]
            ):
            q = bids_for_j.iloc[next_bid_idx]["d"]
            y[previous_buyers[pbidx],j,f] -= q
            y[int(bids_for_j.iloc[next_bid_idx]["i"]),j,f] += q
            min_b = min(min_b, bids_for_j.iloc[next_bid_idx]["b"])
            next_bid_idx += 1
          pbidx += 1
    # compute utilization and update prices
    if len(bids_for_j) > 0:
      u = (ell[j,f] + y[:,j,f].sum()) / capacity[j,f]
      p[j,f] = min_b + auction_options["eta"] * (u - u0[j,f])
    else:
      p[j,f] *= (1 - auction_options["zeta"])
  return y, p, additional_replicas


def neigh_dict_to_matrix(neighborhood_dict: dict, Nn: int) -> np.array:
  neighborhood = np.zeros((Nn,Nn))
  for n1 in range(Nn):
    for n2 in range(Nn):
      if n1 != n2 and neighborhood_dict[(n1+1,n2+1)]:
        neighborhood[n1,n2] = 1
  return neighborhood


def start_additional_replicas(
    memory_bids: pd.DataFrame, 
    r: np.array,
    data: dict,
    rho: np.array
  ) -> Tuple[np.array, np.array]:
  # loop over sellers
  additional_replicas = np.zeros(r.shape)
  residual_capacity = deepcopy(rho)
  for j, bids_for_j in memory_bids.groupby("j"):
    if rho[j] > 0:
      # count the fraction that each function requires
      fractions = bids_for_j["f"].value_counts(normalize = True)
      # assign new replicas proportionally to this fraction
      for f, frac in fractions.items():
        # -- check memory requirement
        ram_f = data[None]["memory_requirement"][f+1]
        # -- determine the maximum number of replicas that fit in the 
        # assignable fraction of the residual memory capacity
        a = int((residual_capacity[j] * frac) // ram_f)
        # -- update
        residual_capacity[j] -= int(ram_f * a)
        additional_replicas[j,f] = a
  return additional_replicas, residual_capacity


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
  patience = config["patience"]
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
  # load globally-optimal solution (if provided)
  opt_solution, opt_replicas, opt_detailed_fwd = None, None, None
  if "opt_solution_folder" in config:
    opt_solution, opt_replicas, opt_detailed_fwd, _, _ = load_solution(
      config["opt_solution_folder"], "LoadManagementModel"
    )
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
    # -- extract optimal solution (if provided)
    sp_data = deepcopy(data)
    if opt_solution is not None:
      _, _, _, opt_r, _ = encode_solution(
        Nn, Nf, opt_solution, opt_detailed_fwd, opt_replicas, t
      )
      sp_data[None]["r_bar"] = {}
      for n in range(Nn):
        for f in range(Nf):
          sp_data[None]["r_bar"][(n+1,f+1)] = int(opt_r[n,f])
    # -- solve subproblem
    sp = LSP() if opt_solution is None else LSP_fixedr()
    spr = LSPr()
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
        f"    sp: DONE ({tc['tot']}; obj = {obj['tot']}; "
        f"runtime = {sp_runtime['tot']})", 
        file = log_stream, 
        flush = True
      )
    total_runtime += sp_runtime["tot"]
    # define target operating point and initial prices
    u0 = np.ones((Nn,Nf)) * 0.8
    p = np.zeros((Nn,Nf))
    # loop over iterations
    it = 0
    stop_searching = False
    best_solution_so_far = None
    best_centralized_solution = None
    best_cost_so_far = np.inf
    spr_obj = np.inf
    best_centralized_cost = 0.0
    best_it_so_far = -1
    best_centralized_it = -1
    y = np.zeros((Nn,Nn,Nf))
    omega = deepcopy(sp_omega)
    fairness = np.zeros((Nn,Nf))
    n_accepted_queue = deque(maxlen = patience)
    while not stop_searching:
      if verbose > 0:
        print(f"    it = {it}", file = log_stream, flush = True)
      # compute residual computational capacity
      s = datetime.now()
      capacity, blackboard, residual_capacity, ell = compute_residual_capacity(
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
      bids, memory_bids = define_bids(
        omega, 
        blackboard, 
        p, 
        sp_data, 
        neighborhood, 
        sp_rho,
        auction_options, 
        latency,
        fairness,
        force_memory_bids = (
          (sp_rho >= 0).any() and
            len(n_accepted_queue) >= n_accepted_queue.maxlen and 
              all(x == n_accepted_queue[0] for x in n_accepted_queue)
        )
      )
      e = datetime.now()
      if verbose > 1:
        print(
          f"        define_bids: DONE; runtime = {(e - s).total_seconds()})", 
          file = log_stream, 
          flush = True
        )
        if verbose > 2:
          print(bids, file = log_stream, flush = True)
      total_runtime += (e - s).total_seconds()
      # sellers accept/reject bids
      rmp_omega = np.zeros((Nn,Nf))
      additional_replicas = np.zeros((Nn,Nf))
      if len(bids) > 0:
        s = datetime.now()
        auction_y, p, additional_replicas = evaluate_bids(
          bids, 
          residual_capacity, 
          data, 
          y,
          ell, 
          p, 
          capacity, 
          u0, 
          auction_options,
          sp_rho,
          sp_r,
          tentatively_start_replicas = (len(memory_bids) == 0)
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
        for n in range(Nn):
          for f in range(Nf):
            rmp_omega[n,f] = y[n,:,f].sum()
            if rmp_omega[n,f] > 0:
              fairness[n,f] += 1
        n_accepted_queue.append(rmp_omega.sum())
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
            f"        solve 'restricted problem': DONE ({spr_tc}; "
            f"obj: {spr_obj}; runtime = {spr_runtime})", 
            file = log_stream, 
            flush = True
          )
        # -- update solution
        sp_x, _, _, _, sp_r, sp_rho = spr_sol
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
      if len(memory_bids) > 0 and not (additional_replicas > 0).any():
        # tentatively start additional replicas
        s = datetime.now()
        additional_replicas, sp_rho = start_additional_replicas(
          memory_bids, sp_r, sp_data, sp_rho
        )
        sp_r += additional_replicas
        e = datetime.now()
        print(
          f"        additional replicas started: DONE "
          f"(a = {additional_replicas.tolist()}; "
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
      if spr_obj < best_cost_so_far or it == 0:
        best_cost_so_far = spr_obj
        best_solution_so_far = csol
        best_it_so_far = it
        if verbose > 0:
          print(
            f"        best solution updated; obj = {spr_obj}",
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
        omega,
        rmp_omega,
        additional_replicas,
        bids,
        memory_bids,
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
