from run_centralized_model import (
  compute_utilization, 
  init_complete_solution, 
  init_problem, 
  decode_solution, 
  encode_solution, 
  extract_solution, 
  get_current_load, 
  join_complete_solution,
  plot_history,
  save_solution
)
from utilities import load_configuration
from generate_data import update_data
from postprocessing import load_solution

from rmp import RMPAbstractModel, LRMP
from sp import SPAbstractModel, LSP, LSPr, LSP_fixedr, LSPr_fixedr
from heuristic_coordinator import GreedyCoordinator

import multiprocessing as mpp
from datetime import datetime
from collections import deque
from copy import deepcopy
from typing import Tuple
import pandas as pd
import numpy as np
import argparse
import json
import sys
import os


# Globals for each parallel worker
_sp_data = None
_solver_options = None
_solver_name = None
_sp = None


def parse_arguments() -> argparse.Namespace:
   """
   Parse input arguments
   """
   parser: argparse.ArgumentParser = argparse.ArgumentParser(
     description = "run", formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
   # Parse the arguments
   args: argparse.Namespace = parser.parse_known_args()[0]
   return args


def check_stopping_criteria(
    it: int, 
    max_iterations: int, 
    sp_omega: np.array, 
    rmp_omega: np.array, 
    pi_queue: deque, 
    dev_queue: deque,
    sw_queue: deque,
    current_sw_queue: deque,
    odev_queue: deque,
    psi: float,
    tolerance: float,
    total_runtime: float,
    time_limit: float
  ) -> Tuple[bool, str]:
  stop = False
  why_stopping = None
  if it >= max_iterations - 1:
    stop = True
    why_stopping = "max iterations reached"
  elif (abs(sp_omega - rmp_omega) <= tolerance).all():
    stop = True
    why_stopping = "SP and RMP solutions are equal"
  elif psi < 1e-12:
    stop = True
    why_stopping = "psi < tol"
  elif (dev_queue[-1] < tolerance).all():
    stop = True
    why_stopping = "dev < tol"
  elif total_runtime >= time_limit:
    stop = True
    why_stopping = f"reached time limit: {total_runtime} >= {time_limit}"
  if not stop and len(odev_queue) >= odev_queue.maxlen:
    stop = True
    why_stopping = "UB/LB diff < tol"
    for odev in odev_queue:
      if odev >= tolerance:
        stop = False
        break
  if not stop and len(current_sw_queue) >= current_sw_queue.maxlen:
    last_sw = None
    for sw in current_sw_queue:
      if last_sw is not None and sw < last_sw:
        stop = True
        why_stopping = "SW starts decreasing"
        break
      last_sw = sw
  if not stop:
    if len(pi_queue) >= pi_queue.maxlen and len(dev_queue) >= dev_queue.maxlen:
      stop = True
      # check whether all prices in pi keep increasing
      last_pi = None
      for pi in pi_queue:
        if last_pi is not None:
          for k, v in last_pi.items():
            if pi[k] < v:
              stop = False
              break
        last_pi = pi
      # check whether the deviation is constantly >= 0
      if stop:
        for dev in dev_queue:
          if (dev < 0).any():
            stop = False
            break
      if stop:
        why_stopping = "no pi/dev improvements in the last iterations"
  # update psi if it is not time for stopping
  new_psi = psi
  if not stop and len(sw_queue) == sw_queue.maxlen:
    last_sw = None
    discount_factor = 0.5
    for sw in sw_queue:
      if last_sw is not None:
        if sw < last_sw:
          discount_factor = 1
          break
      last_sw = sw
    new_psi = psi * discount_factor
    if new_psi < psi:
      sw_queue.clear()
  return stop, why_stopping, new_psi


def compute_centralized_objective(
    sp_data: dict, sp_x: np.array, sp_y: np.array, sp_z: np.array
  ) -> float:
  Nn = sp_data[None]["Nn"][None]
  Nf = sp_data[None]["Nf"][None]
  # objective function weights
  alpha = np.zeros((Nn,Nf))
  for (n,f), a in sp_data[None]["alpha"].items():
    alpha[n-1,f-1] = a
  beta = np.zeros((Nn,Nn,Nf))
  for (n1,n2,f), b in sp_data[None]["beta"].items():
    beta[n1-1,n2-1,f-1] = b
  gamma = np.zeros((Nn,Nf))
  for (n,f), g in sp_data[None]["gamma"].items():
    gamma[n-1,f-1] = g
  # value
  tot = 0.0
  for n1 in range(Nn):
    for f in range(Nf):
      load = sp_data[None]["incoming_load"][(n1+1,f+1)]
      tot += alpha[n1,f] * sp_x[n1,f] / load
      tot -= gamma[n1,f] * sp_z[n1,f] / load
      for n2 in range(Nn):
        tot += beta[n1,n2,f] * sp_y[n1,n2,f] / load
  return tot#(alpha * sp_x).sum() + (beta * sp_y).sum() - (gamma * sp_z).sum()


def compute_deviation(
    rmp_data: dict, 
    sp_x: np.array, 
    sp_omega: np.array, 
    rmp_r: np.array, 
    rmp_omega: np.array
  ) -> Tuple[np.array, list, np.array]:
  Nn = rmp_data[None]["Nn"][None]
  Nf = rmp_data[None]["Nf"][None]
  dev = np.array([sp_omega[:,f].sum() for f in range(Nf)])
  detailed_dev = np.zeros((Nn,Nf))
  Nfthr = []
  for f in range(Nf):
    n_f = 0
    for n in range(Nn):
      d = rmp_data[None]["demand"][n+1,f+1]
      u = rmp_data[None]["max_utilization"][f+1]
      n_f += (u * rmp_r[n,f] / d - sp_x[n,f])
      detailed_dev[n,f] = sp_omega[n,f] - rmp_omega[n,f]
    # deviation
    dev[f] -= n_f
    # if abs(dev[f]) < tolerance:
    #   dev[f] = 0
    Nfthr.append(n_f)
  return dev, Nfthr, detailed_dev


def compute_lower_bound(
    sp_obj: float, pi: dict, Nfthr: list, loadt: dict, Nn: int, sp_omega: np.array
  ) -> float:
  tot_workload = {
    f: sum([loadt[(n+1,f)] for n in range(Nn)]) for f in pi.keys()
  }
  highest_cost = 0
  for f, pi_f in pi.items():
    highest_cost += pi_f# * tot_workload[f]#sp_omega[:,f-1].sum()#Nfthr[f-1]#
  return (sp_obj - highest_cost)


def compute_social_welfare(
    spr: SPAbstractModel,
    data: dict, 
    agents: list, 
    solver_name: str,
    solver_options: dict,
    rmp_y: np.array, 
    rmp_omega: np.array,
    parallelism: int
  ) -> Tuple[list, float, list]:
  Nn = data[None]["Nn"][None]
  Nf = data[None]["Nf"][None]
  # solve restricted SP
  spr_data = deepcopy(data)
  spr_data[None]["y_bar"] = {
    (n1+1,n2+1,f+1): max(rmp_y[n1,n2,f], 0) \
      for n1 in range(Nn) \
        for n2 in range(Nn) \
          for f in range(Nf)
  }
  spr_data[None]["omega_bar"] = {
    (n+1,f+1): max(rmp_omega[n,f], 0) for n in range(Nn) for f in range(Nf)
  }
  # solve for all agents
  agents_sol = {}
  if parallelism != 0:
    results = []
    n_proc = mpp.cpu_count() if parallelism < 0 else parallelism
    with mpp.Pool(
        processes = n_proc,
        initializer = init_parallel_worker,
        initargs = (spr_data, solver_options, solver_name, spr),
      ) as pool:
      results = pool.map(solve_single_agent, agents)
    agents_sol = {agent: sol for agent, sol in results}
  else:
    for agent in agents:
      spr_data[None]["whoami"] = {None: agent + 1}
      spr_instance = spr.generate_instance(spr_data)
      agents_sol[agent] = spr.solve(
        spr_instance, solver_options, solver_name
      )
  # merge solutions
  spr_sol = merge_agents_solutions(
    spr_data, agents_sol
  )
  return (
    list(spr_sol[:-3]), # x, omega, r, rho
    spr_sol[-3]["tot"], # obj
    spr_sol[-2]["tot"], # termination condition
    spr_sol[-1]["tot"]  # runtime
  )


def combine_solutions(
    Nn: int, Nf: int, sp_data: dict, loadt: dict, 
    sp_x: np.array, spr_r: np.array, sp_rho: np.array, 
    rmp_x: np.array, rmp_y: np.array, rmp_z: np.array, 
    rmp_r: np.array, rmp_xi: np.array, rmp_rho: np.array
  ):
  sp_y = rmp_y
  # -- compute xi
  sp_xi = np.zeros((Nn,Nn,Nf))
  for n1 in range(Nn):
    for n2 in range(Nn):
      for f in range(Nf):
        sp_xi[n2,n1,f] = sp_y[n1,n2,f]
  # -- compute utilization
  spr_U = compute_utilization(
    sp_data, 
    {"x": sp_x, "xi": sp_xi, "r": spr_r, "obj": None}
  )
  # -- compute rejections
  sp_z = np.zeros((Nn,Nf))
  for (n,f), l in loadt.items():
    sp_z[n-1,f-1] = (
      l - sp_x[n-1,f-1] - sp_y[n-1,:,f-1].sum()
    )
  return {
    "sp": {
      "x":    sp_x,
      "y":    rmp_y,
      "z":    sp_z,
      "r":    spr_r,
      "xi":   sp_xi,
      "rho":  sp_rho,
      "U":    spr_U
    },
    "rmp": {
      "x":    rmp_x,
      "y":    rmp_y,
      "z":    rmp_z,
      "r":    rmp_r,
      "xi":   rmp_xi,
      "rho":  rmp_rho,
      "U":    spr_U
    }
  }


def decode_solutions(
    sp_data: dict, solution: dict, sp_complete_solution, rmp_complete_solution
  ):
  # -- SP
  sp_x = solution["sp"]["x"]
  sp_y = solution["sp"]["y"]
  sp_z = solution["sp"]["z"]
  sp_xi = solution["sp"]["xi"]
  sp_rho = solution["sp"]["rho"]
  sp_U = solution["sp"]["U"]
  sp_r = solution["sp"]["r"]
  sp_complete_solution = decode_solution(
    sp_x, sp_y, sp_z, sp_r, sp_xi, sp_rho, sp_U, sp_complete_solution
  )
  # -- RMP
  rmp_x = solution["rmp"]["x"]
  rmp_y = solution["rmp"]["y"]
  rmp_z = solution["rmp"]["z"]
  rmp_xi = solution["rmp"]["xi"]
  rmp_rho = solution["rmp"]["rho"]
  rmp_U = solution["rmp"]["U"]
  rmp_r = solution["rmp"]["r"]
  rmp_complete_solution = decode_solution(
    rmp_x, rmp_y, rmp_z, rmp_r, rmp_xi, rmp_rho, rmp_U, 
    rmp_complete_solution
  )
  # centralized objective
  obj = compute_centralized_objective(sp_data, sp_x, sp_y, sp_z)
  return sp_complete_solution, rmp_complete_solution, obj


def init_parallel_worker(
    sp_data: dict, solver_options: dict, solver_name: str, sp: SPAbstractModel
  ):
  """Initializer for each worker: store common data as globals."""
  global _sp_data, _solver_options, _solver_name, _sp
  # Each worker process will get its own copy (once)
  _sp_data = sp_data
  _solver_options = solver_options
  _solver_name = solver_name
  _sp = sp


def merge_agents_solutions(
    data: dict, agents_sol: dict
  ) -> Tuple[np.array, np.array, np.array, np.array, dict, dict]:
  Nn = data[None]["Nn"][None]
  Nf = data[None]["Nf"][None]
  x = np.zeros((Nn,Nf))
  omega = np.zeros((Nn,Nf))
  r = np.zeros((Nn,Nf))
  rho = np.zeros((Nn,))
  temp_data = {
    None: {
      "Nn": {None: 1}, 
      "Nf": data[None]["Nf"],
      "memory_capacity": data[None]["memory_capacity"],
      "memory_requirement": data[None]["memory_requirement"]
    }
  }
  obj_dict = {}
  tc_dict = {}
  runtime_dict = {}
  # loop over all agents
  for agent, agent_solution in agents_sol.items():
    temp_data["indices"] = [agent]
    # -- variables
    a_x, _, _, a_r, _, a_omega, a_rho, a_obj = extract_solution(
      temp_data, agent_solution
    )
    if a_x is not None:
      x[agent,:] = a_x
    if a_r is not None:
      r[agent,:] = a_r
    if a_omega is not None:
      omega[agent,:] = a_omega
    if a_rho is not None:
      rho[agent] = a_rho[0]
    # -- termination condition
    a_tc = agent_solution["termination_condition"]
    tc_dict[agent] = a_tc
    # -- objective function value
    obj_dict[agent] = a_obj
    # -- runtime
    runtime_dict[agent] = agent_solution["runtime"]
  # "total" objective function value, termination condition and runtime 
  # (NOTE: the total runtime is the average runtime among agents)
  obj_dict["tot"] = sum(list(obj_dict.values()))
  tc_dict["tot"] = "-".join(tc_dict.values())
  runtime_dict["tot"] = sum(list(runtime_dict.values())) / len(agents_sol)
  return x, omega, r, rho, obj_dict, tc_dict, runtime_dict


def solve_master_problem(
    base_instance_data: dict, 
    rmp1: RMPAbstractModel, 
    solver_name: str, 
    solver_options: dict, 
    sp_solution: Tuple
  ):
  Nn = base_instance_data[None]["Nn"][None]
  Nf = base_instance_data[None]["Nf"][None]
  # prepare data
  sp_x, sp_omega, sp_r, sp_rho = sp_solution
  rmp_data = deepcopy(base_instance_data)
  rmp_data[None]["x_bar"] = {
    (n+1,f+1): max(sp_x[n,f], 0) for n in range(Nn) for f in range(Nf)
  }
  rmp_data[None]["r_bar"] = {
    (n+1,f+1): max(sp_r[n,f], 0) for n in range(Nn) for f in range(Nf)
  }
  rmp_data[None]["omega_bar"] = {
    (n+1,f+1): max(sp_omega[n,f], 0) \
      for n in range(Nn) \
        for f in range(Nf)
  }
  # solve
  rmp_solution = {}
  if "sorting_rule" not in solver_options:
    rmp_instance = rmp1.generate_instance(rmp_data)
    rmp_solution = rmp1.solve(rmp_instance, solver_options, solver_name)
  else:
    reduced_solver_options = deepcopy(solver_options)
    GC = GreedyCoordinator()
    rmp_instance = {**rmp_data, "sp_rho": sp_rho}
    rmp_solution = GC.solve(rmp_instance, reduced_solver_options)
    _ = reduced_solver_options.pop("sorting_rule")
    # check if the greedy solution should be provided as starting point to the 
    # model
    if not reduced_solver_options.pop("heuristic_only", True):
      rmp_instance = rmp1.generate_instance(rmp_data)
      rmp_solution = rmp1.solve(
        rmp_instance, reduced_solver_options, solver_name, rmp_solution
      )
  tc = rmp_solution["termination_condition"]
  runtime = rmp_solution["runtime"]
  # extract solution
  (
    rmp_x, rmp_y, rmp_z, rmp_r, rmp_xi, rmp_omega, rmp_rho, obj
  ) = extract_solution(
    rmp_data, rmp_solution
  )
  rmp_U = compute_utilization(rmp_data, rmp_solution)
  return (
    rmp_x, rmp_y, rmp_z, rmp_r, rmp_xi, rmp_omega, rmp_rho, rmp_U, obj, tc, runtime
  )


def solve_single_agent(agent: int):
  """Function run in each worker, uses the global data initialized above."""
  # Make a local copy of sp_data to modify safely
  local_data = _sp_data.copy()
  local_data[None]["whoami"] = {None: agent + 1}
  sp_instance = _sp.generate_instance(local_data)
  result = _sp.solve(sp_instance, _solver_options, _solver_name)
  return agent, result


def solve_subproblem(
    base_instance_data: dict, 
    agents: list, 
    sp: SPAbstractModel,
    solver_name: str,
    solver_options: dict,
    parallelism: int,
    pi: dict = None,
    detailed_pi: np.array = None
  ):
  Nn = base_instance_data[None]["Nn"][None]
  Nf = base_instance_data[None]["Nf"][None]
  # update data
  sp_data = deepcopy(base_instance_data)
  if pi is not None:
    sp_data[None]["pi"] = pi
  else:
    sp_data[None]["pi"] = {f+1: 0 for f in range(Nf)}
  # solve for all agents
  agents_sol = {}
  if parallelism != 0:
    results = []
    n_proc = mpp.cpu_count() if parallelism < 0 else parallelism
    with mpp.Pool(
        processes = n_proc,
        initializer = init_parallel_worker,
        initargs = (sp_data, solver_options, solver_name, sp),
      ) as pool:
      results = pool.map(solve_single_agent, agents)
    agents_sol = {agent: sol for agent, sol in results}
  else:
    for agent in agents:
      # generate instance
      sp_data[None]["whoami"] = {None: agent + 1}
      if detailed_pi is not None:
        sp_data[None]["pi"] = {f+1: detailed_pi[agent,f] for f in range(Nf)}
      sp_instance = sp.generate_instance(sp_data)
      # solve
      agents_sol[agent] = sp.solve(
        sp_instance, solver_options, solver_name
      )
  # merge solutions
  sp_x, sp_omega, sp_r, sp_rho, obj, tc, runtime = merge_agents_solutions(
    sp_data, agents_sol
  )
  sp_solution = {
    "x": sp_x, 
    "y": np.zeros((Nn,Nn,Nf)), 
    "r": sp_r, 
    "obj": obj["tot"]
  }
  sp_U = compute_utilization(sp_data, sp_solution)
  return sp_data, sp_x, sp_omega, sp_r, sp_rho, sp_U, obj, tc, runtime


def update_neighborhood(
    original_neighborhood: dict, sp_rho: np.array, sp_omega: np.array
  ) -> dict:
  Nn, _ = sp_omega.shape
  neighborhood = deepcopy(original_neighborhood)
  for n1 in range(Nn):
    for n2 in range(Nn):
      # if a node residual capacity is zero, all incoming edges should 
      # be removed
      if n1 != n2 and sp_rho[n1] <= 0:
        neighborhood[(n2+1, n1+1)] = 0
  return neighborhood


def update_prices(
    dev: np.array, 
    detailed_dev: np.array, 
    psi: float, 
    delta_w: float, 
    old_pi: dict,
    old_detailed_pi: np.array
  ) -> Tuple[dict, np.array]:
  pi = {}
  detailed_pi = deepcopy(old_detailed_pi)
  for f in range(len(old_pi)):
    # update per-function prices
    if (dev != 0).any():
      pi[f+1] = max(
        0,
        old_pi[f+1] + psi * delta_w * dev[f] / (dev**2).sum()
      )
    else:
      pi[f+1] = old_pi[f+1]
    # update detailed prices
    for n in range(detailed_pi.shape[0]):
      if detailed_dev[n,f] != 0:
        detailed_pi[n,f] = max(
          0,
          detailed_pi[n,f] + (
            psi * delta_w * detailed_dev[n,f] / (detailed_dev**2).sum()
          )
        )
  return pi, detailed_pi


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
  max_steps = config["max_steps"]
  min_run_time = config.get("min_run_time", 0)
  max_run_time = config.get("max_run_time", max_steps)
  run_time_step = config.get("run_time_step", 1)
  checkpoint_interval = config["checkpoint_interval"]
  # -- solver name and options
  solver_name = config["solver_name"]
  solver_options = config.get("solver_options", {})
  general_solver_options = solver_options.get("general", {})
  coordinator_options = {k: v for k,v in general_solver_options.items()}
  if "coordinator" in solver_options:
    for k, v in solver_options["coordinator"].items():
      coordinator_options[k] = v
  time_limit = general_solver_options.get("TimeLimit", np.inf)
  start_from_last_pi = solver_options.get("start_from_last_pi", False)
  use_detailed_pi = solver_options.get("use_detailed_pi", False)
  # -- maximum number of iterations
  max_iterations = config["max_iterations"]
  plot_interval = config.get("plot_interval", max_iterations)
  patience = config["patience"]
  sw_patience = config.get("sw_patience", max_iterations)
  verbose = config.get("verbose", 0)
  tolerance = config.get("tolerance", 1e-3)
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
  # load globally-optimal solution (if provided)
  opt_solution, opt_replicas, opt_detailed_fwd = None, None, None
  if "opt_solution_folder" in config:
    opt_solution, opt_replicas, opt_detailed_fwd = load_solution(
      config["opt_solution_folder"], "LoadManagementModel"
    )
  # loop over time
  final_pi = None
  final_detailed_pi = np.zeros((Nn,Nf))
  sp_complete_solution = init_complete_solution()
  rmp_complete_solution = init_complete_solution()
  spc_complete_solution = init_complete_solution()
  rmpc_complete_solution = init_complete_solution()
  obj_dict = {
    "LSP": {it: [] for it in range(max_iterations)}, 
    "LSPr": {it: [] for it in range(max_iterations)}, 
    "LSPr_final": [], 
    "LRMP": {it: [] for it in range(max_iterations)}
  }
  tc_dict = {
    "LSP": {it: [] for it in range(max_iterations)}, 
    "LSPr": [], 
    "LRMP": {it: [] for it in range(max_iterations)}
  }
  ub = (
    max_run_time + run_time_step
  ) if max_run_time == min_run_time else max_run_time
  for t in range(min_run_time, ub, run_time_step):
    if verbose > 0:
      print(f"t = {t}", file = log_stream, flush = True)
    social_welfare = np.inf
    lower_bound = np.inf
    # get current load and generate data
    loadt = get_current_load(input_requests_traces, agents, t)
    data = update_data(base_instance_data, {"incoming_load": loadt})
    # loop over SP/RMP
    sp = LSP() if opt_solution is None else LSP_fixedr()
    spr = LSPr()# if opt_solution is None else LSPr_fixedr()
    rmp = LRMP()
    sp_data = deepcopy(data)
    rmp_data = deepcopy(data)
    rmp_y = None
    pi = None if (not start_from_last_pi or final_pi is None) else final_pi
    detailed_pi = np.zeros(
      (Nn,Nf)
    ) if not start_from_last_pi else final_detailed_pi
    it = 0
    stop_searching = False
    psi = 2
    pi_queue = deque(maxlen = patience)
    dev_queue = deque(maxlen = patience)
    sw_queue = deque(maxlen = patience)
    odev_queue = deque(maxlen = patience)
    current_sw_queue = deque(maxlen = sw_patience)
    best_solution_so_far = None
    best_centralized_solution = None
    best_cost_so_far = np.inf
    best_centralized_cost = 0.0
    best_it_so_far = -1
    best_centralized_it = -1
    total_runtime = 0
    ss = datetime.now()
    while not stop_searching:
      if verbose > 0:
        print(f"    it = {it} (psi = {psi})", file = log_stream, flush = True)
      # if t == 6:
      #   print("here")
      # extract optimal solution (if provided)
      if opt_solution is not None:
        opt_x, _, _, _, _ = encode_solution(
          Nn, Nf, opt_solution, opt_detailed_fwd, opt_replicas, t
        )
        opt_r_for_x = np.zeros((Nn,Nf))
        sp_data[None]["r_bar"] = {}
        for n in range(Nn):
          for f in range(Nf):
            opt_r_for_x[n,f] = sp_data[None][
              "demand"
            ][(n+1,f+1)] * opt_x[n,f] / sp_data[None]["max_utilization"][f+1]
            if np.floor(opt_r_for_x[n,f]) > 0 and (
                (opt_r_for_x[n,f] / np.floor(opt_r_for_x[n,f]) - 1) > 1e-6
              ):
              sp_data[None]["r_bar"][(n+1,f+1)] = int(
                np.ceil(opt_r_for_x[n,f])
              )
            else:
              if int(np.floor(opt_r_for_x[n,f])) == 0 and (
                  opt_r_for_x[n,f] > 1e-6
                ):
                sp_data[None]["r_bar"][(n+1,f+1)] = int(
                  np.ceil(opt_r_for_x[n,f])
                )
              else:
                sp_data[None]["r_bar"][(n+1,f+1)] = int(opt_r_for_x[n,f])
      s = datetime.now()
      # solve sub-problem
      (
        sp_data, sp_x, sp_omega, sp_r, sp_rho, sp_U, obj, tc, sp_runtime
      ) = solve_subproblem(
        sp_data, 
        agents, 
        sp, 
        solver_name, 
        general_solver_options, 
        parallelism,
        pi = pi,
        detailed_pi = detailed_pi if use_detailed_pi else None
      )
      e = datetime.now()
      obj_dict["LSP"][it].append(obj["tot"])
      tc_dict["LSP"][it].append(tc["tot"])
      if verbose > 1:
        print(
          f"        sp: DONE ",
          f"({tc['tot']}; obj = {obj['tot']}; runtime = {sp_runtime['tot']})", 
          file = log_stream, 
          flush = True
        )
      total_runtime += sp_runtime["tot"]
      # update neighborhood given the nodes availability
      if solver_options.get("update_neighborhood", False):
        rmp_data[None]["neighborhood"] = update_neighborhood(
          data[None]["neighborhood"], sp_rho, sp_omega
        )
      # solve master problem
      (
        rmp_x, 
        rmp_y, 
        rmp_z, 
        rmp_r, 
        rmp_xi, 
        rmp_omega, 
        rmp_rho, 
        rmp_U, 
        obj, 
        tc, 
        runtime
      ) = solve_master_problem(
        rmp_data, 
        rmp, 
        solver_name, 
        coordinator_options, 
        (sp_x, sp_omega, sp_r, sp_rho)
      )
      obj_dict["LRMP"][it].append(obj)
      tc_dict["LRMP"][it].append(tc)
      if verbose > 1:
        print(
          f"        rmp: DONE ({tc}; obj = {obj}; runtime = {runtime})", 
          file = log_stream, 
          flush = True
        )
      total_runtime += runtime
      # compute deviation
      s = datetime.now()
      dev, Nfthr, detailed_dev = compute_deviation(
        rmp_data, sp_x, sp_omega, rmp_r, rmp_omega
      )
      e = datetime.now()
      dev_queue.append(dev)
      if verbose > 1:
        print(
          f"        compute_deviation: DONE (dev = {dev}; "
          f"Nf = {Nfthr}; omega = {sp_omega.sum(axis=0)}; ", 
          f"ni = {rmp_omega.sum(axis=0)}; runtime = {(e-s).total_seconds()})", 
          file = log_stream, 
          flush = True
        )
      # compute lower bound
      s = datetime.now()
      lb = compute_lower_bound(
        obj_dict["LSP"][it][-1], 
        sp_data[None]["pi"], 
        Nfthr, 
        loadt, 
        Nn, 
        sp_omega
      )
      lower_bound = min(lower_bound, lb)
      e = datetime.now()
      if verbose > 1:
        print(
          f"        compute_lower_bound: DONE (current: {lb};"
          f" bound: {lower_bound}; runtime = {(e-s).total_seconds()})", 
          file = log_stream, 
          flush = True
        )
      # solve "restricted problem"
      spr_sol, spr_obj, spr_tc, spr_runtime = compute_social_welfare(
        spr, 
        sp_data, 
        agents, 
        solver_name, 
        general_solver_options, 
        rmp_y, 
        rmp_omega,
        parallelism
      )
      total_runtime += spr_runtime
      # -- rejection cost
      rej_cost = 0
      for n in range(Nn):
        for f in range(Nf):
          diff = sp_omega[n,f] - rmp_omega[n,f]
          if diff > 0:
            rej_cost += diff * sp_data[None]["gamma"][(n+1,f+1)]
      spr_obj += rej_cost
      # update social welfare
      social_welfare = min(social_welfare, spr_obj)
      sw_queue.append(social_welfare)
      current_sw_queue.append(spr_obj)
      obj_dict["LSPr"][it].append(spr_obj)
      if verbose > 1:
        print(
          f"        compute_social_welfare: DONE ({spr_tc}; current: {spr_obj}"
          f"; sw: {social_welfare}; runtime = {spr_runtime})", 
          file = log_stream, 
          flush = True
        )
      # compute price deviation
      odev = abs((spr_obj - obj_dict["LSP"][it][-1]) / obj_dict["LSP"][it][-1])
      x_cost = 0
      for n in range(sp_x.shape[0]):
        x_cost_n = 0
        for f in range(sp_x.shape[1]):
          x_cost_n += (sp_x[n,f] * sp_data[None]["alpha"][(n+1,f+1)])
        x_cost += x_cost_n
      odev_queue.append(odev)
      # merge solutions and compute the centralized objective value
      csol = combine_solutions(
        Nn, Nf, sp_data, loadt, 
        sp_x, spr_sol[2], sp_rho,
        rmp_x, rmp_y, rmp_z, rmp_r, rmp_xi, rmp_rho
      )
      cobj = compute_centralized_objective(
        sp_data, csol["sp"]["x"], csol["sp"]["y"], csol["sp"]["z"]
      )
      # update best solution so far
      if spr_obj < best_cost_so_far:
        best_cost_so_far = spr_obj
        best_solution_so_far = csol
        best_it_so_far = it
        final_pi = deepcopy(pi)
        final_detailed_pi = deepcopy(detailed_pi)
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
      # check that the deviation is >= 0 (otherwise, no iterations needed)
      if not (dev < 0).all():
        if social_welfare < lower_bound and (
            lower_bound - social_welfare
          ) >= 1e-6:
          return
        # update prices
        pi, detailed_pi = update_prices(
          dev, 
          detailed_dev, 
          psi, 
          social_welfare - lower_bound, 
          sp_data[None]["pi"],
          detailed_pi
        )
        pi_queue.append(pi)
        if verbose > 1:
          print(
            "        update_prices: DONE "
            f"({pi if not use_detailed_pi else detailed_pi.flatten()})", 
            file = log_stream, 
            flush = True
          )
        # check stopping criterion
        s = datetime.now()
        stop_searching, why_stop_searching, psi = check_stopping_criteria(
          it,
          max_iterations,
          sp_omega,
          rmp_omega,
          pi_queue,
          dev_queue,
          sw_queue,
          current_sw_queue,
          odev_queue,
          psi,
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
            f"--> stop? {stop_searching}", 
            file = log_stream, 
            flush = True
          )
      else:
        stop_searching = True
        why_stop_searching = "dev < 0"
      # move to the next iteration OR update the complete solution
      if not stop_searching:
        it += 1
      else:
        # save SP/RMP solutions
        sp_complete_solution, rmp_complete_solution, objf = decode_solutions(
          sp_data, 
          best_solution_so_far, 
          sp_complete_solution, 
          rmp_complete_solution
        )
        spc_complete_solution, rmpc_complete_solution, _ = decode_solutions(
          sp_data, 
          best_centralized_solution, 
          spc_complete_solution, 
          rmpc_complete_solution
        )
        obj_dict["LSPr_final"].append(objf)
        tc_dict["LSPr"].append(
          f"{why_stop_searching} "
          f"(it: {it}; obj. deviation: {odev}; best it: {best_it_so_far}; "
          f"best centralized it: {best_centralized_it}; "
          f"total runtime: {total_runtime})"
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
  rmp_solution, rmp_offloaded, rmp_detailed_fwd_solution = join_complete_solution(
    rmp_complete_solution
  )
  spc_solution, spc_offloaded, spc_detailed_fwd_solution = join_complete_solution(
    spc_complete_solution
  )
  rmpc_solution, rmpc_offloaded, rmpc_detailed_fwd_solution = join_complete_solution(
    rmpc_complete_solution
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
    plot_history(
      input_requests_traces, 
      min_run_time,
      max_run_time,
      run_time_step,
      rmp_solution, 
      rmp_complete_solution["utilization"], 
      rmp_complete_solution["replicas"], 
      rmp_offloaded,
      obj_dict["LRMP"][max_iterations-1],
      os.path.join(solution_folder, "rmp.png")
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
    rmp_solution,
    rmp_offloaded,
    rmp_complete_solution,
    rmp_detailed_fwd_solution,
    "LRMP",
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
  save_solution(
    rmpc_solution,
    rmpc_offloaded,
    rmpc_complete_solution,
    rmpc_detailed_fwd_solution,
    "LRMPc",
    solution_folder
  )
  # save objective function values
  pd.DataFrame(obj_dict["LSPr_final"], columns = ["SP/coord"]).to_csv(
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
  # load configuration file
  config = load_configuration(config_file)
  # run
  run(config, parallelism)
