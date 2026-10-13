from run_centralized_model import (
  compute_utilization, 
  extract_solution, 
  join_complete_solution
)
from utils.centralized import (
  validate_centralized_solution, 
  encode_solution, 
)
from utils.faasmacro import (
  compute_centralized_objective, 
  relative_objective_gap
)
from utils.common import load_configuration
from postprocessing import plot_history

from models.local_sp import solve_agent_problem
from models.rmp import RMPAbstractModel, LRMP
from models.sp import (
  SPAbstractModel, 
  LSP, 
  LSPr, 
  LSP_fixedr, 
  LSP_v0,
  LSPr_v0,
  LSP_fixedr_v0 
)
from heuristic_coordinator import GreedyCoordinator
from utils import run_frame, run_log, step_time

import multiprocessing as mpp
from contextlib import contextmanager
from contextvars import ContextVar
from collections import deque
from copy import deepcopy
from functools import partial, wraps
from typing import Tuple
import numpy as np
import argparse
import os


log = run_log.logger(__name__)

# Globals for each parallel worker
_sp_data = None
_solver_options = None
_solver_name = None
_sp = None
_parallel_session: ContextVar[dict | None] = ContextVar('parallel_sp_session', default=None)


@contextmanager
def parallel_solver_session():
  """Own one lazy process pool; nested runners reuse it until the outer run ends."""
  current = _parallel_session.get()
  if current is not None:
    yield current
    return
  state: dict = {'pool': None, 'workers': 0}
  token = _parallel_session.set(state)
  try:
    yield state
  except BaseException:
    if state['pool'] is not None:
      state['pool'].terminate()
    raise
  else:
    if state['pool'] is not None:
      state['pool'].close()
  finally:
    if state['pool'] is not None:
      state['pool'].join()
    _parallel_session.reset(token)


def parallel_solver_run(runner):
  """Keep local-solver workers warm across all batches and timesteps of a run."""
  @wraps(runner)
  def wrapped(*args, **kwargs):
    with parallel_solver_session():
      return runner(*args, **kwargs)
  return wrapped


def _available_cpu_count():
  try:
    return max(1, len(os.sched_getaffinity(0)))
  except (AttributeError, OSError):
    return os.cpu_count() or 1


def _solve_parallel_batch(task):
  data, options, solver, model, agents, prices = task
  # Each task carries the current snapshot, even if this worker ran another model.
  init_parallel_worker(data, options, solver, model)
  out = []
  for agent in agents:
    started = step_time.now()  # the node's own computation, without transfer
    agent_id, result = solve_single_agent(agent, prices)
    out.append((agent_id, result, step_time.now() - started))
  return out


def _solve_agents_parallel(data, agents, model, solver, options, parallelism, prices=None):
  agents = list(agents)
  if not agents:
    return {}
  state = _parallel_session.get()
  if state is None:
    with parallel_solver_session():
      return _solve_agents_parallel(data, agents, model, solver, options, parallelism, prices)
  if state['pool'] is None:
    available = _available_cpu_count() if parallelism < 0 else parallelism
    state['workers'] = max(1, min(available, len(agents)))
    state['pool'] = mpp.Pool(processes=state['workers'])
  count = min(state['workers'], len(agents))
  # One fresh snapshot per chunk, rather than serializing the network per node.
  tasks = [(data, options, solver, model, agents[i::count], prices) for i in range(count)]
  with step_time.offloaded(state['workers']):
    results = state['pool'].map(_solve_parallel_batch, tasks, chunksize=1)
    for batch in results:
      for agent, _, seconds in batch:
        step_time.record(agent, seconds)
  by_agent = {agent: result for batch in results for agent, result, _ in batch}
  return {agent: by_agent[agent] for agent in agents}


def parse_arguments() -> argparse.Namespace:
  """
  Parse input arguments
  """
  parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description = "Run FaaS-MACrO", 
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
    default = -1
  )
  parser.add_argument(
    "--disable_plotting",
    help = "True to disable automatic plot generation for each experiment",
    default = False,
    action = "store_true"
  )
  parser.add_argument(
    "--v0",
    help = "Use the v0 FaaS-MACrO iteration variant",
    default = False,
    action = "store_true"
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
  #if not stop and len(current_sw_queue) >= current_sw_queue.maxlen:
  #  last_sw = None
  #  for sw in current_sw_queue:
  #    if last_sw is not None and sw < last_sw:
  #      stop = True
  #      why_stopping = "SW starts decreasing"
  #      break
  #    last_sw = sw
  if not stop and len(sw_queue) >= sw_queue.maxlen:
    stop = True
    why_stopping = "SW deviation < tol"
    last_sw = None
    for sw in sw_queue:
      if last_sw is not None and abs(sw - last_sw) >= tolerance:
        stop = False
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
  if not stop and len(current_sw_queue) == current_sw_queue.maxlen:
    last_sw = None
    discount_factor = 0.5
    for sw in current_sw_queue:
      if last_sw is not None:
        if sw < last_sw:
          discount_factor = 1
          break
      last_sw = sw
    new_psi = psi * discount_factor
    if new_psi < psi:
      current_sw_queue.clear()
  return stop, why_stopping, new_psi


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
    parallelism: int,
    sp_x: np.array = None
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
  if sp_x is not None:
    spr_data[None]["x_bar"] = {
      (n+1,f+1): max(sp_x[n,f], 0) for n in range(Nn) for f in range(Nf)
    }
  # solve for all agents
  agents_sol = {}
  with step_time.parallel_phase():
    if parallelism != 0:
      agents_sol = _solve_agents_parallel(
        spr_data, agents, spr, solver_name, solver_options, parallelism,
      )
    else:
      for agent in agents:
        with step_time.task(agent):
          spr_data[None]["whoami"] = {None: agent + 1}
          agents_sol[agent] = solve_agent_problem(
            spr, spr_data, solver_options, solver_name
          )
  # merge solutions
  spr_sol = merge_agents_solutions(
    spr_data, agents_sol
  )
  return (
    list(spr_sol[:-3]), # x, y, z, omega, r, rho
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
  sp_y = rmp_y.copy()
  # -- compute xi
  sp_xi = np.zeros((Nn,Nn,Nf))
  for n1 in range(Nn):
    for n2 in range(Nn):
      for f in range(Nf):
        sp_xi[n2,n1,f] = sp_y[n1,n2,f]
  # -- tighten replicas to the centralized replica equilibrium
  # r = ceil(utilization / max_utilization) for the realized served load
  # (local processing + received offload). A heuristic may greedily over-acquire
  # replicas to be able to receive offload that neighbour/no-ping-pong
  # restrictions then keep from arriving; r does not enter the welfare objective,
  # so tightening it leaves the objective unchanged while making the solution
  # centrally feasible (utilization_equilibrium / utilization_equilibrium2).
  demand = np.array([
    [sp_data[None]["demand"][(n + 1, f + 1)] for f in range(Nf)]
    for n in range(Nn)
  ])
  max_utilization = np.array([
    sp_data[None]["max_utilization"][f + 1] for f in range(Nf)
  ])
  served = sp_x + sp_y.sum(axis = 0)
  with np.errstate(divide = "ignore", invalid = "ignore"):
    spr_r = np.ceil(demand * served / max_utilization - 1e-9).astype(int)
  spr_r = np.maximum(spr_r, 0)
  # -- compute utilization
  # These are total replicas, even when sp_data contains fixed r_bar inputs.
  spr_U = np.divide(
    demand * served, spr_r, out=np.zeros_like(served, dtype=float), where=spr_r > 0,
  )
  memory = np.array([sp_data[None]["memory_requirement"][f + 1] for f in range(Nf)])
  capacity = np.array([sp_data[None]["memory_capacity"][n + 1] for n in range(Nn)])
  sp_rho = capacity - spr_r @ memory
  rmp_U = None
  if rmp_x is not None and rmp_r is not None:
    rmp_U = np.divide(
      demand * (rmp_x + sp_y.sum(axis=0)), rmp_r,
      out=np.zeros_like(served, dtype=float), where=rmp_r > 0,
    )
  # -- compute rejections
  sp_z = np.zeros((Nn,Nf))
  for (n,f), l in loadt.items():
    sp_z[n-1,f-1] = (
      l - sp_x[n-1,f-1] - sp_y[n-1,:,f-1].sum()
    )
  solution = {
    "sp": {
      "x":    sp_x,
      "y":    sp_y,
      "z":    sp_z,
      "r":    spr_r,
      "xi":   sp_xi,
      "rho":  sp_rho,
      "U":    spr_U
    },
    "rmp": {
      "x":    rmp_x,
      "y":    sp_y,
      "z":    rmp_z,
      "r":    rmp_r,
      "xi":   rmp_xi,
      "rho":  rmp_rho,
      "U":    rmp_U
    }
  }
  validate_centralized_solution(
    solution["sp"]["x"], solution["sp"]["y"], solution["sp"]["z"],
    solution["sp"]["r"], sp_data,
  )
  return solution


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
    data: dict, agents_sol: dict, approx_tol: float = 1e-6
  ) -> Tuple[
    np.array, np.array, np.array, np.array, np.array, np.array, dict, dict
  ]:
  Nn = data[None]["Nn"][None]
  Nf = data[None]["Nf"][None]
  x = np.zeros((Nn,Nf))
  y = np.zeros((Nn,Nn,Nf))
  z = np.zeros((Nn,Nf))
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
    a_x, a_y, a_z, a_r, _, a_omega, a_rho, a_obj = extract_solution(
      temp_data, agent_solution, approx_tol = approx_tol
    )
    if a_x is not None:
      x[agent,:] = a_x
    if a_y is not None:
      y[agent,:,:] = a_y
    if a_z is not None:
      z[agent,:] = a_z
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
  return x, y, z, omega, r, rho, obj_dict, tc_dict, runtime_dict


def prepare_master_data(base_instance_data: dict, sp_solution: Tuple) -> dict:
  Nn = base_instance_data[None]["Nn"][None]
  Nf = base_instance_data[None]["Nf"][None]
  sp_x, sp_y, sp_z, sp_omega, sp_r, _ = sp_solution
  rmp_data = deepcopy(base_instance_data)
  rmp_data[None]["x_bar"] = {
    (n+1,f+1): max(sp_x[n,f], 0) for n in range(Nn) for f in range(Nf)
  }
  rmp_data[None]["d_bar"] = {
    (n1+1,n2+1,f+1): max(sp_y[n1,n2,f], 0) for n1 in range(Nn) \
                                            for n2 in range(Nn) \
                                              for f in range(Nf)
  }
  rmp_data[None]["z_bar"] = {
    (n+1,f+1): max(sp_z[n,f], 0) for n in range(Nn) for f in range(Nf)
  }
  rmp_data[None]["r_bar"] = {
    (n+1,f+1): max(sp_r[n,f], 0) for n in range(Nn) for f in range(Nf)
  }
  rmp_data[None]["omega_bar"] = {
    (n+1,f+1): max(sp_omega[n,f], 0) \
      for n in range(Nn) \
        for f in range(Nf)
  }
  return rmp_data


def solve_master_problem(
    base_instance_data: dict, 
    rmp1: RMPAbstractModel, 
    solver_name: str, 
    solver_options: dict, 
    sp_solution: Tuple
  ):
  # prepare data
  rmp_data = prepare_master_data(base_instance_data, sp_solution)
  # solve
  rmp_solution = {}
  if "sorting_rule" not in solver_options:
    rmp_instance = rmp1.generate_instance(rmp_data)
    rmp_solution = rmp1.solve(rmp_instance, solver_options, solver_name)
  else:
    reduced_solver_options = deepcopy(solver_options)
    GC = GreedyCoordinator()
    rmp_instance = {**rmp_data, "sp_rho": sp_solution[-1]}
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
    rmp_data, rmp_solution, approx_tol = solver_options.get(
      "FeasibilityTol", 1e-6
    )
  )
  rmp_U = compute_utilization(rmp_data, rmp_solution)
  return (
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
  )


def solve_single_agent(agent: int, detailed_pi=None):
  """Function run in each worker, uses the global data initialized above."""
  # Make a local copy of sp_data to modify safely
  local_data = _sp_data.copy()
  local_data[None] = _sp_data[None].copy()
  local_data[None]["whoami"] = {None: agent + 1}
  if detailed_pi is not None:
    local_data[None]["pi"] = {
      f+1: price for f, price in enumerate(detailed_pi[agent])
    }
  result = solve_agent_problem(_sp, local_data, _solver_options, _solver_name)
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
  Nf = base_instance_data[None]["Nf"][None]
  # update data
  sp_data = deepcopy(base_instance_data)
  if pi is not None:
    sp_data[None]["pi"] = pi
  else:
    sp_data[None]["pi"] = {f+1: 0 for f in range(Nf)}
  # solve for all agents
  agents_sol = {}
  with step_time.parallel_phase():
    if parallelism != 0:
      agents_sol = _solve_agents_parallel(
        sp_data, agents, sp, solver_name, solver_options, parallelism, detailed_pi,
      )
    else:
      for agent in agents:
        with step_time.task(agent):
          agent_data = sp_data.copy()
          agent_data[None] = sp_data[None].copy()
          agent_data[None]["whoami"] = {None: agent + 1}
          if detailed_pi is not None:
            agent_data[None]["pi"] = {f+1: detailed_pi[agent,f] for f in range(Nf)}
          agents_sol[agent] = solve_agent_problem(
            sp, agent_data, solver_options, solver_name
          )
  # merge solutions
  (
    sp_x, 
    sp_y, 
    sp_z, 
    sp_omega, 
    sp_r, 
    sp_rho, 
    obj, 
    tc, 
    runtime
  ) = merge_agents_solutions(
    sp_data, agents_sol, approx_tol=solver_options.get("FeasibilityTol", 1e-6)
  )
  sp_solution = {
    "x": sp_x, 
    "y": sp_y, 
    "r": sp_r, 
    "obj": obj["tot"]
  }
  sp_U = compute_utilization(sp_data, sp_solution)
  return (
    sp_data, sp_x, sp_y, sp_z, sp_omega, sp_r, sp_rho, sp_U, obj, tc, runtime
  )


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


OUTPUTS = {
  "LSP": run_frame.Output(checkpoint=True, plot_filename="sp.png"),
  "LRMP": run_frame.Output(checkpoint=False, check=False),
  "LSPc": run_frame.Output(checkpoint=True),
  "LRMPc": run_frame.Output(checkpoint=False, check=False),
}


class _BelowLowerBound(Exception):
  """The social welfare fell below the Lagrangian lower bound: the run stops without outputs."""


@parallel_solver_run
def run(
    config: dict, 
    parallelism: int,
    log_on_file: bool = False, 
    disable_plotting: bool = False,
    v0: bool = False
  ):
  # across steps: prices to restart from (start_from_last_pi), the LRMP objective at the
  # last iteration (rmp.png) and the run context for that plot
  run_state = {"pi": None, "detailed_pi": None, "rmp_objectives": [], "ctx": None}
  try:
    folder = run_frame.run(
      config, parallelism, log_on_file=log_on_file, disable_plotting=disable_plotting,
      method="FaaS-MACrO", outputs=OUTPUTS,
      solve_step=partial(_solve_step, v0=v0, run_state=run_state),
    )
  except _BelowLowerBound:
    return None
  ctx = run_state["ctx"]
  if not disable_plotting and ctx.Nf <= 10 and ctx.Nn <= 10:
    rmp_solution, rmp_offloaded, _ = join_complete_solution(ctx.complete["LRMP"])
    plot_history(
      ctx.traces, config.get("min_run_time", 0), config.get("max_run_time", config["max_steps"]),
      config.get("run_time_step", 1), rmp_solution, ctx.complete["LRMP"]["utilization"],
      ctx.complete["LRMP"]["replicas"], rmp_offloaded, run_state["rmp_objectives"],
      os.path.join(folder, "rmp.png"),
    )
  return folder


def _solve_step(ctx, t: int, data: dict, *, v0: bool, run_state: dict) -> run_frame.StepResult:
  Nn, Nf = ctx.Nn, ctx.Nf
  run_state["ctx"] = ctx
  solver_options = ctx.solver_options
  coordinator_options = {k: v for k,v in ctx.general_options.items()}
  if "coordinator" in solver_options:
    for k, v in solver_options["coordinator"].items():
      coordinator_options[k] = v
  start_from_last_pi = solver_options.get("start_from_last_pi", False)
  use_detailed_pi = solver_options.get("use_detailed_pi", False)
  max_iterations = ctx.max_iterations
  patience = ctx.patience
  tolerance = ctx.config.get("tolerance", 1e-3)
  started_at = step_time.now()
  loadt = data[None]["incoming_load"]
  social_welfare = np.inf
  lower_bound = np.inf
  fixed = ctx.reference is not None
  # loop over SP/RMP
  sp = None
  if v0:
    sp = LSP_fixedr_v0() if fixed else LSP_v0()
  else:
    sp = LSP_fixedr() if fixed else LSP()
  spr = LSPr_v0() if v0 else LSPr()
  rmp = LRMP()
  sp_data = deepcopy(data)
  rmp_data = deepcopy(data)
  rmp_y = None
  pi = None if (not start_from_last_pi or run_state["pi"] is None) else run_state["pi"]
  detailed_pi = np.zeros(
    (Nn,Nf)
  ) if not start_from_last_pi or run_state["detailed_pi"] is None else run_state["detailed_pi"]
  it = 0
  stop_searching = False
  psi = 2
  pi_queue = deque(maxlen = patience)
  dev_queue = deque(maxlen = patience)
  sw_queue = deque(maxlen = patience)
  odev_queue = deque(maxlen = patience)
  current_sw_queue = deque(maxlen = patience)
  best_solution_so_far = None
  best_centralized_solution = None
  best_cost_so_far = np.inf
  best_centralized_cost = -np.inf
  best_it_so_far = -1
  best_centralized_it = -1
  while not stop_searching:
    with step_time.untimed():
      log.info("    it = %s (psi = %s)", it, psi)
    # oracle replicas from a reference solution: experiment input, not algorithm work
    with step_time.untimed():
      if fixed:
        opt_solution, opt_replicas, opt_detailed_fwd = ctx.reference
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
    # solve sub-problem
    (
      sp_data, 
      sp_x, 
      sp_y, 
      sp_z, 
      sp_omega, 
      sp_r, 
      sp_rho, 
      sp_U, 
      obj, 
      tc, 
      sp_runtime
    ) = solve_subproblem(
      sp_data, 
      ctx.agents, 
      sp, 
      ctx.solver_name, 
      ctx.general_options, 
      ctx.parallelism,
      pi = pi,
      detailed_pi = detailed_pi if use_detailed_pi else None
    )
    sp_obj = obj["tot"]
    with step_time.untimed():
      log.debug("        sp: DONE  (%s; obj = %s; runtime = %s)", tc["tot"], obj["tot"], sp_runtime["tot"])
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
      ctx.solver_name, 
      coordinator_options, 
      (sp_x, sp_y, sp_z, sp_omega, sp_r, sp_rho)
    )
    if it == max_iterations - 1:
      run_state["rmp_objectives"].append(obj)
    with step_time.untimed():
      log.debug("        rmp: DONE (%s; obj = %s; runtime = %s)", tc, obj, runtime)
    # compute deviation
    s = step_time.now()
    dev, Nfthr, detailed_dev = compute_deviation(
      rmp_data, sp_x, sp_omega, rmp_r, rmp_omega
    )
    e = step_time.now()
    dev_queue.append(dev)
    with step_time.untimed():
      log.debug(
        "        compute_deviation: DONE (dev = %s; Nf = %s; omega = %s;  ni = %s; runtime = %s)",
        dev, Nfthr, sp_omega.sum(axis=0), rmp_omega.sum(axis=0), e - s,
      )
    # compute lower bound
    s = step_time.now()
    lb = compute_lower_bound(
      sp_obj, 
      sp_data[None]["pi"], 
      Nfthr, 
      loadt, 
      Nn, 
      sp_omega
    )
    lower_bound = min(lower_bound, lb)
    e = step_time.now()
    with step_time.untimed():
      log.debug("        compute_lower_bound: DONE (current: %s; bound: %s; runtime = %s)",
                lb, lower_bound, e - s)
    # solve "restricted problem"
    spr_sol, spr_obj, spr_tc, spr_runtime = compute_social_welfare(
      spr, 
      sp_data, 
      ctx.agents, 
      ctx.solver_name, 
      ctx.general_options, 
      rmp_y, 
      rmp_omega,
      ctx.parallelism
    )
    # update social welfare
    social_welfare = min(social_welfare, spr_obj)
    current_sw_queue.append(social_welfare)
    with step_time.untimed():
      log.debug("        compute_social_welfare: DONE (%s; current: %s; sw: %s; runtime = %s)",
                spr_tc, spr_obj, social_welfare, spr_runtime)
    # compute price deviation
    odev = relative_objective_gap(spr_obj, sp_obj)
    odev_queue.append(odev)
    # merge solutions and compute the centralized objective value
    spr_x, _, _, _, spr_r, spr_rho = spr_sol
    with step_time.untimed():
      csol = combine_solutions(
        Nn, Nf, sp_data, loadt, 
        spr_x, spr_r, spr_rho,
        rmp_x, rmp_y, rmp_z, rmp_r, rmp_xi, rmp_rho
      )
      cobj = compute_centralized_objective(
        sp_data, csol["sp"]["x"], csol["sp"]["y"], csol["sp"]["z"]
      )
    # update best solution so far
    if spr_obj < best_cost_so_far:
      sw_queue.append(social_welfare)
      best_cost_so_far = spr_obj
      best_solution_so_far = csol
      best_it_so_far = it
      run_state["pi"] = deepcopy(pi)
      run_state["detailed_pi"] = deepcopy(detailed_pi)
      with step_time.untimed():
        log.info("        best solution updated; obj = %s", cobj)
    if cobj > best_centralized_cost:
      best_centralized_cost = cobj
      best_centralized_solution = csol
      best_centralized_it = it
      with step_time.untimed():
        log.info("        best centralized solution updated; obj = %s", cobj)
    total_runtime = step_time.current_modelled()
    # check that the deviation is >= 0 (otherwise, no iterations needed)
    if not (dev < 0).all():
      if social_welfare < lower_bound and (
          lower_bound - social_welfare
        ) >= 1e-6:
        raise _BelowLowerBound
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
      with step_time.untimed():
        log.debug("        update_prices: DONE (%s)", pi if not use_detailed_pi else detailed_pi.flatten())
      # check stopping criterion
      s = step_time.now()
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
        ctx.time_limit
      )
      e = step_time.now()
      with step_time.untimed():
        log.debug(
          "        check_stopping_criteria: DONE (runtime = %s; total runtime = %s; wallclock: %s) --> stop? %s (%s)",
          e - s, total_runtime, step_time.now() - started_at, stop_searching, why_stop_searching,
        )
    else:
      stop_searching = True
      why_stop_searching = "dev < 0"
      with step_time.untimed():
        log.info(
          "        check_stopping_criteria: DONE (total runtime = %s; wallclock: %s) --> stop? %s (%s)",
          total_runtime, step_time.now() - started_at, stop_searching, why_stop_searching,
        )
    # move to the next iteration
    if not stop_searching:
      it += 1
  with step_time.untimed():
    best = best_solution_so_far["sp"]
    objective = compute_centralized_objective(sp_data, best["x"], best["y"], best["z"])
  return run_frame.StepResult(
    solutions={"LSP": best, "LRMP": best_solution_so_far["rmp"],
               "LSPc": best_centralized_solution["sp"], "LRMPc": best_centralized_solution["rmp"]},
    objective=objective,
    termination=(
      f"{why_stop_searching} "
      f"(it: {it}; obj. deviation: {odev}; best it: {best_it_so_far}; "
      f"best centralized it: {best_centralized_it}; "
      f"total runtime: {total_runtime})"
    ),
    data=sp_data,
  )


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
    disable_plotting = disable_plotting,
    v0 = args.v0
  )
