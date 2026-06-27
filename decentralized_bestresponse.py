from run_centralized_model import (
  encode_solution,
  get_current_load,
  init_complete_solution,
  init_problem,
  join_complete_solution,
  plot_history,
  save_checkpoint,
  save_solution,
  update_data,
)
from postprocessing import load_solution
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
  neigh_dict_to_matrix,
  start_additional_replicas,
)
from utils.centralized import check_feasibility
from utils.faasmacro import compute_centralized_objective
from utils.common import load_configuration
from models.sp import (
  LSP,
  LSP_capped,
  LSP_capped_fixedr,
  LSP_fixedr,
  LSPr,
  LSPr_fixedr,
)

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


def best_response_sweep(
    omega: np.array,
    residual_capacity: np.array,
    data: dict,
    neighborhood: np.array,
    rho: np.array,
    br_options: dict,
    latency: np.array,
    fairness: np.array,
    force_memory_bids: bool,
    *,
    order: str,
    response: str,
    rng: np.random.Generator = None,
    reopt_fn=None,
    current_y: np.array = None,
  ) -> Tuple[np.array, pd.DataFrame, int, float, float]:
  """Sequential (Gauss-Seidel) best-response sweep.

  Nodes act in `order` ("fixed" = ascending index, "random" = rng permutation);
  Each node releases its current buyer row, computes a new response greedily by
  descending score, and immediately commits it to the shared ledger. Later
  nodes therefore observe earlier coordinate updates. For response=="reopt",
  the target omega row is first recomputed through reopt_fn.
  """
  Nn = data[None]["Nn"][None]
  Nf = data[None]["Nf"][None]
  ledger = np.array(residual_capacity, dtype=float)
  omega = np.array(omega, dtype=float)
  if current_y is None:
    current_y = np.zeros((Nn, Nn, Nf))
  else:
    current_y = np.array(current_y, dtype=float)
  y_increment = np.zeros((Nn, Nn, Nf))
  memory_bids = {"i": [], "j": [], "f": []}
  reopt_runtime = 0.0
  placed_total = 0.0
  active = set()
  if order == "random":
    if rng is None:
      raise ValueError("rng is required when order='random'")
    node_order = [int(i) for i in rng.permutation(Nn)]
  elif order == "fixed":
    node_order = list(range(Nn))
  else:
    raise ValueError("order must be one of: fixed, random")
  if response not in {"greedy", "reopt"}:
    raise ValueError("response must be one of: greedy, reopt")
  for i in node_order:
    neighbours = set(int(j) for j in np.nonzero(neighborhood[i, :])[0])
    previous_row = current_y[i, :, :]
    if not np.any(omega[i, :] > 0) and not np.any(previous_row > 0):
      continue
    active.add(i)
    ledger += previous_row
    new_row = np.zeros((Nn, Nf))
    if response == "reopt" and reopt_fn is not None and np.any(omega[i, :] > 0):
      omega_ub_row = np.array(
        [sum(ledger[j, f] for j in neighbours) for f in range(Nf)]
      )
      capped_row, rt = reopt_fn(i, omega_ub_row)
      reopt_runtime += rt
      omega[i, :] = capped_row
    for f in range(Nf):
      if omega[i, f] <= 0:
        continue
      memory_requirement = data[None]["memory_requirement"][f + 1]
      potential_memory = {
        j for j in neighbours if rho[j] >= memory_requirement
      }
      potential_capacity = {j for j in neighbours if ledger[j, f] >= 1}
      score = {}
      for j in potential_capacity:
        s = (
          data[None]["beta"][(i + 1, j + 1, f + 1)]
          - br_options["latency_weight"] * latency[i, j]
          - br_options["fairness_weight"] * fairness[i, f]
        )
        if s > - data[None]["gamma"][(i + 1, f + 1)]:
          score[j] = s
      placed = 0.0
      for j in sorted(score, key=lambda k: (-score[k], k)):
        if placed >= omega[i, f]:
          break
        q = min(ledger[j, f], omega[i, f] - placed)
        if q <= 0:
          continue
        new_row[j, f] += q
        ledger[j, f] -= q
        placed += q
      placed_total += placed
      if placed < omega[i, f] or force_memory_bids:
        for j in sorted(score, key=lambda k: (-score[k], k)):
          if j in potential_memory:
            memory_bids["i"].append(i)
            memory_bids["j"].append(j)
            memory_bids["f"].append(f)
      if placed < omega[i, f] or force_memory_bids:
        for j in sorted(potential_memory - potential_capacity):
          memory_bids["i"].append(i)
          memory_bids["j"].append(j)
          memory_bids["f"].append(f)
    y_increment[i, :, :] = new_row - previous_row
  return (
    y_increment, pd.DataFrame(memory_bids), len(active),
    placed_total, reopt_runtime,
  )


def reoptimize_node(
    node: int,
    omega_ub_row: np.array,
    sp_data: dict,
    solver_name: str,
    general_solver_options: dict,
    parallelism: int,
    use_fixed_r: bool,
  ) -> Tuple[np.array, float]:
  """Capped local best response for one node.

  Deep-copies sp_data, sets the per-function offloading cap omega_ub to the
  accessible neighbour residual capacity, and re-solves ONLY ``node`` with
  LSP_capped (or LSP_capped_fixedr when fixed replicas are active). Returns the
  node's re-optimized omega row and the solve runtime. Only this omega row is
  consumed by the caller; the capped solve's x/z/r are diagnostic.
  """
  Nf = sp_data[None]["Nf"][None]
  node_data = deepcopy(sp_data)
  node_data[None]["omega_ub"] = {
    (f + 1): float(omega_ub_row[f]) for f in range(Nf)
  }
  model = LSP_capped_fixedr() if use_fixed_r else LSP_capped()
  result = solve_subproblem(
    node_data, [node], model, solver_name, general_solver_options, parallelism
  )
  sp_omega = result[4]
  runtime = result[10]["tot"]
  return np.array(sp_omega[node, :], dtype=float), float(runtime)


def compute_sweep_runtime(
    elapsed: float, reopt_runtime: float, n_active: int
  ) -> float:
  bookkeeping_runtime = max(0.0, elapsed - reopt_runtime)
  amortized_bookkeeping = (
    bookkeeping_runtime / n_active if n_active else bookkeeping_runtime
  )
  return reopt_runtime + amortized_bookkeeping


def parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Run FaaS-MABR",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
    "-c", "--config", help="Configuration file", type=str,
    default="config_files/manual_config.json",
  )
  parser.add_argument(
    "-j", "--parallelism",
    help="Number of parallel processes (-1: auto, 0: sequential)",
    type=int, default=-1,
  )
  parser.add_argument(
    "--disable_plotting",
    help="True to disable automatic plot generation for each experiment",
    default=False, action="store_true",
  )
  parser.add_argument(
    "--variant", choices=["s", "r", "o"], default="s",
    help="FaaS-MABR variant: s (sequential), r (randomized), o (re-optimization)",
  )
  return parser.parse_known_args()[0]


def _run(
    config: dict,
    parallelism: int,
    *,
    order: str,
    response: str,
    method_name: str,
    options_key: str,
    log_on_file: bool = False,
    disable_plotting: bool = False,
  ) -> str:
  base_solution_folder = config["base_solution_folder"]
  seed = config["seed"]
  limits = config["limits"]
  trace_type = config["limits"]["load"].get("trace_type", "fixed_sum")
  verbose = config.get("verbose", 0)
  solver_name = config["solver_name"]
  solver_options = config["solver_options"]
  general_solver_options = solver_options.get("general", {})
  br_options = dict(solver_options[options_key])
  br_options.setdefault("latency_weight", 0.0)
  br_options.setdefault("fairness_weight", 0.0)
  rng = np.random.default_rng(seed)
  time_limit = general_solver_options.get("TimeLimit", np.inf)
  tolerance = config.get("tolerance", 1e-6)
  max_iterations = config["max_iterations"]
  max_steps = config["max_steps"]
  min_run_time = config.get("min_run_time", 0)
  max_run_time = config.get("max_run_time", max_steps)
  run_time_step = config.get("run_time_step", 1)
  checkpoint_interval = config["checkpoint_interval"]
  patience = config["patience"]
  now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
  solution_folder = f"{base_solution_folder}/{now}"
  os.makedirs(solution_folder, exist_ok=True)
  with open(os.path.join(solution_folder, "config.json"), "w") as ostream:
    ostream.write(json.dumps(config, indent=2))
  log_stream = sys.stdout
  if log_on_file:
    log_stream = open(os.path.join(solution_folder, "out.log"), "w")
  base_instance_data, input_requests_traces, agents, graph = init_problem(
    limits, trace_type, max_steps, seed, solution_folder
  )
  Nn = base_instance_data[None]["Nn"][None]
  Nf = base_instance_data[None]["Nf"][None]
  opt_solution, opt_replicas, opt_detailed_fwd = None, None, None
  if "opt_solution_folder" in config:
    opt_solution, opt_replicas, opt_detailed_fwd, _, _ = load_solution(
      config["opt_solution_folder"], "LoadManagementModel"
    )
  neighborhood = neigh_dict_to_matrix(
    base_instance_data[None]["neighborhood"], Nn
  )
  latency = adjacency_matrix(graph, weight="network_latency")
  ub = (
    max_run_time + run_time_step
  ) if max_run_time == min_run_time else max_run_time
  sp_complete_solution = init_complete_solution()
  spc_complete_solution = init_complete_solution()
  obj_dict = {"LSPr_final": []}
  tc_dict = {"LSPr": []}
  runtime_list = []
  for t in range(min_run_time, ub, run_time_step):
    if verbose > 0:
      print(f"t = {t}", file=log_stream, flush=True)
    loadt = get_current_load(input_requests_traces, agents, t)
    data = update_data(base_instance_data, {"incoming_load": loadt})
    total_runtime = 0
    ss = datetime.now()
    sp_data = deepcopy(data)
    if opt_solution is not None:
      _, _, _, opt_r, _ = encode_solution(
        Nn, Nf, opt_solution, opt_detailed_fwd, opt_replicas, t
      )
      sp_data[None]["r_bar"] = {}
      for n in range(Nn):
        for f in range(Nf):
          sp_data[None]["r_bar"][(n + 1, f + 1)] = int(opt_r[n, f])
    sp = LSP() if opt_solution is None else LSP_fixedr()
    spr = LSPr() if opt_solution is None else LSPr_fixedr()
    (
      sp_data, sp_x, _, _, sp_omega, sp_r, sp_rho, sp_U, obj, tc, sp_runtime
    ) = solve_subproblem(
      sp_data, agents, sp, solver_name, general_solver_options, parallelism
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
    n_accepted_queue = deque(maxlen=patience)
    while not stop_searching:
      s = datetime.now()
      capacity, residual_capacity, ell = compute_residual_capacity(
        sp_x, y, sp_r, sp_data
      )
      blackboard = np.maximum(0.0, capacity - sp_x)
      coordination_rho = (
        sp_rho if opt_solution is None else np.zeros_like(sp_rho)
      )
      total_runtime += (datetime.now() - s).total_seconds()
      reopt_fn = None
      if response == "reopt":
        def reopt_fn(node, omega_ub_row):
          return reoptimize_node(
            node, omega_ub_row, sp_data, solver_name,
            general_solver_options, parallelism,
            use_fixed_r=(opt_solution is not None),
          )
      s = datetime.now()
      sweep_y, memory_bids, n_active, _, reopt_runtime = best_response_sweep(
        sp_omega, residual_capacity, sp_data, neighborhood, coordination_rho,
        br_options, latency, fairness,
        force_memory_bids=(
          (coordination_rho > 0).any()
          and len(n_accepted_queue) >= n_accepted_queue.maxlen
          and all(x == n_accepted_queue[0] for x in n_accepted_queue)
        ),
        order=order, response=response, rng=rng, reopt_fn=reopt_fn,
        current_y=y,
      )
      rt = (datetime.now() - s).total_seconds()
      total_runtime += compute_sweep_runtime(rt, reopt_runtime, n_active)
      allocation_changed = (np.abs(sweep_y) > tolerance).any()
      rmp_omega = y.sum(axis=1)
      additional_replicas = np.zeros((Nn, Nf))
      if allocation_changed:
        y += sweep_y
        y[np.abs(y) < tolerance] = 0.0
        rmp_omega = y.sum(axis=1)
        fairness += (rmp_omega > tolerance)
        n_accepted_queue.append(rmp_omega.sum())
        bad_nodes = check_ls_pr_feasibility_from_fixed_y(sp_data, y)
        if bad_nodes:
          raise RuntimeError(f"LSPr infeasible from fixed y assignments: {bad_nodes}")
        spr_sol, spr_obj, spr_tc, spr_runtime = compute_social_welfare(
          spr, sp_data, agents, solver_name, general_solver_options,
          y, rmp_omega, parallelism
        )
        total_runtime += spr_runtime
        sp_x, _, _, _, sp_r, sp_rho = spr_sol
        omega = sp_omega - rmp_omega
        omega[np.abs(omega) < tolerance] = 0.0
      if len(memory_bids) > 0 and not (additional_replicas > 0).any():
        s = datetime.now()
        additional_replicas, sp_rho = start_additional_replicas(
          memory_bids, sp_r, sp_data, sp_rho
        )
        sp_r += additional_replicas
        total_runtime += (datetime.now() - s).total_seconds()
      mabr_no_progress = (
        not allocation_changed and not (additional_replicas > tolerance).any()
      )
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
      if cobj > best_centralized_cost:
        best_centralized_cost = cobj
        best_centralized_solution = deepcopy(csol)
        best_centralized_it = it
      stop_searching, why_stop_searching = check_stopping_criteria(
        it, max_iterations, blackboard, omega, rmp_omega,
        additional_replicas, None, memory_bids,
        tolerance, total_runtime, time_limit
      )
      if not stop_searching and mabr_no_progress:
        stop_searching = True
        why_stop_searching = "no best-response progress"
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
          f"best centralized it: {best_centralized_it}; "
          f"total runtime: {total_runtime})"
        )
        if t % checkpoint_interval == 0 or t == max_steps - 1:
          save_checkpoint(
            sp_complete_solution, os.path.join(solution_folder, "LSP"), t
          )
          save_checkpoint(
            spc_complete_solution, os.path.join(solution_folder, "LSPc"), t
          )
    runtime_list.append(total_runtime)
    if verbose > 0:
      print(
        f"    TOTAL RUNTIME [s] = {total_runtime} "
        f"(wallclock: {(datetime.now() - ss).total_seconds()})",
        file=log_stream, flush=True
      )
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
  pd.DataFrame(obj_dict["LSPr_final"], columns=[method_name]).to_csv(
    os.path.join(solution_folder, "obj.csv"), index=False
  )
  pd.DataFrame(tc_dict["LSPr"]).to_csv(
    os.path.join(solution_folder, "termination_condition.csv")
  )
  pd.DataFrame({"tot": runtime_list}).to_csv(
    os.path.join(solution_folder, "runtime.csv"), index=False
  )
  if verbose > 0:
    print(f"All solutions saved in: {solution_folder}", file=log_stream, flush=True)
  if log_on_file:
    log_stream.close()
  return solution_folder


def run_br_s(config, parallelism, log_on_file=False, disable_plotting=False):
  return _run(config, parallelism, order="fixed", response="greedy",
              method_name="FaaS-MABR-S", options_key="br_s",
              log_on_file=log_on_file, disable_plotting=disable_plotting)


def run_br_r(config, parallelism, log_on_file=False, disable_plotting=False):
  return _run(config, parallelism, order="random", response="greedy",
              method_name="FaaS-MABR-R", options_key="br_r",
              log_on_file=log_on_file, disable_plotting=disable_plotting)


def run_br_o(config, parallelism, log_on_file=False, disable_plotting=False):
  return _run(config, parallelism, order="fixed", response="reopt",
              method_name="FaaS-MABR-O", options_key="br_o",
              log_on_file=log_on_file, disable_plotting=disable_plotting)


if __name__ == "__main__":
  args = parse_arguments()
  config = load_configuration(args.config)
  runner = {"s": run_br_s, "r": run_br_r, "o": run_br_o}[args.variant]
  runner(config, args.parallelism, disable_plotting=args.disable_plotting)
