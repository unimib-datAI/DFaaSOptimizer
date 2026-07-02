from collections import deque
from copy import deepcopy
from datetime import datetime
from typing import Tuple
import argparse
import json
import os
import sys

from networkx import adjacency_matrix
import numpy as np
import pandas as pd

from decentralized_diffusion import evaluate_assignments
from models.sp import LSP, LSP_fixedr, LSPr, LSPr_fixedr
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
  neigh_dict_to_matrix,
  start_additional_replicas,
)
from utils.centralized import check_feasibility
from utils.common import load_configuration
from utils.faasmacro import compute_centralized_objective


def pair_scores(
  data: dict,
  neighborhood: np.array,
  latency: np.array,
  fairness: np.array,
  dual_options: dict,
) -> Tuple[np.array, np.array]:
  """Return unpriced beta-minus-penalty scores and their eligibility mask.

  Only neighboring pairs whose score beats rejection (``score > -gamma``)
  are eligible; all other score entries are ``-inf``.
  """
  values = data[None]
  nn = values["Nn"][None]
  nf = values["Nf"][None]
  scores = np.full((nn, nn, nf), -np.inf)
  eligible = np.zeros((nn, nn, nf), dtype=bool)

  for i in range(nn):
    for j in range(nn):
      for f in range(nf):
        score = (
          values["beta"][(i + 1, j + 1, f + 1)]
          - dual_options["latency_weight"] * latency[i, j]
          - dual_options["fairness_weight"] * fairness[i, f]
        )
        if neighborhood[i, j] and score > -values["gamma"][(i + 1, f + 1)]:
          scores[i, j, f] = score
          eligible[i, j, f] = True

  return scores, eligible


def buyer_price_response(
  omega: np.array,
  capacity: np.array,
  lam: np.array,
  s: np.array,
  elig: np.array,
) -> Tuple[pd.DataFrame, np.array, float]:
  """Return uncapped best-seller demand and capacity-capped waterfill bids.

  Demand and the buyer dual term use the best strictly positive price-adjusted
  score with lower seller indices breaking ties; bids rank all positive sellers
  while exposing the unpriced utility.
  """
  nn, nf = omega.shape
  demand = np.zeros((nn, nf))
  dual_term = 0.0
  rows = []

  for i, f in zip(*np.nonzero(omega > 0)):
    adjusted = np.where(elig[i, :, f], s[i, :, f] - lam[:, f], -np.inf)
    positive = np.flatnonzero(adjusted > 0)
    if not len(positive):
      continue

    ranked = positive[np.argsort(-adjusted[positive], kind="stable")]
    best = ranked[0]
    demand[best, f] += omega[i, f]
    dual_term += omega[i, f] * adjusted[best]

    remaining = omega[i, f]
    for j in ranked:
      quantity = min(remaining, capacity[j, f])
      if quantity > 0:
        rows.append({"i": i, "j": j, "f": f, "d": quantity, "utility": s[i, j, f]})
        remaining -= quantity
      if remaining <= 0:
        break

  bids = pd.DataFrame(rows, columns=["i", "j", "f", "d", "utility"])
  return bids, demand, dual_term


def dual_coordination_round(
  omega: np.array,
  residual_capacity: np.array,
  data: dict,
  neighborhood: np.array,
  rho: np.array,
  dual_options: dict,
  latency: np.array,
  fairness: np.array,
  force_memory_bids: bool,
  ell: np.array,
  r: np.array,
  ) -> Tuple[np.array, np.array, pd.DataFrame, dict, int]:
  """Projected dual subgradient loop with primal recovery and certificate."""
  if int(dual_options["max_inner_iterations"]) < 1:
    raise ValueError("max_inner_iterations must be at least 1")
  if dual_options["step_rule"] not in {"sqrt", "polyak"}:
    raise ValueError("step_rule must be 'sqrt' or 'polyak'")
  nn = data[None]["Nn"][None]
  nf = data[None]["Nf"][None]
  scores, eligible = pair_scores(
    data, neighborhood, latency, fairness, dual_options
  )
  capacity = np.array(residual_capacity, dtype=float)
  lam = np.zeros((nn, nf))
  score_values = np.where(eligible, scores, 0.0)
  best_lb, best_ub = 0.0, np.inf
  best_bids = pd.DataFrame(
    {"i": [], "j": [], "f": [], "d": [], "utility": []}
  )
  lb_history = []
  n_active = int((omega > 0).any(axis=1).sum())
  gap = 0.0
  k = 0

  for k in range(1, int(dual_options["max_inner_iterations"]) + 1):
    bids, demand, buyer_term = buyer_price_response(
      omega, capacity, lam, scores, eligible
    )
    best_ub = min(best_ub, float((lam * capacity).sum() + buyer_term))
    if len(bids) > 0:
      candidate_y, _, _ = evaluate_assignments(
        bids, residual_capacity, data, ell, r, rho,
        tentatively_start_replicas=False, last_y=None,
        diffusion_options=dual_options, latency=latency, fairness=fairness,
      )
      candidate_lb = float((score_values * candidate_y).sum())
      if candidate_lb > best_lb:
        best_lb, best_bids = candidate_lb, bids
    lb_history.append(best_lb)
    gap = (best_ub - best_lb) / max(1.0, abs(best_ub))
    if gap <= dual_options["gap_tolerance"]:
      break
    if demand.sum() <= 0 and len(bids) == 0:
      break
    subgradient = demand - capacity
    if dual_options["step_rule"] == "polyak":
      denominator = float((subgradient * subgradient).sum())
      alpha = (
        0.0 if denominator <= 0.0
        else dual_options["theta"] * max(0.0, best_ub - best_lb) / denominator
      )
    else:
      alpha = dual_options["alpha0"] / np.sqrt(k)
    lam = np.maximum(0.0, lam + alpha * subgradient)

  memory_bids = {"i": [], "j": [], "f": []}
  placed = (
    np.zeros((nn, nf)) if len(best_bids) == 0
    else evaluate_assignments(
      best_bids, residual_capacity, data, ell, r, rho,
      tentatively_start_replicas=False, last_y=None,
      diffusion_options=dual_options, latency=latency, fairness=fairness,
    )[0].sum(axis=1)
  )
  for i, f in zip(*np.nonzero(omega > 0)):
    i, f = int(i), int(f)
    if placed[i, f] < omega[i, f] or force_memory_bids:
      memory_requirement = data[None]["memory_requirement"][f + 1]
      for j in np.nonzero(neighborhood[i, :])[0]:
        if rho[int(j)] >= memory_requirement:
          memory_bids["i"].append(i)
          memory_bids["j"].append(int(j))
          memory_bids["f"].append(f)
  memory_bids = pd.DataFrame(memory_bids)

  y_increment = np.zeros((nn, nn, nf))
  additional_replicas = np.zeros((nn, nf))
  if len(best_bids) > 0:
    y_increment, additional_replicas, _ = evaluate_assignments(
      best_bids, residual_capacity, data, ell, r, rho,
      tentatively_start_replicas=(len(memory_bids) == 0), last_y=None,
      diffusion_options=dual_options, latency=latency, fairness=fairness,
    )
  gap_info = {
    "LB": best_lb, "UB": best_ub, "gap": gap,
    "inner_iterations": k, "lam": lam, "lb_history": lb_history,
  }
  return y_increment, additional_replicas, memory_bids, gap_info, n_active

def parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Run FaaS-MALD",
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
  return parser.parse_known_args()[0]


def run(
    config: dict,
    parallelism: int,
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
  dual_options = dict(solver_options.get("dual", {}))
  dual_options.setdefault("alpha0", 1.0)
  dual_options.setdefault("step_rule", "sqrt")
  dual_options.setdefault("theta", 1.0)
  dual_options.setdefault("max_inner_iterations", 50)
  dual_options.setdefault("gap_tolerance", 0.01)
  dual_options.setdefault("latency_weight", 0.0)
  dual_options.setdefault("fairness_weight", 0.0)
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
      s = datetime.now()
      y_inc, additional_replicas, memory_bids, gap_info, n_active = (
        dual_coordination_round(
          omega, residual_capacity, sp_data, neighborhood, coordination_rho,
          dual_options, latency, fairness,
          force_memory_bids=(
            (coordination_rho > 0).any()
            and len(n_accepted_queue) >= n_accepted_queue.maxlen
            and all(x == n_accepted_queue[0] for x in n_accepted_queue)
          ),
          ell=ell, r=sp_r,
        )
      )
      rt = (datetime.now() - s).total_seconds()
      total_runtime += (rt / n_active) if n_active else rt
      rmp_omega = y.sum(axis=1)
      allocation_changed = (np.abs(y_inc) > tolerance).any()
      if allocation_changed:
        y += y_inc
        y[np.abs(y) < tolerance] = 0.0
        rmp_omega = y.sum(axis=1)
        fairness += (rmp_omega > tolerance)
        n_accepted_queue.append(rmp_omega.sum())
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
        omega = sp_omega - rmp_omega
        omega[np.abs(omega) < tolerance] = 0.0
      if len(memory_bids) > 0 and not (additional_replicas > 0).any():
        s = datetime.now()
        additional_replicas, sp_rho = start_additional_replicas(
          memory_bids, sp_r, sp_data, sp_rho
        )
        sp_r += additional_replicas
        total_runtime += (datetime.now() - s).total_seconds()
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
      if not stop_searching and not allocation_changed and not (
          additional_replicas > tolerance
        ).any():
        stop_searching = True
        why_stop_searching = "no dual progress"
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
          f"(it: {it}; gap: {gap_info['gap']:.6f}; "
          f"LB: {gap_info['LB']:.6f}; UB: {gap_info['UB']:.6f}; "
          f"inner: {gap_info['inner_iterations']}; "
          f"best it: {best_it_so_far}; "
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
  pd.DataFrame(obj_dict["LSPr_final"], columns=["FaaS-MALD"]).to_csv(
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


if __name__ == "__main__":
  args = parse_arguments()
  config = load_configuration(args.config)
  run(config, args.parallelism, disable_plotting=args.disable_plotting)
