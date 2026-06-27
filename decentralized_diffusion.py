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
  VAR_TYPE,
  check_ls_pr_feasibility_from_fixed_y,
  check_stopping_criteria,
  compute_residual_capacity,
  neigh_dict_to_matrix,
  start_additional_replicas,
)
from utils.centralized import check_feasibility
from utils.faasmacro import compute_centralized_objective
from utils.common import load_configuration
from models.sp import LSP, LSP_fixedr, LSPr, LSPr_fixedr

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


def define_assignments(
    omega: np.array,
    blackboard: np.array,
    data: dict,
    neighborhood: np.array,
    rho: np.array,
    diffusion_options: dict,
    latency: np.array,
    fairness: np.array,
    force_memory_bids: bool,
  ) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
  """Price-free counterpart of run_faasmadea.define_bids.

  Greedy-by-utility assignment of residual load to neighbours with spare
  capacity. No bid price is computed (no epsilon/delta); the per-pair score is
  stored in the ``utility`` column, on which evaluate_assignments later sorts.
  """
  potential_buyers, functions_to_share = np.nonzero(omega)
  bids = {"i": [], "j": [], "f": [], "d": [], "utility": []}
  memory_bids = {"i": [], "j": [], "f": []}
  for i, f in zip(potential_buyers, functions_to_share):
    potential_sellers = set(np.nonzero(neighborhood[i, :])[0])
    potential_capacity_sellers = potential_sellers.intersection(
      set(np.where(blackboard[:, f] >= 1)[0])
    )
    memory_requirement = data[None]["memory_requirement"][f + 1]
    potential_memory_sellers = {
      int(j) for j in potential_sellers if rho[int(j)] >= memory_requirement
    }
    utility = []
    candidate_sellers = []
    for j in potential_capacity_sellers:
      ut = (
        data[None]["beta"][(i + 1, j + 1, f + 1)]
        - diffusion_options["latency_weight"] * latency[i, j]
        - diffusion_options["fairness_weight"] * fairness[i, f]
      )
      if ut > - data[None]["gamma"][(i + 1, f + 1)]:
        utility.append(ut)
        candidate_sellers.append(j)
    assigned = 0
    if len(utility) > 0:
      sellers_order = sorted(
        range(len(candidate_sellers)),
        key=lambda idx: (-utility[idx], candidate_sellers[idx]),
      )
      idx = 0
      while idx < len(sellers_order) and assigned < omega[i, f]:
        j = candidate_sellers[sellers_order[idx]]
        if diffusion_options.get("unit_bids", False):
          d = 1
          while (d < int(min(blackboard[j, f], omega[i, f])) + 1) and (
              assigned < omega[i, f]
            ):
            bids["i"].append(i)
            bids["f"].append(f)
            bids["j"].append(j)
            bids["d"].append(1)
            bids["utility"].append(utility[sellers_order[idx]])
            assigned += 1
            d += 1
        else:
          d = VAR_TYPE(min(blackboard[j, f], (omega[i, f] - assigned)))
          bids["i"].append(i)
          bids["f"].append(f)
          bids["j"].append(j)
          bids["d"].append(d)
          bids["utility"].append(utility[sellers_order[idx]])
          assigned += d
        idx += 1
      if assigned < omega[i, f] or force_memory_bids:
        for idx in sellers_order:
          j = candidate_sellers[idx]
          if j in potential_memory_sellers:
            memory_bids["i"].append(i)
            memory_bids["j"].append(j)
            memory_bids["f"].append(f)
    if assigned < omega[i, f] or force_memory_bids:
      for j in potential_memory_sellers - potential_capacity_sellers:
        memory_bids["i"].append(i)
        memory_bids["j"].append(j)
        memory_bids["f"].append(f)
  return pd.DataFrame(bids), pd.DataFrame(memory_bids), len(potential_buyers)


def evaluate_assignments(
    bids: pd.DataFrame,
    residual_capacity: np.array,
    data: dict,
    ell: np.array,
    r: np.array,
    initial_rho: np.array,
    tentatively_start_replicas: bool,
    last_y: np.array = None,
    diffusion_options: dict = None,
    latency: np.array = None,
    fairness: np.array = None,
  ) -> Tuple[np.array, np.array, int]:
  """Price-free counterpart of run_faasmadea.evaluate_bids.

  Pure greedy capacity fill: sellers serve buyers by descending ``utility``
  (tie-break on buyer index ``i`` for reproducibility). No min_b tracking and no
  price update. When the seller is full and ``last_y`` plus score inputs are
  provided, a higher-score request may replace a lower-score incumbent, matching
  the auction's reassignment role without using prices.
  """
  Nn = data[None]["Nn"][None]
  Nf = data[None]["Nf"][None]
  if last_y is None:
    last_y = np.zeros((Nn, Nn, Nf))
  if diffusion_options is None:
    diffusion_options = {"latency_weight": 0.0, "fairness_weight": 0.0}
  if latency is None:
    latency = np.zeros((Nn, Nn))
  if fairness is None:
    fairness = np.zeros((Nn, Nf))
  seller_pairs = {
    (int(j), int(f)) for j, f in zip(*np.nonzero(residual_capacity > 0))
  }
  for row in bids[["j", "f"]].itertuples(index=False):
    j, f = int(row.j), int(row.f)
    if last_y[:, j, f].sum() > 0:
      seller_pairs.add((j, f))
  if tentatively_start_replicas:
    for j in np.nonzero(initial_rho)[0]:
      for f in range(Nf):
        seller_pairs.add((int(j), f))
  y = np.zeros((Nn, Nn, Nf))
  additional_replicas = np.zeros((Nn, Nf))
  rho = deepcopy(initial_rho)
  for j, f in sorted(seller_pairs):
    bids_for_j = bids[(bids["j"] == j) & (bids["f"] == f)].sort_values(
      by=["utility", "i"], ascending=[False, True]
    ).reset_index(drop=True)
    bid_remaining = bids_for_j["d"].to_numpy(dtype=float, copy=True)
    remaining_capacity = float(residual_capacity[j, f])
    next_bid_idx = 0
    while next_bid_idx < len(bids_for_j) and remaining_capacity > 0:
      q = min(remaining_capacity, bid_remaining[next_bid_idx])
      y[int(bids_for_j.iloc[next_bid_idx]["i"]), j, f] += q
      remaining_capacity -= q
      bid_remaining[next_bid_idx] -= q
      if bid_remaining[next_bid_idx] <= 1e-12:
        next_bid_idx += 1

    if tentatively_start_replicas and next_bid_idx < len(bids_for_j):
      memory_requirement = data[None]["memory_requirement"][f + 1]
      max_a = int(rho[j] // memory_requirement)
      if max_a > 0:
        demand = data[None]["demand"][(j + 1, f + 1)]
        max_utilization = data[None]["max_utilization"][f + 1]
        max_capacity = (r[j, f] + max_a) * max_utilization / demand
        replica_capacity = max(0.0, max_capacity - ell[j, f] - y[:, j, f].sum())
        while next_bid_idx < len(bids_for_j) and replica_capacity > 1e-12:
          q = min(replica_capacity, bid_remaining[next_bid_idx])
          i = int(bids_for_j.iloc[next_bid_idx]["i"])
          y[i, j, f] += q
          bid_remaining[next_bid_idx] -= q
          replica_capacity -= q
          if bid_remaining[next_bid_idx] <= 1e-12:
            next_bid_idx += 1
        hosted_load = ell[j, f] + y[:, j, f].sum()
        required_r = int(np.ceil((demand * hosted_load / max_utilization) - 1e-12))
        a = min(max_a, max(0, required_r - int(r[j, f])))
        additional_replicas[j, f] = a
        rho[j] -= a * memory_requirement

    incumbent_remaining = last_y[:, j, f].copy()
    while next_bid_idx < len(bids_for_j):
      i = int(bids_for_j.iloc[next_bid_idx]["i"])
      candidate_score = float(bids_for_j.iloc[next_bid_idx]["utility"])
      while bid_remaining[next_bid_idx] > 1e-12:
        replaceable = []
        for incumbent in np.nonzero(incumbent_remaining > 1e-12)[0]:
          incumbent = int(incumbent)
          if incumbent == i:
            continue
          incumbent_score = (
            data[None]["beta"][(incumbent + 1, j + 1, f + 1)]
            - diffusion_options["latency_weight"] * latency[incumbent, j]
            - diffusion_options["fairness_weight"] * fairness[incumbent, f]
          )
          if candidate_score > incumbent_score:
            replaceable.append((incumbent_score, -incumbent, incumbent))
        if not replaceable:
          break
        _, _, incumbent = min(replaceable)
        q = min(bid_remaining[next_bid_idx], incumbent_remaining[incumbent])
        y[incumbent, j, f] -= q
        y[i, j, f] += q
        incumbent_remaining[incumbent] -= q
        bid_remaining[next_bid_idx] -= q
      next_bid_idx += 1
  return y, additional_replicas, len(seller_pairs)


def parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Run FaaS-MADiG",
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
  diffusion_options = dict(solver_options["diffusion"])
  diffusion_options.setdefault(
    "unit_bids", solver_options.get("auction", {}).get("unit_bids", False)
  )
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
      bids, memory_bids, n_auctions = define_assignments(
        omega, blackboard, sp_data, neighborhood, coordination_rho,
        diffusion_options, latency, fairness,
        force_memory_bids=(
          (coordination_rho > 0).any()
          and len(n_accepted_queue) >= n_accepted_queue.maxlen
          and all(x == n_accepted_queue[0] for x in n_accepted_queue)
        ),
      )
      rt = (datetime.now() - s).total_seconds()
      total_runtime += (rt / n_auctions) if n_auctions else rt
      rmp_omega = y.sum(axis=1)
      additional_replicas = np.zeros((Nn, Nf))
      if len(bids) > 0:
        s = datetime.now()
        diffusion_y, additional_replicas, n_sellers = evaluate_assignments(
          bids, residual_capacity, sp_data, ell, sp_r, coordination_rho,
          tentatively_start_replicas=(len(memory_bids) == 0),
          last_y=y,
          diffusion_options=diffusion_options,
          latency=latency,
          fairness=fairness,
        )
        rt = (datetime.now() - s).total_seconds()
        total_runtime += (rt / n_sellers) if n_sellers else rt
        y += diffusion_y
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
        additional_replicas, bids, memory_bids,
        tolerance, total_runtime, time_limit
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
  pd.DataFrame(obj_dict["LSPr_final"], columns=["FaaS-MADiG"]).to_csv(
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
