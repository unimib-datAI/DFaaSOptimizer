from run_faasmacro import (
  parallel_solver_run,
  compute_social_welfare,
  solve_subproblem,
)
from run_faasmadea import (
  VAR_TYPE,
  check_ls_pr_feasibility_from_fixed_y,
  check_stopping_criteria,
  compute_residual_capacity,
  start_additional_replicas,
)
from utils.centralized import (
  ping_pong_forbidden_hosts
)
from utils.faasmacro import compute_centralized_objective
from utils.common import load_configuration
from models.sp import LSP, LSP_fixedr, LSPr, LSPr_fixedr

from collections import deque
from copy import deepcopy
from typing import Tuple
import pandas as pd
import numpy as np
import argparse
from utils import run_frame, step_time


@step_time.parallel
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
    current_y: np.array = None,
    tolerance: float = 1e-6,
  ) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
  """Price-free counterpart of run_faasmadea.define_bids.

  Greedy-by-utility assignment of residual load to neighbours with spare
  capacity. No bid price is computed (no epsilon/delta); the per-pair score is
  stored in the ``utility`` column, on which evaluate_assignments later sorts.
  """
  if current_y is None:
    current_y = np.zeros((omega.shape[0], omega.shape[0], omega.shape[1]))
  # no_ping_pong: a node that is a sender of f (residual demand or already
  # forwarded) must not be picked as a host for f. Covers both the capacity and
  # the memory seller path, since both derive from ``potential_sellers``.
  forbidden = ping_pong_forbidden_hosts(omega, current_y, tolerance)
  potential_buyers, functions_to_share = np.nonzero(omega)
  bids = {"i": [], "j": [], "f": [], "d": [], "utility": []}
  memory_bids = {"i": [], "j": [], "f": []}
  for i, f in step_time.timed(zip(potential_buyers, functions_to_share), key=lambda item: item[0]):
    potential_sellers = set(np.nonzero(neighborhood[i, :])[0])
    potential_sellers -= {int(j) for j in np.nonzero(forbidden[:, f])[0]}
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


@step_time.parallel
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
  for j, f in step_time.timed(sorted(seller_pairs), key=lambda item: item[0]):
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


OUTPUTS = {"LSP": run_frame.Output(checkpoint=True, plot_filename="sp.png"),
           "LSPc": run_frame.Output(checkpoint=True)}


@parallel_solver_run
def run(
    config: dict,
    parallelism: int,
    log_on_file: bool = False,
    disable_plotting: bool = False,
  ) -> str:
  return run_frame.run(
    config, parallelism, log_on_file=log_on_file, disable_plotting=disable_plotting,
    method="FaaS-MADiG", solve_step=_solve_step, outputs=OUTPUTS,
  )


def _diffusion_options(solver_options: dict) -> dict:
  options = dict(solver_options["diffusion"])
  options.setdefault(
    "unit_bids", solver_options.get("auction", {}).get("unit_bids", False)
  )
  return options


def _solve_step(ctx, t: int, data: dict) -> run_frame.StepResult:
  with step_time.untimed():
    diffusion_options = _diffusion_options(ctx.solver_options)
  Nn, Nf, tolerance = ctx.Nn, ctx.Nf, ctx.tolerance
  loadt = data[None]["incoming_load"]
  fixed = ctx.reference is not None
  sp_data = ctx.step_data(data, t)
  sp = LSP_fixedr() if fixed else LSP()
  spr = LSPr_fixedr() if fixed else LSPr()
  (
    sp_data, sp_x, _, _, sp_omega, sp_r, sp_rho, sp_U, obj, tc, sp_runtime
  ) = solve_subproblem(
    sp_data, ctx.agents, sp, ctx.solver_name, ctx.general_options, ctx.parallelism
  )
  it = 0
  stop_searching = False
  best_solution_so_far = None
  best_cost_so_far = np.inf
  spr_obj = np.inf
  best_it_so_far = -1
  central = run_frame.Incumbent(-np.inf)
  y = np.zeros((Nn, Nn, Nf))
  omega = deepcopy(sp_omega)
  fairness = np.zeros((Nn, Nf))
  n_accepted_queue = deque(maxlen=ctx.patience)
  while not stop_searching:
    capacity, residual_capacity, ell = compute_residual_capacity(
      sp_x, y, sp_r, sp_data
    )
    blackboard = np.maximum(0.0, capacity - sp_x)
    coordination_rho = np.zeros_like(sp_rho) if fixed else sp_rho
    bids, memory_bids, n_auctions = define_assignments(
      omega, blackboard, sp_data, ctx.neighborhood, coordination_rho,
      diffusion_options, ctx.latency, fairness,
      force_memory_bids=(
        (coordination_rho > 0).any()
        and len(n_accepted_queue) >= n_accepted_queue.maxlen
        and all(x == n_accepted_queue[0] for x in n_accepted_queue)
      ),
      current_y=y, tolerance=tolerance,
    )
    rmp_omega = y.sum(axis=1)
    additional_replicas = np.zeros((Nn, Nf))
    if len(bids) > 0:
      diffusion_y, additional_replicas, n_sellers = evaluate_assignments(
        bids, residual_capacity, sp_data, ell, sp_r, coordination_rho,
        tentatively_start_replicas=(len(memory_bids) == 0),
        last_y=y,
        diffusion_options=diffusion_options,
        latency=ctx.latency,
        fairness=fairness,
      )
      y += diffusion_y
      y[np.abs(y) < tolerance] = 0.0
      rmp_omega = y.sum(axis=1)
      fairness += (rmp_omega > tolerance)
      n_accepted_queue.append(rmp_omega.sum())
      with step_time.untimed():
        bad_nodes = check_ls_pr_feasibility_from_fixed_y(sp_data, y)
        if bad_nodes:
          raise RuntimeError(f"LSPr infeasible from fixed y assignments: {bad_nodes}")
      spr_sol, spr_obj, spr_tc, spr_runtime = compute_social_welfare(
        spr, sp_data, ctx.agents, ctx.solver_name, ctx.general_options,
        y, rmp_omega, ctx.parallelism
      )
      sp_x, _, _, _, sp_r, sp_rho = spr_sol
      omega = sp_omega - rmp_omega
      omega[np.abs(omega) < tolerance] = 0.0
    if len(memory_bids) > 0 and not (additional_replicas > 0).any():
      additional_replicas, sp_rho = start_additional_replicas(
        memory_bids, sp_r, sp_data, sp_rho
      )
      sp_r += additional_replicas
    csol, cobj = run_frame.snapshot(
      sp_data, Nn, Nf, sp_data, loadt, sp_x, sp_r, sp_rho,
      None, y, None, None, None, None
    )
    with step_time.untimed():
      if spr_obj < best_cost_so_far or it == 0:
        best_cost_so_far = spr_obj
        best_solution_so_far = deepcopy(csol)
        best_it_so_far = it
    central.consider(csol, cobj, it)
    total_runtime = step_time.current_modelled()
    stop_searching, why_stop_searching = check_stopping_criteria(
      it, ctx.max_iterations, blackboard, omega, rmp_omega,
      a=additional_replicas, bids=bids, memory_bids=memory_bids,
      tolerance=tolerance, total_runtime=total_runtime, time_limit=ctx.time_limit,
    )
    if not stop_searching:
      it += 1
  with step_time.untimed():
    best = best_solution_so_far["sp"]
    objective = compute_centralized_objective(sp_data, best["x"], best["y"], best["z"])
  return run_frame.StepResult(
    solutions={"LSP": best, "LSPc": central.solution["sp"]},
    objective=objective,
    termination=(
      f"{why_stop_searching} "
      f"(it: {it}; obj. deviation: {None}; best it: {best_it_so_far}; "
      f"best centralized it: {central.iteration}; "
      f"total runtime: {total_runtime})"
    ),
    data=sp_data,
  )


if __name__ == "__main__":
  args = parse_arguments()
  config = load_configuration(args.config)
  run(config, args.parallelism, disable_plotting=args.disable_plotting)
