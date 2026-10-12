from copy import deepcopy
from functools import partial
from typing import Tuple
import argparse
import os
from utils import run_frame, step_time

import numpy as np
import pandas as pd

from decentralized_diffusion import evaluate_assignments
from models.sp import LSP, LSP_fixedr, LSPr, LSPr_fixedr
from run_faasmacro import (
  parallel_solver_run,
  compute_social_welfare,
  solve_subproblem,
)
from run_faasmadea import (
  check_ls_pr_feasibility_from_fixed_y,
  check_stopping_criteria,
  compute_residual_capacity,
  start_additional_replicas,
)
from utils.centralized import (
  ping_pong_forbidden_hosts
)
from utils.common import load_configuration
from utils.faasmacro import compute_centralized_objective


@step_time.parallel
def pair_scores(
  data: dict,
  neighborhood: np.array,
  latency: np.array,
  fairness: np.array,
  dual_options: dict,
) -> Tuple[np.array, np.array]:
  """Return Cloud-relative advantages and their eligibility mask.

  Only neighboring pairs with strictly positive advantage are eligible; all
  other entries are ``-inf``.
  """
  values = data[None]
  nn = values["Nn"][None]
  nf = values["Nf"][None]
  scores = np.full((nn, nn, nf), -np.inf)
  eligible = np.zeros((nn, nn, nf), dtype=bool)

  for i in step_time.timed(range(nn)):
    for j in np.flatnonzero(neighborhood[i]):
      j = int(j)
      for f in range(nf):
        score = (
          values["beta"][(i + 1, j + 1, f + 1)]
          - dual_options["latency_weight"] * latency[i, j]
          - dual_options["fairness_weight"] * fairness[i, f]
          + values["gamma"][(i + 1, f + 1)]
        )
        if score > 0.0:
          scores[i, j, f] = score
          eligible[i, j, f] = True

  return scores, eligible


@step_time.parallel
def buyer_price_response(
  omega: np.array,
  capacity: np.array,
  lam: np.array,
  s: np.array,
  elig: np.array,
) -> Tuple[pd.DataFrame, np.array, float]:
  """Return uncapped best-seller demand and capacity-capped waterfill bids.

  Demand and the buyer dual term use the best strictly positive price-adjusted
  advantage with lower seller indices breaking ties; bids rank positive sellers
  sparsely while exposing the Cloud-relative utility.
  """
  nn, nf = omega.shape
  demand = np.zeros((nn, nf))
  dual_term = 0.0
  rows = []

  for i, f in step_time.timed(zip(*np.nonzero(omega > 0)), key=lambda item: int(item[0])):
    ranked = sorted(
      (
        (j, s[i, j, f] - lam[j, f])
        for j in np.flatnonzero(elig[i, :, f])
        if s[i, j, f] - lam[j, f] > 0.0
      ),
      key=lambda item: (-item[1], item[0]),
    )
    if not ranked:
      continue

    best = ranked[0][0]
    demand[best, f] += omega[i, f]
    dual_term += omega[i, f] * ranked[0][1]

    remaining = omega[i, f]
    for j, _ in ranked:
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
  current_y: np.array = None,
  tolerance: float = 1e-6,
  ) -> Tuple[np.array, pd.DataFrame, dict, int]:
  """Projected dual subgradient loop with primal recovery and certificate."""
  max_inner_iterations = dual_options["max_inner_iterations"]
  if isinstance(max_inner_iterations, bool) or not isinstance(
    max_inner_iterations, (int, np.integer)
  ):
    raise ValueError("max_inner_iterations must be a positive integer")
  numeric_options = {
    "alpha0": dual_options["alpha0"],
    "theta": dual_options["theta"],
    "gap_tolerance": dual_options["gap_tolerance"],
    "latency_weight": dual_options["latency_weight"],
    "fairness_weight": dual_options["fairness_weight"],
  }
  for name, value in numeric_options.items():
    if not np.isfinite(value):
      raise ValueError(f"{name} must be finite")
  if dual_options["step_rule"] == "sqrt" and dual_options["alpha0"] <= 0.0:
    raise ValueError("alpha0 must be positive")
  if dual_options["step_rule"] == "polyak" and dual_options["theta"] <= 0.0:
    raise ValueError("theta must be positive")
  if dual_options["gap_tolerance"] < 0.0:
    raise ValueError("gap_tolerance must be nonnegative")
  if max_inner_iterations < 1:
    raise ValueError("max_inner_iterations must be at least 1")
  if dual_options["step_rule"] not in {"sqrt", "polyak"}:
    raise ValueError("step_rule must be 'sqrt' or 'polyak'")
  nn = data[None]["Nn"][None]
  nf = data[None]["Nf"][None]
  scores, eligible = pair_scores(
    data, neighborhood, latency, fairness, dual_options
  )
  capacity = np.array(residual_capacity, dtype=float)
  # no_ping_pong: a node already sending f (residual demand or forwarded load)
  # must not host f. Drop it from the eligible seller set and zero its sellable
  # capacity so no primal bid can place load there.
  if current_y is None:
    current_y = np.zeros((nn, nn, nf))
  forbidden = ping_pong_forbidden_hosts(omega, current_y, tolerance)
  eligible = eligible & ~forbidden[None, :, :]
  capacity[forbidden] = 0.0
  lam = np.zeros((nn, nf))
  score_values = np.where(eligible, scores, 0.0)
  best_lb, best_ub = 0.0, np.inf
  best_lam = lam.copy()
  best_y = np.zeros((nn, nn, nf))
  ell = np.zeros((nn, nf))
  r = np.zeros((nn, nf))
  lb_history = []
  n_active = int((omega > 0).any(axis=1).sum())
  gap = 0.0
  k = 0

  for k in range(1, max_inner_iterations + 1):
    bids, demand, buyer_term = buyer_price_response(
      omega, capacity, lam, scores, eligible
    )
    ub = float((lam * capacity).sum() + buyer_term)
    if ub < best_ub:
      best_ub = ub
      best_lam = lam.copy()
    if len(bids) > 0:
      candidate_y, _, _ = evaluate_assignments(
        bids, residual_capacity, data, ell, r, rho,
        tentatively_start_replicas=False, last_y=None,
        diffusion_options=dual_options, latency=latency, fairness=fairness,
      )
      candidate_lb = float((score_values * candidate_y).sum())
      if candidate_lb > best_lb:
        best_lb = candidate_lb
        best_y = candidate_y.copy()
    lb_history.append(best_lb)
    gap = (best_ub - best_lb) / max(1.0, abs(best_ub))
    if gap <= dual_options["gap_tolerance"]:
      break
    if len(bids) == 0 and k == 1:
      # at k == 1 prices are zero, so the bid-eligible seller set is maximal:
      # no bids now (every positively-advantaged seller lacks capacity, or no
      # pair is eligible at all) means no bids at any later iteration either,
      # and the LP optimum is exactly 0, so the zero incumbent is certified
      # optimal
      best_ub = 0.0
      gap = 0.0
      break
    if len(bids) == 0 and demand.sum() <= 0:
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
    with step_time.parallel_phase() as phase:  # each seller updates its own price
      phase.uniform(nn)
      lam = np.maximum(0.0, lam + alpha * subgradient)

  memory_bids = {"i": [], "j": [], "f": []}
  placed = best_y.sum(axis=1)
  with step_time.parallel_phase():  # each buyer asks for replicas it still needs
    for i, f in step_time.timed(zip(*np.nonzero(omega > 0)), key=lambda item: int(item[0])):
      i, f = int(i), int(f)
      if placed[i, f] + 1e-12 < omega[i, f]:
        memory_requirement = data[None]["memory_requirement"][f + 1]
        for j in np.nonzero(neighborhood[i, :])[0]:
          if rho[int(j)] >= memory_requirement and not forbidden[int(j), f]:
            memory_bids["i"].append(i)
            memory_bids["j"].append(int(j))
            memory_bids["f"].append(f)
  memory_bids = pd.DataFrame(memory_bids)

  gap_info = {
    "LB": best_lb, "UB": best_ub, "gap": gap,
    "inner_iterations": k, "lam": lam, "best_lam": best_lam,
    "lb_history": lb_history,
  }
  return best_y, memory_bids, gap_info, n_active

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


def _capacity_state(
    sp_x: np.array, y: np.array, sp_r: np.array, sp_data: dict
  ) -> Tuple[np.array, np.array, np.array, np.array]:
  capacity, residual_capacity, ell = compute_residual_capacity(
    sp_x, y, sp_r, sp_data
  )
  blackboard = np.maximum(0.0, capacity - sp_x)
  return capacity, residual_capacity, ell, blackboard


CERTIFICATE_COLUMNS = [
  "timestep", "outer_iteration", "LB", "UB", "gap", "inner_iterations", "stop_reason",
]


def _dual_options(solver_options: dict) -> dict:
  options = dict(solver_options.get("dual", {}))
  options.setdefault("alpha0", 1.0)
  options.setdefault("step_rule", "sqrt")
  options.setdefault("theta", 1.0)
  options.setdefault("max_inner_iterations", 50)
  options.setdefault("gap_tolerance", 0.01)
  options.setdefault("latency_weight", 0.0)
  options.setdefault("fairness_weight", 0.0)
  return options


@parallel_solver_run
def run(
    config: dict,
    parallelism: int,
    log_on_file: bool = False,
    disable_plotting: bool = False,
  ) -> str:
  certificate = []  # coordination_certificate.csv rows of the whole run
  folder = run_frame.run(
    config, parallelism, log_on_file=log_on_file, disable_plotting=disable_plotting,
    method="FaaS-MALD", outputs=run_frame.LSP_OUTPUTS,
    solve_step=partial(_solve_step, options=_dual_options(config["solver_options"]),
                       certificate=certificate),
  )
  pd.DataFrame(certificate, columns=CERTIFICATE_COLUMNS).to_csv(
    os.path.join(folder, "coordination_certificate.csv"), index=False,
  )
  return folder


def _solve_step(ctx, t: int, data: dict, *, options: dict, certificate: list) -> run_frame.StepResult:
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
  while not stop_searching:
    capacity, residual_capacity, ell, blackboard = _capacity_state(
      sp_x, y, sp_r, sp_data
    )
    coordination_rho = np.zeros_like(sp_rho) if fixed else sp_rho
    y_inc, memory_bids, gap_info, n_active = dual_coordination_round(
      omega, residual_capacity, sp_data, ctx.neighborhood, coordination_rho,
      options, ctx.latency, fairness,
      current_y=y, tolerance=tolerance,
    )
    additional_replicas = np.zeros((Nn, Nf))
    rmp_omega = y.sum(axis=1)
    allocation_changed = (np.abs(y_inc) > tolerance).any()
    if allocation_changed:
      y += y_inc
      y[np.abs(y) < tolerance] = 0.0
      rmp_omega = y.sum(axis=1)
      fairness += (rmp_omega > tolerance)
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
    if len(memory_bids) > 0:
      additional_replicas, sp_rho = start_additional_replicas(
        memory_bids, sp_r, sp_data, sp_rho
      )
      sp_r += additional_replicas
      if (additional_replicas > tolerance).any():
        capacity, residual_capacity, ell, blackboard = _capacity_state(
          sp_x, y, sp_r, sp_data
        )
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
      a=additional_replicas, memory_bids=memory_bids,
      tolerance=tolerance, total_runtime=total_runtime, time_limit=ctx.time_limit,
    )
    if not stop_searching and not allocation_changed and not (
        additional_replicas > tolerance
      ).any():
      stop_searching = True
      why_stop_searching = "no dual progress"
    certificate.append(
      {
        "timestep": t,
        "outer_iteration": it,
        "LB": gap_info["LB"],
        "UB": gap_info["UB"],
        "gap": gap_info["gap"],
        "inner_iterations": gap_info["inner_iterations"],
        "stop_reason": why_stop_searching if stop_searching else "",
      }
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
      f"(it: {it}; fixed-C gap: {gap_info['gap']:.6f}; "
      f"fixed-C LB: {gap_info['LB']:.6f}; "
      f"fixed-C UB: {gap_info['UB']:.6f}; "
      f"inner: {gap_info['inner_iterations']}; "
      f"best it: {best_it_so_far}; "
      f"best centralized it: {central.iteration}; "
      f"total runtime: {total_runtime})"
    ),
    data=sp_data,
  )


if __name__ == "__main__":
  args = parse_arguments()
  config = load_configuration(args.config)
  run(config, args.parallelism, disable_plotting=args.disable_plotting)
