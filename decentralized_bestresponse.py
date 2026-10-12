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
from utils import run_frame, step_time
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

from collections import deque
from copy import deepcopy
from functools import partial
from typing import Tuple
import pandas as pd
import numpy as np
import argparse


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
    tolerance: float = 1e-6,
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
  # live view of committed forwarding: updated as each mover commits, so the
  # no_ping_pong guard below sees earlier movers' sends within this sweep
  # (matches decentralized_potentialgame's inductive argument).
  live_y = np.array(current_y, dtype=float)
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
    # no_ping_pong: a node already sending f (residual demand or forwarded
    # load, i's own released row excluded) must not host f; and the mover must
    # not offload any f it already hosts inbound.
    sender = live_y.sum(axis=1)
    sender[i, :] = 0.0
    forbidden = (omega > tolerance) | (sender > tolerance)
    inbound_i = live_y[:, i, :].sum(axis=0)
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
      if inbound_i[f] > tolerance:
        continue
      memory_requirement = data[None]["memory_requirement"][f + 1]
      potential_memory = {
        j for j in neighbours
        if rho[j] >= memory_requirement and not forbidden[j, f]
      }
      potential_capacity = {
        j for j in neighbours if ledger[j, f] >= 1 and not forbidden[j, f]
      }
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
    live_y[i, :, :] = new_row
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


@parallel_solver_run
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
  options = dict(config["solver_options"][options_key])
  options.setdefault("latency_weight", 0.0)
  options.setdefault("fairness_weight", 0.0)
  return run_frame.run(
    config, parallelism, log_on_file=log_on_file, disable_plotting=disable_plotting,
    method=method_name, outputs=run_frame.LSP_OUTPUTS,
    solve_step=partial(_solve_step, options=options, order=order, response=response,
                       rng=np.random.default_rng(config["seed"])),
  )


def _solve_step(ctx, t: int, data: dict, *, options: dict, order: str, response: str,
                rng) -> run_frame.StepResult:
  """One step: sequential sweeps run on the step clock (wall time minus observers);
  the local initialization is a parallel phase."""
  Nn, Nf, tolerance = ctx.Nn, ctx.Nf, ctx.tolerance
  time_limit = ctx.general_options.get("TimeLimit", np.inf)
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
    reopt_fn = None
    if response == "reopt":
      def reopt_fn(node, omega_ub_row):
        return reoptimize_node(
          node, omega_ub_row, sp_data, ctx.solver_name,
          ctx.general_options, ctx.parallelism,
          use_fixed_r=fixed,
        )
    sweep_y, memory_bids, n_active, _, _ = best_response_sweep(
      sp_omega, residual_capacity, sp_data, ctx.neighborhood, coordination_rho,
      options, ctx.latency, fairness,
      force_memory_bids=(
        (coordination_rho > 0).any()
        and len(n_accepted_queue) >= n_accepted_queue.maxlen
        and all(x == n_accepted_queue[0] for x in n_accepted_queue)
      ),
      order=order, response=response, rng=rng, reopt_fn=reopt_fn,
      current_y=y, tolerance=tolerance,
    )
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
    mabr_no_progress = (
      not allocation_changed and not (additional_replicas > tolerance).any()
    )
    csol, cobj = run_frame.snapshot(
      sp_data, Nn, Nf, sp_data, loadt, sp_x, sp_r, sp_rho,
      None, y, None, None, None, None
    )
    with step_time.untimed():  # observers: incumbents
      if spr_obj < best_cost_so_far or it == 0:
        best_cost_so_far = spr_obj
        best_solution_so_far = deepcopy(csol)
        best_it_so_far = it
    central.consider(csol, cobj, it)
    total_runtime = step_time.current_modelled()
    stop_searching, why_stop_searching = check_stopping_criteria(
      it, ctx.max_iterations, blackboard, omega, rmp_omega,
      a=additional_replicas, memory_bids=memory_bids,
      tolerance=tolerance, total_runtime=total_runtime, time_limit=time_limit,
    )
    if not stop_searching and mabr_no_progress:
      stop_searching = True
      why_stop_searching = "no best-response progress"
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
