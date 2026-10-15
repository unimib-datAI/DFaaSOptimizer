import numpy as np
import pandas as pd

from run_faasmacro import (
  parallel_solver_run,
  compute_social_welfare,
  solve_subproblem,
)
from run_faasmadea import (
  check_ls_pr_feasibility_from_fixed_y,
  check_stopping_criteria,
  compute_residual_capacity,
  define_bids,
)
from utils.faasmacro import compute_centralized_objective
from utils.common import load_configuration
from models.sp import LSP, LSPr

from copy import deepcopy
from functools import partial
import argparse
from utils import run_frame, run_log, step_time

log = run_log.logger(__name__)


@step_time.parallel
def resolve_gcaa_round(
    bids: pd.DataFrame,
    residual_capacity: np.array,
    current_y: np.array = None,
  ) -> np.array:
  from utils.compute_backend import record_operation
  record_operation('resolve_gcaa_round','cpu',len(residual_capacity),shape=[len(bids),*residual_capacity.shape],
                   reason='proposal CUDA adapter failed full-phase performance gate; shared sender/receiver roles require ordered consensus')
  return _resolve_gcaa_host(bids,residual_capacity,current_y)


def _resolve_gcaa_host(
    bids: pd.DataFrame,
    residual_capacity: np.array,
    current_y: np.array = None,
    best_per_agent=None,
  ) -> np.array:
  """One GCAA consensus round (Braquet & Bakolas, 2021 - Algorithms 1+3):
  each buyer agent (i, f) proposes only its highest-utility task (j, f);
  for each contested task, the single highest-utility proposal wins and
  consumes one unit of residual_capacity[j, f]. Losers are absent from
  the returned allocation and simply re-propose next round via a fresh
  call to define_bids (their omega is left untouched by this function)."""
  Nn, Nf = residual_capacity.shape
  y_round = np.zeros((Nn, Nn, Nf))
  if len(bids) == 0:
    return y_round
  if current_y is None:
    current_y = y_round
  sending = current_y.sum(axis=1) > 1e-10
  receiving = current_y.sum(axis=0) > 1e-10
  if best_per_agent is None:
    best_per_agent = bids.loc[bids.groupby(["i", "f"])["utility"].idxmax()]
  for (j, f), group in step_time.timed(best_per_agent.groupby(["j", "f"]), key=lambda item: int(item[0][0])):
    j, f = int(j), int(f)
    if sending[j, f]:
      continue
    for _, winner in group.sort_values("utility", ascending=False).iterrows():
      i = int(winner["i"])
      if receiving[i, f] or residual_capacity[j, f] < winner["d"]:
        continue
      y_round[i, j, f] += winner["d"]
      sending[i, f] = True
      receiving[j, f] = True
      break
  return y_round


def parse_arguments() -> argparse.Namespace:
  """
  Parse input arguments
  """
  parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description = "Run FaaS-MAGCAA",
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
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
  args: argparse.Namespace = parser.parse_known_args()[0]
  return args


@parallel_solver_run
def run(
    config: dict,
    parallelism: int,
    log_on_file: bool = False,
    disable_plotting: bool = False
  ):
  gcaa_options = config["solver_options"]["gcaa"]
  if gcaa_options.get("unit_bids") is not True:
    raise ValueError("solver_options.gcaa.unit_bids must be true")
  return run_frame.run(
    config, parallelism, log_on_file=log_on_file, disable_plotting=disable_plotting,
    method="FaaS-MAGCAA", outputs=run_frame.LSP_OUTPUTS,
    solve_step=partial(_solve_step, options=gcaa_options),
  )


def _solve_step(ctx, t: int, data: dict, *, options: dict) -> run_frame.StepResult:
  """One GCAA step; a fixed reference (opt_solution_folder) is not used."""
  Nn, Nf, tolerance = ctx.Nn, ctx.Nf, ctx.tolerance
  loadt = data[None]["incoming_load"]
  # GCAA bids are pure utility: price stays at zero forever (no
  # evaluate_bids-style price adaptation)
  no_price = np.zeros((Nn, Nf))
  # zero replica capacity disables define_bids' memory-bid path: GCAA
  # has no replica-bidding, a buyer with no convenient seller just gets
  # the null assignment (utility 0) for this round
  no_replica_sellers = np.zeros((Nn,))
  sp_data = deepcopy(data)
  sp = LSP()
  spr = LSPr()
  (
    sp_data, sp_x, _, _, sp_omega, sp_r, sp_rho, sp_U, obj, tc, sp_runtime
  ) = solve_subproblem(
    sp_data, ctx.agents, sp, ctx.solver_name, ctx.general_options, ctx.parallelism
  )
  with step_time.untimed():
    log.debug("    sp: DONE (%s; obj = %s; runtime = %s)", tc["tot"], obj["tot"], sp_runtime["tot"])
  it = 0
  stop_searching = False
  best_solution_so_far = None
  best_centralized_solution = None
  best_cost_so_far = np.inf
  spr_obj = np.inf
  best_centralized_cost = -np.inf
  best_it_so_far = -1
  y = np.zeros((Nn, Nn, Nf))
  omega = deepcopy(sp_omega)
  fairness = np.zeros((Nn, Nf))
  while not stop_searching:
    with step_time.untimed():
      log.info("    it = %d", it)
    capacity, residual_capacity, ell = compute_residual_capacity(
      sp_x, y, sp_r, sp_data
    )
    blackboard = np.maximum(0.0, capacity - sp_x)
    bids, memory_bids, n_auctions = define_bids(
      omega, blackboard, no_price, sp_data, ctx.neighborhood,
      no_replica_sellers, options, ctx.latency, fairness,
      force_memory_bids = False
    )
    with step_time.untimed():
      if ctx.config.get("verbose", 0) > 2:  # high-volume bid dump
        log.debug("%s", bids)
    rmp_omega = np.zeros((Nn, Nf))
    if len(bids) > 0:
      auction_y = resolve_gcaa_round(bids, residual_capacity, y)
      y += auction_y
      for n in range(Nn):
        for f in range(Nf):
          rmp_omega[n,f] = y[n,:,f].sum()
          if rmp_omega[n,f] > 0:
            fairness[n,f] += 1
      with step_time.untimed():
        bad_nodes = check_ls_pr_feasibility_from_fixed_y(sp_data, y)
        if bad_nodes:
          raise RuntimeError(
            f"LSPr infeasible from fixed y assignments: {bad_nodes}"
          )
      spr_sol, spr_obj, spr_tc, spr_runtime = compute_social_welfare(
        spr, sp_data, ctx.agents, ctx.solver_name, ctx.general_options,
        y, rmp_omega, ctx.parallelism
      )
      sp_x, _, _, _, sp_r, sp_rho = spr_sol
      for i in range(Nn):
        for f in range(Nf):
          omega[i,f] = sp_omega[i,f] - rmp_omega[i,f]
          if abs(omega[i,f]) < tolerance:
            omega[i,f] = 0.0
    csol, cobj = run_frame.snapshot(
      sp_data, Nn, Nf, sp_data, loadt, sp_x, sp_r, sp_rho,
      None, y, None, None, None, None
    )
    with step_time.untimed():
      if spr_obj < best_cost_so_far or it == 0:
        best_cost_so_far = spr_obj
        best_solution_so_far = deepcopy(csol)
        best_it_so_far = it
      if best_centralized_solution is None or cobj > best_centralized_cost:
        best_centralized_cost = cobj
        best_centralized_solution = deepcopy(csol)
    total_runtime = step_time.current_modelled()
    stop_searching, why_stop_searching = check_stopping_criteria(
      it, ctx.max_iterations, blackboard, omega, rmp_omega,
      bids=bids, memory_bids=memory_bids,
      tolerance=tolerance, total_runtime=total_runtime, time_limit=ctx.time_limit,
    )
    if not stop_searching:
      it += 1
  with step_time.untimed():
    best = best_solution_so_far["sp"]
    objective = compute_centralized_objective(sp_data, best["x"], best["y"], best["z"])
  return run_frame.StepResult(
    solutions={"LSP": best, "LSPc": best_centralized_solution["sp"]},
    objective=objective,
    termination=(
      f"{why_stop_searching} "
      f"(it: {it}; obj. deviation: {None}; best it: {best_it_so_far}; "
      f"total runtime: {total_runtime})"
    ),
    data=sp_data,
  )


if __name__ == "__main__":
  args = parse_arguments()
  config = load_configuration(args.config)
  run(
    config, args.parallelism, log_on_file = False,
    disable_plotting = args.disable_plotting
  )
