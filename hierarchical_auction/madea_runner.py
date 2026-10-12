"""Hierarchical auction with the production FaaS-MADeA auction at level one."""

from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
from functools import partial
import logging
from typing import Any

import numpy as np
import pandas as pd

from hierarchical_auction.engine import HierarchicalAuctionEngine
from models.sp import LSP, LSPr_x
from run_faasmacro import (
  parallel_solver_run,
  combine_solutions,
  compute_centralized_objective,
  compute_social_welfare,
  solve_subproblem,
)

from run_faasmadea import (
  check_ls_pr_feasibility_from_fixed_y,
  check_stopping_criteria,
  compute_residual_capacity,
  define_bids,
  evaluate_bids,
  start_additional_replicas,
)
from utils.common import load_configuration
from utils import run_frame, run_log, step_time
from utils.faasmacro import relative_objective_gap

log = run_log.logger(__name__)


def build_auction_options(config: dict[str, Any]) -> dict[str, Any]:
  options = deepcopy(config.get("solver_options", {}).get("auction", {}))
  options.setdefault("epsilon", 0.01)
  options.setdefault("eta", [0.5, 0.3, 0.1])
  options.setdefault("zeta", 0.1)
  options.setdefault("latency_weight", 0.0)
  options.setdefault("fairness_weight", 0.0)
  options.setdefault("unit_bids", False)
  return options


def level1_options(options: dict[str, Any]) -> dict[str, Any]:
  result = deepcopy(options)
  if isinstance(result["eta"], list):
    result["eta"] = result["eta"][0]
  return result


def compute_offloaded_demand(y: np.ndarray) -> np.ndarray:
  return y.sum(axis=1)


def bids_for_stopping(
    bids: pd.DataFrame, hierarchical_allocations_count: int,
  ) -> pd.DataFrame:
  if len(bids) > 0 or hierarchical_allocations_count <= 0:
    return bids
  return pd.DataFrame({"hierarchical_progress": [hierarchical_allocations_count]})


def parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description = "Hierarchical auction extending production FaaS-MADeA",
    formatter_class = argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
    "-c", "--config", default="config_files/manual_config.json"
  )
  parser.add_argument(
    "-j", "--parallelism", type=int, default=-1
  )
  parser.add_argument(
    "--disable_plotting", action="store_true"
  )
  return parser.parse_known_args()[0]


OUTPUTS = {"LSPc": run_frame.Output(checkpoint=True, plot_filename="sp.png")}


@parallel_solver_run
def run(
    config: dict[str, Any],
    parallelism: int = -1,
    log_on_file: bool = False,
    disable_plotting: bool = False,
  ) -> str:
  auction_options = build_auction_options(config)
  return run_frame.run(
    config, parallelism, log_on_file=log_on_file, disable_plotting=disable_plotting,
    method="HierarchicalMADeA", outputs=OUTPUTS,
    solve_step=partial(_solve_step, auction_options=auction_options,
                       first_level_options=level1_options(auction_options)),
  )


def _solve_step(ctx, t: int, data: dict, *, auction_options: dict,
                first_level_options: dict) -> run_frame.StepResult:
  Nn, Nf, tolerance = ctx.Nn, ctx.Nf, ctx.tolerance
  patience = ctx.config.get("patience", 1)
  started_at = step_time.now()
  with step_time.untimed():
    latency = ctx.dense_latency
  loadt = data[None]["incoming_load"]
  sp_data = deepcopy(data)
  sp = LSP()
  spr = LSPr_x()
  (
    sp_data, sp_x, _, _, sp_omega, sp_r, sp_rho, _, obj, tc, sp_runtime
  ) = solve_subproblem(
    sp_data, ctx.agents, sp, ctx.solver_name, ctx.general_options, ctx.parallelism,
  )
  with step_time.untimed():
    if log.isEnabledFor(logging.DEBUG):
      log.debug("    sp: DONE (%s; obj = %s; x = %s; runtime = %s)",
                tc["tot"], obj["tot"], sp_x.tolist(), sp_runtime["tot"])
  u0 = np.ones((Nn, Nf)) * 0.8
  p = np.zeros((Nn, Nf))
  y = np.zeros((Nn, Nn, Nf))
  omega = deepcopy(sp_omega)
  fairness = np.zeros((Nn, Nf))
  accepted_queue = deque(maxlen=patience)
  odev_queue = deque(maxlen = patience)
  best_solution = None
  best_cost = -np.inf
  best_it = -1
  it = 0
  stop = False
  engine = HierarchicalAuctionEngine(
    neighborhood=ctx.neighborhood,
    num_functions=Nf,
    service_quantum=np.ones(Nf),
    max_depth=ctx.config.get("max_hierarchy_depth", 3),
    auction_options=auction_options,
  )

  while not stop:
    with step_time.untimed():
      log.info("    it = %s", it)
    s = step_time.now()
    total_capacity, residual_capacity, ell = compute_residual_capacity(
      sp_x, y, sp_r, sp_data,
    )
    blackboard = np.maximum(0.0, total_capacity - sp_x)
    stalled = (
      len(accepted_queue) >= accepted_queue.maxlen
      and all(value == accepted_queue[0] for value in accepted_queue)
    )
    e = step_time.now()
    with step_time.untimed():
      if log.isEnabledFor(logging.DEBUG):
        log.debug(
          "        compute_residual_capacity: DONE  (%s; blackboard = %s; ell = %s; stalled = %s; runtime = %s)",
          total_capacity.tolist(), blackboard.tolist(), ell.tolist(), stalled, e - s,
        )
    s = step_time.now()
    bids, memory_bids, n_auctions = define_bids(
      omega, blackboard, p, sp_data, ctx.neighborhood, sp_rho,
      first_level_options, latency, fairness,
      force_memory_bids=(sp_rho > 0).any() and stalled,
    )
    rt = step_time.now() - s
    with step_time.untimed():
      log.debug("        define_bids: DONE; runtime = %s; n_auctions = %s; tot runtime = %s)",
                rt / max(n_auctions, 1), n_auctions, rt)
      if ctx.config.get("verbose", 0) > 2:  # high-volume bid dump
        log.debug("%s", bids)
    additional_replicas = np.zeros((Nn, Nf))
    if len(bids) > 0:
      s = step_time.now()
      auction_y, p, additional_replicas, n_auctions = evaluate_bids(
        bids, blackboard, sp_data, y, ell, p, total_capacity, u0,
        first_level_options, sp_rho, sp_r,
        tentatively_start_replicas=(len(memory_bids) == 0),
        residual_capacity = residual_capacity
      )
      rt = step_time.now() - s
      with step_time.untimed():
        log.debug("        evaluate_bids: DONE; runtime = %s; n_auctions = %s; tot runtime = %s)",
                  rt / max(n_auctions, 1), n_auctions, rt)
      y += auction_y
      rmp_omega = compute_offloaded_demand(y)
      with step_time.untimed():
        bad_nodes = check_ls_pr_feasibility_from_fixed_y(sp_data, y)
        if bad_nodes:
          raise RuntimeError(f"LSPr infeasible from fixed y assignments: {bad_nodes}")
      spr_sol, spr_obj, spr_tc, spr_runtime = compute_social_welfare(
        spr, sp_data, ctx.agents, ctx.solver_name, ctx.general_options,
        y, rmp_omega, ctx.parallelism, sp_x
      )
      with step_time.untimed():
        log.debug("        solve 'restricted problem': DONE (%s; obj: %s; runtime = %s)",
                  spr_tc, spr_obj, spr_runtime)
      _, _, _, _, sp_r, sp_rho = spr_sol
      omega = sp_omega - rmp_omega
      omega[np.abs(omega) < tolerance] = 0.0
      with step_time.untimed():
        if log.isEnabledFor(logging.DEBUG):
          log.debug(
            "        solution updated: DONE (auct_y = %s; omega = %s; x = %s; r = %s; rho = %s;  y = %s)",
            auction_y.tolist(), omega.tolist(), sp_x.tolist(), sp_r.tolist(), sp_rho.tolist(),
            y.tolist(),
          )

    if len(memory_bids) > 0 and not (additional_replicas > 0).any():
      s = step_time.now()
      additional_replicas, sp_rho = start_additional_replicas(
        memory_bids, sp_r, sp_data, sp_rho,
      )
      sp_r += additional_replicas
      e = step_time.now()
      with step_time.untimed():
        if log.isEnabledFor(logging.DEBUG):
          log.debug("        additional replicas started: DONE (a = %s; rho = %s; runtime = %s)",
                    additional_replicas.tolist(), sp_rho.tolist(), e - s)
    s = step_time.now()
    _, residual_capacity, _ = compute_residual_capacity(
      sp_x, y, sp_r, sp_data,
    )
    result = engine.run_higher_levels(
      y=y,
      omega=omega,
      residual_capacity=residual_capacity,
      node_prices=p,
      latency=latency,
      fairness=fairness,
    )
    y = result.y
    rmp_omega = compute_offloaded_demand(y)
    e = step_time.now()
    with step_time.untimed():
      if log.isEnabledFor(logging.DEBUG):
        log.debug("        higher-level auction: DONE (y = %s; rmp_omega = %s; runtime = %s)",
                  y.tolist(), rmp_omega.tolist(), e - s)
    if result.accepted_allocations:
      with step_time.untimed():
        bad_nodes = check_ls_pr_feasibility_from_fixed_y(sp_data, y)
        if bad_nodes:
          raise RuntimeError(f"LSPr infeasible from fixed y assignments: {bad_nodes}")
      spr_sol, spr_obj, spr_tc, spr_runtime = compute_social_welfare(
        spr, sp_data, ctx.agents, ctx.solver_name, ctx.general_options,
        y, rmp_omega, ctx.parallelism, sp_x
      )
      _, _, _, _, sp_r, sp_rho = spr_sol
      with step_time.untimed():
        log.debug("        solve 'restricted problem': DONE (%s; obj: %s; runtime = %s)",
                  spr_tc, spr_obj, spr_runtime)
    omega = sp_omega - rmp_omega
    omega[np.abs(omega) < tolerance] = 0.0
    fairness += (rmp_omega > 0).astype(fairness.dtype)
    accepted_queue.append(float(rmp_omega.sum()))

    _, final_residual, _ = compute_residual_capacity(sp_x, y, sp_r, sp_data)
    with step_time.untimed():
      combined = combine_solutions(
        Nn, Nf, sp_data, loadt,
        sp_x, sp_r, sp_rho,
        None, y, None, None, None, None,
      )
      cost = compute_centralized_objective(
        sp_data, combined["sp"]["x"], combined["sp"]["y"], combined["sp"]["z"],
      )
      prev_cobj = best_cost
      if cost > best_cost:
        best_cost = cost
        best_solution = deepcopy(combined)
        best_it = it
        log.info("        best centralized solution updated; obj = %s", cost)
      odev_queue.append(
        relative_objective_gap(prev_cobj, best_cost)
      )

    elapsed = step_time.now() - started_at
    total_runtime = step_time.current_modelled()
    s = step_time.now()
    stop, reason = check_stopping_criteria(
      it=it,
      max_iterations=ctx.max_iterations,
      blackboard=final_residual,
      omega=omega,
      rmp_omega=rmp_omega,
      odev_queue=odev_queue,
      a=additional_replicas,
      bids=bids_for_stopping(bids, len(result.accepted_allocations)),
      memory_bids=memory_bids,
      tolerance=tolerance,
      total_runtime=total_runtime,
      time_limit=ctx.time_limit,
    )
    e = step_time.now()
    with step_time.untimed():
      log.debug(
        "        check_stopping_criteria: DONE (runtime = %s; total runtime = %s; wallclock: %s) --> stop? %s (%s)",
        e - s, total_runtime, elapsed, stop, reason,
      )
    if not stop:
      it += 1

  with step_time.untimed():
    central = best_solution["sp"]
    objective = compute_centralized_objective(sp_data, central["x"], central["y"], central["z"])
  return run_frame.StepResult(
    solutions={"LSPc": central}, objective=objective,
    termination=(
      f"{reason} (it: {it}; obj. deviation: None; best it: {best_it}; "
      f"best centralized it: {best_it}; total runtime: {total_runtime})"
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
    parallelism = parallelism, 
    log_on_file = False, 
    disable_plotting=disable_plotting,
  )
