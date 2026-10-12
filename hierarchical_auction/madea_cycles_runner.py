"""Alternate complete production MADEA cycles with higher-level auctions."""

from __future__ import annotations

import os
from copy import deepcopy
from functools import partial
import logging
from typing import Any

import numpy as np
import pandas as pd

from hierarchical_auction.engine import HierarchicalAuctionEngine
from hierarchical_auction.madea_runner import (
  build_auction_options, parse_arguments,
)
from models.sp import LSP, LSPr_x
from run_faasmacro import (
  parallel_solver_run,
  combine_solutions,
  compute_centralized_objective,
  compute_social_welfare,
  solve_subproblem,
)

from run_faasmadea import (
  MadeaState,
  run_madea_cycle,
  compute_offloaded_demand,
  check_ls_pr_feasibility_from_fixed_y,
  compute_residual_capacity,
)
from utils.common import load_configuration
from utils import run_frame, run_log, step_time

log = run_log.logger(__name__)


def run(
    config: dict[str, Any],
    parallelism: int = -1,
    log_on_file: bool = False,
    disable_plotting: bool = False,
  ) -> str:
  return _run(
    config, parallelism, log_on_file, disable_plotting,
    engine_class=HierarchicalAuctionEngine, result_name="HierarchicalMADeACycles",
  )


OUTPUTS = {"LSPc": run_frame.Output(checkpoint=True, plot_filename="sp.png")}


@parallel_solver_run
def _run(
    config: dict[str, Any],
    parallelism: int,
    log_on_file: bool,
    disable_plotting: bool,
    *,
    engine_class: type[HierarchicalAuctionEngine],
    result_name: str,
    refine_welfare: bool = False,
  ) -> str:
  """Shared cycle orchestration; variants select an engine and output column."""
  max_cycles = config.get("max_cycles")
  if max_cycles is not None and (type(max_cycles) is not int or max_cycles < 1):
    raise ValueError("max_cycles must be a positive integer or null")
  if refine_welfare:
    from madea_pg import refinement_options
    refinement_options(config)  # invalid refinement limits fail before the run
  refinement: list[dict] = []  # refinement.csv rows of the whole run (PG)
  folder = run_frame.run(
    config, parallelism, log_on_file=log_on_file, disable_plotting=disable_plotting,
    method=result_name, outputs=OUTPUTS,
    solve_step=partial(_solve_step, engine_class=engine_class, max_cycles=max_cycles,
                       auction_options=build_auction_options(config),
                       refinement=refinement if refine_welfare else None),
  )
  if refine_welfare:
    pd.DataFrame(refinement).to_csv(os.path.join(folder, "refinement.csv"), index=False)
  return folder


def _solve_step(ctx, t: int, data: dict, *, engine_class, max_cycles, auction_options: dict,
                refinement) -> run_frame.StepResult:
  Nn, Nf, tolerance = ctx.Nn, ctx.Nf, ctx.tolerance
  time_limit = ctx.time_limit
  with step_time.untimed():
    latency = ctx.dense_latency
    if refinement is not None:
      from madea_pg import pg_reserve as reserve
      pg_reserve = reserve(ctx.config, Nn, time_limit)
    else:
      pg_reserve = 0.0
  hierarchy_time_limit = time_limit - pg_reserve
  cycle_started_at = step_time.now()
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
  state = MadeaState(
    y=np.zeros((Nn, Nn, Nf)), omega=deepcopy(sp_omega),
    p=np.zeros((Nn, Nf)), fairness=np.zeros((Nn, Nf)),
    sp_r=sp_r, sp_rho=sp_rho, total_runtime=step_time.current_modelled(),
  )
  engine = engine_class(
    neighborhood=ctx.neighborhood,
    num_functions=Nf,
    service_quantum=np.ones(Nf),
    max_depth=ctx.config.get("max_hierarchy_depth", 3),
    auction_options=auction_options,
  )
  previous_cycle_progress = None
  completed_cycles = 0
  while True:
    state.total_runtime = step_time.current_modelled()
    if state.total_runtime >= hierarchy_time_limit:
      reason = f"reached time limit: {state.total_runtime} >= {hierarchy_time_limit}"
      with step_time.untimed():
        # The initial local solve may exhaust the budget before the first auction.
        if state.best_centralized_solution is None:
          state.best_centralized_solution = combine_solutions(
            Nn, Nf, sp_data, loadt, sp_x, state.sp_r, state.sp_rho,
            None, state.y, None, None, None, None,
          )
      break
    if max_cycles is not None and completed_cycles >= max_cycles:
      reason = f"max cycles reached: {completed_cycles} >= {max_cycles}"
      break
    state = run_madea_cycle(
      state, sp_x=sp_x, sp_omega=sp_omega, sp_data=sp_data, data=sp_data,
      agents=ctx.agents, loadt=loadt, neighborhood=ctx.neighborhood, latency=latency,
      config=ctx.config, auction_options=auction_options,
      parallelism=ctx.parallelism,
      started_at=cycle_started_at, auction_time_limit=hierarchy_time_limit,
    )
    # Use the actual reason, including its existing priority over other tests.
    reason = state.reason
    if reason == "all load assigned":
      break
    if state.total_runtime >= hierarchy_time_limit:
      reason = f"reached time limit: {state.total_runtime} >= {hierarchy_time_limit}"
      break
    # Measure the whole hierarchy -> MADEA pass. Route swaps are not
    # progress unless they increase assigned load or improve the incumbent.
    progress = (float(state.y.sum()), state.best_centralized_cost)
    if previous_cycle_progress is not None and all(
        current <= previous + tolerance
        for current, previous in zip(progress, previous_cycle_progress)
      ):
      reason = f"no progress after hierarchy and MADEA ({reason})"
      break
    previous_cycle_progress = progress

    s = step_time.now()
    _, residual_capacity, _ = compute_residual_capacity(
      sp_x, state.y, state.sp_r, sp_data,
    )
    result = engine.run_higher_levels(
      y=state.y,
      omega=state.omega,
      residual_capacity=residual_capacity,
      node_prices=state.p,
      latency=latency,
      fairness=state.fairness,
    )
    state.y = result.y
    rmp_omega = compute_offloaded_demand(state.y)
    e = step_time.now()
    with step_time.untimed():
      if log.isEnabledFor(logging.DEBUG):
        log.debug("        higher-level auction: DONE (y = %s; rmp_omega = %s; runtime = %s)",
                  state.y.tolist(), rmp_omega.tolist(), e - s)
    if result.accepted_allocations:
      with step_time.untimed():
        bad_nodes = check_ls_pr_feasibility_from_fixed_y(sp_data, state.y)
        if bad_nodes:
          raise RuntimeError(f"LSPr infeasible from fixed y assignments: {bad_nodes}")
      spr_sol, spr_obj, spr_tc, spr_runtime = compute_social_welfare(
        spr, sp_data, ctx.agents, ctx.solver_name, ctx.general_options,
        state.y, rmp_omega, ctx.parallelism, sp_x,
      )
      _, _, _, _, state.sp_r, state.sp_rho = spr_sol
      state.fairness += (rmp_omega > 0).astype(state.fairness.dtype)
      with step_time.untimed():
        log.debug("        solve 'restricted problem': DONE (%s; obj: %s; runtime = %s)",
                  spr_tc, spr_obj, spr_runtime)
    state.omega = sp_omega - rmp_omega
    state.omega[np.abs(state.omega) < tolerance] = 0.0
    with step_time.untimed():
      # Preserve a better hierarchical incumbent even if the next cycle swaps it.
      combined = combine_solutions(
        Nn, Nf, sp_data, loadt, sp_x, state.sp_r, state.sp_rho,
        None, state.y, None, None, None, None,
      )
      cost = compute_centralized_objective(
        sp_data, combined["sp"]["x"], combined["sp"]["y"], combined["sp"]["z"],
      )
      if cost > state.best_centralized_cost:
        state.best_centralized_cost = cost
        state.best_centralized_solution = deepcopy(combined)
        state.best_centralized_it = state.iterations - 1
        log.info("        best centralized solution updated; obj = %s", cost)
    completed_cycles += 1

  if refinement is not None:
    from madea_pg import refine_solution
    state.best_centralized_solution, refined = refine_solution(
      sp_data, state.best_centralized_solution, ctx.config, time_limit - step_time.current_modelled(),
    )
    refinement.append({"time": t, **refined})
  state.total_runtime = step_time.current_modelled()
  with step_time.untimed():
    central = state.best_centralized_solution["sp"]
    objective = compute_centralized_objective(sp_data, central["x"], central["y"], central["z"])
  return run_frame.StepResult(
    solutions={"LSPc": central}, objective=objective,
    termination=(
      f"{reason} (it: {state.iterations - 1}; "
      f"obj. deviation: {state.objective_deviation}; "
      f"best it: {state.best_it_so_far}; "
      f"best centralized it: {state.best_centralized_it}; "
      f"total runtime: {state.total_runtime})"
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
