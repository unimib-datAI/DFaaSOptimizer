"""Hierarchical FaaS-MAFW: Frank-Wolfe at level one, structure auctions above.

Each coordination round has the shape of the hierarchical one-shot runner
(``hierarchical_auction.runner``), with the level-one auction replaced by one
FaaS-MAFW round:

1. level 1: distributed Frank-Wolfe on the relaxed one-hop problem, rounding
   and slack-preserving repair (``frankwolfe_assignments``), then LSPr with the
   forwarded load fixed;
2. levels 2..max_hierarchy_depth: the existing iterative structure auctions
   (``IterativeHierarchicalAuctionEngine``) sell the residual capacity of the
   replicas already running, as indivisible tokens, to the residual demand of
   the structures; flows stay between direct neighbours.  The node price of
   these auctions is the last memory price advertised by each receiver in the
   Frank-Wolfe iterations;
3. optional potential-game refinement of the incumbent (``madea_pg``), with
   the same time reservation as hierarchical-one-shot-pg.

``hierarchy=False`` skips step 2, so the same runner also gives the flat
MAFW + PG variant used as an ablation.  Like the hierarchical runner, the time
budget is wall-clock time per timestep.
"""

from run_faasmacro import (
  compute_social_welfare,
  parallel_solver_run,
  solve_subproblem,
)
from run_faasmadea import (
  check_ls_pr_feasibility_from_fixed_y,
  compute_residual_capacity,
)
from utils.faasmacro import compute_centralized_objective
from models.sp import LSP, LSPr
from decentralized_frankwolfe import (
  frankwolfe_assignments,
  frankwolfe_options,
  fw_parameters,
)
from hierarchical_auction.iterative_engine import IterativeHierarchicalAuctionEngine
from hierarchical_auction.runner import build_auction_options

from copy import deepcopy
from functools import partial
import pandas as pd
import numpy as np
import os
from utils import run_frame, run_log, step_time

log = run_log.logger(__name__)


RESULT_NAMES = {
  (True, False): "Hierarchical-MAFW",
  (True, True): "Hierarchical-MAFW-PG",
  (False, True): "MAFW-PG",
  (False, False): "FaaS-MAFW-flat",
}


def _update_from_lspr(spr, sp_data, agents, solver_name, solver_options, y, parallelism):
  with step_time.untimed():
    bad_nodes = check_ls_pr_feasibility_from_fixed_y(sp_data, y)
    if bad_nodes:
      raise RuntimeError(f"LSPr infeasible from fixed y assignments: {bad_nodes}")
  rmp_omega = y.sum(axis=1)
  spr_sol, spr_obj, _, spr_runtime = compute_social_welfare(
    spr, sp_data, agents, solver_name, solver_options, y, rmp_omega, parallelism
  )
  sp_x, _, _, _, sp_r, sp_rho = spr_sol
  return sp_x, sp_r, sp_rho, rmp_omega


OUTPUTS = {"LSPc": run_frame.Output(checkpoint=True)}


@parallel_solver_run
def run(
    config: dict,
    parallelism: int,
    log_on_file: bool = False,
    disable_plotting: bool = False,
    *,
    hierarchy: bool = True,
    refine_welfare: bool = False,
  ) -> str:
  if "opt_solution_folder" in config:
    raise ValueError("hierarchical FaaS-MAFW does not support fixed replicas")
  if refine_welfare:
    from madea_pg import refinement_options
    refinement_options(config)  # invalid refinement limits fail before the run
  stats, refinement = [], []  # frankwolfe_stats.csv and refinement.csv rows of the whole run
  folder = run_frame.run(
    config, parallelism, log_on_file=log_on_file, disable_plotting=disable_plotting,
    method=RESULT_NAMES[(hierarchy, refine_welfare)], outputs=OUTPUTS,
    solve_step=partial(
      _solve_step, hierarchy=hierarchy, refinement=refinement if refine_welfare else None,
      options=frankwolfe_options(config["solver_options"]), auction_options=build_auction_options(config),
      rng=np.random.default_rng(config["seed"]), stats=stats,
    ),
  )
  pd.DataFrame(stats).to_csv(os.path.join(folder, "frankwolfe_stats.csv"), index=False)
  if refinement:
    pd.DataFrame(refinement).to_csv(os.path.join(folder, "refinement.csv"), index=False)
  return folder


def _pg_reserve(config: dict, Nn: int, time_limit: float) -> float:
  from madea_pg import refinement_options
  pg_options = refinement_options(config)
  if pg_options["max_sweeps"] <= 0:
    return 0.0
  requested = (pg_options["time_limit_per_node"] * Nn
               if "time_limit_per_node" in pg_options else pg_options["time_limit"])
  return min(requested, time_limit / 2)


def _solve_step(ctx, t: int, data: dict, *, hierarchy: bool, refinement, options: dict,
                auction_options: dict, rng, stats: list) -> run_frame.StepResult:
  Nn, Nf, tolerance = ctx.Nn, ctx.Nf, ctx.tolerance
  loadt = data[None]["incoming_load"]
  with step_time.untimed():
    latency = ctx.dense_latency
    pg_reserve = _pg_reserve(ctx.config, Nn, ctx.time_limit) if refinement is not None else 0.0
  coordination_limit = ctx.time_limit - pg_reserve
  sp_data = deepcopy(data)
  (
    sp_data, sp_x, _, _, sp_omega, sp_r, sp_rho, _, _, _, _
  ) = solve_subproblem(
    sp_data, ctx.agents, LSP(), ctx.solver_name, ctx.general_options, ctx.parallelism
  )
  spr = LSPr()
  with step_time.untimed():  # dense copies of the instance (simulation bookkeeping)
    fw_params = fw_parameters(sp_data)
  y = np.zeros((Nn, Nn, Nf))
  omega = deepcopy(sp_omega)
  fairness = np.zeros((Nn, Nf))
  engine = IterativeHierarchicalAuctionEngine(
    neighborhood=ctx.neighborhood, num_functions=Nf, service_quantum=np.ones(Nf),
    max_depth=ctx.config.get("max_hierarchy_depth", 3), auction_options=auction_options,
  ) if hierarchy else None

  def snapshot():  # observer: global incumbent and feasibility check
    return run_frame.snapshot(
      sp_data, Nn, Nf, sp_data, loadt, sp_x, sp_r, sp_rho, None, y, None, None, None, None
    )

  # the local solution without forwarding is the first incumbent
  best = run_frame.Incumbent(*reversed(snapshot()))
  it = 0
  while True:
    y_new, fw_stats = frankwolfe_assignments(
      omega, y, sp_x, sp_r, sp_rho, sp_data, ctx.neighborhood, latency, fairness,
      options, rng, tolerance=tolerance, params=fw_params,
    )
    price = fw_stats.pop("price")
    if y_new.sum() > tolerance:
      y = y + y_new
      y[np.abs(y) < tolerance] = 0.0
      sp_x, sp_r, sp_rho, rmp_omega = _update_from_lspr(
        spr, sp_data, ctx.agents, ctx.solver_name, ctx.general_options, y, ctx.parallelism
      )
      fairness += (rmp_omega > tolerance)
      omega = sp_omega - rmp_omega
      omega[np.abs(omega) < tolerance] = 0.0
    accepted_higher = 0.0
    if engine is not None and (omega > tolerance).any():
      _, blackboard, _ = compute_residual_capacity(sp_x, y, sp_r, sp_data)
      result = engine.run_higher_levels(
        y=y, omega=omega, residual_capacity=blackboard,
        node_prices=np.maximum(price, 0.0), latency=latency, fairness=fairness,
      )
      if result.accepted_allocations:
        accepted_higher = float((result.y - y).sum())
        y = result.y
        y[np.abs(y) < tolerance] = 0.0
        sp_x, sp_r, sp_rho, rmp_omega = _update_from_lspr(
          spr, sp_data, ctx.agents, ctx.solver_name, ctx.general_options, y, ctx.parallelism
        )
        fairness += (rmp_omega > tolerance)
        omega = sp_omega - rmp_omega
        omega[np.abs(omega) < tolerance] = 0.0
    fw_stats.update({"t": t, "round": it, "accepted_higher_levels": accepted_higher})
    stats.append(fw_stats)
    csol, cobj = snapshot()
    best.consider(csol, cobj, it)
    elapsed = step_time.current_modelled()
    if it + 1 >= ctx.max_iterations:
      why = "max iterations"
    elif options["max_rounds"] and it + 1 >= options["max_rounds"]:
      why = "max rounds"
    elif not (omega > tolerance).any():
      why = "all load assigned"
    elif y_new.sum() <= tolerance and accepted_higher <= tolerance:
      why = "no more assignments"
    elif elapsed > coordination_limit:
      why = f"reached time limit: {elapsed} > {coordination_limit}"
    else:
      it += 1
      continue
    break

  incumbent = best.solution
  if refinement is not None:
    from madea_pg import refine_solution
    incumbent, refined = refine_solution(
      sp_data, incumbent, ctx.config, ctx.time_limit - step_time.current_modelled(),
    )
    refinement.append({"time": t, **refined, "coordination_rounds": it + 1,
                       "pg_reserve": pg_reserve})
  total = step_time.current_modelled()
  with step_time.untimed():
    central = incumbent["sp"]
    objective = compute_centralized_objective(sp_data, central["x"], central["y"], central["z"])
    log.info("t = %s: %s, rounds %s, welfare %s", t, why, it + 1, objective)
  return run_frame.StepResult(
    solutions={"LSPc": central}, objective=objective,
    termination=f"{why} (it: {it}; best centralized it: {best.iteration}; total runtime: {total})",
    data=sp_data,
  )


run_hierarchical = partial(run, hierarchy=True, refine_welfare=False)
run_hierarchical_pg = partial(run, hierarchy=True, refine_welfare=True)
run_flat_pg = partial(run, hierarchy=False, refine_welfare=True)


def _price_repair(config):
  """Same configuration, with the receiver-side (price-based) repair order."""
  config = deepcopy(config)
  config.setdefault("solver_options", {}).setdefault("frankwolfe", {})["repair_order"] = "price"
  return config


def run_mafw_b(config, parallelism, log_on_file=False, disable_plotting=False):
  """FaaS-MAFW-b: MAFW whose repair does not use the senders' beta."""
  from decentralized_frankwolfe import run as run_mafw
  return run_mafw(_price_repair(config), parallelism, log_on_file, disable_plotting)


def run_flat_pg_b(config, parallelism, log_on_file=False, disable_plotting=False):
  return run(_price_repair(config), parallelism, log_on_file, disable_plotting,
             hierarchy=False, refine_welfare=True)

