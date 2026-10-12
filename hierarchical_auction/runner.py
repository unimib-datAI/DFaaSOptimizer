"""Standalone runner for the hierarchical multi-structure auction.

Follows the same lifecycle as decentralized_auction.run():
init -> time loop -> iteration loop -> save.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from copy import deepcopy
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from networkx import adjacency_matrix as nx_adjacency_matrix

from decentralized_auction import (
  check_stopping_criteria,
  compute_residual_capacity,
  define_bids,
  evaluate_bids,
  start_additional_replicas,
)
from hierarchical_auction.engine import HierarchicalAuctionEngine
from hierarchical_auction.types import FloatArray
from models.sp import LSP, LSPr_x
from run_faasmacro import (
  parallel_solver_run,
  combine_solutions,
  compute_centralized_objective,
  compute_social_welfare,
  solve_subproblem,
)
from utils.common import load_configuration
from utils import run_frame, run_log, step_time

log = run_log.logger(__name__)


def parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Run Hierarchical Multi-Structure Auction",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
    "-c", "--config",
    help="Configuration file",
    type=str,
    default="config_files/manual_config.json",
  )
  parser.add_argument(
    "-j", "--parallelism",
    help="Number of parallel processes (-1: auto, 0: sequential)",
    type=int,
    default=-1,
  )
  parser.add_argument(
    "--disable_plotting",
    help="Disable automatic plot generation",
    default=False,
    action="store_true",
  )
  return parser.parse_known_args()[0]


def build_auction_options(config: Mapping[str, Any]) -> dict[str, object]:
  solver_options = config.get("solver_options", {})
  auction = solver_options.get("auction", {})
  return {
    "epsilon": auction.get("epsilon", 0.01),
    "eta": auction.get("eta", [0.5, 0.3, 0.1]),
    "zeta": auction.get("zeta", 0.1),
    "latency_weight": auction.get("latency_weight", 0.0),
    "fairness_weight": auction.get("fairness_weight", 0.0),
    "use_internal_value": auction.get("use_internal_value", False),
  }


def compute_offloaded_demand(y: FloatArray) -> FloatArray:
  """Return total offloaded demand per buyer node/function."""
  return y.sum(axis=1)


def bids_for_stopping(
  bids: pd.DataFrame,
  hierarchical_allocations_count: int,
) -> pd.DataFrame:
  """Avoid treating a productive hierarchical round as no-progress."""
  if len(bids) > 0 or hierarchical_allocations_count <= 0:
    return bids
  return pd.DataFrame({"hierarchical_progress": [hierarchical_allocations_count]})


def _extract_latency(graph: Any) -> FloatArray:
  return nx_adjacency_matrix(graph, weight="network_latency").toarray()


@parallel_solver_run
def run(
  config: dict[str, Any],
  parallelism: int = -1,
  log_on_file: bool = False,
  disable_plotting: bool = False,
  *,
  engine_class: type[HierarchicalAuctionEngine] = HierarchicalAuctionEngine,
  refine_welfare: bool = False,
) -> str:
  if refine_welfare:
    from madea_pg import refinement_options
    refinement_options(config)  # invalid refinement limits fail before the run
  auction_options = build_auction_options(config)
  auction_options["unit_bids"] = config.get("solver_options", {}).get("auction", {}).get("unit_bids", False)
  refinement: list[dict] = []  # refinement.csv rows of the whole run (PG)
  folder = run_frame.run(
    config, parallelism, log_on_file=log_on_file, disable_plotting=disable_plotting,
    method="HierarchicalOneShotPG" if refine_welfare else "HierarchicalAuction", outputs=OUTPUTS,
    solve_step=partial(_solve_step, engine_class=engine_class, auction_options=auction_options,
                       refinement=refinement if refine_welfare else None),
  )
  if refinement:
    pd.DataFrame(refinement).to_csv(os.path.join(folder, "refinement.csv"), index=False)
  return folder


OUTPUTS = {"LSPc": run_frame.Output(checkpoint=True, plot_filename="sp.png", plot_max_size=None)}


def _solve_step(ctx, t: int, data: dict, *, engine_class, auction_options: dict,
                refinement) -> run_frame.StepResult:
  Nn, Nf, tolerance = ctx.Nn, ctx.Nf, ctx.tolerance
  loadt = data[None]["incoming_load"]
  time_limit = ctx.time_limit
  with step_time.untimed():
    latency = ctx.dense_latency
    if refinement is not None:
      from madea_pg import pg_reserve as reserve
      pg_reserve = reserve(ctx.config, Nn, time_limit)
    else:
      pg_reserve = 0.0
  hierarchy_time_limit = time_limit - pg_reserve

  u0 = np.ones((Nn, Nf)) * 0.9
  p = np.zeros((Nn, Nf))
  fairness = np.zeros((Nn, Nf))

  sp_data = deepcopy(data)
  sp = LSP()
  spr = LSPr_x()
  (
    sp_data, sp_x, _, _sp_z, sp_omega, sp_r, sp_rho, _sp_u,
    _obj, _tc, sp_runtime,
  ) = solve_subproblem(
    sp_data, ctx.agents, sp, ctx.solver_name, ctx.general_options, ctx.parallelism,
  )

  it = 0
  stop_searching = False
  y = np.zeros((Nn, Nn, Nf))
  omega = deepcopy(sp_omega)
  rmp_omega = np.zeros((Nn, Nf))
  best_centralized_solution: dict | None = None
  best_centralized_cost = -np.inf
  best_centralized_it = -1
  stagnant_rounds = 0
  engine = engine_class(
    neighborhood=ctx.neighborhood,
    num_functions=Nf,
    service_quantum=np.ones(Nf),
    max_depth=ctx.config.get("max_hierarchy_depth", 3),
    auction_options=auction_options,
  )

  while not stop_searching:
    with step_time.untimed():
      log.info("    it = %s", it)

    if pg_reserve > 0:
      previous_y, previous_r, previous_p = y.copy(), sp_r.copy(), p.copy()
      previous_fairness = fairness.copy()

    capacity, blackboard, ell = compute_residual_capacity(
      sp_x, y, sp_r, sp_data,
    )

    level1_options = dict(auction_options)
    if isinstance(level1_options.get("eta"), list):
      level1_options["eta"] = level1_options["eta"][0]

    bids, memory_bids, _ = define_bids(
      omega, blackboard, p, sp_data, ctx.neighborhood, sp_rho,
      level1_options,
      latency=latency,
      fairness=fairness,
      force_memory_bids=False,
    )

    additional_replicas = np.zeros_like(sp_r)
    if len(bids) > 0:
      auction_y, p, _, _ = evaluate_bids(
        bids, blackboard, sp_data, previous_y=y, ell=ell, p=p,
        total_capacity=capacity, u0=u0, auction_options=level1_options,
        may_replace_existing_assignments=False,
      )
      y += auction_y

      rmp_omega = compute_offloaded_demand(y)
      fairness += (rmp_omega > 0).astype(fairness.dtype)

      spr_sol, _spr_obj, _spr_tc, spr_runtime = compute_social_welfare(
        spr, sp_data, ctx.agents, ctx.solver_name, ctx.general_options,
        y, rmp_omega, ctx.parallelism, sp_x
      )

      # sp_x = spr_sol[0]
      sp_r = spr_sol[4]
      sp_rho = spr_sol[5]
      omega = sp_omega - rmp_omega
      omega[np.abs(omega) < tolerance] = 0.0
    else:
      additional_replicas, sp_rho = start_additional_replicas(
        memory_bids, sp_r, sp_data, sp_rho,
      )
      sp_r += additional_replicas

    capacity, blackboard, ell = compute_residual_capacity(
      sp_x, y, sp_r, sp_data,
    )

    result = engine.run_higher_levels(
      y=y,
      omega=omega,
      residual_capacity=blackboard,
      node_prices=p,
      latency=latency,
      fairness=fairness,
    )
    y = result.y
    rmp_omega = compute_offloaded_demand(y)

    if result.accepted_allocations:
      spr_sol, _spr_obj, _spr_tc, spr_runtime = compute_social_welfare(
        spr, sp_data, ctx.agents, ctx.solver_name, ctx.general_options,
        y, rmp_omega, ctx.parallelism, sp_x
      )

      # sp_x = spr_sol[0]
      sp_r = spr_sol[4]
      sp_rho = spr_sol[5]
      omega = sp_omega - rmp_omega
      omega[np.abs(omega) < tolerance] = 0.0

    stopping_bids = bids_for_stopping(
      bids, len(result.accepted_allocations),
    )
    total_runtime = step_time.current_modelled()
    stop_searching, why_stop_searching = check_stopping_criteria(
      it=it, max_iterations=ctx.max_iterations, blackboard=blackboard,
      omega=omega, rmp_omega=rmp_omega, bids=stopping_bids,
      a=additional_replicas,
      memory_bids=memory_bids, tolerance=tolerance,
      total_runtime=total_runtime, time_limit=hierarchy_time_limit,
    )

    if pg_reserve > 0:
      # ponytail: two epsilon-stable rounds are a heuristic, not a convergence certificate.
      price_tolerance = auction_options["epsilon"] + tolerance
      stable = (np.array_equal(y, previous_y) and np.array_equal(sp_r, previous_r)
                and np.allclose(p, previous_p, rtol=0, atol=price_tolerance)
                and np.allclose(auction_options["fairness_weight"] * (fairness - previous_fairness),
                                0, rtol=0, atol=price_tolerance))
      stagnant_rounds = stagnant_rounds + 1 if stable else 0
      if not stop_searching and stagnant_rounds >= 2:
        stop_searching, why_stop_searching = True, "assignments, replicas and prices stalled"

    with step_time.untimed():
      csol = combine_solutions(
        Nn, Nf, sp_data, loadt,
        sp_x, sp_r, sp_rho,
        np.zeros((Nn, Nf)), y, np.zeros((Nn, Nf)),
        np.zeros((Nn, Nf)), np.zeros((Nn, Nn, Nf)),
        np.zeros((Nn,)),
      )
      cobj = compute_centralized_objective(
        sp_data, csol["sp"]["x"], csol["sp"]["y"], csol["sp"]["z"],
      )

      if cobj > best_centralized_cost:
        best_centralized_cost = cobj
        best_centralized_solution = csol
        best_centralized_it = it

    if not stop_searching:
      it += 1

  incumbent = best_centralized_solution if best_centralized_solution is not None else csol
  if refinement is not None:
    from madea_pg import refine_solution
    incumbent, refined = refine_solution(
      sp_data, incumbent, ctx.config, time_limit - step_time.current_modelled(),
    )
    refinement.append({"time": t, **refined,
                       "hierarchy_rounds": it + 1,
                       "hierarchy_pg_reserve": pg_reserve,
                       "hierarchy_time_budget": hierarchy_time_limit})
  total_runtime = step_time.current_modelled()
  with step_time.untimed():
    central = incumbent["sp"]
    objective = compute_centralized_objective(sp_data, central["x"], central["y"], central["z"])
  return run_frame.StepResult(
    solutions={"LSPc": central}, objective=objective,
    termination=(
      f"{why_stop_searching} (it: {it}; "
      f"best centralized it: {best_centralized_it}; "
      f"total runtime: {total_runtime})"
    ),
    data=sp_data,
  )


if __name__ == "__main__":
  args = parse_arguments()
  config = load_configuration(args.config)
  run(
    config,
    parallelism=args.parallelism,
    log_on_file=False,
    disable_plotting=args.disable_plotting,
  )
