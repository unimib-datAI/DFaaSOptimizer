"""Alternate complete production MADEA cycles with higher-level auctions."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from networkx import adjacency_matrix

from hierarchical_auction.engine import HierarchicalAuctionEngine
from hierarchical_auction.madea_runner import (
  build_auction_options, parse_arguments,
)
from models.sp import LSP, LSPr_x
from run_centralized_model import (
  get_current_load,
  init_complete_solution,
  init_problem,
  join_complete_solution,
  save_checkpoint,
  save_solution,
)
from run_faasmacro import (
  combine_solutions,
  compute_centralized_objective,
  compute_social_welfare,
  decode_solutions,
  solve_subproblem,
)

from run_faasmadea import (
  MadeaState,
  run_madea_cycle,
  compute_offloaded_demand,
  check_ls_pr_feasibility_from_fixed_y,
  compute_residual_capacity,
  neigh_dict_to_matrix,
)
from utils.common import load_configuration


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


def _run(
    config: dict[str, Any],
    parallelism: int,
    log_on_file: bool,
    disable_plotting: bool,
    *,
    engine_class: type[HierarchicalAuctionEngine],
    result_name: str,
  ) -> str:
  """Shared cycle orchestration; variants select an engine and output column."""
  del disable_plotting
  base_solution_folder = config["base_solution_folder"]
  seed = config["seed"]
  limits = config["limits"]
  trace_type = limits["load"].get("trace_type", "fixed_sum")
  verbose = config.get("verbose", 0)
  solver_name = config["solver_name"]
  solver_options = config.get("solver_options", {})
  general_solver_options = solver_options.get("general", {})
  auction_options = build_auction_options(config)
  tolerance = config.get("tolerance", 1e-6)
  max_steps = config["max_steps"]
  min_run_time = config.get("min_run_time", 0)
  max_run_time = config.get("max_run_time", max_steps)
  run_time_step = config.get("run_time_step", 1)
  checkpoint_interval = config["checkpoint_interval"]
  max_hierarchy_depth = config.get("max_hierarchy_depth", 3)

  now = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
  solution_folder = f"{base_solution_folder}/{now}"
  os.makedirs(solution_folder, exist_ok=True)
  with open(os.path.join(solution_folder, "config.json"), "w") as stream:
    stream.write(json.dumps(config, indent=2))
  log_stream = open(os.path.join(solution_folder, "out.log"), "w") \
    if log_on_file else sys.stdout

  base_data, traces, agents, graph = init_problem(
    limits, trace_type, max_steps, seed, solution_folder,
  )
  Nn = base_data[None]["Nn"][None]
  Nf = base_data[None]["Nf"][None]
  neighborhood = neigh_dict_to_matrix(base_data[None]["neighborhood"], Nn)
  latency = adjacency_matrix(graph, weight="network_latency").toarray()
  ub = max_run_time + run_time_step if max_run_time == min_run_time else max_run_time

  complete_solution = init_complete_solution()
  objectives = []
  termination_conditions = []
  runtimes = []

  for t in range(min_run_time, ub, run_time_step):
    if verbose > 0:
      print(f"t = {t}", file=log_stream, flush=True)
    started_at = time.monotonic()
    cycle_started_at = datetime.now()
    total_runtime = 0
    loadt = get_current_load(traces, agents, t)
    sp_data = deepcopy(base_data)
    sp_data[None]["incoming_load"] = loadt
    sp = LSP()
    spr = LSPr_x()
    (
      sp_data, sp_x, _, _, sp_omega, sp_r, sp_rho, _, obj, tc, sp_runtime
    ) = solve_subproblem(
      sp_data, agents, sp, solver_name, general_solver_options, parallelism,
    )
    if verbose > 1:
      print(
        f"    sp: DONE ({tc['tot']}; obj = {obj['tot']}; "
        f"x = {sp_x.tolist()}; runtime = {sp_runtime['tot']})",
        file = log_stream,
        flush = True
      )
    total_runtime += sp_runtime['tot']
    state = MadeaState(
      y=np.zeros((Nn, Nn, Nf)), omega=deepcopy(sp_omega),
      p=np.zeros((Nn, Nf)), fairness=np.zeros((Nn, Nf)),
      sp_r=sp_r, sp_rho=sp_rho, total_runtime=total_runtime,
    )
    engine = engine_class(
      neighborhood=neighborhood,
      num_functions=Nf,
      service_quantum=np.ones(Nf),
      max_depth=max_hierarchy_depth,
      auction_options=auction_options,
    )
    previous_cycle_progress = None
    while True:
      state = run_madea_cycle(
        state, sp_x=sp_x, sp_omega=sp_omega, sp_data=sp_data, data=sp_data,
        agents=agents, loadt=loadt, neighborhood=neighborhood, latency=latency,
        config=config, auction_options=auction_options,
        parallelism=parallelism, log_stream=log_stream,
        started_at=cycle_started_at,
      )
      # Use the actual reason, including its existing priority over other tests.
      reason = state.reason
      if reason == "all load assigned":
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

      s = time.monotonic()
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
      e = time.monotonic()
      state.total_runtime += e - s
      if verbose > 1:
        print(
          f"        higher-level auction: DONE (y = {state.y.tolist()}; "
          f"rmp_omega = {rmp_omega.tolist()}; runtime = {(e - s)})",
          file=log_stream, flush=True,
        )
      if result.accepted_allocations:
        bad_nodes = check_ls_pr_feasibility_from_fixed_y(sp_data, state.y)
        if bad_nodes:
          raise RuntimeError(f"LSPr infeasible from fixed y assignments: {bad_nodes}")
        spr_sol, spr_obj, spr_tc, spr_runtime = compute_social_welfare(
          spr, sp_data, agents, solver_name, general_solver_options,
          state.y, rmp_omega, parallelism, sp_x,
        )
        state.total_runtime += spr_runtime
        _, _, _, _, state.sp_r, state.sp_rho = spr_sol
        state.fairness += (rmp_omega > 0).astype(state.fairness.dtype)
        if verbose > 1:
          print(
            f"        solve 'restricted problem': DONE ({spr_tc}; "
            f"obj: {spr_obj}; runtime = {spr_runtime})",
            file=log_stream, flush=True,
          )
      state.omega = sp_omega - rmp_omega
      state.omega[np.abs(state.omega) < tolerance] = 0.0
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
        if verbose > 0:
          print(
            f"        best centralized solution updated; obj = {cost}",
            file=log_stream, flush=True,
          )

    complete_solution, _, objective = decode_solutions(
      sp_data, state.best_centralized_solution, complete_solution, None,
    )
    objectives.append(objective)
    termination_conditions.append(
      f"{reason} (it: {state.iterations - 1}; "
      f"obj. deviation: {state.objective_deviation}; "
      f"best it: {state.best_it_so_far}; "
      f"best centralized it: {state.best_centralized_it}; "
      f"total runtime: {state.total_runtime})"
    )
    runtimes.append(state.total_runtime)
    if t % checkpoint_interval == 0 or t == max_steps - 1:
      save_checkpoint(
        complete_solution, os.path.join(solution_folder, "LSPc"), t,
      )
    if verbose > 0:
      elapsed = time.monotonic() - started_at
      print(
        f"    TOTAL RUNTIME [s] = {state.total_runtime} (wallclock: {elapsed})",
        file=log_stream, flush=True,
      )

  solution, offloaded, detailed = join_complete_solution(complete_solution)
  save_solution(
    solution, offloaded, complete_solution, detailed, "LSPc", solution_folder,
  )
  pd.DataFrame({result_name: objectives}).to_csv(
    os.path.join(solution_folder, "obj.csv"), index=False,
  )
  pd.DataFrame(termination_conditions).to_csv(
    os.path.join(solution_folder, "termination_condition.csv"),
  )
  pd.DataFrame({"tot": runtimes}).to_csv(
    os.path.join(solution_folder, "runtime.csv"), index=False,
  )
  if log_on_file:
    log_stream.close()
  return solution_folder


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
