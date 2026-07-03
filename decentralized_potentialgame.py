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
  decode_solutions,
  solve_subproblem,
)
from run_faasmadea import (
  compute_residual_capacity,
  neigh_dict_to_matrix,
  start_additional_replicas,
)
from utils.centralized import check_feasibility
from utils.faasmacro import compute_centralized_objective
from utils.common import load_configuration
from models.sp import LSP, LSP_fixedr, LSP_pg, LSP_pg_fixedr

from datetime import datetime
from copy import deepcopy
from typing import Callable, Tuple
import pandas as pd
import numpy as np
import argparse
import json
import sys
import os


def compute_z(x: np.array, y: np.array, sp_data: dict) -> np.array:
  """Cloud-forwarded load derived from flow conservation (same convention as
  combine_solutions): z = load - x - outbound."""
  Nn = sp_data[None]["Nn"][None]
  Nf = sp_data[None]["Nf"][None]
  z = np.zeros((Nn, Nf))
  for n in range(Nn):
    for f in range(Nf):
      load = sp_data[None]["incoming_load"][(n + 1, f + 1)]
      z[n, f] = max(0.0, load - x[n, f] - y[n, :, f].sum())
  return z


def compute_node_utility(
    i: int, x: np.array, y: np.array, z: np.array, sp_data: dict
  ) -> float:
  """Node i's share of the centralized objective. Summing over all nodes
  yields exactly compute_centralized_objective (the exact potential)."""
  xm = np.zeros_like(x)
  ym = np.zeros_like(y)
  zm = np.zeros_like(z)
  xm[i, :] = x[i, :]
  ym[i, :, :] = y[i, :, :]
  zm[i, :] = z[i, :]
  return compute_centralized_objective(sp_data, xm, ym, zm)


def split_omega(
    i: int,
    omega_row: np.array,
    ledger: np.array,
    neighbours: set,
    sp_data: dict,
  ) -> np.array:
  """Fractional-knapsack split of node i's proposed offload across eligible
  sellers by descending beta (exact best response for the linear utility).
  Mutates ledger in place; unplaced residual is left to the Cloud (z)."""
  Nn = sp_data[None]["Nn"][None]
  Nf = sp_data[None]["Nf"][None]
  new_row = np.zeros((Nn, Nf))
  for f in range(Nf):
    if omega_row[f] <= 0:
      continue
    gamma_if = sp_data[None]["gamma"][(i + 1, f + 1)]
    score = {}
    for j in neighbours:
      b = sp_data[None]["beta"][(i + 1, j + 1, f + 1)]
      if b > -gamma_if and ledger[j, f] > 0:
        score[j] = b
    placed = 0.0
    for j in sorted(score, key=lambda k: (-score[k], k)):
      if placed >= omega_row[f]:
        break
      q = min(ledger[j, f], omega_row[f] - placed)
      if q <= 0:
        continue
      new_row[j, f] += q
      ledger[j, f] -= q
      placed += q
  return new_row


def propose_node_move(
    node: int,
    omega_ub_row: np.array,
    y: np.array,
    sp_data: dict,
    model,
    solver_name: str,
    general_solver_options: dict,
    parallelism: int,
  ) -> Tuple[np.array, np.array, np.array, float]:
  """MILP proposal for one node: re-solve the local problem with committed
  inbound flows (y_bar) and the accessible-capacity cap (omega_ub).

  The proposal prices aggregate offloading with delta (as P2 does); the
  caller's epsilon-acceptance on the true beta-aware utility is what
  guarantees potential monotonicity, not proposal optimality."""
  Nn = sp_data[None]["Nn"][None]
  Nf = sp_data[None]["Nf"][None]
  node_data = deepcopy(sp_data)
  node_data[None]["omega_ub"] = {
    (f + 1): float(omega_ub_row[f]) for f in range(Nf)
  }
  node_data[None]["y_bar"] = {
    (m + 1, n + 1, f + 1): float(max(y[m, n, f], 0.0))
    for m in range(Nn) for n in range(Nn) for f in range(Nf)
  }
  result = solve_subproblem(
    node_data, [node], model, solver_name, general_solver_options, parallelism
  )
  x_row = np.array(result[1][node, :], dtype=float)
  omega_row = np.array(result[4][node, :], dtype=float)
  r_row = np.array(result[5][node, :], dtype=float)
  runtime = float(result[10]["tot"])
  return x_row, r_row, omega_row, runtime


def node_move(
    i: int,
    x: np.array,
    y: np.array,
    r: np.array,
    sp_data: dict,
    neighborhood: np.array,
    rho: np.array,
    epsilon: float,
    propose_fn: Callable,
    tolerance: float,
  ) -> Tuple[bool, float, dict, float]:
  """One better-response move. Rules that keep Phi an exact potential:
  the mover keeps serving committed inbound flows (enforced by LSP_pg via
  y_bar) and only claims advertised residual capacity of its neighbours.
  The move is committed iff the TRUE utility (per-pair beta) improves by
  more than epsilon; x, y, r are mutated in place only on acceptance."""
  Nn = sp_data[None]["Nn"][None]
  Nf = sp_data[None]["Nf"][None]
  memory_bids = {"i": [], "j": [], "f": []}
  neighbours = set(int(j) for j in np.nonzero(neighborhood[i, :])[0])
  z = compute_z(x, y, sp_data)
  u_old = compute_node_utility(i, x, y, z, sp_data)
  # residual capacity visible to i, with its own row released
  y_trial = np.array(y, dtype=float)
  y_trial[i, :, :] = 0.0
  _, ledger, _ = compute_residual_capacity(x, y_trial, r, sp_data)
  omega_ub_row = np.zeros(Nf)
  for f in range(Nf):
    gamma_if = sp_data[None]["gamma"][(i + 1, f + 1)]
    omega_ub_row[f] = sum(
      ledger[j, f] for j in neighbours
      if sp_data[None]["beta"][(i + 1, j + 1, f + 1)] > -gamma_if
    )
  x_row, r_row, omega_row, runtime = propose_fn(i, omega_ub_row)
  new_row = split_omega(i, omega_row, ledger, neighbours, sp_data)
  # candidate state (copies: commit only on acceptance)
  x_new = np.array(x, dtype=float)
  y_new = y_trial
  x_new[i, :] = x_row
  y_new[i, :, :] = new_row
  z_new = compute_z(x_new, y_new, sp_data)
  u_new = compute_node_utility(i, x_new, y_new, z_new, sp_data)
  delta_u = u_new - u_old
  accepted = delta_u > epsilon
  if accepted:
    x[i, :] = x_row
    y[i, :, :] = new_row
    r[i, :] = r_row
  # unplaced appetite signals a capacity shortage: bid on neighbour memory
  placed = new_row.sum(axis=0)
  for f in range(Nf):
    if omega_row[f] - placed[f] <= tolerance:
      continue
    gamma_if = sp_data[None]["gamma"][(i + 1, f + 1)]
    memory_requirement = sp_data[None]["memory_requirement"][f + 1]
    for j in sorted(neighbours):
      if (
          rho[j] >= memory_requirement
          and sp_data[None]["beta"][(i + 1, j + 1, f + 1)] > -gamma_if
        ):
        memory_bids["i"].append(i)
        memory_bids["j"].append(j)
        memory_bids["f"].append(f)
  return accepted, float(delta_u), memory_bids, runtime


def potential_game_sweep(
    x: np.array,
    y: np.array,
    r: np.array,
    sp_data: dict,
    neighborhood: np.array,
    rho: np.array,
    epsilon: float,
    tolerance: float,
    order: str,
    rng: np.random.Generator,
    propose_fn: Callable,
  ) -> Tuple[int, float, pd.DataFrame, float]:
  """Gauss-Seidel sweep of better-response moves. Every accepted move raises
  the exact potential Phi by more than epsilon, so sweeps terminate."""
  Nn = sp_data[None]["Nn"][None]
  if order == "random":
    node_order = [int(i) for i in rng.permutation(Nn)]
  elif order == "fixed":
    node_order = list(range(Nn))
  else:
    raise ValueError("order must be one of: fixed, random")
  n_accepted = 0
  delta_phi = 0.0
  proposal_runtime = 0.0
  all_bids = {"i": [], "j": [], "f": []}
  for i in node_order:
    accepted, delta_u, bids, runtime = node_move(
      i, x, y, r, sp_data, neighborhood, rho, epsilon, propose_fn, tolerance
    )
    proposal_runtime += runtime
    if accepted:
      n_accepted += 1
      delta_phi += delta_u
    for k in all_bids:
      all_bids[k].extend(bids[k])
  return n_accepted, delta_phi, pd.DataFrame(all_bids), proposal_runtime


def check_pg_stopping(
    it: int,
    max_iterations: int,
    n_accepted: int,
    n_new_replicas: int,
    total_runtime: float,
    time_limit: float,
  ) -> Tuple[bool, str]:
  if it >= max_iterations - 1:
    return True, "max iterations reached"
  if total_runtime >= time_limit:
    return True, f"reached time limit: {total_runtime} >= {time_limit}"
  if n_accepted == 0 and n_new_replicas == 0:
    return True, "epsilon-Nash equilibrium certified"
  return False, None


def parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Run FaaS-MAPG",
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
    "--variant", choices=["s", "r"], default="s",
    help="FaaS-MAPG variant: s (fixed order), r (randomized order)",
  )
  return parser.parse_known_args()[0]


def _run(
    config: dict,
    parallelism: int,
    *,
    order: str,
    method_name: str,
    options_key: str,
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
  pg_options = dict(solver_options.get(options_key, {}))
  epsilon = pg_options.get("epsilon", 1e-6)
  rng = np.random.default_rng(seed)
  time_limit = general_solver_options.get("TimeLimit", np.inf)
  tolerance = config.get("tolerance", 1e-6)
  max_iterations = config["max_iterations"]
  max_steps = config["max_steps"]
  min_run_time = config.get("min_run_time", 0)
  max_run_time = config.get("max_run_time", max_steps)
  run_time_step = config.get("run_time_step", 1)
  checkpoint_interval = config["checkpoint_interval"]
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
  ub = (
    max_run_time + run_time_step
  ) if max_run_time == min_run_time else max_run_time
  sp_complete_solution = init_complete_solution()
  spc_complete_solution = init_complete_solution()
  obj_dict = {"phi_final": []}
  tc_dict = {"pg": []}
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
    pg_model = LSP_pg() if opt_solution is None else LSP_pg_fixedr()
    (
      sp_data, sp_x, _, _, _, sp_r, sp_rho, _, _, _, sp_runtime
    ) = solve_subproblem(
      sp_data, agents, sp, solver_name, general_solver_options, parallelism
    )
    total_runtime += sp_runtime["tot"]
    x = np.array(sp_x, dtype=float)
    r = np.array(sp_r, dtype=float)
    rho = np.array(
      sp_rho if opt_solution is None else np.zeros_like(sp_rho), dtype=float
    )
    y = np.zeros((Nn, Nn, Nf))

    def propose_fn(node, omega_ub_row):
      return propose_node_move(
        node, omega_ub_row, y, sp_data, pg_model,
        solver_name, general_solver_options, parallelism,
      )

    it = 0
    stop_searching = False
    phi_prev = compute_centralized_objective(
      sp_data, x, y, compute_z(x, y, sp_data)
    )
    while not stop_searching:
      s = datetime.now()
      n_accepted, delta_phi, memory_bids, proposal_runtime = (
        potential_game_sweep(
          x, y, r, sp_data, neighborhood, rho, epsilon, tolerance,
          order, rng, propose_fn,
        )
      )
      elapsed = (datetime.now() - s).total_seconds()
      bookkeeping = max(0.0, elapsed - proposal_runtime)
      total_runtime += proposal_runtime + (
        bookkeeping / n_accepted if n_accepted else bookkeeping
      )
      additional_replicas = np.zeros((Nn, Nf))
      if len(memory_bids) > 0 and (rho > 0).any():
        s = datetime.now()
        additional_replicas, rho = start_additional_replicas(
          memory_bids, r, sp_data, rho
        )
        r += additional_replicas
        total_runtime += (datetime.now() - s).total_seconds()
      phi = compute_centralized_objective(
        sp_data, x, y, compute_z(x, y, sp_data)
      )
      assert phi >= phi_prev - 1e-9, (
        f"potential decreased: {phi_prev} -> {phi}"
      )
      phi_prev = phi
      stop_searching, why_stop_searching = check_pg_stopping(
        it, max_iterations, n_accepted,
        int(additional_replicas.sum()), total_runtime, time_limit,
      )
      if not stop_searching:
        it += 1
    csol = combine_solutions(
      Nn, Nf, sp_data, loadt, x, r, rho,
      None, y, None, None, None, None
    )
    feas = check_feasibility(
      csol["sp"]["x"], csol["sp"]["y"].sum(axis=1), csol["sp"]["z"],
      csol["sp"]["r"], csol["sp"]["U"], sp_data
    )
    assert feas[0], feas[1]
    sp_complete_solution, _, objf = decode_solutions(
      sp_data, csol, sp_complete_solution, None
    )
    spc_complete_solution, _, _ = decode_solutions(
      sp_data, csol, spc_complete_solution, None
    )
    obj_dict["phi_final"].append(objf)
    tc_dict["pg"].append(
      f"{why_stop_searching} "
      f"(it: {it}; phi: {phi}; total runtime: {total_runtime})"
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
  spc_solution, spc_offloaded, spc_detailed_fwd_solution = (
    join_complete_solution(spc_complete_solution)
  )
  if not disable_plotting and Nf <= 10 and Nn <= 10:
    plot_history(
      input_requests_traces, min_run_time, max_run_time, run_time_step,
      sp_solution, sp_complete_solution["utilization"],
      sp_complete_solution["replicas"], sp_offloaded,
      obj_dict["phi_final"], os.path.join(solution_folder, "sp.png")
    )
  save_solution(
    sp_solution, sp_offloaded, sp_complete_solution,
    sp_detailed_fwd_solution, "LSP", solution_folder
  )
  save_solution(
    spc_solution, spc_offloaded, spc_complete_solution,
    spc_detailed_fwd_solution, "LSPc", solution_folder
  )
  pd.DataFrame(obj_dict["phi_final"], columns=[method_name]).to_csv(
    os.path.join(solution_folder, "obj.csv"), index=False
  )
  pd.DataFrame(tc_dict["pg"]).to_csv(
    os.path.join(solution_folder, "termination_condition.csv")
  )
  pd.DataFrame({"tot": runtime_list}).to_csv(
    os.path.join(solution_folder, "runtime.csv"), index=False
  )
  if verbose > 0:
    print(
      f"All solutions saved in: {solution_folder}",
      file=log_stream, flush=True
    )
  if log_on_file:
    log_stream.close()
  return solution_folder


def run_pg_s(config, parallelism, log_on_file=False, disable_plotting=False):
  return _run(config, parallelism, order="fixed",
              method_name="FaaS-MAPG-S", options_key="pg_s",
              log_on_file=log_on_file, disable_plotting=disable_plotting)


def run_pg_r(config, parallelism, log_on_file=False, disable_plotting=False):
  return _run(config, parallelism, order="random",
              method_name="FaaS-MAPG-R", options_key="pg_r",
              log_on_file=log_on_file, disable_plotting=disable_plotting)


if __name__ == "__main__":
  args = parse_arguments()
  config = load_configuration(args.config)
  runner = {"s": run_pg_s, "r": run_pg_r}[args.variant]
  runner(config, args.parallelism, disable_plotting=args.disable_plotting)
