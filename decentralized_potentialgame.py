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
