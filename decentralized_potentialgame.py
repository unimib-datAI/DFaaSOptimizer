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
