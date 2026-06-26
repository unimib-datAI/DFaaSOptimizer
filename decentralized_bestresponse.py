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
  compute_social_welfare,
  decode_solutions,
  solve_subproblem,
)
from run_faasmadea import (
  check_ls_pr_feasibility_from_fixed_y,
  check_stopping_criteria,
  compute_residual_capacity,
  neigh_dict_to_matrix,
  start_additional_replicas,
)
from utils.centralized import check_feasibility
from utils.faasmacro import compute_centralized_objective
from utils.common import load_configuration
from models.sp import LSP, LSP_fixedr, LSPr, LSP_capped, LSP_capped_fixedr

from networkx import adjacency_matrix
from collections import deque
from datetime import datetime
from copy import deepcopy
from typing import Tuple
import pandas as pd
import numpy as np
import argparse
import json
import sys
import os


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
  ) -> Tuple[np.array, pd.DataFrame, int, float, float]:
  """Sequential (Gauss-Seidel) best-response sweep.

  Nodes act in `order` ("fixed" = ascending index, "random" = rng permutation);
  each places its residual demand greedily by descending score onto the shared
  `ledger` (a copy of residual_capacity), decremented in place so later nodes
  see what earlier nodes took. For response=="reopt", each node first caps its
  omega row at the accessible neighbour capacity via reopt_fn before placing.
  Replica-expansion memory bids use the FaaS-MADiG two-block emission.
  """
  Nn = data[None]["Nn"][None]
  Nf = data[None]["Nf"][None]
  ledger = np.array(residual_capacity, dtype=float)
  omega = np.array(omega, dtype=float)  # working copy; never mutate the caller's
  y_increment = np.zeros((Nn, Nn, Nf))
  memory_bids = {"i": [], "j": [], "f": []}
  reopt_runtime = 0.0
  active = set()
  memory_seller_nodes = set(int(j) for j in np.nonzero(rho)[0])
  if order == "random":
    node_order = [int(i) for i in rng.permutation(Nn)]
  else:
    node_order = list(range(Nn))
  for i in node_order:
    neighbours = set(int(j) for j in np.nonzero(neighborhood[i, :])[0])
    if response == "reopt" and reopt_fn is not None and np.any(omega[i, :] > 0):
      omega_ub_row = np.array(
        [sum(ledger[j, f] for j in neighbours) for f in range(Nf)]
      )
      capped_row, rt = reopt_fn(i, omega_ub_row)
      reopt_runtime += rt
      omega[i, :] = capped_row
    potential_memory = neighbours & memory_seller_nodes
    for f in range(Nf):
      if omega[i, f] <= 0:
        continue
      score = {}
      candidates = []
      for j in neighbours:
        if ledger[j, f] >= 1:
          s = (
            data[None]["beta"][(i + 1, j + 1, f + 1)]
            - br_options["latency_weight"] * latency[i, j]
            - br_options["fairness_weight"] * fairness[i, f]
          )
          if s > - data[None]["gamma"][(i + 1, f + 1)]:
            score[j] = s
            candidates.append(j)
      placed = 0.0
      for j in sorted(candidates, key=lambda k: (-score[k], k)):
        if placed >= omega[i, f]:
          break
        q = min(ledger[j, f], omega[i, f] - placed)
        if q <= 0:
          continue
        y_increment[i, j, f] += q
        ledger[j, f] -= q
        placed += q
        active.add(i)
      potential_capacity = set(
        j for j in neighbours if residual_capacity[j, f] >= 1
      )
      if placed < omega[i, f]:
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
  placed_total = float(y_increment.sum())
  return (
    y_increment, pd.DataFrame(memory_bids), len(active),
    placed_total, reopt_runtime,
  )
