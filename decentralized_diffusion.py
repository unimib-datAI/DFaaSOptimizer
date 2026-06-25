from run_centralized_model import (
  get_current_load,
  init_complete_solution,
  init_problem,
  join_complete_solution,
  plot_history,
  save_checkpoint,
  save_solution,
  update_data,
)
from run_faasmacro import (
  combine_solutions,
  compute_social_welfare,
  decode_solutions,
  solve_subproblem,
)
from run_faasmadea import (
  VAR_TYPE,
  check_ls_pr_feasibility_from_fixed_y,
  check_stopping_criteria,
  compute_residual_capacity,
  ensure_memory_sellers,
  neigh_dict_to_matrix,
  start_additional_replicas,
)
from utils.centralized import check_feasibility
from utils.faasmacro import compute_centralized_objective
from utils.common import load_configuration
from models.sp import LSP, LSPr

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


def define_assignments(
    omega: np.array,
    blackboard: np.array,
    data: dict,
    neighborhood: np.array,
    rho: np.array,
    diffusion_options: dict,
    latency: np.array,
    fairness: np.array,
    force_memory_bids: bool,
  ) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
  """Price-free counterpart of run_faasmadea.define_bids.

  Greedy-by-utility assignment of residual load to neighbours with spare
  capacity. No bid price is computed (no epsilon/delta); the per-pair score is
  stored in the ``utility`` column, on which evaluate_assignments later sorts.
  """
  potential_buyers, functions_to_share = np.nonzero(omega)
  bids = {"i": [], "j": [], "f": [], "d": [], "utility": []}
  memory_bids = {"i": [], "j": [], "f": []}
  for i, f in zip(potential_buyers, functions_to_share):
    potential_sellers = set(np.nonzero(neighborhood[i, :])[0])
    potential_capacity_sellers = potential_sellers.intersection(
      set(np.where(blackboard[:, f] >= 1)[0])
    )
    potential_memory_sellers = potential_sellers.intersection(
      set(np.nonzero(rho)[0])
    )
    utility = []
    candidate_sellers = []
    for j in potential_capacity_sellers:
      ut = (
        data[None]["beta"][(i + 1, j + 1, f + 1)]
        - diffusion_options["latency_weight"] * latency[i, j]
        - diffusion_options["fairness_weight"] * fairness[i, f]
      )
      if ut > - data[None]["gamma"][(i + 1, f + 1)]:
        utility.append(ut)
        candidate_sellers.append(j)
    assigned = 0
    if len(utility) > 0:
      utility = np.array(utility)
      sellers_order = np.argsort(utility)[::-1]
      idx = 0
      while idx < len(sellers_order) and assigned < omega[i, f]:
        j = candidate_sellers[sellers_order[idx]]
        if diffusion_options.get("unit_bids", False):
          d = 1
          while (d < int(min(blackboard[j, f], omega[i, f])) + 1) and (
              assigned < omega[i, f]
            ):
            bids["i"].append(i)
            bids["f"].append(f)
            bids["j"].append(j)
            bids["d"].append(1)
            bids["utility"].append(utility[sellers_order[idx]])
            assigned += 1
            d += 1
        else:
          d = VAR_TYPE(min(blackboard[j, f], (omega[i, f] - assigned)))
          bids["i"].append(i)
          bids["f"].append(f)
          bids["j"].append(j)
          bids["d"].append(d)
          bids["utility"].append(utility[sellers_order[idx]])
          assigned += d
        idx += 1
      if assigned < omega[i, f]:
        for idx in sellers_order:
          j = candidate_sellers[idx]
          if j in potential_memory_sellers:
            memory_bids["i"].append(i)
            memory_bids["j"].append(j)
            memory_bids["f"].append(f)
    if assigned < omega[i, f] or force_memory_bids:
      for j in potential_memory_sellers - potential_capacity_sellers:
        memory_bids["i"].append(i)
        memory_bids["j"].append(j)
        memory_bids["f"].append(f)
  return pd.DataFrame(bids), pd.DataFrame(memory_bids), len(potential_buyers)


def evaluate_assignments(
    bids: pd.DataFrame,
    residual_capacity: np.array,
    data: dict,
    ell: np.array,
    r: np.array,
    initial_rho: np.array,
    tentatively_start_replicas: bool,
  ) -> Tuple[np.array, np.array, int]:
  """Price-free counterpart of run_faasmadea.evaluate_bids.

  Pure greedy capacity fill: sellers serve buyers by descending ``utility``
  (tie-break on buyer index ``i`` for reproducibility). No min_b tracking, no
  price update, no last_y replacement swap. The price-free
  tentatively_start_replicas branch is kept for parity with the baseline.
  """
  Nn = data[None]["Nn"][None]
  Nf = data[None]["Nf"][None]
  potential_sellers, functions_to_share = np.nonzero(residual_capacity)
  if tentatively_start_replicas:
    potential_sellers, functions_to_share = ensure_memory_sellers(
      potential_sellers, functions_to_share, np.nonzero(initial_rho)[0], Nf
    )
  y = np.zeros((Nn, Nn, Nf))
  additional_replicas = np.zeros((Nn, Nf))
  rho = deepcopy(initial_rho)
  for j, f in zip(potential_sellers, functions_to_share):
    j = int(j)
    f = int(f)
    bids_for_j = bids[(bids["j"] == j) & (bids["f"] == f)].sort_values(
      by=["utility", "i"], ascending=[False, True]
    )
    remaining_capacity = int(residual_capacity[j, f])
    next_bid_idx = 0
    while next_bid_idx < len(bids_for_j) and remaining_capacity > 0:
      q = min(remaining_capacity, bids_for_j.iloc[next_bid_idx]["d"])
      y[int(bids_for_j.iloc[next_bid_idx]["i"]), j, f] += q
      remaining_capacity -= q
      next_bid_idx += 1
    # price-free tentative replica start (decides on utilization, not price)
    if (
        remaining_capacity == 0
        and (next_bid_idx > 0 or len(bids_for_j) > 0)
        and tentatively_start_replicas
      ):
      max_a = int(rho[j] / data[None]["memory_requirement"][f + 1])
      if max_a > 0:
        a = 1
        while next_bid_idx < len(bids_for_j) and a <= max_a:
          q = bids_for_j.iloc[next_bid_idx]["d"]
          u = data[None]["demand"][(j + 1, f + 1)] * (
            ell[j, f] + y[:, j, f].sum() + q
          ) / (r[j, f] + a)
          if u <= data[None]["max_utilization"][f + 1]:
            y[int(bids_for_j.iloc[next_bid_idx]["i"]), j, f] += q
            next_bid_idx += 1
            additional_replicas[j, f] = a
            rho[j] -= (a * data[None]["memory_requirement"][f + 1])
          else:
            a += 1
  return y, additional_replicas, len(potential_sellers)
