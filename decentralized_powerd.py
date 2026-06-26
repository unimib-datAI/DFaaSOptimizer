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
  VAR_TYPE,
  check_ls_pr_feasibility_from_fixed_y,
  check_stopping_criteria,
  compute_residual_capacity,
  ensure_memory_sellers,
  neigh_dict_to_matrix,
  start_additional_replicas,
)
from decentralized_diffusion import evaluate_assignments
from utils.centralized import check_feasibility
from utils.faasmacro import compute_centralized_objective
from utils.common import load_configuration
from models.sp import LSP, LSP_fixedr, LSPr

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


def sample_assignments(
    omega: np.array,
    blackboard: np.array,
    data: dict,
    neighborhood: np.array,
    rho: np.array,
    powerd_options: dict,
    latency: np.array,
    fairness: np.array,
    force_memory_bids: bool,
    rng: np.random.Generator,
  ) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
  """Power-of-d-choices counterpart of decentralized_diffusion.define_assignments.

  For each overloaded (i, f), instead of scanning the whole candidate
  neighbourhood greedily (FaaS-MADiG), repeatedly probe a random sample of ``d``
  candidate sellers (uniform, without replacement) and pick the best by
  ``criterion``: "score" -> max s_{ij}^f = beta - w_lat*L - w_fair*phi;
  "capacity" -> max advertised residual capacity. Ties break on the lower node
  id for reproducibility. The score is always stored in the ``utility`` column,
  on which the reused evaluate_assignments later sorts the seller side.
  """
  d = int(powerd_options["d"])
  criterion = powerd_options.get("criterion", "score")
  unit_bids = powerd_options.get("unit_bids", False)
  potential_buyers, functions_to_share = np.nonzero(omega)
  bids = {"i": [], "j": [], "f": [], "d": [], "utility": []}
  memory_bids = {"i": [], "j": [], "f": []}
  for i, f in zip(potential_buyers, functions_to_share):
    i = int(i)
    f = int(f)
    potential_sellers = set(np.nonzero(neighborhood[i, :])[0])
    potential_capacity_sellers = potential_sellers.intersection(
      set(np.where(blackboard[:, f] >= 1)[0])
    )
    potential_memory_sellers = potential_sellers.intersection(
      set(np.nonzero(rho)[0])
    )
    score = {}
    candidates = []
    for j in potential_capacity_sellers:
      j = int(j)
      s = (
        data[None]["beta"][(i + 1, j + 1, f + 1)]
        - powerd_options["latency_weight"] * latency[i, j]
        - powerd_options["fairness_weight"] * fairness[i, f]
      )
      if s > - data[None]["gamma"][(i + 1, f + 1)]:
        score[j] = s
        candidates.append(j)
    # buyer-local view of advertised capacity, decremented as we bid, so we
    # never request more from a seller than the buyer has observed locally
    remaining = {j: int(blackboard[j, f]) for j in candidates}
    assigned = 0
    while assigned < omega[i, f] and len(candidates) > 0:
      sample_size = min(d, len(candidates))
      sample = rng.choice(candidates, size=sample_size, replace=False)
      if criterion == "capacity":
        j_star = int(max(sample, key=lambda j: (remaining[int(j)], -int(j))))
      else:
        j_star = int(max(sample, key=lambda j: (score[int(j)], -int(j))))
      if unit_bids:
        q = 1
      else:
        q = VAR_TYPE(min(remaining[j_star], omega[i, f] - assigned))
      bids["i"].append(i)
      bids["f"].append(f)
      bids["j"].append(j_star)
      bids["d"].append(q)
      bids["utility"].append(score[j_star])
      assigned += q
      remaining[j_star] -= q
      if remaining[j_star] < 1:
        candidates.remove(j_star)
    if assigned < omega[i, f] or force_memory_bids:
      for j in potential_memory_sellers - potential_capacity_sellers:
        memory_bids["i"].append(i)
        memory_bids["j"].append(int(j))
        memory_bids["f"].append(f)
  return pd.DataFrame(bids), pd.DataFrame(memory_bids), len(potential_buyers)
