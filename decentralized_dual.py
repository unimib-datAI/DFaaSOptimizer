from typing import Tuple

import numpy as np
import pandas as pd

from decentralized_diffusion import evaluate_assignments


def pair_scores(
  data: dict,
  neighborhood: np.array,
  latency: np.array,
  fairness: np.array,
  dual_options: dict,
) -> Tuple[np.array, np.array]:
  """Return unpriced beta-minus-penalty scores and their eligibility mask.

  Only neighboring pairs whose score beats rejection (``score > -gamma``)
  are eligible; all other score entries are ``-inf``.
  """
  values = data[None]
  nn = values["Nn"][None]
  nf = values["Nf"][None]
  scores = np.full((nn, nn, nf), -np.inf)
  eligible = np.zeros((nn, nn, nf), dtype=bool)

  for i in range(nn):
    for j in range(nn):
      for f in range(nf):
        score = (
          values["beta"][(i + 1, j + 1, f + 1)]
          - dual_options["latency_weight"] * latency[i, j]
          - dual_options["fairness_weight"] * fairness[i, f]
        )
        if neighborhood[i, j] and score > -values["gamma"][(i + 1, f + 1)]:
          scores[i, j, f] = score
          eligible[i, j, f] = True

  return scores, eligible


def buyer_price_response(
  omega: np.array,
  capacity: np.array,
  lam: np.array,
  s: np.array,
  elig: np.array,
) -> Tuple[pd.DataFrame, np.array, float]:
  """Return uncapped best-seller demand and capacity-capped waterfill bids.

  Demand and the buyer dual term use the best strictly positive price-adjusted
  score with lower seller indices breaking ties; bids rank all positive sellers
  while exposing the unpriced utility.
  """
  nn, nf = omega.shape
  demand = np.zeros((nn, nf))
  dual_term = 0.0
  rows = []

  for i, f in zip(*np.nonzero(omega > 0)):
    adjusted = np.where(elig[i, :, f], s[i, :, f] - lam[:, f], -np.inf)
    positive = np.flatnonzero(adjusted > 0)
    if not len(positive):
      continue

    ranked = positive[np.argsort(-adjusted[positive], kind="stable")]
    best = ranked[0]
    demand[best, f] += omega[i, f]
    dual_term += omega[i, f] * adjusted[best]

    remaining = omega[i, f]
    for j in ranked:
      quantity = min(remaining, capacity[j, f])
      if quantity > 0:
        rows.append({"i": i, "j": j, "f": f, "d": quantity, "utility": s[i, j, f]})
        remaining -= quantity
      if remaining <= 0:
        break

  bids = pd.DataFrame(rows, columns=["i", "j", "f", "d", "utility"])
  return bids, demand, dual_term


def dual_coordination_round(
  omega: np.array,
  residual_capacity: np.array,
  data: dict,
  neighborhood: np.array,
  rho: np.array,
  dual_options: dict,
  latency: np.array,
  fairness: np.array,
  force_memory_bids: bool,
  ell: np.array,
  r: np.array,
  ) -> Tuple[np.array, np.array, pd.DataFrame, dict, int]:
  """Projected dual subgradient loop with primal recovery and certificate."""
  if int(dual_options["max_inner_iterations"]) < 1:
    raise ValueError("max_inner_iterations must be at least 1")
  if dual_options["step_rule"] not in {"sqrt", "polyak"}:
    raise ValueError("step_rule must be 'sqrt' or 'polyak'")
  nn = data[None]["Nn"][None]
  nf = data[None]["Nf"][None]
  scores, eligible = pair_scores(
    data, neighborhood, latency, fairness, dual_options
  )
  capacity = np.array(residual_capacity, dtype=float)
  lam = np.zeros((nn, nf))
  score_values = np.where(eligible, scores, 0.0)
  best_lb, best_ub = 0.0, np.inf
  best_bids = pd.DataFrame(
    {"i": [], "j": [], "f": [], "d": [], "utility": []}
  )
  lb_history = []
  n_active = int((omega > 0).any(axis=1).sum())
  gap = 0.0
  k = 0

  for k in range(1, int(dual_options["max_inner_iterations"]) + 1):
    bids, demand, buyer_term = buyer_price_response(
      omega, capacity, lam, scores, eligible
    )
    best_ub = min(best_ub, float((lam * capacity).sum() + buyer_term))
    if len(bids) > 0:
      candidate_y, _, _ = evaluate_assignments(
        bids, residual_capacity, data, ell, r, rho,
        tentatively_start_replicas=False, last_y=None,
        diffusion_options=dual_options, latency=latency, fairness=fairness,
      )
      candidate_lb = float((score_values * candidate_y).sum())
      if candidate_lb > best_lb:
        best_lb, best_bids = candidate_lb, bids
    lb_history.append(best_lb)
    gap = (best_ub - best_lb) / max(1.0, abs(best_ub))
    if gap <= dual_options["gap_tolerance"]:
      break
    if demand.sum() <= 0 and len(bids) == 0:
      break
    subgradient = demand - capacity
    if dual_options["step_rule"] == "polyak":
      denominator = float((subgradient * subgradient).sum())
      alpha = (
        0.0 if denominator <= 0.0
        else dual_options["theta"] * max(0.0, best_ub - best_lb) / denominator
      )
    else:
      alpha = dual_options["alpha0"] / np.sqrt(k)
    lam = np.maximum(0.0, lam + alpha * subgradient)

  memory_bids = {"i": [], "j": [], "f": []}
  placed = (
    np.zeros((nn, nf)) if len(best_bids) == 0
    else evaluate_assignments(
      best_bids, residual_capacity, data, ell, r, rho,
      tentatively_start_replicas=False, last_y=None,
      diffusion_options=dual_options, latency=latency, fairness=fairness,
    )[0].sum(axis=1)
  )
  for i, f in zip(*np.nonzero(omega > 0)):
    i, f = int(i), int(f)
    if placed[i, f] < omega[i, f] or force_memory_bids:
      memory_requirement = data[None]["memory_requirement"][f + 1]
      for j in np.nonzero(neighborhood[i, :])[0]:
        if rho[int(j)] >= memory_requirement:
          memory_bids["i"].append(i)
          memory_bids["j"].append(int(j))
          memory_bids["f"].append(f)
  memory_bids = pd.DataFrame(memory_bids)

  y_increment = np.zeros((nn, nn, nf))
  additional_replicas = np.zeros((nn, nf))
  if len(best_bids) > 0:
    y_increment, additional_replicas, _ = evaluate_assignments(
      best_bids, residual_capacity, data, ell, r, rho,
      tentatively_start_replicas=(len(memory_bids) == 0), last_y=None,
      diffusion_options=dual_options, latency=latency, fairness=fairness,
    )
  gap_info = {
    "LB": best_lb, "UB": best_ub, "gap": gap,
    "inner_iterations": k, "lam": lam, "lb_history": lb_history,
  }
  return y_increment, additional_replicas, memory_bids, gap_info, n_active
