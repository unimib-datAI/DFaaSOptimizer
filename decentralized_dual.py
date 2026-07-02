from typing import Tuple

import numpy as np
import pandas as pd


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
