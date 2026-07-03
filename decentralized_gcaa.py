import numpy as np
import pandas as pd


def resolve_gcaa_round(
    bids: pd.DataFrame, residual_capacity: np.array
  ) -> np.array:
  """One GCAA consensus round (Braquet & Bakolas, 2021 - Algorithms 1+3):
  each buyer agent (i, f) proposes only its highest-utility task (j, f);
  for each contested task, the single highest-utility proposal wins and
  consumes one unit of residual_capacity[j, f]. Losers are absent from
  the returned allocation and simply re-propose next round via a fresh
  call to define_bids (their omega is left untouched by this function)."""
  Nn, Nf = residual_capacity.shape
  y_round = np.zeros((Nn, Nn, Nf))
  if len(bids) == 0:
    return y_round
  best_per_agent = bids.loc[bids.groupby(["i", "f"])["utility"].idxmax()]
  for (j, f), group in best_per_agent.groupby(["j", "f"]):
    j, f = int(j), int(f)
    if residual_capacity[j, f] <= 0:
      continue
    winner = group.loc[group["utility"].idxmax()]
    y_round[int(winner["i"]), j, f] += winner["d"]
  return y_round
