import numpy as np
import pandas as pd

from decentralized_dual import buyer_price_response, pair_scores


def make_data(Nn=3, Nf=1, beta=None, gamma=0.05):
  data = {None: {
    "Nn": {None: Nn}, "Nf": {None: Nf},
    "beta": {}, "gamma": {},
    "memory_requirement": {f + 1: 2 for f in range(Nf)},
    "demand": {(j + 1, f + 1): 1.0 for j in range(Nn) for f in range(Nf)},
    "max_utilization": {f + 1: 0.7 for f in range(Nf)},
  }}
  for i in range(Nn):
    for f in range(Nf):
      data[None]["gamma"][(i + 1, f + 1)] = gamma
      for j in range(Nn):
        b = 1.0 if beta is None else beta[i][j][f]
        data[None]["beta"][(i + 1, j + 1, f + 1)] = b
  return data


def full_neighborhood(Nn):
  return np.ones((Nn, Nn)) - np.eye(Nn)


DUAL_OPTIONS = {"latency_weight": 0.0, "fairness_weight": 0.0}


def test_pair_scores_masks_non_neighbors_and_dominated_pairs():
  Nn, Nf = 3, 1
  beta = [[[1.0] for _ in range(Nn)] for _ in range(Nn)]
  beta[0][2][0] = -1.0
  data = make_data(Nn, Nf, beta=beta)
  neighborhood = full_neighborhood(Nn)
  neighborhood[0, 1] = 0
  s, elig = pair_scores(
    data, neighborhood, np.zeros((Nn, Nn)), np.zeros((Nn, Nf)), DUAL_OPTIONS
  )
  assert not elig[0, 1, 0] and not elig[0, 2, 0] and not elig[0, 0, 0]
  assert elig[1, 0, 0] and s[1, 0, 0] == 1.0
  assert s[0, 1, 0] == -np.inf


def test_buyer_response_zero_prices_picks_best_score_seller():
  Nn, Nf = 3, 1
  beta = [[[1.0] for _ in range(Nn)] for _ in range(Nn)]
  beta[0][1][0] = 2.0
  data = make_data(Nn, Nf, beta=beta)
  s, elig = pair_scores(
    data, full_neighborhood(Nn), np.zeros((Nn, Nn)), np.zeros((Nn, Nf)),
    DUAL_OPTIONS,
  )
  omega = np.zeros((Nn, Nf)); omega[0, 0] = 5.0
  capacity = np.full((Nn, Nf), 3.0)
  bids, demand, dual_term = buyer_price_response(
    omega, capacity, np.zeros((Nn, Nf)), s, elig
  )
  assert demand[1, 0] == 5.0 and demand[2, 0] == 0.0
  assert dual_term == 5.0 * 2.0
  first = bids.iloc[0]
  assert (first.j, first.d) == (1, 3.0)
  assert bids["d"].sum() == 5.0


def test_buyer_response_price_shifts_demand():
  Nn, Nf = 3, 1
  beta = [[[1.0] for _ in range(Nn)] for _ in range(Nn)]
  beta[0][1][0] = 2.0
  data = make_data(Nn, Nf, beta=beta)
  s, elig = pair_scores(
    data, full_neighborhood(Nn), np.zeros((Nn, Nn)), np.zeros((Nn, Nf)),
    DUAL_OPTIONS,
  )
  omega = np.zeros((Nn, Nf)); omega[0, 0] = 5.0
  lam = np.zeros((Nn, Nf)); lam[1, 0] = 1.5
  bids, demand, dual_term = buyer_price_response(
    omega, np.full((Nn, Nf), 10.0), lam, s, elig
  )
  assert demand[2, 0] == 5.0 and demand[1, 0] == 0.0
  assert dual_term == 5.0 * 1.0


def test_buyer_response_tie_uses_lower_seller_index():
  Nn, Nf = 3, 1
  data = make_data(Nn, Nf)
  s, elig = pair_scores(
    data, full_neighborhood(Nn), np.zeros((Nn, Nn)), np.zeros((Nn, Nf)),
    DUAL_OPTIONS,
  )
  omega = np.zeros((Nn, Nf)); omega[0, 0] = 5.0
  bids, demand, _ = buyer_price_response(
    omega, np.full((Nn, Nf), 3.0), np.zeros((Nn, Nf)), s, elig
  )
  assert demand[1, 0] == 5.0 and demand[2, 0] == 0.0
  assert bids["j"].tolist() == [1, 2]


def test_buyer_response_all_priced_out_yields_empty():
  Nn, Nf = 2, 1
  data = make_data(Nn, Nf)
  s, elig = pair_scores(
    data, full_neighborhood(Nn), np.zeros((Nn, Nn)), np.zeros((Nn, Nf)),
    DUAL_OPTIONS,
  )
  omega = np.zeros((Nn, Nf)); omega[0, 0] = 4.0
  lam = np.full((Nn, Nf), 10.0)
  bids, demand, dual_term = buyer_price_response(
    omega, np.full((Nn, Nf), 10.0), lam, s, elig
  )
  assert len(bids) == 0 and demand.sum() == 0.0 and dual_term == 0.0
