import numpy as np
import pandas as pd

from decentralized_gcaa import resolve_gcaa_round


def _bids(rows):
  return pd.DataFrame(rows, columns=["i", "j", "f", "d", "b", "utility"])


def test_empty_bids_returns_zero_allocation():
  residual_capacity = np.ones((2, 1))
  y_round = resolve_gcaa_round(_bids([]), residual_capacity)
  assert y_round.shape == (2, 2, 1)
  assert (y_round == 0).all()


def test_single_winner_per_contested_task():
  # agents 0 and 1 both target seller 1 / function 0; agent 1 has the
  # higher utility and must be the sole winner this round
  bids = _bids([
    {"i": 0, "j": 1, "f": 0, "d": 1, "b": 0.5, "utility": 1.0},
    {"i": 1, "j": 1, "f": 0, "d": 1, "b": 0.7, "utility": 2.0},
  ])
  residual_capacity = np.array([[0.0], [5.0]])
  y_round = resolve_gcaa_round(bids, residual_capacity)
  assert y_round[1, 1, 0] == 1.0
  assert y_round[0, 1, 0] == 0.0
  assert y_round.sum() == 1.0


def test_agent_proposes_only_its_best_task():
  # agent 0 has bid rows for two different sellers; only the
  # higher-utility one (seller 2) should be proposed this round
  bids = _bids([
    {"i": 0, "j": 1, "f": 0, "d": 1, "b": 0.1, "utility": 0.5},
    {"i": 0, "j": 2, "f": 0, "d": 1, "b": 0.9, "utility": 3.0},
  ])
  residual_capacity = np.array([[5.0], [5.0], [5.0]])
  y_round = resolve_gcaa_round(bids, residual_capacity)
  assert y_round[0, 2, 0] == 1.0
  assert y_round[0, 1, 0] == 0.0


def test_seller_with_no_residual_capacity_is_skipped():
  bids = _bids([
    {"i": 0, "j": 1, "f": 0, "d": 1, "b": 0.5, "utility": 1.0},
  ])
  residual_capacity = np.array([[5.0], [0.0]])
  y_round = resolve_gcaa_round(bids, residual_capacity)
  assert y_round.sum() == 0.0


def test_bid_larger_than_residual_capacity_is_skipped():
  bids = _bids([
    {"i": 0, "j": 1, "f": 0, "d": 1, "b": 0.5, "utility": 1.0},
  ])
  residual_capacity = np.array([[5.0], [0.4]])
  y_round = resolve_gcaa_round(bids, residual_capacity)
  assert y_round.sum() == 0.0
