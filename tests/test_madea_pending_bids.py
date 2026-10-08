import numpy as np
import pandas as pd
import pytest

from run_faasmadea import evaluate_bids


@pytest.mark.parametrize("other_spare", [0., 1.])
@pytest.mark.parametrize("replicas", [False, True])
def test_saturated_function_is_processed_despite_other_spare(other_spare, replicas):
  data = {None: {
    "Nn": {None: 3}, "Nf": {None: 2}, "memory_requirement": {1: 1, 2: 1},
    "demand": {(2, 1): 1.}, "max_utilization": {1: 1.},
  }}
  previous = np.zeros((3, 3, 2))
  previous[2, 1, 0] = 1.
  board = np.zeros((3, 2)); board[1] = [1., 1.]
  residual = np.zeros((3, 2)); residual[1, 1] = other_spare
  prices = np.zeros((3, 2)); prices[1, 0] = 1.
  r = np.zeros((3, 2)); r[1, 0] = 1.
  bids = pd.DataFrame([dict(i=0, j=1, f=0, d=1., b=2.)])
  delta, _, added, _ = evaluate_bids(
    bids, board, data, previous_y=previous, p=prices,
    residual_capacity=residual, total_capacity=np.ones((3, 2)),
    ell=previous.sum(axis=0), r=r,
    initial_rho=np.array([0., float(replicas), 0.]),
    tentatively_start_replicas=replicas,
  )
  assert delta[0, 1, 0] == 1.
  assert delta[2, 1, 0] == (0. if replicas else -1.)
  assert added[1, 0] == float(replicas)


@pytest.mark.parametrize("functions", [[0, 0, 0], [0, 1, 0]])
def test_reassignment_evaluates_each_bid_and_its_function(functions):
  data = {None: {"Nn": {None: 5}, "Nf": {None: 2}}}
  previous = np.zeros((5, 5, 2))
  previous[2, 1] = [functions.count(0), functions.count(1)]
  board = previous.sum(axis=0)
  bids = pd.DataFrame(dict(i=[0, 3, 4], j=[1]*3, f=functions, d=[1.]*3, b=[4., 3., 2.]))
  delta, _, _, _ = evaluate_bids(
    bids, board, data, previous_y=previous, p=np.ones((5, 2)),
    total_capacity=np.ones((5, 2))*3, residual_capacity=np.zeros((5, 2)),
  )
  np.testing.assert_array_equal(delta[[0, 3, 4], 1, functions], [1., 1., 1.])
  np.testing.assert_array_equal((previous + delta)[2, 1], [0., 0.])
  assert (previous + delta >= 0).all()


@pytest.mark.parametrize("replace", [False, True])
def test_partial_bid_uses_only_remainder_across_incumbents(replace):
  data = {None: {"Nn": {None: 4}, "Nf": {None: 1}}}
  previous = np.zeros((4, 4, 1)); previous[2:, 1, 0] = 1.
  residual = np.zeros((4, 1)); residual[1, 0] = 1.
  board = residual + previous.sum(axis=0) if replace else residual
  bids = pd.DataFrame([dict(i=0, j=1, f=0, d=3., b=2.)])
  original = bids.copy(deep=True)
  delta, _, _, _ = evaluate_bids(
    bids, board, data, previous_y=previous, p=np.ones((4, 1)),
    total_capacity=np.ones((4, 1))*3, residual_capacity=residual,
    may_replace_existing_assignments=replace,
  )
  assert delta[0, 1, 0] == (3. if replace else 1.)
  assert (previous + delta)[:, 1, 0].sum() == 3.
  assert (previous + delta >= 0).all()
  pd.testing.assert_frame_equal(bids, original)
