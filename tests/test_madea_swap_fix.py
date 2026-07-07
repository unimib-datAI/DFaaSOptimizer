"""Regression for the swap-branch over-subtraction in run_faasmadea.evaluate_bids.

When a seller j is full and a higher bid displaces a previous buyer, the swap
loop removed the full bid quantity `d` from the incumbent instead of capping it
at what the incumbent actually holds (`max_to_remove`). The last bid overshot,
driving the accumulated y negative (and violating j's capacity).
"""

import numpy as np
import pandas as pd

from run_faasmadea import evaluate_bids


def _base_data(Nn=4, Nf=1):
  data = {None: {
    "Nn": {None: Nn}, "Nf": {None: Nf},
    "beta": {}, "gamma": {}, "demand": {},
    "memory_requirement": {f + 1: 2 for f in range(Nf)},
    "max_utilization": {f + 1: 0.8 for f in range(Nf)},
  }}
  for i in range(Nn):
    for f in range(Nf):
      data[None]["gamma"][(i + 1, f + 1)] = 0.05
      data[None]["demand"][(i + 1, f + 1)] = 1.0
      for j in range(Nn):
        data[None]["beta"][(i + 1, j + 1, f + 1)] = 1.0
  return data


def test_swap_does_not_oversubtract_incumbent():
  data = _base_data(Nn=4)
  # seller 1 hosts capacity 2; incumbent buyer 3 currently holds 1 unit there
  last_y = np.zeros((4, 4, 1)); last_y[3, 1, 0] = 1.0
  blackboard = np.zeros((4, 1)); blackboard[1, 0] = 2.0
  # buyer 2 fills the capacity in the main loop; buyer 0 overflows with d=5 > 1
  bids = pd.DataFrame({
    "i": [2, 0], "j": [1, 1], "f": [0, 0], "d": [2.0, 5.0], "b": [10.0, 5.0],
  })
  capacity = np.ones((4, 1)) * 10.0
  y, _, _, _ = evaluate_bids(
    bids, blackboard, data, last_y,
    np.zeros((4, 1)), np.zeros((4, 1)), capacity, np.zeros((4, 1)),
    {"eta": 0.0, "zeta": 0.0},
  )
  accumulated = last_y + y
  assert (accumulated >= -1e-9).all(), accumulated[3, 1, 0]
  # incumbent can lose at most what it held (1); the displacer gains that much
  assert accumulated[3, 1, 0] >= 0.0
  assert y[0, 1, 0] <= last_y[3, 1, 0] + 1e-9
  # the swap runs at exhausted residual, so this round's net placement at
  # seller 1 stays within the advertised residual capacity
  assert y[:, 1, 0].sum() <= blackboard[1, 0] + 1e-9
