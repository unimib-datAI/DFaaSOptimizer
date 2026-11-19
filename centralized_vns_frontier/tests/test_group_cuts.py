from dataclasses import replace
from itertools import product
from time import monotonic

import numpy as np

from centralized_vns_frontier.pair_cuts import separate
from centralized_vns_frontier.tests.test_pair_cuts import receiver


def test_joint_ram_and_branch_bounds_preserve_every_feasible_flow():
  p = replace(receiver(12, [2, 2, 2]), memory=[6, 6, 6])
  upper = np.full(3, 2)
  for lower in (np.zeros(3, dtype=int), np.array([1, 0, 0])):
    rows = separate(p, upper, lower, upper, monotonic() + 5, groups=True)
    assert len(rows) == 1 and rows[0][1] == 4
    for y in product(*(range(l, h + 1) for l, h in zip(lower, upper))):
      y = np.asarray(y)
      try:
        p.check(y)
      except ValueError:
        continue
      assert y[rows[0][0]].sum() <= rows[0][1]
