from itertools import product
from time import monotonic

import numpy as np
import pytest

from centralized_vns.core import Problem
from centralized_vns_frontier.pair_cuts import separate
from centralized_vns_frontier.regions import partition


def receiver(ram, capacity):
  return Problem(
    load=[[2, 2, 2], [0, 0, 0]],
    x=np.zeros((2, 3)),
    capacity=[capacity] * 2,
    memory=[6, 6, 4],
    ram=[0, ram],
    alpha=np.zeros((2, 3)),
    gamma=np.zeros((2, 3)),
    edges=[(0, 1, f) for f in range(3)],
    beta=[2, 2, 0],
  )


@pytest.mark.parametrize(
  'ram,capacity', [(10, [2, 2, 2]), (12, [2, 2, 2]), (14, [1.5, 2 - 1e-10, 1])]
)
def test_global_and_branch_cuts_preserve_all_integer_flows(ram, capacity):
  p = receiver(ram, capacity)
  for lower in (np.zeros(3, dtype=int), np.array([0, 0, 1])):
    upper = np.array([2, 2, 2])
    rows = separate(p, upper, lower, upper, monotonic() + 10)
    if ram == 12:
      pair = [v for edges, v in rows if edges.tolist() == [0, 1]]
      assert pair == ([2] if lower[2] else [])
    for flow in product(*(range(l, h + 1) for l, h in zip(lower, upper))):
      y = np.asarray(flow)
      try:
        p.check(y)
      except ValueError:
        continue
      assert all(y[edges].sum() <= limit for edges, limit in rows)


def test_cuts_strictly_improve_lp_and_keep_partition():
  p = receiver(10, [2, 2, 2])
  original, before = partition(p, 2, monotonic() + 10, tighten_receive=True)
  rows = []
  new, after = partition(p, 2, monotonic() + 10, tighten_receive=True, pair_rows=rows)
  assert rows and before > after and after == pytest.approx(2)
  np.testing.assert_array_equal(original.edges, new.edges)
  np.testing.assert_array_equal(original.cuts, new.cuts)


def test_branch_cuts_charge_fixed_work_and_combine_senders():
  from centralized_vns_frontier.tests.test_receiver_bound import shared

  p = shared()
  upper = (p.load - p.x)[p.src, p.fun]
  lower = np.zeros_like(upper)
  rows = separate(p, upper, lower, upper, monotonic() + 10)
  pair = np.flatnonzero(p.dst == 1)
  # Local x already uses one replica of both functions. The spare two units
  # buy just one more replica of function 0, including its fractional slack.
  assert any(np.array_equal(edges, pair) and limit == 2 for edges, limit in rows)
  for flow in product(*(range(h + 1) for h in upper)):
    y = np.array(flow)
    try:
      p.check(y)
    except ValueError:
      continue
    assert all(y[edges].sum() <= limit for edges, limit in rows)


def test_branch_proof_applies_conditional_cut_only_in_its_region():
  from centralized_vns_frontier.proof import Ledger, certify_branch
  from centralized_vns_frontier.regions import Regions

  p = receiver(12, [2, 2, 2])
  book = Ledger(p, Regions([2], [0]), np.zeros(3, dtype=int), 6.0, 0.0, monotonic())
  result = certify_branch(book, 1, monotonic() + 5, pair_rows=[])
  assert result['pair_cuts'] >= 1
  assert book.upper.tolist() == pytest.approx([6.0, 2.0])
  certify_branch(book, 0, monotonic() + 5, pair_rows=[])
  assert book.upper.tolist() == pytest.approx([4.0, 2.0])
