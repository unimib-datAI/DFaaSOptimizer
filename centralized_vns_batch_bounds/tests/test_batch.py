from itertools import product
import numpy as np
import pytest
from centralized_vns_frontier.tests.test_frontier import disjoint
from centralized_vns_batch_bounds.lp import Relaxation
from centralized_vns_batch_bounds.solvers import highs, metal


def test_bound_and_both_prefix_sides():
  p = disjoint()
  lp = Relaxation(p)
  lo, hi = lp.boxes([0, 1], [0, 1], [(1, 0), (2, 1), (2, 3)])
  r = highs(lp, lo, hi, 10)
  rng = np.random.default_rng(7)
  for i in range(len(lo)):
    optimum = max(
      p.check(np.array(y))['welfare']
      for y in product(range(3), repeat=2)
      if np.all(lo[i, :2] <= y) and np.all(y <= hi[i, :2])
    )
    assert r['bounds'][i] == pytest.approx(optimum)
    assert lp.certificate(rng.normal(size=len(lp.b)), lo[i], hi[i]) >= optimum
  with pytest.raises(ValueError):
    lp.certificate(np.full_like(lp.b, np.nan), lo[0], hi[0])


def test_metal_mixed_depth_and_partial_threadgroup():
  mx = pytest.importorskip('mlx.core')
  if not mx.metal.is_available():
    pytest.skip('requires Metal')
  lp = Relaxation(disjoint())
  lo, hi = lp.boxes([0, 1], [0, 1], [(1, 0), (2, 1), (2, 3)])
  cpu = highs(lp, lo, hi, 15)
  gpu = metal(lp, lo, hi, 15, 65)
  assert gpu['iterations'] == 65
  for a, b in zip(gpu['bounds'], cpu['lp_optima']):
    assert a >= b - 1e-10 and np.isfinite(a)


def test_metal_tightens_both_children_with_signed_objective():
  mx = pytest.importorskip('mlx.core')
  if not mx.metal.is_available():
    pytest.skip('requires Metal')
  from centralized_vns_highs.tests.test_hybrid import small

  lp = Relaxation(small())
  lo, hi = lp.boxes([0], [0], [(1, 0), (1, 1)])
  initial = [lp.certificate(-np.asarray(lp.root.row_dual), l, u) for l, u in zip(lo, hi)]
  opt = highs(lp, lo, hi, 10)['lp_optima']
  result = metal(lp, lo, hi, 20, 512)['bounds']
  for start, bound, optimal in zip(initial, result, opt):
    assert optimal - 1e-8 <= bound < start - 0.1
