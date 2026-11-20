from dataclasses import replace
from itertools import product
from time import monotonic

import numpy as np
import pytest

from centralized_vns.core import Options, Problem
from centralized_vns_highs.core import Exchange, solve
from centralized_vns_highs.pool import Pool


def small():
  return Problem(load=[[3], [2], [2]], x=[[1], [0], [0]], capacity=[[2], [1.5], [2]],
                 memory=[2], ram=[2, 2, 4], alpha=[[1], [0], [1]], gamma=[[2], [5], [4]],
                 edges=[(0, 1, 0), (1, 2, 0), (2, 1, 0), (2, 0, 0)], beta=[4, 2, 1, 7])


@pytest.mark.parametrize('huge_capacity', [False, True])
def test_model_optimum_matches_exhaustive_forwarding(huge_capacity):
  p = small()
  if huge_capacity:
    p = replace(p, ram=[2, 0, 4], capacity=np.full((3, 1), 1e12))
  feasible = []
  for y in product(*(range(1 + p.load[i, f] - p.x[i, f]) for i, _, f in p.edges)):
    try:
      feasible.append(p.check(y)['welfare'])
    except ValueError:
      continue
  result = solve(p, Options(iterations=0), relative_gap=0)
  assert result['welfare'] == pytest.approx(max(feasible))
  assert result['upper_bound'] == pytest.approx(max(feasible))
  assert all(t['upper'] >= max(feasible) - 1e-8 for t in result['trace'])
  assert all(t['lower'] <= max(feasible) + 1e-8 for t in result['trace'])
  np.testing.assert_array_equal(result['x'], p.x)
  assert result['welfare'] == p.check(result['y'])['welfare']


def test_gap_and_time_stop_preserve_checked_warm_start():
  p = small()
  y = np.array([0, 2, 0, 0])
  for budget, target, reason in [(5., 100., 'gap'), (1e-10, 0., 'time_limit')]:
    result = solve(p, Options(time_limit=budget), y, relative_gap=target)
    assert result['termination_reason'] == reason
    assert result['welfare'] == p.check(y)['welfare']
    assert result['upper_bound'] >= result['welfare']
  with pytest.raises(ValueError, match='flows'):
    solve(p, initial_y=y + 0.1)


def test_infeasible_point_and_stale_record_cannot_replace_incumbent():
  p = small()
  y = np.zeros(len(p.src), np.int64)
  exchange = Exchange(p, y, 0., monotonic())
  good = np.array([0, 2, 0, 0])
  exchange.offer('highs', good)
  exchange.offer('vns', y)
  with pytest.raises(ValueError):
    exchange.offer('highs', np.full_like(y, 4))
  good[:] = 0
  assert exchange.latest().source == 'highs'
  assert exchange.latest().y[1] == 2


@pytest.mark.parametrize('backend', ['cpu', 'cuda'])
def test_reseed_only_one_trajectory_and_resume(backend):
  if backend == 'cuda':
    pytest.importorskip("numba")
    from numba import cuda
    if not cuda.is_available():
      pytest.skip('CUDA or NUMBA_ENABLE_CUDASIM=1 required')
  p = small()
  o = Options(pool_size=2, candidates=8, backend=backend)
  y = np.zeros(len(p.src), np.int64)
  pool = Pool(p, o, y, p.check(y), 2)
  seed = np.array([0, 2, 0, 0])
  checked = p.check(seed)
  pool.inject(seed, checked)
  engine = pool.engine
  archive = engine.best_y.copy_to_host() if pool.cuda else engine.best_y
  np.testing.assert_array_equal(archive, np.stack([seed, y]))
  best = pool.advance(0, 2)
  assert best >= checked['welfare'] - 1e-12
  assert p.check(engine.result())['welfare'] == pytest.approx(best)


def test_negative_optimum_uses_full_welfare_for_gap():
  p = Problem(load=[[2], [0]], x=[[0], [0]], capacity=[[1], [1]], memory=[1],
              ram=[0, 1], alpha=[[0], [0]], gamma=[[6], [0]], edges=[(0, 1, 0)], beta=[0])
  result = solve(p, Options(iterations=0), relative_gap=0)
  assert result['initial_welfare'] == -6
  assert result['welfare'] == result['upper_bound'] == -3
  assert result['relative_gap'] == 0
