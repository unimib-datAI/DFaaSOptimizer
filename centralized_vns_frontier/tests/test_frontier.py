from dataclasses import replace
from itertools import product
from time import monotonic

import numpy as np
import pytest

from centralized_vns.core import Options, Problem
from centralized_vns_highs.tests.test_hybrid import small
from centralized_vns_frontier.core import redistribute, solve
from centralized_vns_frontier.pool import Pool, allowed
from centralized_vns_frontier.proof import Ledger, certify_branch
from centralized_vns_frontier.regions import Regions
from centralized_vns_frontier.relink import cross_regions, relink


def disjoint():
  return Problem(
    load=[[2], [0], [2], [0]],
    x=[[0]] * 4,
    capacity=[[2]] * 4,
    memory=[1],
    ram=[1] * 4,
    alpha=[[0]] * 4,
    gamma=[[0]] * 4,
    edges=[(0, 1, 0), (2, 3, 0)],
    beta=[2, 2],
  )


def test_each_bound_covers_exactly_its_region():
  p = small()
  groups = Regions([1], [1])
  zero = np.zeros(len(p.src), np.int64)
  ledger = Ledger(p, groups, zero, 100.0, 0, monotonic())
  optima = np.full(groups.count, -np.inf)
  for y in product(range(3), repeat=len(zero)):
    y = np.array(y)
    try:
      value = p.check(y)['welfare']
    except ValueError:
      continue
    b = groups.identify(y)
    optima[b] = max(optima[b], value)
    for branch in range(2):
      seed = groups.seed(p, branch, y)
      p.check(seed)
      assert groups.identify(seed) == branch
  for b in range(2):
    certify_branch(ledger, b, monotonic() + 5.0)
  np.testing.assert_allclose(ledger.upper, optima)
  assert ledger.closed.all()


def test_record_closes_another_region_at_fixed_bound():
  p = disjoint()
  groups = Regions([0], [0])
  book = Ledger(p, groups, np.array([0, 0]), 4.0, 0.0, monotonic())
  book.tighten(2.0, 0)
  assert not book.closed.any()
  book.offer('path_relinking', np.array([2, 2]), 1)
  assert book.closed.all()
  assert book.closures[0]['cause'] == 'path_relinking'
  assert book.closures[0]['upper'] == 2
  with pytest.raises(RuntimeError, match='crossed'):
    book.offer('vns', np.array([1, 0]), 0)


def test_compound_move_uses_net_change():
  moves = np.array([[[0, 1], [0, -1]], [[0, 1], [-1, 0]]])
  assert allowed(moves, np.array([0]), np.array([0]), np.array([0]), 0).tolist() == [True, False]


def test_cross_region_final_child_enters_third_region_with_global_certificate():
  p = disjoint()
  groups = Regions([0, 1], [0, 0])
  book = Ledger(p, groups, np.array([0, 0]), 4.0, 0.0, monotonic())
  book.offer('vns', np.array([2, 0]), 1)
  book.offer('vns', np.array([0, 2]), 2)
  bounds = book.upper.copy()
  stage, parents = cross_regions(book, 2)
  assert stage['pairs'] == stage['improved_children'] == stage['better_than_pre_final'] == 1
  assert stage['winning_pair'] == dict(parents=[1, 2], child_region=3)
  np.testing.assert_array_equal(parents, [[2, 0], [0, 2]])
  np.testing.assert_array_equal(book.best.y, [2, 2])
  np.testing.assert_equal(book.upper, bounds)
  assert book.closed.all()
  assert all(c['cause'] == 'final_path_relinking' for c in book.closures)


def test_final_stage_preserves_only_parent_with_exhausted_search_budget():
  p = small()
  y = np.array([0, 2, 0, 0])
  r = solve(p, Options(time_limit=1e-10), y, final_relink_top=32)
  assert r['final_relink']['pairs'] == 0
  assert r['final_relink']['winning_pair'] is None
  np.testing.assert_equal(r['pre_final_y'], r['y'])
  np.testing.assert_equal(r['final_parent_y'], [y])


def test_relink_keeps_flows_competing_for_receiver_ram_together():
  p = Problem(
    load=[[2, 0], [0, 2], [0, 0], [2, 0], [0, 0]],
    x=[[0, 0]] * 5,
    capacity=[[2, 2]] * 5,
    memory=[1, 1],
    ram=[1] * 5,
    alpha=[[0, 0]] * 5,
    gamma=[[0, 0]] * 5,
    edges=[(0, 2, 0), (1, 2, 1), (3, 4, 0)],
    beta=[2, 4, 2],
  )
  a, g = np.array([2, 0, 2]), np.array([0, 2, 0])
  child = relink(p, a, g)
  np.testing.assert_array_equal(child, [0, 2, 2])
  assert p.check(child)['welfare'] > max(p.check(y)['welfare'] for y in (a, g))
  with pytest.raises(ValueError, match='memory'):
    p.check(np.maximum(a, g))


@pytest.mark.parametrize('backend', ['cpu', 'metal', 'cuda'])
def test_population_and_path_stay_in_region(backend):
  if backend == 'metal':
    pytest.importorskip('mlx.core', exc_type=ImportError)
  if backend == 'cuda':
    pytest.importorskip('numba')
    from numba import cuda

    if not cuda.is_available():
      pytest.skip('CUDA device or simulator required')
  p = disjoint()
  groups = Regions([0], [1])
  pool = Pool(p, Options(backend=backend, pool_size=3, candidates=4), groups)
  zero = np.zeros(2, dtype=np.int64)
  book = Ledger(p, groups, zero, 4.0, 0.0, monotonic())
  for slot, b in enumerate(pool.branches):
    r = book.witness(b)
    pool.seed(slot, r.y, r.checked)
  for i in range(2):
    pool.engine.step(i)
    current = pool.engine.y.copy_to_host() if backend == 'cuda' else pool.engine.state.y
    np.testing.assert_array_equal(groups.identify(current), pool.branches)
    for slot, b in enumerate(pool.branches):
      assert groups.identify(pool.record(slot)) == b
  book.tighten(3.0, 0)
  book.offer('vns', np.array([2, 1]), 1)
  redistribute(pool, book)
  assert np.all(pool.branches == 1)
  pool.engine.step(2)
  for i in range(3):
    assert groups.identify(pool.record(i)) == 1
  a, g = np.array([1, 0]), np.array([0, 2])
  child = relink(p, a, g)
  np.testing.assert_array_equal(child, [1, 2])
  if backend == 'cuda':
    children = pool.engine.relink(np.stack([a, g]), np.stack([g, a]))
    np.testing.assert_array_equal(children, np.stack([child, child]))
    separate = Ledger(p, Regions([0, 1], [0, 0]), a, 4.0, 0.0, monotonic())
    separate.offer('vns', g)
    stage, _ = cross_regions(separate, 2, pool.engine.relink)
    assert stage['backend'] == 'cuda' and stage['winning_pair']['child_region'] == 3
    np.testing.assert_array_equal(separate.best.y, child)


def test_solver_reports_gap_on_coordinator_and_preserves_warm_start():
  p = replace(small(), alpha=np.full((3, 1), 1000.0))
  warm = np.array([0, 2, 0, 0])
  for seconds in (3.0, 1e-10):
    result = solve(p, Options(pool_size=4, candidates=8, time_limit=seconds), warm, relative_gap=0)
    assert p.check(result['y'])['welfare'] >= p.check(warm)['welfare']
    np.testing.assert_array_equal(result['x'], p.x)
    assert result['coordinator_objective'] + result['welfare_offset'] == pytest.approx(
      result['welfare']
    )
    if seconds > 1:
      assert result['coordinator_gap'] < 1e-12
      assert len(result['branch_solves']) <= result['branches']
