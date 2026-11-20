from dataclasses import replace
from itertools import product
from time import monotonic
import numpy as np
import pytest

from centralized_vns_frontier.lagrangian import EndpointBound, certify_regions
from centralized_vns_frontier.proof import Ledger
from centralized_vns_frontier.regions import Regions
from centralized_vns_frontier.tests.test_frontier import disjoint
from centralized_vns_frontier.tests.test_receiver_bound import shared
from centralized_vns_frontier.tests.test_pair_cuts import receiver


def enumerate_bounds(p, regions):
  exact = np.full(regions.count, -np.inf)
  for y in product(*(range(v + 1) for v in (p.load - p.x)[p.src, p.fun])):
    y = np.array(y)
    try:
      value = p.check(y)['welfare']
    except ValueError:
      continue
    b = regions.identify(y)
    exact[b] = max(exact[b], value)
  return exact


@pytest.mark.parametrize(
  'p,regions',
  [
    (shared(), Regions([0], [0])),
    (disjoint(), Regions([0, 1], [0, 0])),
    (
      replace(receiver(12, [1.5, 2 - 5e-10, 2]), memory=[6, 6, 6], beta=[2, 2, 2]),
      Regions([2], [1]),
    ),
  ],
)
def test_endpoint_upper_bounds_and_gradient(p, regions):
  exact = enumerate_bounds(p, regions)
  ref = EndpointBound(p, regions)
  rng = np.random.default_rng(34768)
  for _ in range(6):
    price = p.reward * rng.random(len(p.src))
    bounds, root, g = ref.evaluate(price)
    assert np.all(bounds >= exact - 1e-12) and root >= bounds.max() - 1e-12
    other = p.reward * rng.random(len(p.src))
    _, new_root, _ = ref.evaluate(other)
    assert new_root >= root + np.dot(g, other - price) - 1e-5
  book = Ledger(p, regions, np.zeros(len(p.src), dtype=np.int64), 1e3, 0.0, monotonic())
  report = certify_regions(book, [p.reward, np.zeros_like(price)], monotonic() + 10, 'cpu')
  assert report['evaluations'] and np.all(book.upper >= exact - 1e-12)


def test_forced_traffic_changes_only_its_region():
  p = replace(receiver(12, [2] * 3), beta=[2, 2, 0])
  bounds, _, _ = EndpointBound(p, Regions([2], [0])).evaluate(p.reward)
  np.testing.assert_allclose(bounds, [4, 2], atol=1e-10, rtol=0)


def test_optional_bound_handles_limits_and_expiry():
  p = shared()
  with pytest.raises(MemoryError):
    EndpointBound(p, Regions([], []), max_bytes=1)
  assert EndpointBound(p, Regions([], [])).evaluate(p.reward, deadline=0) is None


def test_cuda_kernels_match_numpy_exactly():
  cp = pytest.importorskip('cupy')
  if not cp.cuda.is_available():
    pytest.skip('CUDA device required')
  p = shared()
  partition = Regions([0], [0])
  price = p.reward * 0.7
  expected = EndpointBound(p, partition).evaluate(price)
  actual = EndpointBound(p, partition, 'cuda').evaluate(price)
  for a, b in zip(actual, expected):
    np.testing.assert_array_equal(cp.asnumpy(a), b)


@pytest.mark.parametrize('groups,dual', [(True, False), (False, True), (True, True)])
def test_options_work_independently_through_solver(groups, dual):
  from centralized_vns.core import Options
  from centralized_vns_frontier.core import solve

  p = replace(receiver(12, [2, 2, 2]), beta=[2, 2, 2], memory=[6, 6, 6])
  r = solve(
    p,
    Options(iterations=0, time_limit=5),
    branches=2,
    group_cuts=groups,
    lagrangian=dual,
    relative_gap=0,
  )
  assert r['group_cuts'] == groups and r['lagrangian'] == dual
  assert r['coordinator_upper'] == pytest.approx(4.0)
  assert r['coordinator_objective'] == pytest.approx(4.0)


def test_local_dp_matches_independent_node_enumeration():
  p = shared()
  groups = Regions([0], [0])
  oracle = EndpointBound(p, groups)
  price = np.array([0.25, 0.75, 0.5, 0.1])
  ub, root, _ = oracle.evaluate(price)
  q = np.rint(np.clip(price, 0, p.reward) * oracle.scale) / oracle.scale
  w = np.ceil(p.reward * oracle.scale) / oracle.scale
  # Enumerate each node's incident copies independently, including edge caps.
  maxima = []
  for branch in (None, 0, 1):
    value = p.base_welfare
    for n in range(len(p.x)):
      outgoing = p.src == n
      incoming = p.dst == n
      incident = np.flatnonzero(outgoing | incoming)
      best = -np.inf
      for traffic in product(
        *(range(p.load[p.src[e], p.fun[e]] - p.x[p.src[e], p.fun[e]] + 1) for e in incident)
      ):
        y = np.zeros(len(p.src), dtype=np.int64)
        y[incident] = traffic
        if branch is not None and 0 in incident and groups.identify(y) != branch:
          continue
        # A single-edge network witness checks the shared upper bound of that arc.
        try:
          for e in incident:
            single = np.zeros_like(y)
            single[e] = y[e]
            p.check(single)
        except ValueError:
          continue
        sent = np.bincount(p.fun[outgoing], weights=y[outgoing], minlength=p.x.shape[1])
        got = np.bincount(p.fun[incoming], weights=y[incoming], minlength=p.x.shape[1])
        work = p.x[n] + got
        replicas = np.maximum(work > 0, np.ceil(work / p.capacity[n] - 1e-9))
        if (
          np.any(sent * got) or np.any(sent > p.load[n] - p.x[n]) or replicas @ p.memory > p.ram[n]
        ):
          continue
        best = max(best, (w - q)[outgoing] @ y[outgoing] + q[incoming] @ y[incoming])
      value += best
    maxima.append(value)
  np.testing.assert_allclose([root, *ub], maxima, atol=1e-10, rtol=0)


def test_invalid_prices_cannot_generate_a_certificate():
  p = shared()
  oracle = EndpointBound(p, Regions([], []))
  for values in (np.array([0.0]), np.full_like(p.reward, np.nan)):
    with pytest.raises(ValueError, match='finite price'):
      oracle.evaluate(values)
