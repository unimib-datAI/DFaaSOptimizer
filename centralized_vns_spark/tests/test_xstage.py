from dataclasses import replace

import numpy as np
import pytest

from centralized_vns.core import Problem
from centralized_vns_r.tests.test_xstage import paired
from centralized_vns_r.xstage import _proposals, _room, local_gains
from centralized_vns_spark.xstage import refine_x


@pytest.fixture
def cuda_available():
  pytest.importorskip('numba')
  from numba import cuda
  if not cuda.is_available():
    pytest.skip('CUDA or NUMBA_ENABLE_CUDASIM=1 required')


def scalar_best(p, y):
  """Independent old CPU neighborhood/checker oracle, with deterministic best gain."""
  floor = local_gains(p, p.x)
  checked = p.check(y)
  best = checked['welfare']
  for x, flow, node in _proposals(p, p.x, y, checked, floor,
                                 np.random.default_rng(0), lambda: True):
    if np.any(local_gains(p, x) < floor - 1e-12):
      continue
    try:
      trial_p = replace(p, x=x)
      trial = trial_p.check(flow)
    except ValueError:
      continue
    for edge in sorted(np.flatnonzero(p.dst == node), key=lambda e: -p.reward[e]):
      src, fun = p.src[edge], p.fun[edge]
      if p.reward[edge] <= 0 or trial['incoming'][src, fun] > 0:
        continue
      if p.load[node, fun] - x[node, fun] - trial['z'][node, fun] > 0:
        continue
      amount = min(int(trial['z'][src, fun]),
                   _room(p, x, trial['incoming'], node, fun))
      if amount <= 0:
        continue
      repaired = flow.copy()
      repaired[edge] += amount
      try:
        repaired_check = trial_p.check(repaired)
      except ValueError:
        continue
      flow, trial = repaired, repaired_check
    if np.isfinite(trial['welfare']):
      best = max(best, trial['welfare'])
  return best


def check_result(p, result):
  checked = replace(p, x=result['x']).check(result['y'])
  assert result['welfare'] == checked['welfare']
  for key in ('incoming', 'z', 'replicas'):
    np.testing.assert_array_equal(result[key], checked[key])
  assert np.all(result['local_gain'] >= result['local_gain_floor'] - 1e-12)
  values = [row['best_welfare'] for row in result['trace']]
  assert values == sorted(values)
  assert result['accepted'] <= result['attempted']


def test_paired_swap_is_best_feasible_and_preserves_input(cuda_available):
  p = paired()
  result = refine_x(p, [], iterations=100, time_limit=60)
  assert result['algorithm'] == 'VNS-RX'
  assert result['backend'] == 'cuda'
  assert result['execution']['variant'] == 'parallel-best-improvement'
  assert result['x'].tolist() == [[0, 4]]
  assert result['welfare'] == scalar_best(p, np.zeros(0, dtype=np.int64)) == 2
  assert p.x.tolist() == [[2, 0]]
  assert result['local_gain_floor'].tolist() == [1]
  assert result['attempted'] <= 100
  check_result(p, result)


def test_global_gain_does_not_override_local_floor(cuda_available):
  p = paired(alpha=(2, 1), gamma=(0, 100))
  result = refine_x(p, [], iterations=100, time_limit=60)
  assert result['x'].tolist() == [[2, 0]]
  assert result['accepted'] == 0
  check_result(p, result)


def test_receiver_repair_and_localization_preserve_roles(cuda_available):
  p = Problem(load=[[1,1],[0,2],[0,0]], x=[[1,0],[0,0],[0,0]],
    capacity=np.full((3,2), 2), memory=[2,1], ram=[2,0,1],
    alpha=np.zeros((3,2)), gamma=np.zeros((3,2)),
    edges=[(0,2,1),(1,0,1)], beta=[1,10])
  y = np.array([1,0])
  result = refine_x(p, y, iterations=500, time_limit=60)
  assert result['x'].tolist() == [[0,1],[0,0],[0,0]]
  assert result['y'].tolist() == [0,2]
  assert result['welfare'] == scalar_best(p, y) == 10
  assert y.tolist() == [1,0]
  check_result(p, result)


def test_trimming_prefers_low_reward_and_keeps_other_node_floor(cuda_available):
  p = Problem(load=[[4],[0],[0]], x=[[0],[0],[0]], capacity=[[2],[4],[4]],
    memory=[1], ram=[1,1,1], alpha=[[8],[0],[0]], gamma=np.zeros((3,1)),
    edges=[(0,1,0),(0,2,0)], beta=[1,3])
  result = refine_x(p, [2,2], iterations=100, time_limit=60)
  assert result['x'].tolist() == [[2],[0],[0]]
  assert result['y'].tolist() == [0,2]
  assert result['welfare'] == 5.5
  check_result(p, result)


@pytest.mark.parametrize('loss,accepted', [(5e-13, True), (2e-12, False)])
def test_float64_floor_tolerance(loss, accepted, cuda_available):
  p = Problem(load=[[1,1]], x=[[1,0]], capacity=[[1,1]], memory=[1,1],
    ram=[1], alpha=[[1,1-loss]], gamma=[[0,1]], edges=[], beta=[])
  result = refine_x(p, [], iterations=100, time_limit=60)
  assert bool(result['accepted']) == accepted
  check_result(p, result)


@pytest.mark.parametrize('gain,accepted', [(5e-13, False), (2e-12, True)])
def test_float64_strict_welfare_improvement(gain, accepted, cuda_available):
  p = Problem(load=[[1]], x=[[0]], capacity=[[1]], memory=[1], ram=[1],
    alpha=[[gain]], gamma=[[0]], edges=[], beta=[])
  result = refine_x(p, [], iterations=10, time_limit=60)
  assert bool(result['accepted']) == accepted
  check_result(p, result)


def test_zero_load_has_no_feasible_proposals(cuda_available):
  p = Problem(load=[[0]], x=[[0]], capacity=[[1]], memory=[1], ram=[0],
    alpha=[[1]], gamma=[[1]], edges=[], beta=[])
  result = refine_x(p, [], iterations=100, time_limit=60)
  assert result['accepted'] == 0 and result['welfare'] == 0
  check_result(p, result)


@pytest.mark.parametrize('kwargs', [{'iterations': 0}, {'time_limit': 0}])
def test_empty_budget_preserves_incumbent(kwargs):
  result = refine_x(paired(), [], **kwargs)
  assert result['x'].tolist() == [[2,0]]
  assert result['attempted'] == result['accepted'] == 0


@pytest.mark.parametrize('kwargs', [
  {'iterations': -1}, {'iterations': True}, {'iterations': 1.5},
  {'time_limit': -1}, {'time_limit': float('nan')}, {'time_limit': float('inf')},
  {'time_limit': True}, {'seed': -1}, {'seed': True}, {'seed': 2147483646},
  {'max_device_bytes': 0}, {'max_device_bytes': True}, {'max_device_bytes': 1.5}])
def test_invalid_options(kwargs):
  with pytest.raises(ValueError):
    refine_x(paired(), [], **kwargs)


def test_invalid_input_is_checked_even_with_empty_budget():
  with pytest.raises(ValueError):
    refine_x(paired(), [1], iterations=0)
  p = paired()
  p.alpha[0,0] = float('nan')
  with pytest.raises(ValueError):
    refine_x(p, [], time_limit=0)


def test_memory_limit_fails_before_cuda_import_or_allocation(monkeypatch):
  import centralized_vns_spark.xstage as module
  def forbidden():
    raise AssertionError('CUDA was initialized before the memory guard')
  monkeypatch.setattr(module, 'kernels', forbidden)
  with pytest.raises(ValueError, match='memory budget'):
    refine_x(paired(), [], max_device_bytes=1)


@pytest.mark.parametrize('iterations,proposals,budget,sms,expected', [
  (100000,100000,1000000,0,4096),
  (100000,100000,1000000,48,24576),
  (100000,100000,1100,48,100),
  (71,100000,1000000,48,71),
  (100000,29,1000000,48,29),
  (0,100000,1,48,0),
])
def test_candidate_batch_respects_hardware_and_work_memory_bounds(
    iterations, proposals, budget, sms, expected):
  from centralized_vns_spark.xstage import _candidate_batch
  assert _candidate_batch(iterations, proposals, 100, 10, budget, sms) == expected


def test_cuda_unavailable_does_not_silently_run_cpu(monkeypatch):
  from types import SimpleNamespace
  import centralized_vns_spark.xstage as module
  monkeypatch.setattr(module, 'kernels', lambda *args: (
    SimpleNamespace(is_available=lambda: False), None, None, None))
  with pytest.raises(RuntimeError, match='CUDA unavailable'):
    refine_x(paired(), [], iterations=10, time_limit=60)


def test_memory_limited_batch_keeps_the_same_best_improvement(cuda_available):
  p = paired()
  wide = refine_x(p, [], iterations=100, time_limit=60)
  narrow = refine_x(p, [], iterations=100, time_limit=60, max_device_bytes=4096)
  assert narrow['execution']['candidate_batch'] < wide['execution']['candidate_batch']
  assert narrow['execution']['device_bytes'] <= 4096
  np.testing.assert_array_equal(narrow['x'], wide['x'])
  np.testing.assert_array_equal(narrow['y'], wide['y'])
  assert narrow['welfare'] == wide['welfare']
  assert narrow['attempted'] == wide['attempted']
  assert narrow['accepted'] == wide['accepted']


def test_nonfinite_proposal_does_not_replace_finite_incumbent(cuda_available):
  p = Problem(load=[[2]], x=[[0]], capacity=[[2]], memory=[1], ram=[1],
    alpha=[[1e308]], gamma=[[0]], edges=[], beta=[])
  result = refine_x(p, [], iterations=30, time_limit=60)
  assert result['x'].tolist() == [[1]]
  assert np.isfinite(result['welfare'])
  check_result(p, result)


@pytest.fixture
def large_floor_problem():
  return Problem(load=np.ones((2,8)), x=[[1]*8,[0]*8],
    capacity=np.ones((2,8)), memory=np.ones(8), ram=[8,8],
    alpha=[[1e16,1,1,1,1,1,1,1],[100]*8], gamma=np.zeros((2,8)),
    edges=[], beta=[])


def test_unchanged_large_local_floor_matches_numpy_reduction(large_floor_problem):
  from centralized_vns_spark.xstage import (
    _add, _multiply, _numerical_functions, _sum_function, _term_function)
  p = large_floor_problem
  sum_terms = _sum_function(_term_function(_multiply, _add), _add)
  _, check = _numerical_functions(_multiply, _add, sum_terms)
  value = check(p.x, np.zeros(0, dtype=np.int64), np.zeros_like(p.x),
    np.zeros_like(p.x), p.load, p.capacity, p.memory, p.ram, p.alpha,
    p.gamma, p.src, p.dst, p.fun, p.beta, local_gains(p, p.x),
    np.empty(64, dtype=np.int64), np.empty(72))
  assert value == p.check([])['welfare']


def test_large_unchanged_floor_does_not_block_another_node(large_floor_problem, cuda_available):
  p = large_floor_problem
  result = refine_x(p, [], iterations=1232, time_limit=60)
  assert result['accepted'] == 1
  assert result['x'][0].tolist() == [1]*8
  assert result['x'][1].sum() == 1
  assert result['welfare'] > result['initial_welfare']
  check_result(p, result)


@pytest.mark.parametrize('length', [0,1,7,8,9,127,128,129,256,1000,8193])
def test_pairwise_sum_matches_numpy_for_mixed_sign_magnitudes(length):
  from centralized_vns_spark.xstage import _add, _multiply, _sum_function, _term_function
  pairwise = _sum_function(_term_function(_multiply, _add), _add)
  values = np.resize(np.array([1e16,1,-1e16,1,1e-8,-1e-8,3,-2,1e10,-1e10]), length)
  ones = np.ones((1,length), dtype=np.int64)
  empty = np.zeros(0, dtype=np.int64)
  result = pairwise(length, 0, 0, ones, empty, ones, ones, values[None], ones,
    empty, empty, empty, np.empty(64, dtype=np.int64), np.empty(72))
  assert result == np.sum(values, dtype=np.float64)


def test_replica_ratio_rejected_before_fixed_width_integer_conversion(monkeypatch):
  import math
  from centralized_vns_spark.xstage import (
    _add, _multiply, _numerical_functions, _sum_function, _term_function)
  sum_terms = _sum_function(_term_function(_multiply, _add), _add)
  _, check = _numerical_functions(_multiply, _add, sum_terms)
  p = Problem(load=[[1]], x=[[0]], capacity=[[1e-30]], memory=[1], ram=[1],
    alpha=[[1]], gamma=[[0]], edges=[], beta=[])
  original = math.ceil
  def bounded_ceil(value):
    if value > 2**40:
      raise OverflowError('CUDA integer conversion would be outside the supported range')
    return original(value)
  monkeypatch.setattr(math, 'ceil', bounded_ceil)
  value = check(np.ones((1,1), dtype=np.int64), np.zeros(0, dtype=np.int64),
    np.zeros_like(p.x), np.zeros_like(p.x), p.load, p.capacity, p.memory, p.ram,
    p.alpha, p.gamma, p.src, p.dst, p.fun, p.beta, local_gains(p, p.x),
    np.empty(64, dtype=np.int64), np.empty(72))
  assert value == -math.inf


@pytest.mark.parametrize('buffer_size', [16,32,8192])
def test_numpy_buffering_matches_both_row_and_flat_reductions(buffer_size):
  from centralized_vns_spark.xstage import _add, _multiply, _sum_function, _term_function
  pairwise = _sum_function(_term_function(_multiply, _add), _add, buffer_size)
  row = np.resize(np.array([1e16,1,-1e16,1,1e-8,-1e-8,3,-2,1e10,-1e10]), 8193)
  values = np.stack((row, row[::-1]))
  ones = np.ones_like(values, dtype=np.int64)
  zeros = np.zeros_like(ones)
  empty = np.zeros(0, dtype=np.int64)
  previous = np.setbufsize(buffer_size)
  try:
    for node in range(2):
      result = pairwise(values.shape[1], 0, node, ones, empty, zeros, ones, values,
        zeros, empty, empty, empty, np.empty(64, dtype=np.int64), np.empty(72))
      assert result == np.sum(values, axis=1)[node]
    result = pairwise(values.size, 2, 0, ones, empty, zeros, ones, values,
      zeros, empty, empty, empty, np.empty(64, dtype=np.int64), np.empty(72))
    assert result == np.sum(values)
  finally:
    np.setbufsize(previous)


@pytest.mark.parametrize('strided', [False, True])
@pytest.mark.parametrize('p,y', [
  (paired(), []),
  (paired(alpha=(2,1), gamma=(0,100)), []),
  (Problem(load=[[2,2,0],[0,0,2]], x=[[2,0,0],[0,0,0]],
    capacity=np.full((2,3),2), memory=[2,1,1], ram=[2,0],
    alpha=[[1,1,0],[0,0,0]], gamma=np.zeros((2,3)),
    edges=[(1,0,2)], beta=[4]), [0]),
  (Problem(load=[[4],[0],[0]], x=[[0],[0],[0]], capacity=[[2],[4],[4]],
    memory=[1], ram=[1,1,1], alpha=[[8],[0],[0]], gamma=np.zeros((3,1)),
    edges=[(0,1,0),(0,2,0)], beta=[1,3]), [2,2]),
])
def test_scalar_transactions_match_independent_reference(p, y, strided):
  from centralized_vns_spark.xstage import (
    _add, _multiply, _numerical_functions, _proposal_function, _sum_function, _term_function)
  sum_terms = _sum_function(_term_function(_multiply, _add), _add)
  room, check = _numerical_functions(_multiply, _add, sum_terms)
  proposal = _proposal_function(room, check, _multiply, _add, sum_terms)
  y = np.asarray(y, dtype=np.int64)
  tx, incoming, outgoing = [np.empty_like(p.x) for _ in range(3)]
  ty = np.empty_like(y)
  if strided:
    tx, incoming, outgoing = [np.empty((*p.x.shape, 3), dtype=np.int64)[:, :, 1]
                              for _ in range(3)]
    ty = np.empty((len(y), 3), dtype=np.int64)[:, 1]
  floor = local_gains(p, p.x)
  best = p.check(y)['welfare']
  for index in range(p.load.size * (5 + 9 * p.load.shape[1])):
    value = proposal(index, p.x, y, tx, ty, incoming, outgoing,
      p.load, p.capacity, p.memory, p.ram, p.alpha, p.gamma,
      p.src, p.dst, p.fun, p.beta, p.reward,
      np.argsort(p.reward, kind='stable'), np.argsort(-p.reward, kind='stable'),
      floor, int(p.load.sum()), np.empty(64, dtype=np.int64), np.empty(72))
    if np.isfinite(value):
      assert value == pytest.approx(replace(p, x=tx).check(ty)['welfare'])
      assert np.all(local_gains(p, tx) >= floor - 1e-12)
      best = max(best, value)
  assert best == pytest.approx(scalar_best(p, y))
