"""Numerical parity requires a real GPU; policy checks never need a driver."""
import os
import numpy as np
import pytest
from models import dp_gpu
from plasma.core.sbm import minimize_replica_costs

real_gpu = pytest.mark.skipif(os.environ.get('DFAAS_TEST_CUDA') != '1', reason='set DFAAS_TEST_CUDA=1 for real CUDA tests')

def test_empty_and_limits_without_cuda():
  assert dp_gpu.minimize_replica_costs_batch([], [], []) == []
  assert dp_gpu.gpu_problem_fits([[0, 0]], [2], 8)
  assert not dp_gpu.gpu_problem_fits([[0]], [1], 2**31)
  assert not dp_gpu.gpu_problem_fits([[0]] * 100, [1] * 100, 200_000)
  assert not dp_gpu.gpu_problem_fits([[0]], [0], 1)

def test_padded_chunks_fit_actual_buffers(monkeypatch):
  monkeypatch.setattr(dp_gpu, '_MAX_DEVICE_BYTES', 6000)
  items = [([[0] * l] * f, [2] * f, b) for l, f, b in [(2, 1, 40), (5, 3, 20), (2, 1, 257)] * 3]
  chunks = list(dp_gpu._chunks(items))
  assert sum(map(len, chunks)) == len(items)
  assert len(chunks) > 1
  assert all(dp_gpu._padded_bytes(chunk) <= 6000 for chunk in chunks)

@real_gpu
def test_real_gpu_matches_cpu_across_heterogeneous_calls():
  rng = np.random.default_rng(47)
  cases = [([[0, 0], [0, 0]], [2, 4], 8), ([[np.inf]], [1], 0),
           ([[0, -1], [0, -2]], [4, 6], 9), ([[0, -1]], [1], 257),
           ([[3, 2]], [3], 0), ([[float("-inf")]], [1], 0),
           ([[0., float("-inf")]], [1], 1)]
  for _ in range(80):
    nf = int(rng.integers(1, 7))
    curves = [rng.integers(-5, 6, size=int(rng.integers(1, 9))).astype(float) for _ in range(nf)]
    for curve in curves:
      curve[rng.random(len(curve)) < .2] = np.inf
    cases.append((curves, rng.integers(1, 9, size=nf) * 2, int(rng.integers(0, 60))))
  for batch in (cases, cases[::-1][:17], cases[:3]):
    actual = dp_gpu.minimize_replica_costs_batch([c for c, _, _ in batch], [m for _, m, _ in batch], [b for _, _, b in batch])
    for got, (c, m, b) in zip(actual, batch):
      try:
        expected = minimize_replica_costs(c, m, b)
      except ValueError:
        assert got is None
      else:
        np.testing.assert_array_equal(got, expected)
  np.testing.assert_array_equal(actual[0], [0, 0])

@real_gpu
def test_real_gpu_chunked_result_parity(monkeypatch):
  monkeypatch.setattr(dp_gpu, '_MAX_DEVICE_BYTES', 6000)
  costs = [[[0., -1., -2.]]] * 30
  got = dp_gpu.minimize_replica_costs_batch(costs, [[1]] * 30, [257] * 30)
  for choice in got:
    np.testing.assert_array_equal(choice, [2])


def test_chunk_planning_scales_linearly(monkeypatch):
  calls = []
  original = dp_gpu._shape
  def shape(*args):
    calls.append(1)
    return original(*args)
  monkeypatch.setattr(dp_gpu, '_shape', shape)
  items = [([[0., -1.]], [1], 30)] * 100
  assert len(list(dp_gpu._chunks(items))) == 1
  assert len(calls) <= 3 * len(items)
