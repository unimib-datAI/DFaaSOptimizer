import numpy as np
import pytest
from centralized_vns.core import Options, Problem
from centralized_vns_highs.core import Record
from centralized_vns_highs.pool import Pool
from centralized_vns_bc_metal.experiment import Model
from centralized_vns_bc_metal.moves import install


def saturated():
  p = Problem(
    load=[[4, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]],
    x=[[0] * 3] * 4,
    capacity=[[4] * 3] * 4,
    ram=[2] * 4,
    memory=[2, 1, 1],
    alpha=[[0] * 3] * 4,
    gamma=[[0] * 3] * 4,
    edges=[(0, 3, 0), (1, 3, 1), (2, 3, 2)],
    beta=[10, 1, 1],
  )
  return p, np.array([0, 1, 1])


def test_complete_ram_witness():
  p, y = saturated()
  m = Model(p)
  for flow in (y, np.array([4, 0, 0]), np.zeros_like(y)):
    v = m.lift(Record('test', flow, p.check(flow)))
    assert np.all(m.A @ v <= m.b + 1e-8)
    assert len(v) == m.model.lp_.num_col_ and np.all(v[:3] == flow)


def test_two_evictions_admit_a_more_valuable_function():
  mx = pytest.importorskip('mlx.core')
  if not mx.metal.is_available():
    pytest.skip('requires Metal')
  p, y = saturated()
  pool = Pool(p, Options(backend='metal', pool_size=2, candidates=128), y, p.check(y), 1)
  e = pool.engine
  install(e)
  e.level[:] = 2
  pool.advance(0, 1)
  assert e.double_evictions > 0
  np.testing.assert_array_equal(e.result(), [4, 0, 0])
  assert p.check(e.result())['welfare'] > p.check(y)['welfare']
