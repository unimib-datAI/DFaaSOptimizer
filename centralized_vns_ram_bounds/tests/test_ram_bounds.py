from itertools import product
import highspy
import numpy as np
import pytest

from centralized_vns.core import Problem
from centralized_vns_batch_bounds.lp import Relaxation
from centralized_vns_batch_bounds.solvers import highs
from centralized_vns_highs.mip import require_ok
from centralized_vns_ram_bounds.formulation import add_ram_rows, levels


def shared_receiver():
  return Problem(
    load=[[2, 0, 0], [2, 2, 0], [0, 0, 2], [0, 0, 0]],
    x=[[0] * 3] * 4,
    capacity=[[2.5] * 3] * 4,
    memory=[2, 2, 3],
    ram=[3] * 4,
    alpha=[[0] * 3] * 4,
    gamma=[[0] * 3] * 4,
    edges=[(0, 3, 0), (1, 3, 0), (1, 3, 1), (2, 3, 2)],
    beta=[1, 1, 1, 1],
  )


def test_levels_use_integer_traffic_and_checker_tolerance():
  assert levels(1, 2.5, 1, 6) == [(0, 1), (1, 4), (2, 6)]
  assert levels(0, 100, 0, 1) == [(1, 1)]
  assert levels(0, 0.25, 0, 0) == []
  assert levels(0, 0.25, 0, 2) == [(4, 1), (8, 2)]


def test_every_feasible_flow_extends_and_each_bound_covers_its_region():
  p = shared_receiver()
  lp = Relaxation(p, strengthen=add_ram_rows)
  assert lp.strengthening['contended_nodes'] == 1
  columns = np.arange(len(p.src), dtype=np.int32)
  optimum = np.full(2, -np.inf)
  for state in product(range(3), repeat=len(columns)):
    y = np.array(state)
    try:
      value = p.check(y)['welfare']
    except ValueError:
      continue
    branch = int(y[0] > 0)
    optimum[branch] = max(optimum[branch], value)
    require_ok(lp.solver.changeColsBounds(len(y), columns, y, y))
    require_ok(lp.solver.run())
    assert lp.solver.getModelStatus() == highspy.HighsModelStatus.kOptimal
  require_ok(lp.solver.changeColsBounds(len(columns), columns, lp.lo[columns], lp.hi[columns]))
  lo, hi = lp.boxes([0], [0], [(1, 0), (1, 1)])
  bounds = highs(lp, lo, hi, 20)['bounds']
  assert np.all(np.array(bounds) >= optimum - 1e-9)
  assert bounds == pytest.approx(optimum)


def test_mandatory_inflow_excludes_competing_replicas_without_losing_feasible_states():
  from functools import partial
  from centralized_vns_ram_bounds.formulation import propagate

  p = shared_receiver()
  options = {}
  lp = Relaxation(p, strengthen=partial(add_ram_rows, options_by_cell=options))
  lo, hi = lp.boxes([0], [0], [(1, 1)])
  propagate(p, lo, hi, options)
  n = len(p.src)
  flags = n + p.x.size
  assert lo[0, flags] == 1 and hi[0, flags + 9] == 0
  assert all(hi[0, z] == 0 for k, q, z in options[3, 1] if k > 0)
  assert all(hi[0, z] == 0 for k, q, z in options[3, 2] if k > 0)
  columns = np.arange(len(lp.c), dtype=np.int32)
  require_ok(lp.solver.changeColsBounds(len(columns), columns, lo[0], hi[0]))
  tested = 0
  for state in product(range(3), repeat=n):
    y = np.array(state)
    if y[0] < 1:
      continue
    try:
      p.check(y)
    except ValueError:
      continue
    require_ok(lp.solver.changeColsBounds(n, columns[:n], y, y))
    require_ok(lp.solver.run())
    assert lp.solver.getModelStatus() == highspy.HighsModelStatus.kOptimal
    tested += 1
  assert tested >= 3
  assert levels(2, 100000.0, 1, 200000) == [(0, 99998), (1, 199998), (2, 200000)]
