from dataclasses import replace
from itertools import product

import highspy
import numpy as np
import pytest

from centralized_vns.core import Problem
from centralized_vns_highs.mip import fixed_model, ram_receive_limit, require_ok


def cycle():
  return Problem(
    load=[[2], [2]],
    x=[[0], [0]],
    capacity=[[1], [1]],
    memory=[1],
    ram=[1, 1],
    alpha=[[0], [0]],
    gamma=[[0], [0]],
    edges=[(0, 1, 0), (1, 0, 0)],
    beta=[2, 2],
  )


def shared():
  return Problem(
    load=[[3, 2], [2, 2], [2, 0]],
    x=[[0, 0], [1, 1], [0, 0]],
    capacity=[[1, 1], [1.5, 1], [2, 1]],
    memory=[2, 3],
    ram=[4, 7, 2],
    alpha=[[1, 1]] * 3,
    gamma=[[1, 1]] * 3,
    edges=[(0, 1, 0), (0, 1, 1), (2, 1, 0), (1, 0, 0)],
    beta=[3, 2, 4, 1],
  )


@pytest.mark.parametrize('p', [cycle(), shared(), replace(cycle(), ram=[0, 1])])
def test_same_integer_flows_and_exact_optimum(p):
  edges = np.arange(len(p.src), dtype=np.int32)
  upper = p.load[p.src, p.fun] - p.x[p.src, p.fun]
  models = [fixed_model(p, tighten_receive=tight) for tight in (False, True)]
  opt = -np.inf
  for candidate in product(*(range(int(u) + 1) for u in upper)):
    y = np.array(candidate)
    try:
      opt = max(opt, p.check(y)['welfare'])
      expected = highspy.HighsModelStatus.kOptimal
    except ValueError:
      expected = highspy.HighsModelStatus.kInfeasible
    for model in models:
      require_ok(model.changeColsBounds(len(edges), edges, y, y))
      require_ok(model.run())
      assert model.getModelStatus() == expected
  for model in models:
    require_ok(model.changeColsBounds(len(edges), edges, np.zeros(len(edges)), upper))
    require_ok(model.run())
    assert p.base_welfare - model.getObjectiveValue() == pytest.approx(opt)


def test_lp_excludes_fractional_sender_receiver_mix():
  p = cycle()
  bounds = []
  for tight in (False, True):
    lp = fixed_model(p, tighten_receive=tight)
    require_ok(lp.setOptionValue('solve_relaxation', True))
    require_ok(lp.run())
    bounds.append(p.base_welfare - lp.getObjectiveValue())
  assert bounds == pytest.approx([2.0, 4.0 / 3.0])


def test_ram_cap_accounts_for_other_functions_and_replica_tolerance():
  p = shared()

  def bound(problem):
    baseline = problem.check(np.zeros(len(problem.src)))['replicas']
    return ram_receive_limit(problem, np.full(problem.x.size, 100), baseline).reshape(problem.x.shape)

  assert bound(p)[1].tolist() == [2, 0]
  near = replace(cycle(), capacity=[[1.0 - 5e-10], [1.0 - 2e-9]])
  assert bound(near).ravel().tolist() == [1, 0]
  assert bound(replace(cycle(), capacity=[[1e300], [1e300]], ram=[0, 1])).ravel().tolist() == [0, 100]
