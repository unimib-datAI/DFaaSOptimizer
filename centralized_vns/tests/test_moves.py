import itertools
import numpy as np
import pytest
from centralized_vns.core import Problem,State,evaluate,apply_move,CPU,Options


def exchange_problem():
  # Two functions compete for the same one-replica memory at each receiver.
  return Problem(load=[[1,0],[0,1],[0,0],[0,0]],x=np.zeros((4,2)),
    capacity=np.ones((4,2)),memory=[1,1],ram=[0,0,1,1],
    alpha=np.ones((4,2)),gamma=np.ones((4,2)),
    edges=[(0,2,0),(0,3,0),(1,2,1),(1,3,1)],beta=[1,4,4,1])


def test_atomic_exchange_releases_memory():
  p=exchange_problem();s=State.empty(p,1)
  apply_move(p,s,0,np.array([[0,1],[3,1],[-1,0],[-1,0]]))
  assert not np.isfinite(evaluate(p,s,0,np.array([[0,-1],[1,1],[-1,0],[-1,0]])))
  move=np.array([[0,-1],[1,1],[3,-1],[2,1]])
  old=p.check(s.y[0])['welfare'];gain=evaluate(p,s,0,move)
  apply_move(p,s,0,move)
  assert gain==pytest.approx(6)
  assert p.check(s.y[0])['welfare']==pytest.approx(old+gain)


def test_all_small_compound_moves_against_full_checker():
  p=exchange_problem();s=State.empty(p,1)
  apply_move(p,s,0,np.array([[0,1],[3,1],[-1,0],[-1,0]]))
  before=p.check(s.y[0])['welfare']
  for deltas in itertools.product((-1,0,1),repeat=4):
    move=np.array(list(enumerate(deltas)))
    result=evaluate(p,s,0,move)
    try:
      full=p.check(s.y[0]+deltas)['welfare']-before
    except ValueError:
      assert not np.isfinite(result)
    else:
      assert result==pytest.approx(full)


def test_generated_trajectories_keep_caches_consistent():
  p=exchange_problem();engine=CPU(p,Options(pool_size=5,candidates=32,iterations=40))
  for i in range(40):
    engine.step(i)
    for k in range(5):
      check=p.check(engine.state.y[k])
      np.testing.assert_array_equal(check['incoming'],engine.state.incoming[k])
      np.testing.assert_array_equal(check['z'],engine.state.rejected[k])
      assert check['welfare']==pytest.approx(engine.value[k])


def test_fractional_capacity_and_replica_boundary():
  p=Problem(load=[[3],[0]],x=[[0],[0]],capacity=[[1],[1.5]],
    memory=[2],ram=[0,2],alpha=[[1],[1]],gamma=[[1],[1]],
    edges=[(0,1,0)],beta=[2])
  assert p.check(np.array([1]))['replicas'][1,0]==1
  with pytest.raises(ValueError): p.check(np.array([2]))


def test_shaking_removes_whole_replica_blocks_and_preserves_incumbent():
  from centralized_vns.moves import generate_move
  p=Problem(load=[[8],[0]],x=[[0],[0]],capacity=[[1],[8]],memory=[1],ram=[0,1],
    alpha=[[1],[1]],gamma=[[1],[1]],edges=[(0,1,0)],beta=[2])
  engine=CPU(p,Options(pool_size=1,candidates=4))
  engine.step(0);before=engine.best.copy()
  engine.stale[:]=9
  engine.step(9)
  assert engine.state.y[0,0]==0
  np.testing.assert_array_equal(engine.best,before)


def test_extreme_capacity_is_rejected_or_bounded_without_integer_overflow():
  p=Problem(load=[[1],[0]],x=[[0],[0]],capacity=[[1],[1e-320]],memory=[1],ram=[0,1],
    alpha=[[1],[1]],gamma=[[1],[1]],edges=[(0,1,0)],beta=[2])
  s=State.empty(p,1)
  assert not np.isfinite(evaluate(p,s,0,np.array([[0,1],[-1,0],[-1,0],[-1,0]])))
  p=Problem(load=[[1],[0]],x=[[0],[0]],capacity=[[1],[1e308]],memory=[1],ram=[0,2],
    alpha=[[1],[1]],gamma=[[1],[1]],edges=[(0,1,0)],beta=[2])
  engine=CPU(p,Options(pool_size=1,candidates=4));engine.step(0)
  assert engine.state.y[0,0]==1
