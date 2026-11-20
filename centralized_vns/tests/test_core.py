import itertools
import numpy as np
import pytest
from centralized_vns.core import Problem, State, evaluate, apply_move, solve, Options


def tiny():
  return Problem(
    load=[[3], [0], [0]], x=[[0], [0], [0]], capacity=[[1], [2], [2]],
    memory=[1], ram=[0, 1, 1], alpha=[[1], [1], [1]],
    gamma=[[1], [1], [1]], edges=[(0, 1, 0), (0, 2, 0)], beta=[3, 2])


def test_compound_move_and_objective():
  p=tiny();s=State.empty(p,1)
  move=np.array([[0,2],[-1,0],[-1,0],[-1,0]],dtype=np.int64)
  before=p.check(s.y[0])['welfare']
  gain=evaluate(p,s,0,move)
  assert gain==pytest.approx(8/3)
  apply_move(p,s,0,move)
  assert p.check(s.y[0])['welfare']-before==pytest.approx(gain)
  assert not np.isfinite(evaluate(p,s,0,np.array([[1,2],[-1,0],[-1,0],[-1,0]])))
  reroute=np.array([[0,-1],[1,1],[-1,0],[-1,0]])
  gain=evaluate(p,s,0,reroute);before=p.check(s.y[0])['welfare']
  apply_move(p,s,0,reroute)
  assert p.check(s.y[0])['welfare']-before==pytest.approx(gain)


def test_exact_tiny_oracle_and_reproducibility():
  p=tiny();best=-np.inf
  for values in itertools.product(range(4),repeat=2):
    try: best=max(best,p.check(np.array(values))['welfare'])
    except ValueError: pass
  opts=Options(pool_size=4,candidates=16,iterations=40,seed=4850)
  a=solve(p,opts);b=solve(p,opts)
  assert a['welfare']==pytest.approx(best)
  assert np.array_equal(a['y'],b['y'])
  assert [r['best_welfare'] for r in a['trace']]==sorted(r['best_welfare'] for r in a['trace'])
  assert np.array_equal(p.x,[[0],[0],[0]])


def test_roles_memory_and_input_rejection():
  p=Problem(load=[[2],[2],[0]],x=[[0],[0],[0]],capacity=[[2],[2],[2]],
    memory=[1],ram=[1,1,1],alpha=np.ones((3,1)),gamma=np.ones((3,1)),
    edges=[(0,1,0),(1,2,0)],beta=[2,2])
  with pytest.raises(ValueError): p.check(np.array([1,1]))
  with pytest.raises(ValueError): p.check(np.array([3,0]))
  with pytest.raises(ValueError): p.check(np.array([.5,0]))
  with pytest.raises(ValueError): p.check(np.array([-1,0]))
  s=State.empty(p,1);apply_move(p,s,0,np.array([[0,1],[-1,0],[-1,0],[-1,0]]))
  assert not np.isfinite(evaluate(p,s,0,np.array([[1,1],[-1,0],[-1,0],[-1,0]])))


def test_no_edges_and_zero_iterations():
  p=Problem(load=[[0]],x=[[0]],capacity=[[1]],memory=[1],ram=[1],
    alpha=[[1]],gamma=[[1]],edges=[],beta=[])
  r=solve(p,Options(iterations=0));assert r['welfare']==0
  assert solve(p,Options(iterations=2))['welfare']==0


@pytest.mark.parametrize('kwargs',[{'pool_size':0},{'candidates':0},{'seed':-1},
  {'iterations':-1},{'time_limit':float('nan')},{'backend':'invalid'}])
def test_invalid_options(kwargs):
  with pytest.raises(ValueError): solve(tiny(),Options(**kwargs))


def test_same_edge_entries_are_aggregated_atomically():
  p=tiny();s=State.empty(p,1)
  apply_move(p,s,0,np.array([[0,1],[-1,0],[-1,0],[-1,0]]))
  # Intermediate -1 flow is irrelevant: the atomic final delta is zero.
  move=np.array([[0,-2],[0,2],[-1,0],[-1,0]])
  assert evaluate(p,s,0,move)==0
  apply_move(p,s,0,move)
  assert s.y.tolist()==[[1,0]]


def test_large_memory_product_cannot_wrap_to_feasible():
  with pytest.raises(ValueError,match='memory'):
    Problem(load=[[2**32]],x=[[2**32]],capacity=[[1]],memory=[2**32],ram=[1],
      alpha=[[1]],gamma=[[1]],edges=[],beta=[])


def test_positive_load_needs_at_least_one_replica():
  with pytest.raises(ValueError,match='memory'):
    Problem(load=[[1]],x=[[1]],capacity=[[1e12]],memory=[1],ram=[0],
      alpha=[[1]],gamma=[[1]],edges=[],beta=[])


def test_corrupt_flows_rejected_before_accumulation():
  edges=[(i,j,0) for i in range(4) for j in range(4) if i!=j]
  p=Problem(load=np.ones((4,1)),x=np.zeros((4,1)),capacity=np.ones((4,1)),
    memory=[1],ram=np.ones(4),alpha=np.ones((4,1)),gamma=np.ones((4,1)),
    edges=edges,beta=np.ones(len(edges)))
  with pytest.raises(ValueError): p.check(np.full(len(edges),2**62,dtype=np.int64))
