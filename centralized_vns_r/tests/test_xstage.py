from dataclasses import replace
import time

import numpy as np
import pytest

from centralized_vns.core import Problem
from centralized_vns_r.xstage import local_gains, refine_x


def paired(alpha=(1, 2), gamma=(0, 1)):
  return Problem(load=[[2, 4]], x=[[2, 0]], capacity=[[2, 4]],
    memory=[1, 1], ram=[1], alpha=[alpha], gamma=[gamma], edges=[], beta=[])


def test_paired_replica_reallocation_improves_with_immutable_floor():
  p=paired(); x=p.x.copy(); y=np.zeros(0,dtype=np.int64)
  r=refine_x(p,y,iterations=100,time_limit=2)
  assert r['algorithm']=='VNS-RX' and r['backend']=='cpu'
  assert r['x'].tolist()==[[0,4]]
  assert r['welfare']==pytest.approx(2)
  assert r['local_gain_floor'].tolist()==[1]
  assert r['local_gain'].tolist()==[2]
  assert r['accepted']>0 and r['accepted']<=r['attempted']<=100
  assert np.array_equal(p.x,x) and y.size==0
  checked=replace(p,x=r['x']).check(r['y'])
  for key in ('z','replicas','incoming'):
    assert np.array_equal(r[key],checked[key])
  values=[row['best_welfare'] for row in r['trace']]
  assert values==sorted(values)


def test_global_improvement_cannot_pay_for_a_local_floor_violation():
  p=paired(alpha=(2,1),gamma=(0,100))
  assert replace(p,x=[[0,4]]).check([])['welfare']>p.check([])['welfare']
  r=refine_x(p,[],iterations=100,time_limit=2)
  assert r['x'].tolist()==[[2,0]]
  assert r['local_gain_floor'].tolist()==[2]
  assert r['accepted']==0


def test_receiver_insertion_repairs_a_neutral_local_pair():
  p=Problem(load=[[2,2,0],[0,0,2]],x=[[2,0,0],[0,0,0]],
    capacity=[[2,2,2],[2,2,2]],memory=[2,1,1],ram=[2,0],
    alpha=[[1,1,0],[0,0,0]],gamma=np.zeros((2,3)),
    edges=[(1,0,2)],beta=[4])
  r=refine_x(p,[0],iterations=100,time_limit=2)
  assert r['x'].tolist()==[[0,2,0],[0,0,0]]
  assert r['y'].tolist()==[2]
  assert r['welfare']==pytest.approx(5)
  assert r['local_gain'].tolist()==[1,0]


def test_local_admission_and_outgoing_localization():
  p=Problem(load=[[4],[0]],x=[[0],[0]],capacity=[[4],[4]],
    memory=[1],ram=[1,1],alpha=[[4],[0]],gamma=[[1],[0]],
    edges=[(0,1,0)],beta=[1])
  y=np.array([3]); r=refine_x(p,y,iterations=50,time_limit=2)
  assert r['x'].tolist()==[[4],[0]] and r['y'].tolist()==[0]
  assert r['welfare']==pytest.approx(4)
  assert y.tolist()==[3] and p.x.tolist()==[[0],[0]]


@pytest.mark.parametrize('loss,accepted',[(5e-13,True),(2e-12,False)])
def test_floor_tolerance_is_float64(loss,accepted):
  p=Problem(load=[[1,1]],x=[[1,0]],capacity=[[1,1]],memory=[1,1],
    ram=[1],alpha=[[1,1-loss]],gamma=[[0,1]],edges=[],beta=[])
  r=refine_x(p,[],iterations=50,time_limit=2)
  assert (r['accepted']>0)==accepted


@pytest.mark.parametrize('gain,accepted',[(5e-13,False),(2e-12,True)])
def test_only_strict_welfare_gains_are_accepted(gain,accepted):
  p=Problem(load=[[1]],x=[[0]],capacity=[[1]],memory=[1],ram=[1],
    alpha=[[gain]],gamma=[[0]],edges=[],beta=[])
  r=refine_x(p,[],iterations=50,time_limit=2)
  assert (r['accepted']>0)==accepted


@pytest.mark.parametrize('budget',[{'iterations':0},{'time_limit':0},
  {'time_limit':1e-12}])
def test_zero_or_expired_budget_preserves_the_incumbent(budget):
  p=paired(); r=refine_x(p,[],**budget)
  assert np.array_equal(r['x'],p.x) and r['accepted']==r['attempted']==0
  assert r['welfare']==p.check([])['welfare']


def test_zero_load_edges_and_reproducibility():
  p=Problem(load=[[0]],x=[[0]],capacity=[[1]],memory=[1],ram=[1],
    alpha=[[1]],gamma=[[1]],edges=[],beta=[])
  assert local_gains(p,p.x).tolist()==[0]
  assert refine_x(p,[],iterations=5)['welfare']==0
  a=refine_x(paired(),[],iterations=80); b=refine_x(paired(),[],iterations=80)
  assert np.array_equal(a['x'],b['x']) and a['attempted']==b['attempted']


def test_iteration_and_time_budgets_include_repairs():
  p=paired()
  r=refine_x(p,[],iterations=1,time_limit=2)
  assert r['attempted']<=1
  start=time.perf_counter()
  r=refine_x(p,[],iterations=10**9,time_limit=.01)
  assert time.perf_counter()-start<1


@pytest.mark.parametrize('kwargs',[{'iterations':-1},{'iterations':1.5},
  {'iterations':True},{'time_limit':-1},{'time_limit':float('nan')},
  {'time_limit':float('inf')},{'seed':-1},{'seed':True},{'seed':2147483646}])
def test_invalid_arguments(kwargs):
  with pytest.raises(ValueError): refine_x(paired(),[],**kwargs)


@pytest.mark.parametrize('x',[[[float('nan'),0]],[[float('inf'),0]],
  [[-1,0]],[[.5,0]],[[3,0]],[[1]]])
def test_local_gains_reject_invalid_local_plans(x):
  with pytest.raises(ValueError): local_gains(paired(),x)


def test_invalid_incumbent_and_nonfinite_problem_are_rejected():
  p=paired()
  with pytest.raises(ValueError): refine_x(p,[1])
  p.alpha[0,0]=float('nan')
  with pytest.raises(ValueError): refine_x(p,[])


def test_nonfinite_candidate_does_not_discard_the_finite_incumbent():
  p=Problem(load=[[2]],x=[[0]],capacity=[[2]],memory=[1],ram=[1],
    alpha=[[1e308]],gamma=[[0]],edges=[],beta=[])
  r=refine_x(p,[],iterations=20)
  assert r['x'].tolist()==[[1]] and np.isfinite(r['welfare'])


def test_receiver_repair_does_not_create_a_sender_receiver_role_conflict():
  p=Problem(load=[[1,1],[0,2],[0,0]],x=[[1,0],[0,0],[0,0]],
    capacity=np.full((3,2),2),memory=[2,1],ram=[2,0,1],
    alpha=[[0,0],[0,0],[0,0]],gamma=np.zeros((3,2)),
    edges=[(0,2,1),(1,0,1)],beta=[1,10])
  r=refine_x(p,[1,0],iterations=50)
  assert r['y'].tolist()==[0,2]
  assert r['x'].tolist()==[[0,1],[0,0],[0,0]]
  assert r['welfare']==pytest.approx(10)
  replace(p,x=r['x']).check(r['y'])


def test_invalid_incumbent_physics_are_rejected_even_with_no_budget():
  p=Problem(load=[[2],[2],[0]],x=[[0],[0],[0]],capacity=[[2],[2],[2]],
    memory=[1],ram=[1,1,1],alpha=np.ones((3,1)),gamma=np.ones((3,1)),
    edges=[(0,1,0),(1,2,0)],beta=[1,1])
  with pytest.raises(ValueError,match='role'):
    refine_x(p,[1,1],iterations=0)
  with pytest.raises(ValueError):
    refine_x(p,[3,0],time_limit=0)
