"""Exact fixed-iteration parity; run CUDA cases with explicit CUDASIM or hardware."""
from dataclasses import replace
import numpy as np
import pytest
from centralized_vns.core import Options
from centralized_vns_r.core import CPU, solve as reference_solve
from centralized_vns_r.tests.test_backends import boundary_problem, initialize
from centralized_vns_r.tests.test_moves import relocation_problem
from centralized_vns_r.tests.test_stopping import saturated_problem
from centralized_vns_spark.gpu import CUDA, solve, estimate_device_bytes


@pytest.fixture
def cuda_available():
  pytest.importorskip('numba')
  from numba import cuda
  if not cuda.is_available():
    pytest.skip('CUDA unavailable; explicitly enable CUDASIM')


@pytest.mark.parametrize('threads',[64,128,256])
@pytest.mark.parametrize('factory,initial',[(relocation_problem,[10,0,0]),
                                         (boundary_problem,[27,0])])
def test_fused_trajectory_matches_reference(cuda_available,threads,factory,initial):
  p=factory();y=np.array(initial,dtype=np.int64)
  o=Options(pool_size=2,candidates=67,iterations=9,backend='cuda',shake_after=1)
  cpu=CPU(p,o);initialize(cpu,p,y)
  gpu=CUDA(p,o,y,patience=0,batch_size=3,threads=threads)
  for iteration in range(9):
    expected=cpu.step(iteration)
    gpu.advance(iteration,1,0.,'all')
    np.testing.assert_array_equal(expected,gpu.best.copy_to_host())
    np.testing.assert_array_equal(cpu.moves,gpu.moves.copy_to_host())
    for actual,expected_array in [(gpu.y,cpu.state.y),(gpu.inc,cpu.state.incoming),
        (gpu.z,cpu.state.rejected),(gpu.best_y,cpu.best_y),(gpu.value,cpu.value),
        (gpu.level,cpu.level),(gpu.stale,cpu.stale)]:
      np.testing.assert_array_equal(actual.copy_to_host(),expected_array)
  np.testing.assert_array_equal(gpu.result(),cpu.result())


@pytest.mark.parametrize('scope',['all','global'])
def test_plateau_masks_remainder_of_batch(cuda_available,scope):
  p=saturated_problem()
  o=Options(pool_size=3,candidates=1,iterations=20,backend='cuda',shake_after=1)
  result=solve(p,o,patience=4,plateau_scope=scope,batch_size=7,threads=64)
  assert result['termination_reason']=='plateau'
  assert result['iterations_completed']==4
  assert result['candidates_evaluated']==12
  assert result['trace'][-1]['iteration']==4
  assert result['welfare']==1


def test_warm_start_solve_parity_and_limit_precedence(cuda_available):
  p=relocation_problem();y=np.array([0,10,10])
  o=Options(pool_size=2,candidates=16,iterations=8,backend='cuda',shake_after=1)
  actual=solve(p,o,y,patience=4,batch_size=3,threads=64)
  expected=reference_solve(p,replace(o,backend='cpu'),y,patience=4)
  for key in ('iterations_completed','termination_reason','welfare','candidates_evaluated'):
    assert actual[key]==expected[key]
  np.testing.assert_array_equal(actual['y'],expected['y'])
  for iterations in (0,4):
    r=solve(saturated_problem(),replace(o,iterations=iterations),patience=4,batch_size=7)
    assert r['termination_reason']=='iteration_limit'
    assert r['iterations_completed']==iterations


@pytest.mark.parametrize('kwargs',[{'threads':32},{'threads':True},{'batch_size':0},
  {'batch_size':True},{'plateau_scope':'unknown'},{'patience':1}])
def test_invalid_options_before_cuda_import(kwargs):
  with pytest.raises(ValueError):
    solve(saturated_problem(),Options(backend='cuda'),**kwargs)


def test_full_memory_guard_includes_plateau_and_batch_buffers():
  p=relocation_problem();o=Options(pool_size=3,candidates=7,backend='cuda',shake_after=1)
  a=estimate_device_bytes(p,o,4,3)
  assert estimate_device_bytes(p,o,8,3)-a==4*3*8
  with pytest.raises(ValueError,match='memory'):
    solve(p,replace(o,max_device_bytes=a-1),patience=4,batch_size=3)


def test_non_cuda_backend_never_silently_falls_back():
  with pytest.raises(ValueError,match='cuda'):
    solve(saturated_problem(),Options(backend='cpu'))


def test_nonfinite_warm_start_welfare_rejected_before_cuda():
  from centralized_vns.core import Problem
  p=Problem(load=[[10],[0]],x=[[0],[0]],capacity=[[1],[1]],memory=[1],ram=[0,10],
    alpha=[[1],[1]],gamma=[[0],[0]],edges=[(0,1,0)],beta=[1e308])
  with np.errstate(over='ignore'),pytest.raises(ValueError,match='welfare'):
    solve(p,Options(backend='cuda'),initial_y=np.array([10]))


def test_nonfinite_final_welfare_is_not_returned(monkeypatch):
  from centralized_vns.core import Problem
  from centralized_vns_spark import gpu
  from types import SimpleNamespace
  p=Problem(load=[[10],[0]],x=[[0],[0]],capacity=[[1],[1]],memory=[1],ram=[0,10],
    alpha=[[1],[1]],gamma=[[0],[0]],edges=[(0,1,0)],beta=[1e308])
  class FakeCUDA:
    def __init__(self,*args,**kwargs):
      self.cuda=SimpleNamespace(synchronize=lambda:None);self.device_bytes=1
    def result(self):
      return np.array([10])
  monkeypatch.setattr(gpu,'CUDA',FakeCUDA)
  with np.errstate(over='ignore'),pytest.raises(RuntimeError,match='welfare'):
    solve(p,Options(backend='cuda',iterations=0))


@pytest.mark.parametrize('scope,stop',[('all',9),('global',5)])
def test_monitor_tracks_subordinate_archive_and_window(cuda_available,scope,stop):
  p=saturated_problem();o=Options(pool_size=2,candidates=1,iterations=30,
    backend='cuda',shake_after=1)
  engine=CUDA(p,o,np.zeros(0,dtype=np.int64),patience=4,batch_size=1,threads=64)
  for step in range(1,stop+1):
    engine.best.copy_to_device(np.array([100.,float(min(step,5))]))
    engine.monitor[1,64](step,30,4,0.,scope=='global',engine.status,
      engine.best,engine.history,engine.trace,0)
    status=engine.status.copy_to_host()
    assert bool(status[0])==(step==stop)
    assert status[1]==step and status[2]==0


@pytest.mark.parametrize('budget,completed',[('total',0),('search',1)])
def test_setup_and_search_deadlines_are_distinct(monkeypatch,budget,completed):
  from centralized_vns_spark import gpu
  from types import SimpleNamespace
  clock=[0.]
  class FakeCUDA:
    def __init__(self,p,o,y,**kwargs):
      clock[0]+=5.
      self.cuda=SimpleNamespace(synchronize=lambda:None)
      self.host_status=np.zeros(3,dtype=np.int64);self.device_bytes=100
    def advance(self,iteration,count,tolerance,scope):
      clock[0]+=3.;self.host_status[1]=iteration+count
      return 1.
    def result(self):
      return np.zeros(0,dtype=np.int64)
  monkeypatch.setattr(gpu,'CUDA',FakeCUDA)
  monkeypatch.setattr(gpu.time,'perf_counter',lambda:clock[0])
  r=solve(saturated_problem(),Options(backend='cuda',iterations=10,time_limit=2.),
    batch_size=2,time_budget=budget)
  assert r['termination_reason']=='time_limit'
  assert r['iterations_completed']==completed
  assert r['execution']['setup_s']==5.
  assert r['execution']['search_elapsed_s']==(3. if completed else 0.)
