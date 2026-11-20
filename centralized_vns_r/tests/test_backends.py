"""Exact CPU parity on real Metal and CUDA (or explicitly enabled CUDASIM)."""
import numpy as np
import pytest
from centralized_vns.core import Options, Problem
from centralized_vns_r.core import CPU, solve
from centralized_vns_r.tests.test_moves import relocation_problem


@pytest.fixture(params=['metal','cuda'])
def backend(request):
  if request.param=='metal':
    mx=pytest.importorskip('mlx.core')
    if not mx.metal.is_available():
      pytest.skip('Metal unavailable')
    from centralized_vns_r.metal import Metal
    return 'metal',Metal
  pytest.importorskip('numba')
  from numba import cuda
  if not cuda.is_available():
    pytest.skip('CUDA unavailable; enable CUDASIM explicitly')
  from centralized_vns_r.gpu import CUDA
  return 'cuda',CUDA


def boundary_problem():
  return Problem(load=[[29],[0],[0]],x=np.zeros((3,1)),
    capacity=[[1],[10-1e-10],[7-1e-7]],memory=[1],ram=[0,3,5],
    alpha=np.ones((3,1)),gamma=np.ones((3,1)),
    edges=[(0,1,0),(0,2,0)],beta=[2,2+1e-10])


def same_function_problem():
  return Problem(load=[[13],[11],[0],[0]],x=np.zeros((4,1)),
    capacity=[[1],[1],[10-1e-10],[7-1e-7]],memory=[1],ram=[0,0,2,4],
    alpha=np.ones((4,1)),gamma=np.ones((4,1)),
    edges=[(0,2,0),(0,3,0),(1,2,0),(1,3,0)],beta=[2,1,4,1])


def initialize(engine,p,y,cuda=False):
  checked=p.check(y)
  if cuda:
    for target,value in ((engine.y,y),(engine.inc,checked['incoming']),
        (engine.z,checked['z']),(engine.best_y,y)):
      target.copy_to_device(np.repeat(value[None],engine.o.pool_size,axis=0))
    for target in (engine.value,engine.best):
      target.copy_to_device(np.full(engine.o.pool_size,checked['welfare']))
  else:
    engine.state.y[:]=y;engine.state.incoming[:]=checked['incoming']
    engine.state.rejected[:]=checked['z'];engine.best_y[:]=y
    engine.value[:]=checked['welfare'];engine.best[:]=checked['welfare']


@pytest.mark.parametrize('factory,initial',[(relocation_problem,[10,0,0]),
  (boundary_problem,[27,0]),(same_function_problem,[13,0,0,0])])
@pytest.mark.parametrize('seed',[0,4850,2147483645])
def test_moves_and_warm_started_trajectory(backend,factory,initial,seed):
  name,cls=backend;p=factory();o=Options(pool_size=2,candidates=65,seed=seed,shake_after=1)
  cpu,gpu=CPU(p,o),cls(p,o);is_cuda=name=='cuda'
  initialize(cpu,p,np.array(initial));initialize(gpu,p,np.array(initial),is_cuda)
  # Start with the coupled neighborhood; exercise all levels and shaking explicitly.
  for iteration in range(12):
    if iteration<4:
      cpu.level[:]=[2,0,1,0][iteration];cpu.stale[:]=3 if iteration==3 else 0
      for attr in ('level','stale'):
        if is_cuda: getattr(gpu,attr).copy_to_device(getattr(cpu,attr))
        else: getattr(gpu,attr)[:]=getattr(cpu,attr)
    np.testing.assert_array_equal(cpu.step(iteration),gpu.step(iteration))
    np.testing.assert_array_equal(cpu.moves,gpu.moves.copy_to_host() if is_cuda else gpu.moves)
    for attr,device in [('y','y'),('incoming','inc'),('rejected','z')]:
      actual=getattr(gpu,device).copy_to_host() if is_cuda else getattr(gpu.state,attr)
      np.testing.assert_array_equal(getattr(cpu.state,attr),actual)
    for attr in ('value','best','best_y','level','stale'):
      actual=getattr(gpu,attr)
      np.testing.assert_array_equal(getattr(cpu,attr),actual.copy_to_host() if is_cuda else actual)
  np.testing.assert_array_equal(cpu.result(),gpu.result())
  p.check(gpu.result())


def test_solve_warm_start_and_empty(backend):
  name,_=backend;p=relocation_problem();y=np.array([10,0,0])
  cpu=solve(p,Options(pool_size=1,candidates=16,iterations=6),initial_y=y)
  gpu=solve(p,Options(pool_size=1,candidates=16,iterations=6,backend=name),initial_y=y)
  np.testing.assert_array_equal(cpu['y'],gpu['y'])
  assert cpu['welfare']==gpu['welfare']
  empty=Problem(load=[[0]],x=[[0]],capacity=[[1]],memory=[1],ram=[1],
    alpha=[[1]],gamma=[[1]],edges=[],beta=[])
  assert solve(empty,Options(pool_size=1,candidates=1,iterations=1,backend=name))['welfare']==0


@pytest.mark.parametrize('name',['cuda','metal'])
def test_memory_guard_before_backend_initialization(name):
  # The new adjacency/tables count against the cap before any accelerator imports.
  import importlib
  cls=getattr(importlib.import_module('centralized_vns_r.'+('gpu' if name=='cuda' else name)),
    'CUDA' if name=='cuda' else 'Metal')
  with pytest.raises(ValueError,match='memory'):
    cls(relocation_problem(),Options(max_device_bytes=1))


def test_plateau_backend_matches_cpu(backend):
  name,_=backend;p=relocation_problem();y=np.array([0,10,10])
  o=Options(pool_size=2,candidates=16,iterations=80,shake_after=1)
  cpu=solve(p,o,initial_y=y,patience=8,plateau_tolerance=0)
  o.backend=name
  gpu=solve(p,o,initial_y=y,patience=8,plateau_tolerance=0)
  assert cpu['termination_reason']==gpu['termination_reason']=='plateau'
  assert cpu['iterations_completed']==gpu['iterations_completed']==8
  assert cpu['candidates_evaluated']==gpu['candidates_evaluated']==256
  assert [r['best_welfare'] for r in cpu['trace']]==[r['best_welfare'] for r in gpu['trace']]
  np.testing.assert_array_equal(cpu['y'],gpu['y'])
