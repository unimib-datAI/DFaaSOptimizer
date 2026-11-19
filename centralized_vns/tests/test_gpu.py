"""Run in a fresh process with NUMBA_ENABLE_CUDASIM=1, or on real CUDA."""
import numpy as np
import pytest
from centralized_vns.core import Options, solve
from centralized_vns.tests.test_core import tiny


def test_selection_ties_shaking_tail_and_parallel_archive():
  pytest.importorskip('numba')
  from centralized_vns.core import CPU,Problem
  from centralized_vns.gpu import CUDA
  p=Problem(load=[[3]]+[[0]]*69,x=np.zeros((70,1)),capacity=np.ones((70,1))*2,
    memory=[1],ram=[0]+[1]*69,alpha=np.ones((70,1)),gamma=np.ones((70,1)),
    edges=[(0,j,0) for j in range(1,70)],beta=np.ones(69))
  o=Options(pool_size=8,candidates=130,shake_after=2)
  cpu,gpu=CPU(p,o),CUDA(p,o)
  cpu.moves[:,:,:,0]=-1;cpu.moves[:,:,:,1]=0
  gains=np.full((8,130),np.nan)
  def move(row,c,g,e=0):
    cpu.moves[row,c,0]=[e,1];gains[row,c]=g
  move(0,63,2.);move(0,64,2.,1)
  move(1,63,-1.);move(1,64,9.,1)
  gains[2,0]=100.;move(2,1,np.inf);move(2,2,-np.inf);move(2,129,2.,68)
  move(3,129,5e-13)
  move(5,0,1.);move(5,129,1.+1e-13,68)
  cpu.moves[6,129]=[[68,1],[68,1],[68,-1],[-1,0]];gains[6,129]=2.
  move(7,129,2.,68);cpu.best[7]+=100.
  cpu.stale[[1,4]]=6
  gpu.moves.copy_to_device(cpu.moves);gpu.gains.copy_to_device(gains)
  gpu.stale.copy_to_device(cpu.stale);gpu.best.copy_to_device(cpu.best)
  for index in range(8):
    cpu.accept(index,3 if cpu.stale[index]>=6 else 0,gains[index])
  # Both layouts cover all trajectories; the new kernel cooperates within blocks.
  gpu.accept_kernel[8,64](o.shake_after,gpu.level,gpu.stale,gpu.moves,gpu.gains,
    gpu.y,gpu.inc,gpu.z,gpu.value,gpu.best,gpu.best_y,*gpu.data[5:8])
  for expected,actual in ((cpu.state.y,gpu.y),(cpu.state.incoming,gpu.inc),
      (cpu.state.rejected,gpu.z),(cpu.value,gpu.value),(cpu.best,gpu.best),
      (cpu.best_y,gpu.best_y),(cpu.level,gpu.level),(cpu.stale,gpu.stale)):
    np.testing.assert_array_equal(expected,actual.copy_to_host())
  assert cpu.state.y[0,0]==1 and cpu.state.y[0,1]==0
  assert cpu.value[1]==p.base_welfare-1.
  assert cpu.best_y[2,68]==cpu.best_y[5,68]==cpu.best_y[6,68]==1
  assert not cpu.best_y[7].any()


@pytest.mark.parametrize('candidates',[1,17,65])
def test_candidate_counts_across_block_boundary(candidates):
  pytest.importorskip('numba')
  p=tiny()
  cpu=solve(p,Options(pool_size=2,candidates=candidates,iterations=3))
  gpu=solve(p,Options(pool_size=2,candidates=candidates,iterations=3,backend='cuda'))
  np.testing.assert_array_equal(cpu['y'],gpu['y'])
  np.testing.assert_array_equal([t['best_welfare'] for t in cpu['trace']],
                              [t['best_welfare'] for t in gpu['trace']])


def test_population_matches_cpu():
  pytest.importorskip('numba')
  p=tiny()
  cpu=solve(p,Options(pool_size=3,candidates=12,iterations=12))
  gpu=solve(p,Options(pool_size=3,candidates=12,iterations=12,backend='cuda'))
  np.testing.assert_array_equal(cpu['y'],gpu['y'])
  np.testing.assert_allclose([t['best_welfare'] for t in cpu['trace']],
                             [t['best_welfare'] for t in gpu['trace']],atol=1e-12)
  assert cpu['candidates_evaluated']==gpu['candidates_evaluated']


def test_device_memory_guard():
  pytest.importorskip('numba')
  with pytest.raises(ValueError,match='memory'):
    solve(tiny(),Options(backend='cuda',max_device_bytes=1))


def test_compound_scores_and_trajectories_match_cpu():
  pytest.importorskip('numba')
  from centralized_vns.core import CPU
  from centralized_vns.gpu import CUDA,kernels
  from centralized_vns.tests.test_moves import exchange_problem
  from centralized_vns.moves import score_move
  import itertools
  p=exchange_problem();o=Options(pool_size=2,candidates=16)
  cpu=CPU(p,o);gpu=CUDA(p,o)
  for iteration in range(20):
    np.testing.assert_allclose(cpu.step(iteration),gpu.step(iteration),atol=1e-12)
    np.testing.assert_array_equal(cpu.state.y,gpu.y.copy_to_host())
    np.testing.assert_array_equal(cpu.state.incoming,gpu.inc.copy_to_host())
    np.testing.assert_array_equal(cpu.state.rejected,gpu.z.copy_to_host())
  # Exhaustively evaluate atomic exchanges, including infeasible combinations.
  cuda,_,_=kernels()
  device_score=cuda.jit(device=True)(score_move)
  globals()['cuda']=cuda
  @cuda.jit
  def check(moves,gains,y,inc,z,load,x,capacity,memory,ram,src,dst,fun,reward):
    # No CUDA intrinsics: a single test thread checks this bounded table.
    for c in range(gains.size):
      gains[c]=device_score(0,c,moves,y,inc,z,load,x,capacity,memory,ram,src,dst,fun,reward)
  moves=np.array([[list(enumerate(ds)) for ds in itertools.product((-1,0,1),repeat=4)]],dtype=np.int64)
  expected=np.array([score_move(0,c,moves,cpu.state.y,cpu.state.incoming,cpu.state.rejected,*p.arrays()) for c in range(moves.shape[1])])
  actual=cuda.to_device(np.zeros(len(expected)))
  check[1,1](cuda.to_device(moves),actual,gpu.y,gpu.inc,gpu.z,*gpu.data)
  np.testing.assert_allclose(expected,actual.copy_to_host(),atol=1e-12)
