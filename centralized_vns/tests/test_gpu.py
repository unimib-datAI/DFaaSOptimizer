"""Run in a fresh process with NUMBA_ENABLE_CUDASIM=1, or on real CUDA."""
import numpy as np
import pytest
from centralized_vns.core import Options, solve
from centralized_vns.tests.test_core import tiny


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
