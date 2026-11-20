"""Exact parity on real Apple hardware, not a CUDA simulator."""
import numpy as np
import pytest
from centralized_vns.core import CPU, Options, Problem, solve
from centralized_vns.tests.test_core import tiny
from centralized_vns.tests.test_moves import exchange_problem


@pytest.fixture
def metal():
  mx = pytest.importorskip('mlx.core')
  if not mx.metal.is_available():
    pytest.skip('Apple Metal GPU unavailable')
  return mx


@pytest.mark.parametrize('factory', [tiny, exchange_problem])
@pytest.mark.parametrize('seed', [0, 4850, 2147483645])
def test_metal_matches_every_cpu_iteration(metal, factory, seed):
  from centralized_vns.metal import Metal
  p = factory(); o = Options(pool_size=3, candidates=32, seed=seed)
  cpu, gpu = CPU(p, o), Metal(p, o)
  for iteration in range(40):
    np.testing.assert_array_equal(cpu.step(iteration), gpu.step(iteration))
    np.testing.assert_array_equal(cpu.moves, gpu.moves)
    for name in ('y', 'incoming', 'rejected'):
      np.testing.assert_array_equal(getattr(cpu.state, name), getattr(gpu.state, name))
    np.testing.assert_array_equal(cpu.best_y, gpu.best_y)
    np.testing.assert_array_equal(cpu.value, gpu.value)


def test_metal_float64_boundaries_and_near_ties(metal):
  p = Problem(load=[[8],[0],[0]], x=[[0],[0],[0]],
    capacity=[[1],[7-1e-10],[7-1e-7]], memory=[1], ram=[0,1,1],
    alpha=np.ones((3,1)), gamma=np.ones((3,1)),
    edges=[(0,1,0),(0,2,0)], beta=[2,2+1e-10])
  cpu = solve(p, Options(pool_size=3, candidates=32, iterations=30))
  gpu = solve(p, Options(pool_size=3, candidates=32, iterations=30, backend='metal'))
  np.testing.assert_array_equal(cpu['y'], gpu['y'])
  assert cpu['welfare'] == gpu['welfare']
  assert [t['best_welfare'] for t in cpu['trace']] == [t['best_welfare'] for t in gpu['trace']]


def test_metal_empty_and_memory_guard(metal):
  p = Problem(load=[[0]], x=[[0]], capacity=[[1]], memory=[1], ram=[1],
    alpha=[[1]], gamma=[[1]], edges=[], beta=[])
  assert solve(p, Options(backend='metal', iterations=2))['welfare'] == 0
  with pytest.raises(ValueError, match='memory'):
    solve(tiny(), Options(backend='metal', max_device_bytes=1))


def test_metal_cli_and_paired_benchmark(metal, tmp_path):
  import json
  import subprocess
  import sys
  from pathlib import Path
  root=Path(__file__).resolve().parents[2]
  common=[sys.executable,'-m','centralized_vns','centralized_vns/examples/tiny',
          '--output',str(tmp_path/'solution'),'--backend','metal',
          '--pool-size','2','--candidates','8','--iterations','12']
  subprocess.run(common,cwd=root,check=True,capture_output=True,text=True)
  result=json.loads((tmp_path/'solution/result.json').read_text())
  assert result['backend']=='metal' and result['feasible']
  assert result['welfare']==pytest.approx(8/3)
  args=[sys.executable,'-m','centralized_vns.benchmark','centralized_vns/examples/tiny',
        '--output',str(tmp_path/'paired'),'--pool-sizes','2','--candidates','8',
        '--iterations','12','--seeds','4850','--repeats','1']
  subprocess.run(args,cwd=root,check=True,capture_output=True,text=True)
  report=json.loads((tmp_path/'paired/comparison.json').read_text())
  assert report['pairs'][0]['allocations_equal']
  assert report['pairs'][0]['traces_equal']
  assert report['pairs'][0]['cpu']['welfare']==pytest.approx(8/3)
  assert subprocess.run(args,cwd=root,capture_output=True).returncode!=0


def test_real_instance_full_trajectory(metal):
  from pathlib import Path
  from centralized_vns.io import load_instance,initialize,from_data
  from centralized_vns.metal import Metal
  root=Path(__file__).resolve().parents[2]
  folder=root/'tests/data/harmonization/cilic-europe30-v1-delaunay__huawei-d1-f5-n30-t144x10m-ln1.6-0.5-s1-s1'
  data,_=load_instance(folder,0); x,_=initialize(data); p=from_data(data,x)
  o=Options(pool_size=4,candidates=32)
  cpu,gpu=CPU(p,o),Metal(p,o)
  for iteration in range(100):
    np.testing.assert_array_equal(cpu.step(iteration),gpu.step(iteration))
    np.testing.assert_array_equal(cpu.moves,gpu.moves)
    np.testing.assert_array_equal(cpu.state.y,gpu.state.y)
    np.testing.assert_array_equal(cpu.state.incoming,gpu.state.incoming)
    np.testing.assert_array_equal(cpu.state.rejected,gpu.state.rejected)
