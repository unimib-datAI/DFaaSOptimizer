import os
import numpy as np
import pytest
from models.numeric_gpu import residual_capacity_cuda
from run_faasmadea import compute_residual_capacity
from utils.compute_backend import compute_session

pytestmark = pytest.mark.skipif(os.environ.get('DFAAS_TEST_CUDA') != '1', reason='real CUDA opt-in required')


@pytest.mark.parametrize('n,f', [(1,1), (7,3), (32,5), (100,10)])
@pytest.mark.parametrize('strided', [False, True])
def test_residual_matches_cpu_updated_inputs(n, f, strided):
  rng = np.random.default_rng(7)
  x, y, r = rng.integers(0,10,(n,f)).astype(float), rng.integers(0,3,(n,n,f)).astype(float), rng.integers(0,8,(n,f)).astype(float)
  y[rng.random(y.shape) < .8] = 0
  demand = rng.uniform(.1, 2., (n,f)); utilization = rng.uniform(.1,1.,f)
  if strided:
    x, y, r = x[:,::-1], y[:,:,::-1], r[:,::-1]
  data={None:{'Nn':{None:n},'Nf':{None:f},'demand':{(i+1,j+1):demand[i,j] for i in range(n) for j in range(f)},'max_utilization':{j+1:utilization[j] for j in range(f)}}}
  before = [a.copy() for a in (x,y,r,demand,utilization)]
  for shift in (0., .25):
    expected = compute_residual_capacity(x+shift,y,r,data)
    actual = residual_capacity_cuda(x+shift,y,r,demand,utilization)
    for a,b in zip(actual,expected):
      np.testing.assert_array_equal(a,b)
  for a,b in zip((x,y,r,demand,utilization),before):
    np.testing.assert_array_equal(a,b)


def test_threshold_decisions_and_dispatch():
  n,f=32,1
  x=np.zeros((n,f));y=np.zeros((n,n,f));r=np.ones((n,f))
  x[:3,0] = [1.-1e-9, 1., 1.+1e-9]
  d={None:{'Nn':{None:n},'Nf':{None:f},'demand':{(i+1,1):1. for i in range(n)},'max_utilization':{1:1.}}}
  expected=compute_residual_capacity(x,y,r,d)
  with compute_session({'compute_device':'cuda'}):
    actual=compute_residual_capacity(x,y,r,d)
  for a,b in zip(actual,expected):
    np.testing.assert_array_equal(a,b)
  np.testing.assert_array_equal(actual[1]>0,expected[1]>0)
  zero=residual_capacity_cuda(x*0,y*0,r*0,np.ones_like(r),np.ones(f))
  assert not any(a.any() for a in zero)


@pytest.mark.parametrize('n', [1, 8, 32, 100, 500])
def test_fractional_source_reduction_order(n):
  rng = np.random.default_rng(9)
  f = 3
  x = rng.random((n,f))
  y = rng.random((n,n,f)) * .001
  r = rng.random((n,f)) * 10
  data = {None:{'Nn':{None:n},'Nf':{None:f},'demand':{(i+1,j+1):.7 for i in range(n) for j in range(f)},'max_utilization':{j+1:.8 for j in range(f)}}}
  expected = compute_residual_capacity(x,y,r,data)
  actual = residual_capacity_cuda(x,y,r,np.full((n,f),.7),np.full(f,.8))
  for a,b in zip(actual,expected):
    np.testing.assert_array_equal(a,b)


def test_enabled_domain_dispatches_real_gpu(monkeypatch):
  from models import numeric_gpu
  original = numeric_gpu.residual_capacity_cuda
  calls = []
  def recording(*args):
    calls.append(True)
    return original(*args)
  monkeypatch.setattr(numeric_gpu,'residual_capacity_cuda',recording)
  n,f=100,10
  x=np.ones((n,f)); y=np.zeros((n,n,f)); r=np.ones((n,f))
  data={None:{'Nn':{None:n},'Nf':{None:f},'demand':{(i+1,j+1):1. for i in range(n) for j in range(f)},'max_utilization':{j+1:1. for j in range(f)}}}
  expected=compute_residual_capacity(x,y,r,data)
  with compute_session({'compute_device':'cuda'}):
    actual=compute_residual_capacity(x,y,r,data)
  assert calls == [True]
  for a,b in zip(actual,expected):
    np.testing.assert_array_equal(a,b)
