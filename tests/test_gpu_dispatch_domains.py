"""CPU-only regressions for inputs outside validated CUDA domains."""
import numpy as np
import pandas as pd
import pytest
from run_faasmadea import compute_residual_capacity
from madea2 import start_replicas_by_value
from utils.compute_backend import compute_session
from test_dual_helpers import make_data

@pytest.mark.parametrize('domain', ['arrays', 'parameters', 'mixed_parameters'])
def test_residual_unsupported_dtypes_preserve_cpu_threshold(domain, monkeypatch):
  from models import numeric_gpu
  def forbidden(*args):pytest.fail('unsupported arithmetic reached CUDA')
  monkeypatch.setattr(numeric_gpu,'residual_capacity_cuda',forbidden)
  n,nf=100,10; data=make_data(n,nf)
  x=np.zeros((n,nf)); y=np.zeros((n,n,nf)); r=np.full((n,nf),2.)
  for f in range(nf):
    data[None]['max_utilization'][f+1]=1.
    for i in range(n):data[None]['demand'][i+1,f+1]=1.
  if domain=='arrays':
    x=x.astype(np.float32); y=y.astype(np.float32); r=r.astype(np.float32)
    y[:10,:,:]=np.float32(.1)
  else:
    u=np.float32(.7); d=np.float32(.3) if domain=='parameters' else .3
    data[None]['max_utilization'][1]=u; data[None]['demand'][1,1]=d
    r[0,0]=1.; x[0,0]=float(u/d)-1.
  expected=compute_residual_capacity(x,y,r,data)
  assert expected[1][0,0]==1.
  with compute_session({'compute_device':'cuda'}) as coverage:
    actual=compute_residual_capacity(x,y,r,data)
  for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
  np.testing.assert_array_equal(actual[1]>=1,expected[1]>=1)
  assert coverage[-1]['backend']=='cpu' and 'dtype' in coverage[-1]['reason']

def test_categorical_function_ties_preserve_cpu_order(monkeypatch):
  from models import replicas_gpu
  def forbidden(*args):pytest.fail('categorical grouping reached CUDA')
  monkeypatch.setattr(replicas_gpu,'start_replicas_by_value_cuda',forbidden)
  n,nf=30,5; data=make_data(n,nf)
  for f in range(nf):
    data[None]['memory_requirement'][f+1]=1.; data[None]['max_utilization'][f+1]=1.
    for j in range(n):data[None]['demand'][j+1,f+1]=1.
  bids=pd.DataFrame([dict(i=k,j=j,f=f,d=4.,v=2.) for j in range(n) for f in range(nf) for k in range(4)])
  bids['f']=pd.Categorical(bids['f'],categories=[4,3,2,1,0],ordered=True)
  r=np.zeros((n,nf)); rho=np.ones(n)
  expected=start_replicas_by_value(bids,r,data,rho)
  np.testing.assert_array_equal(expected[0],np.tile([0,0,0,0,1],(n,1)))
  with compute_session({'compute_device':'cuda'}) as coverage:
    actual=start_replicas_by_value(bids,r,data,rho)
  for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
  assert coverage[-1]['backend']=='cpu' and 'integer' in coverage[-1]['reason']
