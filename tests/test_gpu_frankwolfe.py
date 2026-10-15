import os
from copy import deepcopy
import numpy as np
import pytest
from decentralized_frankwolfe import relaxed_flows, round_flows
from test_frankwolfe_helpers import _data, _opts, fw_parameters

pytestmark = pytest.mark.skipif(os.environ.get('DFAAS_TEST_CUDA') != '1', reason='real CUDA opt-in required')


def fixture(n=8,nf=5,iterations=17,smoothing=0.,growth=1.):
  rng=np.random.default_rng(74)
  p=fw_parameters(_data(n,nf));p['demand']=rng.choice([.3,.7,1.,2.],(n,nf))
  p['memory_requirement']=rng.choice([.3,.7,2.,5.],nf)
  omega=rng.choice([0.,1.,3.,7.],(n,nf));base=rng.random((n,nf))
  replicas=rng.integers(0,3,(n,nf)).astype(float);rho=rng.choice([0.,1.,3.],n)
  gains=rng.choice([0.,1e-12,1.,2.,3.],(n,n,nf));mask=rng.integers(0,2,gains.shape).astype(bool)
  options=_opts();options.update(iterations=iterations,smoothing=smoothing,kappa_growth=growth)
  return p,omega,base,replicas,rho,gains,mask,options


def compare(args):
  from models.frankwolfe_gpu import relaxed_flows_cuda
  before=deepcopy(args);q,stats=relaxed_flows(*args);actual,info=relaxed_flows_cuda(*args)
  np.testing.assert_allclose(actual,q,rtol=1e-9,atol=1e-9)
  assert info['iterations']==stats['iterations']
  for k in stats:np.testing.assert_allclose(info[k],stats[k],rtol=1e-9,atol=1e-9,err_msg=k)
  for mode in ('random','largest_remainder'):
    a=np.random.default_rng(74);b=np.random.default_rng(74)
    np.testing.assert_array_equal(round_flows(actual,a,mode),round_flows(q,b,mode))
    assert a.bit_generator.state==b.bit_generator.state
  for a,b in zip(args[1:7],before[1:7]):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('n,nf',[(1,1),(3,1),(8,5),(30,5)])
@pytest.mark.parametrize('iterations',[0,1,17])
@pytest.mark.parametrize('smoothing,growth',[(0.,1.),(.1,1.03)])
def test_resident_relaxation(n,nf,iterations,smoothing,growth):
  compare(fixture(n,nf,iterations,smoothing,growth))


@pytest.mark.parametrize('case',['empty_mask','zero_load','ties','zero_ram_capacity','slope','adjusted_zero'])
def test_adversarial_domains(case):
  args=list(fixture());p,omega,base,r,rho,gains,mask,options=args
  if case=='empty_mask':mask[:]=False
  if case=='zero_load':omega[:]=0
  if case=='ties':gains[:]=2.;rho[:]=1e6
  if case=='zero_ram_capacity':r[:]=0;rho[:]=0
  if case=='slope':base[:]=np.nextafter(1e-12,np.inf);r[:]=0
  if case=='adjusted_zero':gains[:]=0
  compare(args)


@pytest.mark.parametrize('direction',[-1,0,1])
def test_gap_at_stopping_threshold(direction):
  args=list(fixture(iterations=2));args[-1]['fw_tolerance']=0
  _,stats=relaxed_flows(*args);threshold=stats['relative_gap']
  args[-1]['iterations']=9
  args[-1]['fw_tolerance']=threshold if direction==0 else np.nextafter(threshold,-np.inf if direction<0 else np.inf)
  compare(args)
