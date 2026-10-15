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


def test_shared_dispatch_and_fresh_inputs():
  from utils.compute_backend import compute_session
  args=list(fixture(500,5,17))
  for shift in (0.,.3):
    args[1]=args[1]+shift;args[5]=args[5]+shift;args[6]=~args[6]
    expected=relaxed_flows(*args)
    with compute_session({'compute_device':'cuda'}) as coverage:actual=relaxed_flows(*args)
    assert any(r['operation']=='relaxed_flows' and r['backend']=='cuda' for r in coverage)
    np.testing.assert_array_equal(actual[0],expected[0])
    for key in expected[1]:np.testing.assert_allclose(actual[1][key],expected[1][key],rtol=1e-9,atol=1e-9)


def test_gains_shared_kernel_and_clock(monkeypatch):
  from utils import step_time
  from utils.compute_backend import compute_session
  from models import bids_gpu
  from decentralized_frankwolfe import fw_gains_and_mask
  p,omega,*_=fixture(100,5);rng=np.random.default_rng(7)
  args=p,omega,rng.integers(-1,2,(100,100)).astype(float),rng.integers(0,2,(100,5)).astype(bool),rng.random((100,100)),rng.random((100,5)),_opts()
  expected=fw_gains_and_mask(*args)
  tick=[0.];monkeypatch.setattr(step_time,'perf_counter',lambda:tick[0])
  original=bids_gpu._kernels
  def delayed():tick[0]+=2.;return original()
  monkeypatch.setattr(bids_gpu,'_kernels',delayed)
  with compute_session({'compute_device':'cuda'}) as coverage:
    clock=step_time.start_step()
    with step_time.parallel_phase() as phase:
      phase.uniform(100)
      from models.frankwolfe_gpu import fw_gains_cuda
      actual=fw_gains_cuda(*args)
    timing=step_time.end_step(clock)
  assert timing['algorithm_s']==2.
  assert any(r['operation']=='fw_gains' and r['backend']=='cuda' for r in coverage)
  for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)


def test_iteration_clock_and_cuda_error_propagate(monkeypatch):
  from models import frankwolfe_gpu as gpu
  from utils import step_time
  from utils.compute_backend import compute_session
  tick=[0.];monkeypatch.setattr(step_time,'perf_counter',lambda:tick[0])
  kernels=gpu._kernels()
  def delayed():tick[0]+=3.;return kernels
  monkeypatch.setattr(gpu,'_kernels',delayed)
  with compute_session({'compute_device':'cuda'}):
    clock=step_time.start_step();relaxed_flows(*fixture(500,5,10));timing=step_time.end_step(clock)
  assert timing['algorithm_s']==3.
  def broken():raise RuntimeError('injected CUDA failure')
  monkeypatch.setattr(gpu,'_kernels',broken)
  with compute_session({'compute_device':'cuda'}),pytest.raises(RuntimeError,match='injected CUDA failure'):
    relaxed_flows(*fixture(500,5,10))


@pytest.mark.parametrize('repair_order',['price','gain'])
@pytest.mark.parametrize('rounding',['random','largest_remainder'])
def test_final_assignment_and_rng(repair_order,rounding):
  from decentralized_frankwolfe import frankwolfe_assignments
  from utils.compute_backend import compute_session
  n,nf=500,5;p,omega,base,r,rho,_,_,options=fixture(n,nf,17,.1,1.03)
  options.update(repair_order=repair_order,rounding=rounding)
  omega[n//2:]=0.;prev=np.zeros((n,n,nf));neighbors=np.ones((n,n))-np.eye(n)
  args=omega,prev,base,r,rho,_data(n,nf),neighbors,np.zeros((n,n)),np.zeros((n,nf)),options
  a=np.random.default_rng(45);b=np.random.default_rng(45)
  expected=frankwolfe_assignments(*args,a,params=p)
  with compute_session({'compute_device':'cuda'}) as coverage:actual=frankwolfe_assignments(*args,b,params=p)
  np.testing.assert_array_equal(actual[0],expected[0]);assert a.bit_generator.state==b.bit_generator.state
  assert not ((actual[0].sum(axis=0)>0)&(actual[0].sum(axis=1)>0)).any()
  for key in expected[1]:np.testing.assert_allclose(actual[1][key],expected[1][key],rtol=1e-9,atol=1e-9,err_msg=key)
  assert {'relaxed_flows'}<={r['operation'] for r in coverage if r['backend']=='cuda'}


def test_large_gap_tree_and_stopping_decisions():
  compare(fixture(160,5,9,.1,1.03))


def test_buffer_and_unsupported_domains_without_device(monkeypatch):
  from models import frankwolfe_gpu as gpu
  from utils.compute_backend import compute_session
  args=list(fixture(8,5))
  assert gpu.buffer_bytes(500,5)<64*1024*1024<gpu.buffer_bytes(1000,5)
  args[1]=args[1].astype(np.float32)
  valid,_=gpu.eligible(*args);assert not valid
  def forbidden():raise AssertionError('unexpected CUDA initialization')
  monkeypatch.setattr(gpu,'_kernels',forbidden)
  with compute_session({'compute_device':'cuda'}) as coverage:relaxed_flows(*args)
  assert coverage[-1]['backend']=='cpu'


@pytest.mark.parametrize('protected',[False,True])
@pytest.mark.parametrize('nf',[1,5])
def test_repair_independent_load_preparation(protected,nf):
  from models.frankwolfe_gpu import repair_memory_cuda
  from decentralized_frankwolfe import repair_memory
  p,omega,base,r,rho,gains,mask,options=fixture(30,nf)
  rng=np.random.default_rng(90);y=rng.integers(0,3,gains.shape).astype(float)
  old=np.minimum(y,rng.integers(0,2,y.shape)) if protected else None
  expected=repair_memory(y,p,gains,base,r,rho,protected=old)
  actual=repair_memory_cuda(y,p,gains,base,r,rho,protected=old)
  np.testing.assert_array_equal(actual[0],expected[0]);assert actual[1]==expected[1]


def test_float32_gains_keep_numpy_scalar_arithmetic():
  from decentralized_frankwolfe import fw_gains_and_mask
  from utils.compute_backend import compute_session
  p,omega,*_=fixture(100,5)
  options=_opts();options.update(latency_weight=.3,fairness_weight=.7)
  args=p,omega,np.ones((100,100)),np.zeros((100,5),bool),np.full((100,100),.7,np.float32),np.full((100,5),.3,np.float32),options
  expected=fw_gains_and_mask(*args)
  with compute_session({'compute_device':'cuda'}) as coverage:actual=fw_gains_and_mask(*args)
  for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
  assert coverage[-1]['backend']=='cpu'


@pytest.mark.parametrize('direction',[-1,0,1])
def test_large_gap_threshold(direction):
  args=list(fixture(100,5,2,.1,1.03));args[-1]['fw_tolerance']=0
  _,stats=relaxed_flows(*args);threshold=stats['relative_gap']
  args[-1]['iterations']=9
  args[-1]['fw_tolerance']=threshold if direction==0 else np.nextafter(threshold,-np.inf if direction<0 else np.inf)
  compare(args)
