import os
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest

from decentralized_dual import pair_scores, buyer_price_response
from models.bids_gpu import pair_scores_cuda, buyer_price_response_cuda
from test_dual_helpers import make_data
from models.bids_gpu import define_bids_cuda, define_assignments_cuda
from run_faasmadea import define_bids
from decentralized_diffusion import define_assignments
from models.bids_gpu import sample_assignments_cuda
from decentralized_powerd import sample_assignments

pytestmark = pytest.mark.skipif(os.environ.get('DFAAS_TEST_CUDA') != '1', reason='real CUDA opt-in required')


@pytest.mark.parametrize('n,nf', [(1,1), (8,3), (30,5), (100,3)])
def test_pair_scores_exact_and_fresh(n, nf):
  rng = np.random.default_rng(902)
  data = make_data(n,nf)
  beta = rng.choice([-1.,0.,1e-9,1.,2.], (n,n,nf))
  data[None]['beta'] = {(i+1,j+1,f+1):beta[i,j,f] for i in range(n) for j in range(n) for f in range(nf)}
  neighbors = rng.integers(0,2,(n,n)); neighbors[0] = 0
  latency = rng.random((n,n)); fairness = rng.random((n,nf))
  options = dict(latency_weight=.3, fairness_weight=.7)
  before = deepcopy(data)
  for shift in (0.,1e-9):
    args = data,neighbors,latency+shift,fairness,options
    expected = pair_scores(*args)
    actual = pair_scores_cuda(*args)
    for a,b in zip(actual,expected): np.testing.assert_array_equal(a,b)
  assert data == before


def test_pair_scores_sparse_latency_like_production():
  from scipy.sparse import csr_array
  n,nf=8,3;args=auction_fixture(n,nf)
  data,neighbors,options,latency,fairness=args[3],args[4],args[6],csr_array(args[7]),args[8]
  expected=pair_scores(data,neighbors,latency,fairness,options)
  actual=pair_scores_cuda(data,neighbors,latency,fairness,options)
  for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('n,nf', [(1,1), (8,3), (30,5), (100,3)])
@pytest.mark.parametrize('prices', [0.,1.,100.])
def test_buyer_waterfill_order_and_reductions(n,nf,prices):
  rng = np.random.default_rng(901)
  omega = rng.choice([0.,.5,1.,3.,5.+1e-9],(n,nf))
  capacity = rng.choice([0.,.5,1.,2.,4.],(n,nf))
  lam = rng.random((n,nf))*prices
  scores = rng.choice([-np.inf,-1.,0.,1.,2.,3.],(n,n,nf))
  eligible = rng.integers(0,2,scores.shape).astype(bool) & np.isfinite(scores)
  arrays = omega,capacity,lam,scores,eligible
  before = [a.copy() for a in arrays]
  expected = buyer_price_response(*arrays)
  actual = buyer_price_response_cuda(*arrays)
  pd.testing.assert_frame_equal(actual[0],expected[0],check_exact=True)
  np.testing.assert_array_equal(actual[1],expected[1])
  assert actual[2] == expected[2]
  for a,b in zip(arrays,before): np.testing.assert_array_equal(a,b)


def auction_fixture(n=8,nf=3):
  rng=np.random.default_rng(345)
  data=make_data(n,nf)
  data[None]['incoming_load']={(i+1,f+1):float(i%4) for i in range(n) for f in range(nf)}
  data[None]['beta']={(i+1,j+1,f+1):float((i+j+f)%4-1) for i in range(n) for j in range(n) for f in range(nf)}
  omega=rng.choice([0.,.5,1.,3.,7.],(n,nf));omega[:2]=0
  capacity=rng.choice([0.,.5,1.,2.,4.],(n,nf))
  prices=rng.random((n,nf));neighbors=rng.integers(0,2,(n,n));neighbors[0]=0
  rho=rng.choice([0.,1.,2.,5.],n)
  return omega,capacity,prices,data,neighbors,rho,dict(latency_weight=.3,fairness_weight=.7,epsilon=.01),rng.random((n,n)),rng.random((n,nf))


@pytest.mark.parametrize('unit', [False,True])
@pytest.mark.parametrize('internal', [False,True])
@pytest.mark.parametrize('force', [False,True])
def test_madea_bids_preserve_order(unit,internal,force):
  args=list(auction_fixture());args[6].update(unit_bids=unit,use_internal_value=internal)
  expected=define_bids(*args,force)
  actual=define_bids_cuda(*args,force)
  for a,b in zip(actual[:2],expected[:2]):pd.testing.assert_frame_equal(a,b,check_exact=True)
  assert actual[2]==expected[2]


@pytest.mark.parametrize('unit', [False,True])
@pytest.mark.parametrize('force', [False,True])
def test_diffusion_bids_preserve_order(unit,force):
  args=list(auction_fixture());args.pop(2);args[5].update(unit_bids=unit)
  expected=define_assignments(*args,force)
  actual=define_assignments_cuda(*args,force)
  for a,b in zip(actual[:2],expected[:2]):pd.testing.assert_frame_equal(a,b,check_exact=True)
  assert actual[2]==expected[2]


@pytest.mark.parametrize('mode',['beta','bid'])
def test_madea2_known_memory_values(mode,monkeypatch):
  import madea2
  args=list(auction_fixture());args[6].update(unit_bids=False,memory_bid_value=mode)
  expected=madea2.define_bids(*args,True)
  calls=[]
  def adapter(*a,**kw):
    calls.append(True)
    return define_bids_cuda(*a,**kw)
  monkeypatch.setattr(madea2,'_madea_bids',adapter)
  actual=madea2.define_bids(*args,True)
  assert calls
  from utils.compute_backend import compute_session
  with compute_session({'compute_device':'cuda'}) as coverage:
    actual=madea2.define_bids(*args,True)
  assert any(row['backend']=='cuda' and row['operation']=='define_bids' for row in coverage)
  for a,b in zip(actual[:2],expected[:2]):pd.testing.assert_frame_equal(a,b,check_exact=True)


@pytest.mark.parametrize('mode',['beta','bid'])
def test_madea2_does_not_read_unrequested_memory_values(mode,monkeypatch):
  import madea2
  args=list(auction_fixture(3,1));args[0][:]=0;args[0][0,0]=1
  args[1][:]=0;args[1][1,0]=2;args[2][:]=0
  args[3][None]['beta']={(1,2,1):4.}
  args[4][:]=0;args[4][0,1:]=1;args[5][:]=0;args[5][2]=2
  args[6].update(unit_bids=False,memory_bid_value=mode)
  expected=madea2.define_bids(*args,False)
  monkeypatch.setattr(madea2,'_madea_bids',define_bids_cuda)
  actual=madea2.define_bids(*args,False)
  for a,b in zip(actual[:2],expected[:2]):pd.testing.assert_frame_equal(a,b,check_exact=True)


def test_custom_memory_callback_retains_calls_and_rng():
  from utils.compute_backend import compute_session
  args=list(auction_fixture());args[6].update(unit_bids=False)
  def execute(call):
    rng=np.random.default_rng(391);seen=[]
    def callback(i,j,f):
      seen.append((i,j,f));return rng.random()
    result=call(*args,True,memory_bid_value=callback)
    return result,seen,rng.bit_generator.state
  expected,seen,state=execute(define_bids)
  with compute_session({'compute_device':'cuda'}) as coverage:
    actual,other_seen,other_state=execute(define_bids_cuda)
  assert seen==other_seen and state==other_state
  assert coverage[-1]['backend']=='cpu' and 'callback' in coverage[-1]['reason']
  for a,b in zip(actual[:2],expected[:2]):pd.testing.assert_frame_equal(a,b,check_exact=True)


@pytest.mark.parametrize('unit',[False,True])
@pytest.mark.parametrize('criterion',['score','capacity'])
def test_powerd_preserves_exact_samples_and_rng(unit,criterion):
  args=list(auction_fixture());args.pop(2);args[5].update(d=2,unit_bids=unit,criterion=criterion)
  cpu_rng=np.random.default_rng(929);gpu_rng=np.random.default_rng(929)
  expected=sample_assignments(*args,True,cpu_rng)
  actual=sample_assignments_cuda(*args,True,gpu_rng)
  assert cpu_rng.bit_generator.state==gpu_rng.bit_generator.state
  for a,b in zip(actual[:2],expected[:2]):pd.testing.assert_frame_equal(a,b,check_exact=True)
  assert actual[2]==expected[2]


@pytest.mark.parametrize('kind',['diffusion','powerd'])
def test_price_free_methods_ignore_auction_internal_value(kind):
  args=list(auction_fixture());args.pop(2);args[5].update(unit_bids=False,use_internal_value=True,d=2)
  if kind=='diffusion':
    expected=define_assignments(*args,True);actual=define_assignments_cuda(*args,True)
  else:
    expected=sample_assignments(*args,True,np.random.default_rng(34))
    actual=sample_assignments_cuda(*args,True,np.random.default_rng(34))
  for a,b in zip(actual[:2],expected[:2]):pd.testing.assert_frame_equal(a,b,check_exact=True)


@pytest.mark.parametrize('kind',['madea','diffusion'])
def test_fractional_request_domain_stays_cpu(kind,monkeypatch):
  import run_faasmadea,decentralized_diffusion
  from utils.compute_backend import compute_session
  args=list(auction_fixture());args[6]['unit_bids']=False
  if kind=='madea':
    monkeypatch.setattr(run_faasmadea,'VAR_TYPE',float)
    cpu,gpu=define_bids,define_bids_cuda
  else:
    monkeypatch.setattr(decentralized_diffusion,'VAR_TYPE',float);args.pop(2)
    cpu,gpu=define_assignments,define_assignments_cuda
  expected=cpu(*args,True)
  with compute_session({'compute_device':'cuda'}) as coverage:actual=gpu(*args,True)
  for a,b in zip(actual[:2],expected[:2]):pd.testing.assert_frame_equal(a,b,check_exact=True)
  assert coverage[-1]['backend']=='cpu'


def test_dual_round_dense_dispatch_decisions_and_shared_clock():
  from decentralized_dual import dual_coordination_round
  from utils.compute_backend import compute_session
  from utils import step_time
  n,nf=100,5;data=make_data(n,nf)
  omega=np.zeros((n,nf));omega[:20]=3.
  capacity=np.full((n,nf),2.);neighbors=np.ones((n,n))-np.eye(n)
  latency=np.zeros((n,n));fairness=np.zeros((n,nf));rho=np.full(n,20.)
  options=dict(max_inner_iterations=3,alpha0=.1,theta=.5,gap_tolerance=0.,step_rule='sqrt',
               latency_weight=0.,fairness_weight=0.)
  args=omega,capacity,data,neighbors,rho,options,latency,fairness
  expected=dual_coordination_round(*args)
  clock=step_time.start_step()
  try:
    with compute_session({'compute_device':'cuda'}) as coverage:actual=dual_coordination_round(*args)
  finally:timing=step_time.end_step(clock)
  assert timing['algorithm_s']>0
  assert any(r['operation']=='pair_scores' and r['backend']=='cuda' for r in coverage)
  np.testing.assert_array_equal(actual[0],expected[0]);pd.testing.assert_frame_equal(actual[1],expected[1],check_exact=True)
  assert actual[2].keys()==expected[2].keys() and actual[3]==expected[3]
  for name in actual[2]:np.testing.assert_array_equal(actual[2][name],expected[2][name])


def test_sparse_large_buyer_dispatch():
  from utils.compute_backend import compute_session
  from utils import step_time
  n,nf=500,5;omega=np.ones((n,nf));capacity=np.full_like(omega,.5);lam=np.zeros_like(omega)
  scores=np.full((n,n,nf),-np.inf);eligible=np.zeros(scores.shape,dtype=bool)
  for i in range(n):
    for j in (i+np.arange(1,7))%n:scores[i,j]=1.;eligible[i,j]=True
  args=omega,capacity,lam,scores,eligible
  expected=buyer_price_response(*args)
  clock=step_time.start_step()
  try:
    with compute_session({'compute_device':'cuda'}) as coverage:actual=buyer_price_response(*args)
  finally:timing=step_time.end_step(clock)
  assert timing['algorithm_s']>0
  assert any(r['operation']=='buyer_price_response' and r['backend']=='cpu' and 'performance' in r['reason'] for r in coverage)
  pd.testing.assert_frame_equal(actual[0],expected[0],check_exact=True)
  np.testing.assert_array_equal(actual[1],expected[1]);assert actual[2]==expected[2]
