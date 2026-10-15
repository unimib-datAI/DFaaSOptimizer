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
from models.replicas_gpu import start_additional_replicas_cuda,start_replicas_by_value_cuda
from run_faasmadea import start_additional_replicas
from madea2 import start_replicas_by_value
from models.numeric_gpu import aggregate_groups_cuda,dual_price_cuda
from models.bids_gpu import generate_level_requests_cuda
from models.bids_gpu import evaluate_assignments_cuda
from decentralized_diffusion import evaluate_assignments
from models.bids_gpu import best_agent_bids_cuda,resolve_gcaa_round_cuda
from decentralized_gcaa import resolve_gcaa_round
from models.bids_gpu import dual_memory_requests_cuda

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


@pytest.mark.parametrize('n,nf',[(1,1),(8,3),(100,7)])
@pytest.mark.parametrize('capacity',[.1,.4,.7,1.-1e-9,1.,1.+1e-9,3.])
def test_seller_policies_exact_and_fresh(n,nf,capacity):
  rng=np.random.default_rng(56);data=make_data(n,nf)
  for j in range(n):
    for f in range(nf):data[None]['demand'][j+1,f+1]=.7/capacity
  # Include Python float floor-division boundaries and fractional memory sizes.
  data[None]['memory_requirement']={f+1:[1.,2.3,.7][f%3] for f in range(nf)}
  replicas=rng.integers(0,6,(n,nf)).astype(float);rho=rng.integers(0,30,n).astype(float)
  rows=[dict(i=k%max(n,1),j=j,f=f,d=float(k%4)+.1,v=float(k%3))
        for j in range(n) for f in range(nf) for k in range(1+(j+f)%4)]
  bids=pd.DataFrame(rows);before=deepcopy(data),replicas.copy(),rho.copy(),bids.copy()
  for shift in (0.,1.):
    for cpu,gpu in [(start_additional_replicas,start_additional_replicas_cuda),
                    (start_replicas_by_value,start_replicas_by_value_cuda)]:
      expected=cpu(bids,replicas+shift,data,rho)
      actual=gpu(bids,replicas+shift,data,rho)
      for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
      assert (actual[0]>=0).all() and (actual[0]==np.floor(actual[0])).all()
      assert (actual[1]>=-1e-9).all()
  assert data==before[0];np.testing.assert_array_equal(replicas,before[1]);np.testing.assert_array_equal(rho,before[2]);pd.testing.assert_frame_equal(bids,before[3])


def test_empty_seller_batches():
  bids=pd.DataFrame(columns=['i','j','f','d','v']);r=np.zeros((3,2));rho=np.ones(3);data=make_data(3,2)
  for cpu,gpu in [(start_additional_replicas,start_additional_replicas_cuda),
                  (start_replicas_by_value,start_replicas_by_value_cuda)]:
    for a,b in zip(gpu(bids,r,data,rho),cpu(bids,r,data,rho)):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('kwargs',[
  {},dict(free=2,quantities=(1,)),dict(existing=(1,),quantities=(1,)),
  dict(capacity=(.4,),free=4),dict(capacity=(.1,1.),existing=(0,0),free=10,quantities=(10,10),values=(1,1)),
  dict(capacity=(.3,),existing=(9,),free=1,quantities=(1,)),
  dict(capacity=(.7,),existing=(29,),free=1,quantities=(1,)),
  dict(capacity=(.3333333332,),free=3,quantities=(1,)),
  dict(capacity=(.3333333332,),free=4,quantities=(1,)),
])
def test_productive_replica_regressions_on_device(kwargs):
  from test_productive_replicas import case
  args=case(**kwargs)
  for cpu,gpu in [(start_additional_replicas,start_additional_replicas_cuda),
                  (start_replicas_by_value,start_replicas_by_value_cuda)]:
    for a,b in zip(gpu(*args),cpu(*args)):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('value_policy',[False,True])
def test_continuous_replica_policy_stays_cpu(value_policy,monkeypatch):
  import run_faasmadea
  from utils.compute_backend import compute_session
  from test_productive_replicas import case
  monkeypatch.setattr(run_faasmadea,'VAR_TYPE',float)
  args=case(quantities=(.5,))
  cpu,gpu=(start_replicas_by_value,start_replicas_by_value_cuda) if value_policy else (start_additional_replicas,start_additional_replicas_cuda)
  expected=cpu(*args)
  with compute_session({'compute_device':'cuda'}) as coverage:actual=gpu(*args)
  for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
  assert coverage[-1]['backend']=='cpu' and 'continuous' in coverage[-1]['reason']


@pytest.mark.parametrize('value_policy,n',[(False,100),(True,30)])
def test_seller_production_dispatch_and_clock(value_policy,n):
  from utils.compute_backend import compute_session
  from utils import step_time
  nf=5;data=make_data(n,nf)
  bids=pd.DataFrame([dict(i=k,j=j,f=f,d=4.,v=2.) for j in range(n) for f in range(nf) for k in range(4)])
  r=np.zeros((n,nf));rho=np.full(n,100.)
  call=start_replicas_by_value if value_policy else start_additional_replicas
  expected=call(bids,r,data,rho)
  clock=step_time.start_step()
  try:
    with compute_session({'compute_device':'cuda'}) as coverage:actual=call(bids,r,data,rho)
  finally:timing=step_time.end_step(clock)
  for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
  assert any(row['operation']==('replicas_value' if value_policy else 'replicas_proportional') and row['backend']=='cuda' for row in coverage)
  assert timing['algorithm_s']>=.9*timing['wall_s']


@pytest.mark.parametrize('n,nf',[(1,1),(8,1),(129,1),(500,1),(8,3),(500,5)])
def test_ordered_group_sums_shared_members_and_fresh_inputs(n,nf):
  rng=np.random.default_rng(309)
  values=rng.random((n,nf))*rng.choice([1e-9,1.,1e9],(n,nf))
  groups=[[],[0],list(range(n)),list(range(n-1,-1,-1)),list(range(n//2,n))]
  offsets=np.array([0]+list(np.cumsum([len(g) for g in groups])),dtype=np.int64)
  members=np.array([i for g in groups for i in g],dtype=np.int64)
  before=values.copy()
  for delta in (0.,1e-9):
    expected=np.array([(values+delta)[g,:].sum(axis=0) for g in groups])
    actual=aggregate_groups_cuda(values+delta,members,offsets)
    np.testing.assert_array_equal(actual,expected)
  np.testing.assert_array_equal(values,before)
  assert aggregate_groups_cuda(values,[],[0]).shape==(0,nf)


@pytest.mark.parametrize('alpha',[0.,1e-9,.3,1.])
def test_dual_price_clipping_and_thresholds(alpha):
  lam=np.array([[0.,1e-9,1.,1.],[-1.,0.,1.,2.]])[:,::-1]
  gradient=np.array([[-1.,1.,-1.-1e-9,-1.+1e-9],[1.,0.,2.,-3.]])[:,::-1]
  before=lam.copy(),gradient.copy()
  for shift in (0.,.5):
    np.testing.assert_array_equal(dual_price_cuda(lam+shift,alpha,gradient),np.maximum(0.,lam+shift+alpha*gradient))
  np.testing.assert_array_equal(lam,before[0]);np.testing.assert_array_equal(gradient,before[1])


def test_group_token_sums_preserve_int64_and_overflow():
  values=np.array([[2**62,3],[2**62,7],[-1,11]],dtype=np.int64)
  members=np.array([0,1,2,2,1],dtype=np.int64);offsets=[0,3,3,5]
  expected=np.array([values[[0,1,2],:].sum(axis=0),values[[],:].sum(axis=0),values[[2,1],:].sum(axis=0)])
  actual=aggregate_groups_cuda(values,members,offsets)
  assert actual.dtype==expected.dtype
  np.testing.assert_array_equal(actual,expected)


def test_empty_group_phase_remains_cpu():
  from utils import step_time
  from utils.compute_backend import compute_session
  with compute_session({'compute_device':'cuda'}) as coverage,step_time.parallel_phase() as phase:
    actual=aggregate_groups_cuda(np.zeros((3,2)),[],[0])
    assert not phase.shared_device
  assert actual.shape==(0,2) and coverage[-1]['backend']=='cpu'


def test_hierarchy_group_fields_dispatch_and_fresh_values():
  from hierarchical_auction.engine import HierarchicalAuctionEngine
  from hierarchical_auction.structure import Structure
  from hierarchical_auction.token_manager import CapacityTokenManager
  from utils.compute_backend import compute_session
  n,nf=600,5;rng=np.random.default_rng(31)
  engine=HierarchicalAuctionEngine(np.eye(n),nf,np.ones(nf))
  structures={i:Structure(2,i,set((np.arange(500)+i)%n),set(),nf) for i in range(500)}
  values=rng.random((n,nf))
  manager=CapacityTokenManager(np.full((n,nf),20.),np.ones(nf))
  with compute_session({'compute_device':'cuda'}) as coverage:
    for shift in (0.,1e-9):
      engine._aggregate_residual_demand(structures,values+shift)
      engine._populate_indicative_tokens(structures,manager)
      for s in structures.values():
        members=sorted(s.member_nodes)
        np.testing.assert_array_equal(s.residual_demand,(values+shift)[members,:].sum(axis=0))
        np.testing.assert_array_equal(s.indicative_tokens,manager.tokens[members,:].sum(axis=0))
        assert s.indicative_tokens.dtype==np.int64
  assert sum(row['operation']=='aggregate_groups' and row['backend']=='cuda' for row in coverage)==4


@pytest.mark.parametrize('penalty',[0.,.1,1.+1e-9])
def test_hierarchy_request_preparation_preserves_order_and_roles(penalty):
  from hierarchical_auction.engine import HierarchicalAuctionEngine
  from hierarchical_auction.structure import Structure
  from hierarchical_auction.token_manager import CapacityTokenManager
  from utils.compute_backend import compute_session
  n,nf=6,2;rng=np.random.default_rng(21)
  engine=HierarchicalAuctionEngine(np.ones((n,n))-np.eye(n),nf,np.array([1.,2.]))
  structures={i:Structure(2,i,{i,i+1},set(range(0,n,2))-{i},nf) for i in range(0,n,2)}
  omega=np.zeros((n,nf));omega[::2]=3.
  capacity=np.full((n,nf),4.);prices=rng.random((n,nf))
  engine._aggregate_residual_demand(structures,omega)
  cpu_structures=deepcopy(structures);gpu_structures=deepcopy(structures)
  cpu_manager=CapacityTokenManager(capacity,np.array([1.,2.]));gpu_manager=CapacityTokenManager(capacity,np.array([1.,2.]))
  common=dict(current_y=np.zeros((n,n,nf)),node_prices=prices,latency=rng.random((n,n)),fairness=rng.random((n,nf)),
              omega=omega,eta=.5,epsilon=.01,latency_weight=penalty,fairness_weight=penalty,level=2)
  expected=engine._generate_level_requests(cpu_structures,cpu_structures,token_manager=cpu_manager,**common)
  with compute_session({'compute_device':'cuda'}) as coverage:
    actual=generate_level_requests_cuda(engine,gpu_structures,gpu_structures,token_manager=gpu_manager,**common)
  assert actual==expected
  assert any(row['operation']=='hierarchy_bid_scores' and row['backend']=='cuda' for row in coverage)
  for j in range(n):
    for f in range(nf):assert gpu_manager.pending_requests(j,f)==cpu_manager.pending_requests(j,f)
  for root in structures:np.testing.assert_array_equal(gpu_structures[root].structure_price,cpu_structures[root].structure_price)


def test_custom_hierarchy_scoring_preserves_callback_rng(monkeypatch):
  from hierarchical_auction import engine as module
  from hierarchical_auction.structure import Structure
  from hierarchical_auction.token_manager import CapacityTokenManager
  from utils.compute_backend import compute_session
  engine=module.HierarchicalAuctionEngine(np.ones((3,3))-np.eye(3),1,np.ones(1))
  omega=np.array([[3.],[2.],[0.]])
  def execute(gpu):
    rng=np.random.default_rng(93);seen=[];original=module._GPU_REQUEST_FUNCTIONS[2]
    def scoring(**kw):
      seen.append(tuple(kw.values()));return original(**kw)+rng.random()*1e-9
    monkeypatch.setattr(module,'compute_effective_bid',scoring)
    structures={i:Structure(2,i,{i},set(range(3))-{i},1) for i in range(3)}
    engine._aggregate_residual_demand(structures,omega)
    manager=CapacityTokenManager(np.full((3,1),4.),np.ones(1))
    args=(structures,structures,np.zeros((3,3,1)),np.zeros((3,1)),manager,np.zeros((3,3)),np.zeros((3,1)),omega,.5,.01,0.,0.,2)
    result=generate_level_requests_cuda(engine,*args) if gpu else engine._generate_level_requests(*args)
    return result,seen,rng.bit_generator.state
  expected,seen,state=execute(False)
  with compute_session({'compute_device':'cuda'}) as coverage:actual,other,other_state=execute(True)
  assert actual==expected and seen==other and state==other_state
  assert coverage[-1]['backend']=='cpu' and 'custom' in coverage[-1]['reason']


@pytest.mark.parametrize('detailed',[False,True])
def test_memory_confirmation_batch_order_types_and_freshness(detailed):
  import run_faasmadea as madea
  from utils.compute_backend import compute_session
  from utils import step_time
  rng=np.random.default_rng(92);n,nf=40,3;size=2400
  omega=rng.choice([0.,1e-6-1e-9,1e-6,1e-6+1e-9,2.],(n,nf))
  bids=pd.DataFrame(dict(i=rng.integers(0,n,size),j=rng.integers(0,n,size),f=rng.integers(0,nf,size)),index=pd.Index(rng.integers(0,10,size),name='source'))
  if detailed:bids['d']=1;bids['v']=rng.random(size)
  before=bids.copy(deep=True)
  for shift in (0.,.5):
    expected=pd.concat([madea.confirm_memory_bids(g,(omega+shift)[int(i)],tolerance=1e-6) for i,g in bids.groupby('i')])
    with compute_session({'compute_device':'cuda'}) as coverage,step_time.parallel_phase() as phase:
      actual=madea.confirm_memory_bid_batch(bids,omega+shift,tolerance=1e-6)
      assert phase.shared_device
    pd.testing.assert_frame_equal(actual,expected,check_exact=True)
    assert any(row['operation']=='memory_confirmation' and row['backend']=='cuda' for row in coverage)
  pd.testing.assert_frame_equal(bids,before)


def test_confirmation_custom_callback_preserves_rng():
  import run_faasmadea as madea
  from utils.compute_backend import compute_session
  bids=pd.DataFrame(dict(i=[2,0,2,1],j=[1,1,0,2],f=[0,0,0,0]));omega=np.ones((3,1))
  def execute(batch):
    seen=[];rng=np.random.default_rng(81)
    def callback(group,load,**kwargs):
      seen.append((int(group['i'].iloc[0]),rng.random()))
      return madea.confirm_memory_bids(group,load,**kwargs)
    if batch:result=madea.confirm_memory_bid_batch(bids,omega,confirm=callback)
    else:result=pd.concat([callback(g,omega[int(i)]) for i,g in bids.groupby('i')])
    return result,seen,rng.bit_generator.state
  expected,seen,state=execute(False)
  with compute_session({'compute_device':'cuda'}) as coverage:actual,other,other_state=execute(True)
  pd.testing.assert_frame_equal(actual,expected,check_exact=True)
  assert seen==other and state==other_state
  assert coverage[-1]['backend']=='cpu' and 'callback' in coverage[-1]['reason']


def test_confirmation_coverage_observer_is_outside_clock(monkeypatch):
  from models.numeric_gpu import confirm_memory_rows_cuda
  from utils.compute_backend import compute_session
  from utils import step_time
  ids=np.array([0,1]);functions=np.zeros(2,dtype=int);omega=np.ones((2,1))
  confirm_memory_rows_cuda(ids,functions,omega,1e-6)
  now=[0.];original=np.unique
  def slow_count(*args,**kwargs):
    now[0]+=1.;return original(*args,**kwargs)
  monkeypatch.setattr(np,'unique',slow_count)
  monkeypatch.setattr(step_time,'perf_counter',lambda:now[0])
  clock=step_time.start_step()
  try:
    with compute_session({'compute_device':'cuda'}),step_time.parallel_phase():
      confirm_memory_rows_cuda(ids,functions,omega,1e-6)
  finally:timing=step_time.end_step(clock)
  assert timing['wall_s']==1. and timing['algorithm_s']==0.


@pytest.mark.parametrize('n,nf',[(1,1),(6,3),(30,5),(100,3)])
@pytest.mark.parametrize('tentative',[False,True])
def test_diffusion_vendor_sequence_exact(n,nf,tentative):
  rng=np.random.default_rng(30);data=make_data(n,nf)
  data[None]['memory_requirement']={f+1:float(f+1) for f in range(nf)}
  data[None]['demand']={(j+1,f+1):float([.3,.7,1.][(j+f)%3]) for j in range(n) for f in range(nf)}
  bids=pd.DataFrame(dict(i=rng.integers(0,n,n*nf*4),j=rng.integers(0,n,n*nf*4),f=rng.integers(0,nf,n*nf*4),
                         d=rng.choice([0.,.2,1.,3.],n*nf*4),utility=rng.choice([.5,1.,2.],n*nf*4)))
  capacity=rng.choice([0.,.5,1.,3.],(n,nf));ell=rng.random((n,nf));replicas=rng.integers(0,4,(n,nf)).astype(float)
  rho=rng.integers(0,10,n).astype(float);old=rng.choice([0.,0.,0.,.2,1.],(n,n,nf))
  latency=rng.random((n,n));fairness=rng.random((n,nf));options=dict(latency_weight=.3,fairness_weight=.7)
  before=deepcopy(data),bids.copy(deep=True),[a.copy() for a in (capacity,ell,replicas,rho,old,latency,fairness)]
  for shift in (0.,.25):
    args=bids,capacity,data,ell+shift,replicas,rho,tentative,old,options,latency,fairness
    expected=evaluate_assignments(*args);actual=evaluate_assignments_cuda(*args)
    for a,b in zip(actual[:2],expected[:2]):np.testing.assert_array_equal(a,b)
    assert actual[2]==expected[2]
  assert data==before[0];pd.testing.assert_frame_equal(bids,before[1])
  for a,b in zip((capacity,ell,replicas,rho,old,latency,fairness),before[2]):np.testing.assert_array_equal(a,b)


def test_diffusion_evaluation_dispatch_and_full_clock():
  from utils.compute_backend import compute_session
  from utils import step_time
  n,nf=100,5;data=make_data(n,nf)
  bids=pd.DataFrame([dict(i=i,j=(i+1)%n,f=f,d=2.,utility=2.) for i in range(n) for f in range(nf)])
  args=bids,np.ones((n,nf)),data,np.zeros((n,nf)),np.ones((n,nf)),np.full(n,4.),True
  expected=evaluate_assignments(*args)
  clock=step_time.start_step()
  try:
    with compute_session({'compute_device':'cuda'}) as coverage:actual=evaluate_assignments(*args)
  finally:timing=step_time.end_step(clock)
  for a,b in zip(actual[:2],expected[:2]):np.testing.assert_array_equal(a,b)
  assert actual[2]==expected[2]
  assert any(row['operation']=='evaluate_assignments' and row['backend']=='cuda' for row in coverage)
  assert timing['algorithm_s']>=.9*timing['wall_s']


@pytest.mark.parametrize('n,nf',[(1,1),(6,3),(100,5)])
def test_gcaa_proposals_and_ordered_consensus(n,nf):
  rng=np.random.default_rng(73);size=n*nf*8
  bids=pd.DataFrame(dict(i=rng.integers(0,n,size),j=rng.integers(0,n,size),f=rng.integers(0,nf,size),
                         d=rng.choice([.5,1.,2.],size),utility=rng.choice([-1.,0.,1.,2.],size)),
                    index=pd.Index(rng.permutation(size)+100,name='bid'))
  expected=bids.loc[bids.groupby(['i','f'])['utility'].idxmax()]
  pd.testing.assert_frame_equal(best_agent_bids_cuda(bids),expected,check_exact=True)
  capacity=np.full((n,nf),2.);current=np.zeros((n,n,nf))
  if n>1:current[0,1,0]=1.
  np.testing.assert_array_equal(resolve_gcaa_round_cuda(bids,capacity,current),resolve_gcaa_round(bids,capacity,current))


def test_gcaa_duplicate_index_retains_cpu_semantics():
  from utils.compute_backend import compute_session
  bids=pd.DataFrame(dict(i=[0,1,0],j=[1,2,2],f=[0,0,0],d=[1.,1.,1.],utility=[2.,3.,2.]),index=[7,7,2])
  expected=bids.loc[bids.groupby(['i','f'])['utility'].idxmax()]
  with compute_session({'compute_device':'cuda'}) as coverage:actual=best_agent_bids_cuda(bids)
  pd.testing.assert_frame_equal(actual,expected,check_exact=True)
  assert coverage[-1]['backend']=='cpu'


def test_categorical_confirmation_preserves_pandas_group_order():
  import run_faasmadea as madea
  from utils.compute_backend import compute_session
  bids=pd.DataFrame(dict(i=pd.Categorical(np.repeat(np.arange(30),40),categories=list(range(29,-1,-1)),ordered=True),
                         j=np.zeros(1200,dtype=int),f=np.zeros(1200,dtype=int)))
  omega=np.ones((30,1))
  expected=pd.concat([madea.confirm_memory_bids(group,omega[int(i)]) for i,group in bids.groupby('i',observed=False)])
  with compute_session({'compute_device':'cuda'}) as coverage:actual=madea.confirm_memory_bid_batch(bids,omega)
  pd.testing.assert_frame_equal(actual,expected,check_exact=True)
  assert coverage[-1]['backend']=='cpu'


def test_categorical_gcaa_keeps_first_tie_in_category_order():
  from utils.compute_backend import compute_session
  bids=pd.DataFrame(dict(i=pd.Categorical([0,1,2],categories=[2,0,1],ordered=True),j=[1,2,1],f=[0,0,0],d=[1.,1.,1.],utility=[2.,2.,2.]))
  capacity=np.ones((3,1))
  expected=resolve_gcaa_round(bids,capacity)
  with compute_session({'compute_device':'cuda'}) as coverage:actual=resolve_gcaa_round_cuda(bids,capacity)
  np.testing.assert_array_equal(actual,expected)
  assert coverage[-1]['backend']=='cpu'


@pytest.mark.parametrize('n,nf',[(1,1),(6,3),(100,5)])
def test_dual_memory_request_order_thresholds_and_freshness(n,nf):
  rng=np.random.default_rng(77);data=make_data(n,nf)
  omega=rng.choice([0.,1.,1.+1e-9,2.],(n,nf));placed=rng.choice([0.,1.-1e-12,1.,2.],(n,nf))
  neighbors=rng.integers(0,2,(n,n));rho=rng.choice([0.,1.,2.,3.],n);forbidden=rng.integers(0,2,(n,nf)).astype(bool)
  for shift in (0.,.5):
    rows=dict(i=[],j=[],f=[])
    for i,f in zip(*np.nonzero(omega>0)):
      if placed[i,f]+shift+1e-12<omega[i,f]:
        for j in np.nonzero(neighbors[i,:])[0]:
          if rho[int(j)]>=data[None]['memory_requirement'][f+1] and not forbidden[int(j),f]:
            rows['i'].append(int(i));rows['j'].append(int(j));rows['f'].append(int(f))
    expected=pd.DataFrame(rows)
    actual=dual_memory_requests_cuda(omega,placed+shift,data,neighbors,rho,forbidden)
    pd.testing.assert_frame_equal(actual,expected,check_exact=True)


@pytest.mark.parametrize('kind',['madea','gcaa'])
def test_shared_seller_roles_keep_original_commit_order(kind):
  from run_faasmadea import evaluate_bids
  from utils.compute_backend import compute_session
  bids=pd.DataFrame(dict(i=[0,1],j=[1,2],f=[0,0],d=[1.,1.],b=[2.,2.],utility=[2.,2.]))
  capacity=np.array([[0.],[1.],[1.]])
  call=(lambda:evaluate_bids(bids,capacity,make_data(3,1),may_replace_existing_assignments=False)) if kind=='madea' else (lambda:(resolve_gcaa_round(bids,capacity),))
  expected=call()
  with compute_session({'compute_device':'cuda'}) as coverage:actual=call()
  for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
  assert actual[0][0,1,0]==1. and actual[0][1,2,0]==0.
  assert any(row['backend']=='cpu' and 'shared' in row['reason'] for row in coverage)


def test_incumbent_replacement_releases_role_before_later_vendor():
  from run_faasmadea import evaluate_bids
  from utils.compute_backend import compute_session
  bids=pd.DataFrame(dict(i=[0,2],j=[1,3],f=[0,0],d=[1.,1.],b=[3.,2.],utility=[3.,2.]))
  old=np.zeros((4,4,1));old[3,1,0]=1.
  blackboard=np.array([[0.],[1.],[0.],[1.]])
  residual=np.array([[0.],[0.],[0.],[1.]])
  def call():
    return evaluate_bids(bids,blackboard,make_data(4,1),previous_y=old,p=np.ones((4,1)),
                         residual_capacity=residual,may_replace_existing_assignments=True)
  expected=call()
  with compute_session({'compute_device':'cuda'}):actual=call()
  for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
  assert actual[0][3,1,0]==-1. and actual[0][0,1,0]==1. and actual[0][2,3,0]==1.
  combined=old+actual[0]
  assert not ((combined.sum(axis=0)>1e-10)&(combined.sum(axis=1)>1e-10)).any()


def test_dual_memory_request_production_dispatch():
  from decentralized_dual import dual_coordination_round
  from utils.compute_backend import compute_session
  n,nf=100,5;data=make_data(n,nf);omega=np.zeros((n,nf));omega[:70]=3.
  options=dict(max_inner_iterations=3,alpha0=.1,theta=.5,gap_tolerance=0.,step_rule='sqrt',latency_weight=0.,fairness_weight=0.)
  args=omega,np.zeros((n,nf)),data,np.ones((n,n))-np.eye(n),np.full(n,4.),options,np.zeros((n,n)),np.zeros((n,nf))
  expected=dual_coordination_round(*args)
  with compute_session({'compute_device':'cuda'}) as coverage:actual=dual_coordination_round(*args)
  np.testing.assert_array_equal(actual[0],expected[0]);pd.testing.assert_frame_equal(actual[1],expected[1],check_exact=True)
  for key in actual[2]:np.testing.assert_array_equal(actual[2][key],expected[2][key])
  assert any(row['operation']=='dual_memory_requests' and row['backend']=='cuda' for row in coverage)


def test_diffusion_unused_replica_placeholders_are_expected_cpu_case():
  from utils.compute_backend import compute_session
  n,nf=100,5
  bids=pd.DataFrame(dict(i=np.arange(n),j=(np.arange(n)+1)%n,f=np.zeros(n,dtype=int),d=np.ones(n),utility=np.ones(n)))
  args=bids,np.ones((n,nf)),make_data(n,nf),np.zeros(n),np.zeros(n),np.zeros((n,nf)),False
  expected=evaluate_assignments(*args)
  with compute_session({'compute_device':'cuda'}) as coverage:actual=evaluate_assignments(*args)
  for a,b in zip(actual[:2],expected[:2]):np.testing.assert_array_equal(a,b)
  assert any(row['backend']=='cpu' and 'shape' in row['reason'] for row in coverage)


def test_diffusion_cuda_failure_is_not_hidden(monkeypatch):
  import models.bids_gpu as module
  from utils.compute_backend import compute_session
  n,nf=100,5
  bids=pd.DataFrame(dict(i=np.arange(n),j=(np.arange(n)+1)%n,f=np.zeros(n,dtype=int),d=np.ones(n),utility=np.ones(n)))
  def broken():raise RuntimeError('CUDA kernel failure')
  monkeypatch.setattr(module,'_diffusion_kernel',broken)
  with compute_session({'compute_device':'cuda'}),pytest.raises(RuntimeError,match='CUDA kernel failure'):
    evaluate_assignments(bids,np.ones((n,nf)),make_data(n,nf),np.zeros((n,nf)),np.zeros((n,nf)),np.zeros(n),False)
