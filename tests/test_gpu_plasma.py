import os
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest
from plasma.core.node import NodeParams,PlasmaNode
from plasma.core.sbm import hamiltonian
from plasma.core.types import PlasmaOptions
from plasma.engine import PlasmaEngine
from utils.compute_backend import compute_session
from utils import step_time

pytestmark=pytest.mark.skipif(os.environ.get('DFAAS_TEST_CUDA')!='1',reason='real CUDA opt-in required')
GPU=dict(compute_device='cuda',dp_device='cuda',use_dp=True)


def nodes_fixture(n=6,opts=None):
  opts=opts or PlasmaOptions(k_sb=1,sbm_method='exact',n_hyst=1,p_commit=1.)
  nodes=[]
  for i in range(n):
    nbrs=tuple(j for j in range(n) if j!=i)
    params=NodeParams(i,nbrs,np.array([2.,3.,4.]),np.ones(3),np.ones((len(nbrs),3)),np.array([1.,2.,3.]),float(8+4*(i%2)),np.array([1.,2.,4.]))
    node=PlasmaNode(params,opts,np.random.default_rng(i));node.lam_hat=np.array([6.,5.,4.])+i%3
    nodes.append(node)
  return nodes


def assert_state(a,b):
  for x,y in zip(a,b):
    for name in ('r','lam_hat','D','_spare_last','_pull_last','_recv_from_last'):
      np.testing.assert_array_equal(getattr(x,name),getattr(y,name),err_msg=name)
    assert x._streak==y._streak
    if x._pending is None:assert y._pending is None
    else:np.testing.assert_array_equal(x._pending,y._pending)
    assert x.rng.bit_generator.state==y.rng.bit_generator.state


@pytest.mark.parametrize('p_commit',[0.,.35,1.])
@pytest.mark.parametrize('n_hyst',[1,3])
def test_exact_sb_proposals_hysteresis_rng(p_commit,n_hyst):
  from plasma.gpu import exact_sb_batch
  opts=PlasmaOptions(k_sb=1,sbm_method='exact',p_commit=p_commit,n_hyst=n_hyst)
  cpu=nodes_fixture(opts=opts);cpu[2].alive=False;gpu=deepcopy(cpu)
  for round_ in range(7):
    expected=[node.sb_pass(round_) for node in cpu]
    with compute_session(GPU) as coverage:actual=exact_sb_batch(gpu,round_)
    assert actual==expected;assert_state(cpu,gpu)
    assert any(r['operation']=='plasma_exact_sb' and r['backend']=='cuda' for r in coverage)
    for a,b in zip(cpu,gpu):assert hamiltonian(a.r,a._hamiltonian_ctx(round_))==hamiltonian(b.r,b._hamiltonian_ctx(round_))


@pytest.mark.parametrize('options',[{},dict(dp_device='cuda',use_dp=True),dict(compute_device='cuda'),GPU])
def test_engine_gate_and_pre_sb_snapshot(options):
  opts=PlasmaOptions(k_sb=1,sbm_method='exact',p_commit=1.,n_hyst=1)
  nodes=nodes_fixture(opts=opts)
  for node in nodes:node.params=replace(node.params,ram_cap=128.)
  cpu=PlasmaEngine(nodes,opts,np.random.default_rng(3));gpu=deepcopy(cpu)
  before=np.array([node.r.copy() for node in gpu.nodes]);arrivals=np.full((6,3),10)
  expected=cpu.run_rounds(1,arrivals)
  with compute_session(options) as coverage:actual=gpu.run_rounds(1,arrivals)
  for k in ('x','y','z','r','xi'):np.testing.assert_array_equal(getattr(actual,k),getattr(expected,k))
  np.testing.assert_array_equal(actual.r,before);assert_state(cpu.nodes,gpu.nodes)
  assert cpu.rng.bit_generator.state==gpu.rng.bit_generator.state
  assert (cpu.clock.round,cpu.msg_count,cpu.hb_count)==(gpu.clock.round,gpu.msg_count,gpu.hb_count)
  assert any(r['backend']=='cuda' for r in coverage) is (options==GPU)


def test_expected_cpu_domains_and_mixed_batch(monkeypatch):
  from plasma.gpu import exact_sb_batch
  from models import dp_gpu
  cpu=nodes_fixture();cpu[0].params=replace(cpu[0].params,ram_cap=8.5)
  cpu[1].opts=replace(cpu[1].opts,sbm_method='dsb',n_sb_steps=1,sb_restarts=1)
  gpu=deepcopy(cpu);expected=[n.sb_pass(0) for n in cpu]
  with compute_session(GPU):actual=exact_sb_batch(gpu,0)
  assert actual==expected;assert_state(cpu,gpu)
  def forbidden(*a,**kw):raise AssertionError('unexpected CUDA')
  monkeypatch.setattr(dp_gpu,'minimize_replica_costs_batch',forbidden)
  with compute_session(GPU) as coverage:assert exact_sb_batch(gpu[:1],1)==[cpu[0].sb_pass(1)]
  assert all(r['backend']=='cpu' for r in coverage)


def test_sb_cuda_failure_is_visible(monkeypatch):
  from plasma.gpu import exact_sb_batch
  from models import dp_gpu
  nodes=nodes_fixture();before=deepcopy(nodes)
  def broken(*a,**kw):raise RuntimeError('injected CUDA failure')
  monkeypatch.setattr(dp_gpu,'minimize_replica_costs_batch',broken)
  with compute_session(GPU),pytest.raises(RuntimeError,match='injected CUDA failure'):exact_sb_batch(nodes,0)
  assert_state(nodes,before)


def test_sb_per_problem_infeasibility_keeps_other_proposals(monkeypatch):
  from plasma.gpu import exact_sb_batch
  from plasma.core import sbm
  nodes=nodes_fixture();before=deepcopy(nodes);expected=[node.sb_pass(0) for node in deepcopy(nodes)]
  prepare=sbm.prepare_exact_problem
  def one_infeasible(ctx,r_max):
    costs,memory,cap=prepare(ctx,r_max)
    if cap==8.:costs=[np.full_like(c,np.inf) for c in costs]
    return costs,memory,cap
  monkeypatch.setattr(sbm,'prepare_exact_problem',one_infeasible)
  with compute_session(GPU):actual=exact_sb_batch(nodes,0)
  for i,node in enumerate(nodes):
    if node.params.ram_cap==8.:
      assert not actual[i];assert_state([node],[before[i]])
    else:assert actual[i]==expected[i]


def assert_plan(a,b):
  np.testing.assert_array_equal(a.r,b.r);np.testing.assert_array_equal(a.x,b.x)
  assert a.accepted==b.accepted and a.gain==b.gain


def test_welfare_batch_plans_mixed_domains_and_no_cuda_singleton(monkeypatch):
  from plasma.gpu import welfare_plan_batch
  from test_plasma_welfare import _node
  from plasma.welfare import Reservation,Offer
  from models import dp_gpu
  nodes=[_node(i,(9,),ram=1.5 if i==0 else 3.) for i in range(4)]
  jobs=[]
  for i,node in enumerate(nodes):
    node.start_window(np.array([0,10+i]),0,initialize=False)
    jobs.append((node,[Reservation(9,i,0,1,(Offer(0,False,5,2.),))]))
  expected=[node.plan(r) for node,r in jobs]
  with compute_session(GPU) as coverage:actual=welfare_plan_batch(jobs)
  for a,b in zip(actual,expected):assert_plan(a,b)
  assert any(r['operation']=='plasma_welfare_plan' and r['backend']=='cuda' for r in coverage)
  def forbidden(*a,**kw):raise AssertionError('unexpected CUDA')
  monkeypatch.setattr(dp_gpu,'minimize_replica_costs_batch',forbidden)
  with compute_session(GPU):assert_plan(welfare_plan_batch(jobs[:1])[0],expected[0])


@pytest.mark.parametrize('parallelism',[0,2])
@pytest.mark.parametrize('loss',[0.,.5])
def test_welfare_engine_matches_cpu_pool_no_gpu_workers(parallelism,loss,monkeypatch):
  import networkx as nx
  import plasma.welfare as welfare
  from run_faasmacro import parallel_solver_session
  graph=nx.circular_ladder_graph(6);opts=welfare.WelfareOptions(hb_loss=loss,rounds_per_step=4)
  rng=np.random.default_rng(31);nodes=[]
  for i in range(12):
    nbrs=tuple(sorted(graph[i]));params=NodeParams(i,nbrs,rng.uniform(0,5,3),rng.uniform(0,2,3),rng.uniform(0,8,(len(nbrs),3)),rng.integers(2,6,3),64.,np.ones(3))
    nodes.append(welfare.WelfareNode(params,opts,np.random.default_rng(i)))
  cpu=welfare.WelfareEngine(deepcopy(nodes),opts,np.random.default_rng(31),parallelism=2)
  gpu=welfare.WelfareEngine(deepcopy(nodes),opts,np.random.default_rng(31),parallelism=parallelism)
  cpu_messages=[];gpu_messages=[]
  for engine,messages in ((cpu,cpu_messages),(gpu,gpu_messages)):
    send=engine._send
    def traced(source,target,operation,*payload,_send=send,_messages=messages):
      _messages.append((source,target,operation,payload));return _send(source,target,operation,*payload)
    engine._send=traced
  for epoch in range(3):
    arrivals=rng.integers(200,500,(12,3));cpu.set_alive(0,epoch!=1);gpu.set_alive(0,epoch!=1)
    with parallel_solver_session():expected=cpu.run_rounds(4,arrivals)
    def forbidden(*a,**kw):raise AssertionError('GPU plans must not create CPU workers')
    with monkeypatch.context() as patch,compute_session(GPU) as coverage:
      patch.setattr(welfare.mp,'Pool',forbidden)
      actual=gpu.run_rounds(4,arrivals)
    for key in ('x','y','z','r','xi'):np.testing.assert_array_equal(getattr(actual,key),getattr(expected,key))
    assert gpu.workers==0;assert gpu_messages==cpu_messages
    assert (gpu.msg_count,gpu.hb_count,gpu.accepted_trades)==(cpu.msg_count,cpu.hb_count,cpu.accepted_trades)
    assert gpu.rng.bit_generator.state==cpu.rng.bit_generator.state
    for a,b in zip(gpu.nodes,cpu.nodes):
      for key in ('x','y','z','r','incoming','load'):np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
      assert a.reservation is b.reservation is None
      assert a.rng.bit_generator.state==b.rng.bit_generator.state
    assert any(r['operation']=='plasma_welfare_plan' and r['backend']=='cuda' for r in coverage)


def test_welfare_cuda_failure_releases_every_offer(monkeypatch):
  from test_plasma_welfare import _node
  from plasma.welfare import WelfareEngine,WelfareOptions
  from plasma.core.types import Heartbeat
  from models import dp_gpu
  nodes=[]
  for source,receiver in ((0,1),(2,3)):
    nodes.extend([_node(source,(receiver,),alpha=(0.,),gamma=(1.,),beta=[[2.]],ram=0.),
                  _node(receiver,(source,),alpha=(0.,),gamma=(0.,),beta=[[0.]],ram=128.)])
  for i,node in enumerate(nodes):node.start_window(np.array([10 if i%2==0 else 0]),0)
  for receiver,source in ((1,0),(3,2)):nodes[receiver].on_heartbeat(Heartbeat(source,1,(0.,),(0.,),(10.,)),0)
  engine=WelfareEngine(nodes,WelfareOptions(),np.random.default_rng(0));engine.accepted_trades=0
  def broken(*a,**kw):
    assert nodes[0].reservation is not None and nodes[2].reservation is not None
    raise RuntimeError('injected CUDA reservation failure')
  monkeypatch.setattr(dp_gpu,'minimize_replica_costs_batch',broken)
  with compute_session(GPU),pytest.raises(RuntimeError,match='injected CUDA reservation failure'):
    engine._negotiate_batch(0,[1,3],None)
  assert all(node.reservation is None for node in nodes)
  for i in (0,2):np.testing.assert_array_equal(nodes[i].z,[10])
  for i in (1,3):np.testing.assert_array_equal(nodes[i].incoming,[0])


def test_welfare_shared_participants_never_widen_batches(monkeypatch):
  import plasma.gpu as adapter
  from test_plasma_welfare import _node
  from plasma.welfare import WelfareEngine,WelfareOptions
  n=6;opts=WelfareOptions(rounds_per_step=3)
  nodes=[_node(i,tuple(j for j in range(n) if j!=i),capacity=1.,ram=64.) for i in range(n)]
  cpu=WelfareEngine(deepcopy(nodes),opts,np.random.default_rng(5),parallelism=2)
  gpu=WelfareEngine(deepcopy(nodes),opts,np.random.default_rng(5),parallelism=0)
  assert all(len(batch)==1 for batch in gpu._negotiation_batches(list(range(n))))
  arrivals=np.full((n,2),2000);expected=cpu.run_rounds(3,arrivals)
  sizes=[];original=adapter.welfare_plan_batch
  def counted(jobs):sizes.append(len(jobs));return original(jobs)
  monkeypatch.setattr(adapter,'welfare_plan_batch',counted)
  with compute_session(GPU):actual=gpu.run_rounds(3,arrivals)
  assert sizes==[n]
  for key in ('x','y','z','r','xi'):np.testing.assert_array_equal(getattr(actual,key),getattr(expected,key))
  assert (gpu.msg_count,gpu.hb_count,gpu.accepted_trades)==(cpu.msg_count,cpu.hb_count,cpu.accepted_trades)


def test_gpu_mixed_phase_counts_cpu_prep_jit_and_device_once(monkeypatch):
  from plasma.gpu import exact_sb_batch
  from models import dp_gpu
  tick=[0.];monkeypatch.setattr(step_time,'perf_counter',lambda:tick[0])
  kernel=dp_gpu._get_kernel;batch=dp_gpu.minimize_replica_costs_batch
  def cold():tick[0]+=1.;return kernel()
  def device(*a,**kw):tick[0]+=.4;return batch(*a,**kw)
  monkeypatch.setattr(dp_gpu,'_get_kernel',cold);monkeypatch.setattr(dp_gpu,'minimize_replica_costs_batch',device)
  with compute_session(GPU):
    clock=step_time.start_step()
    with step_time.parallel_phase():
      for i in step_time.timed(range(3)):tick[0]+=.1
      exact_sb_batch(nodes_fixture(),0)
    timing=step_time.end_step(clock)
  assert timing['algorithm_s']==pytest.approx(1.7)
  assert timing['wall_s']==pytest.approx(1.7)


@pytest.mark.parametrize('welfare',[False,True])
def test_gpu_budget_includes_initial_plans_and_resumes_events(welfare,monkeypatch):
  from models import dp_gpu
  from test_plasma_welfare import _node
  from plasma.welfare import WelfareEngine,WelfareOptions
  tick=[0.];monkeypatch.setattr(step_time,'perf_counter',lambda:tick[0])
  batch=dp_gpu.minimize_replica_costs_batch
  def costly(*a,**kw):tick[0]+=.4;return batch(*a,**kw)
  monkeypatch.setattr(dp_gpu,'minimize_replica_costs_batch',costly)
  n=6
  if welfare:
    opts=WelfareOptions(W=.5,hb_latency_rounds=2,staleness_rounds=3)
    nodes=[_node(i,tuple(j for j in range(n) if j!=i),capacity=2.,ram=64.) for i in range(n)]
    engine=WelfareEngine(nodes,opts,np.random.default_rng(7))
    for node in nodes:node.opts=opts
  else:
    opts=PlasmaOptions(W=.5,k_sb=1,hb_latency_rounds=2,staleness_rounds=3)
    nodes=nodes_fixture(opts=opts)
    for node in nodes:node.params=replace(node.params,ram_cap=128.)
    engine=PlasmaEngine(nodes,opts,np.random.default_rng(7))
  arrivals=np.full((n,engine.nodes[0].Nf),8);delivered=[]
  engine.clock.schedule(2,lambda:delivered.append(2))
  with compute_session(GPU):
    clock=step_time.start_step();result=engine.run_rounds(5,arrivals,time_limit=.25);timing=step_time.end_step(clock)
    assert result.completed_rounds==engine.clock.round==1 and result.time_limit_reached
    assert timing['algorithm_s']==pytest.approx(.4)
    assert delivered==[] and any(round_==2 for round_,_,_ in engine.clock._queue)
    for _ in range(2):
      clock=step_time.start_step();next_result=engine.run_rounds(1,arrivals,time_limit=float('inf'));step_time.end_step(clock)
      assert next_result.completed_rounds==1
  assert delivered==[2] and engine.clock.round==3
  np.testing.assert_array_equal(result.x+result.z+result.y.sum(axis=1),arrivals)


@pytest.mark.parametrize('kappa',[1.,2.])
def test_window_batch_numeric_state_exact_and_fresh(kappa):
  from plasma.gpu import end_window_batch_cuda
  nodes=nodes_fixture(8,PlasmaOptions(kappa=kappa));rng=np.random.default_rng(32)
  for node in nodes:
    for key in ('D','_phi','_arrivals','_admitted','lam_hat'):
      arr=getattr(node,key);arr[:]=rng.random(arr.shape)*7
    node.r[:]=rng.integers(0,8,node.Nf)
  for shift in (0.,.3):
    for node in nodes:node._phi+=shift;node._arrivals+=shift
    cpu=deepcopy(nodes);gpu=deepcopy(nodes)
    expected=[node.end_window() for node in cpu]
    actual=end_window_batch_cuda(gpu)
    for a,b in zip(actual,expected):
      for key in ('x','z','y','xi'):np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
    assert_state(cpu,gpu)


@pytest.mark.parametrize('n,degree',[(1,0),(4,3),(16,6),(32,6)])
def test_routing_weights_prototype_preserves_gates(n,degree):
  from plasma.gpu import target_weights_batch_cuda
  from plasma.core.routing import target_weights
  rng=np.random.default_rng(21);D=rng.random((n,5,degree+2));local=rng.integers(0,2,(n,5)).astype(bool)
  spare=rng.choice([0.,.1,2.],(n,5,degree));eps=.01
  expected=np.array([[target_weights(D[i,f],local[i,f],spare[i,f],eps) for f in range(5)] for i in range(n)])
  actual=target_weights_batch_cuda(D,local,spare,eps)
  np.testing.assert_array_equal(actual,expected)


def routing_fixture(n=12,nf=5,degree=2):
  from plasma.core.types import Heartbeat
  nodes=[];opts=PlasmaOptions(k_sb=0)
  for i in range(n):
    nbrs=tuple(sorted({(i+d)%n for d in range(1,degree//2+1)}|{(i-d)%n for d in range(1,degree//2+1)}))
    node=PlasmaNode(NodeParams(i,nbrs,np.ones(nf),np.ones(nf),np.full((len(nbrs),nf),3.),np.ones(nf),16.,np.ones(nf)),opts,np.random.default_rng(i))
    node.r[:]=i%3;node.lam_hat[:]=8.
    for j in nbrs:node.cache.store(Heartbeat(j,1,tuple(np.full(nf,3.)),tuple(np.ones(nf)),tuple(np.full(nf,2.))),0)
    nodes.append(node)
  return PlasmaEngine(nodes,opts,np.random.default_rng(7))


def test_route_all_cuda_preparation_keeps_admission_rng_order():
  from plasma.gpu import route_all_cuda
  cpu=routing_fixture();gpu=deepcopy(cpu)
  for round_,load in ((0,8),(1,13)):
    arrivals=np.full((12,5),load)
    cpu._route_all(round_,arrivals)
    with compute_session({'compute_device':'cuda'}):route_all_cuda(gpu,round_,arrivals)
    assert cpu.msg_count==gpu.msg_count and cpu.rng.bit_generator.state==gpu.rng.bit_generator.state
    for a,b in zip(cpu.nodes,gpu.nodes):
      for key in ('_x','_z','_y','_xi','_phi','_pull','_admitted','_arrivals','_recv_from'):
        np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
      assert a.rng.bit_generator.state==b.rng.bit_generator.state
