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
  cpu=PlasmaEngine(nodes_fixture(opts=opts),opts,np.random.default_rng(3));gpu=deepcopy(cpu)
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
