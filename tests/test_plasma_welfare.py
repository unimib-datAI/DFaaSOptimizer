"""Welfare decisions must use local state and explicit neighbor transactions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plasma.core.node import NodeParams


def _node(i=0, neighbors=(), alpha=(10., 2.), gamma=(1., 1.), beta=None, ram=1., capacity=10.):
  from plasma.welfare import WelfareNode, WelfareOptions
  nf = len(alpha)
  params = NodeParams(i, neighbors, np.array(alpha), np.array(gamma),
                      np.zeros((len(neighbors), nf)) if beta is None else np.array(beta),
                      np.full(nf, capacity), ram, np.ones(nf))
  return WelfareNode(params, WelfareOptions(), np.random.default_rng(i))


@pytest.mark.parametrize('load,alpha,gamma,expected', [
  ([100, 10], (10., 2.), (1., 1.), [0, 10]),
  ([10, 10], (4., 3.), (0., 4.), [0, 10]),
  ([0, 0], (4., 3.), (0., 4.), [0, 0]),
])
def test_local_placement_maximizes_normalized_welfare(load, alpha, gamma, expected):
  node = _node(alpha=alpha, gamma=gamma)
  node.start_window(np.array(load), epoch=0)
  np.testing.assert_array_equal(node.x, expected)
  np.testing.assert_array_equal(node.x + node.z, load)


def test_reservation_prevents_double_sale_and_rejects_stale_confirmation():
  node = _node(neighbors=(1, 2), alpha=(1.,), gamma=(1.,), beta=[[2.], [3.]], ram=0.)
  node.start_window(np.array([10]), epoch=0)
  offer = node.reserve(1, 0)
  assert offer is not None
  assert node.reserve(2, 0) is None
  assert not node.prepare(2, offer, (10,))
  assert not node.prepare(1, offer, (11,))
  assert node.prepare(1, offer, (4,))
  assert node.commit(1, offer, (4,))
  assert not node.commit(1, offer, (4,))
  assert node.z.tolist() == [6]
  assert node.y.sum() == 4
  other = node.reserve(2, 0)
  assert other.items[0].quantity == 6
  node.start_window(np.array([2]), epoch=1)
  assert not node.commit(2, other, (1,))
  assert node.z.tolist() == [2]


def test_receiver_values_sender_gains_and_preserves_accepted_traffic():
  from plasma.welfare import WelfareEngine, WelfareOptions
  opts = WelfareOptions(rounds_per_step=4)
  nodes = [
    _node(0, (1,), alpha=(0., 0.), beta=[[1., 0.]], ram=0.),
    _node(1, (0, 2), alpha=(4., 1.), beta=[[0., 0.], [0., 0.]]),
    _node(2, (1,), alpha=(0., 0.), beta=[[0., 5.]], ram=0.),
  ]
  engine = WelfareEngine(nodes, opts, np.random.default_rng(0))
  result = engine.run_rounds(4, np.array([[10, 0], [0, 0], [0, 10]]))
  assert result.y[2, 1, 1] == 10
  assert result.y[0, 1, 0] == 0
  assert result.r[1].tolist() == [0, 1]
  # An established receiver cannot offer the same function onward.
  assert nodes[1].reserve(0, nodes[1].epoch) is None
  assert all(node.reservation is None for node in nodes)
  assert engine.msg_count > engine.hb_count > 0


def test_new_method_is_registered():
  import run as batch
  from remote_experiments.jobs import SCRIPT_BY_ALGORITHM
  assert batch.METHOD_RESULT_MODELS['plasma-welfare'] == ('LSPc', 'Plasma-Welfare')
  assert 'plasma-welfare' in SCRIPT_BY_ALGORITHM


def test_failed_prepare_releases_all_offers_without_moving_traffic():
  from plasma.core.types import Heartbeat
  receiver = _node(1, (0,), alpha=(0.,), gamma=(0.,), beta=[[0.]])
  source = _node(0, (1,), alpha=(0.,), gamma=(1.,), beta=[[2.]], ram=0.)
  receiver.start_window(np.array([0]), 0)
  source.start_window(np.array([10]), 0)
  receiver.on_heartbeat(Heartbeat(0, 1, (0.,), (0.,), (10.,)), 0)
  def send(target, operation, *payload):
    assert target == 0
    if operation == 'prepare':
      return False
    return getattr(source, operation)(1, *payload)
  assert not receiver.negotiate(0, send)
  assert source.reservation is None
  assert source.z.tolist() == [10]
  assert receiver.incoming.tolist() == [0]


@pytest.mark.parametrize('seed,W,loss', [(7, 1., 0.), (42, 2., .5), (4850, 1., 1.)])
def test_each_transaction_preserves_feasibility_and_increases_actual_welfare(tmp_path, seed, W, loss):
  from test_plasma_e2e import _config
  from plasma.runner import build_nodes
  from plasma.welfare import WelfareEngine, WelfareNode, WelfareOptions
  from run_centralized_model import init_problem, update_data, get_current_load
  from utils.centralized import validate_centralized_solution
  from utils.faasmacro import compute_centralized_objective
  config = _config(tmp_path, Nn=5)
  data, traces, agents, _ = init_problem(config['limits'], 'sinusoidal', 3, seed, str(tmp_path))
  loads = get_current_load(traces, agents, 0)
  arrivals = np.array([[round(loads[i+1,f+1]*W) for f in range(2)] for i in range(5)])
  data = update_data(data, {'incoming_load': {(i+1,f+1):arrivals[i,f] for i in range(5) for f in range(2)}})
  # The capacity validator uses per-second units; adapt demand for a W-second window.
  data[None]['demand'] = {k:v/W for k,v in data[None]['demand'].items()}
  opts = WelfareOptions(W=1., hb_loss=loss, rounds_per_step=6)
  nodes = build_nodes(data, opts, seed, node_class=WelfareNode)
  engine = WelfareEngine(nodes, opts, np.random.default_rng(seed))
  accepted = []
  def snapshot():
    x = np.array([n.x for n in nodes]); z = np.array([n.z for n in nodes])
    r = np.array([n.r for n in nodes]); y = np.zeros((5,5,2))
    for i,n in enumerate(nodes):
      for k,j in enumerate(n.params.nbrs):
        y[i,j] = n.y[k]
    validate_centralized_solution(x,y,z,r,data)
    assert all(np.array_equal(v, np.rint(v)) for v in (x,y,z,r))
    return compute_centralized_objective(data,x,y,z)
  for node in nodes:
    original = node.negotiate
    def checked(round_, send, original=original):
      before = snapshot()
      changed = original(round_, send)
      after = snapshot()
      assert after >= before-1e-9
      if changed:
        assert after > before + opts.epsilon-1e-9
        accepted.append(after-before)
      return changed
    node.negotiate = checked
  result = engine.run_rounds(6, arrivals)
  if loss == 1.:
    assert not result.y.any()
  else:
    assert accepted
  assert all(n.reservation is None for n in nodes)


@pytest.mark.parametrize('W', [1., 2.])
def test_runner_exports_new_method_and_real_termination(tmp_path, W):
  from test_review_distributed_regressions import _two_node_data, _materialized_config
  from plasma.welfare import run
  config = _materialized_config(tmp_path, _two_node_data(), steps=2)
  config['solver_options']['plasma_welfare'] = {'rounds_per_step': 4, 'W': W}
  folder = Path(run(config))
  obj = pd.read_csv(folder/'obj.csv')
  assert list(obj.columns) == ['Plasma-Welfare']
  assert np.isfinite(obj.to_numpy()).all()
  assert (obj.iloc[:,0] > 2.).all()  # local-only welfare is exactly 2
  assert pd.read_csv(folder/'plasma_welfare.csv').accepted_trades.gt(0).all()
  assert 'round budget exhausted' in (folder/'termination_condition.csv').read_text()
  assert (folder/'LSPc_solution.csv').exists()


def test_batch_resume_keeps_plasma_variants_separate(tmp_path, monkeypatch):
  import run as batch
  (tmp_path/'experiments.json').write_text(json.dumps({
    'experiments_list': [[2, 123]], 'plasma': ['existing'], 'plasma-welfare': [None],
  }))
  monkeypatch.setattr('sys.argv', ['run.py', '--methods', 'plasma-welfare'])
  assert batch.parse_arguments().methods == ['plasma-welfare']
  monkeypatch.setattr(batch, 'run_plasma_welfare', lambda *a, **k: 'new-result')
  monkeypatch.setattr(batch, 'run_plasma', lambda *a, **k: pytest.fail('Baseline rerun'))
  monkeypatch.setattr(batch, 'results_postprocessing', lambda *a, **k: None)
  batch.run({'seed':123,'verbose':0,'limits':{'Nn':{'values':[2]}}},str(tmp_path),
            n_experiments=1,methods=['plasma','plasma-welfare'],reference_method='plasma',
            fix_r=False,sp_parallelism=0,enable_plotting=False,loop_over='Nn')
  result = json.loads((tmp_path/'experiments.json').read_text())
  assert result['plasma'] == ['existing']
  assert result['plasma-welfare'] == ['new-result']


def test_source_with_zero_processing_capacity_can_still_forward():
  source = _node(0, (1,), alpha=(1.,), gamma=(1.,), beta=[[2.]], capacity=0.)
  source.start_window(np.array([3]), 0)
  offer = source.reserve(1, 0)
  assert source.commit(1, offer, (3,))
  assert source.r.tolist() == [0]
  assert source.z.tolist() == [0]
  assert source.y.tolist() == [[3]]
