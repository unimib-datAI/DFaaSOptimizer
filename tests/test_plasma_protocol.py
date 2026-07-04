import dataclasses

import numpy as np
import pytest

from plasma.core.protocol import HeartbeatCache, decode_heartbeat, encode_heartbeat
from plasma.core.types import LOCAL, REJ, Heartbeat, PlasmaOptions


def test_target_columns():
  assert LOCAL == 0
  assert REJ == 1


def test_options_defaults():
  opts = PlasmaOptions()
  assert opts.W == 1.0
  assert opts.k_sb == 10
  assert opts.mu == 0.1
  assert opts.p_commit == 0.5
  assert opts.n_hyst == 2
  assert opts.staleness_rounds == 3
  assert opts.rare_function_mode == "sampled"


def test_options_from_config_overrides():
  config = {"solver_options": {"plasma": {"mu": 0.2, "k_sb": 5}}}
  opts = PlasmaOptions.from_config(config)
  assert opts.mu == 0.2
  assert opts.k_sb == 5
  assert opts.W == 1.0


def test_options_frozen():
  with pytest.raises(dataclasses.FrozenInstanceError):
    PlasmaOptions().mu = 0.5


def test_options_rejects_execution_mode():
  config = {"solver_options": {"plasma": {"execution_mode": "async"}}}
  with pytest.raises(TypeError):
    PlasmaOptions.from_config(config)


def _hb(node=3, seq=7):
  return Heartbeat(node=node, seq=seq, spare=(1.0, 0.0), alpha=(2.0, 2.5),
                   pull=(0.0, 4.0))


def test_heartbeat_roundtrip():
  hb = _hb()
  assert decode_heartbeat(encode_heartbeat(hb)) == hb


def test_heartbeat_field_whitelist():
  msg = encode_heartbeat(_hb())
  assert set(msg) == {"node", "seq", "spare", "alpha", "pull"}
  msg["ram_capacity"] = 64  # privacy violation: must be rejected
  with pytest.raises(ValueError):
    decode_heartbeat(msg)


def test_stale_neighbor_treated_as_zero_spare():
  cache = HeartbeatCache()
  cache.store(_hb(node=3), round_=10)
  fresh = cache.spare(3, now_round=12, staleness_rounds=3, Nf=2)
  stale = cache.spare(3, now_round=14, staleness_rounds=3, Nf=2)
  assert fresh.tolist() == [1.0, 0.0]
  assert stale.tolist() == [0.0, 0.0]


def test_unknown_neighbor_is_zero_spare():
  cache = HeartbeatCache()
  assert cache.spare(9, now_round=0, staleness_rounds=3, Nf=2).tolist() == [0.0, 0.0]


def test_pull_in_sums_fresh_neighbors_only():
  cache = HeartbeatCache()
  cache.store(_hb(node=1), round_=10)   # pull (0, 4)
  cache.store(_hb(node=2), round_=1)    # stale at now=12
  pull = cache.pull_in(now_round=12, staleness_rounds=3, Nf=2)
  assert pull.tolist() == [0.0, 4.0]


from plasma.engine import PlasmaEngine  # noqa: F401 (import checks packaging)


def test_message_budget_heartbeats_bounded_by_degree():
  from plasma.core.node import NodeParams, PlasmaNode
  from plasma.core.types import PlasmaOptions
  opts = PlasmaOptions(k_sb=0, hb_loss=0.0)
  rng = np.random.default_rng(0)
  params0 = NodeParams(node_id=0, nbrs=(1,), alpha=np.ones(1),
                       gamma=np.ones(1), beta=np.ones((1, 1)),
                       u_max=np.ones(1), ram_cap=4.0, ram_req=np.ones(1))
  params1 = NodeParams(node_id=1, nbrs=(0,), alpha=np.ones(1),
                       gamma=np.ones(1), beta=np.ones((1, 1)),
                       u_max=np.ones(1), ram_cap=4.0, ram_req=np.ones(1))
  nodes = [PlasmaNode(params0, opts, np.random.default_rng(1)),
           PlasmaNode(params1, opts, np.random.default_rng(2))]
  engine = PlasmaEngine(nodes, opts, rng)
  engine.run_rounds(10, np.zeros((2, 1), dtype=int))
  # exactly deg(i) heartbeats per node per round, no hidden channels
  assert engine.hb_count == 10 * 2 * 1


def test_options_reject_nonpositive_rounds_per_step():
  with pytest.raises(ValueError, match="rounds_per_step"):
    PlasmaOptions(rounds_per_step=0)


def test_options_reject_nonpositive_w():
  with pytest.raises(ValueError, match="W"):
    PlasmaOptions(W=0.0)


def test_options_reject_latency_beyond_staleness():
  with pytest.raises(ValueError, match="hb_latency_rounds"):
    PlasmaOptions(hb_latency_rounds=4, staleness_rounds=3)
