from plasma.sim.clock import RoundClock


def test_events_delivered_before_round_callback():
  clock = RoundClock()
  trace = []
  clock.schedule(1, lambda: trace.append("hb@1"))
  clock.run(2, on_round=lambda r: trace.append(f"round{r}"))
  assert trace == ["round0", "hb@1", "round1"]


def test_fifo_tie_break():
  clock = RoundClock()
  trace = []
  clock.schedule(0, lambda: trace.append("a"))
  clock.schedule(0, lambda: trace.append("b"))
  clock.run(1, on_round=lambda r: None)
  assert trace == ["a", "b"]


def test_round_counter_persists_across_runs():
  clock = RoundClock()
  seen = []
  clock.run(3, on_round=seen.append)
  clock.run(2, on_round=seen.append)
  assert seen == [0, 1, 2, 3, 4]


def test_past_due_events_flush():
  clock = RoundClock()
  trace = []
  clock.run(2, on_round=lambda r: None)
  clock.schedule(0, lambda: trace.append("late"))
  clock.run(1, on_round=lambda r: trace.append(f"round{r}"))
  assert trace == ["late", "round2"]


import numpy as np

from plasma.core.node import NodeParams, PlasmaNode
from plasma.core.types import PlasmaOptions
from plasma.engine import PlasmaEngine


def _line_engine(Nn=2, Nf=1, opts=None, seed=0, u_max=5.0, ram_cap=8.0):
  opts = opts or PlasmaOptions(k_sb=0)
  rng = np.random.default_rng(seed)
  nodes = []
  for i in range(Nn):
    nbrs = tuple(j for j in (i - 1, i + 1) if 0 <= j < Nn)
    params = NodeParams(
      node_id=i, nbrs=nbrs, alpha=np.full(Nf, 2.0), gamma=np.full(Nf, 0.1),
      beta=np.full((len(nbrs), Nf), 1.5), u_max=np.full(Nf, u_max),
      ram_cap=ram_cap, ram_req=np.full(Nf, 2.0),
    )
    node = PlasmaNode(params, opts, np.random.default_rng(seed + i))
    node.init_replicas()
    nodes.append(node)
  return PlasmaEngine(nodes, opts, rng), nodes


def test_traffic_conservation_every_round():
  engine, _ = _line_engine()
  arrivals = np.array([[8], [8]])
  res = engine.run_rounds(5, arrivals)
  total = res.x + res.z + res.y.sum(axis=1)
  np.testing.assert_allclose(total, arrivals.astype(float))


def test_xi_transposes_y():
  engine, _ = _line_engine()
  res = engine.run_rounds(5, np.array([[20], [0]]))
  np.testing.assert_allclose(res.xi[1, 0, :], res.y[0, 1, :])


def test_dead_node_traffic_redistributes():
  # 3-node line, kill the middle node: node 0 must stop forwarding to it.
  # node 0 is deliberately undersized (capacity 5 < 20 arrivals/round) so it
  # relies on node 1 for the overflow -- the default init_replicas() spread
  # would exactly cover 20 arrivals with no forced forwarding at all.
  opts = PlasmaOptions(k_sb=0)
  engine, nodes = _line_engine(Nn=3, opts=opts)
  nodes[0].r = np.array([1])
  engine.run_rounds(5, np.array([[20], [0], [0]]))
  engine.set_alive(1, False)
  # pure evaporation ((1-mu)^n) needs ~90 rounds from the reinforced level
  # reached above to clear the D_min floor at mu=0.1; 10*staleness_rounds
  # only covers the routing/gate-closing side of "redistributes", not full
  # numeric decay to the floor
  res = engine.run_rounds(100, np.array([[20], [0], [0]]))
  assert res.y[0, 1, 0] == 0.0  # nothing ACKed by a dead node
  # conductance toward the dead neighbor decayed to the floor
  col = 2 + nodes[0].params.nbrs.index(1)
  assert nodes[0].D[0, col] <= opts.D_min * 1.01


def test_phase_locked_commit_thrash_vs_randomized():
  # adversarial: 2-node line, shared demand pulse, SB every round
  def run(p_commit):
    opts = PlasmaOptions(k_sb=1, n_hyst=1, p_commit=p_commit, n_sb_steps=150)
    engine, nodes = _line_engine(Nn=2, opts=opts, seed=3, ram_cap=4.0)
    flips = 0
    prev = [n.r.copy() for n in nodes]
    for _ in range(20):
      engine.run_rounds(1, np.array([[12], [12]]))
      for k, n in enumerate(nodes):
        if not np.array_equal(prev[k], n.r):
          flips += 1
        prev[k] = n.r.copy()
    return flips

  thrash = run(p_commit=1.0)
  calm = run(p_commit=0.5)
  assert calm <= thrash


def test_settles_within_20_slow_ticks_with_default_p_commit():
  opts = PlasmaOptions(k_sb=1, n_hyst=2, p_commit=0.5, n_sb_steps=150)
  engine, nodes = _line_engine(Nn=2, opts=opts, seed=3, ram_cap=4.0)
  last_change = 0
  prev = [n.r.copy() for n in nodes]
  for tick in range(1, 21):
    engine.run_rounds(1, np.array([[12], [12]]))
    for k, n in enumerate(nodes):
      if not np.array_equal(prev[k], n.r):
        last_change = tick
        prev[k] = n.r.copy()
  assert last_change < 20
