import numpy as np
import pytest

from plasma.baselines.milp_baseline import routing_lp
from plasma.engine import PlasmaEngine
from plasma.core.types import LOCAL, REJ, PlasmaOptions
from plasma.core.routing import choose_target, target_weights, update_conductance


def test_local_gate_closes_local_column():
  D_f = np.array([10.0, 0.1, 5.0])  # LOCAL, REJ, one neighbor
  w = target_weights(D_f, local_open=False, nbr_spare=np.array([1.0]),
                     eps_explore=0.01)
  assert w[LOCAL] == 0.0
  assert w[2] == 5.0


def test_neighbor_without_spare_gets_exploration_floor():
  D_f = np.array([1.0, 0.1, 4.0])
  w = target_weights(D_f, local_open=True, nbr_spare=np.array([0.0]),
                     eps_explore=0.01)
  assert w[2] == pytest.approx(4.0 * 0.01)


def test_choose_target_unsplittable_is_argmax():
  rng = np.random.default_rng(0)
  w = np.array([1.0, 0.5, 7.0])
  assert choose_target(rng, w, unsplittable=True) == 2


def test_choose_target_sampled_follows_weights():
  rng = np.random.default_rng(0)
  w = np.array([0.0, 0.0, 1.0])
  assert choose_target(rng, w, unsplittable=False) == 2


def test_conductance_reinforces_accepted_traffic():
  opts = PlasmaOptions()
  D = np.full((1, 3), 1.0)
  phi = np.array([[10.0, 0.0, 0.0]])
  rewards = np.array([[2.0, 0.0, 0.0]])
  D2 = update_conductance(D, phi, rewards, opts)
  assert D2[0, LOCAL] == pytest.approx(0.9 * 1.0 + 0.1 * 10.0 * 2.0)


def test_dead_neighbor_conductance_decays_to_floor():
  opts = PlasmaOptions()
  D = np.full((1, 3), 100.0)
  phi = np.zeros((1, 3))
  rewards = np.zeros((1, 3))
  for _ in range(200):
    D = update_conductance(D, phi, rewards, opts)
  # pure evaporation: (1-mu)^200 * 100 << D_min -> clipped at floor
  assert D[0, 2] == pytest.approx(opts.D_min)


def test_conductance_clipped_above():
  opts = PlasmaOptions()
  D = np.full((1, 3), 1.0)
  phi = np.full((1, 3), 1e9)
  rewards = np.full((1, 3), 1e9)
  D2 = update_conductance(D, phi, rewards, opts)
  assert (D2 <= opts.D_max).all()


from plasma.core.node import NodeParams, PlasmaNode


def _node(r=(2,), u_max=(5.0,), nbrs=(1,), opts=None):
  Nf = len(u_max)
  params = NodeParams(
    node_id=0, nbrs=tuple(nbrs), alpha=np.full(Nf, 2.0),
    gamma=np.full(Nf, 0.1), beta=np.full((len(nbrs), Nf), 1.5),
    u_max=np.array(u_max), ram_cap=100.0, ram_req=np.full(Nf, 2.0),
  )
  node = PlasmaNode(params, opts or PlasmaOptions(), np.random.default_rng(0))
  node.r = np.array(r, dtype=int)
  return node


def test_capacity_gate_never_admits_beyond_r_umax():
  node = _node(r=(2,), u_max=(5.0,))  # capacity 10 req/window
  node.begin_window()
  local = sum(node.route_request(0, round_=0) == LOCAL for _ in range(100))
  counts = node.end_window()
  assert local <= 10
  assert counts.x[0] == local


def test_incoming_forwards_share_the_same_capacity():
  node = _node(r=(1,), u_max=(3.0,))
  node.begin_window()
  admitted = sum(node.admit_forward(0) for _ in range(10))
  assert admitted == 3
  assert node.route_request(0, round_=0) != LOCAL  # capacity exhausted


def test_nack_counts_as_origin_rejection_and_pull():
  node = _node()
  node.begin_window()
  node.record_forward_result(0, col=2, accepted=False)
  node.record_forward_result(0, col=2, accepted=True)
  counts = node.end_window()
  assert counts.z[0] == 1
  assert counts.y[0, 0] == 1
  hb = node.make_heartbeat()
  assert hb.pull[0] == 2  # both attempts are offload pressure


def test_zero_replicas_rejects_or_forwards_everything():
  node = _node(r=(0,))
  node.begin_window()
  for _ in range(20):
    assert node.route_request(0, round_=0) != LOCAL


def test_dead_node_admits_nothing():
  node = _node()
  node.alive = False
  assert node.admit_forward(0) is False


def test_end_window_reinforces_local_conductance():
  node = _node(r=(4,), u_max=(100.0,))
  node.begin_window()
  for _ in range(50):
    node.route_request(0, round_=0)
  d_before = node.D[0, LOCAL]
  node.end_window()
  assert node.D[0, LOCAL] > d_before  # phi*alpha > evaporation at D_init


def test_spare_advertises_floored_capacity():
  node = _node(r=(1,), u_max=(2.085,))  # capacity_units = 2
  node.begin_window()
  assert node.admit_forward(0)
  assert node.admit_forward(0)
  assert not node.admit_forward(0)  # floored capacity exhausted
  node.end_window()
  hb = node.make_heartbeat()
  assert hb.spare[0] == 0.0  # not 0.085: nothing more is admittable


def test_physarum_converges_to_lp_routing_fractions():
  # 2-node line, fixed replicas, stationary integer traffic (lambda = 40):
  # node 0 undersized -> LP says: serve 20 locally, forward 20.
  opts = PlasmaOptions(k_sb=0, hb_latency_rounds=1)
  rng = np.random.default_rng(7)
  make = lambda i, nbrs: PlasmaNode(
    NodeParams(
      node_id=i, nbrs=nbrs, alpha=np.array([2.0]), gamma=np.array([0.1]),
      beta=np.full((len(nbrs), 1), 1.5), u_max=np.array([10.0]),
      ram_cap=100.0, ram_req=np.array([2.0]),
    ), opts, np.random.default_rng(10 + i))
  n0, n1 = make(0, (1,)), make(1, (0,))
  n0.r = np.array([2])   # capacity 20
  n1.r = np.array([4])   # capacity 40
  engine = PlasmaEngine([n0, n1], opts, rng)
  arrivals = np.array([[40], [0]])
  engine.run_rounds(150, arrivals)         # burn-in
  x_acc = np.zeros(1); y_acc = 0.0
  for _ in range(50):                      # measure 50 windows
    res = engine.run_rounds(1, arrivals)
    x_acc += res.x[0]; y_acc += res.y[0, 1, 0]
  lp_obj, lp_x, lp_y, lp_z = routing_lp(
    lam=np.array([[40.0], [0.0]]), r=np.array([[2], [4]]),
    u_max=np.full((2, 1), 10.0), alpha=np.full((2, 1), 2.0),
    beta=np.array([[[0.0], [1.5]], [[1.5], [0.0]]]),
    gamma=np.full((2, 1), 0.1), adjacency=np.array([[0, 1], [1, 0]]),
  )
  assert abs(x_acc[0] / 50 - lp_x[0, 0]) / 40.0 <= 0.05   # +-5% band
  assert abs(y_acc / 50 - lp_y[0, 1, 0]) / 40.0 <= 0.05


def test_local_first_admission_fills_local_capacity_before_any_forward():
  # capacity 10: the FIRST 10 requests must all go LOCAL, deterministically
  node = _node(r=(2,), u_max=(5.0,))
  node.begin_window()
  cols = [node.route_request(0, round_=0) for _ in range(15)]
  assert cols[:10] == [LOCAL] * 10
  assert LOCAL not in cols[10:]
