import numpy as np
import pytest

from plasma.core.types import PlasmaOptions
from plasma.core.sbm import (
  HamiltonianContext, bits_per_fn, brute_force, decode_spins, dsb_minimize,
  exact_minimize, hamiltonian, r_max_per_fn, repair,
)


def test_encoding_roundtrip():
  r_max = np.array([5, 0, 1])
  bits = bits_per_fn(r_max)
  assert bits.tolist() == [3, 0, 1]
  # spins for r = (5, -, 1): 5 = 101b -> bits (1,0,1) -> spins (+1,-1,+1)
  s = np.array([1, -1, 1, 1])
  assert decode_spins(s, bits, r_max).tolist() == [5, 0, 1]


def test_decode_clips_to_r_max():
  r_max = np.array([5])           # 3 bits encode up to 7
  s = np.array([1, 1, 1])         # decodes to 7
  assert decode_spins(s, bits_per_fn(r_max), r_max).tolist() == [5]


def _ctx(**over):
  base = dict(
    benefit=np.array([3.0, 1.0]), ram_req=np.array([2.0, 2.0]), ram_cap=8.0,
    demand_hat=np.array([4.0, 1.0]), margin=np.array([1.0, 0.5]),
    u_max=np.array([5.0, 5.0]), r_prev=np.array([1, 0]),
    A=10.0, B=1.0, C=0.1, switch_cost=1.0,
  )
  base.update(over)
  return HamiltonianContext(**base)


def test_hamiltonian_penalizes_ram_violation():
  ctx = _ctx()
  ok = hamiltonian(np.array([2, 2]), ctx)       # RAM = 8 <= 8
  bad = hamiltonian(np.array([3, 2]), ctx)      # RAM = 10 > 8
  assert bad > ok


def test_hamiltonian_churn_term():
  ctx = _ctx(B=0.0, A=0.0, benefit=np.zeros(2))
  h_stay = hamiltonian(np.array([1, 0]), ctx)
  h_move = hamiltonian(np.array([3, 2]), ctx)
  assert h_move == pytest.approx(h_stay + 0.1 * 1.0 * (2 + 2))


def test_repair_restores_feasibility_dropping_lowest_benefit():
  r = np.array([3, 3])  # RAM = 12 > 8
  fixed = repair(r, benefit=np.array([3.0, 1.0]), ram_req=np.array([2.0, 2.0]),
                 ram_cap=8.0)
  assert (fixed * np.array([2.0, 2.0])).sum() <= 8.0
  assert fixed[0] >= fixed[1]  # low-benefit f=1 dropped first


def test_dsb_matches_brute_force_on_random_hamiltonians():
  opts = PlasmaOptions(n_sb_steps=400)
  hits = 0
  trials = 200
  for trial in range(trials):
    rng = np.random.default_rng(trial)
    n = 8
    J = rng.normal(size=(n, n)); J = (J + J.T) / 2; np.fill_diagonal(J, 0.0)
    h = rng.normal(size=n)

    def H(s, J=J, h=h):
      return float(-0.5 * s @ J @ s - h @ s)

    s_star = brute_force(H, n)
    s_dsb = dsb_minimize(H, n, opts, rng)
    if H(s_dsb) <= H(s_star) + 1e-9:
      hits += 1
  assert hits >= 0.95 * trials


def test_brute_force_exact_on_tiny_instance():
  def H(s):
    return float(-(s[0] * s[1]) - s[0])  # ground state (+1, +1)
  assert brute_force(H, 2).tolist() == [1, 1]


from plasma.core.node import NodeParams, PlasmaNode


def _sb_node(p_commit=1.0, n_hyst=1, k_sb=1, ram_cap=8.0):
  params = NodeParams(
    node_id=0, nbrs=(), alpha=np.array([3.0, 1.0]),
    gamma=np.array([0.1, 0.1]), beta=np.zeros((0, 2)),
    u_max=np.array([5.0, 5.0]), ram_cap=ram_cap, ram_req=np.array([2.0, 2.0]),
  )
  opts = PlasmaOptions(p_commit=p_commit, n_hyst=n_hyst, k_sb=k_sb,
                       n_sb_steps=200)
  return PlasmaNode(params, opts, np.random.default_rng(1))


def test_init_replicas_spread_fills_ram_round_robin():
  node = _sb_node()
  node.init_replicas()
  assert (node.r * node.params.ram_req).sum() <= node.params.ram_cap
  assert node.r.sum() == 4  # 8 RAM / 2 per replica


def test_sb_pass_grows_replicas_under_demand():
  node = _sb_node(n_hyst=1)
  node.r = np.zeros(2, dtype=int)
  node.demand_hat = np.array([8.0, 0.0])
  committed = node.sb_pass(round_=0)
  assert committed
  assert node.r[0] >= 1
  assert (node.r * node.params.ram_req).sum() <= node.params.ram_cap


def test_hysteresis_requires_consecutive_confirmations():
  node = _sb_node(n_hyst=2)
  node.r = np.zeros(2, dtype=int)
  node.demand_hat = np.array([8.0, 0.0])
  assert node.sb_pass(round_=0) is False  # first proposal only counts
  assert node.sb_pass(round_=1) is True   # second consecutive -> commit


def test_p_commit_zero_never_commits():
  node = _sb_node(p_commit=0.0, n_hyst=1)
  node.demand_hat = np.array([8.0, 0.0])
  for k in range(5):
    assert node.sb_pass(round_=k) is False
  assert node.r.sum() == 0


def test_committed_r_is_always_ram_feasible():
  node = _sb_node(n_hyst=1, ram_cap=4.0)
  node.demand_hat = np.array([50.0, 50.0])  # wants far more than RAM allows
  node.sb_pass(round_=0)
  assert (node.r * node.params.ram_req).sum() <= 4.0


def test_exact_matches_brute_force_ground_state():
  rng = np.random.default_rng(0)
  for trial in range(50):
    Nf = 3
    ram_req = rng.integers(1, 4, Nf).astype(float)
    ram_cap = float(rng.integers(4, 13))
    r_max = np.floor(ram_cap / ram_req).astype(int)
    ctx = HamiltonianContext(
      benefit=rng.uniform(0, 5, Nf), ram_req=ram_req, ram_cap=ram_cap,
      demand_hat=rng.uniform(0, 10, Nf), margin=rng.uniform(0, 2, Nf),
      u_max=rng.uniform(1, 6, Nf), r_prev=rng.integers(0, 3, Nf),
      A=10.0, B=1.0, C=0.1, switch_cost=1.0,
    )
    r_star = exact_minimize(ctx, r_max)
    assert (ctx.ram_req * r_star).sum() <= ctx.ram_cap + 1e-9
    # brute force over all feasible r
    from itertools import product as iproduct
    best = min(
      (hamiltonian(np.array(rr), ctx)
       for rr in iproduct(*[range(m + 1) for m in r_max])
       if (ctx.ram_req * np.array(rr)).sum() <= ctx.ram_cap + 1e-9))
    assert hamiltonian(r_star, ctx) <= best + 1e-9


def test_exact_rejects_fractional_ram():
  ctx = _ctx(ram_req=np.array([1.5, 2.0]))
  with pytest.raises(ValueError, match="dsb"):
    exact_minimize(ctx, np.array([5, 4]))


def test_sb_pass_exact_is_deterministic():
  a = _sb_node(n_hyst=1)
  b = _sb_node(n_hyst=1)
  for node in (a, b):
    node.demand_hat = np.array([8.0, 3.0])
    node.sb_pass(round_=0)
  assert np.array_equal(a.r, b.r)
