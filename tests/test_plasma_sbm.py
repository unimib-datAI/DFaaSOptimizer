import numpy as np
import pytest

from plasma.core.types import PlasmaOptions
from plasma.core.sbm import (
  HamiltonianContext, bits_per_fn, brute_force, decode_spins, dsb_minimize,
  hamiltonian, r_max_per_fn, repair,
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
