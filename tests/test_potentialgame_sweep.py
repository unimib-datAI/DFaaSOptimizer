import numpy as np
import pytest

from decentralized_potentialgame import (
  check_pg_stopping,
  compute_z,
  node_move,
  potential_game_sweep,
)
from run_faasmadea import compute_residual_capacity
from utils.faasmacro import compute_centralized_objective


def _data_2n_1f():
  # node 0: load 4, initially served locally (r=5 -> cap 4.0)
  # node 1: load 0-ish, big idle capacity (r=10 -> cap 8.0)
  # beta(0->1)=2.0 > alpha=1.0: offloading everything is the improving move
  return {None: {
    "Nn": {None: 2},
    "Nf": {None: 1},
    "incoming_load": {(1, 1): 4.0, (2, 1): 0.001},
    "demand": {(1, 1): 1.0, (2, 1): 1.0},
    "max_utilization": {1: 0.8},
    "memory_capacity": {1: 100, 2: 100},
    "memory_requirement": {1: 2},
    "alpha": {(1, 1): 1.0, (2, 1): 1.0},
    "delta": {(1, 1): 0.2, (2, 1): 0.2},
    "gamma": {(1, 1): 0.1, (2, 1): 0.1},
    "beta": {(1, 2, 1): 2.0, (2, 1, 1): 0.5},
    "neighborhood": {(1, 2): 1, (2, 1): 1},
  }}


def _state():
  x = np.array([[4.0], [0.001]])
  y = np.zeros((2, 2, 1))
  r = np.array([[5.0], [10.0]])
  return x, y, r


def _offload_all_proposal(i, omega_ub_row):
  # node 0 proposes moving everything out; node 1 proposes staying put
  if i == 0:
    return (
      np.array([0.0]), np.array([0.0]),
      np.minimum(np.array([4.0]), omega_ub_row), 0.0,
    )
  return np.array([0.001]), np.array([10.0]), np.array([0.0]), 0.0


def test_node_move_accepts_improving_move_and_updates_state():
  data = _data_2n_1f()
  x, y, r = _state()
  neighborhood = np.array([[0, 1], [1, 0]])
  rho = np.zeros(2)
  accepted, delta_u, bids, _ = node_move(
    0, x, y, r, data, neighborhood, rho, 1e-6, _offload_all_proposal, 1e-9
  )
  assert accepted
  assert delta_u > 0
  # 4.0 offloaded to node 1 (residual cap 8.0 - 0.001)
  assert np.isclose(y[0, 1, 0], 4.0)
  assert np.isclose(x[0, 0], 0.0)


def test_node_move_rejects_non_improving_move():
  data = _data_2n_1f()
  data[None]["beta"][(1, 2, 1)] = 0.5  # now worse than serving locally
  x, y, r = _state()
  neighborhood = np.array([[0, 1], [1, 0]])
  accepted, _, _, _ = node_move(
    0, x, y, r, data, neighborhood, np.zeros(2), 1e-6,
    _offload_all_proposal, 1e-9,
  )
  assert not accepted
  # state untouched
  assert np.isclose(x[0, 0], 4.0)
  assert np.isclose(y.sum(), 0.0)


def test_sweep_raises_potential_then_certifies_equilibrium():
  data = _data_2n_1f()
  x, y, r = _state()
  neighborhood = np.array([[0, 1], [1, 0]])
  rng = np.random.default_rng(0)
  phi0 = compute_centralized_objective(data, x, y, compute_z(x, y, data))
  n_acc, delta_phi, bids, _ = potential_game_sweep(
    x, y, r, data, neighborhood, np.zeros(2), 1e-6, 1e-9,
    "fixed", rng, _offload_all_proposal,
  )
  phi1 = compute_centralized_objective(data, x, y, compute_z(x, y, data))
  assert n_acc == 1
  assert delta_phi > 0
  assert np.isclose(phi1 - phi0, delta_phi)
  # second sweep: same proposal is no longer an improvement -> equilibrium
  n_acc2, delta_phi2, _, _ = potential_game_sweep(
    x, y, r, data, neighborhood, np.zeros(2), 1e-6, 1e-9,
    "fixed", rng, _offload_all_proposal,
  )
  assert n_acc2 == 0
  assert np.isclose(delta_phi2, 0.0)
  stop, why = check_pg_stopping(1, 100, n_acc2, 0, 0.0, np.inf)
  assert stop and why == "epsilon-Nash equilibrium certified"
  # ledger conservation: residual capacity consistent with committed flows
  _, residual, ell = compute_residual_capacity(x, y, r, data)
  assert np.isclose(ell[1, 0], 4.001)


def test_check_pg_stopping_guards():
  stop, why = check_pg_stopping(99, 100, 5, 0, 0.0, np.inf)
  assert stop and why == "max iterations reached"
  stop, why = check_pg_stopping(0, 100, 5, 0, 100.0, 50.0)
  assert stop and "time limit" in why
  stop, why = check_pg_stopping(0, 100, 5, 0, 0.0, np.inf)
  assert not stop


def test_node_move_blocks_offload_of_committed_inbound_function():
  # node 0 already receives f from node 1: node 0 must not offload f
  # (FRALB no_ping_pong), even when offloading would improve its utility
  data = _data_2n_1f()
  data[None]["incoming_load"][(2, 1)] = 4.0
  x = np.array([[4.0], [2.0]])
  y = np.zeros((2, 2, 1))
  y[1, 0, 0] = 2.0  # committed inbound onto node 0
  r = np.array([[10.0], [3.0]])
  neighborhood = np.array([[0, 1], [1, 0]])
  seen_ub = {}

  def probe_proposal(i, omega_ub_row):
    seen_ub[i] = omega_ub_row.copy()
    return np.array([0.0]), np.array([0.0]), np.array([4.0]), 0.0

  accepted, _, _, _ = node_move(
    0, x, y, r, data, neighborhood, np.zeros(2), 1e-6, probe_proposal, 1e-9
  )
  # the cap advertised to the proposal must be zero for the inbound function
  assert np.isclose(seen_ub[0][0], 0.0)
  # and even a rogue proposal cannot place anything (omega clamped to cap)
  assert np.isclose(y[0, :, 0].sum(), 0.0)


def test_node_move_excludes_sellers_that_offload_same_function():
  # 3 nodes: node 1 currently offloads f to node 2, so node 1 must not be a
  # seller of f for node 0 despite having residual capacity
  data = {None: {
    "Nn": {None: 3},
    "Nf": {None: 1},
    "incoming_load": {(1, 1): 4.0, (2, 1): 4.0, (3, 1): 4.0},
    "demand": {(i, 1): 1.0 for i in (1, 2, 3)},
    "max_utilization": {1: 0.8},
    "memory_capacity": {1: 100, 2: 100, 3: 100},
    "memory_requirement": {1: 2},
    "alpha": {(i, 1): 1.0 for i in (1, 2, 3)},
    "delta": {(i, 1): 0.2 for i in (1, 2, 3)},
    "gamma": {(i, 1): 0.1 for i in (1, 2, 3)},
    "beta": {
      (1, 2, 1): 2.0, (1, 3, 1): 1.5,
      (2, 1, 1): 1.0, (2, 3, 1): 1.0,
      (3, 1, 1): 1.0, (3, 2, 1): 1.0,
    },
    "neighborhood": {
      (1, 2): 1, (1, 3): 1, (2, 1): 1, (2, 3): 1, (3, 1): 1, (3, 2): 1,
    },
  }}
  x = np.array([[4.0], [2.0], [4.0]])
  y = np.zeros((3, 3, 1))
  y[1, 2, 0] = 2.0  # node 1 offloads f to node 2
  # node 1 has spare capacity (r=10 -> cap 8, serves only x=2), node 2 is full
  r = np.array([[5.0], [10.0], [8.0]])
  neighborhood = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

  def offload_all(i, omega_ub_row):
    return (
      np.array([0.0]), np.array([0.0]),
      np.minimum(np.array([4.0]), omega_ub_row), 0.0,
    )

  node_move(
    0, x, y, r, data, neighborhood, np.zeros(3), 1e-6, offload_all, 1e-9
  )
  # nothing may be placed on node 1 (it offloads f): no ping-pong
  assert np.isclose(y[0, 1, 0], 0.0)
  # combined y must satisfy the centralized no-ping-pong invariant
  assert not ((y.sum(axis=1) > 1e-9) & (y.sum(axis=0) > 1e-9)).any()


def test_node_move_bids_on_memory_when_cloud_residue_remains():
  # ledger is empty (neighbour saturated), so the capped proposal asks for
  # omega=0 and places 0; the z>0 residue must still trigger memory bids
  # towards neighbours with memory slack
  data = _data_2n_1f()
  x = np.array([[3.0], [4.0]])  # node 0: 1.0 unserved -> z; node 1 full
  y = np.zeros((2, 2, 1))
  r = np.array([[3.75], [5.0]])  # node 1: cap 4.0 == x -> residual 0
  neighborhood = np.array([[0, 1], [1, 0]])
  rho = np.array([0.0, 10.0])  # node 1 has memory slack

  def capped_proposal(i, omega_ub_row):
    return np.array([3.0]), np.array([3.75]), omega_ub_row.copy(), 0.0

  _, _, bids, _ = node_move(
    0, x, y, r, data, neighborhood, rho, 1e-6, capped_proposal, 1e-9
  )
  assert bids["j"] == [1] and bids["f"] == [0]


def test_compute_rho_tracks_committed_replicas():
  from decentralized_potentialgame import compute_rho
  data = _data_2n_1f()  # memory_capacity 100, memory_requirement 2
  r = np.array([[5.0], [10.0]])
  rho = compute_rho(r, data)
  assert np.allclose(rho, [90.0, 80.0])
  # replicas grown by a move: slack must shrink accordingly
  r[0, 0] = 50.0
  assert np.allclose(compute_rho(r, data), [0.0, 80.0])
