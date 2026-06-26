import numpy as np
import pandas as pd

from decentralized_bestresponse import best_response_sweep


def _base_data(Nn=4, Nf=1):
  data = {None: {
    "Nn": {None: Nn},
    "Nf": {None: Nf},
    "beta": {},
    "gamma": {},
  }}
  for i in range(Nn):
    for f in range(Nf):
      data[None]["gamma"][(i + 1, f + 1)] = 0.05
      for j in range(Nn):
        data[None]["beta"][(i + 1, j + 1, f + 1)] = 1.0
  return data


def _full_neighborhood(Nn=4):
  return np.ones((Nn, Nn)) - np.eye(Nn)


def _opts():
  return {"latency_weight": 0.0, "fairness_weight": 0.0}


def test_sweep_ledger_is_order_dependent():
  # buyers 0 and 1 both want 2 from the only capacity seller 2 (cap 2);
  # in a fixed sequential sweep, node 0 (first) takes it all, node 1 gets none.
  data = _base_data(Nn=3, Nf=1)
  omega = np.zeros((3, 1)); omega[0, 0] = 2.0; omega[1, 0] = 2.0
  residual = np.zeros((3, 1)); residual[2, 0] = 2.0
  neighborhood = np.zeros((3, 3))
  neighborhood[0, 2] = 1; neighborhood[1, 2] = 1
  rho = np.zeros((3,))

  y, mem, n_active, placed, rt = best_response_sweep(
    omega, residual, data, neighborhood, rho, _opts(),
    np.zeros((3, 3)), np.zeros((3, 1)), force_memory_bids=False,
    order="fixed", response="greedy")

  assert y[0, 2, 0] == 2.0      # first node claimed the shared capacity
  assert y[1, 2, 0] == 0.0      # later node saw the decremented ledger
  assert placed == 2.0
  assert rt == 0.0


def test_sweep_fixed_order_is_deterministic():
  data = _base_data()
  data[None]["beta"][(1, 2, 1)] = 3.0
  data[None]["beta"][(1, 3, 1)] = 2.0
  omega = np.zeros((4, 1)); omega[0, 0] = 3.0
  residual = np.zeros((4, 1)); residual[1, 0] = 2; residual[2, 0] = 2
  args = (omega, residual, data, _full_neighborhood(), np.zeros((4,)),
          _opts(), np.zeros((4, 4)), np.zeros((4, 1)))

  y1, *_ = best_response_sweep(*args, force_memory_bids=False,
                               order="fixed", response="greedy")
  y2, *_ = best_response_sweep(*args, force_memory_bids=False,
                               order="fixed", response="greedy")
  assert np.array_equal(y1, y2)


def test_sweep_random_order_reproducible_with_same_seed():
  data = _base_data()
  for j in [1, 2, 3]:
    data[None]["beta"][(1, j + 1, 1)] = float(j)
  omega = np.zeros((4, 1)); omega[0, 0] = 2.0
  residual = np.zeros((4, 1)); residual[1, 0] = 1; residual[2, 0] = 1; residual[3, 0] = 1
  args = (omega, residual, data, _full_neighborhood(), np.zeros((4,)),
          _opts(), np.zeros((4, 4)), np.zeros((4, 1)))

  y1, *_ = best_response_sweep(*args, force_memory_bids=False, order="random",
                               response="greedy", rng=np.random.default_rng(7))
  y2, *_ = best_response_sweep(*args, force_memory_bids=False, order="random",
                               response="greedy", rng=np.random.default_rng(7))
  assert np.array_equal(y1, y2)


def test_sweep_threshold_excludes_unconvenient_seller():
  data = _base_data(Nn=2, Nf=1)
  data[None]["beta"][(1, 2, 1)] = -1.0   # score -1.0 <= -0.05 => excluded
  omega = np.zeros((2, 1)); omega[0, 0] = 1.0
  residual = np.zeros((2, 1)); residual[1, 0] = 5
  neighborhood = np.array([[0.0, 1.0], [1.0, 0.0]])

  y, mem, n_active, placed, rt = best_response_sweep(
    omega, residual, data, neighborhood, np.zeros((2,)), _opts(),
    np.zeros((2, 2)), np.zeros((2, 1)), force_memory_bids=False,
    order="fixed", response="greedy")

  assert placed == 0.0
  assert len(mem) == 0


def test_sweep_emits_memory_bids_when_no_capacity():
  data = _base_data(Nn=2, Nf=1)
  omega = np.zeros((2, 1)); omega[0, 0] = 2.0
  residual = np.zeros((2, 1))                 # no capacity sellers
  rho = np.zeros((2,)); rho[1] = 4.0           # neighbour 1 has spare memory
  neighborhood = np.array([[0.0, 1.0], [1.0, 0.0]])

  y, mem, n_active, placed, rt = best_response_sweep(
    omega, residual, data, neighborhood, rho, _opts(),
    np.zeros((2, 2)), np.zeros((2, 1)), force_memory_bids=False,
    order="fixed", response="greedy")

  assert placed == 0.0
  assert list(mem[["i", "j", "f"]].iloc[0]) == [0, 1, 0]


def test_sweep_reopt_branch_caps_omega_via_reopt_fn():
  # response="reopt" calls reopt_fn before placement; a fake fn halves omega.
  data = _base_data(Nn=2, Nf=1)
  omega = np.zeros((2, 1)); omega[0, 0] = 4.0
  residual = np.zeros((2, 1)); residual[1, 0] = 10
  neighborhood = np.array([[0.0, 1.0], [1.0, 0.0]])
  calls = {}

  def fake_reopt(node, omega_ub_row):
    calls["node"] = node
    calls["ub"] = list(omega_ub_row)
    return np.array([2.0]), 0.5     # capped to 2.0, 0.5s runtime

  y, mem, n_active, placed, rt = best_response_sweep(
    omega, residual, data, neighborhood, np.zeros((2,)), _opts(),
    np.zeros((2, 2)), np.zeros((2, 1)), force_memory_bids=False,
    order="fixed", response="reopt", reopt_fn=fake_reopt)

  assert calls["node"] == 0
  assert calls["ub"] == [10.0]      # accessible neighbour capacity
  assert placed == 2.0              # placed the capped amount, not 4.0
  assert rt == 0.5
