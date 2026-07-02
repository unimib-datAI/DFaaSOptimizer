import numpy as np

from decentralized_potentialgame import (
  compute_node_utility,
  compute_z,
  split_omega,
)
from utils.faasmacro import compute_centralized_objective


def _data_3n_1f():
  # 3 nodes, 1 function; node 0 offloads, nodes 1-2 are sellers
  return {None: {
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
  }}


def test_split_omega_prefers_higher_beta_and_respects_ledger():
  data = _data_3n_1f()
  ledger = np.array([[0.0], [1.5], [10.0]])
  row = split_omega(0, np.array([4.0]), ledger, {1, 2}, data)
  # 1.5 to node 1 (beta 2.0, capped by ledger), remaining 2.5 to node 2
  assert np.isclose(row[1, 0], 1.5)
  assert np.isclose(row[2, 0], 2.5)
  # ledger consumed in place
  assert np.isclose(ledger[1, 0], 0.0)
  assert np.isclose(ledger[2, 0], 7.5)


def test_split_omega_skips_inconvenient_sellers():
  data = _data_3n_1f()
  data[None]["beta"][(1, 2, 1)] = -0.2  # below -gamma = -0.1: worse than Cloud
  ledger = np.array([[0.0], [10.0], [10.0]])
  row = split_omega(0, np.array([4.0]), ledger, {1, 2}, data)
  assert np.isclose(row[1, 0], 0.0)
  assert np.isclose(row[2, 0], 4.0)


def test_node_utilities_sum_to_potential():
  data = _data_3n_1f()
  x = np.array([[2.0], [4.0], [4.0]])
  y = np.zeros((3, 3, 1))
  y[0, 1, 0] = 1.0
  z = compute_z(x, y, data)
  assert np.isclose(z[0, 0], 1.0)  # 4 - 2 - 1
  phi = compute_centralized_objective(data, x, y, z)
  total = sum(compute_node_utility(i, x, y, z, data) for i in range(3))
  assert np.isclose(total, phi)
