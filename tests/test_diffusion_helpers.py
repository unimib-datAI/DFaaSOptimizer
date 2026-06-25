import numpy as np
import pandas as pd
import pytest

from decentralized_diffusion import define_assignments, evaluate_assignments


def _base_data(Nn=3, Nf=1):
  data = {None: {
    "Nn": {None: Nn},
    "Nf": {None: Nf},
    "beta": {},
    "gamma": {},
    "demand": {},
    "memory_requirement": {f + 1: 2 for f in range(Nf)},
    "max_utilization": {f + 1: 0.8 for f in range(Nf)},
  }}
  for i in range(Nn):
    for f in range(Nf):
      data[None]["gamma"][(i + 1, f + 1)] = 0.05
      data[None]["demand"][(i + 1, f + 1)] = 1.0
      for j in range(Nn):
        data[None]["beta"][(i + 1, j + 1, f + 1)] = 1.0
  return data


def _ring_neighborhood(Nn=3):
  neighborhood = np.zeros((Nn, Nn))
  for n in range(Nn):
    neighborhood[n, (n + 1) % Nn] = 1
    neighborhood[n, (n - 1) % Nn] = 1
  return neighborhood


def test_define_assignments_greedy_by_utility_no_price_column():
  data = _base_data(Nn=3, Nf=1)
  # buyer 0 prefers seller 1 (beta 2.0) over seller 2 (beta 1.0)
  data[None]["beta"][(1, 2, 1)] = 2.0
  data[None]["beta"][(1, 3, 1)] = 1.0
  omega = np.zeros((3, 1)); omega[0, 0] = 2.0
  blackboard = np.zeros((3, 1)); blackboard[1, 0] = 5.0; blackboard[2, 0] = 5.0
  rho = np.zeros((3,))
  options = {"latency_weight": 0.0, "fairness_weight": 0.0, "unit_bids": False}
  latency = np.zeros((3, 3))
  fairness = np.zeros((3, 1))

  bids, memory_bids, n_buyers = define_assignments(
    omega, blackboard, data, _ring_neighborhood(3), rho,
    options, latency, fairness, force_memory_bids=False,
  )

  assert n_buyers == 1
  assert "b" not in bids.columns
  assert list(bids[["i", "j", "f"]].iloc[0]) == [0, 1, 0]
  assert bids.iloc[0]["d"] == 2.0
  assert bids.iloc[0]["utility"] == 2.0
  assert len(memory_bids) == 0


def test_define_assignments_requests_replicas_when_no_capacity_seller():
  data = _base_data(Nn=2, Nf=1)
  omega = np.zeros((2, 1)); omega[0, 0] = 2.0
  blackboard = np.zeros((2, 1))           # no capacity sellers
  rho = np.zeros((2,)); rho[1] = 4.0       # neighbor 1 has spare memory
  options = {"latency_weight": 0.0, "fairness_weight": 0.0, "unit_bids": False}
  neighborhood = np.array([[0.0, 1.0], [1.0, 0.0]])

  bids, memory_bids, _ = define_assignments(
    omega, blackboard, data, neighborhood, rho,
    options, np.zeros((2, 2)), np.zeros((2, 1)), force_memory_bids=False,
  )

  assert len(bids) == 0
  assert list(memory_bids[["i", "j", "f"]].iloc[0]) == [0, 1, 0]
