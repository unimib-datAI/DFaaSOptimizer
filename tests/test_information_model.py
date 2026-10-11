"""The '-b' variants must not let a seller/receiver see the buyers' beta."""
import numpy as np
import pytest

import madea2
from decentralized_frankwolfe import frankwolfe_assignments, frankwolfe_options


def _auction_data(beta01=5.0, beta02=1.0):
  data = {None: {
    "Nn": {None: 3}, "Nf": {None: 1},
    "beta": {(1, 2, 1): beta01, (1, 3, 1): beta02, (2, 1, 1): 1.0, (2, 3, 1): 1.0,
             (3, 1, 1): 1.0, (3, 2, 1): 1.0},
    "gamma": {(n, 1): 1.0 for n in (1, 2, 3)},
    "incoming_load": {(n, 1): 10 for n in (1, 2, 3)},
  }}
  return data


def _memory_bids(mode, beta01):
  omega = np.array([[4.0], [0.0], [0.0]])
  blackboard = np.zeros((3, 1))  # no spare capacity: only memory bids
  p = np.array([[0.0], [0.2], [0.1]])
  neighborhood = np.ones((3, 3)) - np.eye(3)
  rho = np.array([0.0, 100.0, 100.0])
  options = {"epsilon": 0.01, "latency_weight": 0.0, "fairness_weight": 0.0,
             "unit_bids": True, "memory_bid_value": mode}
  _, memory_bids, _ = madea2.define_bids(
    omega, blackboard, p, _auction_data(beta01), neighborhood, rho, options,
    np.zeros((3, 3)), np.zeros((3, 1)), False)
  return memory_bids.set_index("j")["v"]


def test_beta_mode_reveals_the_private_advantage():
  v = _memory_bids("beta", 5.0)
  assert v[1] == pytest.approx(6.0)  # beta + gamma


def test_bid_mode_sends_auction_bids_not_beta():
  v = _memory_bids("bid", 5.0)
  # utilities: j=1: 6 - 0.2 = 5.8; j=2: 2 - 0.1 = 1.9 -> b1 = 0.2 + 0.01 + 3.9
  assert v[1] == pytest.approx(0.2 + 0.01 + 3.9)
  assert v[2] == pytest.approx(0.1 + 0.01)
  assert all(abs(x - 6.0) > 1e-9 for x in v)


def test_bid_mode_rejects_unknown_values():
  with pytest.raises(ValueError):
    _memory_bids("cheat", 5.0)


def _fw_data():
  Nn = 3
  return {None: {
    "Nn": {None: Nn}, "Nf": {None: 1},
    "beta": {(i + 1, j + 1, 1): (3.0 if i == 0 else 0.5) for i in range(Nn) for j in range(Nn)},
    "gamma": {(i + 1, 1): 0.05 for i in range(Nn)},
    "demand": {(i + 1, 1): 1.0 for i in range(Nn)},
    "memory_capacity": {i + 1: 4 for i in range(Nn)},
    "memory_requirement": {1: 2}, "max_utilization": {1: 1.0},
  }}


def test_price_repair_does_not_depend_on_sender_beta():
  # two senders compete for receiver 2 (4 MB = 2 req/s); with repair by
  # price the outcome must not change when sender 0 raises its beta
  results = []
  for high in (3.0, 30.0):
    data = _fw_data()
    for j in range(3):
      data[None]["beta"][(1, j + 1, 1)] = high
    neighborhood = np.zeros((3, 3))
    neighborhood[0, 2] = neighborhood[2, 0] = neighborhood[1, 2] = neighborhood[2, 1] = 1
    options = frankwolfe_options({"frankwolfe": {"kappa": 1e-6, "repair_order": "price",
                                                 "rounding": "largest_remainder"}})
    y, _ = frankwolfe_assignments(
      np.array([[3.0], [3.0], [0.0]]), np.zeros((3, 3, 1)), np.zeros((3, 1)), np.zeros((3, 1)),
      np.full(3, 4.0), data, neighborhood, np.zeros((3, 3)), np.zeros((3, 1)), options,
      np.random.default_rng(0))
    results.append(y[:, 2, 0].copy())
    assert y[:, 2, 0].sum() <= 2.0 + 1e-9
  assert np.array_equal(results[0], results[1])


def test_repair_order_option_is_validated():
  with pytest.raises(ValueError):
    frankwolfe_options({"frankwolfe": {"repair_order": "beta"}})
