"""Regression tests for the shared no_ping_pong guard.

The FRALB no_ping_pong constraint (validate_centralized_solution) forbids a node
from both sending and receiving the same function. The decentralized methods that
accumulate ``y`` across rounds must not produce such a state. Two reference
implementations already inline this (decentralized_auction / _potentialgame);
these tests pin the behaviour for the ones that had no guard: diffusion, powerd,
dual, bestresponse and run_faasmadea. The invariant enforced everywhere: a node
that is a *sender* of f (residual demand ``omega`` or already-forwarded ``y``)
is never picked as a *host* of f.
"""

import numpy as np
import pandas as pd

from utils.centralized import ping_pong_forbidden_hosts
from decentralized_diffusion import define_assignments
from decentralized_powerd import sample_assignments
from decentralized_bestresponse import best_response_sweep
from decentralized_dual import dual_coordination_round
from run_faasmadea import evaluate_bids


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


def _ring(Nn=3):
  neighborhood = np.zeros((Nn, Nn))
  for n in range(Nn):
    neighborhood[n, (n + 1) % Nn] = 1
    neighborhood[n, (n - 1) % Nn] = 1
  return neighborhood


DIFF_OPTIONS = {"latency_weight": 0.0, "fairness_weight": 0.0, "unit_bids": False}
POWERD_OPTIONS = {**DIFF_OPTIONS, "d": 2, "criterion": "score"}


def _no_ping_pong(y, tol=1e-9):
  sends = y.sum(axis=1) > tol
  receives = y.sum(axis=0) > tol
  return not (sends & receives).any()


def test_helper_flags_senders_only():
  omega = np.array([[1.0], [0.0], [0.0]])
  y = np.zeros((3, 3, 1)); y[1, 2, 0] = 4.0            # node 1 already forwarded f
  forbidden = ping_pong_forbidden_hosts(omega, y)
  assert forbidden[0, 0] and forbidden[1, 0] and not forbidden[2, 0]


def test_diffusion_excludes_seller_with_residual_demand():
  data = _base_data()
  omega = np.zeros((3, 1)); omega[0, 0] = 2.0; omega[1, 0] = 1.0  # 1 is a sender
  blackboard = np.zeros((3, 1)); blackboard[1, 0] = 5.0; blackboard[2, 0] = 5.0
  bids, _, _ = define_assignments(
    omega, blackboard, data, _ring(), np.zeros(3),
    DIFF_OPTIONS, np.zeros((3, 3)), np.zeros((3, 1)), force_memory_bids=False,
  )
  assert (bids["j"] != 1).all()               # node 1 must not host despite capacity
  assert (bids["j"] == 2).all()


def test_diffusion_excludes_seller_that_already_forwarded():
  data = _base_data()
  omega = np.zeros((3, 1)); omega[0, 0] = 2.0
  blackboard = np.zeros((3, 1)); blackboard[1, 0] = 5.0; blackboard[2, 0] = 5.0
  current_y = np.zeros((3, 3, 1)); current_y[1, 2, 0] = 3.0   # node 1 sent already
  bids, _, _ = define_assignments(
    omega, blackboard, data, _ring(), np.zeros(3),
    DIFF_OPTIONS, np.zeros((3, 3)), np.zeros((3, 1)), force_memory_bids=False,
    current_y=current_y,
  )
  assert (bids["j"] != 1).all()


def test_powerd_excludes_seller_with_residual_demand():
  data = _base_data()
  omega = np.zeros((3, 1)); omega[0, 0] = 2.0; omega[1, 0] = 1.0
  blackboard = np.zeros((3, 1)); blackboard[1, 0] = 5.0; blackboard[2, 0] = 5.0
  bids, _, _ = sample_assignments(
    omega, blackboard, data, _ring(), np.zeros(3),
    POWERD_OPTIONS, np.zeros((3, 3)), np.zeros((3, 1)),
    force_memory_bids=False, rng=np.random.default_rng(0),
  )
  assert (bids["j"] != 1).all()


def test_bestresponse_receiver_does_not_offload_and_sender_not_host():
  data = _base_data()
  omega = np.zeros((3, 1)); omega[0, 0] = 2.0     # node 0 wants to offload
  residual_capacity = np.zeros((3, 1)); residual_capacity[1, 0] = 5.0
  current_y = np.zeros((3, 3, 1))
  current_y[2, 0, 0] = 3.0     # node 2 -> node 0: node 0 already RECEIVES f
  current_y[1, 2, 0] = 1.0     # node 1 already SENDS f
  y_inc, _, _, _, _ = best_response_sweep(
    omega, residual_capacity, data, _ring(), np.zeros(3),
    DIFF_OPTIONS, np.zeros((3, 3)), np.zeros((3, 1)), force_memory_bids=False,
    order="fixed", response="greedy", current_y=current_y,
  )
  final_y = current_y + y_inc
  assert final_y[0, :, 0].sum() == 0.0          # receiver 0 must not send f
  assert _no_ping_pong(final_y)


def test_madea_evaluate_bids_recovers_ping_pong_free_y():
  data = _base_data()
  # node 1 already forwarded f (1 -> 0): node 1 is a sender, node 0 a receiver
  last_y = np.zeros((3, 3, 1)); last_y[1, 0, 0] = 2.0
  blackboard = np.zeros((3, 1)); blackboard[0, 0] = 5.0; blackboard[1, 0] = 5.0
  bids = pd.DataFrame({
    "i": [2, 2], "j": [0, 1], "f": [0, 0], "d": [2.0, 2.0], "b": [1.0, 1.0],
  })
  capacity = np.ones((3, 1)) * 10.0
  y, _, _, _ = evaluate_bids(
    bids, blackboard, data, last_y,
    np.zeros((3, 1)), np.zeros((3, 1)), capacity, np.zeros((3, 1)),
    {"eta": 0.0, "zeta": 0.0},
  )
  assert y[:, 1, 0].sum() == 0.0                # node 1 sent already -> no host
  assert y[2, 0, 0] == 2.0                      # host that only received still accepts
  assert _no_ping_pong(last_y + y)


def test_dual_round_recovers_ping_pong_free_y():
  data = _base_data(Nn=3, Nf=1)
  omega = np.zeros((3, 1)); omega[0, 0] = 2.0; omega[1, 0] = 1.0
  residual_capacity = np.zeros((3, 1))
  residual_capacity[1, 0] = 3.0     # node 1 is buyer AND advertises capacity
  residual_capacity[2, 0] = 3.0
  options = {
    "alpha0": 0.5, "step_rule": "sqrt", "theta": 1.0,
    "max_inner_iterations": 100, "gap_tolerance": 0.01,
    "latency_weight": 0.0, "fairness_weight": 0.0,
  }
  y_inc, _, _, _ = dual_coordination_round(
    omega, residual_capacity, data, _ring(), np.zeros(3),
    options, np.zeros((3, 3)), np.zeros((3, 1)),
  )
  assert y_inc[:, 1, 0].sum() == 0.0            # node 1 has residual demand -> no host
  assert _no_ping_pong(y_inc)
