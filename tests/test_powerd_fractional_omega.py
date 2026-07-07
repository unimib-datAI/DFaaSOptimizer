"""powerd.sample_assignments must terminate with fractional residual demand.

With integer decision variables (VAR_TYPE = int) a fractional omega made the
greedy sampling loop spin forever: once the residual dropped below one unit,
q = int(...) became 0, assigned never grew and a seller with spare capacity was
never dropped. The loop must stop instead of appending zero-quantity bids.
"""

import numpy as np

from decentralized_powerd import sample_assignments


def _base_data(Nn=3, Nf=1):
  data = {None: {
    "Nn": {None: Nn}, "Nf": {None: Nf},
    "beta": {}, "gamma": {}, "demand": {},
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
  n = np.zeros((Nn, Nn))
  for k in range(Nn):
    n[k, (k + 1) % Nn] = 1
    n[k, (k - 1) % Nn] = 1
  return n


def test_sample_assignments_terminates_on_fractional_omega():
  data = _base_data()
  omega = np.zeros((3, 1)); omega[0, 0] = 2.5      # fractional residual demand
  blackboard = np.zeros((3, 1)); blackboard[1, 0] = 5.0; blackboard[2, 0] = 5.0
  options = {
    "latency_weight": 0.0, "fairness_weight": 0.0, "unit_bids": False,
    "d": 2, "criterion": "score",
  }
  # would hang before the fix; pytest-timeout not required, the guard bounds it
  bids, _, _ = sample_assignments(
    omega, blackboard, data, _ring(), np.zeros(3),
    options, np.zeros((3, 3)), np.zeros((3, 1)),
    force_memory_bids=False, rng=np.random.default_rng(0),
  )
  placed = bids["d"].sum() if len(bids) else 0
  assert placed <= omega[0, 0] + 1e-9        # never over-offloads
  assert placed == 2                          # integer part of 2.5 is placed
  assert (bids["d"] > 0).all()                # no zero-quantity bids
