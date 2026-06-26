import numpy as np
import pandas as pd

from decentralized_powerd import sample_assignments
from decentralized_diffusion import define_assignments


def _base_data(Nn=4, Nf=1):
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


def _full_neighborhood(Nn=4):
  neighborhood = np.ones((Nn, Nn)) - np.eye(Nn)
  return neighborhood


def _opts(d=2, criterion="score", unit_bids=False):
  return {
    "d": d, "criterion": criterion, "unit_bids": unit_bids,
    "latency_weight": 0.0, "fairness_weight": 0.0,
  }


def test_sample_assignments_is_deterministic_given_seed():
  data = _base_data()
  # distinct betas so the random sampling has something to choose between
  data[None]["beta"][(1, 2, 1)] = 3.0
  data[None]["beta"][(1, 3, 1)] = 2.0
  data[None]["beta"][(1, 4, 1)] = 1.5
  omega = np.zeros((4, 1)); omega[0, 0] = 6.0
  blackboard = np.zeros((4, 1)); blackboard[1, 0] = 2; blackboard[2, 0] = 2; blackboard[3, 0] = 2
  rho = np.zeros((4,))
  args = (omega, blackboard, data, _full_neighborhood(), rho,
          _opts(d=2), np.zeros((4, 4)), np.zeros((4, 1)))

  bids1, _, _ = sample_assignments(*args, force_memory_bids=False,
                                   rng=np.random.default_rng(123))
  bids2, _, _ = sample_assignments(*args, force_memory_bids=False,
                                   rng=np.random.default_rng(123))

  pd.testing.assert_frame_equal(bids1, bids2)


def test_sample_assignments_degenerates_to_madig_when_d_covers_candidates():
  # d >= |candidates|, criterion="score", unique scores => same as FaaS-MADiG
  data = _base_data()
  data[None]["beta"][(1, 2, 1)] = 3.0
  data[None]["beta"][(1, 3, 1)] = 2.0
  data[None]["beta"][(1, 4, 1)] = 1.0
  omega = np.zeros((4, 1)); omega[0, 0] = 4.0
  blackboard = np.zeros((4, 1)); blackboard[1, 0] = 2; blackboard[2, 0] = 2; blackboard[3, 0] = 2
  rho = np.zeros((4,))
  common = (omega, blackboard, data, _full_neighborhood(), rho)
  geo = (np.zeros((4, 4)), np.zeros((4, 1)))

  sampled, _, _ = sample_assignments(
    *common, _opts(d=99, criterion="score"), *geo,
    force_memory_bids=False, rng=np.random.default_rng(0))
  greedy, _, _ = define_assignments(
    *common, _opts(d=99, criterion="score"), *geo, force_memory_bids=False)

  pd.testing.assert_frame_equal(
    sampled.reset_index(drop=True), greedy.reset_index(drop=True))


def test_sample_assignments_capacity_criterion_prefers_largest_capacity():
  # score ranks seller 1 best, but seller 2 advertises far more capacity;
  # with d covering both, criterion="capacity" must pick seller 2 first.
  data = _base_data(Nn=3, Nf=1)
  data[None]["beta"][(1, 2, 1)] = 9.0   # seller 1: best score, small capacity
  data[None]["beta"][(1, 3, 1)] = 1.0   # seller 2: worse score, big capacity
  omega = np.zeros((3, 1)); omega[0, 0] = 3.0
  blackboard = np.zeros((3, 1)); blackboard[1, 0] = 1; blackboard[2, 0] = 10
  rho = np.zeros((3,))

  bids, _, _ = sample_assignments(
    omega, blackboard, data, _full_neighborhood(3), rho,
    _opts(d=99, criterion="capacity"), np.zeros((3, 3)), np.zeros((3, 1)),
    force_memory_bids=False, rng=np.random.default_rng(0))

  first = bids.iloc[0]
  assert first["j"] == 2           # largest-capacity seller picked first
  assert first["d"] == 3.0


def test_sample_assignments_threshold_excludes_unconvenient_seller():
  data = _base_data(Nn=2, Nf=1)
  data[None]["gamma"][(1, 1)] = 0.05    # score 0.0 > -0.05 holds; flip sign below
  data[None]["beta"][(1, 2, 1)] = -1.0  # score -1.0 <= -0.05 => excluded
  omega = np.zeros((2, 1)); omega[0, 0] = 1.0
  blackboard = np.zeros((2, 1)); blackboard[1, 0] = 5
  rho = np.zeros((2,))

  bids, memory_bids, _ = sample_assignments(
    omega, blackboard, data, np.array([[0.0, 1.0], [1.0, 0.0]]), rho,
    _opts(d=2), np.zeros((2, 2)), np.zeros((2, 1)),
    force_memory_bids=False, rng=np.random.default_rng(0))

  assert len(bids) == 0
  assert len(memory_bids) == 0


def test_sample_assignments_requests_replicas_when_no_capacity_seller():
  data = _base_data(Nn=2, Nf=1)
  omega = np.zeros((2, 1)); omega[0, 0] = 2.0
  blackboard = np.zeros((2, 1))           # no capacity sellers
  rho = np.zeros((2,)); rho[1] = 4.0       # neighbour 1 has spare memory

  bids, memory_bids, _ = sample_assignments(
    omega, blackboard, data, np.array([[0.0, 1.0], [1.0, 0.0]]), rho,
    _opts(d=2), np.zeros((2, 2)), np.zeros((2, 1)),
    force_memory_bids=False, rng=np.random.default_rng(0))

  assert len(bids) == 0
  assert list(memory_bids[["i", "j", "f"]].iloc[0]) == [0, 1, 0]


def test_sample_assignments_unit_bids_emits_one_unit_per_request():
  data = _base_data(Nn=2, Nf=1)
  omega = np.zeros((2, 1)); omega[0, 0] = 3.0
  blackboard = np.zeros((2, 1)); blackboard[1, 0] = 3
  rho = np.zeros((2,))

  bids, _, _ = sample_assignments(
    omega, blackboard, data, np.array([[0.0, 1.0], [1.0, 0.0]]), rho,
    _opts(d=2, unit_bids=True), np.zeros((2, 2)), np.zeros((2, 1)),
    force_memory_bids=False, rng=np.random.default_rng(0))

  assert len(bids) == 3
  assert (bids["d"] == 1).all()
  assert (bids["j"] == 1).all()
