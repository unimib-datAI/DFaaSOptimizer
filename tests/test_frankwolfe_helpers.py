import numpy as np
import pytest

from decentralized_frankwolfe import (
  frankwolfe_assignments,
  frankwolfe_options,
  fw_gains_and_mask,
  fw_parameters,
  relaxed_flows,
  repair_memory,
  round_flows,
)
from run_faasmadea import check_ls_pr_feasibility_from_fixed_y


def _data(Nn=4, Nf=1, memory=10, ram=2, umax=1.0, demand=1.0, beta=None, gamma=0.05):
  data = {None: {
    "Nn": {None: Nn},
    "Nf": {None: Nf},
    "beta": {},
    "gamma": {},
    "demand": {},
    "memory_capacity": {n + 1: memory for n in range(Nn)},
    "memory_requirement": {f + 1: ram for f in range(Nf)},
    "max_utilization": {f + 1: umax for f in range(Nf)},
  }}
  for i in range(Nn):
    for f in range(Nf):
      data[None]["gamma"][(i + 1, f + 1)] = gamma
      data[None]["demand"][(i + 1, f + 1)] = demand
      for j in range(Nn):
        b = 1.0 if beta is None else beta[i][j]
        data[None]["beta"][(i + 1, j + 1, f + 1)] = b
  return data


def _opts(**kwargs):
  options = frankwolfe_options({"frankwolfe": kwargs})
  return options


def _full(Nn):
  return np.ones((Nn, Nn)) - np.eye(Nn)


def _setup(data, omega, neighborhood, x=None, r=None, rho=None, y_prev=None, options=None):
  params = fw_parameters(data)
  Nn, Nf = params["Nn"], params["Nf"]
  options = options or _opts()
  omega = np.asarray(omega, dtype=float).reshape(Nn, Nf)
  y_prev = np.zeros((Nn, Nn, Nf)) if y_prev is None else y_prev
  forbidden = (omega > 0) | (y_prev.sum(axis=1) > 0)
  gains, mask = fw_gains_and_mask(
    params, omega, neighborhood, forbidden, np.zeros((Nn, Nn)),
    np.zeros((Nn, Nf)), options,
  )
  x = np.zeros((Nn, Nf)) if x is None else x
  r = np.zeros((Nn, Nf)) if r is None else r
  rho = np.full(Nn, float(data[None]["memory_capacity"][1])) if rho is None else rho
  return params, omega, x + y_prev.sum(axis=0), r, rho, gains, mask, options


def test_options_defaults_and_validation():
  options = frankwolfe_options({})
  assert options["iterations"] == 100 and options["rounding"] == "random"
  with pytest.raises(ValueError):
    frankwolfe_options({"frankwolfe": {"iterations": 0}})
  with pytest.raises(ValueError):
    frankwolfe_options({"frankwolfe": {"kappa": 0}})
  with pytest.raises(ValueError):
    frankwolfe_options({"frankwolfe": {"rounding": "nearest"}})


def test_mask_excludes_non_neighbors_senders_and_unprofitable_pairs():
  # path 0 - 1 - 2; nodes 0 and 1 both have residual demand: 1 is a sender, so
  # it cannot host; node 3 is not a neighbor of 0.
  Nn = 4
  neighborhood = np.zeros((Nn, Nn))
  for a, b in ((0, 1), (1, 2), (0, 2)):
    neighborhood[a, b] = neighborhood[b, a] = 1
  beta = [[0.0] * Nn for _ in range(Nn)]
  for j in range(Nn):
    beta[0][j] = 1.0
  beta[0][2] = -1.0  # a = -1 + 0.05 < 0 -> unprofitable
  data = _data(Nn=Nn, beta=beta)
  params, omega, *_ , gains, mask, options = _setup(
    data, [5, 3, 0, 0], neighborhood
  )
  assert not mask[0, 1, 0]   # 1 is a sender
  assert not mask[0, 2, 0]   # unprofitable
  assert not mask[0, 3, 0]   # not a neighbor
  assert not mask[2, :, 0].any()  # no residual demand at 2
  assert mask[1, 2, 0]


def test_relaxed_flows_send_everything_to_best_neighbor_when_memory_is_ample():
  Nn = 3
  beta = [[0.0, 1.0, 2.0], [0.0] * 3, [0.0] * 3]
  data = _data(Nn=Nn, beta=beta, memory=1000)
  params, omega, base, r, rho, gains, mask, options = _setup(
    data, [6, 0, 0], _full(Nn)
  )
  q, stats = relaxed_flows(params, omega, base, r, rho, gains, mask, options)
  assert q[0, 2, 0] == pytest.approx(6.0)
  assert q[0, 1, 0] == pytest.approx(0.0)
  assert stats["violation"] == pytest.approx(0.0)


def test_relaxed_flows_split_when_one_receiver_cannot_hold_everything():
  # Two equally good receivers, each with memory for 4 req/s of extra load
  # (ram 2, demand 1, umax 1 -> 2 MB per req/s -> 8 MB = 4 req/s).
  Nn = 3
  data = _data(Nn=Nn, memory=8, ram=2)
  neighborhood = _full(Nn)
  options = _opts(iterations=400, kappa=20.0)
  params, omega, base, r, rho, gains, mask, options = _setup(
    data, [8, 0, 0], neighborhood, rho=np.array([8.0, 8.0, 8.0]), options=options
  )
  q, stats = relaxed_flows(params, omega, base, r, rho, gains, mask, options)
  assert q[0, 1, 0] + q[0, 2, 0] == pytest.approx(8.0, abs=0.5)
  assert abs(q[0, 1, 0] - q[0, 2, 0]) < 1.0
  assert stats["violation"] < 1.0
  assert q.sum() <= omega.sum() + 1e-9


def test_relaxed_flows_prefer_cloud_when_no_receiver_is_profitable():
  Nn = 3
  beta = [[0.0, -2.0, -2.0], [0.0] * 3, [0.0] * 3]
  data = _data(Nn=Nn, beta=beta)
  params, omega, base, r, rho, gains, mask, options = _setup(
    data, [5, 0, 0], _full(Nn)
  )
  q, _ = relaxed_flows(params, omega, base, r, rho, gains, mask, options)
  assert q.sum() == 0.0


def test_existing_replica_capacity_is_free_of_memory_price():
  Nn = 2
  data = _data(Nn=Nn, memory=2)
  params, omega, base, r, rho, gains, mask, options = _setup(
    data, [5, 0], _full(Nn), r=np.array([[0.0], [6.0]]), rho=np.array([0.0, 0.0])
  )
  q, stats = relaxed_flows(params, omega, base, r, rho, gains, mask, options)
  assert q[0, 1, 0] == pytest.approx(5.0)
  assert stats["violation"] == pytest.approx(0.0)


def test_round_flows_preserve_total_up_to_one_unit_and_expectation():
  q = np.zeros((3, 3, 2))
  q[0, 1, 0] = 2.4
  q[0, 2, 0] = 1.35
  q[1, 2, 1] = 0.5
  rng = np.random.default_rng(0)
  samples = np.stack([round_flows(q, rng) for _ in range(4000)])
  assert np.allclose(samples, np.round(samples))
  totals = samples.sum(axis=2)  # per sender and function
  assert np.all(np.abs(totals[:, 0, 0] - 3.75) < 1.0)
  assert samples.mean(axis=0) == pytest.approx(q, abs=0.05)


def test_round_flows_largest_remainder_and_continuous_modes():
  q = np.zeros((3, 3, 1))
  q[0, 1, 0], q[0, 2, 0] = 1.4, 1.3  # total 2.7 -> 3 units, extra to node 1
  rng = np.random.default_rng(0)
  y = round_flows(q, rng, mode="largest_remainder")
  assert y[0, 1, 0] == 2.0 and y[0, 2, 0] == 1.0
  assert np.array_equal(round_flows(q, rng, integer=False), q)
  with pytest.raises(ValueError):
    round_flows(q, rng, mode="bad")


def test_repair_memory_makes_flows_feasible_and_respects_protected():
  Nn = 3
  data = _data(Nn=Nn, memory=6, ram=2)  # 3 replicas -> 3 req/s per receiver
  params = fw_parameters(data)
  y = np.zeros((Nn, Nn, 1))
  y[0, 2, 0] = 2.0   # protected earlier flow, already on 2 replicas at node 2
  y[1, 2, 0] = 4.0   # new, would need 6 replicas in total
  gains = np.zeros((Nn, Nn, 1))
  gains[0, 2, 0], gains[1, 2, 0] = 1.0, 0.5
  protected = np.zeros_like(y)
  protected[0, 2, 0] = 2.0
  x = np.zeros((Nn, 1))
  r = np.zeros((Nn, 1))
  r[2, 0] = 2.0
  rho = np.array([6.0, 6.0, 2.0])
  repaired, removed = repair_memory(y, params, gains, x, r, rho, protected=protected)
  assert repaired[0, 2, 0] == 2.0
  assert repaired[1, 2, 0] == 1.0
  assert removed == 3
  assert check_ls_pr_feasibility_from_fixed_y(data, repaired) == []


def test_repair_memory_never_displaces_local_load():
  # node 1 serves 3 req/s locally on 3 replicas and has no free memory: the
  # old total-memory check accepted 2 incoming req/s (4 MB <= 6 MB), forcing
  # LSPr to evict local load; the slack check must reject them.
  Nn = 2
  data = _data(Nn=Nn, memory=6, ram=2)
  params = fw_parameters(data)
  y = np.zeros((Nn, Nn, 1))
  y[0, 1, 0] = 2.0
  gains = np.ones((Nn, Nn, 1))
  x = np.array([[0.0], [3.0]])
  r = np.array([[0.0], [3.0]])
  rho = np.array([6.0, 0.0])
  assert check_ls_pr_feasibility_from_fixed_y(data, y) == []  # the lax check passes
  repaired, removed = repair_memory(y, params, gains, x, r, rho)
  assert repaired.sum() == 0.0 and removed == 2


def test_repair_memory_prefers_cheap_memory_per_unit():
  # receiver 2 must free 4 MB: one unit of f1 (4 MB, gain 0.6) is cheaper per
  # MB than four units of f0 (1 MB each, gain 0.5).
  Nn = 3
  data = _data(Nn=Nn, Nf=2, memory=100, ram=1)
  data[None]["memory_requirement"][2] = 4
  params = fw_parameters(data)
  y = np.zeros((Nn, Nn, 2))
  y[0, 2, 0], y[1, 2, 1] = 4.0, 1.0
  gains = np.zeros((Nn, Nn, 2))
  gains[0, 2, 0], gains[1, 2, 1] = 0.5, 0.6
  x, r = np.zeros((Nn, 2)), np.zeros((Nn, 2))
  rho = np.array([100.0, 100.0, 4.0])  # 8 MB needed, 4 free
  repaired, removed = repair_memory(y, params, gains, x, r, rho)
  assert repaired[0, 2, 0] == 4.0 and repaired[1, 2, 1] == 0.0 and removed == 1


def test_round_flows_never_exceed_residual_demand():
  q = np.zeros((3, 3, 1))
  q[0, 1, 0] = q[0, 2, 0] = 0.6  # 1.2 fractional, but only 1 unit of demand
  limit = np.array([[1.0], [0.0], [0.0]])
  rng = np.random.default_rng(1)
  for mode in ("random", "largest_remainder"):
    for _ in range(200):
      assert round_flows(q, rng, mode, limit=limit)[0, :, 0].sum() <= 1.0


def test_smoothing_is_an_upper_bound_of_the_hinge():
  from decentralized_frankwolfe import _excess_and_slope
  u = np.linspace(-5, 5, 2001)
  for delta in (0.5, 1.0, 2.0):
    excess, slope = _excess_and_slope(u, delta)
    assert (excess >= np.maximum(0.0, u) - 1e-12).all()
    assert np.allclose(excess[np.abs(u) >= delta], np.maximum(0.0, u[np.abs(u) >= delta]))
    assert excess.max() - np.maximum(0.0, u).max() < 1e-9
    assert ((slope >= 0) & (slope <= 1)).all()


def test_frankwolfe_assignments_are_integer_feasible_and_skip_senders():
  Nn = 5
  data = _data(Nn=Nn, memory=8, ram=2)
  neighborhood = _full(Nn)
  omega = np.array([[9.0], [6.0], [0.0], [0.0], [0.0]])
  options = _opts(iterations=200, kappa=10.0)
  rng = np.random.default_rng(3)
  y, stats = frankwolfe_assignments(
    omega, np.zeros((Nn, Nn, 1)), np.zeros((Nn, 1)), np.zeros((Nn, 1)),
    np.full(Nn, 8.0), data, neighborhood, np.zeros((Nn, Nn)), np.zeros((Nn, 1)),
    options, rng,
  )
  assert np.allclose(y, np.round(y))
  assert check_ls_pr_feasibility_from_fixed_y(data, y) == []
  # senders (0 and 1) never host the function they offload
  assert y[:, 0, 0].sum() == 0 and y[:, 1, 0].sum() == 0
  assert (y.sum(axis=1)[:, 0] <= omega[:, 0] + 1e-9).all()
  assert stats["accepted"] == pytest.approx(y.sum())
  assert y.sum() > 0
  # receivers start empty with 8 MB free: at most 4 req/s each
  assert (y.sum(axis=(0, 2)) <= 4.0 + 1e-9).all()
