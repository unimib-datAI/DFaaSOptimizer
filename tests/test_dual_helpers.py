import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog

from decentralized_dual import (
  buyer_price_response,
  dual_coordination_round,
  pair_scores,
)


def make_data(Nn=3, Nf=1, beta=None, gamma=0.05):
  data = {None: {
    "Nn": {None: Nn}, "Nf": {None: Nf},
    "beta": {}, "gamma": {},
    "memory_requirement": {f + 1: 2 for f in range(Nf)},
    "demand": {(j + 1, f + 1): 1.0 for j in range(Nn) for f in range(Nf)},
    "max_utilization": {f + 1: 0.7 for f in range(Nf)},
  }}
  for i in range(Nn):
    for f in range(Nf):
      data[None]["gamma"][(i + 1, f + 1)] = gamma
      for j in range(Nn):
        b = 1.0 if beta is None else beta[i][j][f]
        data[None]["beta"][(i + 1, j + 1, f + 1)] = b
  return data


def full_neighborhood(Nn):
  return np.ones((Nn, Nn)) - np.eye(Nn)


DUAL_OPTIONS = {"latency_weight": 0.0, "fairness_weight": 0.0}


def test_pair_scores_masks_non_neighbors_and_nonpositive_cloud_advantage():
  Nn, Nf = 3, 1
  beta = [[[1.0] for _ in range(Nn)] for _ in range(Nn)]
  beta[0][2][0] = -0.2
  data = make_data(Nn, Nf, beta=beta, gamma=0.05)
  neighborhood = full_neighborhood(Nn)
  neighborhood[0, 1] = 0
  s, elig = pair_scores(
    data, neighborhood, np.zeros((Nn, Nn)), np.zeros((Nn, Nf)), DUAL_OPTIONS
  )
  assert not elig[0, 1, 0] and not elig[0, 2, 0] and not elig[0, 0, 0]
  assert elig[1, 0, 0] and s[1, 0, 0] == 1.05
  assert s[0, 1, 0] == -np.inf
  assert s[0, 2, 0] == -np.inf


def test_buyer_response_zero_prices_picks_best_score_seller():
  Nn, Nf = 3, 1
  beta = [[[1.0] for _ in range(Nn)] for _ in range(Nn)]
  beta[0][1][0] = 2.0
  data = make_data(Nn, Nf, beta=beta, gamma=0.2)
  s, elig = pair_scores(
    data, full_neighborhood(Nn), np.zeros((Nn, Nn)), np.zeros((Nn, Nf)),
    DUAL_OPTIONS,
  )
  omega = np.zeros((Nn, Nf)); omega[0, 0] = 5.0
  capacity = np.full((Nn, Nf), 3.0)
  bids, demand, dual_term = buyer_price_response(
    omega, capacity, np.zeros((Nn, Nf)), s, elig
  )
  assert demand[1, 0] == 5.0 and demand[2, 0] == 0.0
  assert dual_term == 5.0 * 2.2
  first = bids.iloc[0]
  assert (first.j, first.d, first.utility) == (1, 3.0, 2.2)
  assert bids["d"].sum() == 5.0


def test_buyer_response_price_shifts_demand():
  Nn, Nf = 3, 1
  beta = [[[1.0] for _ in range(Nn)] for _ in range(Nn)]
  beta[0][1][0] = 2.0
  data = make_data(Nn, Nf, beta=beta, gamma=0.2)
  s, elig = pair_scores(
    data, full_neighborhood(Nn), np.zeros((Nn, Nn)), np.zeros((Nn, Nf)),
    DUAL_OPTIONS,
  )
  omega = np.zeros((Nn, Nf)); omega[0, 0] = 5.0
  lam = np.zeros((Nn, Nf)); lam[1, 0] = 1.5
  bids, demand, dual_term = buyer_price_response(
    omega, np.full((Nn, Nf), 10.0), lam, s, elig
  )
  assert demand[2, 0] == 5.0 and demand[1, 0] == 0.0
  assert dual_term == 5.0 * 1.2


def test_buyer_response_uses_best_positive_advantage_for_buyer_term():
  Nn, Nf = 3, 1
  beta = [[[1.0] for _ in range(Nn)] for _ in range(Nn)]
  beta[0][1][0] = 3.0
  beta[0][2][0] = 2.0
  data = make_data(Nn, Nf, beta=beta, gamma=0.1)
  s, elig = pair_scores(
    data, full_neighborhood(Nn), np.zeros((Nn, Nn)), np.zeros((Nn, Nf)),
    DUAL_OPTIONS,
  )
  omega = np.zeros((Nn, Nf)); omega[0, 0] = 4.0
  capacity = np.zeros((Nn, Nf)); capacity[2, 0] = 4.0
  lam = np.zeros((Nn, Nf)); lam[1, 0] = 0.5
  bids, demand, dual_term = buyer_price_response(omega, capacity, lam, s, elig)
  assert demand[1, 0] == 4.0 and demand[2, 0] == 0.0
  assert dual_term == 4.0 * 2.6
  assert bids["j"].tolist() == [2]


def test_buyer_response_tie_uses_lower_seller_index():
  Nn, Nf = 3, 1
  data = make_data(Nn, Nf, gamma=0.2)
  s, elig = pair_scores(
    data, full_neighborhood(Nn), np.zeros((Nn, Nn)), np.zeros((Nn, Nf)),
    DUAL_OPTIONS,
  )
  omega = np.zeros((Nn, Nf)); omega[0, 0] = 5.0
  bids, demand, _ = buyer_price_response(
    omega, np.full((Nn, Nf), 3.0), np.zeros((Nn, Nf)), s, elig
  )
  assert demand[1, 0] == 5.0 and demand[2, 0] == 0.0
  assert bids["j"].tolist() == [1, 2]


def test_buyer_response_all_priced_out_yields_empty():
  Nn, Nf = 2, 1
  data = make_data(Nn, Nf, gamma=0.2)
  s, elig = pair_scores(
    data, full_neighborhood(Nn), np.zeros((Nn, Nn)), np.zeros((Nn, Nf)),
    DUAL_OPTIONS,
  )
  omega = np.zeros((Nn, Nf)); omega[0, 0] = 4.0
  lam = np.full((Nn, Nf), 10.0)
  bids, demand, dual_term = buyer_price_response(
    omega, np.full((Nn, Nf), 10.0), lam, s, elig
  )
  assert len(bids) == 0 and demand.sum() == 0.0 and dual_term == 0.0


def coordination_lp_optimum(omega, capacity, s, elig):
  """Brute-force the coordination LP with scipy (test oracle)."""
  Nn, Nf = omega.shape
  pairs = [(i, j, f) for i in range(Nn) for j in range(Nn) for f in range(Nf)
           if elig[i, j, f]]
  if not pairs:
    return 0.0
  c = [-s[i, j, f] for (i, j, f) in pairs]
  A, b = [], []
  for i in range(Nn):
    for f in range(Nf):
      A.append([1.0 if (p[0], p[2]) == (i, f) else 0.0 for p in pairs])
      b.append(float(omega[i, f]))
  for j in range(Nn):
    for f in range(Nf):
      A.append([1.0 if (p[1], p[2]) == (j, f) else 0.0 for p in pairs])
      b.append(float(capacity[j, f]))
  res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None)] * len(pairs))
  assert res.success
  return -res.fun


def dual_round_setup(Nn=4, Nf=2, seed=7):
  rng = np.random.default_rng(seed)
  beta = [[[float(rng.uniform(0.5, 2.0)) for _ in range(Nf)]
           for _ in range(Nn)] for _ in range(Nn)]
  data = make_data(Nn, Nf, beta=beta)
  neighborhood = full_neighborhood(Nn)
  omega = rng.uniform(0.0, 4.0, size=(Nn, Nf))
  capacity = rng.uniform(0.0, 3.0, size=(Nn, Nf))
  return data, neighborhood, omega, capacity


DUAL_ROUND_OPTIONS = {
  "alpha0": 0.5, "step_rule": "sqrt", "theta": 1.0,
  "max_inner_iterations": 300, "gap_tolerance": 0.01,
  "latency_weight": 0.0, "fairness_weight": 0.0,
}


def run_round(data, neighborhood, omega, capacity, options=None):
  Nn = data[None]["Nn"][None]
  Nf = data[None]["Nf"][None]
  return dual_coordination_round(
    omega, capacity, data, neighborhood,
    rho=np.zeros(Nn), dual_options=options or DUAL_ROUND_OPTIONS,
    latency=np.zeros((Nn, Nn)), fairness=np.zeros((Nn, Nf)),
  )


def test_certificate_brackets_lp_optimum_and_gap_closes():
  data, neighborhood, omega, capacity = dual_round_setup()
  y_inc, _, gap_info, _ = run_round(data, neighborhood, omega, capacity)
  s, elig = pair_scores(
    data, neighborhood, np.zeros_like(neighborhood),
    np.zeros(omega.shape), DUAL_ROUND_OPTIONS,
  )
  # the round enforces no_ping_pong: a node with residual demand for f cannot
  # also host f, so the certificate brackets the LP with those sellers removed
  # (their advertised capacity zeroed), not the unconstrained LP.
  constrained_capacity = np.where(omega > 1e-6, 0.0, capacity)
  opt = coordination_lp_optimum(omega, constrained_capacity, s, elig)
  assert gap_info["UB"] >= opt - 1e-6
  assert gap_info["LB"] <= opt + 1e-6
  assert gap_info["gap"] <= 0.05
  val = float((np.where(elig, s, 0.0) * y_inc).sum())
  assert abs(val - gap_info["LB"]) <= 1e-6


def test_best_lb_is_monotone_nondecreasing():
  data, neighborhood, omega, capacity = dual_round_setup(seed=11)
  _, _, gap_info, _ = run_round(data, neighborhood, omega, capacity)
  hist = gap_info["lb_history"]
  assert all(b >= a - 1e-12 for a, b in zip(hist, hist[1:]))


def test_recovered_increment_is_feasible():
  data, neighborhood, omega, capacity = dual_round_setup(seed=3)
  y_inc, _, _, _ = run_round(data, neighborhood, omega, capacity)
  assert (y_inc >= -1e-9).all()
  assert (y_inc.sum(axis=1) <= omega + 1e-6).all()
  assert (y_inc.sum(axis=0) <= capacity + 1e-6).all()


def test_price_rises_on_oversubscribed_seller():
  Nn, Nf = 3, 1
  beta = [[[1.0] for _ in range(Nn)] for _ in range(Nn)]
  beta[0][2][0] = 5.0
  beta[1][2][0] = 5.0
  data = make_data(Nn, Nf, beta=beta)
  omega = np.zeros((Nn, Nf)); omega[0, 0] = 4.0; omega[1, 0] = 4.0
  capacity = np.zeros((Nn, Nf)); capacity[2, 0] = 2.0; capacity[0, 0] = 6.0
  capacity[1, 0] = 6.0
  _, _, gap_info, _ = run_round(data, full_neighborhood(Nn), omega, capacity)
  assert gap_info["lam"][2, 0] > 0.0


def test_no_demand_returns_zero_gap_and_empty_outputs():
  data, neighborhood, _, capacity = dual_round_setup()
  omega = np.zeros((4, 2))
  y_inc, memory_bids, gap_info, n_active = run_round(
    data, neighborhood, omega, capacity
  )
  assert y_inc.sum() == 0.0
  assert len(memory_bids) == 0 and n_active == 0
  assert gap_info["LB"] == 0.0 and gap_info["UB"] == 0.0


def test_zero_capacity_with_demand_exits_after_first_iteration():
  data, neighborhood, omega, _ = dual_round_setup()
  capacity = np.zeros_like(omega)
  y_inc, _, gap_info, _ = run_round(data, neighborhood, omega, capacity)
  assert y_inc.sum() == 0.0
  assert gap_info["LB"] == 0.0
  assert gap_info["inner_iterations"] == 1
  # the zero assignment is certified optimal: no eligible seller has capacity
  assert gap_info["UB"] == 0.0 and gap_info["gap"] == 0.0


def test_single_inner_iteration_yields_valid_certificate():
  data, neighborhood, omega, capacity = dual_round_setup(seed=5)
  options = {**DUAL_ROUND_OPTIONS, "max_inner_iterations": 1}
  y_inc, _, gap_info, _ = run_round(
    data, neighborhood, omega, capacity, options
  )
  assert gap_info["inner_iterations"] == 1
  assert np.isfinite(gap_info["UB"])
  assert gap_info["UB"] + 1e-9 >= gap_info["LB"] >= 0.0
  assert (y_inc >= -1e-9).all()
  assert (y_inc.sum(axis=1) <= omega + 1e-6).all()
  assert (y_inc.sum(axis=0) <= capacity + 1e-6).all()


@pytest.mark.parametrize("value", [0, False, True, 1.5])
def test_positive_demand_rejects_invalid_inner_iterations(value):
  data, neighborhood, omega, capacity = dual_round_setup()
  options = {**DUAL_ROUND_OPTIONS, "max_inner_iterations": value}
  with np.testing.assert_raises_regex(ValueError, "max_inner_iterations"):
    run_round(data, neighborhood, omega, capacity, options)


def test_unknown_step_rule_is_rejected():
  data, neighborhood, omega, capacity = dual_round_setup()
  options = {**DUAL_ROUND_OPTIONS, "step_rule": "unknown"}
  with np.testing.assert_raises_regex(ValueError, "step_rule"):
    run_round(data, neighborhood, omega, capacity, options)


@pytest.mark.parametrize(
  ("overrides", "match"),
  [
    ({"alpha0": 0.0}, "alpha0"),
    ({"step_rule": "polyak", "theta": 0.0}, "theta"),
    ({"gap_tolerance": -0.1}, "gap_tolerance"),
    ({"latency_weight": np.nan}, "latency_weight"),
    ({"fairness_weight": np.inf}, "fairness_weight"),
  ],
)
def test_invalid_dual_numeric_options_are_rejected(overrides, match):
  data, neighborhood, omega, capacity = dual_round_setup()
  options = {**DUAL_ROUND_OPTIONS, **overrides}
  with pytest.raises(ValueError, match=match):
    run_round(data, neighborhood, omega, capacity, options)


def test_gap_info_keeps_best_lambda_not_last_iterate(monkeypatch):
  calls = iter([
    (
      pd.DataFrame([{"i": 0, "j": 1, "f": 0, "d": 1.0, "utility": 1.0}]),
      np.array([[1.0], [0.0]]),
      1.0,
    ),
    (
      pd.DataFrame([{"i": 0, "j": 1, "f": 0, "d": 1.0, "utility": 1.0}]),
      np.array([[1.0], [0.0]]),
      10.0,
    ),
  ])

  def fake_buyer_price_response(*_args, **_kwargs):
    return next(calls)

  def fake_evaluate_assignments(*_args, **_kwargs):
    return np.zeros((2, 2, 1)), np.zeros((2, 1)), None

  monkeypatch.setattr("decentralized_dual.buyer_price_response", fake_buyer_price_response)
  monkeypatch.setattr("decentralized_dual.evaluate_assignments", fake_evaluate_assignments)

  data = make_data(2, 1)
  options = {**DUAL_ROUND_OPTIONS, "max_inner_iterations": 2, "gap_tolerance": 0.0}
  _, _, gap_info, _ = dual_coordination_round(
    omega=np.array([[1.0], [0.0]]),
    residual_capacity=np.array([[0.0], [0.0]]),
    data=data,
    neighborhood=full_neighborhood(2),
    rho=np.zeros(2),
    dual_options=options,
    latency=np.zeros((2, 2)),
    fairness=np.zeros((2, 1)),
  )

  np.testing.assert_array_equal(gap_info["best_lam"], np.zeros((2, 1)))
  assert gap_info["lam"][0, 0] > 0.0


def test_round_keeps_best_assignment_without_post_loop_reevaluation(monkeypatch):
  calls = iter([
    (
      pd.DataFrame([{"i": 0, "j": 1, "f": 0, "d": 1.0, "utility": 1.0}]),
      np.array([[1.0], [0.0]]),
      1.0,
    ),
    (
      pd.DataFrame([{"i": 0, "j": 1, "f": 0, "d": 1.0, "utility": 1.0}]),
      np.array([[1.0], [0.0]]),
      10.0,
    ),
  ])
  evaluations = []

  def fake_buyer_price_response(*_args, **_kwargs):
    return next(calls)

  def fake_evaluate_assignments(*args, **kwargs):
    evaluations.append((args, kwargs))
    y = np.zeros((2, 2, 1))
    y[0, 1, 0] = float(len(evaluations))
    return y, None, None

  monkeypatch.setattr("decentralized_dual.buyer_price_response", fake_buyer_price_response)
  monkeypatch.setattr("decentralized_dual.evaluate_assignments", fake_evaluate_assignments)

  _, _, gap_info, _ = dual_coordination_round(
    omega=np.array([[1.0], [0.0]]),
    residual_capacity=np.array([[0.0], [0.0]]),
    data=make_data(2, 1),
    neighborhood=full_neighborhood(2),
    rho=np.zeros(2),
    dual_options={**DUAL_ROUND_OPTIONS, "max_inner_iterations": 2, "gap_tolerance": 0.0},
    latency=np.zeros((2, 2)),
    fairness=np.zeros((2, 1)),
  )

  assert len(evaluations) == gap_info["inner_iterations"]


def test_memory_bids_are_emitted_only_for_unplaced_demand():
  Nn, Nf = 2, 1
  data = make_data(Nn, Nf)
  neighborhood = full_neighborhood(Nn)
  omega = np.zeros((Nn, Nf)); omega[0, 0] = 1.0
  capacity = np.zeros((Nn, Nf)); capacity[1, 0] = 1.0
  rho = np.array([0.0, 2.0])
  empty = dual_coordination_round(
    omega=omega,
    residual_capacity=capacity,
    data=data,
    neighborhood=neighborhood,
    rho=rho,
    dual_options=DUAL_ROUND_OPTIONS,
    latency=np.zeros((Nn, Nn)),
    fairness=np.zeros((Nn, Nf)),
  )[1]
  assert list(empty.columns) == ["i", "j", "f"] and empty.empty

  blocked = dual_coordination_round(
    omega=omega,
    residual_capacity=np.zeros((Nn, Nf)),
    data=data,
    neighborhood=neighborhood,
    rho=rho,
    dual_options=DUAL_ROUND_OPTIONS,
    latency=np.zeros((Nn, Nn)),
    fairness=np.zeros((Nn, Nf)),
  )[1]
  assert list(blocked.columns) == ["i", "j", "f"]
  assert blocked.to_dict("records") == [{"i": 0, "j": 1, "f": 0}]
