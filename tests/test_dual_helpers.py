import numpy as np
import pandas as pd
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


def test_pair_scores_masks_non_neighbors_and_dominated_pairs():
  Nn, Nf = 3, 1
  beta = [[[1.0] for _ in range(Nn)] for _ in range(Nn)]
  beta[0][2][0] = -1.0
  data = make_data(Nn, Nf, beta=beta)
  neighborhood = full_neighborhood(Nn)
  neighborhood[0, 1] = 0
  s, elig = pair_scores(
    data, neighborhood, np.zeros((Nn, Nn)), np.zeros((Nn, Nf)), DUAL_OPTIONS
  )
  assert not elig[0, 1, 0] and not elig[0, 2, 0] and not elig[0, 0, 0]
  assert elig[1, 0, 0] and s[1, 0, 0] == 1.0
  assert s[0, 1, 0] == -np.inf


def test_buyer_response_zero_prices_picks_best_score_seller():
  Nn, Nf = 3, 1
  beta = [[[1.0] for _ in range(Nn)] for _ in range(Nn)]
  beta[0][1][0] = 2.0
  data = make_data(Nn, Nf, beta=beta)
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
  assert dual_term == 5.0 * 2.0
  first = bids.iloc[0]
  assert (first.j, first.d) == (1, 3.0)
  assert bids["d"].sum() == 5.0


def test_buyer_response_price_shifts_demand():
  Nn, Nf = 3, 1
  beta = [[[1.0] for _ in range(Nn)] for _ in range(Nn)]
  beta[0][1][0] = 2.0
  data = make_data(Nn, Nf, beta=beta)
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
  assert dual_term == 5.0 * 1.0


def test_buyer_response_tie_uses_lower_seller_index():
  Nn, Nf = 3, 1
  data = make_data(Nn, Nf)
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
  data = make_data(Nn, Nf)
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
    force_memory_bids=False, ell=np.zeros((Nn, Nf)),
    r=np.zeros((Nn, Nf)),
  )


def test_certificate_brackets_lp_optimum_and_gap_closes():
  data, neighborhood, omega, capacity = dual_round_setup()
  y_inc, _, _, gap_info, _ = run_round(data, neighborhood, omega, capacity)
  s, elig = pair_scores(
    data, neighborhood, np.zeros_like(neighborhood),
    np.zeros(omega.shape), DUAL_ROUND_OPTIONS,
  )
  opt = coordination_lp_optimum(omega, capacity, s, elig)
  assert gap_info["UB"] >= opt - 1e-6
  assert gap_info["LB"] <= opt + 1e-6
  assert gap_info["gap"] <= 0.05
  val = float((np.where(elig, s, 0.0) * y_inc).sum())
  assert abs(val - gap_info["LB"]) <= 1e-6


def test_best_lb_is_monotone_nondecreasing():
  data, neighborhood, omega, capacity = dual_round_setup(seed=11)
  _, _, _, gap_info, _ = run_round(data, neighborhood, omega, capacity)
  hist = gap_info["lb_history"]
  assert all(b >= a - 1e-12 for a, b in zip(hist, hist[1:]))


def test_recovered_increment_is_feasible():
  data, neighborhood, omega, capacity = dual_round_setup(seed=3)
  y_inc, _, _, _, _ = run_round(data, neighborhood, omega, capacity)
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
  _, _, _, gap_info, _ = run_round(data, full_neighborhood(Nn), omega, capacity)
  assert gap_info["lam"][2, 0] > 0.0


def test_no_demand_returns_zero_gap_and_empty_outputs():
  data, neighborhood, _, capacity = dual_round_setup()
  omega = np.zeros((4, 2))
  y_inc, add_r, memory_bids, gap_info, n_active = run_round(
    data, neighborhood, omega, capacity
  )
  assert y_inc.sum() == 0.0 and add_r.sum() == 0.0
  assert len(memory_bids) == 0 and n_active == 0
  assert gap_info["LB"] == 0.0 and gap_info["UB"] == 0.0


def test_positive_demand_rejects_zero_inner_iterations():
  data, neighborhood, omega, capacity = dual_round_setup()
  options = {**DUAL_ROUND_OPTIONS, "max_inner_iterations": 0}
  with np.testing.assert_raises_regex(ValueError, "max_inner_iterations"):
    run_round(data, neighborhood, omega, capacity, options)


def test_unknown_step_rule_is_rejected():
  data, neighborhood, omega, capacity = dual_round_setup()
  options = {**DUAL_ROUND_OPTIONS, "step_rule": "unknown"}
  with np.testing.assert_raises_regex(ValueError, "step_rule"):
    run_round(data, neighborhood, omega, capacity, options)


def test_memory_bids_keep_schema_and_respect_force_flag():
  Nn, Nf = 2, 1
  data = make_data(Nn, Nf)
  neighborhood = full_neighborhood(Nn)
  omega = np.zeros((Nn, Nf)); omega[0, 0] = 1.0
  capacity = np.zeros((Nn, Nf)); capacity[1, 0] = 1.0
  rho = np.array([0.0, 2.0])
  args = dict(
    omega=omega, residual_capacity=capacity, data=data,
    neighborhood=neighborhood, rho=rho, dual_options=DUAL_ROUND_OPTIONS,
    latency=np.zeros((Nn, Nn)), fairness=np.zeros((Nn, Nf)),
    ell=np.zeros((Nn, Nf)), r=np.zeros((Nn, Nf)),
  )
  empty = dual_coordination_round(force_memory_bids=False, **args)[2]
  forced = dual_coordination_round(force_memory_bids=True, **args)[2]
  assert list(empty.columns) == ["i", "j", "f"] and empty.empty
  assert list(forced.columns) == ["i", "j", "f"]
  assert forced.to_dict("records") == [{"i": 0, "j": 1, "f": 0}]
