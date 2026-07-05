import numpy as np
import pytest

from plasma.baselines.milp_baseline import routing_lp, shed_overflow
from plasma.baselines import madea_iface
from plasma.eval.regret import adaptation_lag, cumulative_regret


def test_routing_lp_prefers_local_when_capacity_allows():
  lam = np.array([[10.0]])
  r = np.array([[5]])
  u_max = np.array([[4.0]])
  alpha = np.array([[2.0]])
  beta = np.zeros((1, 1, 1))
  gamma = np.array([[1.0]])
  adjacency = np.zeros((1, 1))
  obj, x, y, z = routing_lp(lam, r, u_max, alpha, beta, gamma, adjacency)
  assert x[0, 0] == pytest.approx(10.0)
  assert z[0, 0] == pytest.approx(0.0)
  assert obj == pytest.approx(2.0)  # alpha * x / lam


def test_routing_lp_offloads_overflow_to_neighbor():
  lam = np.array([[10.0], [0.0]])
  r = np.array([[1], [5]])
  u_max = np.array([[4.0], [4.0]])
  alpha = np.full((2, 1), 2.0)
  beta = np.zeros((2, 2, 1)); beta[0, 1, 0] = 1.5
  gamma = np.full((2, 1), 5.0)
  adjacency = np.array([[0, 1], [1, 0]])
  obj, x, y, z = routing_lp(lam, r, u_max, alpha, beta, gamma, adjacency)
  assert x[0, 0] == pytest.approx(4.0)
  assert y[0, 1, 0] == pytest.approx(6.0)
  assert z[0, 0] == pytest.approx(0.0)


def test_routing_lp_respects_receiver_capacity():
  lam = np.array([[10.0], [0.0]])
  r = np.array([[0], [1]])
  u_max = np.array([[4.0], [4.0]])
  alpha = np.full((2, 1), 2.0)
  beta = np.zeros((2, 2, 1)); beta[0, 1, 0] = 1.5
  gamma = np.full((2, 1), 0.1)
  adjacency = np.array([[0, 1], [1, 0]])
  obj, x, y, z = routing_lp(lam, r, u_max, alpha, beta, gamma, adjacency)
  assert y[0, 1, 0] == pytest.approx(4.0)
  assert z[0, 0] == pytest.approx(6.0)


def test_shed_overflow_sheds_x_then_phantom_y():
  # held plan: node 0 handles x=30 locally and forwards y=20 to node 1;
  # true load collapses to lam=10 -- x must be shed to 0 first, then the
  # remaining 10 units of overflow scaled out of the outgoing y row so no
  # phantom (never-arrived) traffic is credited to the objective.
  x = np.array([[30.0], [0.0]])
  y = np.zeros((2, 2, 1))
  y[0, 1, 0] = 20.0
  lam = np.array([[10.0], [0.0]])
  x_eff, y_eff, z_eff = shed_overflow(x, y, lam)
  assert x_eff[0, 0] == pytest.approx(0.0)
  assert y_eff[0, 1, 0] == pytest.approx(10.0)
  assert z_eff[0, 0] == pytest.approx(0.0)
  handled = x_eff + y_eff.sum(axis=1) + z_eff
  assert handled == pytest.approx(lam)


def test_madea_iface_reexports_runner():
  import run_faasmadea
  assert madea_iface.run_madea is run_faasmadea.run


def test_greedy_baseline_conserves_traffic():
  from plasma.baselines.greedy_baseline import solve
  data = _tiny_instance()
  x, y, z, r = solve(data, {})
  Nn = data[None]["Nn"][None]
  Nf = data[None]["Nf"][None]
  for n in range(Nn):
    for f in range(Nf):
      load = data[None]["incoming_load"][(n + 1, f + 1)]
      assert x[n, f] + y[n, :, f].sum() + z[n, f] == pytest.approx(load)


def _tiny_instance():
  Nn, Nf = 2, 1
  return {None: {
    "Nn": {None: Nn}, "Nf": {None: Nf},
    "neighborhood": {(1, 1): 0, (1, 2): 1, (2, 1): 1, (2, 2): 0},
    "alpha": {(1, 1): 2.0, (2, 1): 2.0},
    "beta": {(1, 1, 1): 0.0, (1, 2, 1): 1.5, (2, 1, 1): 1.5, (2, 2, 1): 0.0},
    "gamma": {(1, 1): 0.1, (2, 1): 0.1},
    "demand": {(1, 1): 1.0, (2, 1): 1.0},
    "max_utilization": {1: 0.7},
    "memory_capacity": {1: 4, 2: 4},
    "memory_requirement": {1: 2},
    "incoming_load": {(1, 1): 10, (2, 1): 1},
  }}


def test_cumulative_regret():
  method = np.array([1.0, 1.0, 2.0])
  oracle = np.array([2.0, 2.0, 2.0])
  assert cumulative_regret(method, oracle).tolist() == [1.0, 2.0, 2.0]


def test_adaptation_lag_measures_recovery():
  times = np.arange(6, dtype=float)
  oracle = np.full(6, 10.0)
  method = np.array([10.0, 10.0, 2.0, 5.0, 9.5, 9.8])  # change point at t=2
  lag = adaptation_lag(times, method, oracle, change_points=[2.0])
  assert lag == [2.0]  # recovered at t=4 (9.5 >= 9.0)


def test_adaptation_lag_nan_when_never_recovering():
  times = np.arange(3, dtype=float)
  lag = adaptation_lag(times, np.zeros(3), np.full(3, 10.0),
                       change_points=[0.0])
  assert np.isnan(lag[0])


def _scenario_config(tmp_path, Nn=3, max_steps=6):
  return {
    "base_solution_folder": str(tmp_path),
    "verbose": 0,
    "seed": 7,
    "max_steps": max_steps,
    "min_run_time": 0,
    "max_run_time": max_steps,
    "run_time_step": 1,
    "checkpoint_interval": 1,
    "solver_name": "none",
    "solver_options": {
      "plasma": {"rounds_per_step": 5, "k_sb": 2, "n_sb_steps": 100,
                 "n_hyst": 1}
    },
    "limits": {
      "Nn": {"min": Nn, "max": Nn},
      "Nf": {"min": 2, "max": 2},
      "neighborhood": {"m": Nn - 1},
      "demand": {"values": [1.0, 1.2]},
      "memory_capacity": {"min": 12, "max": 12},
      "memory_requirement": {"values": [2, 3]},
      "max_utilization": {"min": 0.65, "max": 0.75},
      "load": {"trace_type": "sinusoidal",
               "min": {"min": 5, "max": 10},
               "max": {"min": 20, "max": 30}},
      "weights": {"alpha": {"min": 1.0, "max": 1.5},
                  "beta_multiplier": {"min": 1.5, "max": 2.5},
                  "gamma": {"min": 0.05, "max": 0.15},
                  "delta_multiplier": {"min": 0.1, "max": 0.2}},
    },
  }


def _solver_available(name="gurobi"):
  try:
    from pyomo.environ import SolverFactory
    return SolverFactory(name).available(exception_flag=False)
  except Exception:
    return False


def test_sweep_driver_runs_grid(tmp_path):
  from plasma.eval.sweep import sweep
  config = _scenario_config(tmp_path)
  df = sweep(config, {"mu": [0.1, 0.3]})
  assert len(df) == 2
  assert set(df["mu"]) == {0.1, 0.3}
  assert df["obj_mean"].notna().all()


@pytest.mark.skipif(not _solver_available(), reason="no MILP solver")
def test_scenario_driver_produces_comparison(tmp_path):
  from plasma.eval.scenario import run_scenario
  config = _scenario_config(tmp_path)
  df = run_scenario(config, kill=(1, 2, 4), solver_name="gurobi",
                    solver_options={"OutputFlag": 0})
  assert list(df.columns) == ["t", "plasma", "oracle", "stale_milp",
                              "greedy", "plasma_msgs_per_node_s"]
  assert len(df) == 6
  assert (df["oracle"] >= df["plasma"] - 1e-6).all()  # oracle upper-bounds
