"""Regressions for the October audit: greedy commitments and sequential timing."""

from functools import partial

import numpy as np
import pytest

import decentralized_bestresponse as br
import decentralized_potentialgame as pg
from heuristic_coordinator import GreedyCoordinator
from models.sp import LSP_pg
from test_review_distributed_regressions import _two_node_data
from utils import run_frame, step_time


def _greedy_instance(loads, memory, edges, requirements=(1,)):
  loads = np.asarray(loads, dtype=float)
  nn, nf = loads.shape
  values = {
    "Nn": {None: nn}, "Nf": {None: nf},
    "incoming_load": {(i + 1, f + 1): loads[i, f] for i in range(nn) for f in range(nf)},
    "omega_bar": {(i + 1, f + 1): loads[i, f] for i in range(nn) for f in range(nf)},
    "x_bar": {(i + 1, f + 1): 0. for i in range(nn) for f in range(nf)},
    "r_bar": {(i + 1, f + 1): 0. for i in range(nn) for f in range(nf)},
    "demand": {(i + 1, f + 1): .08 for i in range(nn) for f in range(nf)},
    "max_utilization": {f + 1: .8 for f in range(nf)},
    "memory_capacity": {i + 1: m for i, m in enumerate(memory)},
    "memory_requirement": {f + 1: m for f, m in enumerate(requirements)},
    "neighborhood": {(i + 1, j + 1): int((i, j) in edges) for i in range(nn) for j in range(nn)},
    "beta": {(i + 1, j + 1, f + 1): 1. for i in range(nn) for j in range(nn) for f in range(nf)},
    "gamma": {(i + 1, f + 1): .1 for i in range(nn) for f in range(nf)},
  }
  return {None: values, "sp_rho": np.array(memory, dtype=float)}


def test_greedy_reuses_replica_capacity_across_buyers():
  data = _greedy_instance([[3], [3], [0]], [0, 0, 1], {(0, 2), (1, 2)})
  result = GreedyCoordinator().solve(data, {})
  np.testing.assert_array_equal(result["y"].reshape(3, 3)[:, 2], [3, 3, 0])
  np.testing.assert_array_equal(result["z"], [0, 0, 0])
  np.testing.assert_array_equal(result["r"], [0, 0, 1])


@pytest.mark.parametrize("edges, expected", [
  ({(0, 1), (1, 2)}, [[0, 3, 0], [0, 0, 0], [0, 0, 0]]),
  ({(0, 2), (1, 0)}, [[0, 0, 3], [0, 0, 0], [0, 0, 0]]),
])
def test_greedy_never_gives_a_function_both_forwarding_roles(edges, expected):
  data = _greedy_instance([[3], [3], [0]], [1, 1, 1], edges)
  result = GreedyCoordinator().solve(data, {})
  np.testing.assert_array_equal(result["y"].reshape(3, 3), expected)
  np.testing.assert_array_equal(result["z"], [0, 3, 0])


@pytest.mark.parametrize("large_ram, expected_large_flow", [(4, 0), (2, 10)])
def test_greedy_keeps_unspent_memory_for_another_function(large_ram, expected_large_flow):
  data = _greedy_instance([[20, 1], [0, 0]], [0, 3], {(0, 1)}, (large_ram, 1))
  result = GreedyCoordinator().solve(data, {})
  np.testing.assert_array_equal(result["y"].reshape(2, 2, 2)[0, 1], [expected_large_flow, 1])
  np.testing.assert_array_equal(result["r"].reshape(2, 2)[1], [int(large_ram == 2), 1])


def test_pg_stall_does_not_certify_nash_when_surrogate_misses_an_improvement():
  data = _two_node_data()
  data[None]["incoming_load"] = {(1, 1): 1, (2, 1): 1}
  data[None]["demand"] = {(1, 1): .5, (2, 1): .5}
  x, r, y = np.ones((2, 1)), np.ones((2, 1)), np.zeros((2, 2, 1))
  propose = partial(pg.propose_node_move, y=y, sp_data=data, model=LSP_pg(),
                    solver_name="missing_solver", general_solver_options={"use_dp": True},
                    parallelism=0)
  accepted, _, bids, _ = pg.potential_game_sweep(
    x, y, r, data, np.array([[0, 1], [1, 0]]), np.zeros(2), 1e-6, 1e-6,
    "fixed", np.random.default_rng(0), propose,
  )
  assert accepted == 0 and bids.empty
  before = pg.compute_node_utility(0, x, y, pg.compute_z(x, y, data), data)
  x[0, 0], y[0, 1, 0] = 0, 1
  after = pg.compute_node_utility(0, x, y, pg.compute_z(x, y, data), data)
  assert after - before == pytest.approx(1.)
  assert x[1, 0] + y[:, 1, 0].sum() == 2  # fits the receiver's existing replica
  assert pg.check_pg_stopping(0, 10, accepted, 0, 0., 10.) == (True, "no improving proposal")


def _sequential_context():
  data = _two_node_data()
  data[None]["incoming_load"] = {(1, 1): 30, (2, 1): 5}
  general = {"use_dp": True, "TimeLimit": 100, "algorithm_time_limit": 1}
  return run_frame.RunContext(
    config={"max_iterations": 5, "patience": 1}, parallelism=0, folder="",
    log=None, solver_name="missing_solver", solver_options={"general": general},
    general_options=general, time_limit=1., tolerance=1e-6, checkpoint_interval=1,
    steps=range(1), base_data=data, traces={}, agents=[0, 1], graph=None, Nn=2, Nf=1,
    neighborhood=np.array([[0, 1], [1, 0]]), latency=np.zeros((2, 2)),
  )


def _sequential_step(module, ctx):
  kwargs = dict(order="fixed", rng=np.random.default_rng(0))
  if module is br:
    kwargs.update(options={"latency_weight": 0., "fairness_weight": 0.}, response="greedy")
  else:
    kwargs.update(epsilon=1e-6)
  return module._solve_step(ctx, 0, ctx.base_data, **kwargs)


@pytest.mark.parametrize("module", [br, pg])
def test_sequential_methods_use_algorithm_budget_not_solver_limit(module, monkeypatch):
  elapsed = [0.]
  monkeypatch.setattr(step_time, "perf_counter", lambda: elapsed[0])
  solve = module.solve_subproblem

  def initialization(*args, **kwargs):
    result = solve(*args, **kwargs)
    elapsed[0] += 2.
    return result

  monkeypatch.setattr(module, "solve_subproblem", initialization)
  clock = step_time.start_step()
  try:
    result = _sequential_step(module, _sequential_context())
    assert "time limit" in result.termination
    assert "it: 0;" in result.termination
  finally:
    step_time.end_step(clock)


def test_best_response_excludes_feasibility_observer_from_time_budget(monkeypatch):
  elapsed = [0.]
  monkeypatch.setattr(step_time, "perf_counter", lambda: elapsed[0])
  validate = br.check_ls_pr_feasibility_from_fixed_y

  def slow_validation(*args, **kwargs):
    elapsed[0] += 5.
    return validate(*args, **kwargs)

  monkeypatch.setattr(br, "check_ls_pr_feasibility_from_fixed_y", slow_validation)
  clock = step_time.start_step()
  try:
    result = _sequential_step(br, _sequential_context())
  finally:
    timing = step_time.end_step(clock)
  assert timing["wall_s"] == 5.
  assert timing["algorithm_s"] == 0.
  assert "time limit" not in result.termination
