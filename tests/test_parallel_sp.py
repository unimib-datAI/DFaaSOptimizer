"""Warm workers must reuse processes, never old optimization inputs."""
from copy import deepcopy
import multiprocessing as mp

import numpy as np
import pytest
import pyomo.environ as pyo

import run_faasmacro as macro
from models.sp import LSP, LSPr_x


class FallbackLSP(LSP):
  """A valid custom model outside the exact-type native backend whitelist."""


def test_accepts_agent_key_views_returned_by_instance_loaders():
  with macro.parallel_solver_session():
    agents = {0: None, 1: None, 2: None}.keys()
    result = macro.solve_subproblem(_data(), agents, LSP(), 'missing_solver', {'use_dp': True}, 2)
    np.testing.assert_array_equal(result[1], [[2], [3], [1]])


def _data():
  return {None: {
    'Nn': {None: 3}, 'Nf': {None: 1},
    'incoming_load': {(1, 1): 3, (2, 1): 4, (3, 1): 2},
    'demand': {(i, 1): 1. for i in (1, 2, 3)},
    'max_utilization': {1: 1.}, 'memory_requirement': {1: 1},
    'memory_capacity': {1: 2, 2: 3, 3: 1},
    'alpha': {(i, 1): 9. for i in (1, 2, 3)},
    'delta': {(i, 1): 0. for i in (1, 2, 3)},
    'gamma': {(i, 1): 1. for i in (1, 2, 3)},
  }}


def _children():
  return {p.pid for p in mp.active_children()}


def test_warm_pool_uses_current_inputs_across_models_and_nested_sessions():
  data = _data()
  baseline = _children()
  with macro.parallel_solver_session():
    first = macro.solve_subproblem(data, [0, 1, 2], LSP(), 'missing_solver', {'use_dp': True}, 2)
    np.testing.assert_array_equal(first[1], [[2], [3], [1]])
    workers = _children() - baseline
    assert len(workers) == 2
    changed = deepcopy(data)
    changed[None]['incoming_load'] = {(1, 1): 1, (2, 1): 2, (3, 1): 3}
    changed[None]['memory_capacity'] = {1: 2, 2: 1, 3: 3}
    prices = np.array([[0.], [2.], [0.]])
    with macro.parallel_solver_session():
      second = macro.solve_subproblem(
        changed, [0, 1, 2], LSP(), 'missing_solver', {'use_dp': True}, 2, detailed_pi=prices,
      )
    np.testing.assert_array_equal(second[1], [[1], [1], [3]])
    np.testing.assert_array_equal(second[3], [[0], [1], [0]])
    np.testing.assert_array_equal(second[4], np.zeros((3, 1)))
    assert _children() - baseline == workers
    oracle = macro.solve_subproblem(
      changed, [0, 1, 2], LSP(), 'missing_solver', {'use_dp': True}, 0, detailed_pi=prices,
    )
    for k in (1, 2, 3, 4, 5, 6, 7):
      np.testing.assert_array_equal(second[k], oracle[k])
    fixed = np.array([[0.], [1.], [2.]])
    args = (LSPr_x(), changed, [0, 1, 2], 'missing_solver', {'use_dp': True},
            np.zeros((3, 3, 1)), np.array([[0.], [0.], [1.]]))
    restricted = macro.compute_social_welfare(*args, 2, fixed)
    sequential = macro.compute_social_welfare(*args, 0, fixed)
    for actual, expected in zip(restricted[0], sequential[0]):
      np.testing.assert_array_equal(actual, expected)
    assert restricted[1:3] == sequential[1:3]
    np.testing.assert_array_equal(restricted[0][4], fixed)
    assert _children() - baseline == workers
    # A smaller batch reuses the same pool; a later timestep can use all workers.
    subset = macro.solve_subproblem(changed, [1], LSP(), 'missing_solver', {'use_dp': True}, 2)
    np.testing.assert_array_equal(subset[1][1], [1])
    assert _children() - baseline == workers
  assert _children() - baseline == set()


@pytest.mark.parametrize('requested,expected', [(-1, 2), (8, 3)])
def test_worker_count_uses_available_cpus_and_caps_processes_at_nodes(monkeypatch, requested, expected):
  monkeypatch.setattr(macro.os, 'sched_getaffinity', lambda _: {0, 1}, raising=False)
  monkeypatch.setattr(macro.mpp, 'cpu_count', lambda: 64)
  baseline = _children()
  with macro.parallel_solver_session():
    macro.solve_subproblem(_data(), [0, 1, 2], LSP(), 'missing_solver', {'use_dp': True}, requested)
    assert len(_children() - baseline) == expected
  assert _children() - baseline == set()


def test_runner_scope_closes_workers_after_worker_failure():
  baseline = _children()
  @macro.parallel_solver_run
  def run():
    data = _data()
    macro.solve_subproblem(data, [0, 1, 2], LSP(), 'missing_solver', {'use_dp': True}, 2)
    del data[None]['demand'][1, 1]
    macro.solve_subproblem(data, [0, 1, 2], LSP(), 'missing_solver', {'use_dp': True}, 2)
  with pytest.raises(ValueError, match='demand'):
    run()
  assert _children() - baseline == set()


def test_sequential_batches_do_not_start_workers():
  baseline = _children()
  with macro.parallel_solver_session():
    macro.solve_subproblem(_data(), [0, 1, 2], LSP(), 'missing_solver', {'use_dp': True}, 0)
    assert _children() - baseline == set()


def test_warm_workers_refresh_solver_and_options_for_milp_fallback():
  if not pyo.SolverFactory('glpk').available(exception_flag=False):
    pytest.skip('GLPK fallback oracle required')
  data = _data()
  baseline = _children()
  with macro.parallel_solver_session():
    macro.solve_subproblem(data, [0, 1, 2], LSP(), 'missing_solver', {'use_dp': True}, 2)
    workers = _children() - baseline
    parallel = macro.solve_subproblem(
      data, [0, 1, 2], FallbackLSP(), 'glpk', {'TimeLimit': 30}, 2,
    )
    serial = macro.solve_subproblem(
      data, [0, 1, 2], FallbackLSP(), 'glpk', {'TimeLimit': 30}, 0,
    )
    for k in (1, 2, 3, 4, 5, 6, 7):
      np.testing.assert_array_equal(parallel[k], serial[k])
    assert parallel[8] == serial[8]
    assert parallel[9] == serial[9]
    assert _children() - baseline == workers
  assert _children() - baseline == set()
