"""Numerical counterexamples from the 2026-10-13 shared-model audit."""
import importlib
from pathlib import Path
from time import perf_counter

import numpy as np
import pyomo.environ as pyo
import pytest

from models import model as model_module
from models.model import BaseAbstractModel
from models.selfish import SelfishLoadManagementModel
from run_faasmacro import combine_solutions


def _data():
  return {None: {
    'Nn': {None: 2}, 'Nf': {None: 1},
    'incoming_load': {(1, 1): 2, (2, 1): 1},
    'demand': {(1, 1): 1., (2, 1): 1.},
    'max_utilization': {1: 1.}, 'memory_requirement': {1: 1},
    'memory_capacity': {1: 3, 2: 4},
    'neighborhood': {(1, 2): 1, (2, 1): 1},
  }}


@pytest.mark.parametrize('fixed_replicas', [False, True])
def test_final_metrics_include_received_load_and_released_replicas(fixed_replicas):
  data = _data()
  if fixed_replicas:
    data[None]['r_bar'] = {(1, 1): 1, (2, 1): 4}
  x = np.array([[0.], [1.]])
  y = np.zeros((2, 2, 1))
  y[0, 1, 0] = 2
  replicas = np.array([[1], [4]])
  residual = np.array([2., 0.])
  result = combine_solutions(
    2, 1, data, data[None]['incoming_load'], x, replicas, residual,
    x, y, np.zeros((2, 1)), replicas, y.transpose(1, 0, 2), residual,
  )
  np.testing.assert_array_equal(result['sp']['r'], [[0], [3]])
  np.testing.assert_allclose(result['sp']['U'], [[0.], [1.]])
  np.testing.assert_array_equal(result['sp']['rho'], [3., 1.])
  np.testing.assert_allclose(result['rmp']['U'], [[0.], [.75]])


@pytest.mark.parametrize('solver_name', ['glpk', 'appsi_highs'])
@pytest.mark.parametrize('initial_value', [None, 1])
def test_infeasible_solve_never_extracts_uninitialized_or_stale_values(
    solver_name, initial_value, monkeypatch, tmp_path,
  ):
  if not pyo.SolverFactory(solver_name).available(exception_flag=False):
    pytest.skip(f'{solver_name} unavailable')
  monkeypatch.chdir(tmp_path)
  monkeypatch.setattr(model_module, '_SOLVER_CACHE', {})
  instance = pyo.ConcreteModel()
  instance.x = pyo.Var(domain=pyo.NonNegativeIntegers, initialize=initial_value)
  instance.balance = pyo.Constraint(expr=instance.x == .5)
  instance.OBJ = pyo.Objective(expr=instance.x)
  options = {'threads': 1} if solver_name == 'appsi_highs' else {}
  result = BaseAbstractModel().solve(instance, options, solver_name)
  assert result['solution_exists'] is False
  assert result['termination_condition'] == 'infeasible'
  assert 'x' not in result and 'obj' not in result


@pytest.mark.parametrize('solver_name', ['appsi_highs', 'highs'])
def test_generic_time_limit_reaches_highs_backend(monkeypatch, solver_name):
  if not pyo.SolverFactory(solver_name).available(exception_flag=False):
    pytest.skip('HiGHS unavailable')
  monkeypatch.setattr(model_module, '_SOLVER_CACHE', {})
  instance = pyo.ConcreteModel()
  instance.x = pyo.Var(bounds=(0, 1))
  instance.OBJ = pyo.Objective(expr=instance.x, sense=pyo.maximize)
  result = BaseAbstractModel().solve(instance, {'TimeLimit': 2.5, 'threads': 1}, solver_name)
  assert result['obj'] == 1
  solver = model_module._SOLVER_CACHE[solver_name]
  assert solver._solver_model.getOptionValue('time_limit')[1] == 2.5


def test_expired_local_reference_deadline_cannot_produce_floors():
  model = SelfishLoadManagementModel()
  instance = model.generate_instance(_data())
  with pytest.raises(TimeoutError):
    model.compute_local_gain_floors(
      instance, {'use_dp': True}, 'missing_solver', deadline=perf_counter() - 1,
    )


def test_local_reference_uses_remaining_shared_budget(monkeypatch):
  import models.selfish as selfish_module
  elapsed = [0.]
  limits = []
  original_solve = selfish_module.solve_agent_problem

  def solve_and_advance(model, data, options, solver):
    limits.append(options['TimeLimit'])
    result = original_solve(model, data, options, solver)
    elapsed[0] += 1.
    return result

  monkeypatch.setattr(selfish_module.step_time, 'perf_counter', lambda: elapsed[0])
  monkeypatch.setattr(selfish_module, 'solve_agent_problem', solve_and_advance)
  model = SelfishLoadManagementModel()
  instance = model.generate_instance(_data())
  model.compute_local_gain_floors(
    instance, {'use_dp': True, 'TimeLimit': 20}, 'missing_solver', deadline=10.,
  )
  assert limits == [10., 9.]
  assert [pyo.value(instance.minimum_local_gain[n]) for n in instance.N] == [1., 1.]


def test_highs_time_limit_without_incumbent_does_not_return_initial_values(monkeypatch):
  if not pyo.SolverFactory('appsi_highs').available(exception_flag=False):
    pytest.skip('HiGHS unavailable')
  monkeypatch.setattr(model_module, '_SOLVER_CACHE', {})
  instance = pyo.ConcreteModel()
  instance.x = pyo.Var(range(10), domain=pyo.Binary, initialize=0)
  instance.capacity = pyo.Constraint(expr=sum(instance.x.values()) <= 3.5)
  instance.OBJ = pyo.Objective(expr=sum(instance.x.values()), sense=pyo.maximize)
  result = BaseAbstractModel().solve(
    instance, {'TimeLimit': 0, 'presolve': 'off', 'threads': 1}, 'appsi_highs',
  )
  assert result['termination_condition'] == 'maxTimeLimit'
  assert result['solution_exists'] is False
  assert 'x' not in result and 'obj' not in result


@pytest.mark.parametrize('initial_value', [None, 0])
def test_gurobi_direct_empty_limit_result_is_not_an_incumbent(monkeypatch, initial_value):
  from solver_support import require_gurobi
  require_gurobi()
  monkeypatch.setattr(model_module, '_SOLVER_CACHE', {})
  instance = pyo.ConcreteModel()
  instance.x = pyo.Var(range(10), domain=pyo.Binary, initialize=initial_value)
  instance.capacity = pyo.Constraint(expr=sum(instance.x.values()) <= 3.5)
  instance.OBJ = pyo.Objective(expr=sum(instance.x.values()), sense=pyo.maximize)
  result = BaseAbstractModel().solve(instance, {'TimeLimit': 0}, 'gurobi_direct')
  assert result['termination_condition'] == 'maxTimeLimit'
  assert result['solution_exists'] is False
  assert 'x' not in result and 'obj' not in result


def test_final_metrics_allow_absent_master_solution():
  data = _data()
  x = np.array([[0.], [1.]])
  y = np.zeros((2, 2, 1))
  y[0, 1, 0] = 2
  result = combine_solutions(
    2, 1, data, data[None]['incoming_load'], x, np.array([[0], [3]]),
    np.array([3., 1.]), None, y, None, None, None, None,
  )
  np.testing.assert_allclose(result['sp']['U'], [[0.], [1.]])


@pytest.mark.parametrize('selfish', [False, True])
def test_catalog_exhausted_budget_returns_no_solution(monkeypatch, selfish):
  root = Path(__file__).resolve().parents[1]
  monkeypatch.syspath_prepend(str(root / 'experiments/madea_pg_compact'))
  catalog = importlib.import_module('catalog_centralized')
  fixture = root / ('tests/data/harmonization/'
                    'cilic-europe30-v1-delaunay__huawei-d1-f5-n30-t144x10m-ln1.6-0.5-s1-s1')
  result = catalog.solve(fixture, 0, selfish=selfish)
  assert result['status'] == 'no_solution'
  assert result['termination'] == 'TerminationCondition.maxTimeLimit'
  assert result['feasible'] is False
  assert result['welfare'] is None


@pytest.mark.parametrize('selfish', [False, True])
def test_catalog_positive_budget_solves_and_validates_incumbent(monkeypatch, tmp_path, selfish):
  root = Path(__file__).resolve().parents[1]
  monkeypatch.syspath_prepend(str(root / 'experiments/madea_pg_compact'))
  catalog = importlib.import_module('catalog_centralized')
  data = _data()
  data[None].update(
    alpha={(1, 1): 1., (2, 1): 1.},
    beta={(1, 2, 1): .9, (2, 1, 1): .9},
    gamma={(1, 1): .8, (2, 1): .8},
  )
  (tmp_path / 'metadata.json').write_text('{}')
  monkeypatch.setattr(catalog.families, 'load_materialized_instance', lambda _path: (
    data, {0: {0: [2], 1: [1]}}, [0, 1], None,
  ))
  result = catalog.solve(tmp_path, 5., threads=1, selfish=selfish)
  assert result['status'] == 'ok'
  assert result['feasible'] is True
  assert result['welfare'] == pytest.approx(2.)
  assert result['solver_objective'] == pytest.approx(2.)
