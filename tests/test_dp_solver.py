"""Opt-in local DP must preserve the detailed Pyomo model and solver fallback."""
from copy import deepcopy
import random

import pyomo.environ as pyo
import pytest

from models.local_sp import solve_agent_problem, try_solve_local
from models.sp import LSP, LSP_detailed, LSP_pg
from solver_support import require_gurobi


def _data():
  return {None: {
    'Nn': {None: 3}, 'Nf': {None: 2}, 'whoami': {None: 2},
    'neighborhood': {(2, 1): 1, (2, 3): 1},
    'incoming_load': {(2, 1): 3, (2, 2): 2},
    'demand': {(2, 1): 1., (2, 2): 1.},
    'max_utilization': {1: 1., 2: 1.},
    'memory_requirement': {1: 1, 2: 1}, 'memory_capacity': {2: 1},
    'alpha': {(2, 1): 4., (2, 2): 1.},
    'beta': {(2, 1, 1): 1., (2, 3, 1): 2., (2, 1, 2): 5.},
    'gamma': {(2, 1): 1., (2, 2): 1.}, 'pi': {1: 0., 2: 1.},
  }}


@pytest.mark.parametrize('options', [{}, {'use_dp': False}])
def test_local_dp_requires_explicit_opt_in(options):
  # A missing external solver must remain visible unless DP was requested.
  with pytest.raises(RuntimeError, match='unavailable'):
    solve_agent_problem(LSP(), _data(), options, 'missing_solver')


@pytest.mark.parametrize('value', ['false', 'true', 0, 1, None])
def test_non_boolean_dp_option_is_rejected(value):
  with pytest.raises(ValueError, match='use_dp.*boolean'):
    solve_agent_problem(LSP(), _data(), {'use_dp': value}, 'missing_solver')


@pytest.mark.parametrize('price,obj,y,z', [
  ({1: 0., 2: 1.}, 20/3, [0, 2, 0, 0, 2, 0], [0, 0]),
  ({1: 5., 2: 9.}, -1/3, [0, 0, 0, 0, 0, 0], [2, 2]),
])
def test_detailed_dp_returns_destination_major_flows_and_positive_gain(price, obj, y, z):
  data = _data()
  data[None]['pi'] = price
  original = deepcopy(data)
  result = solve_agent_problem(LSP_detailed(), data, {'use_dp': True}, 'missing_solver')
  assert result['obj'] == pytest.approx(obj)
  assert result['x'] == [1, 0]
  assert result['y'] == y
  assert result['z'] == z
  assert 'omega' not in result
  assert result['termination_condition'] == 'optimal'
  assert data == original


@pytest.mark.parametrize('options', [{}, {'use_dp': False}])
def test_pg_proposal_does_not_bypass_dp_opt_in(options):
  import numpy as np
  from decentralized_potentialgame import propose_node_move
  with pytest.raises(RuntimeError, match='unavailable'):
    propose_node_move(1, np.zeros(2), np.zeros((3, 3, 2)), _data(),
                      LSP_pg(), 'missing_solver', options, 0)


def test_dp_fallback_strips_application_option_before_external_solver():
  if not pyo.SolverFactory('glpk').available(exception_flag=False):
    pytest.skip('GLPK required')
  data = _data()
  data[None]['demand'][2, 1] = 0.
  model = LSP_detailed()
  assert try_solve_local(model, data) is None
  actual = solve_agent_problem(model, data, {'use_dp': True, 'TimeLimit': 30}, 'glpk')
  expected = model.solve(model.generate_instance(data), {}, 'glpk')
  assert actual['termination_condition'] == 'optimal'
  assert actual['obj'] == pytest.approx(expected['obj'])


def test_selfish_reference_uses_dp_without_an_external_solver():
  from models.selfish import SelfishLoadManagementModel
  data = {None: {
    'Nn': {None: 2}, 'Nf': {None: 1},
    'neighborhood': {(1, 2): 1, (2, 1): 1},
    'incoming_load': {(1, 1): 1, (2, 1): 10},
    'demand': {(1, 1): 1., (2, 1): 1.},
    'max_utilization': {1: 1.}, 'memory_requirement': {1: 1},
    'memory_capacity': {1: 1, 2: 0},
    'alpha': {(1, 1): 1., (2, 1): 1.},
    'beta': {(1, 1, 1): 0., (1, 2, 1): .1,
             (2, 1, 1): 20., (2, 2, 1): 0.},
  }}
  model = SelfishLoadManagementModel()
  instance = model.generate_instance(data)
  runtime = model.compute_local_gain_floors(instance, {'use_dp': True}, 'missing_solver')
  assert runtime >= 0
  assert pyo.value(instance.minimum_local_gain[1]) == 1
  assert pyo.value(instance.minimum_local_gain[2]) == 0


@pytest.mark.parametrize('seed', range(100))
def test_detailed_dp_matches_gurobi_objective_and_all_constraints(seed):
  require_gurobi()
  rng = random.Random(seed)
  nn, nf = rng.choice([1, 2, 5, 10, 30]), rng.choice([1, 2, 5, 10])
  node = rng.randint(1, nn)
  v = {
    'Nn': {None: nn}, 'Nf': {None: nf}, 'whoami': {None: node},
    'incoming_load': {}, 'demand': {}, 'memory_requirement': {},
    'memory_capacity': {node: rng.randint(0, 20)}, 'alpha': {},
    'beta': {}, 'gamma': {}, 'pi': {}, 'max_utilization': {},
    'neighborhood': {(node, j): 1 for j in range(1, nn + 1)
                      if j != node and rng.random() < .35},
  }
  for f in range(1, nf + 1):
    v['incoming_load'][node, f] = rng.randint(0, 15)
    v['demand'][node, f] = rng.choice([.2, .5, 1., 1.5, 3.])
    v['max_utilization'][f] = rng.choice([.8, 1., 1.5])
    v['memory_requirement'][f] = rng.randint(1, 5)
    v['alpha'][node, f] = rng.randint(0, 8)
    v['gamma'][node, f] = rng.randint(0, 3)
    v['pi'][f] = rng.randint(0, 12)
    for j in range(1, nn + 1):
      if rng.random() < .8:
        v['beta'][node, j, f] = rng.randint(0, 8)
  model = LSP_detailed()
  instance = model.generate_instance({None: v})
  oracle = model.solve(instance, {'OutputFlag': 0, 'MIPGap': 0, 'MIPGapAbs': 0}, 'gurobi')
  fast = solve_agent_problem(model, {None: v}, {'use_dp': True}, 'missing_solver')
  assert fast['obj'] == pytest.approx(oracle['obj'], abs=1e-8)
  for key in ('x', 'y', 'r', 'z'):
    variables = getattr(instance, key)
    assert len(fast[key]) == len(variables)
    for index, value in zip(variables, fast[key]):
      assert value in variables[index].domain
      variables[index].set_value(value)
  for constraint in instance.component_data_objects(pyo.Constraint, active=True):
    body = pyo.value(constraint.body)
    assert constraint.lower is None or body >= pyo.value(constraint.lower) - 1e-8
    assert constraint.upper is None or body <= pyo.value(constraint.upper) + 1e-8
  for j in instance.N:
    for f in instance.F:
      assert pyo.value(instance.y[j, f]) <= (
        v['incoming_load'][node, f] * v['neighborhood'].get((node, j), 0)
      )
  assert pyo.value(instance.OBJ) == pytest.approx(oracle['obj'], abs=1e-8)
