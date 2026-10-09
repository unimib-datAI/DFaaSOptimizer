"""Local optimization must preserve objective and constraints without a solver."""
import numpy as np
import pytest
import pyomo.environ as pyo

from models.sp import LSP, LSPr_x
from run_faasmacro import solve_subproblem, compute_social_welfare


def _data():
  return {None: {
    'Nn': {None: 1}, 'Nf': {None: 2},
    'incoming_load': {(1, 1): 3, (1, 2): 4},
    'demand': {(1, 1): 1., (1, 2): 1.},
    'max_utilization': {1: 1., 2: 1.},
    'memory_capacity': {1: 4}, 'memory_requirement': {1: 2, 2: 1},
    'alpha': {(1, 1): 9., (1, 2): 1.},
    'delta': {(1, 1): 0., (1, 2): 0.},
    'gamma': {(1, 1): 1., (1, 2): 1.},
  }}


@pytest.mark.parametrize('parallelism', [0, 2])
def test_local_ram_allocation_does_not_require_external_solver(parallelism):
  result = solve_subproblem(_data(), [0], LSP(), 'missing_solver', {'use_dp': True}, parallelism)
  np.testing.assert_array_equal(result[1], [[2, 0]])
  np.testing.assert_array_equal(result[4], [[1, 4]])
  np.testing.assert_array_equal(result[5], [[2, 0]])
  assert result[8]['tot'] == pytest.approx(-6.)
  assert result[9]['tot'] == 'optimal'


def test_fixed_traffic_computes_replicas_and_rejections_without_solver():
  result, objective, condition, _ = compute_social_welfare(
    LSPr_x(), _data(), [0], 'missing_solver', {'use_dp': True}, np.zeros((1, 1, 2)),
    np.array([[1., 2.]]), 0, np.array([[1., 0.]]),
  )
  np.testing.assert_array_equal(result[2], [[1, 2]])
  np.testing.assert_array_equal(result[4], [[1, 0]])
  assert objective == pytest.approx(-13/6)
  assert condition == 'optimal'


@pytest.mark.parametrize('model_name', [
  'LSP', 'LSP_v0', 'LSP_fixedr', 'LSP_fixedr_v0', 'LSP_capped',
  'LSP_capped_fixedr', 'LSP_pg', 'LSP_pg_fixedr',
  'LSPr', 'LSPr_v0', 'LSPr_x', 'LSPr_fixedr',
])
def test_exact_backend_matches_model_objective_and_constraints(model_name):
  from models import sp
  from models.local_sp import try_solve_local
  solver = pyo.SolverFactory('glpk')
  if not solver.available(exception_flag=False):
    pytest.skip('GLPK oracle required')
  rng = np.random.default_rng(123)
  for _ in range(15):
    data = _data()
    v = data[None]
    v['whoami'] = {None: 1}
    v['memory_capacity'][1] = int(rng.integers(0, 12))
    v['y_bar'] = {(1, 1, f): float(rng.choice([0, .4, 1, 2])) for f in (1, 2)}
    v['pi'] = {f: float(rng.integers(0, 6)) for f in (1, 2)}
    v['omega_ub'] = {f: float(rng.uniform(0, 5)) for f in (1, 2)}
    v['r_bar'] = {(1, f): int(rng.integers(0, 4)) for f in (1, 2)}
    v['omega_bar'] = {(1, f): int(rng.integers(0, v['incoming_load'][1, f] + 1)) for f in (1, 2)}
    v['x_bar'] = {(1, f): int(rng.integers(0, v['incoming_load'][1, f] - v['omega_bar'][1, f] + 1)) for f in (1, 2)}
    for f in (1, 2):
      v['incoming_load'][1, f] = int(rng.integers(0, 5)) if not model_name.startswith('LSPr') else v['incoming_load'][1, f]
      if model_name.startswith('LSPr'):
        v['incoming_load'][1, f] += .5
        v['omega_bar'][1, f] += .4
      v['demand'][1, f] = float(rng.choice([.2, .5, 1, 1.5]))
      v['max_utilization'][f] = float(rng.choice([.8, 1, 1.5]))
      v['alpha'][1, f] = float(rng.integers(0, 5))
      v['delta'][1, f] = float(rng.integers(0, 5))
      v['gamma'][1, f] = float(rng.integers(0, 5))
    model = getattr(sp, model_name)()
    instance = model.generate_instance(data)
    oracle = solver.solve(instance, load_solutions=False)
    fast = try_solve_local(model, data)
    if oracle.solver.termination_condition == pyo.TerminationCondition.infeasible:
      assert fast is None
      continue
    assert oracle.solver.termination_condition == pyo.TerminationCondition.optimal
    instance.solutions.load_from(oracle)
    assert fast is not None
    assert fast['obj'] == pytest.approx(pyo.value(instance.OBJ), abs=1e-8)
    # Independently evaluate every original constraint on the native allocation.
    for key in ('x', 'r', 'omega', 'z'):
      if key in fast:
        for f, value in enumerate(fast[key], 1):
          assert value in getattr(instance, key)[f].domain
          getattr(instance, key)[f].set_value(value)
    for constraint in instance.component_data_objects(pyo.Constraint, active=True):
      body = pyo.value(constraint.body)
      if constraint.lower is not None:
        assert body >= pyo.value(constraint.lower) - 1e-8, constraint.name
      if constraint.upper is not None:
        assert body <= pyo.value(constraint.upper) + 1e-8, constraint.name
    assert fast['obj'] == pytest.approx(pyo.value(instance.OBJ), abs=1e-8)


@pytest.mark.parametrize('change', ['zero_memory', 'zero_capacity', 'fractional_load', 'large_dp'])
def test_unsupported_parameters_keep_original_solver_path(change):
  from models.local_sp import try_solve_local, solve_agent_problem
  data = _data()
  v = data[None]
  v['whoami'] = {None: 1}
  if change == 'zero_memory':
    v['memory_requirement'][1] = 0
  elif change == 'zero_capacity':
    v['max_utilization'][1] = 0
  elif change == 'fractional_load':
    v['incoming_load'][1, 1] = 3.5
  else:
    v['memory_capacity'][1] = 1_000_000
  assert try_solve_local(LSP(), data) is None
  if change in ('zero_memory', 'zero_capacity'):
    solver = pyo.SolverFactory('glpk')
    if not solver.available(exception_flag=False):
      pytest.skip('GLPK fallback required')
    result = solve_agent_problem(LSP(), data, {}, 'glpk')
    assert result['solution_exists']
    assert result['termination_condition'] == 'optimal'


@pytest.mark.parametrize('model_name,missing', [
  ('LSPr', 'y_bar'), ('LSPr_v0', 'y_bar'), ('LSPr_x', 'y_bar'),
  ('LSPr_fixedr', 'y_bar'), ('LSPr_fixedr', 'r_bar'),
])
def test_missing_required_commitments_keep_pyomo_validation(model_name, missing):
  from models import sp
  from models.local_sp import try_solve_local
  data = _data()
  v = data[None]
  v.update(whoami={None: 1}, y_bar={(1, 1, f): 0 for f in (1, 2)},
           omega_bar={(1, f): 0 for f in (1, 2)},
           x_bar={(1, f): 0 for f in (1, 2)},
           r_bar={(1, f): 0 for f in (1, 2)})
  del v[missing]
  model = getattr(sp, model_name)()
  assert try_solve_local(model, data) is None
  with pytest.raises(ValueError, match=missing):
    model.generate_instance(data)


def test_fixed_traffic_with_fractional_load_and_offloading_uses_direct_calculation():
  data = _data()
  data[None]['incoming_load'][1, 1] = 3.5
  result, objective, condition, _ = compute_social_welfare(
    LSPr_x(), data, [0], 'missing_solver', {'use_dp': True}, np.zeros((1, 1, 2)),
    np.array([[.5, 2.]]), 0, np.array([[1., 0.]]),
  )
  np.testing.assert_array_equal(result[2], [[2., 2.]])
  np.testing.assert_array_equal(result[4], [[1, 0]])
  assert objective == pytest.approx(-1.5)
  assert condition == 'optimal'
