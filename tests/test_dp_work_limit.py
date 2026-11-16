"""Bound DP work without excluding the measured Ether IIoT cases."""
from copy import deepcopy

from models.dp_solver import _prepare_local_parameters, try_solve_local
from models.sp import LSP_detailed


def instance(load, ram):
  return {None: {
    'Nn': {None: 2}, 'Nf': {None: 1}, 'whoami': {None: 1},
    'neighborhood': {(1, 2): 1}, 'incoming_load': {(1, 1): load},
    'demand': {(1, 1): 1.}, 'max_utilization': {1: 1.},
    'memory_requirement': {1: 1}, 'memory_capacity': {1: ram},
    'alpha': {(1, 1): 10.}, 'beta': {(1, 1, 1): 0., (1, 2, 1): 1.},
    'gamma': {(1, 1): 1.},
  }}


def test_dp_solves_above_old_limit_without_changing_data():
  data = instance(1200, 2000)  # (2000+1)*(1200+2) = 2,405,202
  original = deepcopy(data)
  result = try_solve_local(LSP_detailed(), data)
  assert result is not None
  assert result['x'] == [1200] and result['r'] == [1200]
  assert result['y'] == [0, 0] and result['z'] == [0]
  assert result['obj'] == 10. and result['termination_condition'] == 'optimal'
  assert data == original


def test_new_work_limit_is_inclusive_and_still_bounds_growth():
  # 2500 * 2000 = 5,000,000, followed by 2500 * 2001 > the limit.
  assert _prepare_local_parameters(LSP_detailed(), instance(1998, 2499)) is not None
  assert _prepare_local_parameters(LSP_detailed(), instance(1999, 2499)) is None
