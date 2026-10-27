"""The ablation fixes initial local service inside both proposal backends."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest

from models import sp
from models.dp_solver import try_solve_local
from decentralized_potentialgame import propose_node_move, run_pg_r
from test_potentialgame_model import _tiny_pg_data
from test_potentialgame_e2e import _e2e_config


def test_fixed_x_proposal_optimizes_remaining_variables():
  data = _tiny_pg_data()
  data[None]['x_bar'] = {(1, 1): 1, (2, 1): 0}
  before = deepcopy(data)
  model = sp.LSP_pg_fixedx()
  fast = try_solve_local(model, data)
  assert fast['x'] == [1]
  assert fast['omega'] == [1]
  assert fast['z'] == [2]
  assert fast['r'] == [5]  # must still serve 2.4 inbound requests
  assert try_solve_local(sp.LSP_pg(), data)['x'] == [4]
  instance = model.generate_instance(data)
  pyo.SolverFactory('highs').solve(instance)
  assert pyo.value(instance.x[1]) == 1
  assert pyo.value(instance.omega[1]) == 1
  assert data == before
  y = np.zeros((2, 2, 1))
  y[1, 0, 0] = 2.4
  x, r, omega, _ = propose_node_move(
    0, np.array([1.]), y, data, model, 'missing_solver', {'use_dp': True}, 0)
  np.testing.assert_array_equal(x, fast['x'])
  np.testing.assert_array_equal(r, fast['r'])
  np.testing.assert_array_equal(omega, fast['omega'])


@pytest.mark.parametrize('fixed_x', [False, True])
def test_detailed_run_records_initial_state_and_sweep_diagnostics(tmp_path, fixed_x):
  config = _e2e_config(tmp_path)
  config.update(max_steps=1, min_run_time=0, max_run_time=1, pg_diagnostics=True,
                communication_metrics=True)
  config['solver_name'] = 'missing_solver'
  config['solver_options']['general'] = {'use_dp': True, 'algorithm_time_limit': 60}
  folder = Path(run_pg_r(config, 0, disable_plotting=True, detailed_init=True, fixed_x=fixed_x))
  trace = pd.read_csv(folder / 'pg_diagnostics.csv')
  assert trace.iloc[0]['sweep'] == 0
  assert trace.iloc[0]['x_changed_entries'] == 0
  assert len(trace) >= 2
  assert np.isfinite(trace['welfare']).all()
  assert (trace['welfare'].diff().dropna() >= -1e-9).all()
  assert (trace['algorithm_s'].diff().dropna() >= 0).all()
  assert pd.notna(trace.iloc[-1]['termination'])
  assert (folder / 'pg_initial_state.npz').is_file()
  assert (folder / 'communication.json').is_file()
  if fixed_x:
    assert (trace['x_changed_entries'] == 0).all()
  assert pd.read_csv(folder / 'obj.csv').iloc[0, 0] == pytest.approx(trace.iloc[-1]['welfare'])
