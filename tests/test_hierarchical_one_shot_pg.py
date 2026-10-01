"""The hierarchical one-shot incumbent must survive a local PG refinement."""
from copy import deepcopy
from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import madea_pg
from hierarchical_auction.iterative_engine import IterativeHierarchicalAuctionEngine
from hierarchical_auction.runner import run as run_hierarchy
from postprocessing import load_solution
from remote_experiments.batch import Experiment
from remote_experiments.instances import materialize_instance, load_materialized_instance
from run_centralized_model import encode_solution, get_current_load, update_data
from utils.centralized import validate_centralized_solution
from utils.faasmacro import compute_centralized_objective


def test_hierarchical_pg_preserves_incumbent_and_accepts_without_global_observer(tmp_path, monkeypatch):
  import one_shot_pg
  assert hasattr(one_shot_pg, 'run_hierarchical'), 'Missing hierarchical one-shot-PG runner'
  config = json.loads((Path(__file__).resolve().parents[1] /
                       'experiments/madea_pg_compact/config.json').read_text())
  config.update(seed=7, max_steps=2, max_run_time=2, run_time_step=1,
                verbose=0, checkpoint_interval=1, solver_name='glpk',
                max_hierarchy_depth=3, base_solution_folder=str(tmp_path / 'runs'))
  config['limits']['Nn'] = {'min': 10, 'max': 10}
  config['limits']['Nf'] = {'min': 3, 'max': 3}
  source = tmp_path / 'instance'
  materialize_instance(Experiment('hpg', 'hpg', 'hierarchical-one-shot-pg', 7, {}, {}, config), source)
  data, traces, agents, _ = load_materialized_instance(source)
  config['limits'].update(instance_type='materialized', path=str(source))
  original = deepcopy(config)
  baseline = Path(run_hierarchy(config, 0, log_on_file=True, disable_plotting=True,
                               engine_class=IterativeHierarchicalAuctionEngine))
  rounds = []
  real_higher_levels = IterativeHierarchicalAuctionEngine.run_higher_levels
  def record(self, **kwargs):
    result = real_higher_levels(self, **kwargs)
    rounds.append(result)
    return result
  monkeypatch.setattr(IterativeHierarchicalAuctionEngine, 'run_higher_levels', record)
  refined = Path(one_shot_pg.run_hierarchical(config, 0, log_on_file=True, disable_plotting=True))
  assert len(rounds) < 2 * config['max_iterations'] // 2
  disabled = deepcopy(config)
  disabled['solver_options']['madea_pg'] = {'time_limit': 0}
  unchanged = Path(one_shot_pg.run_hierarchical(disabled, 0, log_on_file=True, disable_plotting=True))
  monkeypatch.setattr(madea_pg, 'compute_centralized_objective', lambda *args: 0.)
  observed = Path(one_shot_pg.run_hierarchical(config, 0, log_on_file=True, disable_plotting=True))
  assert config == original
  assert rounds and any(result.accepted_allocations for result in rounds)
  assert not (baseline / 'refinement.csv').exists()
  assert json.loads((refined / 'config.json').read_text())['algorithm'] == 'hierarchical-one-shot-pg'
  stats = pd.read_csv(refined / 'refinement.csv')
  assert stats.time.tolist() == [0, 1]
  assert stats.moves.sum() > 0
  assert stats.hierarchy_rounds.max() < config['max_iterations'] // 2
  np.testing.assert_allclose(stats.hierarchy_pg_reserve, 2.5)
  np.testing.assert_allclose(stats.hierarchy_time_budget, 27.5)
  assert (stats.time_budget > 0).all()
  np.testing.assert_allclose(stats.welfare_before, pd.read_csv(baseline / 'obj.csv').iloc[:, 0])
  np.testing.assert_allclose(stats.welfare_after, pd.read_csv(refined / 'obj.csv')['HierarchicalOneShotPG'])
  assert (stats.welfare_after >= stats.welfare_before - 1e-6).all()
  assert (pd.read_csv(refined / 'runtime.csv')['tot'] >= stats.seconds).all()
  solutions = [load_solution(str(folder), 'LSPc') for folder in (baseline, refined, unchanged, observed)]
  for step in range(2):
    arrays = [encode_solution(10, 3, sol[0], sol[2], sol[1], step) for sol in solutions]
    current = update_data(data, {'incoming_load': get_current_load(traces, list(agents), step)})
    for values in arrays:
      validate_centralized_solution(*values[:4], current)
      for value in values[:4]:
        np.testing.assert_allclose(value, np.rint(value), atol=1e-6)
    for actual, expected in zip(arrays[2], arrays[0]):
      np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(arrays[3], arrays[1]):
      np.testing.assert_array_equal(actual, expected)
    assert compute_centralized_objective(current, *arrays[1][:3]) >= compute_centralized_objective(current, *arrays[0][:3]) - 1e-6


def test_hierarchical_one_shot_pg_can_be_selected_and_resumed(tmp_path, monkeypatch):
  import run as batch
  from remote_experiments.jobs import experiment_to_job
  monkeypatch.setattr('sys.argv', ['run.py', '--methods', 'hierarchical', 'hierarchical-one-shot-pg'])
  assert batch.parse_arguments().methods == ['hierarchical', 'hierarchical-one-shot-pg']
  experiment = Experiment('hpg', 'hpg', 'hierarchical-one-shot-pg', 7, {}, {}, {'seed': 7})
  job = experiment_to_job(experiment, tmp_path / 'configs')
  assert job.command == ('python', 'one_shot_pg.py', '-c', 'config.json', '--disable_plotting', '--variant', 'hierarchical')
  (tmp_path / 'experiments.json').write_text(json.dumps({
    'experiments_list': [[2, 123]], 'hierarchical': ['baseline'], 'hierarchical-one-shot-pg': [None],
  }))
  monkeypatch.setattr(batch, 'run_hierarchical_one_shot_pg', lambda *a, **kw: 'refined')
  monkeypatch.setattr(batch, 'results_postprocessing', lambda *a, **kw: None)
  config = {'seed': 123, 'verbose': 0, 'limits': {'Nn': {'values': [2]}}}
  batch.run(config, str(tmp_path), 1, ['hierarchical-one-shot-pg'], 'hierarchical', False, 0, False, 'Nn')
  result = json.loads((tmp_path / 'experiments.json').read_text())
  assert result['hierarchical'] == ['baseline']
  assert result['hierarchical-one-shot-pg'] == ['refined']
  def unexpected(*args, **kwargs):
    raise AssertionError('Finished hierarchical-one-shot-pg was rerun')
  monkeypatch.setattr(batch, 'run_hierarchical_one_shot_pg', unexpected)
  batch.run(config, str(tmp_path), 1, ['hierarchical-one-shot-pg'], 'hierarchical', False, 0, False, 'Nn')


def test_new_replicas_are_auctioned_before_stopping_with_disabled_pg(tmp_path):
  from one_shot_pg import run_hierarchical
  from test_review_distributed_regressions import _two_node_data, _materialized_config
  data = _two_node_data()
  data[None]['incoming_load'] = {(1, 1): 5, (2, 1): 0}
  data[None]['memory_capacity'][1] = 0
  config = _materialized_config(tmp_path, data)
  config['max_hierarchy_depth'] = 1
  config['solver_options'] = {'madea_pg': {'time_limit': 0}}
  folder = Path(run_hierarchical(config, 0, log_on_file=True, disable_plotting=True))
  sol, replicas, detailed, *_ = load_solution(str(folder), 'LSPc')
  x, y, z, r, _ = encode_solution(2, 1, sol, detailed, replicas, 0)
  validate_centralized_solution(x, y, z, r, data)
  # Buyer 1 cannot serve locally. Seller 2 needs one memory round first.
  np.testing.assert_array_equal(x, [[0], [0]])
  np.testing.assert_array_equal(y[:, :, 0], [[0, 5], [0, 0]])
  np.testing.assert_array_equal(z, [[0], [0]])
  np.testing.assert_array_equal(r, [[0], [1]])


@pytest.mark.parametrize('mode', ['stalled', 'changing_prices', 'budget'])
def test_hierarchy_stops_stalled_rounds_but_preserves_price_progress_and_pg_time(tmp_path, monkeypatch, mode):
  import one_shot_pg
  import hierarchical_auction.runner as runner
  config = json.loads((Path(__file__).resolve().parents[1] /
                       'experiments/madea_pg_compact/config.json').read_text())
  config.update(seed=7, max_steps=1, max_run_time=1, run_time_step=1,
                checkpoint_interval=1, solver_name='glpk', max_iterations=10,
                base_solution_folder=str(tmp_path / 'runs'))
  config['limits']['Nn'] = {'min': 40, 'max': 40}
  config['limits']['Nf'] = {'min': 5, 'max': 5}
  source = tmp_path / 'instance'
  materialize_instance(Experiment('hpg', 'hpg', 'hierarchical-one-shot-pg', 7, {}, {}, config), source)
  data, traces, agents, _ = load_materialized_instance(source)
  config['limits'].update(instance_type='materialized', path=str(source))
  calls = []
  clock = [0.]
  actual = IterativeHierarchicalAuctionEngine.run_higher_levels
  def record(self, **kwargs):
    result = actual(self, **kwargs)
    calls.append(result)
    if mode == 'changing_prices':
      kwargs['node_prices'][:] += 0.05
    if mode == 'budget' and len(calls) == 2:
      clock[0] = 20.
    return result
  monkeypatch.setattr(IterativeHierarchicalAuctionEngine, 'run_higher_levels', record)
  if mode == 'budget':
    # Only the orchestration clock is simulated; local solves and PG remain real.
    monkeypatch.setattr(runner, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
  folder = Path(one_shot_pg.run_hierarchical(config, 0, log_on_file=True, disable_plotting=True))
  stats = pd.read_csv(folder / 'refinement.csv').iloc[0]
  termination = (folder / 'termination_condition.csv').read_text()
  if mode == 'stalled':
    assert len(calls) < config['max_iterations']
    assert 'assignments, replicas and prices stalled' in termination
  elif mode == 'changing_prices':
    assert len(calls) == config['max_iterations']
  else:
    assert len(calls) == 2
    assert 'time limit' in termination
  assert stats.time_budget == 10.
  assert stats.hierarchy_rounds == len(calls)
  assert stats.moves > 0
  sol, replicas, detailed, *_ = load_solution(str(folder), 'LSPc')
  arrays = encode_solution(40, 5, sol, detailed, replicas, 0)
  current = update_data(data, {'incoming_load': get_current_load(traces, list(agents), 0)})
  validate_centralized_solution(*arrays[:4], current)
