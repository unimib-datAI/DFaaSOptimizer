"""One-shot refinement must improve locally and remain independently selectable."""
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pandas as pd

import decentralized_auction
import madea_pg
from postprocessing import load_solution
from remote_experiments.batch import Experiment
from remote_experiments.instances import materialize_instance, load_materialized_instance
from run_centralized_model import get_current_load, update_data
from utils.centralized import encode_solution, validate_centralized_solution
from utils.faasmacro import compute_centralized_objective


def test_one_shot_pg_preserves_baseline_and_uses_local_acceptance(tmp_path, monkeypatch):
  from one_shot_pg import run
  config = json.loads((Path(__file__).resolve().parents[1] /
                       'experiments/madea_pg_compact/config.json').read_text())
  config.update(seed=7, max_steps=2, max_run_time=2, run_time_step=1,
                verbose=0, checkpoint_interval=1, solver_name='glpk',
                base_solution_folder=str(tmp_path / 'runs'))
  config['limits']['Nn'] = {'min': 10, 'max': 10}
  config['limits']['Nf'] = {'min': 3, 'max': 3}
  source = tmp_path / 'instance'
  materialize_instance(Experiment('pg', 'pg', 'one-shot-pg', 7, {}, {}, config), source)
  data, traces, agents, _ = load_materialized_instance(source)
  config['limits'].update(instance_type='materialized', path=str(source))
  original = deepcopy(config)
  baseline = Path(decentralized_auction.run(config, 0, log_on_file=True, disable_plotting=True))
  refined = Path(run(config, 0, log_on_file=True, disable_plotting=True))
  disabled = deepcopy(config)
  disabled['solver_options']['madea_pg'] = {'time_limit': 0}
  unchanged = Path(run(disabled, 0, log_on_file=True, disable_plotting=True))
  # The observer must never select moves, even if it reports useless values.
  monkeypatch.setattr(madea_pg, 'compute_centralized_objective', lambda *args: 0.)
  observed = Path(run(config, 0, log_on_file=True, disable_plotting=True))
  assert config == original
  assert not (baseline / 'refinement.csv').exists()
  assert json.loads((refined / 'config.json').read_text())['algorithm'] == 'one-shot-pg'
  stats = pd.read_csv(refined / 'refinement.csv')
  assert stats.time.tolist() == [0, 1]
  assert stats.moves.sum() > 0
  assert (stats.welfare_after >= stats.welfare_before - 1e-6).all()
  np.testing.assert_allclose(stats.welfare_before, pd.read_csv(baseline / 'obj.csv').iloc[:, 0])
  np.testing.assert_allclose(stats.welfare_after, pd.read_csv(refined / 'obj.csv')['One-shot-PG'])
  assert (pd.read_csv(refined / 'runtime.csv')['tot'] >= stats.seconds).all()
  solutions = [load_solution(str(folder), 'LSPc') for folder in (baseline, refined, unchanged, observed)]
  for step in range(2):
    arrays = [encode_solution(10, 3, sol[0], sol[2], sol[1], step) for sol in solutions]
    current = update_data(data, {'incoming_load': get_current_load(traces, list(agents), step)})
    for values in arrays:
      validate_centralized_solution(*values[:4], current)
    for actual, expected in zip(arrays[2], arrays[0]):
      np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(arrays[3], arrays[1]):
      np.testing.assert_array_equal(actual, expected)
    assert compute_centralized_objective(current, *arrays[1][:3]) >= compute_centralized_objective(current, *arrays[0][:3]) - 1e-6


def test_one_shot_pg_can_be_selected_and_resumed_separately(tmp_path, monkeypatch):
  import run as batch
  from remote_experiments.jobs import experiment_to_job
  monkeypatch.setattr('sys.argv', ['run.py', '--methods', 'faas-madea-1s', 'one-shot-pg'])
  assert batch.parse_arguments().methods == ['faas-madea-1s', 'one-shot-pg']
  experiment = Experiment('pg', 'pg', 'one-shot-pg', 7, {}, {}, {'seed': 7})
  job = experiment_to_job(experiment, tmp_path / 'configs')
  assert job.command == ('python', 'one_shot_pg.py', '-c', 'config.json', '--disable_plotting')
  (tmp_path / 'experiments.json').write_text(json.dumps({
    'experiments_list': [[2, 123]], 'faas-madea-1s': ['baseline'], 'one-shot-pg': [None],
  }))
  monkeypatch.setattr(batch, 'run_one_shot_pg', lambda *a, **kw: 'refined')
  monkeypatch.setattr(batch, 'results_postprocessing', lambda *a, **kw: None)
  batch.run({'seed': 123, 'verbose': 0, 'limits': {'Nn': {'values': [2]}}},
    str(tmp_path), 1, ['one-shot-pg'], 'faas-madea-1s', False, 0, False, 'Nn')
  result = json.loads((tmp_path / 'experiments.json').read_text())
  assert result['faas-madea-1s'] == ['baseline']
  assert result['one-shot-pg'] == ['refined']
  def unexpected(*args, **kwargs):
    raise AssertionError('Finished one-shot-pg was rerun')
  monkeypatch.setattr(batch, 'run_one_shot_pg', unexpected)
  batch.run({'seed': 123, 'verbose': 0, 'limits': {'Nn': {'values': [2]}}},
    str(tmp_path), 1, ['one-shot-pg'], 'faas-madea-1s', False, 0, False, 'Nn')
