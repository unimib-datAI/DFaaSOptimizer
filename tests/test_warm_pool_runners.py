"""Production runs must own one pool across timesteps and close it afterwards."""
from copy import deepcopy
from pathlib import Path
import json
import multiprocessing as mp

import networkx as nx
import numpy as np
import pytest

import madea_pg
import run_faasmacro as macro
from postprocessing import load_solution
from remote_experiments.batch import Experiment
from remote_experiments.instances import materialize_instance, load_materialized_instance
from run_centralized_model import encode_solution, get_current_load, update_data
from utils.centralized import validate_centralized_solution


@pytest.mark.parametrize('runner', [madea_pg.run, madea_pg.run_hierarchical])
def test_planar_runs_reuse_one_pool_and_match_sequential_timesteps(tmp_path, monkeypatch, runner):
  config = json.loads((Path(__file__).resolve().parents[1] /
                       'experiments/madea_pg_compact/config.json').read_text())
  config.update(seed=7, max_steps=2, max_run_time=2, run_time_step=1,
                base_solution_folder=str(tmp_path / 'runs'), solver_name='glpk')
  config['solver_options']['general'] = {'TimeLimit': 30, 'MIPGap': 1e-9}
  config['limits']['Nn'] = {'min': 10, 'max': 10}
  config['limits']['Nf'] = {'min': 3, 'max': 3}
  source = tmp_path / 'instance'
  materialize_instance(Experiment('warm-pool', 'warm-pool', 'faas-madea-pg', 7, {}, {}, config), source)
  data, traces, agents, graph = load_materialized_instance(source)
  assert nx.check_planarity(graph)[0] and nx.is_connected(graph)
  config['limits'].update(instance_type='materialized', path=str(source))
  pools = []
  original_pool = mp.Pool
  def create_pool(*args, **kwargs):
    pool = original_pool(*args, **kwargs)
    pools.append(pool)
    return pool
  monkeypatch.setattr(macro.mpp, 'Pool', create_pool)
  sequential = Path(runner(deepcopy(config), 0, log_on_file=True, disable_plotting=True))
  parallel = Path(runner(deepcopy(config), 2, log_on_file=True, disable_plotting=True))
  assert len(pools) == 1
  assert all(not child.is_alive() for child in pools[0]._pool)
  serial = load_solution(str(sequential), 'LSPc')
  actual = load_solution(str(parallel), 'LSPc')
  nn, nf = data[None]['Nn'][None], data[None]['Nf'][None]
  for step in range(2):
    expected_arrays = encode_solution(nn, nf, serial[0], serial[2], serial[1], step)
    arrays = encode_solution(nn, nf, actual[0], actual[2], actual[1], step)
    current = update_data(data, {'incoming_load': get_current_load(traces, list(agents), step)})
    validate_centralized_solution(*arrays[:4], current)
    for result, expected in zip(arrays, expected_arrays):
      np.testing.assert_array_equal(result, expected)
