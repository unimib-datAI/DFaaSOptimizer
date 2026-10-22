"""Detailed initialization is opt-in; proposed offloads still need acceptance."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pyomo.environ as pyo
from models.selfish import SelfishLoadManagementModel

import methods
import run_faasmadea
import decentralized_auction
import decentralized_potentialgame
import decentralized_greedy_rr
from experiments.harmonization.capture_baseline import config_for, DATA
from postprocessing import load_solution
from remote_experiments.instances import load_materialized_instance
from run_centralized_model import get_current_load, update_data
from utils.centralized import encode_solution, validate_centralized_solution

CASES = [
  ('Greedy-RR', decentralized_greedy_rr),
  ('MADEA-detailed', run_faasmadea),
  ('MADEA-one-shot-detailed', decentralized_auction),
  ('FaaS-MAPG-R-detailed', decentralized_potentialgame),
  ('MADEA-selective', run_faasmadea),
  ('MADEA-one-shot-selective', decentralized_auction),
]


def runner(name):
  return {**methods.HARMONIZED, **methods.SEQUENTIAL_METHODS}[name]


@pytest.mark.parametrize('name,module', CASES)
@pytest.mark.parametrize('workers', [0, 2])
def test_detailed_variants_keep_feasible_artifacts_and_local_reference(name, module, workers, tmp_path, monkeypatch):
  source = next(p for p in sorted(DATA.iterdir()) if (p/'metadata.json').exists())
  config = config_for(source, tmp_path, steps=2)
  config['max_iterations'] = 3
  original = deepcopy(config)
  initial = []
  solve = module.solve_subproblem

  def record(data, agents, model, *args, **kwargs):
    result = solve(data, agents, model, *args, **kwargs)
    if model.name == 'LSP_detailed':
      initial.append(result)
    assert model.name != 'LSP'
    return result

  monkeypatch.setattr(module, 'solve_subproblem', record)
  folder = Path(runner(name)(config, workers, log_on_file=True, disable_plotting=True))
  assert config == original
  saved = json.loads((folder/'config.json').read_text())
  assert saved['algorithm'] == name and saved['local_reference'] == 'LSP_detailed'
  if name.endswith('-selective'):
    assert saved['memory_requests'] == 'selective'
  assert list(pd.read_csv(folder/'obj.csv').columns) == [name]
  assert len(initial) == 2
  solution, replicas, detailed, *_ = load_solution(str(folder), 'LSPc')
  data, traces, agents, _ = load_materialized_instance(source)
  for step, init in enumerate(initial):
    np.testing.assert_array_equal(init[4], init[2].sum(axis=1))
    assert init[4].sum() > 0
    current = update_data(data, {'incoming_load': get_current_load(traces, list(agents), step)})
    arrays = encode_solution(len(agents), data[None]['Nf'][None], solution, detailed, replicas, step)
    validate_centralized_solution(*arrays[:4], current)
    if name != 'FaaS-MAPG-R-detailed':
      np.testing.assert_array_equal(arrays[0], init[1])
      central = SelfishLoadManagementModel()
      instance = central.generate_instance(current)
      central.compute_local_gain_floors(instance, {'use_dp': True}, 'missing_solver')
      for n in instance.N:
        gain = sum(pyo.value(instance.alpha[n, f]) * arrays[0][n-1, f-1]
                   / (pyo.value(instance.incoming_load[n, f]) or 1) for f in instance.F)
        assert gain >= pyo.value(instance.minimum_local_gain[n]) - 1e-9
  runtime = pd.read_csv(folder/'runtime.csv')
  assert len(runtime) == 2 and (runtime.algorithm_s >= 0).all()


@pytest.mark.parametrize('name,module', CASES)
def test_detailed_variant_rejects_fixed_replica_reference(name, module, tmp_path):
  # No detailed fixed-r model exists: do not silently revert to LSP_fixedr.
  with pytest.raises(ValueError, match='opt_solution_folder'):
    runner(name)({'opt_solution_folder': str(tmp_path)}, 0)
