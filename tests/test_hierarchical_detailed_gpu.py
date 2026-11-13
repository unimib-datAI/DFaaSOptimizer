"""Real GPU parity for the registered hierarchical detailed method.

DFAAS_TEST_CUDA=1 DFAAS_GPU_INSTANCE=/path/to/materialized/instance pytest ...
Use a catalogue instance with at least 30 nodes to exercise numeric batching.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import unittest

import methods
from experiments.harmonization.capture_baseline import config_for
from postprocessing import load_solution
from remote_experiments.instances import load_materialized_instance
from run_centralized_model import get_current_load, update_data
from utils.centralized import encode_solution, validate_centralized_solution


@unittest.skipUnless(os.environ.get('DFAAS_TEST_CUDA') == '1', 'real CUDA opt-in required')
def test_hierarchical_detailed_cpu_gpu_parity_and_coverage(tmp_path):
  source = Path(os.environ['DFAAS_GPU_INSTANCE'])
  data, traces, agents, _ = load_materialized_instance(source)
  assert len(agents) >= 30
  current = update_data(data, {'incoming_load': get_current_load(traces, list(agents), 0)})
  outputs = {}
  for device in ('cpu', 'cuda'):
    config = config_for(source, tmp_path / device, steps=1)
    config.update(max_iterations=3, max_cycles=2, max_hierarchy_depth=3, communication_metrics=True)
    config['solver_options']['general'].update(
      use_dp=True, dp_device=device, compute_device=device, algorithm_time_limit=1e5)
    folder = Path(methods.HARMONIZED['hierarchical-madea-detailed'](
      config, 0, log_on_file=True, disable_plotting=True))
    saved = json.loads((folder / 'config.json').read_text())
    assert saved['local_reference'] == 'LSP_detailed'
    solution, replicas, detailed, *_ = load_solution(str(folder), 'LSPc')
    arrays = encode_solution(len(agents), data[None]['Nf'][None], solution, detailed, replicas, 0)
    validate_centralized_solution(*arrays[:4], current)
    metrics = json.loads((folder / 'communication.json').read_text())
    assert metrics['complete'] and metrics['messages'] > 0
    outputs[device] = (arrays[:4], pd.read_csv(folder / 'obj.csv').to_numpy(),
                       pd.read_csv(folder / 'communication.csv'))
    if device == 'cuda':
      coverage = json.loads((folder / 'compute_coverage.json').read_text())
      assert not coverage['zero_coverage']
      assert any(r['backend'] == 'cuda' and r['operation'] == 'local_numeric'
                 for r in coverage['operations'])
  for cpu, gpu in zip(outputs['cpu'][0], outputs['cuda'][0]):
    np.testing.assert_allclose(cpu, gpu, rtol=1e-9, atol=1e-9)
  np.testing.assert_allclose(outputs['cpu'][1], outputs['cuda'][1], rtol=1e-9, atol=1e-9)
  pd.testing.assert_frame_equal(outputs['cpu'][2], outputs['cuda'][2])


if __name__ == "__main__":
  import sys
  output = Path(sys.argv[1])
  output.mkdir(parents=True, exist_ok=False)
  test_hierarchical_detailed_cpu_gpu_parity_and_coverage(output)
  print("PASS: CPU/GPU allocations, welfare, feasibility, communication counts and CUDA coverage")
