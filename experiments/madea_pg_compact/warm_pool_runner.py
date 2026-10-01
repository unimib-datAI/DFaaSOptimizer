"""Two-step planar MADEA-PG comparison of cold and persistent process pools."""
from pathlib import Path
from unittest.mock import patch
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import networkx as nx
import numpy as np
import madea_pg
import run_faasmacro as macro
from experiments.madea_pg_compact.warm_pool import cold_solve
from postprocessing import load_solution
from remote_experiments.instances import load_materialized_instance
from run_centralized_model import encode_solution, get_current_load, update_data
from utils.centralized import validate_centralized_solution
from utils.faasmacro import compute_centralized_objective


if __name__ == '__main__':
  output = Path('solutions/madea-pg-warm-pool-planar-2026-10-01')
  source = Path('solutions/madea-pg-compact-planar-temporal-2026-10-01/instances/n40-f5-s7')
  data, traces, agents, graph = load_materialized_instance(source)
  assert nx.check_planarity(graph)[0] and nx.is_connected(graph)
  config = json.loads((ROOT / 'experiments/madea_pg_compact/config.json').read_text())
  config.update(seed=7, max_steps=2, max_run_time=2, run_time_step=1,
                base_solution_folder=str(output / 'runs'))
  config['limits'].update(instance_type='materialized', path=str(source.resolve()))
  rows, references = [], []
  for mode in ('warm', 'cold'):
    with patch.object(macro, '_solve_agents_parallel',
                      cold_solve if mode == 'cold' else macro._solve_agents_parallel):
      started = time.perf_counter()
      folder = madea_pg.run(config, 2, log_on_file=True, disable_plotting=True)
      elapsed = time.perf_counter() - started
    solution, replicas, detailed, *_ = load_solution(folder, 'LSPc')
    welfare = []
    for step in range(2):
      arrays = encode_solution(40, 5, solution, detailed, replicas, step)
      current = update_data(data, {'incoming_load': get_current_load(traces, list(agents), step)})
      validate_centralized_solution(*arrays[:4], current)
      if mode == 'warm':
        references.append(arrays)
      else:
        for actual, expected in zip(arrays, references[step]):
          np.testing.assert_array_equal(actual, expected)
      welfare.append(compute_centralized_objective(current, *arrays[:3]))
    row = dict(mode=mode, processes=2, steps=2, nodes=40, functions=5, seed=7,
               planar=True, wall_seconds=elapsed, welfare=welfare, exact_match=True)
    rows.append(row)
    print(json.dumps(row), flush=True)
    (output / 'runner.json').write_text(json.dumps(rows, indent=2))
