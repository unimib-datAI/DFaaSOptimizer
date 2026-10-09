"""Measure real changing-input SP batches, including snapshot transfer and merge."""
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from unittest.mock import patch
import json
import multiprocessing as mp
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import networkx as nx
import numpy as np
import run_faasmacro as macro
from models.sp import LSP
from remote_experiments.instances import load_materialized_instance
from run_centralized_model import get_current_load, update_data


def cold_solve(data, agents, model, solver, options, processes, prices=None):
  """Previous lifecycle, with the same number of workers as the warm variant."""
  with mp.Pool(processes=min(processes, len(agents)),
               initializer=macro.init_parallel_worker,
               initargs=(data, options, solver, model)) as pool:
    return dict(pool.map(partial(macro.solve_single_agent, detailed_pi=prices), agents))


def run(instance, output):
  data, traces, agents, graph = load_materialized_instance(instance)
  agents = list(agents)
  assert nx.check_planarity(graph)[0] and nx.is_connected(graph)
  nf = data[None]['Nf'][None]
  batches = [update_data(data, {'incoming_load': get_current_load(traces, agents, t)})
             for t in range(3)]
  prices = [np.full((len(agents), nf), float(t)) for t in range(3)]
  references, rows = [], []
  for mode, processes in [('sequential', 0), ('cold', 2), ('warm', 2), ('warm', 4), ('cold', 4)]:
    with macro.parallel_solver_session(), patch.object(
        macro, '_solve_agents_parallel', cold_solve if mode == 'cold' else macro._solve_agents_parallel):
      for step, batch in enumerate(batches):
        started = time.perf_counter()
        result = macro.solve_subproblem(batch, agents, LSP(), 'missing_solver', {'use_dp': True},
                                        processes, detailed_pi=prices[step])
        elapsed = time.perf_counter() - started
        if mode == 'sequential':
          references.append(result)
        else:
          for k in (1, 2, 3, 4, 5, 6, 7):
            np.testing.assert_array_equal(result[k], references[step][k])
          assert result[8] == references[step][8]
          assert result[9] == references[step][9]
        row = dict(mode=mode, processes=processes, step=step, seconds=elapsed,
                   lifecycle='cold/startup' if step == 0 or mode == 'cold' else 'warm',
                   exact_match=True)
        rows.append(row)
        print(json.dumps(row), flush=True)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(dict(nodes=len(agents), functions=nf, planar=True,
    start_method=mp.get_start_method(), available_cpus=macro._available_cpu_count(),
    scope='solve_subproblem wall time including fresh snapshots, IPC and merge; three timesteps with changing prices',
    rows=rows), indent=2))


if __name__ == '__main__':
  parser = ArgumentParser(description=__doc__)
  parser.add_argument('--instance', default='solutions/madea-pg-compact-planar-temporal-2026-10-01/instances/n80-f10-s7')
  parser.add_argument('--output', default='solutions/madea-pg-warm-pool-planar-2026-10-01/batches.json')
  args = parser.parse_args()
  run(args.instance, Path(args.output))
