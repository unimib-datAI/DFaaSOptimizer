"""Three-timestep production runs without replay or neutralized solver runtimes."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import json
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import madea_pg
import run_faasmacro as macro
from benchmark_madea_pg_compact import full_propose, milp_only, digest, timeout_handler
from models.sp import LSP
from postprocessing import load_solution
from remote_experiments.batch import Experiment
from remote_experiments.instances import materialize_instance, load_materialized_instance
from run_centralized_model import encode_solution, get_current_load, update_data
from utils.centralized import validate_centralized_solution
from utils.faasmacro import compute_centralized_objective


def run(output, seconds):
  output.mkdir(parents=True, exist_ok=True)
  config = json.loads((ROOT / 'experiments/madea_pg_compact/config.json').read_text())
  config.update(max_steps=3, max_run_time=3, run_time_step=1)
  deadline = time.perf_counter() + seconds
  signal.signal(signal.SIGALRM, timeout_handler)
  native = macro.solve_agent_problem
  rows = []
  cases = [(40, 7, 10), (80, 7, 10), (40, 7, 5), (80, 7, 5), (40, 42, 5), (80, 42, 5)]
  protocol = dict(config=config, cases=cases, steps=3, frozen_initialization=False,
                  runtime_accounting='unchanged production counters; total measured wall time',
                  observer_trace_hashes=False, seconds=seconds, added_messages=0)
  (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
  for nn, seed, nf in cases:
    if time.perf_counter() >= deadline:
      break
    generation = deepcopy(config)
    generation.update(seed=seed)
    generation['limits']['Nn'] = dict(min=nn, max=nn)
    generation['limits']['Nf'] = dict(min=nf, max=nf)
    source = output / 'instances' / f'n{nn}-f{nf}-s{seed}'
    materialize_instance(Experiment(source.name, 'compact-pg-temporal', 'faas-madea-pg', seed, {}, {}, generation), source)
    data, traces, agents, _ = load_materialized_instance(source)
    agents = list(agents)
    for backend in ('native', 'milp'):
      def solve(model, local, options, solver):
        return milp_only(model, local, options, solver) if backend == 'milp' and type(model) is LSP else native(model, local, options, solver)
      for proposal in ('full', 'compact'):
        if time.perf_counter() >= deadline:
          break
        run_config = deepcopy(config)
        run_config.update(seed=seed, base_solution_folder=str(output / 'runs' / f'n{nn}-f{nf}-s{seed}-{backend}-{proposal}'))
        run_config['limits'].update(instance_type='materialized', path=str(source.resolve()))
        started = time.perf_counter()
        signal.setitimer(signal.ITIMER_REAL, max(.01, min(300., deadline - started)))
        try:
          with patch.object(macro, 'solve_agent_problem', solve), patch.object(madea_pg, 'propose_node_move', full_propose if proposal == 'full' else madea_pg.propose_node_move):
            folder = madea_pg.run(run_config, parallelism=0, log_on_file=True, disable_plotting=True)
          signal.setitimer(signal.ITIMER_REAL, 0)
          wall = time.perf_counter() - started
          sol, replicas, detailed, *_ = load_solution(folder, 'LSPc')
          stats = pd.read_csv(Path(folder) / 'refinement.csv')
          assert len(stats) == 3
          for t in range(3):
            current = update_data(data, {'incoming_load': get_current_load(traces, agents, t)})
            x, y, z, r, _ = encode_solution(nn, nf, sol, detailed, replicas, t)
            validate_centralized_solution(x, y, z, r, current)
            for array in (x, y, z, r):
              np.testing.assert_allclose(array, np.rint(array), atol=1e-6, rtol=0)
            row = dict(nodes=nn, functions=nf, seed=seed, step=t, initial_backend=backend, proposal=proposal,
                       welfare=compute_centralized_objective(current, x, y, z),
                       pg_seconds=float(stats.iloc[t].seconds), pg_reason=stats.iloc[t].reason,
                       wall_seconds=wall, final_flow=digest(x, y), final_replicas=digest(r),
                       feasible=True, status='ok', folder=folder)
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k not in ('folder', 'final_flow', 'final_replicas')}), flush=True)
        except TimeoutError:
          rows.append(dict(nodes=nn, functions=nf, seed=seed, initial_backend=backend, proposal=proposal, status='timeout'))
        finally:
          signal.setitimer(signal.ITIMER_REAL, 0)
        pd.DataFrame(rows).to_csv(output / 'results.csv', index=False)
  print(f'COMPLETE {len(rows)} timestep records', flush=True)


if __name__ == '__main__':
  run(Path(sys.argv[1]), float(sys.argv[2]))
