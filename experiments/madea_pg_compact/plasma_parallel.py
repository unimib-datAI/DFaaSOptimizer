"""Short paired planar PLASMA-Welfare benchmark, including process startup."""
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import networkx as nx
import numpy as np
import pandas as pd
from benchmark_madea_pg_compact import timeout_handler
from plasma.welfare import run as run_plasma_welfare
from postprocessing import load_solution
from remote_experiments.instances import load_materialized_instance
from run_centralized_model import encode_solution, get_current_load, update_data
from utils.centralized import validate_centralized_solution
from utils.faasmacro import compute_centralized_objective
from run_faasmacro import _available_cpu_count


def run(output, seconds, repeats):
  output.mkdir(parents=True, exist_ok=True)
  config = json.loads((ROOT / 'experiments/madea_pg_compact/config.json').read_text())
  config.update(max_steps=1, min_run_time=0, max_run_time=0, run_time_step=1)
  config['solver_options']['plasma_welfare'] = dict(W=1., rounds_per_step=20,
    epsilon=1e-6, hb_latency_rounds=1, hb_loss=0., staleness_rounds=3)
  cases = [(40, 7), (40, 42), (80, 7), (80, 42)]
  files = ['run_faasmacro.py', 'plasma/welfare.py', 'plasma/runner.py']
  protocol = dict(config=config, cases=cases, parallelism=[0, 2, 4], repeats=repeats,
    available_logical_cpus=_available_cpu_count(), overall_limit_seconds=seconds,
    observation='20 identical rounds; full wall time includes startup, transfers and pool shutdown',
    source_hashes={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files})
  (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
  signal.signal(signal.SIGALRM, timeout_handler)
  deadline = time.perf_counter() + seconds
  references, rows = {}, []
  for repeat in range(repeats):
    for index, (nn, seed) in enumerate(cases):
      source = ROOT / f'solutions/madea-pg-compact-planar-2026-10-01/instances/n{nn}-f5-s{seed}-l1-k3'
      data, traces, agents, graph = load_materialized_instance(source)
      assert nx.check_planarity(graph)[0] and nx.is_connected(graph)
      current = update_data(data, {'incoming_load': get_current_load(traces, list(agents), 0)})
      choices = [0, 2, 4]
      offset = (index + repeat) % 3
      for workers in choices[offset:] + choices[:offset]:
        if time.perf_counter() >= deadline:
          return
        cfg = deepcopy(config)
        cfg.update(seed=seed, base_solution_folder=str(output / 'runs' / f'n{nn}-s{seed}-j{workers}-r{repeat}'))
        cfg['limits'].update(instance_type='materialized', path=str(source))
        row = dict(nodes=nn, functions=5, seed=seed, workers=workers, repeat=repeat)
        started = time.perf_counter()
        signal.setitimer(signal.ITIMER_REAL, max(.01, min(60., deadline - started)))
        try:
          folder = run_plasma_welfare(cfg, workers, log_on_file=True, disable_plotting=True)
          wall = time.perf_counter() - started
          signal.setitimer(signal.ITIMER_REAL, 0)
          sol, replicas, detailed, *_ = load_solution(folder, 'LSPc')
          arrays = encode_solution(nn, 5, sol, detailed, replicas, 0)
          x, y, z, r = arrays[:4]
          validate_centralized_solution(x, y, z, r, current)
          for array in arrays:
            np.testing.assert_array_equal(array, np.rint(array))
          counters = [pd.read_csv(Path(folder) / name)
                      for name in ('obj.csv', 'plasma_messages.csv', 'plasma_welfare.csv')]
          key = nn, seed
          if key in references:
            expected, frames = references[key]
            for actual, oracle in zip(arrays, expected):
              np.testing.assert_array_equal(actual, oracle)
            for actual, oracle in zip(counters, frames):
              pd.testing.assert_frame_equal(actual, oracle)
          else:
            references[key] = arrays, counters
          welfare = compute_centralized_objective(current, x, y, z)
          np.testing.assert_allclose(welfare, counters[0].iloc[0, -1], atol=1e-6, rtol=0)
          row.update(status='ok', wall_seconds=wall, welfare=welfare, feasible=True,
                     accepted_trades=int(counters[2].accepted_trades.sum()))
        except TimeoutError:
          row.update(status='timeout', wall_seconds=time.perf_counter() - started)
        finally:
          signal.setitimer(signal.ITIMER_REAL, 0)
        rows.append(row)
        pd.DataFrame(rows).to_csv(output / 'results.csv', index=False)
        print(json.dumps(row), flush=True)
  assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h
             for p, h in protocol['source_hashes'].items())
  print(f'COMPLETE {len(rows)} runs', flush=True)


if __name__ == '__main__':
  parser = ArgumentParser(description=__doc__)
  parser.add_argument('--output', default='solutions/plasma-parallel-planar-2026-10-01')
  parser.add_argument('--seconds', type=float, default=300.)
  parser.add_argument('--repeats', type=int, default=2)
  args = parser.parse_args()
  run(ROOT / args.output, args.seconds, args.repeats)
