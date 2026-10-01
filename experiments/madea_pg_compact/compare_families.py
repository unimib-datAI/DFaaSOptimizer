"""Bounded planar MADEA / MADEA-PG / PLASMA-Welfare comparison."""
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
import madea_pg
import run_faasmadea
from benchmark_madea_pg_compact import timeout_handler
from plasma.welfare import run as run_plasma_welfare
from postprocessing import load_solution
from remote_experiments.instances import load_materialized_instance
from run_centralized_model import encode_solution, get_current_load, update_data
from utils.centralized import validate_centralized_solution
from utils.faasmacro import compute_centralized_objective


def run(output, seconds):
  output.mkdir(parents=True, exist_ok=True)
  config = json.loads((ROOT / 'experiments/madea_pg_compact/config.json').read_text())
  config.update(max_steps=1, min_run_time=0, max_run_time=0, run_time_step=1)
  config['solver_options']['plasma_welfare'] = dict(W=1., rounds_per_step=20,
    epsilon=1e-6, hb_latency_rounds=1, hb_loss=0., staleness_rounds=3)
  cases = [(40, 7), (40, 42), (80, 7), (80, 42)]
  methods = [('MADEA', run_faasmadea.run), ('MADEA-PG', madea_pg.run),
             ('PLASMA-Welfare', run_plasma_welfare)]
  deadline = time.perf_counter() + seconds
  signal.signal(signal.SIGALRM, timeout_handler)
  source_files = ['run_faasmacro.py', 'models/local_sp.py', 'madea_pg.py',
                  'run_faasmadea.py', 'plasma/welfare.py', 'plasma/runner.py']
  protocol = dict(config=config, cases=cases, parallelism=0, per_run_limit_seconds=90,
    overall_limit_seconds=seconds, observation='one timestep; actual production runtime accounting',
    limits='MADEA 100 iterations and its native stopping; PG 5 sweeps and 0.25 seconds per node, bounded by remaining time; PLASMA-Welfare 20 rounds',
    source_hashes={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_files})
  (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
  rows = []
  for index, (nn, seed) in enumerate(cases):
    source = ROOT / f'solutions/madea-pg-compact-planar-2026-10-01/instances/n{nn}-f5-s{seed}-l1-k3'
    data, traces, agents, graph = load_materialized_instance(source)
    assert nx.check_planarity(graph)[0] and nx.is_connected(graph)
    assert all(degree == 3 for _, degree in graph.degree())
    load = get_current_load(traces, list(agents), 0)
    assert all(value == round(value) for value in load.values())
    current = update_data(data, {'incoming_load': load})
    input_hash = hashlib.sha256((source / 'metadata.json').read_bytes()).hexdigest()
    order = methods[index % len(methods):] + methods[:index % len(methods)]
    for name, runner in order:
      if time.perf_counter() >= deadline:
        break
      cfg = deepcopy(config)
      cfg.update(seed=seed, base_solution_folder=str(output / 'runs' / f'n{nn}-s{seed}-{name}'))
      cfg['limits'].update(instance_type='materialized', path=str(source))
      row = dict(nodes=nn, functions=5, seed=seed, method=name, input_hash=input_hash)
      started = time.perf_counter()
      signal.setitimer(signal.ITIMER_REAL, max(.01, min(90., deadline - started)))
      try:
        folder = runner(cfg, 0, log_on_file=True, disable_plotting=True)
        wall = time.perf_counter() - started
        signal.setitimer(signal.ITIMER_REAL, 0)
        sol, replicas, detailed, *_ = load_solution(folder, 'LSPc')
        x, y, z, r, _ = encode_solution(nn, 5, sol, detailed, replicas, 0)
        validate_centralized_solution(x, y, z, r, current)
        for a in (x, y, z, r):
          np.testing.assert_allclose(a, np.rint(a), atol=1e-6, rtol=0)
        welfare = compute_centralized_objective(current, x, y, z)
        exported = float(pd.read_csv(Path(folder) / 'obj.csv').iloc[0, -1])
        np.testing.assert_allclose(welfare, exported, rtol=0, atol=1e-6)
        row.update(status='ok', welfare=welfare, wall_seconds=wall, feasible=True,
          served_pct=100 * float((x.sum() + y.sum()) / (x.sum() + y.sum() + z.sum())),
          termination=str(pd.read_csv(Path(folder) / 'termination_condition.csv').iloc[0, -1]),
          folder=str(Path(folder).relative_to(ROOT)))
        if name == 'MADEA-PG':
          stats = pd.read_csv(Path(folder) / 'refinement.csv').iloc[0]
          row.update(pg_before=float(stats.welfare_before), pg_seconds=float(stats.seconds),
                     pg_reason=stats.reason, pg_budget=float(stats.time_budget))
      except TimeoutError:
        row.update(status='timeout', wall_seconds=time.perf_counter() - started)
      finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
      rows.append(row)
      pd.DataFrame(rows).to_csv(output / 'results.csv', index=False)
      print(json.dumps({k: v for k, v in row.items() if k not in ('folder', 'input_hash')}), flush=True)
  assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h
             for p, h in protocol['source_hashes'].items())
  print(f'COMPLETE {len(rows)} runs', flush=True)


if __name__ == '__main__':
  parser = ArgumentParser(description=__doc__)
  parser.add_argument('--output', default='solutions/madea-plasma-planar-2026-10-01')
  parser.add_argument('--seconds', type=float, default=480.)
  args = parser.parse_args()
  run(ROOT / args.output, args.seconds)
