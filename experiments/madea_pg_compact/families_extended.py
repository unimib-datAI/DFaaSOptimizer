"""Time-bounded comparison of four accelerated families on shared planar inputs."""
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import networkx as nx
import numpy as np
import pandas as pd
import madea_pg
import run_faasmadea
import decentralized_auction
from benchmark_madea_pg_compact import timeout_handler, digest
from plasma.welfare import run as run_plasma_welfare
from postprocessing import load_solution
from remote_experiments.batch import Experiment
from remote_experiments.instances import materialize_instance, load_materialized_instance
from run_centralized_model import get_current_load, update_data
from utils.centralized import encode_solution, validate_centralized_solution
from utils.faasmacro import compute_centralized_objective
from run_faasmacro import _available_cpu_count

METHODS = [('MADEA', run_faasmadea.run), ('MADEA-one-shot', decentralized_auction.run),
           ('MADEA-PG', madea_pg.run), ('PLASMA-Welfare', run_plasma_welfare)]


def cases():
  result = []
  for seed in (7, 42, 99, 2026):
    for nf in (5, 10):
      for nn in (40, 80, 160):
        result.append(dict(phase='static', nodes=nn, functions=nf, seed=seed,
                           steps=1, profile=[1.], repeat=0))
  for seed in (7, 42):
    for nf in (5, 10):
      for scale in (.75, 1.5):
        result.append(dict(phase='stress', nodes=80, functions=nf, seed=seed,
                           steps=1, profile=[scale], repeat=0))
  for seed in (7, 42):
    for nf in (5, 10):
      for nn in (40, 80):
        result.append(dict(phase='temporal', nodes=nn, functions=nf, seed=seed,
                           steps=3, profile=[.75, 1.25, 1.], repeat=0))
  for case in result[:24]:
    if case['seed'] in (7, 42):
      result.append(dict(case, repeat=1, phase='repeat'))
  for case in result:
    case['id'] = f"{case['phase']}-n{case['nodes']}-f{case['functions']}-s{case['seed']}-l{case['profile'][0]:g}-r{case['repeat']}"
  return result


def instance(config, case, output):
  generation = deepcopy(config)
  generation.update(seed=case['seed'], max_steps=case['steps'],
                    max_run_time=case['steps'], min_run_time=0, run_time_step=1)
  generation['limits']['Nn'] = dict(min=case['nodes'], max=case['nodes'])
  generation['limits']['Nf'] = dict(min=case['functions'], max=case['functions'])
  profile = '-'.join(f'{value:g}' for value in case['profile'])
  source = output / 'instances' / f"n{case['nodes']}-f{case['functions']}-s{case['seed']}-t{case['steps']}-l{profile}"
  materialize_instance(Experiment(source.name, 'families-planar', 'faas-madea-pg',
                                 case['seed'], {}, {}, generation), source)
  metadata_path = source / 'metadata.json'
  metadata = json.loads(metadata_path.read_text())
  if 'trace_profile' not in metadata:
    traces_path = source / 'input_requests_traces.json'
    traces = json.loads(traces_path.read_text())
    for by_node in traces.values():
      for node, values in by_node.items():
        by_node[node] = np.rint(np.array(values) * case['profile']).astype(int).tolist()
    traces_path.write_text(json.dumps(traces))
    metadata['files']['input_requests_traces.json'] = hashlib.sha256(traces_path.read_bytes()).hexdigest()
    metadata['trace_profile'] = case['profile']
    metadata['trace_rounding'] = 'numpy.rint, applied to the materialized integer traces'
    metadata_path.write_text(json.dumps(metadata, indent=2))
  assert metadata['trace_profile'] == case['profile']
  data, traces, agents, graph = load_materialized_instance(source)
  assert nx.check_planarity(graph)[0] and nx.is_connected(graph)
  assert all(degree == 3 for _, degree in graph.degree())
  return source, data, traces, list(agents)


def run(output, deadline, workers, max_cases, plasma_workers=None, *, config=None):
  output.mkdir(parents=True, exist_ok=True)
  config = (deepcopy(config) if config is not None else
            json.loads((ROOT / 'experiments/madea_pg_compact/config.json').read_text()))
  config.update(run_time_step=1, checkpoint_interval=1)
  config['solver_options']['plasma_welfare'] = dict(W=1., rounds_per_step=20,
    epsilon=1e-6, hb_latency_rounds=1, hb_loss=0., staleness_rounds=3)
  planned = cases()[:max_cases]
  watched = ['run_faasmacro.py', 'run_faasmadea.py', 'decentralized_auction.py',
             'madea_pg.py', 'decentralized_potentialgame.py', 'models/local_sp.py',
             'models/sp.py', 'plasma/welfare.py', 'plasma/runner.py', 'plasma/core/sbm.py',
             'generators/generate_load.py', 'generators/generate_data.py',
             'experiments/madea_pg_compact/families_extended.py']
  by_method = {name: (plasma_workers if name == 'PLASMA-Welfare' and plasma_workers is not None
                     else workers) for name, _ in METHODS}
  protocol = dict(config=config, cases=planned, parallelism=by_method,
    available_logical_cpus=_available_cpu_count(), deadline_utc=deadline.isoformat(),
    started_utc=datetime.now(timezone.utc).isoformat(),
    source_hashes={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in watched},
    runtime_accounting='actual full runner wall time including pool startup/shutdown; no replay or observer inside decision loop',
    native_limits='MADEA and one-shot <=100 iterations; general TimeLimit=max(30,0.5*N); PG <=5 sweeps and 0.25 seconds/node bounded by remaining native time; PLASMA 20 rounds/step',
    note='Different stopping criteria; not an equal-wall-budget comparison. fixed_sum integer rounding is unchanged; actual input totals are recorded.')
  (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
  signal.signal(signal.SIGALRM, timeout_handler)
  runs, steps = [], []
  for index, case in enumerate(planned):
    remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
    if remaining < 60:
      break
    source, base, traces, agents = instance(config, case, output)
    input_hash = hashlib.sha256((source / 'metadata.json').read_bytes()).hexdigest()
    order = METHODS[index % len(METHODS):] + METHODS[:index % len(METHODS)]
    for name, runner in order:
      method_workers = by_method[name]
      remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
      if remaining <= 0:
        break
      cfg = deepcopy(config)
      cfg.update(seed=case['seed'], max_steps=case['steps'], min_run_time=0,
        max_run_time=case['steps'], base_solution_folder=str(output / 'runs' / case['id'] / name))
      cfg['limits'].update(instance_type='materialized', path=str(source.resolve()))
      cfg['solver_options']['general']['TimeLimit'] = max(30., .5 * case['nodes'])
      row = dict(case_id=case['id'], phase=case['phase'], nodes=case['nodes'],
        functions=case['functions'], seed=case['seed'], repeat=case['repeat'],
        load_scale=case['profile'][0], steps=case['steps'], method=name,
        workers=method_workers, input_hash=input_hash)
      started = time.perf_counter()
      limit = min(150. * case['steps'], remaining)
      signal.setitimer(signal.ITIMER_REAL, max(.01, limit))
      records = []
      try:
        folder = Path(runner(cfg, method_workers, log_on_file=True, disable_plotting=True))
        wall = time.perf_counter() - started
        signal.setitimer(signal.ITIMER_REAL, 0)
        sol, replicas, detailed, *_ = load_solution(str(folder), 'LSPc')
        objectives = pd.read_csv(folder / 'obj.csv')
        termination = pd.read_csv(folder / 'termination_condition.csv')
        runtime = None if name == 'MADEA-one-shot' else pd.read_csv(folder / 'runtime.csv')
        refinements = (pd.read_csv(folder / 'refinement.csv')
                       if name in {'MADEA-PG', 'one-shot-pg', 'hierarchical-one-shot-pg',
                                   'hierarchical-madea-pg'} else None)
        trades = pd.read_csv(folder / 'plasma_welfare.csv') if name == 'PLASMA-Welfare' else None
        previous = None
        for step in range(case['steps']):
          loads = get_current_load(traces, agents, step)
          current = update_data(base, {'incoming_load': loads})
          arrays = encode_solution(case['nodes'], case['functions'], sol, detailed, replicas, step)
          x, y, z, r = arrays[:4]
          validate_centralized_solution(x, y, z, r, current)
          for array in arrays:
            np.testing.assert_allclose(array, np.rint(array), atol=1e-6, rtol=0)
          welfare = compute_centralized_objective(current, x, y, z)
          np.testing.assert_allclose(welfare, objectives.iloc[step, -1], atol=1e-6, rtol=0)
          item = dict(row, step=step, welfare=welfare, feasible=True,
            input_requests=float(sum(loads.values())), served_requests=float(x.sum() + y.sum()),
            rejected_requests=float(z.sum()), termination=str(termination.iloc[step, -1]),
            final_flow=digest(x, y), final_replicas=digest(r),
            replica_churn=None if previous is None else float(np.abs(r - previous).sum()))
          previous = r
          if refinements is not None:
            pg = refinements.iloc[step]
            item.update(pg_before=float(pg.welfare_before), pg_seconds=float(pg.seconds),
              pg_reason=pg.reason, pg_budget=float(pg.time_budget), pg_moves=int(pg.moves))
          if trades is not None:
            item['accepted_trades'] = int(trades.iloc[step].accepted_trades)
          records.append(item)
        row.update(status='ok', wall_seconds=wall,
          native_seconds=None if runtime is None else float(runtime['tot'].sum()),
          welfare_sum=float(sum(item['welfare'] for item in records)), feasible=True,
          folder=str(folder.relative_to(ROOT)))
        steps.extend(records)
      except TimeoutError:
        row.update(status='timeout', wall_seconds=time.perf_counter() - started)
      except Exception as error:
        row.update(status='error', wall_seconds=time.perf_counter() - started, error=str(error))
        (output / f"error-{case['id']}-{name}.txt").write_text(traceback.format_exc())
      finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
      runs.append(row)
      pd.DataFrame(runs).to_csv(output / 'runs.csv', index=False)
      pd.DataFrame(steps).to_csv(output / 'timesteps.csv', index=False)
      print(json.dumps({k: v for k, v in row.items() if k not in ('folder', 'input_hash')}), flush=True)
    if any(item['status'] == 'error' for item in runs if item['case_id'] == case['id']):
      break
  assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h
             for p, h in protocol['source_hashes'].items())
  finished = dict(completed_runs=len(runs), timestep_records=len(steps),
                  finished_utc=datetime.now(timezone.utc).isoformat())
  (output / 'finished.json').write_text(json.dumps(finished, indent=2))
  print('COMPLETE ' + json.dumps(finished), flush=True)


if __name__ == '__main__':
  parser = ArgumentParser(description=__doc__)
  parser.add_argument('--output', default='solutions/families-planar-hour-2026-10-01')
  parser.add_argument('--deadline', required=True, help='UTC ISO timestamp')
  parser.add_argument('--workers', type=int, default=4)
  parser.add_argument('--plasma-workers', type=int, default=None)
  parser.add_argument('--max-cases', type=int, default=52)
  args = parser.parse_args()
  run(ROOT / args.output, datetime.fromisoformat(args.deadline), args.workers, args.max_cases,
      args.plasma_workers)
