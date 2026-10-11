"""Centralized reference (LoadManagementModel) for one sampled instance, step 0.

Solved with HiGHS (appsi) under a wall-clock limit; records the incumbent
welfare (same metric as the decentralized runs, ``compute_centralized_objective``
on a solution checked by ``validate_centralized_solution``), the dual bound
and the termination status.  The gap to the bound tells how far the
incumbent can be from the true optimum when the time limit is hit.
"""
from argparse import ArgumentParser
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.contrib.appsi.solvers import Highs

import families_extended as families
from models.model import LoadManagementModel
from run_centralized_model import get_current_load, update_data
from utils.centralized import validate_centralized_solution
from utils.faasmacro import compute_centralized_objective


def _array(var, shape):
  return np.array([pyo.value(var[idx]) for idx in var], dtype=float).reshape(shape)


def solve(path, time_limit, threads=1, gap=1e-6):
  data, traces, agents, graph = families.load_materialized_instance(path)
  data = update_data(data, {'incoming_load': get_current_load(traces, list(agents), 0)})
  Nn, Nf = data[None]['Nn'][None], data[None]['Nf'][None]
  started = time.perf_counter()
  instance = LoadManagementModel().generate_instance(data)
  build = time.perf_counter() - started
  opt = Highs()
  opt.config.time_limit = max(1.0, time_limit - build)
  opt.config.load_solution = False
  opt.config.stream_solver = False
  opt.highs_options.update({'threads': threads, 'mip_rel_gap': gap, 'random_seed': 0})
  result = opt.solve(instance)
  wall = time.perf_counter() - started
  row = dict(nodes=Nn, functions=Nf, termination=str(result.termination_condition),
             build_seconds=build, wall_seconds=wall,
             bound=result.best_objective_bound, solver_objective=result.best_feasible_objective,
             input_hash=hashlib.sha256((Path(path) / 'metadata.json').read_bytes()).hexdigest())
  if result.best_feasible_objective is None:
    return dict(row, status='no_solution', welfare=None, feasible=False)
  result.solution_loader.load_vars()
  x = np.rint(_array(instance.x, (Nn, Nf)))
  y = np.rint(_array(instance.y, (Nn, Nn, Nf)))
  z = np.rint(_array(instance.z, (Nn, Nf)))
  r = np.rint(_array(instance.r, (Nn, Nf)))
  validate_centralized_solution(x, y, z, r, data)
  welfare = compute_centralized_objective(data, x, y, z)
  return dict(row, status='ok', welfare=welfare, feasible=True,
              rel_gap_to_bound=(row['bound'] - welfare) / abs(row['bound']) if row['bound'] else None)


if __name__ == '__main__':
  parser = ArgumentParser(description=__doc__)
  parser.add_argument('--folder', default='solutions/catalog-sample-2026-10-10')
  parser.add_argument('--case')
  parser.add_argument('--path', help='instance folder (instead of --case)')
  parser.add_argument('--time-limit', type=float, default=300.)
  args = parser.parse_args()
  folder = families.ROOT / args.folder
  if args.path:
    case = dict(id=Path(args.path).name, path=args.path)
  else:
    case = {x['id']: x for x in json.loads((folder / 'sample.json').read_text())}[args.case]
    local = folder / 'instances' / case['name']
    if local.exists():
      case['path'] = str(local)
  out = folder / 'runs' / case['id'] / 'centralized'
  out.mkdir(parents=True, exist_ok=True)
  if not (out / 'finished.json').exists():
    row = dict(case_id=case['id'], method='centralized', time_limit=args.time_limit,
               **solve(case['path'], args.time_limit))
    pd.DataFrame([row]).to_csv(out / 'runs.csv', index=False)
    (out / 'finished.json').write_text(json.dumps({'status': row['status']}))
    print(json.dumps({k: v for k, v in row.items() if k != 'input_hash'}, default=str))
