"""Centralized reference (LoadManagementModel) for one sampled instance, step 0.

With ``selfish``, SelfishLoadManagementModel: every node first solves its own
problem (DP, HiGHS where DP does not apply) and the central solve must leave
it at least that local gain.

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

import highspy
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.contrib.appsi.solvers import Highs

import families_extended as families
from models.model import LoadManagementModel, _solver_option_name
from models.selfish import SelfishLoadManagementModel
from run_centralized_model import get_current_load, update_data
from utils.centralized import validate_centralized_solution
from utils.faasmacro import compute_centralized_objective
from utils import run_log

log = run_log.logger(__name__)


def _array(var, shape):
  return np.array([pyo.value(var[idx]) for idx in var], dtype=float).reshape(shape)


def solve(path, time_limit, threads=1, gap=1e-6, selfish=False, log_file=None,
          warm_start=None, warm_start_sha256=None, solution_file=None):
  from utils.warm_start import (data_fingerprint, evaluated_data_and_floors,
    initialize_variables, load_start, save_decisions, validate_decisions)
  data, traces, agents, graph = families.load_materialized_instance(path)
  data = update_data(data, {'incoming_load': get_current_load(traces, list(agents), 0)})
  Nn, Nf = data[None]['Nn'][None], data[None]['Nf'][None]
  overall_started = time.perf_counter()
  seed = None
  seed_info = None
  floors = None
  local = 0.0
  validation = 0.0
  if warm_start is not None:
    # Validate before spending memory/time on the global MIP. This phase is
    # recorded separately from the optimization budget and inside the job deadline.
    validation_started = time.perf_counter()
    data, floors = evaluated_data_and_floors(data)
    local = time.perf_counter()-validation_started
    seed, seed_info = load_start(warm_start, data, warm_start_sha256,
                                floors if selfish else None, Path(path).name)
    validation = time.perf_counter()-validation_started
    if solution_file:
      save_decisions(solution_file, seed, dict(seed_info, solution_source='warm_start'))
  started = time.perf_counter()
  deadline = started+time_limit
  model = SelfishLoadManagementModel() if selfish else LoadManagementModel()
  instance = None
  result = None
  candidate = None
  candidate_welfare = None
  rejected = None
  if time_limit > 0:
    instance = model.generate_instance(data)
    if selfish and seed is None:
      local_started = time.perf_counter()
      try:
        model.compute_local_gain_floors(instance, {'use_dp': True}, 'appsi_highs', deadline=deadline)
      except TimeoutError:
        pass
      local = time.perf_counter()-local_started
    if selfish:
      # A partially populated reference is never silently accepted.
      values = [instance.minimum_local_gain[n].value for n in instance.N]
      if all(v is not None for v in values):
        floors = np.array(values)
      else:
        instance = None
  build = time.perf_counter()-started
  if instance is not None and time.perf_counter() < deadline:
    evaluated = {None: {p.name:p.extract_values() for p in instance.component_objects(pyo.Param)}}
    if seed is not None:
      initialize_variables(instance, seed)
      if solution_file:
        save_decisions(solution_file, seed, dict(seed_info, solution_source='warm_start'))
    highspy.Highs.resetGlobalScheduler(True)
    opt = Highs()
    opt.config.time_limit = max(0.0, deadline-time.perf_counter())
    opt.config.load_solution = False
    opt.config.stream_solver = False
    opt.config.warmstart = seed is not None
    opt.highs_options.update({'threads': threads, 'mip_rel_gap': gap, 'random_seed': 0})
    if log_file:
      opt.highs_options[_solver_option_name('appsi_highs', 'LogFile')] = log_file
    result = opt.solve(instance)
    data = evaluated
    if result.best_feasible_objective is not None:
      result.solution_loader.load_vars()
      try:
        arrays = {name:_array(getattr(instance, name), shape) for name,shape in
                  [('x',(Nn,Nf)),('y',(Nn,Nn,Nf)),('z',(Nn,Nf)),('r',(Nn,Nf))]}
        candidate, candidate_welfare = validate_decisions(arrays, data, floors if selfish else None)
      except ValueError as error:
        if seed is None:
          raise
        rejected = str(error)
  bound = result.best_objective_bound if result else None
  if bound is not None and not np.isfinite(bound):
    bound = None
  termination = str(result.termination_condition) if result else 'TerminationCondition.maxTimeLimit'
  row = dict(model=model.name, nodes=Nn, functions=Nf, termination=termination,
    mip_rel_gap=gap, solver_log=str(log_file) if log_file else None,
    solver_threads=threads, build_seconds=build, local_reference_seconds=local,
    warm_start_validation_seconds=validation, wall_seconds=time.perf_counter()-overall_started,
    bound=bound, solver_objective=result.best_feasible_objective if result else None,
    input_hash=hashlib.sha256((Path(path)/'metadata.json').read_bytes()).hexdigest(),
    warm_start=dict(seed_info, accepted=True) if seed_info else None,
    solver_solution_rejected=rejected, proven_optimal=False)
  chosen, welfare, source = candidate, candidate_welfare, 'solver'
  if seed is not None and (candidate is None or candidate_welfare < seed_info['welfare']-1e-6):
    chosen, welfare, source = seed, seed_info['welfare'], 'warm_start'
  if chosen is None:
    return dict(row, status='no_solution', welfare=None, feasible=False,
                solution_source=None, rel_gap_to_bound=None)
  if bound is not None and bound < welfare-1e-6:
    row['bound'] = bound = None
  relative_gap = ((bound-welfare)/abs(bound) if bound else (0. if welfare == 0 else None)) if bound is not None else None
  row['proven_optimal'] = bool(source == 'solver' and bound is not None and abs(bound-welfare)<=1e-6)
  if solution_file:
    save_decisions(solution_file, chosen, dict(instance=Path(path).name, step=0,
      algorithm='selfish-centralized' if selfish else 'centralized', welfare=welfare,
      data_sha256=data_fingerprint(data), solution_source=source))
  return dict(row, status='ok', welfare=welfare, feasible=True, solution_source=source,
              solution_file=str(solution_file) if solution_file else None, rel_gap_to_bound=relative_gap)


if __name__ == '__main__':
  with run_log.open_run_log(None, 1):
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
      log.info("%s", json.dumps({k: v for k, v in row.items() if k != 'input_hash'}, default=str))
