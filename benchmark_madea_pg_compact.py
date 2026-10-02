"""Bounded, paired compact/full PG experiments; Gurobi solves node-local MILPs.

Outputs include immutable instances, per-move traces, source hashes and checks.
The frozen initialization isolates tie choices; its measured wall time is added
back to total wall time. Observer hashes/feasibility never select algorithm moves.
"""
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import signal
import time

import numpy as np
import pandas as pd
import networkx as nx

import madea_pg
import run_faasmacro as macro
import run_faasmadea as madea
from models.local_sp import try_solve_local
from models.sp import LSP
from postprocessing import load_solution
from remote_experiments.batch import Experiment
from remote_experiments.instances import materialize_instance, load_materialized_instance
from run_centralized_model import get_current_load, update_data
from utils.centralized import encode_solution, validate_centralized_solution
from utils.faasmacro import compute_centralized_objective


def full_propose(node, cap, y, data, model, solver, options, parallelism):
  """Pre-compaction proposal: full copy and Nn*Nn*Nf commitment dictionary."""
  nn, nf = data[None]['Nn'][None], data[None]['Nf'][None]
  local = deepcopy(data)
  local[None]['omega_ub'] = {f + 1: float(cap[f]) for f in range(nf)}
  local[None]['y_bar'] = {
    (m + 1, n + 1, f + 1): float(max(y[m, n, f], 0.))
    for m in range(nn) for n in range(nn) for f in range(nf)
  }
  result = macro.solve_subproblem(local, [node], model, solver, options, parallelism)
  return (np.array(result[1][node], dtype=float), np.array(result[5][node], dtype=float),
          np.array(result[4][node], dtype=float), float(result[10]['tot']))


def milp_only(model, data, options, solver):
  return model.solve(model.generate_instance(data), options, solver)


def digest(*arrays):
  result = hashlib.sha256()
  for array in arrays:
    result.update(np.rint(array).astype(np.int64).tobytes())
  return result.hexdigest()


def timeout_handler(*_):
  raise TimeoutError('experiment wall-clock limit')


def run(config, cases, output, seconds):
  output.mkdir(parents=True, exist_ok=True)
  watched = ['models/local_sp.py', 'models/sp.py', 'models/model.py', 'run_faasmacro.py',
             'run_faasmadea.py', 'madea_pg.py', 'decentralized_potentialgame.py',
             'plasma/core/sbm.py', 'benchmark_madea_pg_compact.py']
  hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in watched}
  started_all = time.perf_counter()
  deadline = started_all + seconds
  signal.signal(signal.SIGALRM, timeout_handler)
  original_move = madea_pg.node_move
  compact_propose = madea_pg.propose_node_move
  rows, checks = [], []
  protocol = dict(config=config, cases=cases, source_hashes=hashes, overall_seconds=seconds,
                  added_messages=0, initial_runtime_accounting='neutralized for replay; measured wall time added back',
                  trace_overhead='same observer hashes after each move in both paths',
                  initial_solver='Gurobi local MILP or exact local DP; no global initialization selection')
  (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
  for case_index, case in enumerate(cases):
    if time.perf_counter() >= deadline:
      break
    nn, seed = case['nodes'], case['seed']
    nf, scale = case.get('functions', 5), case.get('load_scale', 1.)
    name = f"n{nn}-f{nf}-s{seed}-l{scale:g}-k{case.get('degree', 3)}"
    generation = deepcopy(config)
    generation.update(seed=seed, max_steps=1)
    generation['limits']['Nn'] = dict(min=nn, max=nn)
    generation['limits']['Nf'] = dict(min=nf, max=nf)
    generation['limits']['neighborhood']['k'] = case.get('degree', 3)
    source = output / 'instances' / name
    if not source.exists():
      materialize_instance(Experiment(name, 'compact-pg', 'faas-madea-pg', seed, {}, {}, generation), source)
      traces_path = source / 'input_requests_traces.json'
      traces = json.loads(traces_path.read_text())
      for values in traces.values():
        for node, trace in values.items():
          values[node] = np.rint(np.array(trace) * scale).astype(int).tolist()
      traces_path.write_text(json.dumps(traces))
      metadata_path = source / 'metadata.json'
      metadata = json.loads(metadata_path.read_text())
      metadata['trace_scale'] = scale
      metadata['files'][traces_path.name] = hashlib.sha256(traces_path.read_bytes()).hexdigest()
      metadata_path.write_text(json.dumps(metadata, indent=2))
    data, traces, agents, graph = load_materialized_instance(source)
    assert nx.check_planarity(graph)[0] and nx.is_connected(graph)
    assert all(degree == 3 for _, degree in graph.degree())
    agents = list(agents)
    data = update_data(data, {'incoming_load': get_current_load(traces, agents, 0)})
    print(f'CASE {case_index + 1}/{len(cases)} {name}', flush=True)
    initials, initial_seconds = {}, {}
    for backend in ('native', 'milp'):
      started = time.perf_counter()
      signal.setitimer(signal.ITIMER_REAL, max(.01, min(180., deadline - started)))
      try:
        # Native here still retains production fallback for unsupported inputs.
        solve = macro.solve_agent_problem if backend == 'native' else milp_only
        with patch.object(macro, 'solve_agent_problem', solve):
          initials[backend] = macro.solve_subproblem(data, agents, LSP(), 'gurobi', config['solver_options']['general'], 0)
        assert set(str(initials[backend][9]['tot']).split('-')) == {'optimal'}
        initial_seconds[backend] = time.perf_counter() - started
      finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    assert abs(initials['native'][8]['tot'] - initials['milp'][8]['tot']) < 1e-7
    checks.append(dict(case=name, primary_objective_equal=True, planar=True, connected=True, degree=3,
                       x_differences=int(np.count_nonzero(initials['native'][1] != initials['milp'][1])),
                       replica_differences=int(np.count_nonzero(initials['native'][5] != initials['milp'][5]))))
    for backend in ('native', 'milp'):
      paired_steps = {}
      for path in (('full', 'compact') if case_index % 2 == 0 else ('compact', 'full')):
        if time.perf_counter() >= deadline:
          break
        frozen = list(deepcopy(initials[backend]))
        for key in frozen[10]:
          frozen[10][key] = 0.
        steps, intercepted, fallbacks = [], [0], [0]
        def initial_hook(*args, **kwargs):
          assert type(args[2]) is LSP
          intercepted[0] += 1
          return deepcopy(tuple(frozen))
        def native_tail(model, local, options, solver):
          result = try_solve_local(model, local)
          if result is None:
            fallbacks[0] += 1
            return milp_only(model, local, options, solver)
          return result
        def traced_move(node, x, y, r, *args, **kwargs):
          result = original_move(node, x, y, r, *args, **kwargs)
          steps.append(dict(node=node, accepted=bool(result[0]), flow=digest(x, y), replicas=digest(r)))
          return result
        run_config = deepcopy(config)
        run_config.update(seed=seed, max_steps=1, base_solution_folder=str(output / 'runs' / f'{name}-{backend}-{path}'))
        run_config['limits'].update(instance_type='materialized', path=str(source.resolve()))
        row = dict(case=name, nodes=nn, functions=nf, seed=seed, load_scale=scale,
                   degree=case.get('degree', 3), initial_backend=backend, proposal=path,
                   initial_seconds=initial_seconds[backend])
        started = time.perf_counter()
        signal.setitimer(signal.ITIMER_REAL, max(.01, min(180., deadline - started)))
        try:
          with patch.object(madea, 'solve_subproblem', initial_hook), patch.object(macro, 'solve_agent_problem', native_tail), patch.object(madea_pg, 'propose_node_move', full_propose if path == 'full' else compact_propose), patch.object(madea_pg, 'node_move', traced_move):
            folder = madea_pg.run(run_config, parallelism=0, log_on_file=True, disable_plotting=True)
          signal.setitimer(signal.ITIMER_REAL, 0)
          tail_seconds = time.perf_counter() - started
          assert intercepted == [1]
          sol, replicas, detailed, *_ = load_solution(folder, 'LSPc')
          x, y, z, r, _ = encode_solution(nn, nf, sol, detailed, replicas, 0)
          validate_centralized_solution(x, y, z, r, data)
          for array in (x, y, z, r):
            np.testing.assert_allclose(array, np.rint(array), atol=1e-6, rtol=0)
          stats = pd.read_csv(Path(folder) / 'refinement.csv').iloc[0]
          row.update(status='ok', welfare=compute_centralized_objective(data, x, y, z),
                     auction_welfare=float(stats.welfare_before), pg_seconds=float(stats.seconds),
                     pg_moves=int(stats.moves), pg_reason=stats.reason, pg_budget=float(stats.time_budget),
                     pg_proposals=len(steps), completed_sweeps=len(steps) // nn,
                     final_flow=digest(x, y), final_replicas=digest(r), feasible=True,
                     fallbacks=fallbacks[0], tail_seconds=tail_seconds,
                     wall_seconds=tail_seconds + initial_seconds[backend], folder=folder)
          paired_steps[path] = steps
          if len(paired_steps) == 2:
            common = min(map(len, paired_steps.values()))
            assert paired_steps['full'][:common] == paired_steps['compact'][:common], name
            row.update(prefix_equal=True, common_prefix=common)
            partner = next(a for a in rows if a['case'] == name and a['initial_backend'] == backend)
            if partner['pg_reason'] != 'time budget exhausted' and row['pg_reason'] != 'time budget exhausted':
              assert row['final_flow'] == partner['final_flow'] and row['final_replicas'] == partner['final_replicas'], name
        except TimeoutError as exc:
          row.update(status='timeout', wall_seconds=time.perf_counter() - started + initial_seconds[backend], error=str(exc))
        finally:
          signal.setitimer(signal.ITIMER_REAL, 0)
        (output / f'{name}-{backend}-{path}-trace.json').write_text(json.dumps(steps))
        rows.append(row)
        pd.DataFrame(rows).to_csv(output / 'results.csv', index=False)
        (output / 'initial_checks.json').write_text(json.dumps(checks, indent=2))
        print(json.dumps({k: v for k, v in row.items() if k not in ('final_flow', 'final_replicas', 'folder')}), flush=True)
  assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
  protocol.update(elapsed_seconds=time.perf_counter() - started_all, rows=len(rows))
  (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
  print(f"COMPLETE {len(rows)} runs in {protocol['elapsed_seconds']:.1f}s", flush=True)


if __name__ == '__main__':
  parser = ArgumentParser(description=__doc__)
  parser.add_argument('--config', default='experiments/madea_pg_compact/config.json')
  parser.add_argument('--cases', default='experiments/madea_pg_compact/cases.json')
  parser.add_argument('--output', default='solutions/madea-pg-compact-planar-2026-10-01')
  parser.add_argument('--seconds', type=float, default=3600.)
  args = parser.parse_args()
  run(json.loads(Path(args.config).read_text()), json.loads(Path(args.cases).read_text()), Path(args.output), args.seconds)
