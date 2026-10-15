"""Reproducible real-GPU DP, complete local batch, and complete step measurements.

Repeated local problems are synthetic batches, never networks of 500/1000 nodes.
The optional GPU group must be installed. No absolute timing assertions are used.
"""
import argparse
import csv
import json
import multiprocessing as mp
import platform
import re
import subprocess
import sys
from collections import defaultdict
from copy import deepcopy
from functools import partial
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from models import dp_gpu
from models.dp_solver import prepare_local_problem, try_solve_local
from models.local_sp import solve_agent_batch
from models.sp import LSP
from plasma.core.sbm import minimize_replica_costs
from remote_experiments.instances import load_materialized_instance
from utils import run_log
from utils.centralized import get_current_load


def positive(value):
  n = int(value)
  if n <= 0:
    raise argparse.ArgumentTypeError('must be positive')
  return n


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--instance', type=Path, required=True)
  parser.add_argument('--nodes', nargs='+', type=positive, default=[1, 32, 128, 500, 1000])
  parser.add_argument('--repeats', type=positive, default=10)
  parser.add_argument('--workers', type=positive, default=4)
  parser.add_argument('--threads', nargs='+', type=int, choices=[64, 128, 256], default=[128, 64, 256])
  parser.add_argument('--output', type=Path, required=True)
  return parser.parse_args(argv)


def expanded_data(base, nodes):
  """Repeat the existing node-local inputs with new labels, without generating a network."""
  data = deepcopy(base)
  v, original = data[None], base[None]['Nn'][None]
  v['Nn'] = {None: nodes}
  for key in ('incoming_load', 'demand', 'alpha', 'delta', 'gamma'):
    if key in v:
      v[key] = {(n, f): base[None][key].get(((n - 1) % original + 1, f),
                    .9 if key == 'delta' else .1 if key == 'gamma' else 1.)
                for n in range(1, nodes + 1) for f in range(1, v['Nf'][None] + 1)}
  v['memory_capacity'] = {n: base[None]['memory_capacity'][(n - 1) % original + 1]
                          for n in range(1, nodes + 1)}
  return data


def local_data(data, agent):
  result = data.copy()
  result[None] = data[None].copy()
  result[None]['whoami'] = {None: agent + 1}
  return result


def init_worker(data, items):
  global _data, _items, _model
  _data, _items, _model = data, items, LSP()


def worker_dp(agent):
  return minimize_replica_costs(*_items[agent])


def worker_local(agent):
  return try_solve_local(_model, local_data(_data, agent))


def check_choices(actual, expected):
  assert len(actual) == len(expected)
  for a, b in zip(actual, expected):
    if b is None:
      assert a is None
    else:
      np.testing.assert_array_equal(a, b)


def check_solutions(actual, expected):
  assert len(actual) == len(expected)
  for a, b in zip(actual, expected):
    assert a is not None and b is not None
    assert a.keys() - {'runtime', 'runtime_kind'} == b.keys() - {'runtime', 'runtime_kind'}
    for key in b.keys() - {'runtime', 'runtime_kind'}:
      if key == 'obj':
        assert abs(a[key] - b[key]) <= 1e-9
      else:
        assert a[key] == b[key], key


def measure(rows, fn, expected, check, *, scope, nodes, backend, threads=0,
            gpu_fraction=0., repeats=10, synthetic=False, cold=False):
  for repeat in range(-1, repeats):
    profile = []
    started = perf_counter()
    actual = fn(profile)
    elapsed = perf_counter() - started
    check(actual, expected)  # correctness is checked outside the measured interval
    row = dict(scope=scope, problems=nodes, backend=backend,
               effective_backend='cuda' if gpu_fraction else 'cpu', gpu_fraction=gpu_fraction,
               phase='first_call' if repeat == -1 else 'warm', repeat=repeat,
               jit_cold=cold and repeat == -1, threads=threads, synthetic=synthetic,
               total_s=elapsed, packing_s=sum(p['packing_s'] for p in profile) if profile else None,
               kernel_s=sum(p['kernel_s'] for p in profile) if profile else None,
               upload_alloc_s=sum(p['upload_alloc_s'] for p in profile) if profile else None,
               download_s=sum(p['download_s'] for p in profile) if profile else None,
               device_bytes=max((p['device_bytes'] for p in profile), default=0),
               chunks=len(profile), method='', welfare=None, runner_wall_s=None, algorithm_s=None)
    rows.append(row)


def benchmark_batches(base, args, rows):
  model, first_cuda = LSP(), True
  warmups = []
  cold_nodes = next((n for n in args.nodes if n >= 2), None)
  if cold_nodes is not None:
    cold_data = expanded_data(base, cold_nodes)
    cold_agents = list(range(cold_nodes))
    expected_cold = [try_solve_local(model, local_data(cold_data, a)) for a in cold_agents]
    def cold_batch(profile):
      with patch.object(dp_gpu, 'minimize_replica_costs_batch',
                        partial(dp_gpu._minimize_batch, profile=profile)):
        return list(solve_agent_batch(cold_data, cold_agents, model,
                    {'use_dp': True, 'dp_device': 'cuda'}, 'missing_solver').values())
    measure(rows, cold_batch, expected_cold, check_solutions, scope='batch', nodes=cold_nodes,
            backend='cuda', threads=128, gpu_fraction=1., repeats=0, cold=True,
            synthetic=cold_nodes > base[None]['Nn'][None])
    rows[-1]['phase'] = 'cold'
    first_cuda = False
  for nodes in args.nodes:
    data = expanded_data(base, nodes)
    agents = list(range(nodes))
    problems = [prepare_local_problem(model, local_data(data, a)) for a in agents]
    if any(p is None or p.direct or not dp_gpu.gpu_problem_fits(p.costs, p.memory, p.ram_cap) for p in problems):
      raise ValueError('benchmark requires eligible ordinary local DP problems')
    items = [(p.costs, p.memory, p.ram_cap) for p in problems]
    expected = [minimize_replica_costs(*item) for item in items]
    solutions = [try_solve_local(model, local_data(data, a)) for a in agents]
    common = dict(nodes=nodes, repeats=args.repeats,
                  synthetic=nodes > base[None]['Nn'][None])
    # Spawn prevents inheriting a CUDA context when a later size creates its CPU pool.
    with mp.get_context('spawn').Pool(min(nodes, args.workers), init_worker, (data, items)) as pool:
      start = perf_counter()
      pool.map(worker_local, agents)
      warmups.append(dict(problems=nodes, pool_startup_warmup_s=perf_counter()-start))
      measure(rows, lambda _: [minimize_replica_costs(*i) for i in items], expected,
              check_choices, scope='dp', backend='cpu_sequential', **common)
      measure(rows, lambda _: pool.map(worker_dp, agents), expected,
              check_choices, scope='dp', backend='cpu_pool', **common)
      for threads in args.threads:
        measure(rows, lambda profile: dp_gpu._minimize_batch(
                  [i[0] for i in items], [i[1] for i in items], [i[2] for i in items],
                  threads=threads, profile=profile), expected, check_choices,
                scope='dp', backend='cuda', threads=threads, gpu_fraction=1., cold=first_cuda, **common)
        first_cuda = False
      measure(rows, lambda _: [try_solve_local(model, local_data(data, a)) for a in agents],
              solutions, check_solutions, scope='batch', backend='cpu_sequential', **common)
      measure(rows, lambda _: pool.map(worker_local, agents), solutions,
              check_solutions, scope='batch', backend='cpu_pool', **common)
      def gpu_batch(profile):
        # Instrument only this benchmark; the production API and default remain unchanged.
        with patch.object(dp_gpu, 'minimize_replica_costs_batch',
                          partial(dp_gpu._minimize_batch, profile=profile)):
          return list(solve_agent_batch(data, agents, model,
                      {'use_dp': True, 'dp_device': 'cuda'}, 'missing_solver').values())
      measure(rows, gpu_batch, solutions, check_solutions, scope='batch', backend='cuda',
              threads=128, gpu_fraction=1. if nodes >= 2 else 0., **common)
  return warmups


def compare_step(before, after):
  sys.path.insert(0, str(ROOT / 'experiments/harmonization'))
  from compare_artifacts import normalize_termination
  ignore = {'config.json', 'runtime.csv', 'out.log'}
  files = lambda d: {p.relative_to(d) for p in d.rglob('*') if p.is_file()}
  assert files(before) == files(after), 'step artifact inventory differs'
  for name in files(before):
    if name.name in ignore or name.suffix == '.png':
      continue
    a, b = before / name, after / name
    if name.name == 'termination_condition.csv':
      assert normalize_termination(a.read_text()) == normalize_termination(b.read_text())
    else:
      assert a.read_bytes() == b.read_bytes(), str(name)


def benchmark_steps(args, rows):
  sys.path.insert(0, str(ROOT / 'experiments/harmonization'))
  from capture_baseline import capture_config, baseline_runners
  import pandas as pd
  runners = baseline_runners()
  for method in ('MADEA', 'faas-macro'):
    for repeat in range(-1, args.repeats):
      folders = []
      for backend in ('cpu_sequential', 'cuda'):
        destination = args.output / 'steps' / method / str(repeat) / backend
        config = capture_config(args.instance, destination, method)
        config.update(max_steps=1, max_run_time=1, verbose=1)
        if backend == 'cuda':
          config['solver_options']['general']['dp_device'] = 'cuda'
        started = perf_counter()
        folder = Path(runners[method](config, 0, log_on_file=True, disable_plotting=True))
        elapsed = perf_counter() - started
        folders.append(folder)
        runtime = pd.read_csv(folder / 'runtime.csv')
        log = (folder / 'out.log').read_text()
        counts = re.findall(r'gpu_problems=(\d+) cpu_problems=(\d+)', log)
        gpu, cpu = sum(int(a) for a, _ in counts), sum(int(b) for _, b in counts)
        rows.append(dict(scope='full_step', problems=int(json.loads((folder / 'base_instance_data.json').read_text())['None']['Nn']['None']),
                         backend=backend, effective_backend='cuda' if gpu else 'cpu',
                         gpu_fraction=gpu / (gpu+cpu) if gpu+cpu else 0., threads=128 if gpu else 0,
                         phase='first_call' if repeat == -1 else 'warm', repeat=repeat, jit_cold=False,
                         synthetic=False, total_s=float(runtime.wall_s.sum()), packing_s=None,
                         kernel_s=None, upload_alloc_s=None, download_s=None, device_bytes=None,
                         chunks=None, method=method, welfare=float(pd.read_csv(folder / 'obj.csv').iloc[-1,-1]),
                         runner_wall_s=elapsed, algorithm_s=float(runtime.algorithm_s.sum())))
      compare_step(*folders)
    # Binding budgets are reported separately: faster phases can change achieved welfare.
    for backend in ('cpu_sequential', 'cuda'):
      destination = args.output / 'budget' / method / backend
      config = capture_config(args.instance, destination, method)
      config.update(max_steps=1, max_run_time=1, verbose=1)
      config['solver_options']['general']['algorithm_time_limit'] = .03
      if backend == 'cuda':
        config['solver_options']['general']['dp_device'] = 'cuda'
      folder = Path(runners[method](config, 0, log_on_file=True, disable_plotting=True))
      runtime = pd.read_csv(folder / 'runtime.csv')
      counts = re.findall(r'gpu_problems=(\d+) cpu_problems=(\d+)', (folder / 'out.log').read_text())
      gpu, cpu = sum(int(a) for a, _ in counts), sum(int(b) for _, b in counts)
      rows.append(dict(rows[-1], scope='binding_budget', backend=backend, repeat=0, phase='budget_0.03s',
                       effective_backend='cuda' if gpu else 'cpu',
                       gpu_fraction=gpu / (gpu+cpu) if gpu+cpu else 0., threads=128 if gpu else 0,
                       total_s=float(runtime.wall_s.sum()), algorithm_s=float(runtime.algorithm_s.sum()),
                       welfare=float(pd.read_csv(folder / 'obj.csv').iloc[-1,-1]), runner_wall_s=None))


def hardware():
  from numba import cuda
  device = cuda.get_current_device()
  free, total = cuda.current_context().get_memory_info()
  git = lambda *args: subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
  return dict(gpu=str(device.name), compute_capability=list(device.compute_capability),
              platform=platform.platform(), python=platform.python_version(), numpy=np.__version__,
              numba=version('numba'), numba_cuda=version('numba-cuda'),
              revision=git('rev-parse','HEAD'), dirty=bool(git('status','--porcelain')),
              parent_revision=subprocess.check_output(['git','-C',str(ROOT.parent.parent),'rev-parse','HEAD'], text=True).strip(),
              driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'], text=True).strip(),
              cuda_toolkit=subprocess.check_output(['/usr/local/cuda/bin/nvcc','--version'], text=True).strip(),
              measured_device_free_bytes=int(free), measured_device_total_bytes=int(total),
              occupancy=None, occupancy_note='Nsight Compute required; buffer bytes are allocated array extents, not peak VRAM',
              sm_count=int(device.MULTIPROCESSOR_COUNT))


def write_results(output, rows, metadata):
  groups = defaultdict(list)
  keys = ('scope','problems','backend','effective_backend','gpu_fraction','phase','threads','method')
  for row in rows:
    groups[tuple(row[k] for k in keys)].append(row['total_s'])
  summary = [dict(zip(keys, key), median_s=float(np.median(values)), p95_s=float(np.percentile(values,95)), samples=len(values))
             for key, values in groups.items()]
  (output / 'results.json').write_text(json.dumps(dict(hardware=metadata, summary=summary, samples=rows), indent=2))
  with (output / 'samples.csv').open('w', newline='') as stream:
    context = {key: metadata.get(key) for key in ('revision', 'parent_revision', 'python', 'numpy', 'numba_cuda')}
    context['hardware_gpu'] = metadata.get('gpu')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]) + list(context))
    writer.writeheader()
    writer.writerows(dict(row, **context) for row in rows)


def main(argv=None):
  args = parse_args(argv)
  args.output.mkdir(parents=True, exist_ok=False)
  base, traces, agents, _ = load_materialized_instance(args.instance)
  base[None]['incoming_load'] = get_current_load(traces, agents, 0)
  rows = []
  with run_log.open_run_log(None, 0):
    warmups = benchmark_batches(base, args, rows)
    # Keep partial numeric results if an artifact comparison fails in the full step benchmark.
    metadata = dict(hardware(), pool_warmups=warmups, instance=args.instance.name,
                    original_nodes=base[None]['Nn'][None], repeats=args.repeats,
                    warmup='one first call per shape/backend, excluded from warm median/p95',
                    cold_note='jit_cold marks the first real CUDA call; later first calls reuse compiled code')
    write_results(args.output, rows, metadata)
    benchmark_steps(args, rows)
    write_results(args.output, rows, metadata)
  print(f'Validated benchmark: {args.output / "results.json"}')


if __name__ == '__main__':
  main()
