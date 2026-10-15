"""Serial complete-run measurements; use separate processes for cold CUDA runs.

Profile runs are diagnostic only and must not be compared with clean timings.
The first call is reported separately from the requested warm repetitions.
"""
import argparse
import cProfile
import csv
import json
import platform
import re
import resource
import subprocess
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from methods import HARMONIZED


def positive(value):
  value = int(value)
  if value < 1:
    raise argparse.ArgumentTypeError('must be positive')
  return value


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--config', type=Path, required=True)
  parser.add_argument('--methods', nargs='+', choices=HARMONIZED, default=list(HARMONIZED))
  parser.add_argument('--repeats', type=positive, default=10)
  parser.add_argument('--parallelism', type=int, default=0)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--profile', action='store_true')
  return parser.parse_args(argv)


def write_results(output, rows, coverage):
  groups = defaultdict(list)
  keys = ('method', 'seed', 'dp_device', 'compute_device', 'parallelism', 'phase', 'diagnostic')
  for row in rows:
    groups[tuple(row[k] for k in keys)].append(row)
  summaries = []
  for key, samples in groups.items():
    group = dict(zip(keys, key), samples=len(samples))
    for metric in ('wall_s', 'algorithm_s', 'runner_wall_s'):
      values = [r[metric] for r in samples]
      group[f'median_{metric}'] = float(np.median(values))
      group[f'p95_{metric}'] = float(np.percentile(values, 95))
    summaries.append(group)
  revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
  (output/'summary.json').write_text(json.dumps(dict(groups=summaries, revision=revision,
    python=platform.python_version(), numpy=np.__version__,
    cold_note='first call in this process; only the first CUDA use includes context/JIT',
    memory_note='peak_rss_kib is cumulative process high-water RSS, not per-run VRAM'), indent=2))
  for name, records, fields in [('samples.csv', rows, list(rows[0]) if rows else []),
      ('coverage.csv', coverage, ['method', 'repeat', 'operation', 'backend', 'count', 'reason'])]:
    with (output/name).open('w', newline='') as f:
      writer = csv.DictWriter(f, fieldnames=fields)
      writer.writeheader()
      writer.writerows(records)


def main(argv=None):
  args = parse_args(argv)
  args.output.mkdir(parents=True, exist_ok=False)
  base = json.loads(args.config.read_text())
  (args.output/'input-config.json').write_text(json.dumps(base, indent=2))
  rows, coverage = [], []
  for method in args.methods:
    for repeat in range(-1, args.repeats):
      config = deepcopy(base)
      config.update(base_solution_folder=str((args.output/'runs'/method/str(repeat)).resolve()), verbose=2)
      options = config.get('solver_options', {}).get('general', {})
      profiler = cProfile.Profile() if args.profile else None
      if profiler:
        profiler.enable()
      started = perf_counter()
      folder = HARMONIZED[method](config, args.parallelism, log_on_file=True, disable_plotting=True)
      elapsed = perf_counter() - started
      if profiler:
        profiler.disable()
        profiler.dump_stats(str(args.output/f'{method}-{repeat}.prof'))
      if folder is None:
        raise RuntimeError(f'{method} returned no valid run')
      folder = Path(folder)
      runtime = pd.read_csv(folder/'runtime.csv')
      termination = pd.read_csv(folder/'termination_condition.csv').iloc[:,-1].tolist()
      # Successful runner export executes the shared centralized feasibility check.
      row = dict(method=method, seed=config['seed'], dp_device=options.get('dp_device', 'cpu'),
        compute_device=options.get('compute_device', 'cpu'), parallelism=args.parallelism,
        phase='first_call' if repeat < 0 else 'warm', repeat=repeat,
        wall_s=float(runtime.wall_s.sum()), algorithm_s=float(runtime.algorithm_s.sum()),
        runner_wall_s=elapsed, rounds=json.dumps(termination),
        objective=float(pd.read_csv(folder/'obj.csv').iloc[-1,-1]), feasible=True,
        diagnostic=args.profile, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        folder=str(folder))
      rows.append(row)
      text = (folder/'out.log').read_text()
      coverage_file = folder/'compute_coverage.json'
      if coverage_file.exists():
        for item in json.loads(coverage_file.read_text())['operations']:
          coverage.append(dict(method=method,repeat=repeat,operation=item['operation'],
            backend=item['backend'],count=item['count'],reason=item.get('reason','')))
      for gpu, cpu in re.findall(r'gpu_problems=(\d+) cpu_problems=(\d+)', text):
        for backend, count in [('cuda', gpu), ('cpu', cpu)]:
          coverage.append(dict(method=method, repeat=repeat, operation='local_dp',
            backend=backend, count=int(count), reason='see run log cpu_reasons'))
      # No logged CUDA work is coverage zero, even for an explicitly requested device.
      if not re.search(r'gpu_problems=', text):
        coverage.append(dict(method=method, repeat=repeat, operation='local_dp',
          backend='cpu', count=0, reason='no reported CUDA batch'))
      write_results(args.output, rows, coverage)
      print(json.dumps(row), flush=True)


if __name__ == '__main__':
  main()
