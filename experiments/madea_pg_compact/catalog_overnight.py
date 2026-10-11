"""Resumable driver: run every (instance, method) job of the catalogue sample.

Jobs are claimed with an atomic mkdir lock, so any number of workers can be
(re)started at any time; finished jobs are skipped.  Each job runs in its own
process (catalog_campaign.py or catalog_centralized.py) with a hard timeout.
"""
from argparse import ArgumentParser
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
TIER1 = ['MADEA', 'MADEA-one-shot', 'one-shot-pg', 'hierarchical-one-shot-pg', 'PLASMA-Welfare',
         'MADEA2', 'MADEA2-one-shot', 'one-shot2-pg', 'FaaS-MAFW', 'MAFW-PG', 'MADEA-PG']
TIER2 = ['hierarchical-madea-pg', 'FaaS-MAPG-S', 'FaaS-MAPG-R', 'hierarchical-MAFW',
         'hierarchical-MAFW-PG', 'hierarchical-madea', 'hierarchical-one-shot']
GROUP_ORDER = {'main': 0, 'small': 1, 'scale': 2}


def jobs(cases, role):
  order = sorted(cases, key=lambda c: (GROUP_ORDER[c['group']], c['family'], c['id']))
  # case-major: every tier-1 method on an instance before the next instance,
  # so that partial campaigns still give paired comparisons
  decentralized = [(c, m) for c in order for m in TIER1] + [(c, m) for c in order for m in TIER2]
  centralized = [(c, 'centralized') for c in order]
  return centralized + decentralized if role == 'centralized-first' else decentralized + centralized


def run_job(folder, case, method, config, log_dir):
  out = folder / 'runs' / case['id'] / method
  if (out / 'finished.json').exists():
    return 'done'
  lock = folder / 'locks' / f"{case['id']}__{method}"
  try:
    lock.mkdir(parents=True)
  except FileExistsError:
    return 'locked'
  env = dict(os.environ, PYTHONPATH=f"{ROOT}:{ROOT / 'experiments/madea_pg_compact'}")
  if method == 'centralized':
    limit = 120. if case["nodes"] >= 150 else 60.
    cmd = [sys.executable, str(ROOT / 'experiments/madea_pg_compact/catalog_centralized.py'),
           '--folder', str(folder.relative_to(ROOT)), '--case', case['id'], '--time-limit', str(limit)]
    hard = limit + 600
  else:
    cmd = [sys.executable, str(ROOT / 'experiments/madea_pg_compact/catalog_campaign.py'),
           '--folder', str(folder.relative_to(ROOT)), '--config', str(config),
           '--case', case['id'], '--method', method]
    hard = 900
  started = time.time()
  with open(log_dir / f"{case['id']}__{method}.log", 'w') as log:
    try:
      code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=hard).returncode
    except subprocess.TimeoutExpired:
      code = 'hard-timeout'
  with open(folder / 'overnight_progress.csv', 'a') as progress:
    progress.write(f"{time.strftime('%H:%M:%S')},{case['id']},{method},{code},{time.time() - started:.1f}\n")
  if code != 0 and not (out / 'finished.json').exists():
    (folder / 'failed').mkdir(exist_ok=True)
    (folder / 'failed' / f"{case['id']}__{method}").write_text(str(code))
  return code


if __name__ == '__main__':
  parser = ArgumentParser(description=__doc__)
  parser.add_argument('--folder', default='solutions/catalog-sample-2026-10-10')
  parser.add_argument('--config', default='experiments/madea_pg_compact/catalog_config_overnight.json')
  parser.add_argument('--role', choices=['decentralized-first', 'centralized-first'], default='decentralized-first')
  parser.add_argument('--stop-at', help='HH:MM local wall clock after which no new job starts')
  parser.add_argument('--methods', nargs='*', help='run only these methods, on instances where every tier-1 method finished')
  args = parser.parse_args()
  folder = ROOT / args.folder
  log_dir = folder / 'logs'
  log_dir.mkdir(parents=True, exist_ok=True)
  cases = json.loads((folder / 'sample.json').read_text())
  if args.methods:
    done = [c for c in cases if all((folder / 'runs' / c['id'] / m / 'finished.json').exists() for m in TIER1)]
    queue = [(c, m) for c in sorted(done, key=lambda c: (GROUP_ORDER[c['group']], c['family'], c['id']))
             for m in args.methods]
  else:
    queue = jobs(cases, args.role)
  for case, method in queue:
    if args.stop_at and time.strftime('%H:%M') >= args.stop_at and time.strftime('%H:%M') < '12:00':
      break
    run_job(folder, case, method, ROOT / args.config, log_dir)
