"""Run a sparse hierarchical VNS on a fixed local allocation."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from centralized_vns.core import Options
from centralized_vns.io import from_data, initialize, load_instance
from centralized_vns_r.io import load_solution, save_result
from .core import solve


def main():
  cli = argparse.ArgumentParser(description=__doc__)
  cli.add_argument('instance', type=Path)
  cli.add_argument('--output', type=Path, required=True)
  cli.add_argument('--step', type=int, default=0)
  cli.add_argument('--solution', type=Path)
  cli.add_argument('--backend', choices=['cpu', 'cuda'], default='cpu')
  cli.add_argument('--depth', type=int, default=20)
  cli.add_argument('--pool-size', type=int, default=512)
  cli.add_argument('--candidates', type=int, default=512)
  cli.add_argument('--refine-every', type=int, default=256)
  cli.add_argument('--time-limit', type=float, default=300)
  cli.add_argument('--seed', type=int, default=4850)
  cli.add_argument('--relative-gap', type=float, default=0.001)
  args = cli.parse_args()
  if args.output.exists():
    cli.error('choose a fresh output directory')
  data, hashes = load_instance(args.instance, args.step)
  if args.solution:
    provenance = args.solution.parent / 'result.json'
    if provenance.exists():
      saved = json.loads(provenance.read_text())
      if 'input_sha256' in saved and (
        saved['input_sha256'] != hashes or saved.get('step') != args.step
      ):
        cli.error('warm start does not match input or step')
    p, y, meta = load_solution(data, args.solution)
  else:
    x, coverage = initialize(data, args.backend)
    p, y, meta = from_data(data, x), None, dict(initialization='DP', coverage=coverage)
  o = Options(
    pool_size=args.pool_size,
    candidates=args.candidates,
    backend=args.backend,
    seed=args.seed,
    time_limit=args.time_limit,
    iterations=10**9,
    max_device_bytes=3 * 1024**3,
  )
  r = solve(
    p, o, y, depth=args.depth, refine_every=args.refine_every, relative_gap=args.relative_gap
  )
  save_result(
    args.output,
    p,
    r,
    dict(
      **meta, options=asdict(o), instance=str(args.instance), step=args.step, input_sha256=hashes
    ),
  )
  print(
    json.dumps(
      {
        k: r[k]
        for k in (
          'potential_regions',
          'materialized_leaves',
          'coordinator_objective',
          'coordinator_upper',
          'coordinator_gap',
        )
      }
    )
  )


if __name__ == '__main__':
  main()
