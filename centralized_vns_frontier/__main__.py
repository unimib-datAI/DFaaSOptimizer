"""Search fixed-x forwarding regions and certify the coordinator gap."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np

from centralized_vns.core import Options
from centralized_vns.io import from_data, initialize, load_instance
from centralized_vns_r.io import load_solution, save_result
from .core import solve


def save_frontier(folder, p, result, metadata):
  saved = save_result(folder, p, result, metadata)
  if 'final_relink' in result:
    np.savez_compressed(
      Path(folder) / 'final-parents.npz',
      x=result['x'],
      edges=p.edges,
      regions=result['final_relink']['parent_regions'],
      y=result['final_parent_y'],
      search_y=result['search_y'],
      pre_final_y=result['pre_final_y'],
    )
  return saved


def main():
  cli = argparse.ArgumentParser(description=__doc__)
  cli.add_argument('instance', type=Path)
  cli.add_argument('--output', type=Path, required=True)
  cli.add_argument('--solution', type=Path)
  cli.add_argument('--step', type=int, default=0)
  cli.add_argument('--backend', choices=['cpu', 'metal', 'cuda'], default='cpu')
  cli.add_argument('--init-device', choices=['cpu', 'cuda'], default='cpu')
  cli.add_argument('--branches', type=int, default=256)
  cli.add_argument('--pool-size', type=int, default=256)
  cli.add_argument('--candidates', type=int, default=512)
  cli.add_argument('--seed', type=int, default=4850)
  cli.add_argument('--time-limit', type=float, default=300.0)
  cli.add_argument('--relative-gap', type=float, default=1e-3)
  cli.add_argument('--max-device-mib', type=int, default=3072)
  cli.add_argument('--slice-seconds', type=float, default=3.0)
  cli.add_argument(
    '--final-relink-top',
    type=int,
    default=0,
    metavar='K',
    help='after the search, relink the best K distinct region leaders (0 disables)',
  )
  cli.add_argument(
    '--global-share', type=float, default=0.20, help='fraction of proof time for the global MIP'
  )
  cli.add_argument(
    '--tighten-receive',
    action='store_true',
    help='bound incoming traffic in the role constraint by receiver RAM and fixed local service',
  )
  cli.add_argument(
    '--pair-cuts',
    action='store_true',
    help='separate joint receiver RAM bounds after partitioning and within regions',
  )
  cli.add_argument(
    '--group-cuts',
    action='store_true',
    help='separate integer RAM bounds on three or more functions',
  )
  cli.add_argument(
    '--lagrangian',
    action='store_true',
    help='compute endpoint-decomposition bounds for every region',
  )
  args = cli.parse_args()
  if args.output.exists():
    cli.error('choose a new output directory')
  o = Options(
    pool_size=args.pool_size,
    candidates=args.candidates,
    seed=args.seed,
    backend=args.backend,
    iterations=10**9,
    time_limit=args.time_limit,
    max_device_bytes=args.max_device_mib * 1024**2,
  )
  o.validate()
  data, hashes = load_instance(args.instance, args.step)
  if args.solution:
    provenance = args.solution.parent / 'result.json'
    if provenance.exists():
      saved = json.loads(provenance.read_text())
      if 'input_sha256' in saved and (
        saved['input_sha256'] != hashes or saved.get('step') != args.step
      ):
        cli.error('warm start does not match input or step')
    p, y, metadata = load_solution(data, args.solution)
  else:
    x, coverage = initialize(data, args.init_device)
    p, y, metadata = (
      from_data(data, x),
      None,
      dict(initialization='DP', init_device=args.init_device, coverage=coverage),
    )
  result = solve(
    p,
    o,
    y,
    branches=args.branches,
    relative_gap=args.relative_gap,
    slice_seconds=args.slice_seconds,
    global_share=args.global_share,
    final_relink_top=args.final_relink_top,
    tighten_receive=args.tighten_receive,
    pair_cuts=args.pair_cuts,
    group_cuts=args.group_cuts,
    lagrangian=args.lagrangian,
  )
  saved = save_frontier(
    args.output,
    p,
    result,
    dict(
      **metadata,
      options=asdict(o),
      instance=str(args.instance.resolve()),
      input_sha256=hashes,
      step=args.step,
      requested_branches=args.branches,
      gap_target=args.relative_gap,
      slice_seconds=args.slice_seconds,
    ),
  )
  summary = {
    k: saved[k]
    for k in (
      'algorithm',
      'branches',
      'coordinator_objective',
      'coordinator_upper',
      'coordinator_gap',
      'pairs_relinked',
      'termination_reason',
      'elapsed_s',
    )
  }
  if 'final_relink' in saved:
    stage = saved['final_relink']
    summary.update(
      final_pairs=stage['pairs'],
      final_gain=stage['post_objective'] - stage['pre_objective'],
      final_seconds=stage['seconds'],
    )
  print(json.dumps(summary))


if __name__ == '__main__':
  main()
