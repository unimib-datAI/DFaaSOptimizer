"""Run the cooperative fixed-x VNS-HiGHS algorithm on one instance and step."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from centralized_vns.core import Options
from centralized_vns.io import from_data, initialize, load_instance
from centralized_vns_r.io import load_solution, save_result
from .core import solve


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('instance', type=Path)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--solution', type=Path, help='optional checked NPZ with initial x and y')
  parser.add_argument('--step', type=int, default=0)
  parser.add_argument('--backend', choices=['cpu', 'metal', 'cuda'], default='cpu')
  parser.add_argument('--init-device', choices=['cpu', 'cuda'], default='cpu')
  parser.add_argument('--pool-size', type=int, default=32)
  parser.add_argument('--candidates', type=int, default=128)
  parser.add_argument('--iterations', type=int, default=100000)
  parser.add_argument('--time-limit', type=float, default=300.)
  parser.add_argument('--relative-gap', type=float, default=1e-4)
  parser.add_argument('--batch-size', type=int, default=8, help='CUDA iterations between exchanges')
  parser.add_argument('--seed', type=int, default=4850)
  parser.add_argument('--max-device-mib', type=int, default=2048)
  args = parser.parse_args()
  if args.output.exists():
    parser.error('choose a new output directory')
  options = Options(backend=args.backend, pool_size=args.pool_size, candidates=args.candidates,
                    iterations=args.iterations, time_limit=args.time_limit, seed=args.seed,
                    max_device_bytes=args.max_device_mib * 1024**2)
  options.validate()
  data, hashes = load_instance(args.instance, args.step)
  if args.solution is None:
    x, coverage = initialize(data, args.init_device)
    p, y, metadata = from_data(data, x), None, dict(initialization='local DP',
      init_device=args.init_device, initialization_coverage=coverage)
  else:
    if not args.solution.is_file():
      parser.error('--solution requires an NPZ file')
    provenance = args.solution.parent / 'result.json'
    if provenance.exists():
      saved = json.loads(provenance.read_text())
      if 'input_sha256' in saved and (saved['input_sha256'] != hashes or saved.get('step') != args.step):
        parser.error('solution provenance does not match instance and step')
    p, y, metadata = load_solution(data, args.solution)
  result = solve(p, options, y, relative_gap=args.relative_gap, batch_size=args.batch_size)
  report = save_result(args.output, p, result, dict(**metadata, instance=str(args.instance.resolve()),
    step=args.step, input_sha256=hashes, options=asdict(options), gap_target=args.relative_gap,
    batch_size=args.batch_size))
  print(json.dumps({key: report[key] for key in ('algorithm', 'welfare', 'upper_bound',
                                                'relative_gap', 'termination_reason')}))


if __name__ == '__main__':
  main()
