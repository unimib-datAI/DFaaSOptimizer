"""Replay regional LP bounds on the local Mac from a saved fixed-x VNS-Tree result."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import monotonic

import numpy as np
from centralized_vns.io import from_data, load_instance
from .lp import Relaxation
from .solvers import highs, metal


def compare(instance, archive, count, seconds, iterations, gap):
  start = monotonic()
  report = json.loads((archive / 'result.json').read_text())
  data, hashes = load_instance(instance, report['step'])
  if report['input_sha256'] != hashes or report['algorithm'] != 'VNS-Tree':
    raise ValueError('expected matching instance and VNS-Tree frontier')
  with np.load(archive / 'solution.npz') as saved:
    p = from_data(data, saved['x'])
    np.testing.assert_array_equal(saved['edges'], p.edges)
    p = replace(p, alpha=np.zeros_like(p.alpha))
    y = saved['y'].copy()
  incumbent = p.check(y)['welfare']
  np.testing.assert_allclose(incumbent, report['coordinator_objective'], rtol=0, atol=1e-8)
  threshold = incumbent + gap * max(1, abs(incumbent))
  edges, cuts = report['split_edges'], report['split_thresholds']
  leaves = report['leaves']
  before = np.array([l['upper'] for l in leaves])
  keys = [(l['depth'], l['code']) for l in leaves]
  keyset = set(keys)
  depth = len(edges)
  if (
    len(keys) != len(keyset)
    or len(cuts) != depth
    or not np.isfinite(before).all()
    or any(not 0 <= d <= depth or not 0 <= c < 1 << d for d, c in keys)
    or any((k, c % (1 << k)) in keyset for d, c in keys for k in range(d))
    or sum(1 << (depth - d) for d, c in keys) != 1 << depth
  ):
    raise ValueError('invalid saved covering frontier')
  open_ids = np.flatnonzero(before > threshold)
  if not len(open_ids):
    raise ValueError('no open leaves at requested gap')
  selected = open_ids[np.linspace(0, len(open_ids) - 1, min(count, len(open_ids)), dtype=int)]
  chosen = [keys[i] for i in selected]
  lp = Relaxation(p)
  lo, hi = lp.boxes(edges, cuts, chosen)
  setup = monotonic() - start
  results = {}
  for name, method in [('highs', highs), ('metal', metal)]:
    r = method(lp, lo, hi, seconds, **({'iterations': iterations} if name == 'metal' else {}))
    effective = before.copy()
    for idx, bound in zip(selected, r['bounds']):
      if bound is not None:
        effective[idx] = min(effective[idx], bound)
    r.update(
      certified=sum(v is not None for v in r['bounds']),
      closed=int(np.sum(effective[selected] <= threshold)),
      improved=int(np.sum(effective[selected] < before[selected])),
      selected_upper=float(effective[selected].max()),
      global_upper=float(effective.max()),
      global_gap=(float(effective.max()) - incumbent) / max(1, abs(incumbent)),
    )
    results[name] = r
    print(name, {k: v for k, v in r.items() if not isinstance(v, list)}, flush=True)
  for k, opt in enumerate(results['highs']['lp_optima']):
    if opt is not None:
      for r in results.values():
        if r['bounds'][k] is not None and r['bounds'][k] < opt - 1e-7 * max(1, abs(opt)):
          raise ArithmeticError('certificate inconsistent with regional LP optimum')
  return dict(
    algorithm='VNS-Tree-BatchBounds',
    instance=Path(instance).name,
    step=report['step'],
    input_sha256=hashes,
    incumbent=incumbent,
    relative_gap=gap,
    threshold=threshold,
    seconds_per_method=seconds,
    iteration_limit=iterations,
    shared_setup_seconds=setup,
    pair_cuts=lp.cuts,
    columns=len(lp.c),
    rows=len(lp.b),
    leaves_total=len(keys),
    selected=[
      dict(index=int(i), depth=keys[i][0], code=keys[i][1], inherited=before[i]) for i in selected
    ],
    initial_global_upper=float(before.max()),
    methods=results,
  )


def main():
  cli = argparse.ArgumentParser(description=__doc__)
  cli.add_argument('instance', type=Path)
  cli.add_argument('archive', type=Path, help='saved VNS-Tree directory')
  cli.add_argument('--output', type=Path, required=True)
  cli.add_argument('--regions', type=int, default=64)
  cli.add_argument('--seconds', type=float, default=300.0)
  cli.add_argument('--iterations', type=int, default=8192)
  cli.add_argument('--gap', type=float, default=0.001)
  args = cli.parse_args()
  if (
    args.output.exists()
    or args.regions < 1
    or args.iterations < 1
    or not np.isfinite(args.seconds)
    or args.seconds <= 0
    or not 0 <= args.gap <= 1
  ):
    cli.error('choose a new output path, positive budget/size and a gap in [0,1]')
  result = compare(
    args.instance, args.archive, args.regions, args.seconds, args.iterations, args.gap
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open('x') as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
    stream.write('\n')


if __name__ == '__main__':
  main()
