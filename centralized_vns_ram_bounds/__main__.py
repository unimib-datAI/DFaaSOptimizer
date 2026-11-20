"""Evaluate integer-RAM traffic envelopes on the same archived regional LPs."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from functools import partial
from time import monotonic

import numpy as np
from centralized_vns.io import from_data, load_instance
from centralized_vns_batch_bounds.lp import Relaxation
from centralized_vns_batch_bounds.solvers import highs
from .formulation import add_ram_rows, propagate


def compare(instance, archive, baseline, seconds, *, conditional=True):
  start = monotonic()
  reference = json.loads(baseline.read_text())
  tree = json.loads((archive / 'result.json').read_text())
  data, hashes = load_instance(instance, reference['step'])
  if (
    reference['algorithm'] != 'VNS-Tree-BatchBounds'
    or tree['algorithm'] != 'VNS-Tree'
    or hashes != reference['input_sha256']
    or hashes != tree['input_sha256']
    or tree['step'] != reference['step']
  ):
    raise ValueError('instance and run provenance mismatch')
  with np.load(archive / 'solution.npz') as s:
    p = from_data(data, s['x'])
    p = replace(p, alpha=np.zeros_like(p.alpha))
    np.testing.assert_array_equal(s['edges'], p.edges)
    incumbent = p.check(s['y'])['welfare']
  np.testing.assert_allclose(incumbent, reference['incumbent'], rtol=0, atol=1e-8)
  subset = reference['selected']
  leaves = tree['leaves']
  if not subset or len({r['index'] for r in subset}) != len(subset):
    raise ValueError('need distinct archived leaves')
  for r in subset:
    if leaves[r['index']] != {'depth': r['depth'], 'code': r['code'], 'upper': r['inherited']}:
      raise ValueError('leaf differs from reference')
  options = {} if conditional else None
  lp = Relaxation(p, strengthen=partial(add_ram_rows, options_by_cell=options))
  keys = [(r['depth'], r['code']) for r in subset]
  lo, hi = lp.boxes(tree['split_edges'], tree['split_thresholds'], keys)
  root_lp = float(lp.base) - lp.solver.getObjectiveValue()
  root_upper = lp.certificate(-np.asarray(lp.root.row_dual), lp.lo, lp.hi)
  if conditional:
    propagate(p, lo, hi, options)
  setup = monotonic() - start
  run = highs(lp, lo, hi, seconds)
  effective = np.minimum(root_upper, [l['upper'] for l in leaves])
  threshold = incumbent + reference['relative_gap'] * max(1, abs(incumbent))
  rows = []
  for r, new, old, opt in zip(
    subset, run['bounds'], reference['methods']['highs']['bounds'], run['lp_optima'], strict=True
  ):
    i = r['index']
    if new is not None:
      if opt is None or abs(opt - new) > 1e-6 * max(1, abs(opt)):
        raise ArithmeticError('LP objective and checked upper bound disagree')
      effective[i] = min(new, effective[i])
    rows.append(
      dict(
        **r,
        old_lp=old,
        new_lp=new,
        lp_optimum=opt,
        effective=float(effective[i]),
        improved=bool(effective[i] < r['inherited']),
        closed=bool(effective[i] <= threshold),
      )
    )
  return dict(
    algorithm='VNS-Tree-RAMBounds',
    instance=instance.name,
    input_sha256=hashes,
    step=tree['step'],
    gap_target=reference['relative_gap'],
    incumbent=incumbent,
    threshold=threshold,
    selected=rows,
    certified=sum(r['new_lp'] is not None for r in rows),
    improved=sum(r['improved'] for r in rows),
    closed=sum(r['closed'] for r in rows),
    initial_global_upper=max(l['upper'] for l in leaves),
    global_upper=float(effective.max()),
    root_lp=root_lp,
    root_upper=root_upper,
    conditional=conditional,
    pair_cuts=lp.cuts,
    **lp.strengthening,
    rows=len(lp.b),
    columns=len(lp.c),
    setup_seconds=setup,
    solve_seconds=run['seconds'],
    solve_budget_seconds=seconds,
  )


def main():
  cli = argparse.ArgumentParser(description=__doc__)
  for name in ('instance', 'archive', 'baseline', 'output'):
    cli.add_argument(name, type=Path)
  cli.add_argument(
    '--unconditional', action='store_true', help='skip propagation of mandatory branch flows'
  )
  cli.add_argument('--seconds', type=float, default=300.0)
  a = cli.parse_args()
  if a.output.exists() or not np.isfinite(a.seconds) or a.seconds <= 0:
    cli.error('choose a new output file and a positive finite budget')
  result = compare(a.instance, a.archive, a.baseline, a.seconds, conditional=not a.unconditional)
  a.output.parent.mkdir(parents=True, exist_ok=True)
  with a.output.open('x') as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
    stream.write('\n')
  print({k: v for k, v in result.items() if k not in ('input_sha256', 'selected')})


if __name__ == '__main__':
  main()
