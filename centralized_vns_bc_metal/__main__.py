import argparse
from dataclasses import replace, asdict
import json
from pathlib import Path
from time import monotonic
import highspy
import numpy as np
from centralized_vns.core import Options
from centralized_vns.io import from_data, load_instance
from centralized_vns_r.io import digest
from .experiment import Model, trial


def main():
  cli = argparse.ArgumentParser()
  for k in ('instance', 'archive', 'output'):
    cli.add_argument(k, type=Path)
  cli.add_argument('modes', nargs='+', choices=['highs', 'vns', 'coordinated'])
  cli.add_argument('--seconds', type=float, default=300)
  a = cli.parse_args()
  if not np.isfinite(a.seconds) or a.seconds <= 0:
    cli.error('expected positive finite limit')
  for mode in a.modes:
    if (a.output / mode).exists():
      cli.error('output already exists: ' + mode)
  start = monotonic()
  data, hashes = load_instance(a.instance, 0)
  old = json.loads((a.archive / 'result.json').read_text())
  assert old['input_sha256'] == hashes and old['step'] == 0
  with np.load(a.archive / 'solution.npz') as s:
    p = from_data(data, s['x'])
    p = replace(p, alpha=np.zeros_like(p.alpha))
    np.testing.assert_array_equal(s['edges'], p.edges)
    y = s['y'].copy()
  np.testing.assert_allclose(p.check(y)['welfare'], old['coordinator_objective'], atol=1e-8, rtol=0)
  model = Model(p)
  setup = monotonic() - start
  highspy.Highs.resetGlobalScheduler(True)
  o = Options(
    backend='metal',
    pool_size=64,
    candidates=512,
    seed=4850,
    max_device_bytes=3 * 1024**3,
    time_limit=a.seconds,
  )
  for mode in a.modes:
    print('start', mode, flush=True)
    record, r = trial(model, y, o, mode)
    r.update(
      options=asdict(o),
      setup_seconds=setup,
      rows=model.A.shape[0],
      cols=model.A.shape[1],
      pair_cuts=model.cuts,
      root_upper=model.upper,
      solver_threads=1,
      highspy=highspy.Highs().version(),
      seed_sha256=digest(a.archive / 'solution.npz'),
      input_sha256=hashes,
    )
    folder = a.output / mode
    folder.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
      folder / 'solution.npz',
      x=p.x,
      y=record.y,
      edges=p.edges,
      replicas=record.checked['replicas'],
      z=record.checked['z'],
    )
    (folder / 'result.json').write_text(json.dumps(r, indent=2, allow_nan=False) + '\n')
    print(mode, r['lower'], r['upper'], r['gap'], r['nodes'], r['iterations'], flush=True)


if __name__ == '__main__':
  main()
