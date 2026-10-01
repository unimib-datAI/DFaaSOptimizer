"""Small paired planar comparison using the existing validated family harness."""
from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json

import pandas as pd
import numpy as np
import families_extended as families
import one_shot_pg


def run(output, seconds):
  planned = [case for case in families.cases() if case['phase'] == 'static'
             and case['seed'] in (7, 42) and case['nodes'] in (40, 80)]
  families.cases = lambda: planned
  families.METHODS = [('MADEA-one-shot', families.decentralized_auction.run),
                     ('one-shot-pg', one_shot_pg.run), ('MADEA-PG', families.madea_pg.run)]
  files = ['one_shot_pg.py', 'experiments/madea_pg_compact/compare_one_shot_pg.py']
  hashes = {path: hashlib.sha256((families.ROOT / path).read_bytes()).hexdigest() for path in files}
  families.run(output, datetime.now(timezone.utc) + timedelta(seconds=seconds), 0, len(planned))
  protocol_path = output / 'protocol.json'
  protocol = json.loads(protocol_path.read_text())
  protocol['source_hashes'].update(hashes)
  protocol_path.write_text(json.dumps(protocol, indent=2))
  assert all(hashlib.sha256((families.ROOT / path).read_bytes()).hexdigest() == h for path, h in hashes.items())
  runs = pd.read_csv(output / 'runs.csv')
  methods = [name for name, _ in families.METHODS]
  complete = [key for key, group in runs.groupby('case_id')
              if len(group) == 3 and group.status.eq('ok').all() and set(group.method) == set(methods)]
  paired = runs[runs.case_id.isin(complete)]
  assert paired.groupby('case_id').input_hash.nunique().eq(1).all()
  w = paired.pivot(index='case_id', columns='method', values='welfare_sum')[methods]
  t = paired.pivot(index='case_id', columns='method', values='wall_seconds')[methods]
  assert (w['one-shot-pg'] >= w['MADEA-one-shot'] - 1e-6).all()
  refinements = []
  for row in paired[paired.method.eq('one-shot-pg')].itertuples():
    stats = pd.read_csv(families.ROOT / row.folder / 'refinement.csv').iloc[0].to_dict()
    np.testing.assert_allclose(stats['welfare_before'], w.loc[row.case_id, 'MADEA-one-shot'], atol=1e-6, rtol=0)
    np.testing.assert_allclose(stats['welfare_after'], row.welfare_sum, atol=1e-6, rtol=0)
    refinements.append(dict(case_id=row.case_id, **stats))
  pd.DataFrame(refinements).to_csv(output / 'refinements.csv', index=False)
  gains = (w.div(w['MADEA-one-shot'], axis=0) - 1) * 100
  w.to_csv(output / 'welfare.csv')
  t.to_csv(output / 'wall_seconds.csv')
  gains.to_csv(output / 'gains_pct.csv')
  metrics = dict(complete_cases=len(complete), successful_runs=int(runs.status.eq('ok').sum()),
    errors=int(runs.status.eq('error').sum()), timeouts=int(runs.status.eq('timeout').sum()),
    welfare_gain_mean_pct=gains.mean().to_dict(), welfare_gain_min_pct=gains.min().to_dict(),
    winner_including_ties={method:int(np.isclose(w[method], w.max(axis=1), atol=1e-6, rtol=0).sum()) for method in methods})
  (output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
  print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
  parser = ArgumentParser(description=__doc__)
  parser.add_argument('--output', default='solutions/one-shot-pg-planar-2026-10-01')
  parser.add_argument('--seconds', type=float, default=300.)
  args = parser.parse_args()
  run(families.ROOT / args.output, args.seconds)
