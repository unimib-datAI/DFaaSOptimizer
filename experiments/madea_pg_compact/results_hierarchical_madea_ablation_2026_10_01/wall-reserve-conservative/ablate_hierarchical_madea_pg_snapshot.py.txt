"""Measure one production revision on a fixed subset of the shared planar inputs."""
from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json

import pandas as pd
import numpy as np
import networkx as nx
import families_extended as families


SCREEN = [
  'static-n40-f5-s7-l1-r0', 'static-n40-f10-s42-l1-r0',
  'static-n80-f5-s99-l1-r0', 'static-n80-f10-s2026-l1-r0',
  'static-n160-f5-s42-l1-r0', 'static-n160-f10-s7-l1-r0',
  'stress-n80-f10-s42-l1.5-r0', 'temporal-n40-f10-s7-l0.75-r0',
]
HOLDOUT = [
  'static-n80-f5-s7-l1-r0', 'static-n80-f10-s42-l1-r0',
  'stress-n80-f5-s7-l0.75-r0', 'temporal-n80-f5-s7-l0.75-r0',
]


def run(output, seconds, holdout):
  source = families.ROOT / 'solutions/families-five-methods-2026-10-01'
  original = pd.read_csv(source / 'runs.csv')
  protocol = json.loads((source / 'protocol.json').read_text())
  shared = families.ROOT / protocol['extension_protocol']['instance_source']
  requested = HOLDOUT if holdout else SCREEN
  planned = [case for case in protocol['cases'] if case['id'] in requested]
  assert len(planned) == len(requested)
  hashes = original.groupby('case_id').input_hash.first()

  def instance(config, case, output):
    profile = '-'.join(f'{value:g}' for value in case['profile'])
    path = shared / 'instances' / f"n{case['nodes']}-f{case['functions']}-s{case['seed']}-t{case['steps']}-l{profile}"
    assert hashlib.sha256((path / 'metadata.json').read_bytes()).hexdigest() == hashes[case['id']]
    data, traces, agents, graph = families.load_materialized_instance(path)
    assert nx.check_planarity(graph)[0] and nx.is_connected(graph)
    assert all(degree == 3 for _, degree in graph.degree())
    return path, data, traces, list(agents)

  files = ['hierarchical_auction/madea_cycles_runner.py', 'run_faasmadea.py',
           'madea_pg.py', 'experiments/madea_pg_compact/ablate_hierarchical_madea_pg.py']
  code = {name: (families.ROOT / name).read_bytes() for name in files}
  families.cases = lambda: planned
  families.instance = instance
  families.METHODS = [('hierarchical-madea-pg', families.madea_pg.run_hierarchical)]
  families.run(output, datetime.now(timezone.utc) + timedelta(seconds=seconds), 0, len(planned))
  saved = json.loads((output / 'protocol.json').read_text())
  for name, contents in code.items():
    assert (families.ROOT / name).read_bytes() == contents
    saved['source_hashes'][name] = hashlib.sha256(contents).hexdigest()
    (output / (Path(name).stem + '_snapshot.py.txt')).write_bytes(contents)
  saved.update(instance_source=str(shared.relative_to(families.ROOT)),
               previous_measurements=str(source.relative_to(families.ROOT)),
               selection='holdout' if holdout else 'screening')
  (output / 'protocol.json').write_text(json.dumps(saved, indent=2))
  runs = pd.read_csv(output / 'runs.csv')
  steps = pd.read_csv(output / 'timesteps.csv')
  for row in runs[runs.status.eq('ok')].itertuples():
    stats = pd.read_csv(families.ROOT / row.folder / 'refinement.csv')
    for step, pg in stats.iterrows():
      selected = steps.case_id.eq(row.case_id) & steps.step.eq(step)
      assert selected.sum() == 1
      np.testing.assert_allclose(pg.welfare_after, steps.loc[selected, 'welfare'], rtol=0, atol=1e-6)
      assert pg.welfare_after >= pg.welfare_before - 1e-6
      for key, value in dict(pg_before=pg.welfare_before, pg_seconds=pg.seconds,
                             pg_budget=pg.time_budget, pg_moves=pg.moves, pg_reason=pg.reason).items():
        steps.loc[selected, key] = value
  steps.to_csv(output / 'timesteps.csv', index=False)
  assert set(runs.case_id) == set(requested), 'Not all planned cases were measured'
  print('MEASURED ' + json.dumps(runs.status.value_counts().to_dict()), flush=True)


if __name__ == '__main__':
  parser = ArgumentParser(description=__doc__)
  parser.add_argument('--output', required=True)
  parser.add_argument('--seconds', type=float, default=900.)
  parser.add_argument('--holdout', action='store_true')
  args = parser.parse_args()
  run(families.ROOT / args.output, args.seconds, args.holdout)
