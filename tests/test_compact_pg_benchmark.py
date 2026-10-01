import json

import pandas as pd

from experiments.madea_pg_compact.summarize import summarize


def test_report_separates_incomplete_runs_and_actual_caps(tmp_path):
  rows = []
  for backend in ('native', 'milp'):
    for proposal in ('full', 'compact'):
      rows.append(dict(case='paired', nodes=40, functions=5, seed=7, load_scale=1.,
                       degree=3, initial_backend=backend, proposal=proposal, status='ok',
                       feasible=True, prefix_equal=True if proposal == 'compact' else None,
                       pg_seconds=2. if proposal == 'full' else 1., pg_proposals=10,
                       wall_seconds=4. if proposal == 'full' else 2., welfare=100.,
                       pg_reason='no improving proposal', final_flow='same', final_replicas='same',
                       fallbacks=0, pg_budget=2.5 if proposal == 'full' else 3.))
  orphan = dict(rows[0], case='orphan', prefix_equal=None)
  rows.append(orphan)
  rows.append(dict(orphan, case='interrupted', status='timeout'))
  pd.DataFrame(rows).to_csv(tmp_path / 'results.csv', index=False)
  summarize(tmp_path)
  metrics = json.loads((tmp_path / 'metrics.json').read_text())
  assert metrics['runs'] == 6
  assert metrics['pairs'] == 2
  assert metrics['unpaired_ok_runs'] == 1
  assert metrics['unsuccessful_runs'] == 1
  assert metrics['prefix_checks'] == 2
  assert metrics['pg_throughput_median'] == 2.
  assert metrics['compact_within_lower_paired_cap'] == 2
