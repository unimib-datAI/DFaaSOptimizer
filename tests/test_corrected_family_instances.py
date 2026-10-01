"""Economic correction must preserve the exact physical instance and traces."""
import json
from pathlib import Path

from remote_experiments.batch import Experiment
from remote_experiments.instances import materialize_instance, load_materialized_instance


def test_alpha_gt_beta_correction_preserves_physical_inputs(tmp_path, monkeypatch):
  root = Path(__file__).resolve().parents[1]
  monkeypatch.syspath_prepend(str(root / 'experiments/madea_pg_compact'))
  from rerun_alpha_gt_beta import correct_instance
  config = json.loads((root / 'experiments/madea_pg_compact/config.json').read_text())
  config.update(seed=7, max_steps=3, max_run_time=3)
  config['limits']['Nn'] = {'min': 40, 'max': 40}
  config['limits']['Nf'] = {'min': 5, 'max': 5}
  source = materialize_instance(Experiment('test', 'test', 'faas-madea', 7, {}, {}, config), tmp_path/'old')
  # Preserve a deliberately changed temporal trace, not just the regenerated baseline trace.
  traces = source/'input_requests_traces.json'
  payload = json.loads(traces.read_text())
  payload['0']['0'][1] += 3
  traces.write_text(json.dumps(payload))
  import hashlib
  metadata = json.loads((source/'metadata.json').read_text())
  metadata['files'][traces.name] = hashlib.sha256(traces.read_bytes()).hexdigest()
  metadata['trace_profile'] = [.75, 1.25, 1.]
  (source/'metadata.json').write_text(json.dumps(metadata))
  original_bytes = {p.name: p.read_bytes() for p in source.iterdir()}
  target = correct_instance(source, tmp_path/'new')
  old, *_ = load_materialized_instance(source)
  new, *_ = load_materialized_instance(target)
  for key, value in old[None].items():
    if key not in {'beta', 'delta'}:
      assert new[None][key] == value
  assert old[None]['beta'] != new[None]['beta']
  assert all(0 <= beta < new[None]['alpha'][i, f]
             for (i, j, f), beta in new[None]['beta'].items())
  for (i, f), delta in new[None]['delta'].items():
    assert delta == new[None]['beta'][1, 2, f]
  for name in ('graph.json', 'load_limits.json', 'input_requests_traces.json'):
    assert (target/name).read_bytes() == original_bytes[name]
  assert all(p.read_bytes() == original_bytes[p.name] for p in source.iterdir())
  assert json.loads((target/'metadata.json').read_text())['trace_profile'] == [.75, 1.25, 1.]
