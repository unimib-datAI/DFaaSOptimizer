"""Rerun seven methods with alpha > beta on exact copies of shared planar inputs."""
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json
import shutil

import networkx as nx
import numpy as np
import pandas as pd
import families_extended as families
import one_shot_pg
from ablate_hierarchical_madea_pg import SCREEN, HOLDOUT
from remote_experiments.batch import Experiment
from remote_experiments.instances import materialize_instance, validate_instance


def correct_instance(source, target):
  metadata = validate_instance(source)
  original, *_ = families.load_materialized_instance(source)
  generation = deepcopy(metadata['generation'])
  generation['limits']['weights']['beta_multiplier'] = {'min': .1, 'max': .9}
  seed = generation.pop('generation_seed')
  config = dict(generation, seed=seed, instance_seed=seed)
  materialize_instance(Experiment(target.name, 'families-alpha-gt-beta',
                                  'faas-madea-pg', seed, {}, {}, config), target)
  corrected, *_ = families.load_materialized_instance(target)
  for key, value in original[None].items():
    if key not in {'beta', 'delta'}:
      assert corrected[None][key] == value, f'Physical input changed: {key}'
  assert (source/'graph.json').read_bytes() == (target/'graph.json').read_bytes()
  assert all(0 <= value < corrected[None]['alpha'][i, f]
             for (i, j, f), value in corrected[None]['beta'].items())
  for name in ('load_limits.json', 'input_requests_traces.json'):
    shutil.copyfile(source/name, target/name)
  new_metadata = json.loads((target/'metadata.json').read_text())
  new_metadata['alpha_beta_correction'] = dict(
    source_metadata_hash=hashlib.sha256((source/'metadata.json').read_bytes()).hexdigest(),
    source_instance=source.name, beta_multiplier={'min': .1, 'max': .9},
    changed_parameters=['beta', 'delta'],
  )
  if 'trace_profile' in metadata:
    new_metadata['trace_profile'] = metadata['trace_profile']
  new_metadata['files'] = {name: hashlib.sha256((target/name).read_bytes()).hexdigest()
                           for name in new_metadata['files']}
  (target/'metadata.json').write_text(json.dumps(new_metadata, indent=2))
  validate_instance(target)
  return target


def run(output, seconds):
  started = datetime.now(timezone.utc)
  source = families.ROOT/'solutions/families-five-methods-2026-10-01'
  original_protocol = json.loads((source/'protocol.json').read_text())
  shared = families.ROOT/original_protocol['extension_protocol']['instance_source']
  config = deepcopy(original_protocol['config'])
  config['limits']['weights']['beta_multiplier'] = {'min': .1, 'max': .9}
  output.mkdir(parents=True, exist_ok=True)
  (output/'corrected_config.json').write_text(json.dumps(config, indent=2))
  files = ['one_shot_pg.py', 'experiments/madea_pg_compact/rerun_alpha_gt_beta.py',
           'experiments/madea_pg_compact/families_report.py']
  files += [str(path.relative_to(families.ROOT))
            for path in sorted((families.ROOT/'hierarchical_auction').glob('*.py'))]
  code = {name: (families.ROOT/name).read_bytes() for name in files}
  hashes = pd.read_csv(source/'runs.csv').groupby('case_id').input_hash.first()
  available_cases = [case for case in original_protocol['cases'] if case['id'] in hashes.index]
  audit = []
  for case in available_cases:
    profile = '-'.join(f'{value:g}' for value in case['profile'])
    name = f"n{case['nodes']}-f{case['functions']}-s{case['seed']}-t{case['steps']}-l{profile}"
    old = shared/'instances'/name
    assert hashlib.sha256((old/'metadata.json').read_bytes()).hexdigest() == hashes[case['id']]
    target = correct_instance(old, output/'instances'/name)
    ratios = []
    for path in (old, target):
      data, *_ = families.load_materialized_instance(path)
      values = data[None]
      ratios.append([b/values['alpha'][i, f] for (i, j, f), b in values['beta'].items()
                     if values['neighborhood'][i, j]])
    audit.append(dict(case_id=case['id'], instance=name,
      original_hash=hashes[case['id']], corrected_hash=hashlib.sha256((target/'metadata.json').read_bytes()).hexdigest(),
      active_coefficients=len(ratios[0]), violations_before=sum(b >= 1 for b in ratios[0]),
      violations_after=sum(b >= 1 for b in ratios[1]),
      ratio_min=min(ratios[1]), ratio_max=max(ratios[1]), physical_inputs_identical=True))
  pd.DataFrame(audit).to_csv(output/'instance_audit.csv', index=False)
  print('CORRECTED ' + json.dumps(dict(instances=len(audit),
        violations_before=sum(row['violations_before'] for row in audit),
        violations_after=sum(row['violations_after'] for row in audit))), flush=True)
  planned = [case for case in available_cases if case['id'] in SCREEN+HOLDOUT]
  assert len(planned) == 12

  def corrected_instance(config, case, output):
    profile = '-'.join(f'{value:g}' for value in case['profile'])
    name = f"n{case['nodes']}-f{case['functions']}-s{case['seed']}-t{case['steps']}-l{profile}"
    path = output/'instances'/name
    data, traces, agents, graph = families.load_materialized_instance(path)
    assert nx.is_connected(graph) and nx.check_planarity(graph)[0]
    assert all(degree == 3 for _, degree in graph.degree())
    return path, data, traces, list(agents)

  families.cases = lambda: planned
  families.instance = corrected_instance
  families.METHODS += [('one-shot-pg', one_shot_pg.run),
                      ('hierarchical-one-shot-pg', one_shot_pg.run_hierarchical),
                      ('hierarchical-madea-pg', families.madea_pg.run_hierarchical)]
  families.run(output, started+timedelta(seconds=seconds), 0, len(planned), config=config)
  protocol = json.loads((output/'protocol.json').read_text())
  protocol.update(instance_source=str((output/'instances').relative_to(families.ROOT)),
    original_instance_source=str(shared.relative_to(families.ROOT)),
    correction='beta/alpha uniform in [0.1,0.9]; delta regenerated coherently; all other input data identical',
    extension_note='Tutte le sette colonne sono rieseguite sulle nuove istanze con α > β. '
    'Grafi, α, γ, risorse e tracce originali sono identici. β/α è in [0,1; 0,9]; '
    'δ è ricalcolato dal generatore perché deriva da β. Sono corrette 35 istanze '
    'e misurati 12 casi selezionati, con seed 7, 42, 99 e 2026. '
    'Non confrontare direttamente questi welfare con quelli della precedente funzione obiettivo.')
  for name, contents in code.items():
    assert (families.ROOT/name).read_bytes() == contents
    protocol['source_hashes'][name] = hashlib.sha256(contents).hexdigest()
    (output/(name.replace('/', '__')+'.txt')).write_bytes(contents)
  (output/'protocol.json').write_text(json.dumps(protocol, indent=2))
  runs = pd.read_csv(output/'runs.csv')
  assert len(runs) == 84 and runs.status.eq('ok').all(), 'Not all 12 seven-method cases completed'
  steps = pd.read_csv(output/'timesteps.csv')
  pg = steps[steps.pg_before.notna()]
  np.testing.assert_array_less(pg.pg_before.to_numpy()-1e-6, pg.welfare.to_numpy())
  print('VERIFIED ' + json.dumps(dict(runs=len(runs), timesteps=len(steps))), flush=True)


if __name__ == '__main__':
  parser = ArgumentParser(description=__doc__)
  parser.add_argument('--output', default='solutions/families-alpha-gt-beta-2026-10-01')
  parser.add_argument('--seconds', type=float, default=1800.)
  args = parser.parse_args()
  run(families.ROOT/args.output, args.seconds)
