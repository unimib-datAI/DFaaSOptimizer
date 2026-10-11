"""Run FaaS-MAFW on the shared instances of an existing families campaign and merge the results."""
from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json

import networkx as nx
import pandas as pd
import families_extended as families
from decentralized_frankwolfe import run as run_frankwolfe
import hierarchical_frankwolfe as hfw
import madea2

def _one_shot2(config, parallelism, log_on_file=False, disable_plotting=False):
  return madea2.run_one_shot(config, parallelism, log_on_file, disable_plotting)


def _one_shot2_b(config, parallelism, log_on_file=False, disable_plotting=False):
  return madea2.run_one_shot_b(config, parallelism, log_on_file, disable_plotting)


ALL_METHODS = {'FaaS-MAFW': run_frankwolfe, 'MAFW-PG': hfw.run_flat_pg,
               'hierarchical-MAFW': hfw.run_hierarchical,
               'hierarchical-MAFW-PG': hfw.run_hierarchical_pg,
               'MADEA2': madea2.run, 'MADEA2-one-shot': _one_shot2,
               'one-shot2-pg': madea2.run_one_shot_pg,
               'MADEA2-b': madea2.run_b, 'MADEA2-b-one-shot': _one_shot2_b,
               'one-shot2-b-pg': madea2.run_one_shot_b_pg,
               'FaaS-MAFW-b': hfw.run_mafw_b, 'MAFW-b-PG': hfw.run_flat_pg_b}
METHODS = [('FaaS-MAFW', run_frankwolfe)]


def run(reference, output, merged, seconds, only=None, fw=None, instance_dir=None, general=None):
  """`reference` is a campaign folder (possibly in another worktree) with protocol.json/runs.csv."""
  protocol = json.loads((reference / 'protocol.json').read_text())
  runs = pd.read_csv(reference / 'runs.csv')
  assert runs.status.eq('ok').all() and not {name for name, _ in METHODS} & set(runs.method)
  hashes = runs.groupby('case_id').input_hash.first()
  repo = reference.parents[1]  # <worktree>/solutions/<campaign>
  instances = Path(instance_dir) if instance_dir else repo / protocol['instance_source']
  # only the cases the reference campaign actually completed
  planned = [c for c in protocol['cases'] if c['id'] in hashes.index and (not only or c['id'] in only)]

  def shared_instance(config, case, output):
    profile = '-'.join(f'{value:g}' for value in case['profile'])
    path = instances / f"n{case['nodes']}-f{case['functions']}-s{case['seed']}-t{case['steps']}-l{profile}"
    assert hashlib.sha256((path / 'metadata.json').read_bytes()).hexdigest() == hashes[case['id']]
    data, traces, agents, graph = families.load_materialized_instance(path)
    assert nx.is_connected(graph) and nx.check_planarity(graph)[0]
    return path, data, traces, list(agents)

  config = protocol['config']
  config['solver_options']['frankwolfe'] = dict(iterations=400, kappa=1.0, smoothing=1.0, max_rounds=20)
  config['solver_options']['frankwolfe'].update(fw or {})
  config['solver_options']['general'].update(general or {})
  families.cases = lambda: planned
  families.instance = shared_instance
  families.METHODS = METHODS
  deadline = datetime.now(timezone.utc) + timedelta(seconds=seconds)
  if not (output / 'finished.json').exists():
    families.run(output, deadline, 0, len(planned), config=config)
  new_runs, new_steps = pd.read_csv(output / 'runs.csv'), pd.read_csv(output / 'timesteps.csv')
  merged.mkdir(parents=True, exist_ok=True)
  pd.concat([runs[runs.case_id.isin([c['id'] for c in planned])], new_runs]).to_csv(merged / 'runs.csv', index=False)
  pd.concat([pd.read_csv(reference / 'timesteps.csv'), new_steps]).to_csv(merged / 'timesteps.csv', index=False)
  protocol['parallelism'].update({name: 0 for name, _ in METHODS})
  protocol['extension_note'] = protocol.get('extension_note', '') + (
    ' FaaS-MAFW (Frank-Wolfe distribuito con repliche rilassate e arrotondamento) misurato in una sessione '
    'separata sulle stesse istanze (hash di input verificati), con gurobipy a licenza ristretta (use_dp=true); opzioni frankwolfe in config.')
  (merged / 'protocol.json').write_text(json.dumps(protocol, indent=2))


if __name__ == '__main__':
  parser = ArgumentParser(description=__doc__)
  parser.add_argument('--reference', required=True)
  parser.add_argument('--output', required=True)
  parser.add_argument('--merged', required=True)
  parser.add_argument('--seconds', type=float, default=3000.)
  parser.add_argument('--only', nargs='*')
  parser.add_argument('--methods', nargs='*', choices=sorted(ALL_METHODS), default=['FaaS-MAFW'])
  parser.add_argument('--instances', help='folder of materialized instances (default: protocol instance_source)')
  parser.add_argument('--general', type=json.loads, default=None, help='JSON overrides of the general solver options (e.g. use_dp)')
  parser.add_argument('--fw', type=json.loads, default=None, help='JSON overrides of the frankwolfe options')
  args = parser.parse_args()
  METHODS[:] = [(name, ALL_METHODS[name]) for name in args.methods]
  run(Path(args.reference).resolve(), families.ROOT / args.output, families.ROOT / args.merged,
      args.seconds, set(args.only) if args.only else None, args.fw, args.instances, args.general)
