"""Rerun the seven alpha>beta methods plus standalone PG (S/R) on the same 12 inputs."""
from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone
import hashlib
import json

import networkx as nx
import pandas as pd
import families_extended as families
import one_shot_pg
from ablate_hierarchical_madea_pg import SCREEN, HOLDOUT
from decentralized_potentialgame import run_pg_s, run_pg_r

SOURCE = families.ROOT / 'solutions/families-alpha-gt-beta-2026-10-01'


def run(output, seconds):
  started = datetime.now(timezone.utc)
  protocol = json.loads((SOURCE / 'protocol.json').read_text())
  config = json.loads((SOURCE / 'corrected_config.json').read_text())
  # The DP became opt-in after the reference campaign, which always used it.
  config['solver_options']['general']['use_dp'] = True
  hashes = pd.read_csv(SOURCE / 'runs.csv').groupby('case_id').input_hash.first()
  planned = [case for case in protocol['cases'] if case['id'] in SCREEN + HOLDOUT]
  assert len(planned) == 12

  def reference_instance(config, case, output):
    profile = '-'.join(f'{value:g}' for value in case['profile'])
    path = SOURCE / 'instances' / f"n{case['nodes']}-f{case['functions']}-s{case['seed']}-t{case['steps']}-l{profile}"
    assert hashlib.sha256((path / 'metadata.json').read_bytes()).hexdigest() == hashes[case['id']]
    data, traces, agents, graph = families.load_materialized_instance(path)
    assert nx.is_connected(graph) and nx.check_planarity(graph)[0]
    assert all(degree == 3 for _, degree in graph.degree())
    return path, data, traces, list(agents)

  families.cases = lambda: planned
  families.instance = reference_instance
  families.METHODS += [('one-shot-pg', one_shot_pg.run),
                       ('hierarchical-one-shot-pg', one_shot_pg.run_hierarchical),
                       ('hierarchical-madea-pg', families.madea_pg.run_hierarchical),
                       ('FaaS-MAPG-S', run_pg_s), ('FaaS-MAPG-R', run_pg_r)]
  families.run(output, started + timedelta(seconds=seconds), 0, len(planned), config=config)
  saved = json.loads((output / 'protocol.json').read_text())
  saved.update(instance_source=str((SOURCE / 'instances').relative_to(families.ROOT)),
    reference_results='experiments/madea_pg_compact/results_families_alpha_gt_beta_2026_10_01',
    extension_note='Stesse 12 istanze α > β della campagna del 1° ottobre (hash verificati). '
    'Tutti i metodi sono rieseguiti sul codice attuale, con DP locale attiva; si aggiungono '
    'FaaS-MAPG-S e FaaS-MAPG-R (PG da solo, ordine fisso e casuale), con al massimo 100 '
    'iterazioni e lo stesso TimeLimit max(30, 0,5 × N).')
  (output / 'protocol.json').write_text(json.dumps(saved, indent=2))
  runs = pd.read_csv(output / 'runs.csv')
  assert len(runs) == 12 * len(families.METHODS) and runs.status.eq('ok').all(), 'Incomplete campaign'


if __name__ == '__main__':
  parser = ArgumentParser(description=__doc__)
  parser.add_argument('--output', default='solutions/families-pg-standalone-2026-10-09')
  parser.add_argument('--seconds', type=float, default=3600.)
  args = parser.parse_args()
  run(families.ROOT / args.output, args.seconds)
