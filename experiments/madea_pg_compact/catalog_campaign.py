"""Static comparison (step 0) on a seeded sample of the edge-dataset catalogue.

``--sample`` draws the instances and writes ``sample.json``; ``--case`` and
``--method`` run one method on one sampled instance through the validated
families harness (one run per call, so long sweeps can be resumed).

Sample (seed ``--seed``): for each family, two instances at the middle size
(n80; n30 for the published 30-node families; the ether-urban topology
closest to 80 nodes), Huawei traces with Nf = 5, one with random beta
(trace seed 1) and one with latency-based beta (``-blat``, trace seed 2), the
topology drawn at random; plus a scale check of four n160 instances with
Huawei Nf = 10.  A second stratum (``extend_sample``) adds four instances per
family (Nf = 10 at the middle size, and the smallest size) and n160 scale
checks for the other families.  Every catalogue instance has beta < alpha by
construction.
"""
from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import random
import re

import networkx as nx
import families_extended as families
import one_shot_pg
from extend_frankwolfe import ALL_METHODS

REFERENCE_PROTOCOL = 'solutions/families-pg-standalone-hier-2026-10-09/protocol.json'
METHODS = {**dict(families.METHODS), 'one-shot-pg': one_shot_pg.run,
           'hierarchical-one-shot-pg': one_shot_pg.run_hierarchical, **ALL_METHODS}
SCALE_FAMILIES = ('synthetic-planar-cubic', 'zhang-regional', 'data7-pisa', 'wondernetwork')
NAME = re.compile(r'^(?P<topo>.+)__(?P<src>huawei|azure)-d1-f(?P<f>\d+)-n(?P<n>\d+)-.*-s(?P<s>\d+)(?P<blat>-blat)?$')


def _parse(path):
  m = NAME.match(path.name)
  return None if m is None else dict(path=str(path), name=path.name, topology=m['topo'], source=m['src'],
    functions=int(m['f']), nodes=int(m['n']), seed=int(m['s']), beta='latency' if m['blat'] else 'random')


def sample(catalog, seed):
  rng = random.Random(seed)
  chosen = []
  for family in sorted(p.name for p in catalog.iterdir() if (p / 'instances').is_dir()):
    items = [x for x in map(_parse, sorted((catalog / family / 'instances').iterdir())) if x and x['source'] == 'huawei']
    sizes = sorted({x['nodes'] for x in items})
    middle = min(sizes, key=lambda n: abs(n - 80))
    for trace_seed, beta in ((1, 'random'), (2, 'latency')):
      pool = [x for x in items if x['nodes'] == middle and x['functions'] == 5 and x['beta'] == beta]
      # prefer the planned trace seed; some topologies exist with one seed only
      pool = [x for x in pool if x['seed'] == trace_seed] or pool
      chosen.append(dict(rng.choice(pool), family=family, group='main'))
    if family in SCALE_FAMILIES:
      pool = [x for x in items if x['nodes'] == max(sizes) and x['functions'] == 10 and x['beta'] == 'random']
      pool = [x for x in pool if x['seed'] == 1] or pool
      chosen.append(dict(rng.choice(pool), family=family, group='scale'))
  chosen += extend_sample(catalog, seed + 1, {x['name'] for x in chosen})
  for x in chosen:
    x.update(id=x['name'], phase='static', repeat=0, profile=[1.0], steps=1)
  return chosen


def extend_sample(catalog, seed, taken):
  """Second stratum (drawn after the first, so the first is unchanged).

  Per family: middle size with Nf = 10 (random beta, trace seed 3; latency
  beta, seed 4) and smallest size (n40 where available) with Nf = 5 and 10
  (random and latency beta, seed 5); one n160 Nf = 10 instance for the
  families with a 160-node size not already in the scale group.
  """
  rng = random.Random(seed)
  extra = []

  def draw(items, family, group, nodes, functions, beta, trace_seed):
    pool = [x for x in items if x['nodes'] == nodes and x['functions'] == functions
            and x['beta'] == beta and x['name'] not in taken]
    pool = [x for x in pool if x['seed'] == trace_seed] or pool
    if pool:
      pick = dict(rng.choice(pool), family=family, group=group)
      taken.add(pick['name'])
      extra.append(pick)

  for family in sorted(p.name for p in catalog.iterdir() if (p / 'instances').is_dir()):
    items = [x for x in map(_parse, sorted((catalog / family / 'instances').iterdir())) if x and x['source'] == 'huawei']
    sizes = sorted({x['nodes'] for x in items})
    middle = min(sizes, key=lambda n: abs(n - 80))
    small = min(sizes, key=lambda n: abs(n - 40))
    draw(items, family, 'main', middle, 10, 'random', 3)
    draw(items, family, 'main', middle, 10, 'latency', 4)
    draw(items, family, 'small', small, 5, 'random', 5)
    draw(items, family, 'small', small, 10, 'latency', 5)
    if family not in SCALE_FAMILIES and max(sizes) >= 160:
      draw(items, family, 'scale', max(sizes), 10, 'random', 1)
  return extra


def run_one(case, method, output, config):
  def instance(config, case, output):
    path = Path(case['path'])
    data, traces, agents, graph = families.load_materialized_instance(path)
    assert nx.is_connected(graph)
    return path, data, traces, list(agents)

  families.cases = lambda: [case]
  families.instance = instance
  families.METHODS = [(method, METHODS[method])]
  families.run(output, datetime.now(timezone.utc) + timedelta(hours=1), 0, 1, config=config)


if __name__ == '__main__':
  parser = ArgumentParser(description=__doc__)
  parser.add_argument('--catalog', type=Path, required=True)
  parser.add_argument('--reference-root', type=Path, required=True)
  parser.add_argument('--folder', default='solutions/catalog-sample-2026-10-10')
  parser.add_argument('--seed', type=int, default=20261010)
  parser.add_argument('--sample', action='store_true')
  parser.add_argument('--case')
  parser.add_argument('--method', choices=sorted(METHODS))
  args = parser.parse_args()
  folder = families.ROOT / args.folder
  folder.mkdir(parents=True, exist_ok=True)
  if args.sample:
    cases = sample(args.catalog, args.seed)
    (folder / 'sample.json').write_text(json.dumps(cases, indent=2))
    for x in cases:
      print(x['group'], x['family'], x['name'])
  else:
    cases = {x['id']: x for x in json.loads((folder / 'sample.json').read_text())}
    config = json.loads((args.reference_root / REFERENCE_PROTOCOL).read_text())['config']
    config['solver_options']['frankwolfe'] = dict(iterations=400, kappa=10.0, smoothing=1.0, max_rounds=20)
    out = folder / 'runs' / args.case / args.method
    if not (out / 'finished.json').exists():
      run_one(cases[args.case], args.method, out, config)
