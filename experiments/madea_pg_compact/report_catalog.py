"""Report of the catalogue campaign: gap to the centralized reference, ranks, head-to-head."""
from argparse import ArgumentParser
from pathlib import Path
import json
import re

import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root: utils.run_log
from utils import run_log  # noqa: E402

log = run_log.logger(__name__)

PAIRS = [('MADEA2', 'MADEA'), ('MADEA2-one-shot', 'MADEA-one-shot'), ('one-shot2-pg', 'one-shot-pg'),
         ('FaaS-MAFW', 'MADEA'), ('MAFW-PG', 'one-shot-pg'), ('one-shot2-pg', 'hierarchical-one-shot-pg'),
         ('MAFW-PG', 'hierarchical-one-shot-pg'),
         ('MADEA2', 'MADEA2-b'), ('MADEA2-b', 'MADEA'), ('MADEA2-b-one-shot', 'MADEA-one-shot'),
         ('one-shot2-b-pg', 'one-shot-pg'), ('FaaS-MAFW', 'FaaS-MAFW-b'), ('FaaS-MAFW-b', 'MADEA'),
         ('MAFW-b-PG', 'one-shot-pg')]


def load(folder):
  cases = {x['id']: x for x in json.loads((folder / 'sample.json').read_text())}
  rows = []
  for f in folder.glob('runs/*/*/runs.csv'):
    r = pd.read_csv(f)
    for _, x in r.iterrows():
      method = f.parent.name
      case = cases.get(f.parent.parent.name)
      if case is None:
        continue
      rounds = None
      for tc in f.parent.glob('runs/*/*/*/termination_condition.csv'):
        its = [int(m.group(1)) + 1 for line in pd.read_csv(tc).iloc[:, -1].astype(str)
               for m in [re.search(r'it: (\d+)', line)] if m]
        rounds = sum(its) if its else None
      rows.append(dict(case_id=case['id'], family=case['family'], group=case['group'], nodes=case['nodes'],
                       functions=case['functions'], beta=case['beta'], method=method,
                       status=x.get('status'), welfare=x.get('welfare_sum', x.get('welfare')),
                       wall=x.get('wall_seconds'), bound=x.get('bound'), rounds=rounds,
                       termination=x.get('termination')))
  return pd.DataFrame(rows)


def main(folder, output):
  df = load(folder)
  ok = df[df.status.eq('ok')]
  W = ok.pivot(index='case_id', columns='method', values='welfare')
  info = df.drop_duplicates('case_id').set_index('case_id')[['family', 'group', 'nodes', 'functions', 'beta']]
  cent = df[df.method.eq('centralized')].set_index('case_id')
  lines = ['# Campagna sul catalogo edge-datasets', '']
  counts = df.groupby('method').status.value_counts().unstack(fill_value=0)
  lines += ['## Run completate', '', counts.to_markdown(), '']
  if 'centralized' in W:
    ref = W['centralized']
    gap = (W.drop(columns='centralized').rsub(ref, axis=0)).div(ref.abs(), axis=0) * 100
    bound = cent['bound'].reindex(W.index)
    gapb = (W.rsub(bound, axis=0)).div(bound.abs(), axis=0) * 100
    summ = pd.DataFrame({'casi': gap.notna().sum(), 'distanza media dal centralizzato %': gap.mean(),
                         'mediana %': gap.median(), 'max %': gap.max(),
                         'distanza media dal bound %': gapb.drop(columns='centralized', errors='ignore').mean()})
    summ.loc['centralized'] = [ref.notna().sum(), 0, 0, 0, gapb['centralized'].mean()]
    lines += ['## Distanza dalla soluzione centralizzata (positivo = peggio del centralizzato)', '',
              'Centralizzato: HiGHS con limite 60 s (120 s per n160); il bound duale dà la distanza massima dall\'ottimo.', '',
              summ.sort_values('distanza media dal centralizzato %').round(3).to_markdown(), '']
    term = cent['termination'].value_counts()
    lines += ['Stato del centralizzato: ' + ', '.join(f'{k}: {v}' for k, v in term.items()), '']
    lines += ['## Per famiglia (distanza media dal centralizzato, %)', '',
              gap.join(info).groupby('family').mean(numeric_only=True).drop(columns=['nodes', 'functions'], errors='ignore')
              .round(2).T.to_markdown(), '']
  rank = W.drop(columns='centralized', errors='ignore')
  # methods with at least 80% of the most-run method's cases, on the cases all of them completed
  counts_ok = rank.notna().sum()
  full = rank.loc[:, counts_ok >= 0.8 * counts_ok.max()].dropna()
  if len(full):
    r = full.rank(axis=1, ascending=False, method='average')
    wins = full.eq(full.max(axis=1), axis=0).sum()
    lines += [f'## Classifica tra i decentralizzati ({len(full)} casi completi per tutti i metodi mostrati)', '',
              pd.DataFrame({'posizione media': r.mean(), 'vittorie': wins}).sort_values('posizione media').round(2).to_markdown(), '']
  every = rank.dropna(axis=1, how='all').dropna()
  if len(every) and every.shape[1] > full.shape[1]:
    r = every.rank(axis=1, ascending=False, method='average')
    wins = every.eq(every.max(axis=1), axis=0).sum()
    dist = None
    if 'centralized' in W:
      ref = W['centralized'].reindex(every.index)
      dist = (every.rsub(ref, axis=0)).div(ref.abs(), axis=0).mul(100).mean()
    table = pd.DataFrame({'posizione media': r.mean(), 'vittorie': wins})
    if dist is not None:
      table['distanza media dal centralizzato %'] = dist
    lines += [f'## Classifica con tutti i metodi, varianti "-b" comprese ({len(every)} casi completi per tutti)', '',
              table.sort_values('posizione media').round(3).to_markdown(), '']
  rows = []
  for a, b in PAIRS:
    if a in W and b in W:
      d = ((W[a] - W[b]) / W[b].abs() * 100).dropna()
      if len(d):
        rows.append(dict(confronto=f'{a} vs {b}', casi=len(d), vince=int((d > 1e-9).sum()),
                         pari=int((d.abs() <= 1e-9).sum()), perde=int((d < -1e-9).sum()),
                         diff_media=d.mean(), min=d.min(), max=d.max()))
  if rows:
    lines += ['## Confronti diretti (differenza % di welfare)', '', pd.DataFrame(rows).round(3).to_markdown(index=False), '']
  tm = ok.groupby('method')[['wall', 'rounds']].median()
  lines += ['## Tempo di parete e round per timestep (mediane)', '', tm.round(2).to_markdown(), '']
  output.mkdir(parents=True, exist_ok=True)
  (output / 'comparison.md').write_text('\n'.join(lines))
  df.to_csv(output / 'runs.csv', index=False)
  W.to_csv(output / 'welfare.csv')
  log.info("%s", '\n'.join(lines))


if __name__ == '__main__':
  with run_log.open_run_log(None, 1):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=Path('solutions/catalog-sample-2026-10-10'))
    parser.add_argument('--output', type=Path, default=Path('experiments/madea_pg_compact/results_catalog_2026_10_11'))
    args = parser.parse_args()
    main(args.folder, args.output)
