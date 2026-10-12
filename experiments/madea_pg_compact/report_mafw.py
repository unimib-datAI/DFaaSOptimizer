"""Overall table of all methods plus FaaS-MAFW on the two shared 12-case campaigns of 2026-10-09."""
from argparse import ArgumentParser
from pathlib import Path
import json

import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root: utils.run_log
from utils import run_log  # noqa: E402

log = run_log.logger(__name__)

ORDER = ['MADEA', 'MADEA-PG', 'MADEA-one-shot', 'one-shot-pg', 'hierarchical-madea', 'hierarchical-madea-pg',
         'hierarchical-one-shot', 'hierarchical-one-shot-pg', 'PLASMA-Welfare', 'FaaS-MAPG-S', 'FaaS-MAPG-R',
         'FaaS-MAFW', 'MAFW-PG', 'hierarchical-MAFW', 'hierarchical-MAFW-PG',
         'MADEA2', 'MADEA2-one-shot', 'one-shot2-pg']


def load(reference, parts_root, tags):
  """Reference runs plus every MAFW variant stored under ``parts_root/<tag>`` (tags joined by '+')."""
  runs = pd.read_csv(reference / 'runs.csv')
  new, steps = [], [pd.read_csv(reference / 'timesteps.csv')]
  for tag in tags.split('+'):
    done = [p for p in sorted((parts_root / tag).glob('*-r0')) if (p / 'finished.json').exists()]
    part = pd.concat([pd.read_csv(p / 'runs.csv') for p in done])
    assert part.status.eq('ok').all() and len(part) == part.method.nunique() * runs.case_id.nunique(), (tag, len(part))
    new.append(part)
    steps += [pd.read_csv(p / 'timesteps.csv') for p in done]
  merged = pd.concat([runs, *new], ignore_index=True)
  assert not merged.duplicated(['case_id', 'method']).any()
  # same instance for every method
  assert merged.groupby('case_id').input_hash.nunique().eq(1).all()
  return merged, pd.concat(steps)


def summarize(runs):
  order = [m for m in ORDER if m in set(runs.method)]
  welfare = runs.pivot(index='case_id', columns='method', values='welfare_sum')[order]
  wall = runs.pivot(index='case_id', columns='method', values='wall_seconds')[order]
  phase = runs.groupby('case_id').phase.first()
  gain = welfare.sub(welfare['MADEA'], axis=0).div(welfare['MADEA'].abs(), axis=0) * 100
  ratio = wall.div(wall['MADEA'], axis=0)
  out = pd.DataFrame({
    'std': gain[phase == 'static'].mean(), 'stress': gain[phase == 'stress'].mean(),
    'temp': gain[phase == 'temporal'].mean(), 'tempo': ratio.median(),
    'wall_media_s': wall.mean()})
  rank = welfare[phase == 'static'].rank(axis=1, ascending=False, method='average').mean()
  wins = welfare[phase == 'static'].eq(welfare[phase == 'static'].max(axis=1), axis=0).sum()
  out['rank_std'] = rank
  out['vittorie_std'] = wins
  return out, welfare, gain


def fmt(out):
  lines = ['| Metodo | std | stress | temp | tempo (mediana vs MADEA) | tempo medio [s] | posizione media (std) | vittorie (std) |',
           '| --- | --- | --- | --- | --- | --- | --- | --- |']
  for m, r in out.iterrows():
    lines.append(f"| {m} | {r['std']:+.2f}% | {r['stress']:+.2f}% | {r['temp']:+.2f}% | {r['tempo']:.2f}× | "
                 f"{r['wall_media_s']:.1f} | {r['rank_std']:.2f} | {int(r['vittorie_std'])} |")
  return '\n'.join(lines)


if __name__ == '__main__':
  with run_log.open_run_log(None, 1):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='worktree with the 09/10 reference campaigns')
    parser.add_argument('--parts', required=True, help='folder with mafw-parts/{ab,pu}')
    parser.add_argument('--output', required=True)
    parser.add_argument('--campaign', action='append', help='title=reference_dir,parts_tag (repeatable; replaces the default pair)')
    parser.add_argument('--previous', help='folder with summary_{ab,pu}.csv of an earlier MAFW version')
    parser.add_argument('--notes', help='markdown file appended to the report')
    args = parser.parse_args()
    root, parts, output = Path(args.root), Path(args.parts), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    campaigns = {'alpha-gt-beta (grafi originali)': ('solutions/families-pg-standalone-hier-2026-10-09', 'ab'),
                 'planar-uniform (grafi ricampionati)': ('solutions/families-planar-uniform-2026-10-09', 'pu')}
    if args.campaign:
      campaigns = {c.split('=')[0]: tuple(c.split('=', 1)[1].split(',')) for c in args.campaign}
    sections, tables = [], {}
    for title, (ref, tag) in campaigns.items():
      runs, steps = load(root / ref, parts, tag)
      tag = tag.split('+')[0]
      out, welfare, gain = summarize(runs)
      tables[tag] = out
      runs.to_csv(output / f'runs_{tag}.csv', index=False)
      steps.to_csv(output / f'timesteps_{tag}.csv', index=False)
      out.round(4).to_csv(output / f'summary_{tag}.csv')
      gain.round(4).to_csv(output / f'gain_vs_madea_pct_{tag}.csv')
      welfare.round(4).to_csv(output / f'welfare_{tag}.csv')
      sections.append(f'## {title}\n\n' + fmt(out))
    if not {'ab', 'pu'} <= set(tables):
      text = '# ' + ', '.join(campaigns) + '\n\n' + '\n\n'.join(sections) + '\n'
      (output / 'comparison.md').write_text(text + (Path(args.notes).read_text() if args.notes else ''))
      log.info("%s", text)
      raise SystemExit
    both = pd.DataFrame({'std (orig.)': tables['ab']['std'], 'stress (orig.)': tables['ab']['stress'],
                         'temp (orig.)': tables['ab']['temp'], 'std (ric.)': tables['pu']['std'],
                         'stress (ric.)': tables['pu']['stress'], 'temp (ric.)': tables['pu']['temp'],
                         'tempo (orig.)': tables['ab']['tempo'], 'tempo (ric.)': tables['pu']['tempo']})
    rows = ['| Metodo | std orig. | stress orig. | temp orig. | tempo orig. | std ric. | stress ric. | temp ric. | tempo ric. |',
            '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for m, r in both.iterrows():
      rows.append(f"| {m} | {r['std (orig.)']:+.2f}% | {r['stress (orig.)']:+.2f}% | {r['temp (orig.)']:+.2f}% | "
                  f"{r['tempo (orig.)']:.2f}× | {r['std (ric.)']:+.2f}% | {r['stress (ric.)']:+.2f}% | "
                  f"{r['temp (ric.)']:+.2f}% | {r['tempo (ric.)']:.2f}× |")
    text = ('# Tabella complessiva con FaaS-MAFW\n\nStesse 12 istanze per campagna, stessa configurazione e stesso codice dei '
            'metodi di riferimento (hash di input verificati). Guadagno medio del welfare rispetto a MADEA nella fase '
            '(std = 8 casi statici, stress = 2, temp = 2); tempo = rapporto mediano dei tempi di parete rispetto a MADEA.\n\n'
            '## Riepilogo (entrambe le topologie)\n\n' + '\n'.join(rows) + '\n\n' + '\n\n'.join(sections) + '\n')
    if args.previous:
      prev = {tag: pd.read_csv(Path(args.previous) / f'summary_{tag}.csv', index_col=0).loc['FaaS-MAFW'] for tag in ('ab', 'pu')}
      lines = ['## FaaS-MAFW prima e dopo le correzioni', '',
               '| Versione | std orig. | stress orig. | temp orig. | std ric. | stress ric. | temp ric. | pos. media std orig. | pos. media std ric. |',
               '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
      for name, src in (('prima', prev), ('dopo', {tag: tables[tag].loc['FaaS-MAFW'] for tag in ('ab', 'pu')})):
        a, b = src['ab'], src['pu']
        lines.append(f"| {name} | {a['std']:+.2f}% | {a['stress']:+.2f}% | {a['temp']:+.2f}% | {b['std']:+.2f}% | "
                     f"{b['stress']:+.2f}% | {b['temp']:+.2f}% | {a['rank_std']:.2f} | {b['rank_std']:.2f} |")
      text += '\n' + '\n'.join(lines) + '\n'
    if args.notes:
      text += '\n' + Path(args.notes).read_text()
    (output / 'comparison.md').write_text(text)
    log.info("%s", text)
