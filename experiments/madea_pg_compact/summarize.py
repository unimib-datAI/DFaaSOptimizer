"""Summarize paired runs without treating time-censored searches as equal work."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


def summarize(output):
  raw = pd.read_csv(output / 'results.csv')
  valid = raw[raw.status == 'ok']
  assert valid.feasible.all()
  second = valid.dropna(subset=['prefix_equal'])
  assert second.prefix_equal.all(), 'decision prefix mismatch'
  keys = ['case', 'nodes', 'functions', 'seed', 'load_scale', 'degree', 'initial_backend']
  full = valid[valid.proposal == 'full']
  compact = valid[valid.proposal == 'compact']
  pairs = full.merge(compact, on=keys, suffixes=('_full', '_compact'))
  assert len(second) == len(pairs), 'missing prefix verification for a complete pair'
  compact = compact.merge(pairs[keys], on=keys)
  pairs['pg_throughput_speedup'] = (pairs.pg_seconds_full / pairs.pg_proposals_full) / (pairs.pg_seconds_compact / pairs.pg_proposals_compact)
  pairs['wall_speedup'] = pairs.wall_seconds_full / pairs.wall_seconds_compact
  pairs['welfare_gain_pct'] = 100 * (pairs.welfare_compact - pairs.welfare_full) / pairs.welfare_full.abs()
  pairs['uncensored'] = (pairs.pg_reason_full != 'time budget exhausted') & (pairs.pg_reason_compact != 'time budget exhausted')
  pairs['compact_within_lower_cap'] = pairs.pg_seconds_compact <= pairs[['pg_budget_full', 'pg_budget_compact']].min(axis=1)
  assert (pairs.welfare_gain_pct >= -1e-6).all(), 'compact prefix extension reduced welfare'
  clean = pairs[pairs.uncensored]
  assert (clean.final_flow_full == clean.final_flow_compact).all()
  assert (clean.final_replicas_full == clean.final_replicas_compact).all()
  hybrid = compact[compact.initial_backend == 'milp'].merge(compact[compact.initial_backend == 'native'], on=keys[:-1], suffixes=('_milp', '_native'))
  hybrid['hybrid_welfare_gain_pct'] = 100 * (hybrid.welfare_milp - hybrid.welfare_native) / hybrid.welfare_native.abs()
  hybrid['hybrid_wall_ratio'] = hybrid.wall_seconds_milp / hybrid.wall_seconds_native
  summary = pairs.groupby(['nodes', 'functions']).agg(
    pairs=('case', 'count'), pg_throughput_median=('pg_throughput_speedup', 'median'),
    total_wall_median=('wall_speedup', 'median'), welfare_gain_mean_pct=('welfare_gain_pct', 'mean'),
    welfare_gain_max_pct=('welfare_gain_pct', 'max'), uncensored_pairs=('uncensored', 'sum'),
  ).reset_index()
  metrics = dict(runs=len(raw), cases=raw.case.nunique(), pairs=len(pairs),
                 all_feasible=bool(valid.feasible.all()), all_prefixes_equal=bool(second.prefix_equal.all()),
                 successful_runs=len(valid), unpaired_ok_runs=len(valid) - 2 * len(pairs),
                 unsuccessful_runs=len(raw) - len(valid),
                 compact_within_lower_paired_cap=int(pairs.compact_within_lower_cap.sum()),
                 prefix_checks=len(second), uncensored_pairs=len(clean),
                 full_budget_exhausted=int((full.pg_reason == 'time budget exhausted').sum()),
                 compact_budget_exhausted=int((compact.pg_reason == 'time budget exhausted').sum()),
                 fallbacks=int(valid.fallbacks.sum()),
                 pg_throughput_median=float(pairs.pg_throughput_speedup.median()),
                 total_wall_median=float(pairs.wall_speedup.median()),
                 welfare_improved=int((pairs.welfare_gain_pct > 1e-6).sum()),
                 welfare_equal=int((pairs.welfare_gain_pct.abs() <= 1e-6).sum()),
                 welfare_worse=int((pairs.welfare_gain_pct < -1e-6).sum()),
                 welfare_gain_mean_pct=float(pairs.welfare_gain_pct.mean()),
                 welfare_gain_max_pct=float(pairs.welfare_gain_pct.max()),
                 hybrid_better=int((hybrid.hybrid_welfare_gain_pct > 1e-6).sum()),
                 hybrid_equal=int((hybrid.hybrid_welfare_gain_pct.abs() <= 1e-6).sum()),
                 hybrid_worse=int((hybrid.hybrid_welfare_gain_pct < -1e-6).sum()),
                 hybrid_mean_gain_pct=float(hybrid.hybrid_welfare_gain_pct.mean()),
                 hybrid_min_gain_pct=float(hybrid.hybrid_welfare_gain_pct.min()),
                 hybrid_max_gain_pct=float(hybrid.hybrid_welfare_gain_pct.max()))
  pairs.to_csv(output / 'pairs.csv', index=False)
  hybrid.to_csv(output / 'hybrid.csv', index=False)
  summary.to_csv(output / 'summary.csv', index=False)
  (output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
  html = f'''<!doctype html><html lang="it"><meta charset="utf-8"><title>MADEA-PG planare</title>
<style>body{{font:16px system-ui;margin:2rem;max-width:1400px}}table{{border-collapse:collapse;font-size:14px}}td,th{{border:1px solid #ddd;padding:6px}}th{{background:#eee}}</style>
<h1>MADEA-PG: proposta compatta, istanze planari</h1>
<p>{metrics['runs']} esecuzioni, {metrics['cases']} istanze, {metrics['pairs']} confronti appaiati.
Le {metrics['successful_runs']} soluzioni completate sono ammissibili e intere.
Tutti i prefissi delle {metrics['prefix_checks']} coppie verificate coincidono.
{metrics['unpaired_ok_runs']} esecuzioni completate ma non appaiate e {metrics['unsuccessful_runs']} esecuzioni interrotte
sono escluse dalle statistiche appaiate.
Nei {metrics['uncensored_pairs']} confronti senza arresto temporale coincidono anche tutte le allocazioni finali.</p>
<p>Il throughput PG confronta secondi per proposta: evita di confondere il tempo di una ricerca interrotta con quello di una ricerca completata.
Il tempo totale include inizializzazione locale, aste, raffinamento e salvataggi.
La variante compatta migliora il welfare in {metrics['welfare_improved']} coppie, lo conserva in {metrics['welfare_equal']} e lo riduce in {metrics['welfare_worse']}.</p>
<h2>Dimensioni</h2>{summary.to_html(index=False, float_format=lambda a:f'{a:.2f}')}
<h2>Inizializzazione Gurobi locale</h2><p>Con proposte compatte, migliora il welfare in {metrics['hybrid_better']} casi,
lo conserva in {metrics['hybrid_equal']} e lo riduce in {metrics['hybrid_worse']}. Guadagno medio {metrics['hybrid_mean_gain_pct']:.2f}%,
intervallo {metrics['hybrid_min_gain_pct']:.2f}% / {metrics['hybrid_max_gain_pct']:.2f}%.
Non è attivata automaticamente negli algoritmi.</p>
<h2>Protocollo e limiti</h2><p>Grafi connessi planari di grado 3, famiglia di espansione dei vertici del generatore esistente.
Carichi condivisi materializzati e verificati con SHA-256; arrotondamento intero dei carichi scalati.
Un timestep per istanza, ordine delle due proposte alternato fra istanze, un thread Gurobi.
L'inizializzazione è congelata per isolare i pareggi fra ottimi locali; il suo tempo reale viene aggiunto al tempo totale,
mentre i contatori incorporati nel replay sono azzerati.
Le tracce aggiungono lo stesso osservatore a entrambe le proposte.
La configurazione dei limiti è condivisa; il limite PG effettivo è ridotto dal tempo residuo delle aste e può differire leggermente.
In {metrics['compact_within_lower_paired_cap']} delle {metrics['pairs']} coppie il compatto completa la PG entro il minore dei due limiti effettivi.
L'eventuale guadagno di welfare della compattazione deriva da ulteriori mosse migliorative consentite dal tempo risparmiato,
non da una diversa funzione obiettivo. Nessun messaggio o selezione globale viene aggiunto.
I tempi sono misure su un solo computer e una ripetizione per variante/istanza; non sono una garanzia universale.</p>
<h2>Confronti</h2>{pairs[keys + ['pg_throughput_speedup','wall_speedup','welfare_gain_pct','pg_budget_full','pg_budget_compact','compact_within_lower_cap','uncensored']].to_html(index=False, float_format=lambda a:f'{a:.3f}')}
</html>'''
  (output / 'report.html').write_text(html)
  print(json.dumps(metrics, indent=2))
  print(summary.to_string(index=False))


if __name__ == '__main__':
  summarize(Path(sys.argv[1]))
