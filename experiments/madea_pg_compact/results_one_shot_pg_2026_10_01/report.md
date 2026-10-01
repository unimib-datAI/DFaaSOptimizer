# One-shot-PG: integrazione e piccolo confronto planare

Il nuovo algoritmo è selezionabile come `one-shot-pg` e usa one-shot per
l'inizializzazione, seguito dal raffinamento locale già usato in MADEA-PG.
La fase aggiunta decide con l'utilità del singolo nodo e le capacità residue
dei vicini; non usa il welfare globale per accettare le mosse. Il simulatore
ordina i turni e gestisce gli export. Non è un'implementazione di un protocollo
di rete distribuito. La baseline `faas-madea-1s` resta disponibile separatamente.

La configurazione PG è `solver_options.madea_pg`: al massimo cinque sweep,
0,25 secondi per nodo se non è specificato un budget, limitati dal tempo
nativo restante. Una DP già avviata può superare il budget nominale. Con
fallback GLPK, sotto un secondo residuo non vengono avviate proposte.

Il confronto contiene otto istanze planari connesse di grado 3, con seed 7 e
42, 40/80 nodi e 5/10 funzioni. Sono 24 esecuzioni reali, tutte sequenziali,
senza replay e con gli stessi input per i tre metodi. I tempi includono output
e tutti i passaggi del runner. Gurobi è configurato come fallback, con DP
nativa per i modelli supportati. Il limite complessivo era cinque minuti;
le esecuzioni sono terminate prima della scadenza. Ogni soluzione è stata
verificata per ammissibilità, integrità e coerenza del welfare esportato.

## Risultati

Ogni cella indica welfare medio e tempo mediano totale; due seed per riga.

| Nodi | Funzioni | One-shot | One-shot-PG | MADEA-PG |
| --- | --- | --- | --- | --- |
| 40 | 5 | 751.0 (1.03 s) | 854.8 (1.53 s) | 865.3 (2.48 s) |
| 40 | 10 | 1365.9 (1.75 s) | 1610.8 (2.43 s) | 1617.3 (3.37 s) |
| 80 | 5 | 1304.3 (3.54 s) | 1739.7 (5.06 s) | 1750.3 (8.56 s) |
| 80 | 10 | 3073.5 (6.14 s) | 3437.7 (7.95 s) | 3422.2 (12.29 s) |

One-shot-PG migliora one-shot in tutti gli otto casi: +20.1% di welfare medio per istanza. Rispetto a MADEA-PG, il guadagno medio è -0.38%, con un caso peggiore del -2.32% e tre casi migliori. È più rapida in 8 casi su otto; il rapporto mediano tra i tempi è 0.686. Le percentuali sono medie dei rapporti per istanza, non rapporti tra medie di welfare.

È un campione piccolo, con un'esecuzione per metodo e istanza. I metodi
mantengono criteri d'arresto differenti: il risultato non certifica l'ottimo
e non confronta il welfare a parità di tempo. Il generatore fixed_sum mantiene
il problema noto nell'arrotondamento; si usano gli stessi carichi effettivi.

## Verifica del codice

I due nuovi test di integrazione passano: due timestep, budget nullo con
allocazioni identiche alla baseline, ammissibilità, welfare non decrescente,
export coerenti e selezione/ripresa separata nel batch e nei job remoti.
Alterando il misuratore globale di welfare, le mosse e allocazioni finali
restano identiche: il misuratore non è un oracolo di accettazione.
Ruff e Mypy sui quattro file di produzione modificati passano.

La suite completa ha 902 test passati e tre fallimenti preesistenti.
I due test gerarchici segnalano `define_bids(..., delta=...)`, parametro
non più accettato dopo l'integrazione precedente dei helper condivisi.
Entrambi si riproducono caricando il runner one-shot da HEAD, senza le nuove
modifiche. Il terzo è il problema noto di fixed_sum (9 richieste invece di 10).
I nomi completi e il commit di riferimento sono in [verification.json](verification.json).

## File e riproduzione

- [Esecuzioni e tempi](runs.csv)
- [Welfare per istanza](welfare.csv)
- [Statistiche PG](refinements.csv)
- [Metriche](metrics.json)
- [Protocollo e hash del codice](protocol.json)

Dalla radice del repository, nell'ambiente virtuale esistente:

```sh
MPLCONFIGDIR=/tmp/dfaas-mpl XDG_CACHE_HOME=/tmp/dfaas-test-cache .venv/bin/python experiments/madea_pg_compact/compare_one_shot_pg.py --seconds 300
```

Gli output completi restano nella cartella ignorata `solutions/`.
