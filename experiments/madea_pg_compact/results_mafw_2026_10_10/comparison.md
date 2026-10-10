# Tabella complessiva con FaaS-MAFW

Stesse 12 istanze per campagna, stessa configurazione e stesso codice dei metodi di riferimento (hash di input verificati). Guadagno medio del welfare rispetto a MADEA nella fase (std = 8 casi statici, stress = 2, temp = 2); tempo = rapporto mediano dei tempi di parete rispetto a MADEA.

## Riepilogo (entrambe le topologie)

| Metodo | std orig. | stress orig. | temp orig. | tempo orig. | std ric. | stress ric. | temp ric. | tempo ric. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MADEA | +0.00% | +0.00% | +0.00% | 1.00× | +0.00% | +0.00% | +0.00% | 1.00× |
| MADEA-PG | +0.57% | +3.33% | +0.94% | 1.19× | +0.59% | +3.47% | +1.19% | 1.23× |
| MADEA-one-shot | +0.08% | -0.07% | +0.18% | 0.53× | -0.07% | -0.10% | +0.24% | 0.50× |
| one-shot-pg | +0.62% | +3.23% | +1.07% | 0.69× | +0.48% | +3.29% | +1.32% | 0.69× |
| hierarchical-madea | +0.00% | +0.01% | +0.01% | 1.82× | +0.00% | +0.00% | +0.01% | 1.46× |
| hierarchical-madea-pg | +0.56% | +3.37% | +0.95% | 1.93× | +0.59% | +3.47% | +1.18% | 1.72× |
| hierarchical-one-shot | +0.22% | -0.11% | +0.23% | 2.72× | +0.08% | -0.14% | +0.50% | 3.15× |
| hierarchical-one-shot-pg | +0.80% | +3.26% | +1.15% | 1.06× | +0.66% | +3.37% | +1.68% | 1.09× |
| PLASMA-Welfare | +0.10% | +4.26% | +1.31% | 1.63× | +0.00% | +4.07% | +1.75% | 1.57× |
| FaaS-MAPG-S | -0.04% | +3.24% | +0.62% | 0.62× | +0.01% | +3.07% | +1.00% | 0.58× |
| FaaS-MAPG-R | -0.12% | +2.96% | +0.47% | 0.60× | -0.08% | +3.03% | +0.84% | 0.58× |
| FaaS-MAFW | +0.91% | +3.46% | +1.42% | 0.70× | +0.95% | +3.62% | +1.84% | 0.72× |

## alpha-gt-beta (grafi originali)

| Metodo | std | stress | temp | tempo (mediana vs MADEA) | tempo medio [s] | posizione media (std) | vittorie (std) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MADEA | +0.00% | +0.00% | +0.00% | 1.00× | 12.0 | 9.31 | 0 |
| MADEA-PG | +0.57% | +3.33% | +0.94% | 1.19× | 14.0 | 4.12 | 1 |
| MADEA-one-shot | +0.08% | -0.07% | +0.18% | 0.53× | 5.4 | 8.50 | 0 |
| one-shot-pg | +0.62% | +3.23% | +1.07% | 0.69× | 7.4 | 3.75 | 0 |
| hierarchical-madea | +0.00% | +0.01% | +0.01% | 1.82× | 40.6 | 8.94 | 0 |
| hierarchical-madea-pg | +0.56% | +3.37% | +0.95% | 1.93× | 20.3 | 4.38 | 1 |
| hierarchical-one-shot | +0.22% | -0.11% | +0.23% | 2.72× | 37.3 | 6.62 | 0 |
| hierarchical-one-shot-pg | +0.80% | +3.26% | +1.15% | 1.06× | 12.5 | 2.62 | 2 |
| PLASMA-Welfare | +0.10% | +4.26% | +1.31% | 1.63× | 14.8 | 7.25 | 0 |
| FaaS-MAPG-S | -0.04% | +3.24% | +0.62% | 0.62× | 6.1 | 9.38 | 0 |
| FaaS-MAPG-R | -0.12% | +2.96% | +0.47% | 0.60× | 6.5 | 10.50 | 0 |
| FaaS-MAFW | +0.91% | +3.46% | +1.42% | 0.70× | 8.2 | 2.62 | 5 |

## planar-uniform (grafi ricampionati)

| Metodo | std | stress | temp | tempo (mediana vs MADEA) | tempo medio [s] | posizione media (std) | vittorie (std) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MADEA | +0.00% | +0.00% | +0.00% | 1.00× | 11.6 | 9.38 | 0 |
| MADEA-PG | +0.59% | +3.47% | +1.19% | 1.23× | 13.7 | 5.00 | 1 |
| MADEA-one-shot | -0.07% | -0.10% | +0.24% | 0.50× | 5.8 | 8.38 | 0 |
| one-shot-pg | +0.48% | +3.29% | +1.32% | 0.69× | 7.7 | 4.62 | 1 |
| hierarchical-madea | +0.00% | +0.00% | +0.01% | 1.46× | 18.7 | 8.88 | 0 |
| hierarchical-madea-pg | +0.59% | +3.47% | +1.18% | 1.72× | 20.0 | 4.50 | 2 |
| hierarchical-one-shot | +0.08% | -0.14% | +0.50% | 3.15× | 37.6 | 8.38 | 0 |
| hierarchical-one-shot-pg | +0.66% | +3.37% | +1.68% | 1.09× | 13.4 | 3.62 | 0 |
| PLASMA-Welfare | +0.00% | +4.07% | +1.75% | 1.57× | 15.0 | 6.62 | 1 |
| FaaS-MAPG-S | +0.01% | +3.07% | +1.00% | 0.58× | 6.6 | 7.38 | 0 |
| FaaS-MAPG-R | -0.08% | +3.03% | +0.84% | 0.58× | 6.8 | 8.50 | 0 |
| FaaS-MAFW | +0.95% | +3.62% | +1.84% | 0.72× | 8.3 | 2.75 | 4 |

## FaaS-MAFW prima e dopo le correzioni

| Versione | std orig. | stress orig. | temp orig. | std ric. | stress ric. | temp ric. | pos. media std orig. | pos. media std ric. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prima | -0.32% | +2.06% | -0.20% | -0.41% | +2.01% | -0.02% | 10.75 | 10.00 |
| dopo | +0.91% | +3.46% | +1.42% | +0.95% | +3.62% | +1.84% | 2.62 | 2.75 |

## Note di metodo

- **Versione di FaaS-MAFW.** Correzioni del 10/10: riparazione rispetto allo slack di memoria (un inoltro non scaccia mai carico locale del ricevente), greedy sul guadagno perso per MB liberato, penalità in unità di replica media con gap di Frank-Wolfe relativo, smoothing che maggiora la cerniera, tetto dell'arrotondamento a ⌊ω⌋, ripiego sulla soluzione senza inoltro, nessun arresto spurio quando l'arrotondamento dà zero, tempo nativo per nodo.
- **Parametri:** `iterations=400`, `kappa=10`, `smoothing=1`, `max_rounds=20` (mai raggiunto: le run chiudono in 2-4 round). κ è stato scelto fra {1, 3, 10, 30} su 4 dei 12 casi della campagna sui grafi originali: su quelle istanze il confronto è leggermente ottimista per MAFW. Fra κ=3 e κ=30 la differenza media è di qualche centesimo di punto.
- **Ambiente.** MAFW gira nella VM Linux sul Mac con `gurobipy` 13 a licenza ristretta e `use_dp=true`, una run alla volta. I metodi di riferimento sono quelli misurati il 09/10. Il welfare è deterministico (stesso valore in due esecuzioni), ma i rapporti di tempo confrontano macchine e sessioni diverse e vanno letti come indicativi.
- **Medie.** Guadagni ricalcolati dai `runs.csv` sui welfare sommati per caso: la colonna temp può differire di qualche decimo dal report del 09/10, che media per timestep.
