# Interventi sul gerarchico MADEA-PG — 1 ottobre 2026

Ho tenuto **tempo effettivo e riserva per il PG**. Ho scartato entrambe le
versioni sperimentali dell’arresto locale. Le modifiche sono attive soltanto
nel gerarchico MADEA-PG con budget PG positivo e almeno uno sweep.
MADEA, one-shot e gli altri algoritmi senza PG conservano i criteri e i budget
precedenti. Anche MADEA-PG piatto e gerarchico one-shot-PG restano invariati.

## Che cosa è stato provato e deciso

| Intervento | Esito | Decisione |
|---|---|---|
| Riserva PG da sola, con il vecchio conteggio del tempo | Stesso welfare nei sette casi confrontabili; resta un timeout | Non sufficiente da sola |
| Tempo effettivo nei cicli MADEA e nella gerarchia | Otto casi completati; stesso welfare nei sette confrontabili | Tenuto |
| Tempo effettivo più riserva PG | Otto casi completati; il caso grande beneficia del raffinamento finale | Tenuti insieme |
| Arresto locale con tentativo anticipato di nuove repliche | Welfare peggiore in cinque dei sette casi confrontabili; variazione media −0,0978% | Scartato |
| Arresto conservativo con tempi originali del tentativo di nuove repliche | Nessun beneficio attribuibile al nuovo arresto nei casi misurati | Scartato |

La riserva è il budget PG richiesto, limitato a metà del limite complessivo
del timestep. L’asta usa il tempo trascorso dall’inizio del timestep; il PG
riceve il tempo realmente residuo. Le opzioni del solver locale restano
quelle originali: un budget frazionario dell’asta non viene passato a GLPK.
Il relativo test forza un vero fallback MILP e riproduce l’errore senza la correzione.
Il round, il modello locale o la proposta corrente possono finire oltre il
limite: questo controllo del tempo non è un’interruzione rigida del processo.

## Risultati delle prove separate

Ogni fase misura gli stessi otto input planari già materializzati, con
40/80/160 nodi, 5/10 funzioni e seed diversi. Comprendono sei casi standard,
un carico stress 1,5 e una sequenza temporale con profilo 0,75 → 1,25 → 1.
I grafi sono connessi, planari e con grado tre. Solver Gurobi, esecuzione
sequenziale, limite generale `max(30, 0.5*N)` secondi e PG `0.25*N` secondi.
Gli hash degli input sono verificati prima di ogni esecuzione. Tutti gli
output completati superano i controlli di ammissibilità, integrità dei flussi,
conservazione delle richieste e coerenza fra welfare ed export; il PG non
peggiora il proprio incumbent iniziale.

| Fase | Casi completati | Variazione media del welfare | Accelerazione mediana |
|---|---:|---:|---:|
| baseline | 7/8 | +0.0000% | 1.000× |
| reserve | 7/8 | +0.0000% | 0.993× |
| wall | 8/8 | +0.0000% | 1.090× |
| stagnation | 8/8 | -0.0978% | 1.136× |
| wall-reserve | 8/8 | +0.0000% | 1.129× |
| wall-reserve-conservative | 8/8 | +0.0000% | 1.150× |
| selected | 8/8 | +0.0000% | 1.151× |

Il motivo finale esportato non riporta l’arresto conservativo in nessun caso.
Non sono stati registrati contatori per ogni arresto dei cicli interni,
quindi questo non prova che il criterio non sia mai intervenuto internamente.

Variazione e accelerazione sono calcolate sui **sette casi con baseline
valida**, usando coppie sullo stesso input. `selected` è una nuova esecuzione
del codice definitivo, dopo la rimozione dell’arresto e la correzione GLPK.
Il rapporto mediano di 1,151× corrisponde a circa il 13% di tempo in meno.
Una sola misura per fase non separa completamente il beneficio dalle
variazioni della macchina; i piccoli scarti temporali sono indicativi. Non sono stati eseguiti casi indipendenti di conferma o tutti i
35 casi della precedente tabella: questi risultati non vanno estesi a quella tabella.

## Valori assoluti del codice definitivo

Ogni cella riporta **welfare / secondi reali dell’intera esecuzione**. Per il
caso temporale, il welfare è la media dei tre timestep e il tempo è il totale.

| Caso: fase, nodi/funzioni, seed, carico iniziale | Prima | Definitivo |
|---|---:|---:|
| static, 40/5, seed 7, carico 1 | 1003.41 / 3.38 | 1003.41 / 2.92 |
| static, 160/10, seed 7, carico 1 | timeout | 8597.95 / 52.85 |
| static, 160/5, seed 42, carico 1 | 4589.76 / 34.01 | 4589.76 / 32.45 |
| static, 40/10, seed 42, carico 1 | 1447.87 / 5.23 | 1447.87 / 5.10 |
| static, 80/5, seed 99, carico 1 | 2012.39 / 17.19 | 2012.39 / 16.23 |
| static, 80/10, seed 2026, carico 1 | 3613.23 / 32.02 | 3613.23 / 23.72 |
| stress, 80/10, seed 42, carico 1.5 | 2910.64 / 25.57 | 2910.64 / 21.93 |
| temporal, 40/10, seed 7, carico 0.75 | 1815.25 / 21.68 | 1815.25 / 18.84 |

Per 160 nodi, 10 funzioni e seed 7, la baseline originaria supera il timeout
esterno di 150 secondi: non le attribuisco un welfare incompleto. Con il
solo tempo effettivo si ottengono **7256.79** in
**86.11 s**; con il codice definitivo si ottengono
**8597.95** in **52.85 s**, cioè
**+18.48%** di welfare rispetto
al solo tempo effettivo. Il risultato varia leggermente fra riesecuzioni
perché la soglia temporale può tagliare il ciclo un’iterazione prima o dopo.
Negli altri sette input il welfare definitivo è identico alla baseline,
con accelerazione mediana **1.151×**.

## Ambito e decentralizzazione

La modifica ripartisce il tempo fra fasi e legge un orologio. Non introduce
un ottimizzatore globale, nuove informazioni per le offerte o nuovi archi
di comunicazione. Le proposte PG mantengono le decisioni locali e gli impegni
già accettati. Il criterio storico di avanzamento della gerarchia e l’archivio
del migliore welfare globale erano già presenti e non sono cambiati; questo
intervento non dimostra che tutta l’orchestrazione storica sia decentralizzata.

I test controllano anche non-PG, PG con budget zero e PG con zero sweep,
mantenendo reali i modelli locali e simulando soltanto il ritardo iniziale.
La revisione statica finale non ha trovato criticità nei tre file modificati.

La suite completa: **916 test passati e
1 fallimento preesistente**,
`test_integer_fixed_sum_traces_preserve_system_workload`.
Ruff e Mypy sui quattro file di codice e test della modifica, e il controllo
del diff, sono passati. GitNexus segnala il ciclo condiviso
come CRITICAL per numero di dipendenze: i nuovi argomenti sono disattivati
per default e forniti soltanto dal percorso PG abilitato.

## Riproducibilità

Le sottocartelle conservano CSV, protocolli con hash e snapshot dei runner
effettivamente misurati. Gli snapshot delle varianti scartate servono a
verificare gli esperimenti; non sono il codice corrente.

```sh
MPLCONFIGDIR=/tmp/dfaas-mpl XDG_CACHE_HOME=/tmp/dfaas-test-cache .venv/bin/python experiments/madea_pg_compact/ablate_hierarchical_madea_pg.py --output solutions/hierarchical-madea-ablation-repeat --seconds 900
```

Questo comando riesegue soltanto il codice attuale, senza ricreare le
varianti sperimentali rimosse. Dati completi originali:
`solutions/hierarchical-madea-ablation-2026-10-01/`.
