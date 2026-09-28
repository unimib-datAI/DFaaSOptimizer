# Confronto centralizzato originale e selfish

## Formulazione

`SelfishLoadManagementModel` massimizza lo stesso guadagno globale di `LoadManagementModel`. Per ogni nodo risolve prima `LSP_detailed`, massimizzando il negativo del precedente costo completo (con x, y, z e π). Dalla soluzione ottenuta estrae soltanto `sum_f alpha[n,f]*x[f]/D[n,f]` e impone che la stessa componente nel centralizzato sia almeno pari a tale soglia. `D` è il carico in ingresso, oppure 1 se il carico è zero.

## Protocollo

Quattro istanze già salvate in `test_instances/`, con 10 o 20 nodi e 3 funzioni; snapshot t=0, 50, 75. I dati e le tracce non sono stati rigenerati. Gurobi 12.0.3, 1 thread, seed solver 0, MIPGap=1e-5, FeasibilityTol=1e-8, limite di 30 secondi per ciascuna ottimizzazione; π=0. Entrambi i centralizzati ricevono lo stesso snapshot e gli stessi coefficienti. Tutti i riferimenti locali hanno terminazione ottima entro la tolleranza.

Su 210 controlli nodo–snapshot: **87 violazioni nell’originale, 0 nello selfish** (tolleranza di controllo 1e-6). Entrambe le soluzioni di ogni coppia superano anche il validatore fisico esistente: bilancio del carico, memoria, utilizzo, vicinato e assenza di ping-pong.

Gli incumbent selfish hanno un guadagno globale inferiore del **0.99%–11.86%** rispetto agli incumbent originali. Questa è una differenza fra soluzioni trovate: nei casi con limite di tempo non equivale a una differenza esatta fra gli ottimi.

Terminazioni ottime entro tolleranza: **7/12 originali, 11/12 selfish**. Le altre terminazioni sono per limite di tempo, con incumbent fisicamente valido. La colonna gap riporta `(bound-guadagno)/abs(guadagno)`. Per gli selfish ottimi delle prime 11 coppie il wrapper non espone il bound: è riportata la tolleranza massima certificata dalla terminazione.

## Risultati per snapshot

| Nodi / seed istanza | t | Guadagno originale | Guadagno selfish | Riduzione incumbent | Nodi sotto soglia orig. / selfish | Gap orig. | Gap selfish | Stato orig. / selfish |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 10 / 3865 | 0 | 25.672856 | 25.318523 | 1.38% | 3 / 0 | 0.0000% | ≤0.0010% | ottimo / ottimo |
| 10 / 3865 | 50 | 24.791889 | 24.381996 | 1.65% | 3 / 0 | 0.0009% | ≤0.0010% | ottimo / ottimo |
| 10 / 3865 | 75 | 24.941875 | 24.695026 | 0.99% | 2 / 0 | 0.0005% | ≤0.0010% | ottimo / ottimo |
| 20 / 4850 | 0 | 30.959674 | 27.812218 | 10.17% | 8 / 0 | 0.0010% | ≤0.0010% | ottimo / ottimo |
| 20 / 4850 | 50 | 34.845046 | 30.711609 | 11.86% | 12 / 0 | 0.1303% | ≤0.0010% | limite tempo / ottimo |
| 20 / 4850 | 75 | 32.380284 | 29.434922 | 9.10% | 8 / 0 | 0.0007% | ≤0.0010% | ottimo / ottimo |
| 20 / 7521 | 0 | 31.504677 | 30.220348 | 4.08% | 8 / 0 | 0.1889% | ≤0.0010% | limite tempo / ottimo |
| 20 / 7521 | 50 | 37.209237 | 33.162802 | 10.87% | 12 / 0 | 0.0636% | ≤0.0010% | limite tempo / ottimo |
| 20 / 7521 | 75 | 30.922090 | 29.950543 | 3.14% | 8 / 0 | 0.0010% | ≤0.0010% | ottimo / ottimo |
| 20 / 3865 | 0 | 37.191007 | 33.264796 | 10.56% | 8 / 0 | 0.0186% | ≤0.0010% | limite tempo / ottimo |
| 20 / 3865 | 50 | 38.557190 | 34.910766 | 9.46% | 6 / 0 | 0.0010% | ≤0.0010% | ottimo / ottimo |
| 20 / 3865 | 75 | 39.736275 | 37.755079 | 4.99% | 9 / 0 | 0.0914% | 0.0114% | limite tempo / limite tempo |

## Riproduzione e file

```bash
PYTHONPATH=. .venv/bin/python outputs/selfish-comparison/compare.py
```

`summary.csv`: guadagni, terminazioni, bound disponibili, tempi e carico locale/offloaded/cloud. `nodes.csv`: soglia e guadagno x per nodo, modello e snapshot. `settings.json`: opzioni Gurobi. `metrics.json`: riepilogo numerico. Il benchmark usa direttamente Pyomo per conservare gli incumbent al limite di tempo; il wrapper comune `BaseAbstractModel.solve` può scartarli dopo l’autoload di Pyomo. `compute_local_gain_floors()` rende disponibili le soglie prima della chiamata diretta al solver.

## Verifica del codice

- 40 test mirati superati: `tests/test_selfish_model.py` e `tests/test_model_construction.py`.
- I quattro nuovi casi coprono segno/prezzo nell’obiettivo locale, protezione x, effetto dell’offloading sulla scelta della x di riferimento e carico nullo.
- Suite completa eseguita prima dell’ultima estrazione del metodo di calcolo delle soglie: 808 test superati e un errore in `test_integer_fixed_sum_traces_preserve_system_workload`. Il generatore produce 9 richieste invece di 10; lo stesso difetto è stato riprodotto eseguendo il codice del generatore letto direttamente da HEAD. Log in `tests.log`.
- Ruff e `git diff --check` passati.

Le soglie dipendono dalla x restituita dal solver in caso di ottimi locali multipli. Il vincolo protegge la componente di guadagno delle sole x, aggregata sulle funzioni.
