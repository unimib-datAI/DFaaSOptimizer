# Campagna sul catalogo edge-datasets

## Run completate

| method                   |   ok |   timeout |
|:-------------------------|-----:|----------:|
| FaaS-MAFW                |   32 |         0 |
| FaaS-MAFW-b              |   18 |         0 |
| MADEA                    |   33 |         0 |
| MADEA-PG                 |   32 |         0 |
| MADEA-one-shot           |   33 |         0 |
| MADEA2                   |   32 |         0 |
| MADEA2-b                 |   18 |         0 |
| MADEA2-b-one-shot        |   18 |         0 |
| MADEA2-one-shot          |   32 |         0 |
| MAFW-PG                  |   32 |         0 |
| MAFW-b-PG                |   18 |         0 |
| PLASMA-Welfare           |   32 |         1 |
| centralized              |   67 |         0 |
| hierarchical-one-shot-pg |   33 |         0 |
| one-shot-pg              |   33 |         0 |
| one-shot2-b-pg           |   18 |         0 |
| one-shot2-pg             |   32 |         0 |

## Distanza dalla soluzione centralizzata (positivo = peggio del centralizzato)

Centralizzato: HiGHS con limite 60 s (120 s per n160); il bound duale dà la distanza massima dall'ottimo.

| method                   |   casi |   distanza media dal centralizzato % |   mediana % |   max % |   distanza media dal bound % |
|:-------------------------|-------:|-------------------------------------:|------------:|--------:|-----------------------------:|
| centralized              |     67 |                                0     |       0     |   0     |                        0.106 |
| PLASMA-Welfare           |     32 |                                0.988 |       0.657 |   5.905 |                        1.082 |
| one-shot2-pg             |     32 |                                1.724 |       0.656 |  11.772 |                        1.817 |
| hierarchical-one-shot-pg |     33 |                                1.755 |       0.757 |   9.945 |                        1.846 |
| one-shot-pg              |     33 |                                1.773 |       0.78  |   9.583 |                        1.865 |
| MADEA2                   |     32 |                                1.791 |       0.975 |   9.89  |                        1.883 |
| MADEA2-one-shot          |     32 |                                1.855 |       0.762 |  12.42  |                        1.948 |
| MAFW-PG                  |     32 |                                1.87  |       1.05  |   9.452 |                        1.962 |
| MADEA-one-shot           |     33 |                                1.913 |       0.786 |  10.138 |                        2.004 |
| MADEA-PG                 |     32 |                                1.931 |       0.899 |   8.959 |                        2.022 |
| FaaS-MAFW                |     32 |                                1.943 |       1.075 |   9.726 |                        2.035 |
| MADEA                    |     33 |                                2.071 |       0.99  |   9.54  |                        2.162 |
| one-shot2-b-pg           |     18 |                                2.445 |       0.966 |  10.859 |                        2.541 |
| MADEA2-b                 |     18 |                                2.573 |       1.322 |   9.305 |                        2.669 |
| MADEA2-b-one-shot        |     18 |                                2.617 |       1.293 |  11.214 |                        2.713 |
| MAFW-b-PG                |     18 |                                2.691 |       1.368 |   8.972 |                        2.785 |
| FaaS-MAFW-b              |     18 |                                2.753 |       1.398 |   9.255 |                        2.847 |

Stato del centralizzato: TerminationCondition.maxTimeLimit: 43, TerminationCondition.optimal: 24

## Per famiglia (distanza media dal centralizzato, %)

|                          |   cilic-europe30 |   data7-pisa |   ether-iiot |   ether-urban |   eua-melbourne |   synthetic-euclidean-planar |   synthetic-planar-cubic |   topo4mec-milan |   wondernetwork |   zhang-national |   zhang-regional |
|:-------------------------|-----------------:|-------------:|-------------:|--------------:|----------------:|-----------------------------:|-------------------------:|-----------------:|----------------:|-----------------:|-----------------:|
| FaaS-MAFW                |             0.93 |         2.67 |         0.96 |          6.81 |            1.52 |                         1.13 |                     1.07 |             0.45 |          nan    |              nan |              nan |
| FaaS-MAFW-b              |             0.89 |         2.96 |         1.02 |          6.84 |            1.36 |                       nan    |                   nan    |           nan    |          nan    |              nan |              nan |
| MADEA                    |             1.05 |         3.01 |         0.86 |          7.35 |            1.54 |                         1.12 |                     1.18 |             0.67 |            1.24 |              nan |              nan |
| MADEA-PG                 |             1.02 |         2.95 |         0.71 |          6.8  |            1.32 |                         0.95 |                     1.08 |             0.62 |          nan    |              nan |              nan |
| MADEA-one-shot           |             0.83 |         2.12 |         0.97 |          8.2  |            1.23 |                         0.8  |                     0.99 |             0.56 |            0.32 |              nan |              nan |
| MADEA2                   |             0.61 |         1.53 |         0.93 |          6.87 |            1.72 |                         1.05 |                     1.05 |             0.57 |          nan    |              nan |              nan |
| MADEA2-b                 |             0.81 |         1.88 |         1.11 |          7.02 |            1.53 |                       nan    |                   nan    |           nan    |          nan    |              nan |              nan |
| MADEA2-b-one-shot        |             0.75 |         1.55 |         0.91 |          7.91 |            1.33 |                       nan    |                   nan    |           nan    |          nan    |              nan |              nan |
| MADEA2-one-shot          |             0.79 |         1.33 |         0.9  |          7.92 |            1.48 |                         0.95 |                     1.01 |             0.47 |          nan    |              nan |              nan |
| MAFW-PG                  |             0.85 |         2.59 |         0.96 |          6.63 |            1.45 |                         1.06 |                     1.02 |             0.41 |          nan    |              nan |              nan |
| MAFW-b-PG                |             0.84 |         2.87 |         1.02 |          6.71 |            1.34 |                       nan    |                   nan    |           nan    |          nan    |              nan |              nan |
| PLASMA-Welfare           |             0.83 |         2.02 |         0.3  |          1.93 |            0.77 |                         0.56 |                     0.65 |             0.72 |            0.8  |              nan |              nan |
| hierarchical-one-shot-pg |             0.79 |         2.14 |         0.74 |          7.56 |            1.06 |                         0.68 |                     0.91 |             0.51 |            0.36 |              nan |              nan |
| one-shot-pg              |             0.81 |         2.08 |         0.83 |          7.7  |            1.05 |                         0.68 |                     0.9  |             0.51 |            0.31 |              nan |              nan |
| one-shot2-b-pg           |             0.72 |         1.48 |         0.77 |          7.53 |            1.02 |                       nan    |                   nan    |           nan    |          nan    |              nan |              nan |
| one-shot2-pg             |             0.75 |         1.22 |         0.82 |          7.56 |            1.3  |                         0.85 |                     0.89 |             0.41 |          nan    |              nan |              nan |

## Classifica tra i decentralizzati (31 casi completi per tutti i metodi mostrati)

| method                   |   posizione media |   vittorie |
|:-------------------------|------------------:|-----------:|
| PLASMA-Welfare           |              3.32 |         18 |
| one-shot-pg              |              4.66 |          4 |
| hierarchical-one-shot-pg |              4.82 |          1 |
| one-shot2-pg             |              4.94 |          6 |
| MAFW-PG                  |              6.24 |          2 |
| MADEA-PG                 |              6.32 |          1 |
| MADEA2-one-shot          |              6.53 |          1 |
| MADEA-one-shot           |              6.53 |          2 |
| MADEA2                   |              7.02 |          2 |
| FaaS-MAFW                |              7.76 |          0 |
| MADEA                    |              7.85 |          0 |

## Classifica con tutti i metodi, varianti "-b" comprese (17 casi completi per tutti)

| method                   |   posizione media |   vittorie |   distanza media dal centralizzato % |
|:-------------------------|------------------:|-----------:|-------------------------------------:|
| PLASMA-Welfare           |             4.118 |         10 |                                1.273 |
| one-shot2-b-pg           |             7.118 |          1 |                                2.538 |
| one-shot-pg              |             7.147 |          2 |                                2.733 |
| one-shot2-pg             |             7.206 |          3 |                                2.502 |
| hierarchical-one-shot-pg |             7.353 |          0 |                                2.696 |
| MADEA-PG                 |             7.735 |          1 |                                2.777 |
| MAFW-PG                  |             8.441 |          1 |                                2.68  |
| MADEA2                   |             8.559 |          2 |                                2.457 |
| MAFW-b-PG                |             8.853 |          2 |                                2.772 |
| MADEA2-one-shot          |             9.118 |          0 |                                2.655 |
| MADEA2-b                 |             9.265 |          1 |                                2.644 |
| MADEA-one-shot           |             9.882 |          1 |                                2.913 |
| MADEA                    |            10     |          0 |                                2.971 |
| MADEA2-b-one-shot        |            10.029 |          0 |                                2.701 |
| FaaS-MAFW                |            10.412 |          0 |                                2.764 |
| FaaS-MAFW-b              |            10.765 |          0 |                                2.838 |

## Confronti diretti (differenza % di welfare)

| confronto                                |   casi |   vince |   pari |   perde |   diff_media |    min |   max |
|:-----------------------------------------|-------:|--------:|-------:|--------:|-------------:|-------:|------:|
| MADEA2 vs MADEA                          |     32 |      15 |      2 |      15 |        0.33  | -0.744 | 6.477 |
| MADEA2-one-shot vs MADEA-one-shot        |     32 |      15 |      1 |      16 |        0.115 | -2.54  | 4.24  |
| one-shot2-pg vs one-shot-pg              |     32 |      15 |      1 |      16 |        0.102 | -2.421 | 4.232 |
| FaaS-MAFW vs MADEA                       |     32 |      14 |      0 |      18 |        0.166 | -0.918 | 2.182 |
| MAFW-PG vs one-shot-pg                   |     32 |      11 |      0 |      21 |       -0.042 | -1.167 | 3.771 |
| one-shot2-pg vs hierarchical-one-shot-pg |     32 |      15 |      0 |      17 |        0.08  | -2.029 | 4.58  |
| MAFW-PG vs hierarchical-one-shot-pg      |     32 |      11 |      0 |      21 |       -0.065 | -1.154 | 2.817 |
| MADEA2 vs MADEA2-b                       |     18 |      10 |      3 |       5 |        0.207 | -0.646 | 1.332 |
| MADEA2-b vs MADEA                        |     18 |      10 |      2 |       6 |        0.352 | -0.844 | 5.077 |
| MADEA2-b-one-shot vs MADEA-one-shot      |     18 |       7 |      3 |       8 |        0.234 | -1.197 | 3.548 |
| one-shot2-b-pg vs one-shot-pg            |     18 |       8 |      3 |       7 |        0.216 | -1.411 | 3.541 |
| FaaS-MAFW vs FaaS-MAFW-b                 |     18 |      10 |      1 |       7 |        0.09  | -0.519 | 0.839 |
| FaaS-MAFW-b vs MADEA                     |     18 |       8 |      0 |      10 |        0.157 | -1.127 | 1.781 |
| MAFW-b-PG vs one-shot-pg                 |     18 |       6 |      1 |      11 |       -0.028 | -1.878 | 3.422 |

## Tempo di parete e round per timestep (mediane)

| method                   |   wall |   rounds |
|:-------------------------|-------:|---------:|
| FaaS-MAFW                |   4.7  |      4   |
| FaaS-MAFW-b              |   6.73 |      4   |
| MADEA                    |  11.46 |     20   |
| MADEA-PG                 |  14.14 |     20   |
| MADEA-one-shot           |   5.52 |     18   |
| MADEA2                   |  11.29 |     21.5 |
| MADEA2-b                 |  18.77 |     30.5 |
| MADEA2-b-one-shot        |   5.65 |     17.5 |
| MADEA2-one-shot          |   4.97 |     17   |
| MAFW-PG                  |   4.7  |      4   |
| MAFW-b-PG                |   6.47 |      4   |
| PLASMA-Welfare           |   8.3  |     21   |
| centralized              |  60.93 |    nan   |
| hierarchical-one-shot-pg |   9.75 |      6   |
| one-shot-pg              |   6.45 |     18   |
| one-shot2-b-pg           |   6.89 |     17.5 |
| one-shot2-pg             |   6.11 |     17   |
## Note di metodo (campagna catalogo edge-datasets, 11/10/2026)

- Campione: 75 istanze estratte con seme fisso (`catalog_campaign.py --sample`, seed 20261010) dal catalogo rigenerato la sera del 10/10 e congelate in `solutions/catalog-sample-2026-10-10/instances` (non versionate). 44 "main" (taglia ~80, Nf 5 e 10), 22 "small" (~40), 9 "scale" (taglia massima della famiglia: n160, n500 in 5 famiglie). Tutte con β < α.
- Eseguito nel workspace cloud (2 core, 7 GB), una run per core alla volta, config `catalog_config_overnight.json` (config del 09/10, Gurobi a licenza ristretta per LSPr, use_dp=true). Solo passo 0 (statico).
- Centralizzato: LoadManagementModel con HiGHS, limite 60 s (120 s per n160); incumbent validato con `validate_centralized_solution` e misurato con la stessa funzione di welfare dei metodi decentralizzati. 24 ottimi, 43 al limite di tempo (gap medio dal bound 0,1%); i 5 casi n500 non entrano in memoria e mancano.
- Copertura: la sessione si è fermata la notte del 10/10 (dalle ~22:10 alle 6:37) e il workspace è stato riavviato verso le 7:56; la campagna è girata dalle 6:39 alle 7:55 e dalle 9:30 alle 10:00. Risultato: 32-33 istanze per gli 11 metodi principali, 18 per le varianti "-b" (solo istanze già complete, confronti appaiati), nessun metodo del secondo gruppo (hierarchical-madea(-pg), MAPG-S/R, hierarchical-MAFW(-PG), hierarchical-one-shot).
- Le distanze medie delle varianti "-b" sono su un sottoinsieme diverso: confrontarle solo tramite i confronti diretti.
- Varianti "-b": vedi `docs/decentralized-algorithms/information-model.md` (MADEA2 e la riparazione di MAFW usano β dei mittenti; le "-b" no).
