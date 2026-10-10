# Tabella complessiva con FaaS-MAFW

Stesse 12 istanze per campagna, stessa configurazione e stesso codice dei metodi di riferimento (hash di input verificati). Guadagno medio del welfare rispetto a MADEA nella fase (std = 8 casi statici, stress = 2, temp = 2); tempo = rapporto mediano dei tempi di parete rispetto a MADEA.

## Riepilogo (entrambe le topologie)

| Metodo | std orig. | stress orig. | temp orig. | tempo orig. | std ric. | stress ric. | temp ric. | tempo ric. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FaaS-MAFW | +0.91% | +3.46% | +1.42% | 0.70× | +0.95% | +3.62% | +1.84% | 0.72× |
| FaaS-MAPG-R | -0.12% | +2.96% | +0.47% | 0.60× | -0.08% | +3.03% | +0.84% | 0.58× |
| FaaS-MAPG-S | -0.04% | +3.24% | +0.62% | 0.62× | +0.01% | +3.07% | +1.00% | 0.58× |
| MADEA | +0.00% | +0.00% | +0.00% | 1.00× | +0.00% | +0.00% | +0.00% | 1.00× |
| MADEA-PG | +0.57% | +3.33% | +0.94% | 1.19× | +0.59% | +3.47% | +1.19% | 1.23× |
| MADEA-one-shot | +0.08% | -0.07% | +0.18% | 0.53× | -0.07% | -0.10% | +0.24% | 0.50× |
| MADEA2 | +0.13% | +0.27% | +0.74% | 1.15× | -0.05% | +0.43% | +0.97% | 1.03× |
| MADEA2-one-shot | +0.37% | +0.24% | +0.97% | 0.69× | +0.38% | +0.36% | +1.19% | 0.65× |
| MAFW-PG | +1.03% | +3.60% | +1.51% | 0.84× | +nan% | +nan% | +nan% | nan× |
| PLASMA-Welfare | +0.10% | +4.26% | +1.31% | 1.63× | +0.00% | +4.07% | +1.75% | 1.57× |
| hierarchical-madea | +0.00% | +0.01% | +0.01% | 1.82× | +0.00% | +0.00% | +0.01% | 1.46× |
| hierarchical-madea-pg | +0.56% | +3.37% | +0.95% | 1.93× | +0.59% | +3.47% | +1.18% | 1.72× |
| hierarchical-one-shot | +0.22% | -0.11% | +0.23% | 2.72× | +0.08% | -0.14% | +0.50% | 3.15× |
| hierarchical-one-shot-pg | +0.80% | +3.26% | +1.15% | 1.06× | +0.66% | +3.37% | +1.68% | 1.09× |
| one-shot-pg | +0.62% | +3.23% | +1.07% | 0.69× | +0.48% | +3.29% | +1.32% | 0.69× |
| one-shot2-pg | +0.86% | +3.58% | +1.43% | 0.90× | +0.85% | +3.78% | +1.84% | 0.89× |

## alpha-gt-beta (09/10)

| Metodo | std | stress | temp | tempo (mediana vs MADEA) | tempo medio [s] | posizione media (std) | vittorie (std) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MADEA | +0.00% | +0.00% | +0.00% | 1.00× | 12.0 | 12.81 | 0 |
| MADEA-PG | +0.57% | +3.33% | +0.94% | 1.19× | 14.0 | 6.38 | 1 |
| MADEA-one-shot | +0.08% | -0.07% | +0.18% | 0.53× | 5.4 | 11.75 | 0 |
| one-shot-pg | +0.62% | +3.23% | +1.07% | 0.69× | 7.4 | 5.88 | 0 |
| hierarchical-madea | +0.00% | +0.01% | +0.01% | 1.82× | 40.6 | 12.44 | 0 |
| hierarchical-madea-pg | +0.56% | +3.37% | +0.95% | 1.93× | 20.3 | 6.62 | 1 |
| hierarchical-one-shot | +0.22% | -0.11% | +0.23% | 2.72× | 37.3 | 9.62 | 0 |
| hierarchical-one-shot-pg | +0.80% | +3.26% | +1.15% | 1.06× | 12.5 | 4.12 | 1 |
| PLASMA-Welfare | +0.10% | +4.26% | +1.31% | 1.63× | 14.8 | 10.00 | 0 |
| FaaS-MAPG-S | -0.04% | +3.24% | +0.62% | 0.62× | 6.1 | 12.88 | 0 |
| FaaS-MAPG-R | -0.12% | +2.96% | +0.47% | 0.60× | 6.5 | 14.00 | 0 |
| FaaS-MAFW | +0.91% | +3.46% | +1.42% | 0.70× | 8.2 | 3.88 | 0 |
| MAFW-PG | +1.03% | +3.60% | +1.51% | 0.84× | 9.0 | 1.88 | 5 |
| MADEA2 | +0.13% | +0.27% | +0.74% | 1.15× | 11.3 | 11.25 | 0 |
| MADEA2-one-shot | +0.37% | +0.24% | +0.97% | 0.69× | 7.3 | 8.62 | 0 |
| one-shot2-pg | +0.86% | +3.58% | +1.43% | 0.90× | 10.0 | 3.88 | 1 |

## planar-uniform (09/10)

| Metodo | std | stress | temp | tempo (mediana vs MADEA) | tempo medio [s] | posizione media (std) | vittorie (std) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MADEA | +0.00% | +0.00% | +0.00% | 1.00× | 11.6 | 11.50 | 0 |
| MADEA-PG | +0.59% | +3.47% | +1.19% | 1.23× | 13.7 | 6.50 | 1 |
| MADEA-one-shot | -0.07% | -0.10% | +0.24% | 0.50× | 5.8 | 10.75 | 0 |
| one-shot-pg | +0.48% | +3.29% | +1.32% | 0.69× | 7.7 | 6.12 | 1 |
| hierarchical-madea | +0.00% | +0.00% | +0.01% | 1.46× | 18.7 | 11.00 | 0 |
| hierarchical-madea-pg | +0.59% | +3.47% | +1.18% | 1.72× | 20.0 | 6.00 | 2 |
| hierarchical-one-shot | +0.08% | -0.14% | +0.50% | 3.15× | 37.6 | 10.50 | 0 |
| hierarchical-one-shot-pg | +0.66% | +3.37% | +1.68% | 1.09× | 13.4 | 5.00 | 0 |
| PLASMA-Welfare | +0.00% | +4.07% | +1.75% | 1.57× | 15.0 | 8.62 | 1 |
| FaaS-MAPG-S | +0.01% | +3.07% | +1.00% | 0.58× | 6.6 | 9.50 | 0 |
| FaaS-MAPG-R | -0.08% | +3.03% | +0.84% | 0.58× | 6.8 | 10.62 | 0 |
| FaaS-MAFW | +0.95% | +3.62% | +1.84% | 0.72× | 8.3 | 2.88 | 4 |
| MADEA2 | -0.05% | +0.43% | +0.97% | 1.03× | 12.0 | 10.00 | 0 |
| MADEA2-one-shot | +0.38% | +0.36% | +1.19% | 0.65× | 7.0 | 7.50 | 0 |
| one-shot2-pg | +0.85% | +3.78% | +1.84% | 0.89× | 9.8 | 3.50 | 0 |

## Note (MADEA2 / one-shot2, 10/10)

- `madea2.py` non modifica MADEA né one-shot: esegue i loro runner con tre funzioni sostituite per la durata della chiamata (offerte di memoria con quantità scoperta e valore β+γ; avvio di repliche greedy per valore per MB; offerte in attesa ordinate per valore per MB). Stessi messaggi; le offerte di memoria portano due numeri in più. Il prezzo di congestione è opzionale (`auction.congestion_weight`) e qui è spento.
- Config delle campagne del 09/10 (DP, Gurobi a licenza ristretta per LSPr). Le varianti "2" girano sulla VM Linux, due run alla volta; i riferimenti sono le misure del 09/10 (la VM riproduce esattamente MADEA su alpha-gt-beta, a ~1,1× il tempo).
- Nessuna taratura: le regole nuove non hanno parametri.
