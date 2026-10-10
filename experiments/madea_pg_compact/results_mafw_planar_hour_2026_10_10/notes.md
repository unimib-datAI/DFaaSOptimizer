## Note di metodo (planar-hour, 10/10)

- Riferimenti: `families-six-balanced-2026-10-01` (35 casi: 24 statici, 8 stress, 3 temporali). In quella campagna il DP locale era il default (reso opt-in solo il 09/10, commit d6b4924), quindi MAFW gira con `use_dp=true`. Una prima esecuzione con Gurobi (`solutions/mafw-parts/hour`, 27 casi) è fuori protocollo ed è esclusa: dà welfare diverso dal DP (−0,19% in media) e tempi ~6× più alti.
- Deriva del codice: MADEA rieseguito oggi con DP sulla stessa VM non riproduce esattamente il welfare del 01/10 (+0,2…+2,3% su 5 casi); i distacchi fra MAFW e i metodi migliori (~10 punti) sono molto più grandi.
- Tempi: la colonna tempo confronta la VM di oggi con la macchina del 01/10. Sulla stessa VM, MAFW/MADEA = 0,2–0,6 su 5 casi di questa famiglia e 0,6–1,0 su 5 casi della famiglia alpha-gt-beta (dove la VM riproduce MADEA esattamente e ha velocità ~1,1× la macchina di riferimento).
- Parametri MAFW invariati (κ=10, 400 iterazioni, smoothing 1): nessuna taratura su questa famiglia.
