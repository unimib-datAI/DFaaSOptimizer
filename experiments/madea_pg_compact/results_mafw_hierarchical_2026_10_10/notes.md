## Note di metodo (varianti gerarchiche di FaaS-MAFW, 10/10)

- `hierarchical_frankwolfe.py`: ogni round = un round FaaS-MAFW (livello 1) + LSPr, poi i livelli 2–3 di `IterativeHierarchicalAuctionEngine` (lo stesso motore di hierarchical-one-shot-pg) sulla capacità residua delle repliche già avviate, con i prezzi di memoria di Frank-Wolfe come prezzi dei nodi; raffinamento PG finale opzionale con la stessa riserva di tempo di hierarchical-one-shot-pg (`madea_pg`: 5 sweep, 0,25 s/nodo). `MAFW-PG` è la stessa pipeline senza gerarchia (ablazione).
- Parametri FW invariati (κ=10, 400 iterazioni). Config di ciascuna campagna; planar-hour con `use_dp=true` (default del codice al 01/10).
- Le varianti sono state eseguite due alla volta su 4 core (una famiglia per flusso): il budget PG è a tempo di parete, quindi la contesa può ridurre leggermente le mosse PG. I tempi confrontano macchine diverse (vedi note planar-hour).
- Diagnosi su planar-hour n40-f5-s7: le soluzioni MAFW inoltrano ~850–920 req/s (β·y ≈ 2,4–2,8 k) contro ~2400 di hierarchical-one-shot-pg (β·y ≈ 17,3 k). MAFW usa solo memoria libera e non sposta mai carico locale; gerarchia e PG non rimuovono questo vincolo.
