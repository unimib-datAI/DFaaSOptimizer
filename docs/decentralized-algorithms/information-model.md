# Modello di informazione dei metodi decentralizzati

Questo documento registra chi conosce cosa nei metodi decentralizzati e perché
esistono le varianti "-b" di MADEA2 e di FaaS-MAFW. Nasce da un'osservazione
del 11/10/2026: nelle versioni MADEA2 e MAFW i venditori/riceventi usavano
un'informazione (β dei compratori) che in MADEA non hanno.

## Informazione privata

- **β_ij^f** (valore per il mittente i di far servire f dal vicino j) e
  **γ_i^f** (penalità del Cloud) sono del **mittente**. α e la memoria sono del
  nodo che li possiede.
- In un sistema decentralizzato realistico i nodi possono avere interessi
  propri: rivelare β al ricevente gli permette di discriminare (o di
  approfittarne), quindi un metodo che lo richiede non è confrontabile alla
  pari con uno che non lo richiede, anche se il numero di messaggi è lo stesso.

## Cosa vede il venditore/ricevente

| Metodo | Messaggi del mittente | Cosa usa il ricevente | Rivela β? |
|---|---|---|---|
| MADEA / one-shot | offerta b = p_j + ε + δ (δ = u_j − u_secondo), quantità; offerte di memoria con solo (i, j, f) | b per ordinare e fissare il prezzo; offerte di memoria contate per funzione | no (δ ne rivela solo una differenza) |
| MADEA2 / one-shot2 (10/10) | come sopra, **più** nelle offerte di memoria la quantità scoperta d e il valore v = β + γ | v per scegliere le repliche per valore per MB | **sì** |
| **MADEA2-b** / one-shot2-b | come MADEA2, ma v = offerta b calcolata come per la capacità | b per MB | no (come MADEA) |
| FaaS-MAFW (10/10) | frazioni di flusso; i riceventi pubblicano solo un prezzo | prezzi propri; nella **riparazione** ordina gli inoltri per β + γ del mittente | **sì, solo nella riparazione** |
| **FaaS-MAFW-b** | come MAFW | riparazione ordinata per il proprio prezzo per funzione | no |

La scelta del vicino in MAFW è sempre privata: il mittente confronta il proprio
β con il prezzo del ricevente.

## Perché si è arrivati qui

1. Su istanze alpha-gt-beta MAFW batteva i metodi d'asta, e l'analisi delle
   soluzioni indicava che il vantaggio veniva da come si avviano le repliche:
   MADEA divide la memoria libera in proporzione al *numero* di offerte di
   memoria per funzione, MAFW per valore marginale.
2. MADEA2 ha importato la regola "valore per MB" senza messaggi in più, ma
   aggiungendo β + γ alle offerte di memoria. È un cambio del modello di
   informazione, non solo della regola.
3. Le varianti "-b" ripristinano il modello di MADEA: il venditore vede solo
   offerte d'asta (MADEA2-b) o i propri prezzi (MAFW-b). Il confronto
   MADEA2 vs MADEA2-b e MAFW vs MAFW-b misura quanto del guadagno dipende
   dall'informazione in più.

## Dove sono nel codice

- `madea2.py`: opzione `solver_options.auction.memory_bid_value` = `"beta"`
  (default, MADEA2) o `"bid"` (MADEA2-b); `run_b`, `run_one_shot_b`,
  `run_one_shot_b_pg`.
- `decentralized_frankwolfe.py`: opzione `solver_options.frankwolfe.repair_order`
  = `"gain"` (default, MAFW) o `"price"` (MAFW-b);
  `hierarchical_frankwolfe.run_mafw_b`, `run_flat_pg_b`.
- Test: `tests/test_information_model.py` (in modalità `bid` le offerte di
  memoria non contengono β; con `price` la riparazione non cambia se il
  mittente alza il proprio β).
