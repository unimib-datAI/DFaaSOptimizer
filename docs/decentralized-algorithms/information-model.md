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

Con richieste intere, gli allocatori condivisi di MADEA e MADEA2 avviano gruppi
di repliche che aumentano la capacità intera servibile, considerando anche le
repliche già presenti. Partendo da zero, una replica da 0,889 richieste non viene
avviata da sola: ne occorrono due, se la memoria disponibile lo consente. MADEA redistribuisce
le quote inutilizzabili tra le funzioni con gruppi utili che entrano in memoria;
MADEA2 confronta il valore delle richieste servibili con il costo in memoria
dell'intero gruppo. La decisione usa soltanto i dati già disponibili al venditore
e si applica anche ai runner che riusano questi allocatori. Con richieste continue,
la capacità frazionaria resta utilizzabile.

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

## Conferma locale delle richieste di memoria (15/10/2026)

Nel ciclo MADEA (incluse MADEA2/-b e i wrapper), in Diffuse/PowD e nel
ciclo autonomo hierarchical-madea-legacy, le richieste formulate prima
dell'asta sono candidate locali. Dopo l'esito dell'asta e l'aggiornamento
della domanda, ciascun compratore conferma solo quelle per funzioni con
residuo maggiore della tolleranza del runner. Solo allora il venditore
riceve i messaggi confermati. La conferma riceve esclusivamente la riga
locale di domanda: il venditore non accede alla matrice globale `omega`.
MADEA conserva `(i,j,f)`; MADEA2/-b conserva `(i,j,f,d,v)`, aggiornando `d`
al residuo corrente (anche da zero a positivo dopo un rifiuto), e conserva
`v`, destinatari, ordine e molteplicità. Non si ripete l'asta dopo il ritiro.
La decisione di avviare repliche durante l'asta resta basata sulle candidate
originarie; una creazione diretta impedisce la seconda allocazione.

La conferma aggiunge una fase causale e computazionale parallela per compratore
solo quando esistono candidate, senza un nuovo costo di rete nel modello.
Best Response conferma dopo la decisione di ogni nodo con il target locale
aggiornato da reopt, mantenendo sequenziale il costo dello sweep.
One-shot e gerarchico classico mantengono l'allocazione nel solo ramo senza
offerte di capacità; Dual costruisce le richieste dalla domanda non assegnata
nel coordinamento. Non viene aggiunta loro una fase ridondante.

FaaS-MAPG ha un contratto diverso: la scarsità della proposta locale e del Cloud
può generare richieste anche quando la proposta è rifiutata e la soluzione
corrente serve già il nodo. Queste richieste esplorano una diversa decisione
locale, quindi non vengono filtrate con il residuo MADEA. I wrapper PG che
eseguono un ciclo MADEA ereditano invece la conferma di quel ciclo.
Richieste vive verso più venditori possono ancora creare capacità invenduta;
la conferma non prova convergenza universale e non cambia stop o budget.

## Varianti selective (21/10/2026)

`MADEA-selective` e `MADEA-one-shot-selective` usano LSP_detailed all'inizio e
mantengono x fisso. Le offerte di capacità restano quelle originali. Dopo l'asta,
il compratore conferma il residuo e seleziona un solo destinatario per funzione
tra le richieste di memoria candidate, usando beta propri, prezzi e latenza già
noti. Il venditore riceve soltanto `(i,j,f)`: nessuna quantità residua o utilità
privata è aggiunta. L'identità del destinatario è naturalmente osservabile.

I tentativi successivi esplorano gli altri candidati; il conteggio riparte al
cambiare del residuo e a ogni passo temporale. Gli stop per stallo possono essere
rinviati finché restano candidati, mai oltre i budget. One-shot-selective conferma
e gestisce le richieste residue anche dopo un'asta di capacità, mantenendo la sua
regola di accettazione senza rimpiazzare assegnazioni precedenti. La riduzione dei
messaggi per round non garantisce una riduzione del totale, da misurare.

## Greedy-RR (21/10/2026)

Metodo senza aste. Il mittente sceglie il vicino usando le proprie beta e la
capacità annunciata, e invia `(i,j,f,quantità)`. Il destinatario usa solo queste
richieste e la propria capacità, con accettazione round-robin; non legge beta,
gamma o alpha dei mittenti. Per le repliche riceve solo `(i,j,f)` e conta i vicini
richiedenti per funzione; domanda computazionale, RAM e repliche usate nella
scelta sono locali al destinatario. Nessun prezzo viene scambiato.

Il ruolo mittente/ricevente per funzione è fissato localmente all'inizio del
passo: un mittente annuncia capacità ricevente zero e rifiuta richieste di memoria
per la stessa funzione, senza pubblicare un nuovo campo di ruolo. Le conferme
riguardano solo traffico realmente accettato. Si mantengono x di LSP_detailed e
flussi già accettati; il budget limita i round. Questa regola può sprecare alcune
opportunità di scambio, ma previene il ping-pong per costruzione.
