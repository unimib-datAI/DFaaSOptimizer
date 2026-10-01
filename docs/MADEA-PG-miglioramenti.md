# Cambiamenti in MADEA-PG e nelle varianti collegate

Le modifiche rendono più rapidi i problemi locali e permettono a MADEA-PG di
dedicare più tempo alle mosse che migliorano il welfare. Ogni nodo continua a
decidere sulla base della propria utilità e dei vincoli locali. Il pool di
processi accelera l'esecuzione del simulatore senza introdurre un ottimizzatore
centrale che scelga le allocazioni.

## Gli algoritmi coinvolti

MADEA costruisce una soluzione attraverso aste fra nodi. MADEA-PG parte da quella
soluzione e prova ulteriori mosse locali: un nodo può cambiare le richieste servite,
quelle inoltrate e le repliche, rispettando gli impegni già presi verso gli altri.
La mossa viene accettata soltanto se migliora la sua utilità oltre la soglia prevista.

Le varianti gerarchiche aggiungono aste tra gruppi di nodi. Quelle a cicli alternano
una fase MADEA completa con una fase gerarchica; quelle con iterazioni per livello
ripetono le aste dello stesso livello finché trovano progressi. La variante
gerarchica MADEA-PG aggiunge il raffinamento locale alla soluzione così ottenuta.

La costruzione della gerarchia si arresta quando nessuna struttura acquista nuovi
nodi. Controllare soltanto l'unione delle strutture non sarebbe utile, perché
copre già la rete al primo livello. In una rete disconnessa, l'assenza di crescita
può significare aver raggiunto tutti i nodi della propria componente.

## Calcoli locali più semplici e veloci

Per i modelli supportati abbiamo sostituito la costruzione e la risoluzione di un
MILP con calcoli locali esatti. Un MILP è un problema di ottimizzazione che combina
variabili intere e vincoli lineari.

| Problema locale | Calcolo usato |
| --- | --- |
| `LSPr_x`, con traffico già fissato | Calcolo diretto delle repliche minime necessarie e del carico restante. |
| `LSP`, `LSPr` e varianti supportate | Programmazione dinamica, o DP: confronto delle scelte di repliche rispettando la RAM del nodo. |
| Modelli con repliche fissate | Calcolo separato per funzione, senza una ricerca sulle repliche. |

Questi calcoli conservano l'obiettivo e i vincoli del problema locale. Il solver
Pyomo resta disponibile per modelli personalizzati, parametri non supportati,
traffico continuo e casi troppo grandi per la DP. Nei pareggi fra soluzioni
ottime, DP e MILP possono scegliere allocazioni diverse: le aste successive
possono quindi seguire percorsi diversi, pur partendo da problemi locali risolti
correttamente.

MADEA-PG usa inoltre una proposta compatta. Prima ogni proposta copiava i dati
dell'intera rete e ricostruiva tutti i flussi. Ora passa alla DP soltanto i
parametri del nodo, il totale del traffico in ingresso già impegnato e i limiti
di inoltro. Se il caso non è supportato, conserva il percorso precedente.

## Un pool di processi che rimane caldo

Prima ogni batch creava nuovi processi, risolveva i modelli locali e li terminava.
Ora i runner condivisi creano il pool al primo batch parallelo e lo riutilizzano
fra iterazioni, cicli gerarchici e timestep. Lo chiudono alla fine dell'esecuzione
e lo terminano se si verifica un errore.

Con `-j -1` il numero di processi dipende dai core disponibili al processo, tenendo
conto dell'affinità CPU dove supportata. Non supera il numero di agenti del primo
batch parallelo. `-j N` permette di scegliere il numero manualmente; `-j 0`
mantiene l'esecuzione sequenziale. La dimensione del pool resta fissa durante il
run, anche quando i batch successivi contengono meno nodi.

Ogni blocco di lavoro porta dati, prezzi, modello e opzioni del solver aggiornati.
I processi restano caldi, ma gli input vengono sostituiti. Le risposte vengono
ricomposte nell'ordine originale degli agenti. Il trasferimento dei dati costa
comunque tempo: usare più processi non garantisce sempre il risultato più rapido.

Il pool risolve in parallelo i modelli locali indipendenti. Le mosse del
raffinamento PG continuano a essere accettate in sequenza, usando lo stato
aggiornato dopo ogni mossa. Il welfare complessivo coincide con la somma delle
utilità dei nodi nella convenzione usata dal codice; la sua misurazione ed
esportazione non diventano una nuova decisione centralizzata.

## Risultati delle prove planari

I confronti usano grafi planari connessi di grado 3. Sono misure su un solo
computer, con una ripetizione per variante e istanza; non garantiscono lo stesso
vantaggio su ogni rete o processore.

| Confronto | Risultato osservato |
| --- | --- |
| Proposta compatta rispetto alla precedente, entrambe con DP: 27 istanze complete e 54 coppie | Tempo totale 2,45 volte più rapido in mediana. Welfare migliore in 26 coppie, uguale in 28, mai peggiore; massimo guadagno del 43,3%. |
| Proposta compatta su tre timestep reali: 7 coppie di esecuzioni | Tempo totale 2,96 volte più rapido in mediana. Welfare migliore in 6 dei 21 confronti di timestep e uguale negli altri 15. |
| Batch da 80 nodi e 10 funzioni, con nuovi input e prezzi | Circa 174 ms con 4 processi caldi, contro 242 ms in sequenziale. La misura include trasferimento dei dati e ricomposizione dei risultati. |
| Runner MADEA-PG da 40 nodi e 5 funzioni, due timestep e 2 processi | 8,9 secondi con pool persistente, contro 163,5 secondi ricreando il pool a ogni batch: circa 18,4 volte più rapido. Allocazioni e welfare identici. |

Il vantaggio del pool caldo sul batch da 80 nodi esclude il primo avvio. Considerando
soltanto tre batch e includendo l'avvio, il sequenziale resta più rapido del pool
persistente: circa 0,73 secondi contro 1,52 secondi con 4 processi. Il pool conviene
quando l'esecuzione contiene abbastanza lavoro da compensare quel costo iniziale.

I guadagni di welfare della proposta compatta derivano dalle ulteriori mosse
migliorative completate nel tempo disponibile. Il pool, nel confronto completo
riportato sopra, accelera il calcolo senza cambiare la soluzione. Un esperimento
con inizializzazione locale Gurobi ha dato un welfare mediamente migliore del
3,64% rispetto alla DP, ma anche un caso peggiore: non è stato attivato come scelta
automatica. Nessuna di queste modifiche garantisce l'ottimo globale.

## Verifica finale e riferimenti

La suite completa ha prodotto **896 test passati e un fallimento preesistente**:
`test_integer_fixed_sum_traces_preserve_system_workload`. Il generatore
`fixed_sum` perde un resto nell'arrotondamento e produce 9 richieste invece delle
10 attese; generatore e test non sono stati modificati in questo intervento.
I nuovi test del pool e dei runner passano. Ruff, Mypy sui 13 file di produzione
coinvolti e il controllo del diff passano.

I test verificano riuso dei processi, aggiornamento degli input, cambio di modello
e solver, fallback GLPK, esecuzione sequenziale, chiusura dopo errori e supporto
alle viste degli agenti restituite dai loader. Gli smoke test MADEA-PG normale e
gerarchico verificano un solo pool per due timestep, soluzioni ammissibili e
allocazioni identiche al sequenziale.

I dettagli sono nel [report delle proposte compatte](../experiments/madea_pg_compact/results_2026_10_01/report.html)
e nei [risultati del pool persistente](../experiments/madea_pg_compact/results_warm_pool_2026_10_01/metrics.json).
Gli script di verifica sono nella cartella
[degli esperimenti](../experiments/madea_pg_compact/README.md).

## Parallelismo in PLASMA-Welfare

Anche PLASMA-Welfare può usare il pool persistente: `-j 0` mantiene il calcolo
sequenziale, `-j N` usa N processi e `-j -1` sceglie in base ai core logici
disponibili, senza superare il numero iniziale di nodi attivi. L'inizializzazione
dei nodi è indipendente. Durante i round, le DP vengono eseguite insieme soltanto
quando i rispettivi nodi e vicini non si sovrappongono; le transazioni che
condividono partecipanti mantengono l'ordine originale. Il processo principale
gestisce prenotazioni e commit, mentre ciascun worker riceve solo lo stato locale
del destinatario e le offerte riservate. Non viene aggiunta una scelta globale
delle allocazioni. La variante PLASMA originale resta sequenziale.

La prova su quattro istanze planari da 40 e 80 nodi, con cinque funzioni e
20 round, comprende 24 esecuzioni: due ripetizioni con zero, due e quattro
processi. Allocazioni, welfare, messaggi e scambi accettati sono identici. Con
quattro processi, i casi da 80 nodi impiegano circa la metà del tempo, includendo
l'avvio e la chiusura del pool. Le ultime due misure si sono brevemente
sovrapposte ai controlli finali: il risultato indica un beneficio, ma non è uno
studio completo della scalabilità hardware.

La suite dopo l'integrazione dei commit remoti e delle modifiche PLASMA ha dato
874 test passati, 27 saltati e il solo fallimento già noto di `fixed_sum`.
I sei test finali del parallelismo passano, compresa la cancellazione delle
offerte e la chiusura dei processi dopo un errore nella DP di negoziazione.
I [tempi per istanza](../experiments/madea_pg_compact/results_plasma_parallel_2026_10_01/summary.csv)
e i [dettagli della verifica](../experiments/madea_pg_compact/results_plasma_parallel_2026_10_01/metrics.json)
sono salvati insieme allo script riproducibile.
