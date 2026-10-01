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

## One-shot-PG

Il nuovo metodo `one-shot-pg` usa l'asta one-shot come punto di partenza e
applica la stessa fase di raffinamento di MADEA-PG a ogni timestep. Riutilizza
DP e proposte compatte. Le mosse vengono accettate dal nodo che le propone,
in base alla sua utilità e alle capacità residue annunciate dai vicini,
preservando gli impegni in ingresso. Il welfare globale serve solo per
misurare ed esportare il risultato della fase aggiunta. Il simulatore
gestisce i turni; non è un'implementazione di un protocollo di rete distribuito.

La configurazione del raffinamento è condivisa: `solver_options.madea_pg`.
In assenza di un limite esplicito, il budget è 0,25 secondi per nodo;
viene limitato dal tempo nativo restante dell'asta. Restano al massimo
cinque sweep e la soglia di miglioramento `epsilon`. Un budget nullo
mantiene la soluzione one-shot. Il limite controlla l'avvio delle proposte:
una DP già avviata termina normalmente. Con fallback GLPK, meno di un
secondo residuo impedisce di avviare una proposta; per istanze piccolissime
si può impostare un `time_limit` maggiore.

```sh
MPLCONFIGDIR=/tmp/dfaas-mpl .venv/bin/python one_shot_pg.py -c experiments/madea_pg_compact/config.json -j 0 --disable_plotting
```

Nel batch si seleziona con `--methods one-shot-pg`. I risultati hanno la
colonna `One-shot-PG`, un `refinement.csv` per timestep e il `runtime.csv`.
Il metodo originale `faas-madea-1s` mantiene comportamento ed export
precedenti. I test verificano ammissibilità, welfare non decrescente,
identità con la baseline a budget nullo e mosse identiche anche alterando
il misuratore globale di welfare. La variante può essere ripresa nel batch
indipendentemente dalla baseline ed è selezionabile nei job remoti.

Il [piccolo confronto planare](../experiments/madea_pg_compact/results_one_shot_pg_2026_10_01/report.md)
comprende otto istanze e 24 esecuzioni. One-shot-PG migliora la baseline in
tutti i casi, con un guadagno percentuale medio del 20,1%. Rispetto a MADEA-PG,
il welfare è mediamente inferiore dello 0,38%, ma è più alto in tre casi;
il rapporto mediano dei tempi è 0,686, cioè circa il 31% di tempo in meno.
È una prima misura con due seed e una sola esecuzione per metodo e istanza.

La suite completa dopo questa aggiunta ha dato 902 test passati e tre
fallimenti preesistenti: il noto `fixed_sum` e due test gerarchici dovuti
alla chiamata `define_bids(..., delta=...)`. Questi ultimi si riproducono
anche caricando il runner one-shot da HEAD, senza le modifiche PG.
I nomi completi e il commit di riferimento sono nella
[verifica](../experiments/madea_pg_compact/results_one_shot_pg_2026_10_01/verification.json).

One-shot-PG è stato poi eseguito da solo sui 35 input del confronto esteso,
senza rilanciare gli altri quattro metodi. La
[tabella aggiornata](../experiments/madea_pg_compact/results_families_five_methods_2026_10_01/report.md)
include quattro seed per i 24 casi standard, otto casi di carico modificato
e tre sequenze temporali. In tutti i 41 timestep il punto di partenza coincide
con one-shot e il welfare non peggiora. Le colonne precedenti sono conservate.
Nei casi standard il guadagno medio rispetto a one-shot è +20,4%; rispetto
a MADEA-PG il welfare medio per istanza è inferiore dello 0,52%, con un rapporto
mediano dei tempi pari a 0,524. Le misure nuove provengono da una sessione
successiva sulla stessa macchina, come indicato nel protocollo.

## Hierarchical-one-shot-PG

Il metodo `hierarchical-one-shot-pg` riutilizza il runner gerarchico one-shot
con il motore iterativo dei livelli già impiegato dalla variante MADEA.
Mantiene l’allocazione locale iniziale durante le aste, accumula le assegnazioni
accettate senza sostituire quelle precedenti, attraversa i livelli gerarchici
e infine applica una sola fase PG per timestep. La profondità massima è
`max_hierarchy_depth`, pari a 3 se non specificata.

Le strutture ampliano il coordinamento, ma gli inoltri restano tra vicini
diretti: non vengono autorizzati nuovi collegamenti tra nodi lontani.
La fase PG aggiunta conserva gli impegni in ingresso e decide ogni mossa
con l’utilità del nodo proponente, senza un criterio globale di accettazione.
Rimane la selezione storica dell’incumbent tramite welfare già presente
nel runner gerarchico originale; la nuova fase PG non aggiunge questo meccanismo.

Il budget PG usa le stesse opzioni di one-shot-PG. Il tempo residuo viene
calcolato sul tempo effettivamente trascorso nel timestep, includendo aste e
gerarchia. La versione aggiornata riserva al PG il budget richiesto, fino a
metà del limite totale: con 80 nodi e limite di 40 secondi, la gerarchia dispone
di 20 secondi e il PG di altri 20. Se il loop si arresta prima, PG mantiene
comunque il proprio limite configurato. Il controllo avviene tra round e
proposte: una singola operazione già avviata può consumare parte della riserva
o terminare oltre il limite.

Il loop esterno si arresta anche dopo due round consecutivi con assegnazioni
e repliche immutate e variazioni dei prezzi entro l’epsilon dell’asta più la
tolleranza numerica. Controlla anche la variazione della penalità di fairness,
quando attiva. Sono segnali ricavabili dagli aggiornamenti locali dei nodi;
questo nuovo criterio non usa il welfare globale. Si tratta di una soglia
euristica: piccole variazioni dei prezzi potrebbero produrre effetti dopo
altri round. Non certifica convergenza o ottimalità. Con budget PG nullo o
zero sweep, riserva e nuovo arresto sono disattivati.

```sh
MPLCONFIGDIR=/tmp/dfaas-mpl XDG_CACHE_HOME=/tmp/dfaas-test-cache .venv/bin/python one_shot_pg.py --variant hierarchical -c experiments/madea_pg_compact/config.json -j 0 --disable_plotting
```

Nel batch si seleziona con `--methods hierarchical-one-shot-pg`; è disponibile
anche nei job remoti. Gli output usano la colonna `HierarchicalOneShotPG` e
comprendono `refinement.csv` e `runtime.csv`. Il primo registra anche numero
di round gerarchici, budget della gerarchia e riserva richiesta per il PG.
Il runner gerarchico originale
mantiene il motore non iterativo e il raffinamento disattivato per default.
Sono state aggiornate le sue chiamate alle funzioni d’asta condivise:
argomenti e risultati di `define_bids`, `evaluate_bids` e criterio d’arresto,
oltre all’opzione `unit_bids`. Le aste rispettano gli impegni già accettati.
Il criterio d’arresto riceve anche il numero di repliche appena avviate:
non si ferma prima che una successiva asta possa utilizzarle.

I test reali controllano due timestep, allocazioni dei livelli superiori,
ammissibilità, integrità dei flussi, export coerente con il welfare,
identità con la baseline gerarchica allo stesso motore quando il budget PG
è nullo e decisioni PG identiche alterando l’osservatore globale del welfare.
Verificano anche l’arresto dei round stagnanti su un’istanza planare da 40 nodi,
la prosecuzione quando cambiano i prezzi e il tempo riservato al PG, simulando
solo l’orologio del coordinamento e mantenendo reali i modelli locali.
La suite completa dopo questi cambi ha dato **910 test passati e un fallimento preesistente**, in 118,29 secondi:
`tests/test_review_optimization_regressions.py::test_integer_fixed_sum_traces_preserve_system_workload`.
I due precedenti fallimenti gerarchici sono risolti dall’aggiornamento delle chiamate.

Il [primo confronto a sei metodi, prima di riserva e arresto per stagnazione](../experiments/madea_pg_compact/results_families_six_methods_2026_10_01/report.md)
aggiunge soltanto questa variante sugli stessi 35 input e 41 timestep,
preservando tutte le cinque colonne precedenti. La prima finestra di 30 minuti
ha coperto 34 casi; l’ultima sequenza è stata completata separatamente in
60,23 secondi, senza cambiare il codice. Tutte le soluzioni sono ammissibili.
Nei 24 casi standard, rispetto a one-shot-PG il guadagno percentuale medio
per istanza è **−1,78%**, con un rapporto mediano dei tempi di **6,24×**.
La gerarchia dà risultati migliori nelle istanze da 40 nodi, ma a 80 e
160 nodi il costo del coordinamento esaurisce il budget: PG migliora la
soluzione in 14 timestep, mentre in altri 27 non può avviare proposte.
Nei casi stress il welfare medio per istanza è −11,83% rispetto a one-shot-PG;
nelle tre sequenze temporali è +1,37%, con tempo mediano 8,45 volte maggiore.
Questa prima integrazione funziona, ma non conviene complessivamente
rispetto a one-shot-PG con i budget scelti. Non è stata misurata qui la
variante gerarchica MADEA-PG: la sua sostituzione non viene quindi validata
da un confronto diretto tra le due versioni gerarchiche.

Il [confronto aggiornato dopo arresto per stagnazione e riserva PG](../experiments/madea_pg_compact/results_hierarchical_balanced_2026_10_01/report.md)
ha rieseguito solo questa colonna sugli stessi 35 input, in 10 minuti e
11 secondi inclusa la validazione. Le altre cinque colonne e i loro aggregati
sono identici al confronto precedente. Nei 41 timestep l’incumbent prima
del PG è rimasto identico; il PG ora migliora tutti i 41, senza budget nullo.
Il loop usa 4–6 round (mediana 4), contro 12–100 (mediana 66), contando il
primo round indicizzato con zero.

Rispetto alla versione gerarchica precedente, il guadagno percentuale medio
per istanza è +9,95% sul carico standard, +20,30% nello stress e +3,93%
nelle sequenze temporali. Il rapporto mediano tra tempo precedente e nuovo
è rispettivamente 4,90×, 5,57× e 7,28×. Non si osservano peggioramenti sui
35 casi, ma il criterio resta euristico.

Rispetto a one-shot-PG, il welfare medio per istanza cresce del 7,42% sul
carico standard, del 5,10% nello stress e del 5,02% nei timestep temporali.
Il rapporto mediano dei tempi gerarchica/one-shot-PG è 1,29×, 1,21× e 0,99×.
La variante gerarchica aggiornata ha il welfare maggiore in 15 dei 24 casi
standard; PLASMA-Welfare negli altri 9. La media dei guadagni rispetto a
MADEA resta vicina fra questi due metodi (25,98% e 26,27%), quindi non emerge
un metodo migliore in ogni istanza.

## Gerarchico MADEA-PG: prove separate dei controlli del tempo

Il [confronto degli interventi sul gerarchico MADEA-PG](../experiments/madea_pg_compact/results_hierarchical_madea_ablation_2026_10_01/report.md)
misura separatamente riserva per il PG, tempo effettivo e arresto locale su
otto input planari comuni. Sono rimasti soltanto tempo effettivo e riserva:
la variante aggressiva dell’arresto riduceva il welfare, mentre quella
conservativa non interveniva nei casi misurati.

La riserva è il budget PG richiesto, limitato a metà del tempo totale del
timestep. Si applica al ciclo d’asta senza modificare le opzioni del solver
locale, mantenendo GLPK compatibile anche con riserve frazionarie. I nuovi
controlli sono disattivati per non-PG, budget PG nullo e zero sweep. MADEA-PG
piatto e gerarchico one-shot-PG mantengono il comportamento precedente.
Il report contiene valori assoluti, varianti scartate e limiti del confronto:
non aggiorna la tabella precedente dei 35 casi usando solo questo campione.

## Istanze con α sempre maggiore di β

Il vecchio intervallo di β/α, 0,1–1,5, consentiva che l’inoltro ricevesse un
coefficiente superiore all’esecuzione locale. Per il nuovo confronto è stato
scelto l’intervallo 0,1–0,9. Il driver `rerun_alpha_gt_beta.py` crea nuove copie
delle 35 istanze, verifica α > β e mantiene identici grafi, risorse, α, γ e
tracce. Ricalcola anche δ, derivato da β dal generatore. Le istanze precedenti
restano disponibili; il codice degli algoritmi non cambia. Il confronto viene
rieseguito per tutte le sette colonne su dodici input comuni, entro 30 minuti.

Il [report con α > β](../experiments/madea_pg_compact/results_families_alpha_gt_beta_2026_10_01/report.md)
registra 84 esecuzioni riuscite e 112 risultati per timestep, in 22,07 minuti
inclusa la preparazione. Sugli otto casi standard, rispetto a MADEA i guadagni
medi sono +0,58% per MADEA-PG, +0,84% per one-shot-PG e +1,02% per il gerarchico
one-shot-PG. Il gerarchico MADEA-PG dà +0,59%: su questo campione aggiunge poco
welfare al metodo piatto. PLASMA-Welfare ha il welfare maggiore nei due casi
stress; nel caso a carico 1,5 dà +8,39% rispetto a MADEA. I valori assoluti,
i tempi e i risultati temporali sono riportati separatamente. La suite completa
ha dato 917 test passati e il fallimento preesistente di fixed_sum.
