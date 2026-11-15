# cercaPattern - Analisi forense "cross" di una colonna target in un CSV

Pipeline Python (pandas / scikit-learn / mlxtend / xgboost / shap / plotly) che analizza un CSV qualsiasi,
una riga per record. Si scelgono una colonna e uno o più valori, cioè il *caso da spiegare* (per esempio
`ESITO_ALFA = ERROR`), e lo script produce perimetro, pattern, cluster, anomalie e deduzioni sul **perché**
quel caso si verifica.

Il report si apre con le conclusioni ("In breve"): la causa più probabile, quanti problemi distinti ci
sono e quanto pesa ciascuno, cosa succede nel tempo, quanto i dati disponibili spiegano il caso.

Lo script non presuppone un tracciato. Il ruolo delle colonne (data e ora, misura principale, sessione...)
e i gruppi di colonne che descrivono l'esito di uno stesso sistema vengono riconosciuti dai nomi e dai
dati: vedi "Struttura del tracciato".

## Installazione

```bash
uv venv --python 3.12 .venv            # oppure python3.12 -m venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Su Windows l'interprete del venv è `.venv\Scripts\python.exe` (negli esempi sotto sostituisce `.venv/bin/python`).

## Uso

```bash
.venv/bin/python run_analysis.py dati.csv --output-dir output
```

Dopo il caricamento del CSV lo script mostra l'elenco delle colonne (cardinalità, % nulli, esempi), i
**candidati plausibili** (colonne con pochi esiti in cui un valore minoritario ha l'aspetto di un
errore, di un blocco o di un livello alto: `ESITO_BETA = KO`, ...) e chiede:

1. **quale colonna analizzare** (numero o nome; invio = il candidato più plausibile);
2. **quali valori di quella colonna definiscono il caso da spiegare** (uno o più, separati da virgola; invio = il valore più plausibile della colonna).

Tutte le misure delle nove fasi, i grafici, i report e i nomi dei file sono costruiti su quella scelta: l'analisi è "cross", funziona su qualunque colonna categorica, che sia l'esito di un sistema (`ESITO_BETA = KO`, `GAMMA_AZIONE = DENY`) o una colonna descrittiva (`TIPO_RICHIESTA = RESO`). Quando il target è l'esito di un sistema, le altre colonne di quel sistema (dettaglio, score) vengono escluse dai modelli perché descrivono l'esito stesso, e il sistema sparisce dal confronto con gli altri. Quando il target non appartiene a un sistema, le ipotesi di causa della fase 9a vanno lette come indicazioni generiche (il report lo segnala).

Per lanci non presidiati (batch, scheduler) si passano colonna e valori da riga di comando. Con `--non-interactive` ciò che manca viene scelto tra i candidati plausibili, e il log dice quale:

```bash
.venv/bin/python run_analysis.py dati.csv -o out --target-col ESITO_BETA --target-value KO
.venv/bin/python run_analysis.py dati.csv -o out --target-col TIPO_RICHIESTA --target-value RESO,PRENOTAZIONE
.venv/bin/python run_analysis.py dati.csv -o out --non-interactive     # il candidato più plausibile
```

Se la colonna indicata non esiste il messaggio di errore elenca i candidati plausibili. Se nessuna colonna ha l'aspetto di un esito il target va indicato a mano. Se tutte le righe (o nessuna) ricadono nel caso scelto l'analisi si ferma con un messaggio: non c'è nulla con cui confrontarle.

Opzioni principali (`--help` per l'elenco completo):

| Opzione | Default | Note |
|---|---|---|
| `--output-dir`, `-o` | `<cartella del csv>/<nome csv>_analisi` | con più CSV e `-o` una sottocartella per file |
| `--pause` | auto | `yes`/`no`/`auto`: attende INVIO prima di chiudere (auto = solo come eseguibile) |
| `--open-report` | auto | `yes`/`no`/`auto`: apre il report HTML dell'analisi appena conclusa (auto = solo come eseguibile) |
| `--target-col`, `--target-value` | chiesti a terminale | colonna e valore/i (anche `A,B`) che definiscono il caso analizzato |
| `--non-interactive`, `-y` | - | nessuna domanda: per ciò che manca usa il candidato più plausibile |
| `--group NOME=REGEX` | - | dichiara un sistema: le colonne che corrispondono alla regex ne fanno parte (ripetibile) |
| `--no-auto-groups` | - | non riconosce i sistemi dai nomi delle colonne (restano quelli dichiarati con `--group`) |
| `--critical-values VAL ...` | - | valori da considerare esito critico degli altri sistemi, oltre a quelli già noti (`SOSP*` = parole che iniziano così) |
| `--family [COLONNA:]NOME=REGEX` | - | dichiara una famiglia di valori da analizzare insieme (ripetibile) |
| `--options FILE` | - | legge altre opzioni da un file (vedi "Opzioni su file") |
| `--no-local-options` | - | ignora il file `cercaPattern.opzioni` accanto allo script |
| `--sep`, `--encoding` | auto | separatore (anche di più caratteri, es. `"||"`) ed encoding rilevati automaticamente |
| `--min-support`, `--min-confidence` | 0.005, 0.2 | soglie dei pattern; si adattano da sole se il caso target è raro (vedi note) |
| `--max-cardinality` | 40 | max valori distinti per usare una colonna come categorica nei modelli |
| `--sample-rows` | 250000 | campione stratificato per albero e XGBoost; i pattern sono sempre contati su tutte le righe |
| `--dims COL ...` | - | dimensioni aggiuntive per i tassi |
| `--exclude COL ...` | - | colonne da tenere fuori da pattern e modelli (codifiche del target, campi noti come conseguenza) |
| `--no-shap` | - | salta SHAP (solo gain XGBoost) |
| `--plotly-inline` | - | HTML autonomo senza CDN (~4 MB) |
| `--no-progress` | - | disattiva la barra di avanzamento (si spegne da sola se stderr non è un terminale) |
| `--log-level` | INFO | con `WARNING` restano a video solo le barre di avanzamento |
| `--version` | - | versione del pacchetto |

Durante l'esecuzione ogni fase ha la propria barra di avanzamento (12 barre: caricamento, setup, fasi 1-9b), che resta a video con tempo ed esito: verde completata, rossa fallita, gialla saltata. Le fasi lunghe mostrano i sotto-passi: fase 3 una tacca per dimensione analizzata, fase 4 regole / correlazioni / albero / XGBoost+SHAP, fase 9b summary / PNG / plotly / CSV / Excel / HTML. I log compaiono sopra le barre senza sporcarle.

Per un collaudo senza dati reali:

```bash
.venv/bin/python generate_sample_data.py --rows 40000 --out sample_dati.csv
.venv/bin/python run_analysis.py sample_dati.csv -o output_sample
```

Il dataset sintetico simula richieste a una piattaforma, ciascuna valutata da più sistemi (`ALFA`, `BETA`, `GAMMA`...), con quattro cause "nascoste" del caso `LIVELLO_ALFA = ERROR` che l'analisi deve ritrovare.

Tempi misurati sul dataset sintetico (portatile 6 core): 40.000 righe circa 40 secondi, 1.000.000 di righe circa 4 minuti con 2,7 GB di memoria (vedi "Prestazioni").

## Struttura del tracciato: ruoli e sistemi

L'analisi ricava la struttura del CSV senza configurazione. Ciò che riconosce è riportato nella tabella "Ruoli e sistemi riconosciuti" del report (tab `00_mappa_colonne` dell'Excel).

**Ruoli.** Alcune colonne hanno un compito preciso; si riconoscono da parole generiche nel nome (maiuscole e minuscole non contano):

| Ruolo | Nomi riconosciuti (esempi) | A cosa serve |
|---|---|---|
| data e ora | `DATETIME`, `TIMESTAMP`, `DATA_ORA`; oppure `DATA` + `ORA`/`TIME`; oppure `TIME_KEY` | fase 5 |
| misura principale | prima colonna numerica con `IMPORTO`, `AMOUNT`, `PREZZO`, `TOTALE`, `QTY`... | priorità delle anomalie, descrizione dei cluster |
| sessione / contesto | nome con `SESSION` (`SESSIONE_ALFA`, `sessionId`) | ipotesi H5 |
| regola applicata | `REGOLA`, `RULE_NAME`, `POLICY` | ipotesi H6 |
| decisione complessiva | `DECISIONE`, `AZIONE_FINALE`, `FINAL_ACTION` (non di un singolo sistema) | casi ad alta criticità |
| identificativo del record | colonna con un valore diverso per ogni riga | etichette dei grafici |

Tutto il resto viene dai dati: ogni colonna con pochi valori distinti è una dimensione di analisi; le colonne con molti valori distinti sono dati *raccolti* (identificativi, token, coordinate, testi), di cui conta la presenza; le colonne numeriche continue sono misure e vengono divise in fasce, gli identificativi numerici no.

**Sistemi.** Un sistema è un gruppo di colonne che descrive l'esito di uno stesso componente. Il nome di una colonna come `ESITO_ALFA` o `BETA_SCORE` è fatto di una parola che dice *cosa* contiene (esito, score, livello, stato, azione, decisione...) e di un nome proprio: il nome proprio identifica il sistema. Ne fanno parte anche le colonne che ripetono quel nome accanto a parole di contesto (`DETTAGLIO_ALFA`, `SESSIONE_ALFA`, `ALFA_CODE`). Nomi fatti solo di parole generiche (`ESITO_FINALE`, `STATUS_CODE`, `ID_UTENTE`) non indicano alcun sistema.

Riconoscere i sistemi serve a tre cose:

- le colonne dello stesso sistema del target descrivono l'esito, non lo spiegano: dettaglio e score escono dai modelli, e la colonna di dettaglio viene classificata in cause tecniche (fase 6);
- gli altri sistemi dicono se il problema è isolato o condiviso: se anche loro falliscono sulle stesse righe la causa è a monte (ipotesi H4);
- gli esiti critici degli altri sistemi (`DENY`, `BLOCK`, `HIGH`, `KO`...) indicano le righe del caso da guardare per prime (fase 8). Altri valori si aggiungono con `--critical-values`.

Se i nomi delle colonne non seguono questo schema i gruppi si dichiarano a mano, e hanno la precedenza:

```bash
.venv/bin/python run_analysis.py dati.csv --group ALFA="^A_|ALFA" --group BETA="^B_"
.venv/bin/python run_analysis.py dati.csv --no-auto-groups        # nessun sistema: ogni colonna vale per sé
```

## Opzioni su file

Ciò che lo script non può ricavare dai dati (gruppi con nomi irregolari, famiglie di valori, esiti critici, il target abituale) si passa con le opzioni. Per non riscriverle a ogni lancio si mettono in un file, scritte come sulla riga di comando:

```text
# opzioni per i CSV di questo progetto
--group 'ALFA=^A_|ALFA'
--critical-values 'SOSP*' FERMO
--family 'TIPO_RICHIESTA:ORDINI=^ORDINE|CONSEGNA'
--family 'CHIUSURE=RESO|DISDETTA'
--target-col ESITO_ALFA
--target-value ERROR
```

- Una o più opzioni per riga; le righe vuote e ciò che segue `#` sono ignorati. Un valore con spazi o barre rovesciate va tra apici singoli.
- Il file si indica con `--options FILE`. Un file di nome `cercaPattern.opzioni` nella cartella di `run_analysis.py` (o dell'eseguibile) viene letto da solo: è il modo di passare opzioni anche quando i CSV vengono trascinati sull'eseguibile. `--no-local-options` lo ignora.
- Le opzioni del file valgono come default: quelle date sulla riga di comando hanno la precedenza, mentre le opzioni ripetibili (`--group`, `--family`) si sommano. Lo script scrive a video da quali file ha letto.
- `cercaPattern.opzioni` e i file `*.opzioni` sono esclusi da git: descrivono i dati di chi usa lo script, non lo script.

Una **famiglia** raggruppa valori che vanno letti insieme. Senza dichiarazioni lo script forma da solo le famiglie dei valori che iniziano con la stessa parola (`ORDINE_STANDARD`, `ORDINE_ESTERO`); con `--family NOME=REGEX` si dichiarano gruppi che dal nome non si vedono. La regola si applica ai valori di tutte le dimensioni descrittive, oppure a una sola colonna con `--family COLONNA:NOME=REGEX`; maiuscole e minuscole non contano.

## Eseguibile e drag & drop (py2exe / pyinstaller)

Lo script accetta uno o più CSV come argomenti posizionali: su Windows **trascinare i file sull'eseguibile** equivale a lanciarlo con i loro percorsi. In quel caso:

- l'output va accanto a ciascun CSV, nella cartella `<nome csv>_analisi` (nessun `-o` richiesto);
- la scelta di colonna e valori viene chiesta nella finestra della console (l'eseguibile va costruito come applicazione **console**, non GUI, altrimenti non c'è stdin);
- a fine analisi il report HTML si apre nel browser e la finestra attende INVIO prima di chiudersi, così anche gli eventuali errori restano leggibili;
- con più file trascinati insieme vengono analizzati in sequenza; un file che fallisce non blocca gli altri (codice di uscita 1 se almeno uno fallisce, 2 se nessun file esiste);
- i caratteri che la console di Windows non sa rappresentare vengono sostituiti da `?` invece di interrompere la stampa del riepilogo.

Per forzare questi comportamenti anche da terminale: `--pause yes`, `--open-report yes`. Per disattivarli nell'eseguibile: `--pause no`, `--open-report no` (utile negli scheduler, insieme a `--non-interactive`).

Nota per il packaging: xgboost, shap e scikit-learn caricano librerie native e dati a runtime; vanno inclusi come `data_files`/`packages` nella configurazione di py2exe, e `plotly` richiede la cartella `plotly/package_data`.

## Output (cartella `--output-dir`)

Ogni file ha come prefisso il target analizzato, in minuscolo (`<colonna>_<valori>`, es. `esito_alfa_error`, `tipo_richiesta_reso`), così analisi di colonne diverse possono convivere nella stessa cartella.

| File | Contenuto |
|---|---|
| `<prefisso>_report.xlsx` | report multi-tab (executive, findings, tabelle delle fasi 1-9) |
| `<prefisso>_report.html` | report interattivo (plotly) con executive summary e tabelle |
| `<prefisso>_executive_summary.md` | executive summary in Markdown ("In breve" + 10 punti) |
| `<prefisso>_righe_target.csv` | sole righe del caso analizzato |
| `<prefisso>_patterns.csv` | pattern, copertura cumulativa e regole dell'albero decisionale |
| `<prefisso>_clusters.csv`, `<prefisso>_clusters_summary.csv` | assegnazione cluster per riga e sintesi |
| `<prefisso>_anomalies.csv` | ranking anomalie (IF, LOF, DBSCAN, consenso, casi unici, alta criticità) |
| `<prefisso>_findings.csv`, `<prefisso>_cause_probabili.csv` | deduzioni e ipotesi di root cause con score |
| `<prefisso>_dizionario_dati.csv` | dizionario dati inferito |
| `png_<prefisso>/*.png` | grafici statici |

I CSV sono in UTF-8 con BOM e separatore `;`: Excel li apre con le lettere accentate corrette. Nell'Excel le due tabelle con una riga per riga target (assegnazioni ai cluster, anomalie) contengono le prime 5.000 righe: il dettaglio completo è nei CSV. I testi del CSV che iniziano per `=` restano testo (non diventano formule) e i caratteri di controllo vengono rimossi.

Nelle tabelle e nei testi "errori" e "tasso di errore" indicano le righe del caso analizzato e la loro incidenza, qualunque sia il valore scelto.

## Fasi

1. **Profiling** - tipi inferiti, nulli, cardinalità, campi costanti, distribuzioni.
2. **Dizionario dati** - significato, dominio e ruolo nell'analisi di ogni colonna (parole generiche nel nome, sistema di appartenenza, euristiche sul contenuto).
3. **Analisi errori** - tasso di errore con intervallo di confidenza, lift, quota errori e Cramér's V per ogni colonna con pochi valori distinti (più quelle chieste con `--dims`). Ogni valore è marcato come significativo o no dopo la correzione per test multipli.
4. **Pattern discovery** - pattern orientati al target con conteggi esatti, **copertura cumulativa** (i problemi distinti in ordine di peso e il tasso residuo fuori dai pattern), correlazioni, albero decisionale esplicativo, XGBoost + SHAP con AUC su righe tenute fuori dall'addestramento.
5. **Temporale** - fasce orarie ricorrenti, heatmap giorno x ora, serie storiche, **finestre di degrado** statisticamente fondate con il loro ambito (trasversale o localizzato su un valore), **cambi di livello** del tasso giornaliero con il valore che li guida.
6. **Funzionale** - campi mancanti (per i dati raccolti, la sessione e la regola applicata: quanto l'assenza è associata al caso); relazione con gli altri sistemi (co-fallimento vs fallimento isolato); classificazione della colonna di dettaglio in cause tecniche; famiglie di valori (valori che iniziano con la stessa parola o dichiarati con `--family`, analizzati insieme); passaggi facoltativi (colonne in cui un valore vuoto o negativo dice "passaggio non eseguito").
7. **Clustering** - KMeans sui profili delle righe target (one-hot pesato per importanza), k scelto via silhouette, etichette automatiche dai campi discriminanti.
8. **Anomaly detection** - Isolation Forest, LOF, DBSCAN con consenso; errori unici; errori ad alta criticità (esito critico negli altri sistemi / misura principale >= p99).
9. **Report** - regole deduttive delle cause probabili (H1..H9 con score) ed executive summary.

Le ipotesi: H1 infrastruttura, H2 alimentazione dati, H3 integrazione applicativa, H4 problema a monte, H5 sessione, H6 configurazione, H7 passaggio facoltativo, H8 osservabilità, H9 rilascio / cambio di configurazione. Le prove vengono incrociate tra fasi: una finestra di degrado trasversale a tutte le dimensioni rafforza H1, una localizzata su un solo flusso rafforza H3, un salto persistente del tasso da una certa data rafforza H9, i dettagli di tipo timeout *dentro* le finestre di degrado rafforzano H1.

## Come ragiona (note di progettazione)

- **Pattern contati su tutte le righe.** Una regola `A -> target` può superare il supporto minimo solo se `A` è frequente tra le righe target: le combinazioni candidate si cercano quindi (FP-growth) sulle sole righe target, poi righe ed errori di ogni candidato vengono contati in modo esatto su tutto il dataset con insiemi di bit. Nessun campionamento: confidenza e lift non sono distorti, e la fase è molte volte più veloce di un FP-growth su tutte le righe.
- **Solo ciò che è significativo.** Con centinaia di valori e migliaia di combinazioni a confronto, un tasso alto su venti righe è quasi sempre rumore. Tassi per dimensione, pattern, ore anomale e passi della copertura cumulativa passano un test binomiale con correzione per test multipli (FDR di Benjamini-Hochberg); le confidenze sono accompagnate dal limite inferiore dell'intervallo di Wilson. Un pattern più lungo sopravvive solo se migliora in modo significativo quello più corto che contiene.
- **Soglie che si adattano al target.** `--min-support` è una quota di tutte le righe: se il caso target è più raro, la soglia scende al 5% delle righe target (minimo 3 righe, e mai sotto l'1% delle righe target). Allo stesso modo la confidenza minima scende a 3 volte il tasso globale quando il 20% è irraggiungibile.
- **Copertura cumulativa.** A ogni passo si sceglie il pattern con più errori in eccesso tra le righe *non ancora spiegate*. Un pattern che è solo l'ombra di uno già scelto (un'APP che ha tanti errori perché lì manca un campo) non viene contato due volte. Il risultato risponde a "quanti problemi distinti ci sono, quanto pesa ciascuno, quale tasso resterebbe a problemi risolti". Anche "dentro una finestra di degrado" partecipa come condizione, così cause legate agli attributi e cause legate al tempo sono pesate insieme.
- **Pattern equivalenti.** Due pattern sono fusi in una riga solo se selezionano esattamente le stesse righe (es. `APP=412` e `CANALE=MOBILE`), non se hanno per caso gli stessi conteggi.
- **Finestre di degrado.** Un'ora è anomala se i suoi errori superano in modo significativo quelli attesi dato il suo volume e il tasso dei giorni vicini (3,5 giorni prima e dopo). Il tasso "di zona" segue i cambi di livello duraturi, che sono analizzati a parte, e non è gonfiato dal degrado stesso.
- **Cambi di livello.** Segmentazione binaria della serie giornaliera (senza le ore di degrado): un taglio è accettato se è significativo e se anche la mediana dei tassi giornalieri si sposta di almeno il 50%, così il singolo giorno anomalo non passa per un cambio di livello.
- **Ambito di finestre e cambi di livello.** Si cerca il valore che raccoglie quasi tutti gli errori senza essere quasi tutto il volume, tra le dimensioni descrittive (quelle che non riportano l'esito di un sistema), a partire dalle più associate al target.
- **Date con fuso orario** (`2026-10-26 17:43:19+00:00`, `...Z`, `+0200`, anche offset misti solare/legale): l'offset viene scartato e si usa l'ora di orologio scritta nel file, così le fasce orarie coincidono con quanto si legge nel CSV. Il log riporta gli offset trovati.
- **Chiavi temporali compatte** (`yyyymmdd`, `yyyymmddHH`, `yyyymmddHHMM`, `yyyymmddHHMMSS`): il formato dipende dalla lunghezza del valore. Sono riconosciuti anche epoch Unix in secondi e millisecondi, formati italiani e US (`mm/gg/aaaa` solo se il formato italiano non regge sull'intera colonna). Date sentinella lontanissime dal periodo dei dati vengono ignorate e segnalate.
- **Numeri**: la convenzione (virgola o punto decimale, separatore delle migliaia) è decisa sull'intera colonna, non valore per valore: `0.123` in una colonna di punteggi resta 0,123.
- **Formato del file**: encoding rilevato sui primi 256 KB (UTF-8 con o senza BOM, UTF-16, cp1252, latin-1) con ripiego automatico se più avanti compaiono byte non validi; separatore scelto in base alla coerenza del numero di campi, rispettando le virgolette. Nomi colonna duplicati dopo la rimozione degli spazi vengono rinominati (`A`, `A_2`).
- Le colonne **conseguenza** del caso (es. lo score e il dettaglio dello stesso sistema, campi vuoti nel >=99% delle righe target)
  sono escluse automaticamente dai modelli per evitare pattern tautologici (vengono comunque mostrate in fase 3 e 6).
- **Alias del target.** Una colonna che determina *ogni* valore della colonna target (una codifica più fine dello stesso
  dato, es. un codice tipo quando il target è `TIPO_RICHIESTA`) viene esclusa dai modelli: spiegherebbe il caso solo in modo
  tautologico. Una colonna che separa perfettamente il solo caso analizzato (ogni suo valore sta tutto dentro o tutto fuori)
  può essere invece la causa deterministica ("APP=412 va sempre in errore") oppure una codifica: i dati non bastano a
  distinguere, quindi resta nei modelli, il report la segnala ("SEPARATORE PERFETTO") e si può escludere con `--exclude`.
- Per le colonne ad alta cardinalità (identificativi, token, sessione, coordinate) i modelli usano la feature derivata
  `<COL>_PRESENTE` = SI/NO: di un identificativo non conta il valore, conta se il dato è stato raccolto.
- Le colonne **numeriche continue** (punteggi, importi) vengono discretizzate in quantili (`<COL>_BIN`)
  ed entrano in pattern mining, albero, SHAP e clustering. Restano fuori i codici numerici: valori interi quasi tutti
  diversi, o con lo stesso numero di cifre, o sotto un nome da identificativo (`ID_UTENTE`, `COD_SEDE`).
- **Errori unici.** La "firma" di una riga target è la combinazione dei valori delle otto dimensioni descrittive più
  associate al target: una firma che compare una volta sola è un caso da guardare a mano.
- **Campi mancanti.** Un vuoto in una colonna degli altri sistemi è un co-fallimento (ipotesi H4), non un dato mancante (H2): le due
  cose non vengono contate due volte. Se il target non è l'esito di un sistema, anche quelle colonne sono dati come gli altri.
- Parole riconosciute nei nomi, ruoli e tassonomia delle cause tecniche sono in `cercaPattern/config.py`
  (`ROLE_COLUMNS`, `OUTCOME_WORDS`, `FAMILY_WORDS`, `DETAIL_TAXONOMY`).
- Ogni fase è isolata: un errore in una fase viene loggato e registrato come finding senza bloccare le altre.

## Prestazioni

- Nulli, cardinalità, conversioni numeriche e feature derivate sono calcolati una volta per colonna e riusati da tutte le fasi.
- Clustering e anomaly detection lavorano sui *profili* distinti delle righe target, pesati per numero di righe; oltre 20.000 profili LOF e DBSCAN sono stimati su un campione e applicati agli altri.
- I grafici a dispersione mostrano al più 5.000 punti; l'Excel limita le tabelle riga-per-riga.
- Con `--sample-rows` si controlla il campione usato da albero e XGBoost (i pattern non sono mai campionati).

## Test

```bash
uv pip install --python .venv/bin/python pytest pytest-xdist
.venv/bin/python -m pytest tests -q            # in sequenza
.venv/bin/python -m pytest tests -q -n 6       # in parallelo (pytest-xdist)
```

- `tests/test_units.py`: test mirati e veloci su conversioni del loader (numeri, date, encoding, separatori), statistica, struttura del tracciato (ruoli, sistemi, gruppi e famiglie dichiarati, campi mancanti), ricerca dei pattern (conteggi esatti, copertura cumulativa, equivalenze, soglie adattive), analisi temporale (finestre, ambito, cambi di livello), anomalie, selezione del target, scrittura dei report.
- `tests/test_pipeline.py`: la pipeline end-to-end su varianti del dataset sintetico: separatori ed encoding diversi, nomi colonna scritti in altri modi, in minuscolo o con caratteri speciali, colonne mancanti o costanti, formati data (ISO, italiano, solo TIME_KEY, solo TIME, DATA+TIME, nessuna colonna temporale), pochi o pochissimi errori, tasso di errore altissimo, valori target con spazi o case diverso, decimali con virgola, nomi riservati già presenti nel tracciato, dataset minuscoli, regressione da una certa data, target su un altro sistema o su una colonna descrittiva. Un test fallisce anche se una fase ha sollevato un'eccezione interna catturata dalla pipeline (le fasi sono isolate ma gli errori non devono passare inosservati).
- `tests/test_cli.py`: percorsi di input e output, più file, pausa e apertura del report in modalità eseguibile, opzioni non valide, opzioni su file, console che non sa rappresentare un carattere.

In tutto 158 test (77 unitari, 49 end-to-end, 32 sulla CLI).

Non modificare i sorgenti mentre la suite gira: una modifica a metà esecuzione può far fallire alcuni test per motivi che non dipendono dal codice finale.
