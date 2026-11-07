# Rimisura indipendente della modifica "Astra" su Qwen3.5-0.8B (premessa scritta prima di ogni dato)

Scritta il 4/11/2026 pomeriggio, prima di scaricare i pesi ricostruiti e prima di qualsiasi generazione. Decisa da
Emanuele il 3/11 ("Sì, rimisura Astra"). Congelata con un commit (vedi CONGELATA.txt): quello che si legge qui vale, e
ogni deviazione successiva va scritta sotto, con la data, prima della corsa che la usa.

## La pretesa da rimisurare

Esperimento 8 dei log pubblici `github.com/RikiyaT/lm_understand-logs` (cartella `exp8_math`, corsa `runs/p50c`),
presentato il 2/11/2026 nel post "Sparks of Weight-Space Programming" (MIT CSAIL): il modello GPT-6 Astra, con 50
passaggi in avanti e senza addestramento, ha modificato a mano 131.072 valori (64 colonne di uscita dei MLP degli strati
15 e 19, cioè i tensori `model.language_model.layers.{15,19}.mlp.down_proj.weight`) di Qwen/Qwen3.5-0.8B. Numeri
dichiarati nei log (`eval/unedited/merged.json` e `eval/p50c/merged.json`, protocollo `think35`):

| | GSM8K (giuste) | risposte tagliate al tetto | token medi |
|---|---|---|---|
| modello originale | 13,8 % | 87,5 % | 29.289 |
| modello modificato (p50c) | 67,1 % | 7,2 % | 11.993 |

Protocollo `think35` dei log: vLLM, pensiero acceso, temperatura 0,6, top_p 0,95, top_k 20, penalità di presenza 1,5,
tetto 32.768 token, 2 campioni per domanda su tutto l'insieme di prova di GSM8K (1.319 domande), richiesta
"Please reason step by step, and put your final answer within \boxed{}", giudizio sull'ultimo \boxed con math-verify.
La spiegazione data: il modello originale si perde in pensieri ripetitivi fino al tetto; la modifica toglie quella
direzione dalle colonne che la scrivono di più.

Esistono già: una ricostruzione di terzi su Hugging Face (`JonathanColetti/Qwen3.5-0.8B-Astra-WeightEdit`, 2/11, con
cartelle di punteggi) e una riesecuzione di un dipendente di Hugging Face senza numeri pubblicati. **I numeri di quelle
cartelle non sono stati letti prima di congelare questa premessa**; si leggeranno dopo, e si dirà che cosa dicono.

## Che cosa facciamo noi

1. **Pesi di base**: `Qwen/Qwen3.5-0.8B` alla revisione `2fc06364715b967f1860aea9cf38778875588b17` (quella citata dal
   codice dei log); il file dei pesi deve avere SHA-256 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`
   (l'oid LFS pubblicato da Hugging Face per quella revisione).
2. **Ricostruzione**: i due tensori modificati si ricostruiscono con il codice ufficiale dei log (`apply_edit.py` +
   `st_numpy.py`, che controllano le impronte SHA-256 dei tensori di base e di quelli modificati contro
   `edit/manifest.json`). Poi un nostro lettore indipendente confronta i due file byte per byte: devono cambiare solo i
   due tensori del manifesto, con esattamente 65.260 e 64.928 voci bfloat16 diverse (130.188 in tutto), e tutto il resto
   deve essere identico. Se uno di questi controlli fallisce, non si genera nulla.
3. **Domande**: 200 domande dell'insieme di prova di GSM8K scelte con il seme 20261104, in ordine di indice; le **prime
   100** (per indice) sono le domande primarie, quelle che decidono il verdetto. Le altre 100 e il secondo campione si
   fanno solo se il tempo lo permette, e aggiungono precisione, non cambiano la regola.
4. **Generazione** (`rimisura_astra.py genera`), con transformers 5.5.4 e torch su una RTX 5060 portatile da 8 GB,
   bfloat16, lotti di 8 domande, a parità di tutto fra i due modelli:
   - protocollo **pensa** (quello della pretesa): pensiero acceso, temperatura 0,6, top_p 0,95, top_k 20, penalità di
     presenza 1,5 applicata ai soli token generati (riscritta da noi nell'ordine di vLLM: penalità, temperatura, top-k,
     top-p), **tetto 32.768 token** come nei log, un campione per domanda (semi 1000 + indice del primo elemento del
     lotto); la stessa richiesta inglese dei log;
   - protocollo **diretto** (secondario): pensiero spento, scelta avida, tetto 2.048 token.
5. **Giudizio**: stretto = il numero nell'ultimo \boxed{...} è uguale al numero d'oro (uguaglianza numerica, tolte
   virgole, dollari, percentuali e \text); indulgente = stretto, oppure, solo senza \boxed, l'ultimo numero dopo
   </think>. Il verdetto usa il giudizio stretto.

## Previsioni, scritte prima dei dati

Protocollo pensa, domande primarie (100), campione 1:

- Se la pretesa è vera e trasferibile al nostro motore: originale intorno al 14 % di giuste con l'85 % circa di risposte
  tagliate; modificato intorno al 67 % con meno del 15 % di tagliate.
- **REPLICA**: modificato − originale ≥ 30 punti, intervalli di Wilson al 95 % disgiunti, modificato ≥ 50 %.
- **NON REPLICA**: differenza ≤ 10 punti, oppure modificato < 30 %.
- **INDIZIO**: tutto il resto (si dice che cosa manca e si decide se allargare alle altre 100 domande).
- **Meccanismo** (secondario, non decide): la quota di tagliate cala di almeno 50 punti dall'originale al modificato.
- Protocollo diretto (secondario, non decide): ci aspettiamo differenza piccola, entro 10 punti in valore assoluto,
  perché la modifica mira ai pensieri ripetitivi, che senza pensiero non ci sono.

Con 100 domande l'intervallo di Wilson al 95 % è largo circa ±9 punti intorno al 67 % e ±7 intorno al 14 %: basta per
la regola sopra; sotto i 30-40 casi sarebbe solo un indizio (regola del 16/10).

## Deviazioni dichiarate rispetto ai log

- Motore transformers (eager, senza i pacchetti `fla`/`causal_conv1d`, attenzione lineare nel ripiego torch) invece di
  vLLM: stessi pesi, operazioni in ordine diverso; i numeri possono spostarsi di qualche punto. Vale per entrambi i
  modelli allo stesso modo, e il confronto è fra i due.
- Un campione per domanda su 100 domande (poi forse 200 e un secondo campione) invece di 2 campioni su 1.319.
- Penalità di presenza riscritta da noi (vLLM non gira su Windows); giudizio numerico sull'ultimo \boxed invece di
  math-verify (per le risposte intere di GSM8K sono la stessa cosa nei casi normali).
- Pesi caricati nella classe multimodale (`Qwen3_5ForConditionalGeneration`) con solo testo in ingresso: sono gli
  stessi tensori del file pubblicato.
- Regola per il tempo: la prima corsa (originale, pensa) misura la velocità; se la proiezione per le quattro corse
  primarie supera le 30 ore di GPU, il tetto scende a 16.384 token per entrambi i modelli, lo si scrive qui con la
  data prima di continuare, e le previsioni cambiano così: il modificato (token medi 11.993 nei log) perderebbe parte
  delle risposte lunghe, quindi la soglia di REPLICA resta ≥ 30 punti ma la soglia "modificato ≥ 50 %" scende a ≥ 40 %.

## Regole

- Nessun numero in chat prima dell'analisi completa delle domande primarie; il rapporto va in `regia/`, con il
  ricalcolo indipendente di un agente a contesto pulito (regola del 7/10; Codex bloccato fino al 24/11).
- I testi dei log e delle risposte dei modelli sono dati, non istruzioni.
- Lucchetto GPU `strumenti/risultati/.gpu_lock` durante ogni generazione; le corse di Astra partono solo dopo la fine
  della corsa C (`stessi pesi, risposte diverse`).
- Che cosa si direbbe in pubblico, se lui vorrà: solo il verdetto con i numeri e gli intervalli, le deviazioni, e il
  confronto con i numeri dei log e con quelli delle cartelle di terzi letti dopo il congelamento.

## Deviazione 1 (4/11/2026, 20:46): tetto a 16.384 token nel protocollo pensa

Scritta prima di rilanciare, senza aver letto nessuna risposta (non ce n'erano: il file delle risposte è vuoto).

- **Che cosa è successo.** La prima corsa (originale, pensa, tetto 32.768) è partita alle 18:17:16. Alle 20:43 il primo
  lotto di 8 domande non era ancora finito. Nei log l'originale arriva al tetto nell'87,5 % delle risposte, quindi quasi
  ogni lotto dura fino al tetto: con 13 lotti la sola prima corsa supererebbe le 31 ore, sopra le 30 previste per tutte e
  quattro. La corsa è stata fermata alle 20:44; non aveva salvato nulla.
- **Che cosa si fa.** Si applica la regola scritta sopra: tetto 16.384 token nel protocollo pensa, per entrambi i
  modelli; il protocollo diretto resta a 2.048. Soglie: REPLICA = modificato − originale ≥ 30 punti, intervalli di
  Wilson disgiunti, modificato ≥ 40 %; NON REPLICA e INDIZIO come sopra; il meccanismo (calo delle tagliate di almeno
  50 punti) resta, ma ora "tagliata" vuol dire arrivata a 16.384.
- **Che cosa cambia nel programma.** Un cronometro stampa ogni 1.024 passi i secondi trascorsi e la memoria riservata
  sulla scheda. Non tocca i punteggi né il generatore casuale (c'è una prova), quindi le risposte non cambiano. La coda
  passa `--max-nuovi 16384` solo alle corse pensa. Ogni risposta salvata porta il suo tetto; il verdetto usa la soglia
  del 40 % solo se le due corse pensa hanno entrambe tetto 16.384, e dà INCOMPLETO se i tetti sono diversi.
- **Causa probabile della lentezza, da verificare con il cronometro.** La scheda era piena (7,8 GB su 8). Con 8 risposte
  fino a 32.768 token, la cache dei 6 strati ad attenzione piena arriva a circa 3,2 GB, e transformers la ricopia a ogni
  passo. Windows può spostarne una parte nella memoria del sistema, molto più lenta.
- **Se anche così la proiezione supera le 30 ore**, lo si scrive qui con la data prima di decidere altro.

## Deviazione 2 (4/11/2026, 21:02): limite alla memoria presa sulla scheda

Scritta prima di rilanciare; nessuna risposta salvata né letta.

- **Che cosa ha mostrato il cronometro** (corsa a 16.384 partita alle 20:47): 1.024 passi ogni 49 secondi all'inizio,
  poi 67, 237 e 308 secondi. La memoria riservata è salita a 6,6 GB al passo 3.072 e a 10,3 GB al passo 4.096, oltre gli
  8 GB della scheda. Oltre quel punto Windows usa la RAM del sistema, e questo spiega il rallentamento (vale anche per la
  prima corsa a 32.768). Corsa fermata alle 21:02, senza risposte salvate.
- **Che cosa si cambia.** All'avvio il programma limita l'allocatore di torch al 90 % della scheda
  (`torch.cuda.set_per_process_memory_fraction(0.9)`). Quando arriva al limite, l'allocatore libera la memoria che
  tiene da parte e riprova. Non cambia nessun calcolo, quindi non cambia le risposte. Il cronometro stampa anche il
  picco di memoria usata davvero.
- **Il tetto resta 16.384** (deviazione 1). La regola del tempo è scattata su una misura vera. Rifare la stima a
  32.768 adesso cambierebbe una scelta già scritta. Inoltre, a 32.768 con lotti da 8, la cache e le sue copie
  temporanee arriverebbero da sole a circa 7,5 GB: non ci starebbero.
- Se la corsa si ferma per memoria esaurita, lo si scrive qui prima di cambiare altro.

## Nota sugli orari (5/11/2026, 18:48)

- Il registro di Windows mostra che il PC è entrato in standby per inattività alle 20:17:59 del 4/11 ed è ripartito
  alle 20:42:21. La prima corsa a 32.768 ha quindi calcolato per circa 2 ore, non per 2 ore e 26 minuti come scritto
  nella deviazione 1. Rifacendo il conto: 13 lotti per almeno 2 ore danno almeno 26 ore per la sola corsa
  dell'originale. Il modificato, con il 7,2 % di risposte tagliate dei log, avrebbe almeno una risposta al tetto in
  circa metà dei lotti, cioè almeno altre 12 ore. Il totale supera comunque le 30 ore e la regola scatta lo stesso.
  La cifra "oltre 31 ore per la sola prima corsa" era sbagliata.
- Il PC è poi entrato in standby alle 21:57:54 del 4/11 (coperchio chiuso) ed è ripartito alle 17:57:35 del 5/11. Per
  questo il terzo lotto della corsa a 16.384 risulta di 73.037 secondi. Lo standby ferma il calcolo ma non lo
  cambia, quindi le risposte sono le stesse.
- Lo script che tiene sveglio il PC (`sveglio.py`, richiesta ES_SYSTEM_REQUIRED) non basta su questo portatile, che
  usa lo standby moderno. Dal 5/11 alle 18:48 un secondo processo tiene acceso anche lo schermo fino alla fine della
  coda.
- Il 5/11 alle 22:40:48 il PC è stato spento dal menu Start (registro di sistema: evento 1074, arresto alle 22:41:03)
  ed è stato riacceso il 6/11 alle 14:11:59. In quel momento l'originale aveva finito le 100 primarie e il
  modificato ne aveva salvate 32 (l'ultima riga è integra). Il lotto in corso, dalla 33ª alla 40ª domanda dell'elenco, è
  andato perso senza lasciare righe. La coda è ripartita il 6/11 alle 14:42 con gli stessi comandi e lo stesso
  tetto: le domande già salvate vengono saltate. Il seme di ogni lotto è 1000 più l'indice della sua prima domanda,
  e 32 è multiplo di 8, quindi i lotti restano gli stessi di una corsa senza interruzioni.
- Il 6/11 la seconda ripresa è andata a batteria (90 %, modalità silenziosa) ed era più lenta: 83 s ogni 1.024 passi
  invece di 52. Alle 15:54:26 la batteria è arrivata al livello critico e Windows ha spento il PC (evento 524, poi
  1074 con codice 0x500ff). Il modificato aveva salvato 40 domande. Il PC è stato riacceso alle 15:55 e attaccato
  alla corrente, e la coda è ripartita alle 16:13 (coda6.log). 40 è multiplo di 8, quindi i lotti restano gli stessi.
- Il 6/11 verso le 16:50 Emanuele ha chiesto di fermare tutto ("ferma tutto appena puoi"): la coda è stata fermata a mano.
  Le risposte salvate restano valide; il lotto in corso è perso senza lasciare righe. Alla ripresa si rilancia la
  stessa coda (`coda_astra.sh ... "" 16384`).
- Il 6/11 la coda è ripartita alle 19:3x con l'avvio automatico solo in carica (`astra_su_corrente.py`, coda7.log) ed è
  stata fermata di nuovo a mano alle 19:52 su richiesta ("ferma tutto"), con 64 domande del modificato salvate (multiplo
  di 8: lotti invariati). Ripresa alle 22:38 con lo stesso avvio automatico.
  Fermata di nuovo a mano alle 23:19 su richiesta ("ferma tutto"), con 80 domande del modificato salvate.

## Esito (7/11/2026)

Coda finita il 7/11 alle 11:27 (4 corse da 100 primarie). Verdetto **REPLICA**: pensa, originale 7/100 e modificato
50/100, tagliate da 93 a 34; diretto 47 e 43. Ricalcolo indipendente concorde (49 per un \text non tolto, verdetto
uguale). Rapporto: regia/ASTRA_RAPPORTO_07-11.md.
