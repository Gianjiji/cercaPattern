"""
cercaPattern
============
Pacchetto per l'analisi forense "cross" di una colonna target di un CSV: si scelgono una colonna
e uno o più valori (il *caso da spiegare*) e l'analisi cerca dove, quando e perché quel caso si
verifica. Non presuppone un tracciato: ruoli delle colonne e gruppi di colonne che descrivono
l'esito di uno stesso sistema vengono riconosciuti dai nomi e dai dati.

Struttura (una fase = un modulo):

    config.py         ruoli e sistemi del tracciato, dataclass di contesto, cache per colonna
    loader.py         caricamento robusto del CSV (encoding / separatore), numeri e date
    stats_utils.py    intervalli di Wilson, test binomiale, FDR, insiemi di righe a bit
    select_target.py  scelta guidata di colonna e valori target, candidati suggeriti
    profiling.py      FASE 1 - data profiling
    dictionary.py     FASE 2 - dizionario dati automatico
    error_analysis.py FASE 3 - tassi di errore per dimensione, con significatività
    patterns.py       FASE 4 - pattern orientati al target, copertura cumulativa, albero, XGBoost + SHAP
    temporal.py       FASE 5 - finestre di degrado, cambi di livello, fasce orarie
    functional.py     FASE 6 - analisi funzionale (campi mancanti, altri sistemi, famiglie di valori)
    clustering.py     FASE 7 - clustering dei profili delle righe target
    anomaly.py        FASE 8 - anomaly detection (IF, DBSCAN, LOF)
    hypotheses.py     regole deduttive: cause probabili (H1..H9)
    reporting.py      FASE 9 - Excel, HTML, CSV, PNG, Markdown
    progress.py       barre di avanzamento per fase
    pipeline.py       orchestrazione delle fasi
"""

__version__ = "2.0.0"
