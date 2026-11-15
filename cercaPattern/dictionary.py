"""
FASE 2 - Dizionario dati automatico.

Per ogni colonna viene inferito significato, tipo, dominio dei valori e ruolo
nell'analisi. L'inferenza combina:

1. una base di conoscenza di parole generiche che compaiono nei nomi delle colonne;
2. il sistema a cui la colonna appartiene, se il nome ne contiene uno (vedi ``config.py``);
3. euristiche sul contenuto (cardinalità, tipo, esempi).
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

import pandas as pd

from .config import AnalysisContext

# regex sul nome -> (descrizione, ruolo nell'analisi). Vince la prima che corrisponde.
KNOWLEDGE_BASE: List[Tuple[str, str, str]] = [
    (r"DATE_?TIME|TIMESTAMP|DATA_?ORA", "Data e ora del record", "Analisi temporale, finestre di degrado"),
    (r"^(DATA|DATE|DT|GIORNO|DAY)$", "Data del record", "Analisi temporale"),
    (r"^(TIME|ORA|ORARIO)$", "Ora del record", "Fasce orarie critiche"),
    (r"TIME_?KEY", "Chiave temporale (bucket orario/giornaliero)", "Aggregazioni temporali"),
    (r"SESSION", "Identificativo di sessione / contesto", "Se assente il record non è riconducibile a una sessione"),
    (r"DETAIL|DETTAGLI|REASON|MOTIV|MESSAG|MSG", "Dettaglio / motivazione di un esito", "Permette di classificare la causa tecnica"),
    (r"SCORE|PUNTEGGIO|RATING", "Punteggio numerico", "Output quantitativo di una valutazione"),
    (r"RULE|REGOLA|POLICY", "Regola / policy applicata", "Spiega una decisione; se assente nessuna regola è stata applicata"),
    (r"RISK|RISCHIO|LEVEL|LIVELLO", "Livello / classe di valutazione", "Esito di una valutazione"),
    (r"(^|_)(ACTION|AZIONE|DECISION[E]?)(_|$)|ACTION$", "Decisione / azione risultante", "Esito di un passo decisionale"),
    (r"ESITO|OUTCOME|RESULT|RISULTATO|STATUS|STATO", "Stato / esito", "Esito di un passo del processo"),
    (r"(^|_)(ID|UUID|GUID|KEY)(_|$)|^ID[A-Z]{1,6}$|^[A-Z]{2,10}ID$", "Identificativo", "Chiave / correlazione tra record"),
    (r"IMPORTO|AMOUNT|PREZZO|PRICE|COSTO|TOTALE", "Importo / grandezza economica", "Misura; priorità delle anomalie"),
    (r"FLOW|FLUSSO|JOURNEY|PROCESS", "Flusso / percorso applicativo", "Dimensione di analisi"),
    (r"^APP(_?ID)?$|APPLICAZIONE|APPLICATION|SERVICE|SERVIZIO", "Applicazione / servizio chiamante", "Dimensione di analisi: punto di integrazione"),
    (r"CANALE|CHANNEL", "Canale di provenienza", "Dimensione di analisi"),
    (r"SEGMENT", "Segmento", "Dimensione di analisi"),
    (r"PRODOTTO|PRODUCT|TIPO|TYPE|CATEGOR", "Tipologia / categoria", "Dimensione di analisi"),
    (r"VERSION", "Versione", "Dimensione di analisi; utile per i cambi di livello"),
    (r"USER|UTENTE|ACCOUNT", "Utente / soggetto", "Analisi di ricorrenza per soggetto"),
    (r"DEVICE|DISPOSITIVO|BROWSER|CLIENT", "Informazione sul client / dispositivo", "Dato raccolto lato client"),
    (r"LAT(ITUDE)?$|LON(GITUDE)?$|LNG$|COUNTRY|PAESE|NAZIONE|CITY|CITTA|GEO", "Informazione geografica (coordinate/paese/città)", "Dimensione geografica"),
    (r"^IP|_IP$|IP_", "Indirizzo IP / attributi di rete", "Dato di rete"),
    (r"DURATION|DURATA|LATENCY|LATENZA|ELAPSED", "Durata / tempo di risposta", "Misura"),
    (r"COUNT|QTA|QTY|QUANT", "Conteggio / quantità", "Misura"),
    (r"FLAG|^IS_|^HAS_", "Indicatore sì/no", "Dimensione binaria"),
    (r"CODE|CODICE|^COD_", "Codice", "Dimensione o chiave"),
    (r"DESCR|NOTE|COMMENT|TESTO|TEXT", "Descrizione testuale", "Testo libero"),
]

UNKNOWN = ("Campo non riconosciuto dal nome: significato da confermare con il data owner", "Non classificato")

# descrizione per tipo inferito, quando il nome non dice nulla
BY_TYPE = {
    "categorico": ("Dimensione categorica", "Dimensione di analisi"),
    "booleano": ("Indicatore sì/no", "Dimensione binaria"),
    "numerico": ("Grandezza numerica", "Misura"),
    "identificativo": ("Identificativo (valori quasi tutti diversi)", "Chiave / correlazione tra record"),
    "identificativo numerico": ("Identificativo numerico (valori quasi tutti diversi)", "Chiave / correlazione tra record"),
    "data/ora": ("Data e/o ora", "Analisi temporale"),
    "testo libero": ("Testo libero", "Testo libero"),
}


def _infer_type(s: pd.Series, n_unique: int, n_rows: int, num_ratio: float, dt_ratio: float) -> str:
    nn = s.dropna()
    if nn.empty:
        return "vuoto"
    vals = set(nn.astype(str).str.upper().unique()[:10])
    if n_unique <= 2 and vals <= {"0", "1", "S", "N", "Y", "TRUE", "FALSE", "SI", "NO", "T", "F"}:
        return "booleano"
    if dt_ratio > 0.9:
        return "data/ora"
    if num_ratio > 0.95:
        # numerico ma identificativo?
        if n_rows and n_unique / max(len(nn), 1) > 0.9 and n_unique > 50:
            return "identificativo numerico"
        return "numerico"
    if n_rows and n_unique / max(len(nn), 1) > 0.9 and n_unique > 50:
        return "identificativo"
    avg_len = nn.astype(str).str.len().mean()
    if avg_len > 40:
        return "testo libero"
    return "categorico"


def _domain(ctx: AnalysisContext, col: str, tipo: str) -> str:
    """Dominio dei valori: intervallo per numeri e date, elenco o valori più frequenti per il resto."""
    nn = ctx.df[col].dropna()
    if nn.empty:
        return "(solo nulli)"
    if tipo in ("numerico", "identificativo numerico"):
        v = ctx.numeric(col)
        return f"[{v.min():g} .. {v.max():g}]"
    if tipo == "data/ora":
        # minimo e massimo vanno presi sulle date, non sulle stringhe: "01/01/2027" < "31/12/2026" in ordine alfabetico
        ts = ctx.datetime(col).dropna()
        if ts.empty:
            return f"[{nn.min()} .. {nn.max()}]"
        fmt = "%H:%M:%S" if ts.dt.year.max() <= 1900 else "%Y-%m-%d %H:%M:%S"
        return f"[{ts.min():{fmt}} .. {ts.max():{fmt}}]"
    n_distinct = ctx.nunique(col)
    if n_distinct > 5000:     # identificativi e testo libero: i "più frequenti" non dicono nulla
        return f"{n_distinct} valori distinti"
    vc = nn.value_counts()
    if len(vc) <= 12:
        return "{" + ", ".join(map(str, vc.index.tolist())) + "}"
    return f"{len(vc)} valori distinti; top: " + ", ".join(map(str, vc.index[:6].tolist()))


def describe_column(name: str, tipo: Optional[str] = None) -> Tuple[str, str]:
    """Descrizione e ruolo di una colonna: dal nome (base di conoscenza) e, in mancanza, dal tipo inferito."""
    up = name.upper()
    for rx, desc, role in KNOWLEDGE_BASE:
        if re.search(rx, up):
            return desc, role
    return BY_TYPE.get(tipo or "", UNKNOWN)


def build_dictionary(ctx: AnalysisContext, profile: pd.DataFrame) -> pd.DataFrame:
    """Costruisce il dizionario dati a partire dal profiling."""
    rows = []
    prof = profile.set_index("CAMPO")
    for c in ctx.df.columns:
        s = ctx.df[c]
        tipo = prof.loc[c, "TIPO_INFERITO"] if c in prof.index else "n/d"
        desc, role = describe_column(c, tipo)
        system = ctx.cols.system_of(c)
        if system:
            desc = f"{desc} (sistema {system})"
            if c != ctx.target_col:
                role = f"{role}. " + ("Stesso sistema del target: descrive l'esito analizzato" if system == ctx.system_key
                                     else "Altro sistema: confronto con il caso analizzato")
        if c == ctx.target_col:
            role = f"TARGET DELL'ANALISI ({ctx.target_value} = caso analizzato). {role}"
        ex = [str(v) for v in s.dropna().unique()[:5]]
        rows.append({
            "COLONNA": c,
            "DESCRIZIONE": desc,
            "TIPO": tipo,
            "ESEMPI": " | ".join(ex),
            "DOMINIO": _domain(ctx, c, tipo),
            "RUOLO_NELL_ANALISI": role,
            "PCT_NULL": prof.loc[c, "PCT_NULL"] if c in prof.index else None,
            "VALORI_DISTINTI": prof.loc[c, "VALORI_DISTINTI"] if c in prof.index else None,
        })
    return pd.DataFrame(rows)
