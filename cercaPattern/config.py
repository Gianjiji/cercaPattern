"""
Configurazione centrale, struttura del tracciato e contesto condiviso.

Il programma non conosce in anticipo il dataset: la struttura del tracciato viene ricavata dai
nomi delle colonne e dai dati.

- *Ruoli*: alcune colonne hanno un compito preciso nell'analisi (data/ora, misura principale,
  identificativo di sessione, regola applicata, decisione complessiva). Si riconoscono da parole
  generiche nel nome, con match esatto senza distinzione tra maiuscole e minuscole e, in
  subordine, tramite espressioni regolari.
- *Sistemi*: gruppi di colonne che descrivono l'esito di uno stesso componente. Un nome come
  ``ESITO_ALFA`` o ``BETA_SCORE`` è fatto di una parola che dice *cosa* contiene la colonna
  (esito, score, livello, stato...) e di un nome proprio (``ALFA``, ``BETA``): il nome proprio
  identifica il sistema, e le colonne che lo ripetono (``DETTAGLIO_ALFA``, ``SESSIONE_ALFA``)
  ne fanno parte. I gruppi si possono anche dichiarare in modo esplicito (``--group NOME=REGEX``).

Quando la colonna target appartiene a un sistema, le altre colonne di quel sistema descrivono
l'esito stesso (dettaglio, score) e restano fuori dai modelli; gli altri sistemi servono a capire
se il problema è isolato o condiviso.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .progress import Progress

LOG = logging.getLogger("cercaPattern")

# ---------------------------------------------------------------------------
# Valori considerati "assenti" oltre al NaN di pandas (dopo strip e upper)
# ---------------------------------------------------------------------------
NULL_TOKENS = {"", "NULL", "NONE", "NAN", "N/A", "NA", "-", "--", "?", "UNKNOWN", "UNDEFINED", "NIL"}

# ---------------------------------------------------------------------------
# Ruoli: nome del ruolo -> candidati (match esatto) e regex di fallback
# ---------------------------------------------------------------------------
ROLE_COLUMNS: Dict[str, Dict[str, Any]] = {
    "datetime": {"exact": ["DATETIME", "DATA_ORA", "TIMESTAMP", "TS"], "regex": r"DATE_?TIME|TIMESTAMP|DATA_?ORA"},
    "date":     {"exact": ["DATA", "DATE", "DT"], "regex": r"^(DATA|DATE)$"},
    "time":     {"exact": ["TIME", "ORA"], "regex": r"^(TIME|ORA)$"},
    "time_key": {"exact": ["TIME_KEY", "TIMEKEY"], "regex": r"TIME_?KEY"},
    "regola":   {"exact": ["RULE_NAME", "RULE", "REGOLA", "POLICY"], "regex": r"RULE|REGOLA|POLICY"},
}
TIME_ROLES = ("datetime", "date", "time", "time_key")

# parole che indicano una grandezza economica o una quantità (ruolo "misura": la prima colonna numerica che ne contiene una)
MEASURE_WORDS = {"IMPORTO", "AMOUNT", "AMT", "PREZZO", "PRICE", "COSTO", "COST", "TOTALE", "TOTAL", "QUANTITA", "QUANTITY", "QTY"}
SESSION_RX = r"SESSION"                                   # anche SESSIONE, SESSIONID, SESSION_ID
DECISION_RX = r"ACTION|AZIONE|DECISION"                   # decisione complessiva (se non appartiene a un sistema)
DETAIL_RX = r"DETAIL|DETTAGLI|REASON|MOTIV|DESC|MESSAG|MSG"
SCORE_RX = r"SCORE|PUNTEGGIO"
COORDINATE_RX = r"LAT(ITUDE)?$|LON(GITUDE)?$|LNG$"

# ---------------------------------------------------------------------------
# Sistemi: vocabolario generico per leggere i nomi delle colonne
# ---------------------------------------------------------------------------
# parole che dicono che la colonna contiene l'esito di una valutazione/elaborazione
OUTCOME_WORDS = {"RISK", "RISCHIO", "LEVEL", "LIVELLO", "SCORE", "PUNTEGGIO", "RATING", "ESITO", "OUTCOME", "RESULT", "RISULTATO",
                 "STATUS", "STATO", "STATE", "ACTION", "AZIONE", "DECISION", "DECISIONE", "VERDICT"}
# parole che descrivono un esito o il suo contesto: una colonna fatta di queste parole e del nome di
# un sistema appartiene a quel sistema
FAMILY_WORDS = OUTCOME_WORDS | {"DETAIL", "DETAILS", "DETTAGLIO", "DETTAGLI", "REASON", "MOTIVO", "MOTIVAZIONE", "CAUSE", "CAUSA",
                                "CODE", "CODICE", "COD", "MSG", "MESSAGE", "MESSAGGIO", "ERROR", "ERRORE", "ERR", "DESC", "DESCR",
                                "DESCRIZIONE", "DESCRIPTION", "SESSION", "SESSIONE", "ID", "FLAG", "TIME", "TS", "TIMESTAMP",
                                "DURATION", "DURATA", "LATENCY", "LATENZA", "RESPONSE", "RISPOSTA", "INFO", "NOTE"}
# altre parole generiche: non sono mai il nome di un sistema
GENERIC_WORDS = FAMILY_WORDS | {"DATE", "DATA", "ORA", "DATETIME", "KEY", "VALUE", "VALORE", "TYPE", "TIPO", "NAME", "NOME", "TEXT",
                                "TESTO", "CATEGORY", "CATEGORIA", "LABEL", "VERSION", "VERSIONE", "COUNT", "NUM", "TOT", "TOTAL",
                                "TOTALE", "FINAL", "FINALE", "GLOBAL", "GLOBALE", "GENERALE", "OVERALL", "SISTEMA", "SISTEMI",
                                "SYSTEM", "SYSTEMS", "MAIN", "PRINCIPALE", "OLD", "NEW", "PREV", "NEXT", "MIN", "MAX", "AVG", "SUM"}


def _words_rx(words: set) -> "re.Pattern[str]":
    """Token fatto solo di parole dell'insieme, anche attaccate ('STATUSCODE', 'SESSIONID') o seguite da una versione ('V2')."""
    return re.compile(r"^(?:%s|V?\d+)+$" % "|".join(sorted(words, key=len, reverse=True)))


_GENERIC_TOKEN = _words_rx(GENERIC_WORDS)
_FAMILY_TOKEN = _words_rx(FAMILY_WORDS)
_OUTCOME_PART = re.compile("|".join(sorted(OUTCOME_WORDS, key=len, reverse=True)))
_ID_TOKENS = {"ID", "UUID", "GUID", "KEY", "PK", "CODE", "CODICE", "COD", "NR", "NUM", "NUMERO", "NUMBER", "REF", "RIF"}


def name_tokens(name: str) -> List[str]:
    """Parole del nome di una colonna, in maiuscolo: 'Esito-Alfa [v2]' -> ['ESITO', 'ALFA', 'V2'],
    'sessionId' -> ['SESSION', 'ID']."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z][a-z])", "_", str(name))
    return re.findall(r"[A-Z0-9]+", spaced.upper())


def looks_like_identifier(name: str) -> bool:
    """Il nome suggerisce un identificativo o un codice (ID_RICHIESTA, userId, TXID, COD_SEDE)?"""
    return any(t in _ID_TOKENS or re.fullmatch(r"ID[A-Z]{1,6}|[A-Z]{2,10}ID", t) for t in name_tokens(name))


def system_name(column: str) -> Optional[str]:
    """Nome del sistema di cui la colonna riporta un esito, oppure None.

    'ESITO_ALFA' -> 'ALFA', 'BETA_SCORE' -> 'BETA', 'Stato Gamma' -> 'GAMMA'. Serve una parola
    di esito (ESITO, SCORE, LIVELLO, STATO...) e almeno una parola che non sia generica: 'ESITO_FINALE',
    'ID_UTENTE' o 'TIME_KEY' non nominano alcun sistema."""
    toks = name_tokens(column)
    names = [t for t in toks if not _GENERIC_TOKEN.match(t)]
    if not names or any(len(t) < 2 or t.isdigit() for t in names):
        return None
    if not any(_GENERIC_TOKEN.match(t) and _OUTCOME_PART.search(t) for t in toks):
        return None
    return "_".join(names)


def _belongs_to(column: str, system: str) -> bool:
    """La colonna è fatta del nome del sistema e di sole parole che descrivono un esito o il suo contesto."""
    toks, key = name_tokens(column), system.split("_")
    for i in range(len(toks) - len(key) + 1):
        if toks[i:i + len(key)] == key:
            rest = toks[:i] + toks[i + len(key):]
            return bool(rest) and all(_FAMILY_TOKEN.match(t) for t in rest)
    return False


def detect_systems(columns: List[str], groups: Optional[Dict[str, str]] = None, auto: bool = True) -> Dict[str, List[str]]:
    """Sistemi presenti nel tracciato: nome -> colonne (nell'ordine del file).

    ``groups`` sono i gruppi dichiarati (nome -> regex sul nome della colonna) e hanno la precedenza
    sul riconoscimento automatico, che si disattiva con ``auto=False``."""
    systems: Dict[str, List[str]] = {}
    taken: set = set()
    for name, rx_s in (groups or {}).items():
        rx = re.compile(rx_s, re.IGNORECASE)
        cols = [c for c in columns if c not in taken and rx.search(str(c))]
        if cols:
            systems[str(name)] = cols
            taken.update(cols)
        else:
            LOG.warning("Gruppo '%s': nessuna colonna corrisponde a '%s'", name, rx_s)
    if not auto:
        return systems
    found: List[str] = []
    for c in columns:
        s = system_name(c) if c not in taken else None
        if s and s not in found and s not in systems:
            found.append(s)
    # prima i nomi più lunghi: 'ALFA_BETA' non deve essere assorbito da 'ALFA'
    for s in sorted(found, key=lambda k: (-len(k.split("_")), found.index(k))):
        cols = [c for c in columns if c not in taken and (system_name(c) == s or _belongs_to(c, s))]
        if cols:
            systems[s] = cols
            taken.update(cols)
    order = {c: i for i, c in enumerate(columns)}
    return dict(sorted(systems.items(), key=lambda kv: order[kv[1][0]]))


def plural(n: int, singolare: str, plurale: str) -> str:
    """'1 finestra' / '3 finestre': i testi finiscono in un report letto da persone."""
    return f"{n} {singolare if n == 1 else plurale}"


def slugify(text: str) -> str:
    """Nome file sicuro: minuscolo, solo [a-z0-9_]."""
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(text).lower())).strip("_") or "target"

# Parole chiave per classificare la colonna di dettaglio in cause tecniche (vince la prima categoria che corrisponde)
DETAIL_TAXONOMY: Dict[str, List[str]] = {
    "TIMEOUT / LATENZA":          ["TIMEOUT", "TIME OUT", "TIMED", "LATENC", "SLOW", "DEADLINE"],
    "CONNETTIVITA / HTTP 5xx":    ["CONNECT", "REFUSED", "UNREACHABLE", "503", "502", "500", "504", "GATEWAY", "SOCKET", "HOST"],
    "SESSIONE / CONTESTO":        ["SESSION", "SESSIONE", "CONTEXT", "NOT FOUND", "NOTFOUND", "UNKNOWN SESSION", "EXPIRED", "SCADUT"],
    "DATI MANCANTI / PAYLOAD":    ["MISSING", "MANCANT", "NULL", "EMPTY", "REQUIRED", "INVALID", "PAYLOAD", "PARSE", "FORMAT", "SCHEMA", "MALFORMED"],
    "CLIENT / TELEMETRIA":        ["FINGERPRINT", "DEVICE", "SDK", "AGENT", "BROWSER", "TELEMETRY", "CLIENT"],
    "AUTENTICAZIONE / API":       ["UNAUTHORIZED", "401", "403", "FORBIDDEN", "TOKEN", "APIKEY", "API KEY", "CREDENTIAL"],
    "ECCEZIONE APPLICATIVA":      ["EXCEPTION", "ERROR", "ERRORE", "FAIL", "STACK", "INTERNAL"],
    "CONFIGURAZIONE / REGOLA":    ["CONFIG", "RULE", "POLICY", "NOT ENABLED", "DISABLED", "NOT CONFIGURED", "UNSUPPORTED"],
}

# Esiti che indicano una situazione critica. Sono parole intere, non pezzi di altre parole:
# "RED" non deve scattare su "CREDIT", "HIGH" non su "HIGHWAY".
CRITICAL_PARTS = ["DENY", "DENIED", "BLOCK[A-Z]*", "REJECT[A-Z]*", "DECLINE[A-Z]*", "HIGH", "ALTO", "CRITIC[A-Z]*", "ALERT", "KO", "RED", "ROSSO"]


def critical_regex(extra: Optional[List[str]] = None) -> str:
    """Regex (senza lookaround, vedi :func:`contains_token`) degli esiti critici, più i valori indicati
    dall'utente. Un valore che termina con '*' vale per tutte le parole che iniziano così ('SOSP*')."""
    parts = list(CRITICAL_PARTS)
    for v in extra or []:
        v = str(v).strip().upper()
        if v.strip("*"):
            parts.append(re.escape(v.rstrip("*")) + ("[A-Z]*" if v.endswith("*") else ""))
    return r"(?:^|[^A-Z0-9])(?:%s)(?:$|[^A-Z0-9])" % "|".join(parts)


CRITICAL_RX = critical_regex()

# Valori che indicano "passaggio non eseguito / opzione non attiva" (oltre ai valori assenti)
NEGATIVE_TOKENS = {"N", "NO", "0", "FALSE", "F", "OFF", "NESSUNA", "NESSUNO", "ASSENTE", "ABSENT", "SKIPPED", "SALTATO", "SALTATA",
                   "DISABLED", "DISATTIVO", "DISATTIVA", "EXEMPT", "ESENTE", "NOT_REQUIRED", "NON_RICHIESTO", "NON_RICHIESTA"}


@dataclass
class Finding:
    """Deduzione strutturata prodotta da una fase di analisi."""
    fase: str                       # es. "FASE 6"
    categoria: str                  # es. "ALIMENTAZIONE DATI"
    titolo: str                     # frase breve
    evidenza: str                   # numeri a supporto
    confidenza: str                 # ALTA / MEDIA / BASSA
    raccomandazione: str = ""       # azione operativa suggerita

    def as_row(self) -> Dict[str, str]:
        return {
            "FASE": self.fase, "CATEGORIA": self.categoria, "TITOLO": self.titolo,
            "EVIDENZA": self.evidenza, "CONFIDENZA": self.confidenza,
            "RACCOMANDAZIONE": self.raccomandazione,
        }


class ColumnMap:
    """Ruoli e sistemi ricavati dai nomi delle colonne del DataFrame."""

    def __init__(self, columns: List[str], groups: Optional[Dict[str, str]] = None, auto_groups: bool = True):
        self.columns = list(columns)
        self._upper = {c.strip().upper(): c for c in self.columns}
        self.map: Dict[str, Optional[str]] = {}
        self.systems: Dict[str, List[str]] = detect_systems(self.columns, groups, auto_groups)
        self._system_of: Dict[str, str] = {c: s for s, cols in self.systems.items() for c in cols}
        self._resolve()

    def _resolve(self) -> None:
        used: set = set()
        for key, spec in ROLE_COLUMNS.items():
            found = None
            for cand in spec["exact"]:
                if cand.upper() in self._upper and self._upper[cand.upper()] not in used:
                    found = self._upper[cand.upper()]
                    break
            if found is None and spec.get("regex"):
                rx = re.compile(spec["regex"], re.IGNORECASE)
                for up, orig in self._upper.items():
                    if orig in used:
                        continue
                    if rx.search(up):
                        found = orig
                        break
            self.map[key] = found
            if found:
                used.add(found)
        # decisione complessiva: una colonna di azione/decisione che non appartiene a nessun sistema
        # (parole intere: "AUTORIZZAZIONE" contiene "AZIONE" ma non è una decisione)
        self.map["decisione"] = next((c for c in self.columns if c not in self._system_of and any(
            _GENERIC_TOKEN.match(t) and re.search(DECISION_RX, t) for t in name_tokens(c))), None)

    def get(self, key: str) -> Optional[str]:
        return self.map.get(key)

    def has(self, key: str) -> bool:
        return self.map.get(key) is not None

    def system(self, key: Optional[str]) -> List[str]:
        """Colonne del sistema indicato."""
        return self.systems.get(key, []) if key else []

    def system_of(self, column: str) -> Optional[str]:
        """Sistema a cui appartiene la colonna, oppure None."""
        return self._system_of.get(column)

    def measure_candidates(self) -> List[str]:
        """Colonne il cui nome indica una grandezza economica o una quantità, nell'ordine del file."""
        return [c for c in self.columns if c not in self._system_of and MEASURE_WORDS & set(name_tokens(c))]

    def session_candidates(self) -> List[str]:
        return [orig for up, orig in self._upper.items() if re.search(SESSION_RX, up)]

    def summary(self) -> pd.DataFrame:
        rows = [{"NOME_LOGICO": f"ruolo:{k}", "COLONNA_FISICA": v or "(non trovata)"} for k, v in self.map.items()]
        for s, cols in self.systems.items():
            rows.append({"NOME_LOGICO": f"sistema:{s}", "COLONNA_FISICA": ", ".join(cols)})
        return pd.DataFrame(rows)


@dataclass
class AnalysisContext:
    """Stato condiviso tra le fasi della pipeline."""
    df: pd.DataFrame
    out_dir: Path
    target_col: str
    target_value: str                 # etichetta dei valori target (es. "ERROR" o "ERROR|HIGH")
    cols: ColumnMap
    target_values: List[str] = field(default_factory=list)  # valori (maiuscoli) che definiscono il caso analizzato
    is_error: pd.Series = field(default=None)
    results: Dict[str, Any] = field(default_factory=dict)   # tabelle (DataFrame) per i report
    figures: Dict[str, Any] = field(default_factory=dict)   # figure plotly per l'HTML
    png_paths: List[Path] = field(default_factory=list)     # grafici PNG salvati
    findings: List[Finding] = field(default_factory=list)   # deduzioni
    meta: Dict[str, Any] = field(default_factory=dict)      # numeri chiave per l'executive summary
    random_state: int = 42
    max_cardinality: int = 40      # soglia per considerare una colonna "categorica" nei modelli
    sample_rows: int = 250_000     # campionamento per i modelli su dataset grandi
    critical_values: List[str] = field(default_factory=list)   # valori aggiuntivi da trattare come esito critico (fase 8)
    families: List[Tuple[Optional[str], str, str]] = field(default_factory=list)   # famiglie di valori dichiarate: (colonna o None, nome, regex)
    progress: Progress = field(default_factory=lambda: Progress(0, enabled=False))  # barra di avanzamento
    _cache: Dict[str, Any] = field(default_factory=dict, repr=False)  # risultati per colonna riusati tra le fasi

    # -- cache per colonna ----------------------------------------------------
    # Nulli e cardinalità servono a quasi tutte le fasi: su milioni di righe ricalcolarli a ogni
    # chiamata costa minuti. Il DataFrame non viene modificato dopo il caricamento, quindi si
    # possono calcolare una volta sola.
    def isnull(self, col: str) -> pd.Series:
        """Maschera booleana dei valori assenti (NaN o token di nullità) della colonna."""
        key = ("null", col)
        if key not in self._cache:
            self._cache[key] = is_null_series(self.df[col])
        return self._cache[key]

    def nunique(self, col: str) -> int:
        """Numero di valori distinti non nulli della colonna."""
        key = ("nunique", col)
        if key not in self._cache:
            self._cache[key] = int(self.df[col].nunique(dropna=True))
        return self._cache[key]

    def numeric(self, col: str) -> pd.Series:
        """Colonna convertita in float (NaN dove non numerica). La convenzione numerica (virgola o
        punto decimale) è decisa sull'intera colonna: convertire un sottoinsieme di righe per conto
        suo potrebbe leggerle con una convenzione diversa dal resto."""
        key = ("numeric", col)
        if key not in self._cache:
            from .loader import coerce_numeric
            self._cache[key] = coerce_numeric(self.df[col])
        return self._cache[key]

    def datetime(self, col: str) -> pd.Series:
        """Colonna letta come data/ora (NaT dove non interpretabile), calcolata una volta sola."""
        key = ("datetime", col)
        if key not in self._cache:
            from .loader import parse_datetime
            self._cache[key] = parse_datetime(self.df[col])
        return self._cache[key]

    # -- helper -------------------------------------------------------------
    @property
    def n_rows(self) -> int:
        return len(self.df)

    @property
    def n_errors(self) -> int:
        return int(self.is_error.sum())

    @property
    def error_rate(self) -> float:
        return self.n_errors / self.n_rows if self.n_rows else 0.0

    @property
    def errors(self) -> pd.DataFrame:
        return self.df[self.is_error]

    def add_finding(self, f: Finding) -> None:
        LOG.info("[%s] %s -> %s", f.fase, f.categoria, f.titolo)
        self.findings.append(f)

    def add_table(self, name: str, df: pd.DataFrame) -> None:
        self.results[name] = df

    def pretty_errors(self) -> str:
        return f"{self.n_errors:,} / {self.n_rows:,} ({self.error_rate:.2%})"

    # -- target generico ("cross" su qualunque colonna) -----------------------
    @property
    def system_key(self) -> Optional[str]:
        """Sistema a cui appartiene la colonna target, oppure None."""
        return self.cols.system_of(self.target_col)

    @property
    def is_system_target(self) -> bool:
        """True se il target è l'esito di un sistema riconosciuto nel tracciato."""
        return self.system_key is not None

    @property
    def system_label(self) -> str:
        """Nome del sistema del target; se il target non appartiene a un sistema, il nome della colonna."""
        return self.system_key or self.target_col

    @property
    def event_label(self) -> str:
        """Etichetta breve del caso analizzato, es. 'ALFA ERROR' o 'TIPO = RESO'."""
        return f"{self.system_label} {self.target_value}" if self.is_system_target else f"{self.target_col} = {self.target_value}"

    @property
    def target_item(self) -> str:
        """Nome dell'item target nelle association rules / colonne derivate."""
        return f"{self.target_col}={self.target_value}"

    @property
    def target_group_cols(self) -> List[str]:
        """Altre colonne dello stesso sistema del target (descrivono l'esito stesso: escluse dai modelli)."""
        return [c for c in self.cols.system(self.system_key) if c != self.target_col]

    @property
    def other_systems(self) -> Dict[str, List[str]]:
        """Sistemi diversi da quello del target: nome -> colonne."""
        return {s: cols for s, cols in self.cols.systems.items() if s != self.system_key}

    @property
    def other_system_cols(self) -> List[str]:
        return [c for cols in self.other_systems.values() for c in cols]

    def _group_col(self, rx: str) -> Optional[str]:
        for c in self.target_group_cols:
            if re.search(rx, c, re.IGNORECASE):
                return c
        return None

    @property
    def detail_col(self) -> Optional[str]:
        """Colonna di dettaglio/motivazione dell'esito target, se esiste."""
        return self._group_col(DETAIL_RX)

    @property
    def score_col(self) -> Optional[str]:
        return self._group_col(SCORE_RX)

    @property
    def session_col(self) -> Optional[str]:
        """Identificativo di sessione/contesto: quello del sistema del target, altrimenti uno che non
        appartiene a un altro sistema."""
        cands = self.cols.session_candidates()
        own = set(self.target_group_cols)
        other = set(self.other_system_cols)
        return next((c for c in cands if c in own), None) or next((c for c in cands if c not in other and c != self.target_col), None)

    @property
    def measure_col(self) -> Optional[str]:
        """Misura principale (importo, quantità): la prima colonna numerica il cui nome la indica."""
        if "measure_col" not in self._cache:
            from .loader import numeric_ratio
            self._cache["measure_col"] = next((c for c in self.cols.measure_candidates()
                                               if c != self.target_col and numeric_ratio(self.df[c]) >= 0.9), None)
        return self._cache["measure_col"]

    @property
    def decision_col(self) -> Optional[str]:
        """Colonna con la decisione complessiva (non di un singolo sistema), se esiste."""
        c = self.cols.get("decisione")
        return c if c and c != self.target_col else None

    @property
    def id_col(self) -> Optional[str]:
        """Identificativo del record: una colonna con un valore diverso per (quasi) ogni riga, meglio
        se il nome è quello di un identificativo."""
        if "id_col" not in self._cache:
            skip = {self.cols.get(k) for k in TIME_ROLES} | {self.target_col, self.measure_col}
            unique = [c for c in self.df.columns if c not in skip and self.n_rows and self.nunique(c) >= 0.999 * self.n_rows]
            self._cache["id_col"] = next((c for c in unique if looks_like_identifier(c)), None) or next(
                (c for c in unique if not _has_decimals(self, c)), None)
        return self._cache["id_col"]

    @property
    def critical_rx(self) -> str:
        """Regex dei valori che indicano un esito critico, compresi quelli aggiunti con --critical-values."""
        return critical_regex(self.critical_values)

    @property
    def file_prefix(self) -> str:
        """Prefisso dei file di output, es. 'esito_alfa_error'."""
        return slugify(f"{self.target_col}_{self.target_value}")


def _has_decimals(ctx: "AnalysisContext", col: str) -> bool:
    """True se la colonna è numerica con valori non interi (una misura, non un codice)."""
    v = ctx.numeric(col).dropna()
    return bool(len(v)) and bool((v != np.floor(v)).any())


def _is_code_like(ctx: "AnalysisContext", col: str) -> bool:
    """Colonna numerica che è un codice e non una misura: valori interi che sono quasi tutti diversi,
    oppure hanno tutti lo stesso numero di cifre (almeno 5), oppure stanno sotto un nome da identificativo."""
    v = ctx.numeric(col).dropna()
    if v.empty or (v != np.floor(v)).any():
        return False
    if looks_like_identifier(col) or ctx.nunique(col) > 0.9 * len(v):
        return True
    a = np.abs(v.to_numpy())
    digits = np.floor(np.log10(np.maximum(a, 1))).astype(np.int64) + 1
    return bool(digits.min() == digits.max() and digits.min() >= 5)


def key_dimensions(ctx: "AnalysisContext", limit: Optional[int] = None) -> List[str]:
    """Dimensioni descrittive del tracciato: colonne categoriche utilizzabili nei modelli che non
    riportano l'esito di un sistema. Sono ordinate per forza dell'associazione con il target
    (Cramér's V della fase 3) quando è disponibile, altrimenti nell'ordine del file."""
    skip = set(ctx.target_group_cols) | set(ctx.other_system_cols)
    dims = [c for c in categorical_columns(ctx) if c not in skip]
    summ = ctx.meta.get("dim_summary")
    if summ is not None and not summ.empty:
        rank = {d: i for i, d in enumerate(summ["DIMENSIONE"])}
        dims.sort(key=lambda c: rank.get(c, len(rank)))
    return dims[:limit] if limit else dims


def derived_presence_frame(ctx: AnalysisContext, min_null: float = 0.005) -> pd.DataFrame:
    """Per le colonne che i modelli non possono usare direttamente (identificativi, alta cardinalità,
    numeriche) crea una feature binaria '<COL>_PRESENTE' = SI/NO: di un identificativo non conta
    il valore, conta se il dato è stato raccolto o meno."""
    cols: Dict[str, Any] = {}
    leak = set(ctx.meta.get("leakage_cols", []))
    for c in ctx.df.columns:
        if c == ctx.target_col or c in leak:
            continue
        if ctx.nunique(c) <= ctx.max_cardinality:
            continue  # già usata come categorica
        nul = ctx.isnull(c)
        share = float(nul.mean())
        if min_null <= share <= 1 - min_null:
            cols[f"{c}_PRESENTE"] = np.where(nul, "NO", "SI")
    return pd.DataFrame(cols, index=ctx.df.index)


def derived_numeric_bins(ctx: AnalysisContext, n_bins: int = 5) -> pd.DataFrame:
    """Discretizza in quantili le colonne numeriche continue (punteggi, importi, contatori)
    così che entrino nel pattern mining, nell'albero e nel clustering come '<COL>_BIN' = 'Q1 <=x' ...
    I nulli diventano il livello '(NULL)'."""
    from .loader import numeric_ratio
    cols: Dict[str, Any] = {}
    leak = set(ctx.meta.get("leakage_cols", []))
    # chiavi temporali e coordinate non hanno senso come bin quantilici (e nemmeno i codici: vedi sotto)
    skip = {ctx.cols.get(k) for k in TIME_ROLES} - {None}
    skip |= {c for c in ctx.df.columns if re.search(COORDINATE_RX, c, re.IGNORECASE)}
    for c in ctx.df.columns:
        if c == ctx.target_col or c in leak or c in skip:
            continue
        s = ctx.df[c]
        if ctx.nunique(c) <= ctx.max_cardinality or numeric_ratio(s) < 0.95:
            continue
        v = ctx.numeric(c)
        if v.nunique() <= 1 or _is_code_like(ctx, c):     # identificativi e codici numerici non sono misure
            continue
        try:
            q = pd.qcut(v, q=n_bins, duplicates="drop")
        except ValueError:
            continue
        if q.cat.categories.size < 2:
            continue
        labels = [f"Q{i + 1} ({cat.left:g}, {cat.right:g}]" for i, cat in enumerate(q.cat.categories)]
        codes = q.cat.codes.to_numpy()
        cols[f"{c}_BIN"] = np.where(codes < 0, "(NULL)", np.array(labels, dtype=object)[np.clip(codes, 0, None)])
    ctx.meta["binned_cols"] = [c[:-4] for c in cols]
    return pd.DataFrame(cols, index=ctx.df.index)


def derived_feature_frame(ctx: AnalysisContext) -> pd.DataFrame:
    """Unione delle feature derivate: presenza SI/NO (identificativi) e bin quantilici (numeriche).
    Una colonna numerica discretizzata non riceve anche la feature di presenza (il bin '(NULL)' basta).
    Il risultato è calcolato una volta e riusato dalle fasi 4, 7 e 8."""
    if "derived" not in ctx._cache:
        bins = derived_numeric_bins(ctx)
        pres = derived_presence_frame(ctx)
        pres = pres[[c for c in pres.columns if c[:-len("_PRESENTE")] not in ctx.meta.get("binned_cols", [])]]
        ctx._cache["derived"] = pd.concat([pres, bins], axis=1)
    return ctx._cache["derived"]


def is_null_series(s: pd.Series) -> pd.Series:
    """True dove il valore è NaN oppure un token di nullità (vedi NULL_TOKENS)."""
    if s.dtype != object and not pd.api.types.is_string_dtype(s):
        return s.isna()
    na = s.isna()
    # i token si cercano tra i valori distinti: di solito sono pochi anche su milioni di righe
    tokens = [u for u in s[~na].unique() if isinstance(u, str) and u.strip().upper() in NULL_TOKENS]
    return (na | s.isin(tokens)) if tokens else na


def contains_token(s: pd.Series, pattern: str) -> pd.Series:
    """True dove il valore, in maiuscolo, contiene la regex ``pattern``. La regex gira sui soli valori
    distinti (pochi, nelle colonne categoriche, anche su milioni di righe).

    Le regex passate qui non devono usare lookaround: con stringhe pyarrow pandas usa la libreria
    RE2, che non li supporta. Per delimitare un token si usa ``(?:^|[^A-Z0-9])TOKEN(?:$|[^A-Z0-9])``."""
    uniq = pd.Series(s.dropna().unique(), dtype=object)
    if uniq.empty:
        return pd.Series(False, index=s.index)
    hit = uniq[uniq.astype(str).str.upper().str.contains(pattern, na=False, regex=True).to_numpy()]
    return s.isin(set(hit)).fillna(False).astype(bool)


def detect_leakage_columns(ctx: AnalysisContext) -> List[str]:
    """Colonne che sono *conseguenza* del caso target e non causa (es. un punteggio o una regola
    vuoti proprio perché l'elaborazione non è stata completata): nulli nel >=99% delle righe target ma non
    nel resto del dataset. Vanno escluse dai modelli per evitare pattern tautologici."""
    leak: List[str] = []
    is_err = ctx.is_error.to_numpy()
    n_err, n_ok = int(is_err.sum()), int((~is_err).sum())
    for c in ctx.df.columns:
        if c == ctx.target_col or not n_err or not n_ok:
            continue
        nul = ctx.isnull(c).to_numpy()
        # nullo in (quasi) tutti gli errori ma non strutturalmente nullo nel resto
        if nul[is_err].mean() >= 0.99 and nul[~is_err].mean() <= 0.95:
            leak.append(c)
    for c in (ctx.detail_col, ctx.score_col):  # dettaglio e score dell'esito target descrivono l'esito stesso
        if c and c not in leak:
            leak.append(c)
    ctx.meta["leakage_cols"] = leak
    if leak:
        LOG.info("Colonne escluse dai modelli perché conseguenza dell'errore: %s", leak)
    return leak


def detect_alias_columns(ctx: AnalysisContext, min_purity: float = 0.999) -> List[str]:
    """Colonne da cui il valore della colonna target si ricava esattamente: codifiche più fini dello
    stesso dato (es. un codice 'T07' per TIPO 'RESO'). Usarle per spiegare il target
    produce solo tautologie ("COD_TIPO=T07 -> 100% RESO"), quindi vanno fuori dai modelli.

    Non basta che una colonna predica bene il caso analizzato: "APP=412 -> sempre ERROR" è una
    causa, non un alias. Una colonna è un alias solo se determina TUTTI i valori della colonna
    target (almeno tre valori distinti): a ogni suo valore corrisponde un solo valore del target."""
    if ctx.nunique(ctx.target_col) < 3:
        return []
    n = ctx.n_rows
    t_codes, t_uniques = pd.factorize(ctx.df[ctx.target_col].to_numpy(dtype=object), use_na_sentinel=False)
    k_t = len(t_uniques)
    known = set(ctx.meta.get("leakage_cols", []))
    alias: List[str] = []
    for c in ctx.df.columns:
        if c == ctx.target_col or c in known or not 3 <= ctx.nunique(c) <= ctx.max_cardinality:
            continue
        x_codes, x_uniques = pd.factorize(ctx.df[c].to_numpy(dtype=object), use_na_sentinel=False)
        ct = np.bincount(x_codes * k_t + t_codes, minlength=len(x_uniques) * k_t).reshape(len(x_uniques), k_t)
        # quota di righe il cui valore target è quello "previsto" dal valore della colonna
        if ct.max(axis=1).sum() / n >= min_purity and len(set(ct.argmax(axis=1))) >= 3:
            alias.append(c)
    return alias


def detect_perfect_separators(ctx: AnalysisContext, max_impurity: float = 0.001) -> List[str]:
    """Colonne categoriche che da sole separano il caso target: ogni loro valore sta (quasi) tutto
    dentro o tutto fuori dal caso. Può essere la causa deterministica ("APP=412 va sempre in
    errore") oppure una codifica dello stesso dato che il controllo più severo di
    :func:`detect_alias_columns` non riconosce: i dati non bastano a distinguere i due casi, quindi
    le colonne restano nei modelli e il report lo segnala (si escludono con ``--exclude``)."""
    err = ctx.is_error.to_numpy()
    known = set(ctx.meta.get("leakage_cols", [])) | set(ctx.target_group_cols)
    out: List[str] = []
    for c in ctx.df.columns:
        if c == ctx.target_col or c in known or not 2 <= ctx.nunique(c) <= ctx.max_cardinality:
            continue
        codes, uniques = pd.factorize(ctx.df[c].to_numpy(dtype=object), use_na_sentinel=False)
        n = np.bincount(codes, minlength=len(uniques))
        e = np.bincount(codes[err], minlength=len(uniques))
        if np.minimum(e, n - e).sum() / max(ctx.n_rows, 1) <= max_impurity:
            out.append(c)
    return out


def categorical_columns(ctx: AnalysisContext, exclude: Optional[List[str]] = None, max_card: Optional[int] = None) -> List[str]:
    """Colonne categoriche utilizzabili nei modelli: cardinalità contenuta, non costanti, non identificativi, non leakage."""
    exclude = set(exclude or []) | set(ctx.meta.get("leakage_cols", []))
    max_card = max_card or ctx.max_cardinality
    out = []
    n = len(ctx.df)
    for c in ctx.df.columns:
        if c in exclude or c == ctx.target_col:
            continue
        nun = ctx.nunique(c)
        if nun <= 1 or nun > max_card:
            continue
        if n and nun / n > 0.5:        # quasi-identificativo
            continue
        out.append(c)
    return out
