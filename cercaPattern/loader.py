"""
Caricamento robusto del CSV e conversioni di tipo.

- rileva l'encoding su un campione di testa (BOM UTF-16 incluso) e ripiega
  su cp1252/latin-1 se il resto del file contiene byte non validi;
- rileva il separatore misurando la coerenza del numero di campi (rispettando
  le virgolette), con ``csv.Sniffer`` e un conteggio grezzo come ripieghi;
- legge tutto come stringa (nessuna inferenza aggressiva di pandas) e
  normalizza i token di nullità in NaN;
- ``coerce_numeric`` deduce la convenzione numerica (virgola o punto decimale,
  separatore delle migliaia) a livello di colonna, non valore per valore;
- ``parse_datetime`` riconosce il formato su un campione e lo applica a tutta
  la colonna (ISO, italiano, US, con fuso, chiavi compatte yyyymmdd[HH[MM[SS]]],
  epoch Unix).
"""
from __future__ import annotations

import codecs
import csv
import io
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import LOG, NULL_TOKENS

ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
SEPARATORS = [";", ",", "|", "\t"]
SNIFF_BYTES = 262_144


# ---------------------------------------------------------------------------
# Formato del file
# ---------------------------------------------------------------------------
def _read_head(path: Path, n: int = SNIFF_BYTES) -> bytes:
    """Solo i primi ``n`` byte: il file può pesare diversi GB."""
    with open(path, "rb") as fh:
        return fh.read(n)


def _decodes(raw: bytes, enc: str) -> bool:
    """True se ``raw`` è decodificabile con ``enc``. Il campione è tagliato a un numero fisso di
    byte, quindi può finire a metà di un carattere multibyte: il decoder incrementale lo tollera."""
    try:
        codecs.getincrementaldecoder(enc)().decode(raw, final=False)
        return True
    except UnicodeDecodeError:
        return False


def _detect_encoding(raw: bytes) -> str:
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"
    for enc in ENCODINGS:
        if _decodes(raw, enc):
            return enc
    return "latin-1"


def _sniff(sample: str) -> Optional[str]:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(SEPARATORS))
        return dialect.delimiter
    except csv.Error:
        return None


def _best_separator(sample: str) -> str:
    """Ultimo ripiego: separatore con conteggio più stabile tra le prime righe."""
    lines = [l for l in sample.splitlines() if l.strip()][:50]
    best, best_score = ",", -1.0
    for sep in SEPARATORS:
        counts = [l.count(sep) for l in lines]
        if not counts or max(counts) == 0:
            continue
        score = np.mean(counts) - np.std(counts) * 2
        if score > best_score:
            best, best_score = sep, score
    return best


def _consistent_separator(sample: str, truncated: bool, max_records: int = 300) -> Optional[str]:
    """Separatore che divide l'intestazione in più campi e dà lo stesso numero di campi sul
    maggior numero di righe. Usa ``csv.reader``, quindi un separatore dentro un campo tra
    virgolette (testo libero, importi "1.234,56") non conta."""
    best, best_key = None, (0.0, 0)
    for sep in SEPARATORS:
        try:
            rows = []
            for i, r in enumerate(csv.reader(io.StringIO(sample), delimiter=sep)):
                if i >= max_records:
                    break
                if r:
                    rows.append(len(r))
        except csv.Error:
            continue
        if truncated and len(rows) > 2:
            rows = rows[:-1]  # l'ultima riga del campione può essere tagliata a metà
        if not rows or rows[0] < 2:
            continue
        share = sum(n == rows[0] for n in rows) / len(rows)
        key = (round(share, 3), rows[0])
        if key > best_key:
            best, best_key = sep, key
    return best if best_key[0] >= 0.6 else None


def detect_format(path: Path, encoding: Optional[str] = None, sep: Optional[str] = None) -> Tuple[str, str]:
    """Restituisce (encoding, separatore) rilevati o forzati."""
    raw = _read_head(Path(path))
    enc_found = encoding or _detect_encoding(raw)
    sample = raw.decode(enc_found, errors="replace")
    sep_found = sep or _consistent_separator(sample, truncated=len(raw) >= SNIFF_BYTES) or _sniff(sample) or _best_separator(sample)
    LOG.info("Formato rilevato: encoding=%s separatore=%r", enc_found, sep_found)
    return enc_found, sep_found


def _unique_columns(columns) -> List[str]:
    """Nomi colonna senza spazi ai bordi e senza duplicati (' A' e 'A' diventano 'A' e 'A_2')."""
    out: List[str] = []
    seen: set = set()
    for c in columns:
        name = str(c).strip().lstrip("﻿")
        base, i = name, 2
        while name in seen:
            name = f"{base}_{i}"
            i += 1
        seen.add(name)
        out.append(name)
    return out


def load_csv(path: Path, encoding: Optional[str] = None, sep: Optional[str] = None,
             nrows: Optional[int] = None) -> pd.DataFrame:
    """Carica il CSV come stringhe e normalizza i valori nulli."""
    enc, sp = detect_format(path, encoding, sep)
    kw = dict(dtype=str, keep_default_na=False, na_values=[], nrows=nrows, on_bad_lines="warn")
    if len(sp) > 1:   # separatore multi-carattere: solo il motore python lo supporta (e lo legge come regex)
        kw.update(sep=re.escape(sp), engine="python")
    else:
        kw.update(sep=sp, engine="c", low_memory=False)
    # l'encoding è stato dedotto dai primi 256 KB: se più avanti compaiono byte non validi si ripiega
    candidates = [enc] if encoding else [enc] + [e for e in ("cp1252", "latin-1") if e != enc]
    df = None
    for i, e in enumerate(candidates):
        try:
            df = pd.read_csv(path, encoding=e, **kw)
            if i:
                LOG.warning("Il file contiene byte non validi per %s: riletto come %s", enc, e)
            break
        except UnicodeDecodeError:
            if i == len(candidates) - 1:
                raise
    renamed = _unique_columns(df.columns)
    if renamed != [str(c) for c in df.columns]:
        dup = [n for n, o in zip(renamed, df.columns) if n != str(o).strip()]
        if dup:
            LOG.warning("Nomi colonna duplicati rinominati: %s", dup)
    df.columns = renamed
    # rimuove colonne completamente vuote con nome vuoto/"Unnamed" (separatore in coda alle righe)
    drop = [c for c in df.columns if (c == "" or c.startswith("Unnamed")) and df[c].astype(str).str.strip().eq("").all()]
    if drop:
        df = df.drop(columns=drop)
    # strip + token nulli -> NaN
    for c in df.columns:
        df[c] = _normalize_column(df[c])
    LOG.info("Caricate %s righe x %s colonne da %s", f"{len(df):,}", len(df.columns), Path(path).name)
    return df


def _normalize_column(s: pd.Series) -> pd.Series:
    """Toglie gli spazi ai bordi e trasforma i token di nullità in NaN. Se la colonna ha pochi valori
    distinti si normalizzano quelli e si rimappano le righe: molto più veloce su file grandi."""
    uniq = s.unique()
    if len(uniq) * 4 <= len(s):
        clean = {}
        for u in uniq:
            t = str(u).strip()
            clean[u] = np.nan if t.upper() in NULL_TOKENS else t
        out = s.map(clean)
        return out.astype(s.dtype) if out.dtype != s.dtype else out
    t = s.astype(str).str.strip()
    return t.mask(t.str.upper().isin(NULL_TOKENS), np.nan)


# ---------------------------------------------------------------------------
# Numeri
# ---------------------------------------------------------------------------
_NUMERIC_SHAPE = r"[+-]?[\d.,]+"


def number_convention(values: pd.Series) -> Tuple[str, str]:
    """(separatore decimale, separatore migliaia) dedotti dall'INTERA colonna.

    Decidere valore per valore è sbagliato: "0.123" e "12.500" in una colonna di punteggi sono
    decimali, non 123 e 12500. Regole:
    - punto e virgola insieme nello stesso valore: il separatore più a destra è quello decimale;
    - solo virgole: decimale (uso italiano), salvo gruppi di tre cifre ripetuti ("1,234,567");
    - solo punti: decimale, salvo gruppi di tre cifre ripetuti ("1.234.567") oppure una colonna
      di interi all'italiana ("950", "1.234", "12.500": ogni punto è seguito da tre cifre,
      nessun valore inizia per zero e i valori senza punto hanno al massimo tre cifre).
    """
    t = values.dropna().astype(str)
    t = t[t.str.fullmatch(_NUMERIC_SHAPE)]
    if t.empty:
        return ".", ""
    has_dot = t.str.contains(".", regex=False)
    has_com = t.str.contains(",", regex=False)
    both = has_dot & has_com
    if both.any():
        v = t[both]
        dot_last = (v.str.rfind(".") > v.str.rfind(",")).mean() >= 0.5
        return (".", ",") if dot_last else (",", ".")
    if has_com.any():
        v = t[has_com]
        if v.str.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+").all() and int(v.str.count(",").max()) >= 2:
            return ".", ","
        return ",", ""
    if has_dot.any():
        v = t[has_dot]
        if v.str.fullmatch(r"[+-]?[1-9]\d{0,2}(?:\.\d{3})+").all():
            rest = t[~has_dot].str.lstrip("+-")
            many_groups = int(v.str.count(r"\.").max()) >= 2
            small_rest = (not rest.empty) and bool((rest.str.len() <= 3).all()) and len(v) >= 3
            if many_groups or small_rest:
                return ",", "."
    return ".", ""


def coerce_numeric(s: pd.Series) -> pd.Series:
    """Converte in float64 (NaN dove non convertibile) stringhe numeriche in qualunque convenzione:
    "1234.56", "1234,56", "1.234,56", "1,234.56". La convenzione è decisa sulla colonna intera."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype("float64")
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype("float64")
    t = s.astype("string").str.replace(r"[\s ]", "", regex=True)
    dec, thou = number_convention(t)
    if thou:
        t = t.str.replace(thou, "", regex=False)
    if dec != ".":
        t = t.str.replace(dec, ".", regex=False)
    return pd.to_numeric(t, errors="coerce").astype("float64")


def numeric_ratio(s: pd.Series, sample: int = 50_000) -> float:
    """Quota di valori non nulli convertibili in numero (stimata su un campione se la colonna è grande)."""
    nn = s.dropna()
    if nn.empty:
        return 0.0
    if len(nn) > sample:
        nn = nn.sample(sample, random_state=0)
    return float(coerce_numeric(nn).notna().mean())


# ---------------------------------------------------------------------------
# Date e ore
# ---------------------------------------------------------------------------
# fuso in coda a un orario: "17:43:19+00:00", "17:43Z", "17:43:19.123 +0200". Va riconosciuto solo
# dopo un orario, altrimenti "14-09-2026" verrebbe letto come data "14-09" con offset "-2026".
_TZ_RX = r"(\d{1,2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?)\s*([+-]\d{2}:?\d{2}|Z|UTC|GMT)$"
_TZ_FIND = r"\d{1,2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?\s*(?:[+-]\d{2}:?\d{2}|Z|UTC|GMT)$"   # stessa regex senza gruppi
TZ_SUFFIX = _TZ_FIND  # nome storico

DATETIME_FORMATS = [
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M",
    "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M", "%d.%m.%Y %H:%M:%S",
    "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M",
    "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M",          # US: solo se il formato italiano non regge (giorni > 12)
    "%d/%m/%y %H:%M:%S", "%d/%m/%y %H:%M",
    "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%y",
    "%H:%M:%S", "%H:%M:%S.%f", "%H:%M",
]
_EPOCH_MIN, _EPOCH_MAX = 946_684_800, 4_102_444_800   # 2000-01-01 .. 2100-01-01 in secondi


def _to_naive(parsed: pd.Series, full_index: pd.Index) -> pd.Series:
    """Riporta il risultato di to_datetime su tutto l'indice, senza fuso orario e in ns."""
    if parsed.dtype == object:  # offset misti nel fallback "mixed": conversione valore per valore
        parsed = parsed.map(lambda v: v.tz_localize(None) if isinstance(v, pd.Timestamp) and v.tzinfo else v)
        parsed = pd.to_datetime(parsed, errors="coerce")
    if getattr(parsed.dtype, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    parsed = parsed.astype("datetime64[ns]")
    return parsed.reindex(full_index)


def _parse_compact(nn: pd.Series) -> Optional[pd.Series]:
    """Chiavi temporali di sole cifre: yyyymmdd, yyyymmddHH, yyyymmddHHMM, yyyymmddHHMMSS, epoch Unix.

    Il formato dipende dalla LUNGHEZZA: provare "%Y%m%d%H%M" su "2026091410" riesce comunque
    (ora 1, minuto 0) e sposta in silenzio tutte le fasce orarie. Qui ogni valore viene
    completato a 14 cifre e letto con l'unico formato non ambiguo."""
    lens = nn.str.len()
    ok_len = lens.isin([8, 10, 12, 14])
    padded = nn.where(ok_len).str.pad(14, side="right", fillchar="0")
    parsed = pd.to_datetime(padded, format="%Y%m%d%H%M%S", errors="coerce")
    if parsed.notna().mean() > 0.9:
        return parsed
    num = pd.to_numeric(nn, errors="coerce").astype("float64")
    for unit, scale in (("s", 1), ("ms", 1000)):
        inside = num.between(_EPOCH_MIN * scale, _EPOCH_MAX * scale)
        if inside.mean() > 0.9:
            LOG.info("Colonna temporale in epoch Unix (%s): convertita in UTC", unit)
            return pd.to_datetime(num.where(inside), unit=unit, errors="coerce")
    return None


def parse_datetime(s: pd.Series, dayfirst: bool = True) -> pd.Series:
    """Parsing robusto di date/ore con diversi formati (ISO, italiano, US, con fuso, TIME_KEY numerico).

    Le date con fuso ("2026-10-26 17:43:19+00:00", "...Z", offset misti solare/legale) vengono
    lette all'ora di orologio scritta nel file: l'offset viene scartato, così le analisi per
    fascia oraria mostrano gli stessi orari che l'utente vede nel CSV.
    """
    nn = s.dropna().astype(str).str.strip()
    nn = nn[nn != ""]
    if nn.empty:
        return pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    has_tz = nn.str.contains(_TZ_FIND, regex=True)
    if has_tz.any():
        offs = nn[has_tz].str.extract(_TZ_RX)[1].value_counts()
        LOG.info("Date con fuso orario %s su %d valori: si usa l'ora di orologio del file, l'offset viene scartato",
                 offs.head(5).to_dict(), int(has_tz.sum()))
        nn = nn.str.replace(_TZ_RX, r"\1", regex=True)
    if nn.str.fullmatch(r"\d{8,14}").mean() > 0.9:
        parsed = _parse_compact(nn)
        if parsed is not None:
            return _to_naive(parsed, s.index)
    # il formato si riconosce su un campione, poi si applica una volta sola a tutta la colonna
    sample = nn if len(nn) <= 5000 else nn.sample(5000, random_state=0)
    for fmt in DATETIME_FORMATS:
        if pd.to_datetime(sample, format=fmt, errors="coerce").notna().mean() <= 0.9:
            continue
        parsed = pd.to_datetime(nn, format=fmt, errors="coerce")
        if parsed.notna().mean() > 0.9:
            return _to_naive(parsed, s.index)
    try:
        parsed = pd.to_datetime(nn, errors="coerce", dayfirst=dayfirst, format="mixed")
    except (ValueError, OverflowError):  # fusi misti residui o date fuori intervallo
        try:
            parsed = pd.to_datetime(nn, errors="coerce", dayfirst=dayfirst, format="mixed", utc=True)
        except (ValueError, OverflowError):
            parsed = pd.Series(pd.NaT, index=nn.index, dtype="datetime64[ns]")
    return _to_naive(parsed, s.index)
