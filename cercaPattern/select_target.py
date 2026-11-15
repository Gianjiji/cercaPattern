"""
Selezione interattiva della colonna target e dei valori da analizzare.

Dopo il caricamento del CSV lo script mostra le colonne disponibili (con
cardinalità, % nulli ed esempi) e chiede quale analizzare; poi mostra la
distribuzione dei valori della colonna scelta e chiede quali valori
definiscono il "caso da spiegare" (uno o più, separati da virgola).
Tutte le misure delle fasi successive vengono calcolate su quella scelta.

Il programma non ha un target predefinito: propone le colonne che hanno l'aspetto di un esito
(pochi valori, uno dei quali minoritario e con il nome di un errore, un blocco o un livello alto)
e usa la più plausibile quando l'utente non sceglie (INVIO a terminale, ``--non-interactive``).

Le funzioni accettano ``input_fn``/``print_fn`` per poter essere testate
senza terminale. In assenza di stdin (EOF) si ricade sui default.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional, Tuple

import pandas as pd

from .config import LOG, is_null_series

MAX_VALUES_SHOWN = 30


def describe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Tabella sintetica delle colonne: distinti, % nulli, esempi."""
    rows = []
    for i, c in enumerate(df.columns, 1):
        s = df[c]
        nul = is_null_series(s)
        ex = s[~nul].astype(str).str.slice(0, 18).unique()[:3].tolist()
        rows.append({"N": i, "COLONNA": c, "DISTINTI": int(s[~nul].nunique()),
                     "PCT_NULL": round(100 * float(nul.mean()), 1), "ESEMPI": ", ".join(ex)})
    return pd.DataFrame(rows)


# valori che di solito indicano l'esito "da spiegare" di un controllo (parole intere)
_OUTCOME_RX = re.compile(r"(?:^|[^A-Z0-9])(?:ERR(?:OR|ORE)?|KO|FAIL(?:ED|URE)?|TIMEOUT|DENY|DENIED|BLOCK(?:ED)?|REJECT(?:ED)?|DECLINE(?:D)?"
                         r"|RIFIUTAT[AO]|ANNULLAT[AO]|HIGH|ALTO|ALERT|RED)(?:$|[^A-Z0-9])")
_TARGET_NAME_RX = re.compile(r"RISK|ESITO|OUTCOME|STATUS|STATO|RESULT|RISULTATO|LEVEL|LIVELLO|ACTION|AZIONE|DECISION|ERROR")


def suggest_targets(df: pd.DataFrame, max_suggestions: int = 5, max_values: int = 30) -> pd.DataFrame:
    """Coppie (colonna, valore) plausibili come caso da spiegare: colonne con pochi esiti in cui un
    valore minoritario ha l'aspetto di un errore/blocco/rischio. Serve a orientare la scelta, non a
    sostituirla. Colonne: COLONNA, VALORE, N_RIGHE, PCT."""
    rows = []
    n = max(len(df), 1)
    for c in df.columns:
        vc = df[c].value_counts()                      # prima i distinti (veloce), poi la normalizzazione sui soli distinti
        if not 2 <= len(vc) <= 3 * max_values:
            continue
        vc = vc.groupby(vc.index.astype(str).str.strip().str.upper()).sum().sort_values(ascending=False)
        if not 2 <= len(vc) <= max_values:
            continue
        name_bonus = 1.5 if _TARGET_NAME_RX.search(str(c).upper()) else 1.0
        for val, cnt in vc.items():
            share = cnt / n
            if not _OUTCOME_RX.search(str(val)) or share >= 0.5 or cnt < 5:
                continue
            # meglio un esito netto (ERROR, KO) su una colonna che ha il nome di un esito, e non troppo raro
            strong = 2.0 if re.search(r"ERR|KO|FAIL|TIMEOUT", str(val)) else 1.0
            rows.append({"COLONNA": c, "VALORE": str(val), "N_RIGHE": int(cnt), "PCT": round(100 * share, 2),
                         "_score": name_bonus * strong * min(share, 0.05) / 0.05})
    if not rows:
        return pd.DataFrame(columns=["COLONNA", "VALORE", "N_RIGHE", "PCT"])
    out = pd.DataFrame(rows).sort_values(["_score", "N_RIGHE"], ascending=False).drop_duplicates("COLONNA")
    return out.head(max_suggestions).drop(columns="_score").reset_index(drop=True)


def describe_suggestions(df: pd.DataFrame) -> str:
    """Suggerimenti in una riga, per i messaggi di errore: 'ESITO_BETA=KO (2.0%), ...'."""
    s = suggest_targets(df)
    return ", ".join(f"{r.COLONNA}={r.VALORE} ({r.PCT}%)" for r in s.itertuples())


def default_target(df: pd.DataFrame, target_col: Optional[str] = None, target_values: Optional[List[str]] = None) -> Tuple[Optional[str], Optional[str]]:
    """Target usato quando l'utente non lo indica del tutto: (colonna, valore), None per ciò che non si trova.

    - nessuna indicazione: il candidato più plausibile (vedi :func:`suggest_targets`);
    - solo i valori: la colonna più plausibile che li contiene;
    - solo la colonna: il suo valore più plausibile."""
    if target_col is not None:
        s = suggest_targets(df[[target_col]], max_suggestions=1)
        return target_col, (str(s.iloc[0]["VALORE"]) if not s.empty else None)
    if not target_values:
        s = suggest_targets(df, max_suggestions=1)
        return (s.iloc[0]["COLONNA"], str(s.iloc[0]["VALORE"])) if not s.empty else (None, None)
    wanted = {str(v).strip().upper() for v in target_values}
    cands = list(suggest_targets(df, max_suggestions=len(df.columns))["COLONNA"])
    cands += [c for c in df.columns if c not in cands and df[c].nunique() <= 3 * MAX_VALUES_SHOWN]
    for c in cands:
        present = {str(v).strip().upper() for v in df[c].dropna().unique()}
        if wanted & present:
            return c, None
    return None, None


def _match_column(answer: str, df: pd.DataFrame) -> Optional[str]:
    a = answer.strip()
    if not a:
        return None
    if a.isdigit() and 1 <= int(a) <= len(df.columns):
        return df.columns[int(a) - 1]
    upper = {c.strip().upper(): c for c in df.columns}
    return upper.get(a.upper())


def _parse_values(answer: str, dist: pd.DataFrame) -> List[str]:
    """Accetta numeri di riga (dalla distribuzione mostrata) o valori letterali, separati da virgola."""
    out: List[str] = []
    upper = {str(v).upper(): str(v) for v in dist["VALORE"]}
    for tok in answer.split(","):
        t = tok.strip()
        if not t:
            continue
        if t.isdigit() and 1 <= int(t) <= len(dist) and t.upper() not in upper:
            out.append(str(dist.iloc[int(t) - 1]["VALORE"]))
        elif t.upper() in upper:
            out.append(upper[t.upper()])
        else:
            raise ValueError(f"valore '{t}' non presente nella colonna")
    return out


def value_distribution(s: pd.Series) -> pd.DataFrame:
    vc = s.astype("string").str.strip().str.upper().fillna("(NULL)").value_counts()
    d = vc.rename_axis("VALORE").reset_index(name="N_RIGHE")
    d["PCT"] = (100 * d["N_RIGHE"] / max(len(s), 1)).round(2)
    d.insert(0, "N", range(1, len(d) + 1))
    return d


def select_target(df: pd.DataFrame, target_col: Optional[str] = None, target_value: Optional[str] = None,
                  input_fn: Callable[[str], str] = input, print_fn: Callable[[str], None] = print) -> Tuple[str, List[str]]:
    """Chiede (solo per le parti mancanti) colonna e valori target. Ritorna (colonna, [valori])."""
    default_col = default_target(df)[0]                   # il candidato più plausibile, proposto come scelta di INVIO

    # --- colonna ------------------------------------------------------------
    col = _match_column(target_col, df) if target_col else None
    if target_col and not col:
        print_fn(f"\nColonna '{target_col}' non trovata nel CSV.")
    while not col:
        print_fn("\n=== COLONNE DISPONIBILI NEL CSV ===")
        with pd.option_context("display.max_rows", None, "display.width", 160, "display.max_colwidth", 60):
            print_fn(describe_columns(df).to_string(index=False))
        sugg = suggest_targets(df)
        if not sugg.empty:
            print_fn("\nCandidati plausibili (colonna = valore da spiegare): "
                     + "; ".join(f"{r.COLONNA} = {r.VALORE} ({r.PCT}% delle righe)" for r in sugg.itertuples()))
        hint = f" [invio = {default_col}]" if default_col else ""
        try:
            ans = input_fn(f"\nColonna oggetto di analisi (numero o nome){hint}: ")
        except EOFError:
            ans = ""
        if not ans.strip() and default_col:
            col = default_col
        else:
            col = _match_column(ans, df)
            if not col:
                print_fn("Scelta non valida, riprova.")
                if not ans.strip():  # EOF senza default: impossibile proseguire interattivamente
                    raise SystemExit("Nessuna colonna target indicata.")
    dist = value_distribution(df[col])
    if len(dist) > 50:
        print_fn(f"\nATTENZIONE: '{col}' ha {len(dist)} valori distinti: è una colonna poco adatta come target "
                 f"(meglio una colonna categorica con pochi esiti). Puoi comunque indicare i valori da analizzare.")

    # --- valori -------------------------------------------------------------
    values: List[str] = []
    if target_value:
        try:
            values = _parse_values(target_value, dist)
        except ValueError as e:
            print_fn(f"\n{e}.")
    default_val = default_target(df, col)[1]              # il valore di quella colonna con l'aspetto di un esito da spiegare
    while not values:
        print_fn(f"\n=== DISTRIBUZIONE DI {col} ===")
        print_fn(dist.head(MAX_VALUES_SHOWN).to_string(index=False))
        if len(dist) > MAX_VALUES_SHOWN:
            print_fn(f"... altri {len(dist) - MAX_VALUES_SHOWN} valori non mostrati (indicali per nome)")
        hint = f" [invio = {default_val}]" if default_val else ""
        try:
            ans = input_fn(f"\nValore/i di {col} che definiscono il caso da analizzare (numeri o valori, separati da virgola){hint}: ")
        except EOFError:
            ans = ""
        if not ans.strip():
            if default_val:
                values = [default_val]
            else:
                raise SystemExit("Nessun valore target indicato.")
            continue
        try:
            values = _parse_values(ans, dist)
        except ValueError as e:
            print_fn(f"{e}. Riprova.")
    share = float(dist[dist["VALORE"].isin(values)]["PCT"].sum())
    if share >= 50:
        print_fn(f"\nNOTA: i valori scelti coprono il {share:.1f}% delle righe: l'analisi confronta una maggioranza con una minoranza.")
    print_fn(f"\nTarget: {col} in {values}  ({share:.2f}% delle righe)\n")
    LOG.info("Target selezionato interattivamente: %s in %s", col, values)
    return col, values
