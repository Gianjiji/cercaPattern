"""
FASE 1 - Data profiling.

Produce:
- numeri chiave (righe, errori, percentuale errori);
- tabella per campo: tipo inferito, % nulli, valori distinti, campo costante,
  top valori, descrizione inferita;
- distribuzione dei valori (top-N) per ogni colonna categorica.
"""
from __future__ import annotations

import re

import pandas as pd

from .config import AnalysisContext, Finding, LOG
from .dictionary import _infer_type, describe_column
from .loader import numeric_ratio, parse_datetime

def _temporal_name(name: str) -> bool:
    """Il nome suggerisce una data/ora? Le sigle corte valgono solo come parola intera
    ("CORPORATE" contiene "ORA", "COUNTS" contiene "TS")."""
    up = name.upper()
    return bool(re.search(r"DATE|DATA|TIME|GIORNO|DAY", up)) or any(t in {"ORA", "ORARIO", "TS", "DT"} for t in re.split(r"[^A-Z]+", up))


def run_profiling(ctx: AnalysisContext, top_n: int = 10) -> pd.DataFrame:
    df = ctx.df
    n = len(df)
    rows, dist_rows = [], []
    for c in df.columns:
        s = df[c]
        nulls = ctx.isnull(c)
        nn = s[~nulls]
        n_unique = int(nn.nunique())
        num_ratio = numeric_ratio(nn)
        # parsing date solo se il nome lo suggerisce
        dt_ratio = 0.0
        if _temporal_name(c) and not nn.empty:
            dt_ratio = float(parse_datetime(nn.head(2000)).notna().mean())
        tipo = _infer_type(nn, n_unique, n, num_ratio, dt_ratio)
        vc = nn.value_counts()
        top = "; ".join(f"{k}={v}" for k, v in vc.head(5).items())
        desc, _ = describe_column(c, tipo)
        rows.append({
            "CAMPO": c,
            "TIPO_INFERITO": tipo,
            "PCT_NULL": round(100 * nulls.mean(), 2),
            "VALORI_DISTINTI": n_unique,
            "COSTANTE": "SI" if n_unique <= 1 else "NO",
            "TOP_VALORI": top,
            "DESCRIZIONE_INFERITA": desc,
        })
        if tipo in ("categorico", "booleano") or n_unique <= 50:
            tot = len(nn)
            for k, v in vc.head(top_n).items():
                dist_rows.append({"CAMPO": c, "VALORE": k, "N": int(v), "PCT": round(100 * v / tot, 2) if tot else 0})
    profile = pd.DataFrame(rows)
    dist = pd.DataFrame(dist_rows)
    ctx.add_table("01_profiling", profile)
    ctx.add_table("01_distribuzioni", dist)

    const_cols = profile.loc[profile["COSTANTE"] == "SI", "CAMPO"].tolist()
    ctx.meta.update({
        "n_rows": n, "n_errors": ctx.n_errors, "error_rate": ctx.error_rate,
        "n_cols": len(df.columns), "const_cols": const_cols,
    })
    LOG.info("Profiling: %s righe, %s colonne, righe target %s, campi costanti=%s",
             f"{n:,}", len(df.columns), ctx.pretty_errors(), len(const_cols))
    if const_cols:
        ctx.add_finding(Finding(
            "FASE 1", "QUALITA' DATI", f"{len(const_cols)} campi costanti (non informativi)",
            ", ".join(const_cols[:15]), "ALTA",
            "Escludere dal perimetro analitico; verificare se la costanza è attesa o indica mancata alimentazione."))
    # colonne con nulli elevati
    high_null = profile[profile["PCT_NULL"] >= 50].sort_values("PCT_NULL", ascending=False)
    if not high_null.empty:
        ctx.add_finding(Finding(
            "FASE 1", "QUALITA' DATI", f"{len(high_null)} campi con >=50% di valori nulli",
            "; ".join(f"{r.CAMPO}={r.PCT_NULL}%" for r in high_null.head(10).itertuples()), "MEDIA",
            "Verificare se i nulli sono strutturali (campo non applicabile) o indicano mancata alimentazione dati."))
    return profile
