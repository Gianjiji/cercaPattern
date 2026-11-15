"""
FASE 3 - Analisi del caso target per dimensione (tasso, lift, significatività).

Per ciascuna dimensione (ogni colonna con un numero contenuto di valori distinti) calcola:
- N righe, N errori, tasso di errore con intervallo di confidenza di Wilson;
- LIFT rispetto al tasso globale;
- QUOTA_ERRORI = quota di tutti gli errori che ricade nel valore;
- significatività del singolo valore (test binomiale contro il tasso globale) corretta per
  test multipli su tutte le coppie dimensione/valore (FDR di Benjamini-Hochberg);
- test chi-quadro sull'intera dimensione (associazione con il target);
- confronto della distribuzione ERROR vs NON-ERROR.

Le deduzioni automatiche scattano solo su scostamenti significativi: con centinaia di valori
confrontati, un tasso alto su venti righe è quasi sempre rumore.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .config import AnalysisContext, Finding, LOG
from .stats_utils import binom_excess_pvalue, fdr_bh, wilson_interval

MAX_Q_VALUE = 0.05
NULL_LABEL = "(NULL)"


def _labelled(ctx: AnalysisContext, col: str, null_label: str = NULL_LABEL) -> pd.Series:
    """Valori della colonna con i nulli resi espliciti."""
    return ctx.df[col].astype(object).where(~ctx.isnull(col), null_label)


def error_rate_table(ctx: AnalysisContext, col: str, min_n: int = 1, null_label: str = NULL_LABEL) -> pd.DataFrame:
    """Tabella del tasso di errore per valore di ``col``."""
    codes, uniques = pd.factorize(_labelled(ctx, col, null_label).to_numpy(dtype=object))
    err = ctx.is_error.to_numpy()
    n = np.bincount(codes, minlength=len(uniques))
    e = np.bincount(codes[err], minlength=len(uniques))
    lo, hi = wilson_interval(e, n)
    rate = e / np.maximum(n, 1)
    agg = pd.DataFrame({
        "DIMENSIONE": col, "VALORE": list(uniques), "N": n, "N_ERROR": e,
        "TASSO_ERRORE_PCT": (100 * rate).round(3),
        "TASSO_MIN_95_PCT": (100 * lo).round(3), "TASSO_MAX_95_PCT": (100 * hi).round(3),
        "LIFT": (rate / ctx.error_rate).round(2) if ctx.error_rate else np.nan,
        "QUOTA_ERRORI_PCT": (100 * e / max(ctx.n_errors, 1)).round(2),
        "QUOTA_VOLUME_PCT": (100 * n / max(ctx.n_rows, 1)).round(2),
        "P_VALUE": binom_excess_pvalue(e, n, ctx.error_rate),
    })
    agg = agg[agg["N"] >= min_n]
    return agg.sort_values(["N_ERROR", "TASSO_ERRORE_PCT"], ascending=False).reset_index(drop=True)


def chi_square(ctx: AnalysisContext, col: str) -> Dict[str, float]:
    """Chi-quadro e Cramér's V tra la colonna e il target."""
    codes, uniques = pd.factorize(_labelled(ctx, col).to_numpy(dtype=object))
    y = ctx.is_error.to_numpy().astype(np.int64)
    ct = np.bincount(codes * 2 + y, minlength=2 * len(uniques)).reshape(-1, 2)
    ct = ct[:, ct.sum(axis=0) > 0]
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return {"chi2": np.nan, "p_value": np.nan, "cramers_v": 0.0}
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    n = ct.sum()
    v = np.sqrt(chi2 / (n * (min(ct.shape) - 1))) if n else 0.0
    return {"chi2": float(chi2), "p_value": float(p), "cramers_v": float(v)}


def _dimensions(ctx: AnalysisContext, extra_dims: Optional[List[str]]) -> List[str]:
    """Dimensioni analizzate: quelle chieste con --dims (anche ad alta cardinalità), poi ogni colonna
    con un numero contenuto di valori distinti, nell'ordine del file."""
    dims: List[str] = []
    for c in extra_dims or []:
        if c in ctx.df.columns and c not in dims and c != ctx.target_col:
            dims.append(c)
        elif c not in ctx.df.columns:
            LOG.warning("Dimensione aggiuntiva '%s' non presente nel CSV: ignorata", c)
    known = set(dims)
    for c in ctx.df.columns:
        if c in known or c == ctx.target_col:
            continue
        if 1 < ctx.nunique(c) <= ctx.max_cardinality:
            dims.append(c)
    return dims


def run_error_analysis(ctx: AnalysisContext, extra_dims: Optional[List[str]] = None, top_k: int = 25) -> pd.DataFrame:
    """Esegue il ranking dei tassi di errore su tutte le dimensioni disponibili."""
    dims = _dimensions(ctx, extra_dims)
    tables: Dict[str, pd.DataFrame] = {}
    chi: Dict[str, Dict[str, float]] = {}
    ctx.progress.set_total(len(dims) + 1)
    for col in dims:
        ctx.progress.note(col)
        tables[col] = error_rate_table(ctx, col)
        chi[col] = chi_square(ctx, col)
        ctx.progress.step(col)
    # significatività corretta per test multipli su TUTTE le coppie dimensione/valore
    if tables:
        q = fdr_bh(np.concatenate([t["P_VALUE"].to_numpy() for t in tables.values()]))
        pos = 0
        for t in tables.values():
            t["Q_VALUE"] = q[pos:pos + len(t)]
            t["SIGNIFICATIVO"] = np.where((t["Q_VALUE"] <= MAX_Q_VALUE) & (t["LIFT"] > 1), "SI", "NO")
            pos += len(t)

    summary_rows = []
    min_n = max(20, int(0.001 * ctx.n_rows))
    leak = set(ctx.meta.get("leakage_cols", []))
    for col, t in tables.items():
        ctx.add_table(f"03_err_{col}", t.head(200))
        cs = chi[col]
        top = t.iloc[0] if not t.empty else None
        hi = t[t["N"] >= min_n].sort_values("TASSO_MIN_95_PCT", ascending=False)
        worst = hi.iloc[0] if not hi.empty else top
        summary_rows.append({
            "DIMENSIONE": col,
            "VALORI_DISTINTI": int(t.shape[0]),
            "CRAMERS_V": round(cs["cramers_v"], 4),
            "CHI2_P_VALUE": cs["p_value"],
            "VALORE_CON_PIU_ERRORI": top["VALORE"] if top is not None else None,
            "QUOTA_ERRORI_TOP_PCT": top["QUOTA_ERRORI_PCT"] if top is not None else None,
            "VALORE_TASSO_MAX": worst["VALORE"] if worst is not None else None,
            "TASSO_MAX_PCT": worst["TASSO_ERRORE_PCT"] if worst is not None else None,
            "LIFT_MAX": worst["LIFT"] if worst is not None else None,
            "VALORI_SIGNIFICATIVI": int((t["SIGNIFICATIVO"] == "SI").sum()),
        })
        # deduzioni automatiche: concentrazione forte (non per colonne conseguenza dell'errore)
        if col in leak:
            continue
        if top is not None and top["QUOTA_ERRORI_PCT"] >= 50 and top["LIFT"] >= 1.5 and top["SIGNIFICATIVO"] == "SI":
            ctx.add_finding(Finding(
                "FASE 3", "CONCENTRAZIONE", f"{col}={top['VALORE']} concentra il {top['QUOTA_ERRORI_PCT']:.1f}% degli errori",
                f"N={int(top['N'])}, errori={int(top['N_ERROR'])}, tasso={top['TASSO_ERRORE_PCT']:.2f}% (IC95 {top['TASSO_MIN_95_PCT']:.1f}-{top['TASSO_MAX_95_PCT']:.1f}), lift={top['LIFT']}",
                "ALTA", f"Analizzare l'integrazione con {ctx.system_label} specifica per {col}={top['VALORE']}." if ctx.is_system_target
                else f"Approfondire {col}={top['VALORE']}: concentra una quota rilevante di {ctx.event_label}."))
        # tasso sistematico: anche nel caso peggiore (limite inferiore dell'intervallo) almeno metà delle
        # righe è nel caso target, e il tasso è ben sopra quello globale
        bad = hi[(hi["TASSO_MIN_95_PCT"] >= 50) & (hi["LIFT"] >= 1.5) & (hi["SIGNIFICATIVO"] == "SI")]
        if not bad.empty:
            w = bad.iloc[0]
            if len(bad) == 1:
                title = f"{col}={w['VALORE']}: tasso di errore {w['TASSO_ERRORE_PCT']:.1f}% (sistematico)"
            else:
                title = (f"{col}: {len(bad)} valori con tasso di errore sistematico: "
                         + ", ".join(f"{r.VALORE} ({r.TASSO_ERRORE_PCT:.1f}%)" for r in bad.head(4).itertuples()))
            vals = ", ".join(f"{col}={v}" for v in bad["VALORE"].head(4))
            ctx.add_finding(Finding(
                "FASE 3", "TASSO ANOMALO", title,
                "; ".join(f"{r.VALORE}: N={int(r.N)}, errori={int(r.N_ERROR)}, lift={r.LIFT}, IC95 >= {r.TASSO_MIN_95_PCT:.1f}%" for r in bad.head(4).itertuples()), "ALTA",
                f"Un tasso >=50% indica un flusso non integrato o malconfigurato verso {ctx.system_label} per {vals}." if ctx.is_system_target
                else f"Per {vals} il caso {ctx.event_label} è la norma, non l'eccezione."))

    summary_cols = ["DIMENSIONE", "VALORI_DISTINTI", "CRAMERS_V", "CHI2_P_VALUE", "VALORE_CON_PIU_ERRORI", "QUOTA_ERRORI_TOP_PCT",
                    "VALORE_TASSO_MAX", "TASSO_MAX_PCT", "LIFT_MAX", "VALORI_SIGNIFICATIVI"]
    summary = pd.DataFrame(summary_rows, columns=summary_cols).sort_values("CRAMERS_V", ascending=False).reset_index(drop=True)
    if not dims:
        LOG.warning("Nessuna dimensione categorica analizzabile per i tassi di errore")
    ctx.add_table("03_ranking_dimensioni", summary)
    full = pd.concat(list(tables.values()), ignore_index=True) if tables else pd.DataFrame()
    ctx.add_table("03_tassi_errore_completo", full)
    ctx.meta["dims"] = dims
    ctx.meta["dim_summary"] = summary
    # profilo ERROR vs NON-ERROR per le dimensioni principali (top valori tra gli errori)
    comp_rows = []
    n_err, n_ok = max(ctx.n_errors, 1), max(ctx.n_rows - ctx.n_errors, 1)
    for col, t in tables.items():
        for r in t.head(top_k).itertuples():
            if r.N_ERROR == 0:
                continue
            pe, pn = r.N_ERROR / n_err, (r.N - r.N_ERROR) / n_ok
            comp_rows.append({"DIMENSIONE": col, "VALORE": r.VALORE, "PCT_TRA_ERRORI": round(100 * pe, 2),
                              "PCT_TRA_NON_ERRORI": round(100 * pn, 2),
                              "RAPPORTO": round(pe / pn, 2) if pn > 0 else np.inf})
    ctx.add_table("03_profilo_error_vs_ok", pd.DataFrame(comp_rows))
    LOG.info("Analisi errori completata su %d dimensioni", len(dims))
    return summary
