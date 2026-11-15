"""
FASE 9 - Report.

Output prodotti nella cartella di output:
Tutti i file hanno come prefisso il target analizzato (es. ``esito_alfa_error``), così
analisi di colonne diverse possono convivere nella stessa cartella:
- <prefisso>_report.xlsx        (multi-tab: una tab per tabella + executive + findings)
- <prefisso>_report.html        (interattivo: plotly + tabelle + executive summary)
- <prefisso>_righe_target.csv   (solo le righe del caso analizzato)
- <prefisso>_patterns.csv       (association rules + regole albero)
- <prefisso>_clusters.csv       (assegnazione cluster per riga) + <prefisso>_clusters_summary.csv
- <prefisso>_anomalies.csv      (ranking anomalie)
- <prefisso>_findings.csv / <prefisso>_cause_probabili.csv / <prefisso>_dizionario_dati.csv
- <prefisso>_executive_summary.md
- png_<prefisso>/*.png          (grafici statici)

I CSV sono scritti in UTF-8 con BOM e separatore ';' così Excel li apre con gli accenti corretti.
Nell'Excel le tabelle riga-per-riga (assegnazioni ai cluster, anomalie) sono limitate alle prime
righe: il dettaglio completo è nei CSV.

Convenzioni grafiche: una sola scala per asse (mai doppio asse), colori
categorici in ordine fisso, heatmap sequenziale monocromatica, marcatori
sottili, griglia recessiva.
"""
from __future__ import annotations

import html
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .anomaly import HIGH_PRIORITY_VIEW
from .config import AnalysisContext, LOG, key_dimensions, plural

# palette categorica (ordine fisso) + colori di stato
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA, C_GREEN, C_VIOLET, C_RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
CATEGORICAL = [C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA, C_GREEN, C_VIOLET, C_RED]
C_ERR = C_ORANGE      # "errore" come serie identitaria (non stato)
C_VOL = C_BLUE        # volumi
C_TEXT, C_MUTED, C_GRID = "#0b0b0b", "#52514e", "#e6e5e1"

EXCEL_ROW_LEVEL = {"07_cluster_assegnazioni", "08_anomalie_tutte"}   # una riga per riga target: il completo è nei CSV
EXCEL_ROW_LEVEL_MAX = 5_000
SCATTER_MAX_POINTS = 5_000


def _pca_points(ctx: AnalysisContext):
    """Proiezione PCA 2D delle righe target per i grafici a dispersione: (punti, indici delle righe
    mostrate) oppure None. Oltre SCATTER_MAX_POINTS righe se ne mostra un campione: un grafico con
    centinaia di migliaia di punti non si legge e gonfia l'HTML di decine di MB."""
    if "pca_points" in ctx._cache:
        return ctx._cache["pca_points"]
    X, labels = ctx.meta.get("cluster_X"), ctx.meta.get("cluster_labels")
    out = None
    if X is not None and labels is not None and X.shape[1] >= 2 and len(X) >= 2:
        from sklearn.decomposition import PCA
        P = PCA(n_components=2, random_state=ctx.random_state).fit_transform(X)
        idx = np.arange(len(X))
        if len(X) > SCATTER_MAX_POINTS:
            idx = np.sort(np.random.RandomState(ctx.random_state).choice(len(X), SCATTER_MAX_POINTS, replace=False))
        out = (P[idx], idx)
    ctx._cache["pca_points"] = out
    return out


# ---------------------------------------------------------------------------
# Grafici PNG (matplotlib / seaborn)
# ---------------------------------------------------------------------------
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": C_GRID, "grid.color": C_GRID, "axes.labelcolor": C_MUTED,
                                         "xtick.color": C_MUTED, "ytick.color": C_MUTED, "text.color": C_TEXT,
                                         "axes.titleweight": "bold", "axes.titlesize": 12, "font.size": 9})
    return plt, sns


def _save(ctx: AnalysisContext, fig, name: str) -> None:
    png_dir = ctx.out_dir / f"png_{ctx.file_prefix}"
    png_dir.mkdir(exist_ok=True)
    p = png_dir / f"{name}.png"
    fig.savefig(p, dpi=140, bbox_inches="tight", facecolor="white")
    ctx.png_paths.append(p)
    import matplotlib.pyplot as plt
    plt.close(fig)


def png_error_rate_dims(ctx: AnalysisContext, n_dims: int = 6, top: int = 12) -> None:
    plt, sns = _mpl()
    summ = ctx.results.get("03_ranking_dimensioni", pd.DataFrame())
    if summ.empty:
        return
    for dim in summ["DIMENSIONE"].head(n_dims):
        t = ctx.results.get(f"03_err_{dim}")
        if t is None or t.empty:
            continue
        t = t.head(top).iloc[::-1]
        fig, axes = plt.subplots(1, 2, figsize=(11, 0.35 * len(t) + 1.8), sharey=True)
        y = t["VALORE"].astype(str).str.slice(0, 35)
        axes[0].barh(y, t["N_ERROR"], color=C_ERR, height=0.6)
        axes[0].set_title(f"{ctx.event_label} per {dim}")
        axes[0].set_xlabel("N errori")
        axes[1].barh(y, t["TASSO_ERRORE_PCT"], color=C_VOL, height=0.6)
        axes[1].axvline(100 * ctx.error_rate, color=C_MUTED, ls="--", lw=1)
        axes[1].set_title("Tasso di errore (%)  --- media globale")
        axes[1].set_xlabel("% errore")
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
        _save(ctx, fig, f"03_tasso_errore_{re.sub(r'[^A-Za-z0-9]+', '_', dim)}")


def png_patterns(ctx: AnalysisContext, top: int = 15) -> None:
    plt, sns = _mpl()
    r = ctx.results.get("04_association_rules", pd.DataFrame())
    if r.empty:
        return
    r = r.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 0.4 * len(r) + 1.5))
    lab = r["PATTERN"].str.slice(0, 80)
    ax.barh(lab, r["COVERAGE_ERRORI_PCT"], color=C_VOL, height=0.6, label="% errori spiegati (coverage)")
    ax.scatter(r["CONFIDENZA_PCT"], lab, color=C_ERR, s=40, zorder=3, label="% errore nel pattern (confidenza)")
    ax.set_xlabel("%")
    ax.set_xlim(0, 100)
    ax.set_title(f"Top pattern associati a {ctx.target_item}")
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save(ctx, fig, "04_pattern_top")


def png_importance(ctx: AnalysisContext, top: int = 20) -> None:
    plt, sns = _mpl()
    agg = ctx.results.get("04_importance_per_colonna", pd.DataFrame())
    if not agg.empty:
        a = agg.head(top).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 0.35 * len(a) + 1.5))
        ax.barh(a["COLONNA"], a["QUOTA_PCT"], color=C_VOL, height=0.6)
        ax.set_xlabel("quota importanza (%)")
        ax.set_title("Importanza per colonna (XGBoost / SHAP)")
        ax.spines[["top", "right"]].set_visible(False)
        _save(ctx, fig, "04_importanza_colonne")
    imp = ctx.results.get("04_feature_importance", pd.DataFrame())
    if not imp.empty and "SHAP_QUANDO_ATTIVA" in imp:
        a = imp.head(top).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 0.35 * len(a) + 1.5))
        colors = [C_ERR if v > 0 else C_VOL for v in a["SHAP_QUANDO_ATTIVA"]]
        ax.barh(a["FEATURE"].str.slice(0, 60), a["SHAP_QUANDO_ATTIVA"], color=colors, height=0.6)
        ax.axvline(0, color=C_MUTED, lw=1)
        ax.set_xlabel(f"SHAP medio quando la feature è attiva  (>0 spinge verso {ctx.target_value}, in arancio)")
        ax.set_title("Direzione delle feature (SHAP)")
        ax.spines[["top", "right"]].set_visible(False)
        _save(ctx, fig, "04_shap_direzione")
    expl = ctx.meta.get("shap_expl")
    if expl is not None:
        try:
            import shap
            sv, Xs, feat = expl
            if len(Xs) > SCATTER_MAX_POINTS:   # il beeswarm disegna un punto per riga e per feature
                pick = np.random.RandomState(ctx.random_state).choice(len(Xs), SCATTER_MAX_POINTS, replace=False)
                sv, Xs = sv[pick], Xs[pick]
            plt.figure(figsize=(9, 7))
            shap.summary_plot(sv, Xs, feature_names=feat, max_display=20, show=False, color_bar=False)
            fig = plt.gcf()
            fig.suptitle("SHAP beeswarm (top 20 feature)", fontweight="bold")
            _save(ctx, fig, "04_shap_beeswarm")
        except Exception as e:
            LOG.warning("Beeswarm SHAP non generato: %s", e)


def png_temporal(ctx: AnalysisContext) -> None:
    plt, sns = _mpl()
    h = ctx.results.get("05_tasso_per_ora", pd.DataFrame())
    if not h.empty:
        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        axes[0].bar(h["ORA"], h["N_ERROR"], color=C_ERR, width=0.7)
        axes[0].set_ylabel("N errori")
        axes[0].set_title(f"{ctx.event_label} per ora del giorno")
        axes[1].bar(h["ORA"], h["TASSO_ERRORE_PCT"], color=C_VOL, width=0.7)
        axes[1].axhline(100 * ctx.error_rate, color=C_MUTED, ls="--", lw=1)
        axes[1].set_ylabel("% errore")
        axes[1].set_xlabel("ora")
        axes[1].set_xticks(range(0, 24))
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
        _save(ctx, fig, "05_istogramma_ore")
    hm = ctx.meta.get("heatmap_rate")
    if hm is not None:
        fig, ax = plt.subplots(figsize=(12, 3.8))
        sns.heatmap(hm, cmap="Blues", ax=ax, linewidths=0.5, linecolor="white", cbar_kws={"label": "% errore"})
        ax.set_title("Heatmap tasso di errore: giorno della settimana x ora")
        ax.set_xlabel("ora")
        ax.set_ylabel("")
        _save(ctx, fig, "05_heatmap_dow_ora")
    d = ctx.results.get("05_serie_giornaliera", pd.DataFrame())
    if not d.empty and len(d) > 1:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(d["GIORNO"], d["N"], color=C_VOL, lw=2)
        axes[0].set_ylabel("righe")
        axes[0].set_title("Serie storica giornaliera")
        axes[1].plot(d["GIORNO"], d["N_ERROR"], color=C_ERR, lw=2)
        axes[1].set_ylabel(f"righe {ctx.event_label}")
        axes[2].plot(d["GIORNO"], d["TASSO_ERRORE_PCT"], color=C_ERR, lw=2)
        axes[2].axhline(100 * ctx.error_rate, color=C_MUTED, ls="--", lw=1)
        axes[2].set_ylabel("% errore")
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
        fig.autofmt_xdate()
        _save(ctx, fig, "05_serie_giornaliera")
    hb = ctx.results.get("05_serie_oraria", pd.DataFrame())
    if not hb.empty and len(hb) > 24:
        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.plot(hb["BUCKET_ORA"], hb["N_ERROR"], color=C_ERR, lw=1.5)
        b = hb[hb["BURST"]]
        if not b.empty:
            ax.scatter(b["BUCKET_ORA"], b["N_ERROR"], s=36, facecolors="none", edgecolors=C_RED, lw=1.5, label="burst", zorder=3)
            ax.legend(frameon=False)
        ax.set_ylabel("errori / ora")
        ax.set_title("Serie oraria degli errori con burst evidenziati")
        ax.spines[["top", "right"]].set_visible(False)
        fig.autofmt_xdate()
        _save(ctx, fig, "05_serie_oraria_burst")


def png_functional(ctx: AnalysisContext) -> None:
    plt, sns = _mpl()
    det = ctx.results.get("06_detail_classificazione", pd.DataFrame())
    if not det.empty:
        a = det.iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 0.4 * len(a) + 1.5))
        ax.barh(a["CAUSA_TECNICA_INFERITA"], a["PCT"], color=C_ERR, height=0.6)
        ax.set_xlabel("% degli errori")
        ax.set_title(f"Cause tecniche inferite da {ctx.detail_col}")
        ax.spines[["top", "right"]].set_visible(False)
        _save(ctx, fig, "06_cause_tecniche_detail")
    pres = ctx.results.get("06_presenza_campi", pd.DataFrame())
    if not pres.empty:
        a = pres.head(25).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 0.45 * len(a) + 1.5))
        y = np.arange(len(a))
        ax.barh(y + 0.2, a["PCT_ASSENTE_TRA_ERRORI"], height=0.4, color=C_ERR, label=f"tra {ctx.event_label}")
        ax.barh(y - 0.2, a["PCT_ASSENTE_TRA_NON_ERRORI"], height=0.4, color=C_VOL, label="tra le altre righe")
        ax.set_yticks(y)
        ax.set_yticklabels(a["CAMPO"])
        ax.set_xlabel("% valori assenti")
        ax.set_title("Campi assenti: errori vs altre righe")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        _save(ctx, fig, "06_presenza_campi")


def png_clusters_anomalies(ctx: AnalysisContext) -> None:
    plt, sns = _mpl()
    cl = ctx.results.get("07_cluster", pd.DataFrame())
    if not cl.empty:
        a = cl.iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, 0.5 * len(a) + 1.5))
        ax.barh(a["ETICHETTA"].str.slice(0, 70), a["DIMENSIONE"], color=C_VOL, height=0.6)
        ax.set_xlabel("N errori")
        ax.set_title(f"Cluster {ctx.event_label}")
        ax.spines[["top", "right"]].set_visible(False)
        _save(ctx, fig, "07_cluster_dimensioni")
    pts = _pca_points(ctx)
    if pts is not None:
        P, idx = pts
        labels = ctx.meta["cluster_labels"].iloc[idx]
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, k in enumerate(sorted(labels.unique())):
            m = (labels == k).values
            ax.scatter(P[m, 0], P[m, 1], s=12, color=CATEGORICAL[i % len(CATEGORICAL)], alpha=0.6, label=f"cluster {k}", edgecolors="white", linewidths=0.3)
        an = ctx.results.get("08_anomalie_tutte")
        if an is not None and not an.empty:
            m = (an.loc[labels.index, "CONSENSO"] >= 2).values
            ax.scatter(P[m, 0], P[m, 1], s=50, facecolors="none", edgecolors=C_RED, linewidths=1.2, label="anomalia (consenso >= 2)")
        ax.set_title("Proiezione PCA degli errori: cluster e anomalie")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(frameon=False, fontsize=8, markerscale=1.5)
        ax.spines[["top", "right"]].set_visible(False)
        _save(ctx, fig, "08_pca_cluster_anomalie")


# ---------------------------------------------------------------------------
# Figure interattive plotly
# ---------------------------------------------------------------------------
def build_plotly_figures(ctx: AnalysisContext) -> None:
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        LOG.warning("plotly non disponibile: HTML senza grafici interattivi")
        return
    layout = dict(template="plotly_white", font=dict(size=12, color=C_TEXT), margin=dict(l=40, r=20, t=50, b=40),
                  plot_bgcolor="white", paper_bgcolor="white")
    summ = ctx.results.get("03_ranking_dimensioni", pd.DataFrame())
    for dim in summ["DIMENSIONE"].head(6) if not summ.empty else []:
        t = ctx.results.get(f"03_err_{dim}")
        if t is None or t.empty:
            continue
        t = t.head(15)
        fig = go.Figure()
        fig.add_bar(x=t["VALORE"].astype(str), y=t["TASSO_ERRORE_PCT"], marker_color=C_VOL, name="% errore",
                    customdata=np.stack([t["N"], t["N_ERROR"], t["QUOTA_ERRORI_PCT"]], axis=1),
                    hovertemplate="<b>%{x}</b><br>tasso errore: %{y:.2f}%<br>N: %{customdata[0]:,}<br>errori: %{customdata[1]:,}<br>quota errori: %{customdata[2]:.1f}%<extra></extra>")
        fig.add_hline(y=100 * ctx.error_rate, line_dash="dash", line_color=C_MUTED, annotation_text="media globale")
        fig.update_layout(title=f"Tasso {ctx.event_label} per {dim} (top 15 per numero di casi)", yaxis_title="% errore", **layout)
        ctx.figures[f"03_{dim}"] = fig
    r = ctx.results.get("04_association_rules", pd.DataFrame())
    if not r.empty:
        rr = r.head(40)
        fig = px.scatter(rr, x="COVERAGE_ERRORI_PCT", y="CONFIDENZA_PCT", size="N_ERRORI", hover_name="PATTERN",
                         hover_data={"LIFT": True, "N_RIGHE": True, "N_ERRORI": True}, color_discrete_sequence=[C_ERR])
        fig.update_layout(title="Pattern: coverage (x) vs confidenza (y); dimensione = N errori",
                          xaxis_title="% errori spiegati dal pattern", yaxis_title="% errore dentro il pattern", **layout)
        ctx.figures["04_pattern_scatter"] = fig
    agg = ctx.results.get("04_importance_per_colonna", pd.DataFrame())
    if not agg.empty:
        a = agg.head(20).iloc[::-1]
        fig = go.Figure(go.Bar(x=a["QUOTA_PCT"], y=a["COLONNA"], orientation="h", marker_color=C_VOL,
                               hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
        fig.update_layout(title="Importanza per colonna (XGBoost/SHAP)", xaxis_title="quota %", height=max(350, 24 * len(a) + 120), **layout)
        ctx.figures["04_importance"] = fig
    hm = ctx.meta.get("heatmap_rate")
    if hm is not None:
        fig = go.Figure(go.Heatmap(z=hm.values, x=[str(c) for c in hm.columns], y=list(hm.index), colorscale="Blues",
                                   colorbar=dict(title="% errore"), hovertemplate="%{y} %{x}h: %{z:.2f}%<extra></extra>"))
        fig.update_layout(title="Heatmap tasso di errore: giorno x ora", xaxis_title="ora", **layout)
        ctx.figures["05_heatmap"] = fig
    d = ctx.results.get("05_serie_giornaliera", pd.DataFrame())
    if not d.empty and len(d) > 1:
        fig = go.Figure()
        fig.add_scatter(x=d["GIORNO"], y=d["N_ERROR"], mode="lines", line=dict(color=C_ERR, width=2), name=ctx.event_label,
                        hovertemplate="%{x|%Y-%m-%d}: %{y} errori<extra></extra>")
        fig.update_layout(title=f"Serie storica giornaliera {ctx.event_label}", yaxis_title="errori", hovermode="x unified", **layout)
        ctx.figures["05_serie_errori"] = fig
        fig2 = go.Figure()
        fig2.add_scatter(x=d["GIORNO"], y=d["TASSO_ERRORE_PCT"], mode="lines", line=dict(color=C_VOL, width=2), name="% errore",
                         hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}%<extra></extra>")
        fig2.add_hline(y=100 * ctx.error_rate, line_dash="dash", line_color=C_MUTED)
        fig2.update_layout(title="Tasso di errore giornaliero", yaxis_title="% errore", hovermode="x unified", **layout)
        ctx.figures["05_serie_tasso"] = fig2
    hb = ctx.results.get("05_serie_oraria", pd.DataFrame())
    if not hb.empty and len(hb) > 24:
        fig = go.Figure()
        fig.add_scatter(x=hb["BUCKET_ORA"], y=hb["N_ERROR"], mode="lines", line=dict(color=C_ERR, width=1.5), name="errori/ora")
        b = hb[hb["BURST"]]
        if not b.empty:
            fig.add_scatter(x=b["BUCKET_ORA"], y=b["N_ERROR"], mode="markers", name="burst",
                            marker=dict(size=10, color="rgba(0,0,0,0)", line=dict(color=C_RED, width=2)))
        fig.update_layout(title="Serie oraria con burst", yaxis_title="errori / ora", hovermode="x unified", **layout)
        ctx.figures["05_serie_oraria"] = fig
    cl = ctx.results.get("07_cluster", pd.DataFrame())
    if not cl.empty:
        fig = go.Figure(go.Bar(x=cl["DIMENSIONE"], y=cl["ETICHETTA"], orientation="h", marker_color=C_VOL,
                               customdata=cl["CAMPI_DISCRIMINANTI"], hovertemplate="<b>%{y}</b><br>N=%{x}<br>%{customdata}<extra></extra>"))
        fig.update_layout(title="Cluster di errori", xaxis_title="N errori", height=max(300, 40 * len(cl) + 120), **layout)
        ctx.figures["07_cluster"] = fig
    pts = _pca_points(ctx)
    an = ctx.results.get("08_anomalie_tutte")
    if pts is not None and an is not None and not an.empty:
        P, idx = pts
        labels = ctx.meta["cluster_labels"].iloc[idx]
        idc = ctx.id_col
        shown = ctx.df.loc[labels.index]
        ids = shown[idc].astype(str).values if idc else shown.index.astype(str).values
        cons = an.loc[labels.index, "CONSENSO"].values
        fig = go.Figure()
        for i, k in enumerate(sorted(labels.unique())):
            m = (labels == k).values
            fig.add_scatter(x=P[m, 0], y=P[m, 1], mode="markers", name=f"cluster {k}",
                            marker=dict(size=6, color=CATEGORICAL[i % len(CATEGORICAL)], opacity=0.7),
                            text=ids[m], customdata=cons[m], hovertemplate="id %{text}<br>consenso anomalia: %{customdata}<extra>cluster " + str(k) + "</extra>")
        m = cons >= 2
        if m.any():
            fig.add_scatter(x=P[m, 0], y=P[m, 1], mode="markers", name="anomalia (consenso>=2)",
                            marker=dict(size=12, color="rgba(0,0,0,0)", line=dict(color=C_RED, width=1.5)), hoverinfo="skip")
        fig.update_layout(title="Proiezione PCA: cluster e anomalie", xaxis_title="PC1", yaxis_title="PC2", height=600, **layout)
        ctx.figures["08_pca"] = fig


# ---------------------------------------------------------------------------
# Executive summary (Markdown)
# ---------------------------------------------------------------------------
def _top_list(t: Optional[pd.DataFrame], n: int = 3) -> str:
    if t is None or t.empty:
        return "n/d"
    return "; ".join(f"{r.VALORE} ({r.QUOTA_ERRORI_PCT}% degli errori, tasso {r.TASSO_ERRORE_PCT}%)" for r in t.head(n).itertuples())


def _in_brief(ctx: AnalysisContext) -> List[str]:
    """Le conclusioni in quattro righe: causa più probabile, quanti problemi distinti, andamento
    nel tempo, quanto i dati disponibili spiegano il caso."""
    m, R = ctx.meta, ctx.results
    out: List[str] = []
    hyp = R.get("09_cause_probabili", pd.DataFrame())
    if not ctx.is_system_target:
        pass    # le ipotesi H1..H9 descrivono i guasti di un sistema che elabora le righe: per le altre colonne parlano i pattern
    elif not hyp.empty and hyp.iloc[0]["SCORE"] >= 0.4:
        h = hyp.iloc[0]
        also = [r.CODICE for r in hyp.iloc[1:].itertuples() if r.SCORE >= 0.4]
        out.append(f"**Causa più probabile**: {h['CAUSA_PROBABILE']} ({h['CODICE']}, confidenza {h['CONFIDENZA']})"
                   + (f"; concorrono {', '.join(also)}" if also else "") + ".")
    elif not hyp.empty:
        out.append("**Causa più probabile**: nessuna ipotesi raggiunge una confidenza sufficiente con i dati disponibili.")
    cover = R.get("04_copertura_cumulativa", pd.DataFrame())
    if not cover.empty and m.get("cover_steps"):
        first = cover.iloc[0]
        k = m["cover_steps"]
        out.append(f"**Problemi distinti**: {k} pattern {'spiega' if k == 1 else 'spiegano'} il {m['cover_explained_pct']:.0f}% dei casi"
                   + (f"; il primo, {first['PATTERN']}, da solo il {first['QUOTA_ERRORI_PCT']}%" if k > 1 else f" ({first['PATTERN']})")
                   + f". Fuori da {'questo pattern' if k == 1 else 'questi pattern'} il tasso è {m['cover_residual_rate_pct']:.2f}% (globale {ctx.error_rate:.2%}).")
    elif "04_association_rules" in R:
        out.append("**Problemi distinti**: nessuna combinazione di valori isola il caso target in modo significativo.")
    win = R.get("05_finestre_degrado", pd.DataFrame())
    shifts = m.get("level_shifts", [])
    if m.get("ts_source") and not m.get("ts_only_time"):
        parts = []
        if not win.empty:
            parts.append(f"{plural(len(win), 'finestra di degrado contiene', 'finestre di degrado contengono')} il {m.get('degrado_share', 0):.1f}% dei casi "
                         f"(ambito: {', '.join(win['AMBITO'].str.split(' (', regex=False).str[0].unique()[:3])})")
        for s in shifts[:2]:
            parts.append(f"dal {s['DATA_CAMBIO']:%Y-%m-%d} il tasso {'sale' if s['DIREZIONE'] == 'AUMENTO' else 'scende'} dal {s['TASSO_PRIMA_PCT']:.2f}% al {s['TASSO_DOPO_PCT']:.2f}%")
        out.append("**Nel tempo**: " + ("; ".join(parts) if parts else "fenomeno costante, senza finestre di degrado né cambi di livello") + ".")
    auc = m.get("xgb_auc")
    if auc is not None:
        verdict = ("le colonne disponibili separano molto bene i casi" if auc >= 0.9 else "le colonne disponibili separano discretamente i casi" if auc >= 0.7
                   else "le colonne disponibili separano poco i casi: la causa è probabilmente in dati non presenti nel tracciato")
        out.append(f"**Quanto spiegano i dati**: AUC {m.get('xgb_auc_kind')} {auc:.2f}, {verdict}.")
    return out


def build_executive_summary(ctx: AnalysisContext) -> str:
    m, R = ctx.meta, ctx.results
    period = m.get("period")
    L: List[str] = []
    L.append(f"# Executive Summary - Analisi {ctx.target_col} = {ctx.target_value}\n")
    L.append(f"File analizzato: `{Path(m.get('input_file', '')).name}`  ")
    if period:
        L.append(f"Periodo: {period[0]:%Y-%m-%d %H:%M} -> {period[1]:%Y-%m-%d %H:%M}  ")
    L.append("")
    brief = _in_brief(ctx)
    if brief:
        L.append("## In breve")
        L += [f"- {b}" for b in brief]
        L.append("")
    L.append(f"## 1. Perimetro {ctx.event_label}")
    L.append(f"Righe totali: **{ctx.n_rows:,}**. Righe con `{ctx.target_col}` = `{ctx.target_value}`: **{ctx.n_errors:,}** (**{ctx.error_rate:.2%}**).")
    tv = ctx.df[ctx.target_col].value_counts(dropna=False).head(8)
    L.append("Distribuzione del target: " + ", ".join(f"{k}={v:,}" for k, v in tv.items()) + ".")
    if "ERROR" not in ctx.target_values and "ERR" not in ctx.target_value and "KO" not in ctx.target_values:
        L.append(f"\n_Nota terminologica: nelle tabelle e nei testi 'errori', 'ERROR' e 'tasso di errore' indicano le righe del caso analizzato ({ctx.event_label}) e la loro incidenza._")
    bullets = m.get("behaviour_bullets", [])
    if bullets:
        L.append(f"\nLe righe {ctx.event_label} risultano prevalentemente associate a:")
        L += [f"- {b}" for b in bullets]
    L.append("\n## 2. Numero di righe coinvolte")
    L.append(f"| Totale | {ctx.target_value} | % | Cluster | Anomalie estreme | Alta criticità |\n|---|---|---|---|---|---|")
    an = R.get("08_anomalie_sintesi", pd.DataFrame())
    n_ext = int(an.loc[an["VISTA"].str.startswith("Consenso >= 2"), "N"].iloc[0]) if not an.empty else 0
    n_hr = int(an.loc[an["VISTA"] == HIGH_PRIORITY_VIEW, "N"].iloc[0]) if not an.empty else 0
    L.append(f"| {ctx.n_rows:,} | {ctx.n_errors:,} | {ctx.error_rate:.2%} | {m.get('cluster_k', 'n/d')} | {n_ext} | {n_hr} |")
    # dimensioni descrittive, dalla più associata al caso analizzato: le prime tre, poi le successive
    dims = [d for d in key_dimensions(ctx) if f"03_err_{d}" in R]
    L.append("\n## 3. Dimensioni maggiormente coinvolte")
    L += [f"{d}: {_top_list(R[f'03_err_{d}'])}  " for d in dims[:3]] or ["n/d"]
    L.append("\n## 4. Altre dimensioni coinvolte")
    L += [f"{d}: {_top_list(R[f'03_err_{d}'], 2)}  " for d in dims[3:8]] or ["n/d"]
    L.append("\n## 5. Famiglie di valori, campi mancanti e altri sistemi")
    fam = R.get("06_famiglie_valori", pd.DataFrame())
    fam = fam[(fam["LIFT"] >= 2) & (fam["N_ERROR"] >= 10)] if not fam.empty else fam
    L.append("Valori e famiglie sovra-rappresentati: " + ("; ".join(
        f"{r.DIMENSIONE}={r.FAMIGLIA} (tasso {r.TASSO_ERRORE_PCT}%, lift {r.LIFT})" for r in fam.head(4).itertuples()) or "nessuno") + "  ")
    pres = R.get("06_presenza_campi", pd.DataFrame())
    if not pres.empty:      # senza le colonne che sono conseguenza del caso (vuote proprio perché il caso si è verificato)
        pres = pres[(pres["ODDS_RATIO_ASSENZA"] >= 2) & (pres["PCT_ASSENTE_TRA_ERRORI"] >= 20) & ~pres["CAMPO"].isin(m.get("leakage_cols", []))]
    L.append("Campi mancanti associati al caso: " + ("; ".join(
        f"{r.CAMPO} (assente nel {r.PCT_ASSENTE_TRA_ERRORI}% dei casi, {r.PCT_ASSENTE_TRA_NON_ERRORI}% nel resto)" for r in pres.head(4).itertuples()) or "nessuno") + "  ")
    L.append("Altri sistemi: " + next((f.titolo for f in ctx.findings if f.categoria == "INTEGRAZIONE"), "nessun altro sistema riconosciuto nel tracciato"))
    L.append("\n## 6. Principali pattern")
    rules = R.get("04_association_rules", pd.DataFrame())
    if not rules.empty:
        for r in rules.head(6).itertuples():
            L.append(f"- **{r.PATTERN}** -> {r.CONFIDENZA_PCT:.1f}% delle righe con questo pattern sono {ctx.event_label}; spiega il {r.COVERAGE_ERRORI_PCT:.1f}% dei casi (lift {r.LIFT}, n={r.N_RIGHE:,})")
    else:
        L.append("Nessun pattern significativo con le soglie correnti (ridurre --min-support / --min-confidence).")
    cover = R.get("04_copertura_cumulativa", pd.DataFrame())
    if not cover.empty:
        L.append(f"\nProblemi distinti in ordine di peso (ogni pattern conta solo i casi {ctx.event_label} non già spiegati dai precedenti):")
        for r in cover.itertuples():
            if r.PATTERN.startswith("(nessun pattern"):
                L.append(f"- _Fuori da questi pattern_: {r.N_ERRORI_NUOVI:,} casi ({r.QUOTA_ERRORI_PCT}%), tasso residuo {r.TASSO_NELLE_NUOVE_PCT}% contro {ctx.error_rate:.2%} globale")
            else:
                L.append(f"- {r.PASSO}. **{r.PATTERN}**: {r.N_ERRORI_NUOVI:,} casi ({r.QUOTA_ERRORI_PCT}%), tasso {r.TASSO_NELLE_NUOVE_PCT}%; cumulato {r.COPERTURA_CUMULATA_PCT}%")
    tree = R.get("04_albero_regole", pd.DataFrame())
    if not tree.empty:
        t = tree[tree["N_ERRORI"] >= 10].head(3)
        if not t.empty:
            L.append("\nRegole dall'albero decisionale:")
            L += [f"- {r.REGOLA} -> tasso {r.TASSO_ERRORE_PCT}% su {r.N_RIGHE:,} righe (coverage {r.COVERAGE_ERRORI_PCT}%)" for r in t.itertuples()]
    L.append("\n## 7. Cause probabili")
    hyp = R.get("09_cause_probabili", pd.DataFrame())
    if m.get("hypotheses_note"):
        L.append(f"_{m['hypotheses_note']}_")
    if not hyp.empty:
        L.append("| Cod. | Tipo | Causa probabile | Score | Confidenza |\n|---|---|---|---|---|")
        for r in hyp.itertuples():
            if r.SCORE > 0:
                L.append(f"| {r.CODICE} | {r.TIPO} | {r.CAUSA_PROBABILE} | {r.SCORE} | {r.CONFIDENZA} |")
        for r in hyp.head(3).itertuples():
            if r.SCORE > 0:
                L.append(f"\n**{r.CODICE} - {r.CAUSA_PROBABILE}**  \nEvidenze: {r.EVIDENZE}")
    L.append("\n## 8. Variabili discriminanti")
    corr = R.get("04_correlazioni", pd.DataFrame())
    if not corr.empty:
        L.append("Per Cramér's V: " + ", ".join(f"{r.COLONNA} ({r.CRAMERS_V})" for r in corr.head(6).itertuples()))
    agg = R.get("04_importance_per_colonna", pd.DataFrame())
    if not agg.empty:
        auc = f"AUC {m.get('xgb_auc_kind')} {m['xgb_auc']:.3f}" if m.get("xgb_auc") is not None else "AUC n/d"
        L.append(f"  \nPer importanza XGBoost/SHAP ({auc}): " + ", ".join(f"{r.COLONNA} ({r.QUOTA_PCT}%)" for r in agg.head(6).itertuples()))
    cl = R.get("07_cluster", pd.DataFrame())
    if not cl.empty:
        L.append("\nCluster di errore:")
        L += [f"- **C{r.CLUSTER}** {r.ETICHETTA}: {r.DIMENSIONE:,} errori ({r.PCT_ERRORI}%). Discriminanti: {r.CAMPI_DISCRIMINANTI}" for r in cl.itertuples()]
    L.append("\n## 9. Possibili anomalie tecniche")
    tech = [f for f in ctx.findings if f.categoria in ("BURST", "FINESTRE DI DEGRADO", "CAMBIO DI LIVELLO", "ALIMENTAZIONE DATI", "INTEGRAZIONE", "CAUSA TECNICA",
                                                       "OSSERVABILITA'", "TASSO ANOMALO", "PATTERN DETERMINISTICO", "ERRORI AD ALTA CRITICITA'", "QUALITA' DATI")]
    L += [f"- [{f.confidenza}] {f.titolo}. _{f.evidenza}_" for f in tech] or ["- Nessuna anomalia tecnica rilevata dalle regole automatiche."]
    L.append("\n## 10. Raccomandazioni operative")
    recs, seen = [], set()
    for r in hyp.itertuples() if not hyp.empty else []:
        if r.SCORE >= 0.4 and r.RACCOMANDAZIONE not in seen:
            recs.append(f"{r.CODICE}: {r.RACCOMANDAZIONE}")
            seen.add(r.RACCOMANDAZIONE)
    for f in ctx.findings:
        if f.raccomandazione and f.raccomandazione not in seen and f.confidenza == "ALTA":
            recs.append(f.raccomandazione)
            seen.add(f.raccomandazione)
    L += [f"{i}. {r}" for i, r in enumerate(recs[:12], 1)] or ["1. Nessuna raccomandazione automatica."]
    L.append("\n---\n_Report generato automaticamente da cercaPattern. Le cause sono ipotesi statistiche da validare con i log applicativi._")
    text = "\n".join(L)
    ctx.meta["executive_summary_text"] = text
    return text


# ---------------------------------------------------------------------------
# Scrittura file
# ---------------------------------------------------------------------------
def _safe_sheet(name: str, used: set) -> str:
    s = re.sub(r"[\[\]\:\*\?\/\\]", "_", name)[:31]
    base, i = s, 1
    while s in used:
        s = f"{base[:28]}_{i}"
        i += 1
    used.add(s)
    return s


def _excel_value(v):
    """Valore scrivibile in una cella: tipi nativi, niente NaN/inf, niente caratteri di controllo
    (openpyxl li rifiuta e farebbe fallire l'intero file), testo entro il limite di Excel."""
    if v is None:
        return None
    if isinstance(v, str):
        return _ILLEGAL_XLSX.sub("", v)[:32_000]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return None if f != f else (f if abs(f) != float("inf") else ("inf" if f > 0 else "-inf"))
    if v is pd.NaT or v is pd.NA:
        return None
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime().replace(tzinfo=None)
    if isinstance(v, (pd.Timedelta, set, frozenset, list, dict, tuple)):
        return str(v)
    return _ILLEGAL_XLSX.sub("", str(v))[:32_000]


_ILLEGAL_XLSX = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def write_excel(ctx: AnalysisContext, max_rows: int = 200_000, row_level_max: int = EXCEL_ROW_LEVEL_MAX) -> Path:
    """Scrive il report multi-tab. Le celle sono scritte direttamente con openpyxl (alcune volte più
    veloce di ``DataFrame.to_excel``); un testo del CSV che inizia per '=' resta testo, non formula."""
    from openpyxl import Workbook
    from openpyxl.cell.cell import Cell
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    p = ctx.out_dir / f"{ctx.file_prefix}_report.xlsx"
    used: set = set()
    wb = Workbook()
    wb.remove(wb.active)
    head_font, head_fill, head_align = Font(bold=True, color="FFFFFF"), PatternFill("solid", fgColor="2A78D6"), Alignment(vertical="center")

    def add_sheet(name: str, df: pd.DataFrame) -> None:
        ws = wb.create_sheet(_safe_sheet(name, used))
        cols = [str(c) for c in df.columns]
        if not cols:
            return
        ws.append([_excel_value(c) for c in cols])
        widths = [len(c) for c in cols]
        for i, row in enumerate(df.itertuples(index=False, name=None)):
            vals = [_excel_value(v) for v in row]
            for j, v in enumerate(vals):
                if isinstance(v, str):
                    if v.startswith("="):          # dato del CSV, non una formula
                        cell = Cell(ws, value=v)
                        cell.data_type = "s"
                        vals[j] = cell
                    if i < 200 and len(v) > widths[j]:
                        widths[j] = len(v)
                elif i < 200 and v is not None and len(str(v)) > widths[j]:
                    widths[j] = len(str(v))
            ws.append(vals)
        ws.freeze_panes = "A2"
        for cell in ws[1]:
            cell.font, cell.fill, cell.alignment = head_font, head_fill, head_align
        for j, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(j)].width = min(max(10, w + 2), 80)

    # executive summary come testo
    add_sheet("00_Executive_Summary", pd.DataFrame({"EXECUTIVE_SUMMARY": ctx.meta.get("executive_summary_text", "").splitlines()}))
    add_sheet("00_Findings", pd.DataFrame([f.as_row() for f in ctx.findings]))
    for name, df in ctx.results.items():
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            continue
        cap = row_level_max if name in EXCEL_ROW_LEVEL else max_rows
        if len(df) > cap:
            LOG.info("Excel: tabella %s limitata alle prime %s righe su %s (il dettaglio completo è nei CSV)", name, f"{cap:,}", f"{len(df):,}")
        add_sheet(name, df.head(cap))
    wb.save(p)
    LOG.info("Excel scritto: %s", p)
    return p


def _md_to_html(md: str) -> str:
    """Conversione minimale Markdown -> HTML (titoli, liste, tabelle, bold/italic/code)."""
    out, in_ul, in_ol, in_tbl = [], False, False, False
    inline = lambda s: re.sub(r"`([^`]+)`", r"<code>\1</code>", re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"<em>\1</em>", re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", html.escape(s))))
    for line in md.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r"-+", c) for c in cells):
                continue
            if not in_tbl:
                out.append("<table class='md'>")
                in_tbl = True
                out.append("<tr>" + "".join(f"<th>{inline(c)}</th>" for c in cells) + "</tr>")
            else:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
            continue
        if in_tbl:
            out.append("</table>")
            in_tbl = False
        m_ul, m_ol = re.match(r"^- (.*)", line), re.match(r"^\d+\. (.*)", line)
        if m_ul and not in_ul:
            out.append("<ul>"); in_ul = True
        if not m_ul and in_ul:
            out.append("</ul>"); in_ul = False
        if m_ol and not in_ol:
            out.append("<ol>"); in_ol = True
        if not m_ol and in_ol:
            out.append("</ol>"); in_ol = False
        if m_ul:
            out.append(f"<li>{inline(m_ul.group(1))}</li>")
        elif m_ol:
            out.append(f"<li>{inline(m_ol.group(1))}</li>")
        elif line.startswith("# "):
            out.append(f"<h1>{inline(line[2:])}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.strip() == "---":
            out.append("<hr>")
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
    for flag, tag in ((in_ul, "ul"), (in_ol, "ol"), (in_tbl, "table")):
        if flag:
            out.append(f"</{tag}>")
    return "\n".join(out)


HTML_CSS = """
:root{--bg:#fcfcfb;--fg:#0b0b0b;--muted:#52514e;--line:#e6e5e1;--accent:#2a78d6;--card:#ffffff}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#1a1a19;--fg:#f0efec;--muted:#c3c2b7;--line:#3a3a38;--accent:#3987e5;--card:#232322}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:24px 16px}nav{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 16px;z-index:5;display:flex;gap:14px;flex-wrap:wrap;font-size:13px}
nav a{color:var(--accent);text-decoration:none}h1{font-size:26px;margin:.2em 0}h2{font-size:20px;margin:1.6em 0 .4em;border-bottom:1px solid var(--line);padding-bottom:4px}h3{font-size:16px;margin:1.2em 0 .3em;color:var(--muted)}
.kpis{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}.kpi{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 16px;min-width:140px}.kpi b{display:block;font-size:22px}.kpi span{font-size:12px;color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin:8px 0 16px;display:block;overflow-x:auto;white-space:nowrap}th,td{border:1px solid var(--line);padding:4px 8px;text-align:left}th{background:var(--card);position:sticky;top:0}
table.md{display:table;white-space:normal}details{margin:6px 0}summary{cursor:pointer;color:var(--accent)}code{background:var(--card);padding:1px 4px;border-radius:3px}
.finding{border-left:3px solid var(--accent);padding:4px 10px;margin:6px 0;background:var(--card)}.finding .tag{font-size:11px;color:var(--muted)}
.plot{background:#fff;border:1px solid var(--line);border-radius:8px;padding:6px;margin:10px 0}
"""


def write_html(ctx: AnalysisContext, plotly_cdn: bool = True, max_rows: int = 60) -> Path:
    p = ctx.out_dir / f"{ctx.file_prefix}_report.html"
    first = True
    parts: List[str] = []

    def fig_html(key: str) -> str:
        nonlocal first
        f = ctx.figures.get(key)
        if f is None:
            return ""
        inc = ("cdn" if plotly_cdn else True) if first else False
        first = False
        return f"<div class='plot'>{f.to_html(full_html=False, include_plotlyjs=inc, config={'displaylogo': False, 'responsive': True})}</div>"

    def table(name: str, title: Optional[str] = None, n: int = max_rows, open_: bool = False) -> str:
        df = ctx.results.get(name)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return ""
        t = df.head(n).to_html(index=False, border=0, na_rep="", float_format=lambda x: f"{x:,.4g}", escape=True)
        return f"<details{' open' if open_ else ''}><summary>{html.escape(title or name)} ({len(df):,} righe, mostrate {min(n, len(df))})</summary>{t}</details>"

    sections = [("exec", "Executive summary"), ("f1", "1. Profiling"), ("f2", "2. Dizionario"), ("f3", "3. Tassi di errore"),
                ("f4", "4. Pattern"), ("f5", "5. Temporale"), ("f6", "6. Funzionale"), ("f7", "7. Cluster"), ("f8", "8. Anomalie"), ("f9", "9. Findings")]
    nav = "<nav>" + "".join(f"<a href='#{a}'>{b}</a>" for a, b in sections) + "</nav>"
    parts.append(f"<section id='exec'>{_md_to_html(ctx.meta.get('executive_summary_text', ''))}</section>")
    m = ctx.meta
    ev, eng = html.escape(str(ctx.event_label)), html.escape(str(ctx.system_label))   # nomi e valori vengono dal CSV
    kpis = f"""<div class='kpis'><div class='kpi'><b>{ctx.n_rows:,}</b><span>righe</span></div><div class='kpi'><b>{ctx.n_errors:,}</b><span>{ev}</span></div>
    <div class='kpi'><b>{ctx.error_rate:.2%}</b><span>incidenza</span></div><div class='kpi'><b>{m.get('cluster_k','n/d')}</b><span>cluster</span></div>
    <div class='kpi'><b>{len(ctx.findings)}</b><span>findings</span></div></div>"""
    parts.append(f"<section id='f1'><h2>Fase 1 - Data profiling</h2>{kpis}{table('01_profiling', 'Profilo campi', 500, True)}{table('01_distribuzioni', 'Distribuzioni (top valori)', 400)}{table('00_mappa_colonne', 'Ruoli e sistemi riconosciuti')}</section>")
    parts.append(f"<section id='f2'><h2>Fase 2 - Dizionario dati</h2>{table('02_dizionario', 'Dizionario', 500, True)}</section>")
    s3 = "<section id='f3'><h2>Fase 3 - Analisi errori per dimensione</h2>" + table("03_ranking_dimensioni", "Ranking dimensioni (Cramér's V)", 60, True)
    for dim in m.get("dims", []):
        s3 += fig_html(f"03_{dim}") + table(f"03_err_{dim}", f"Tasso di errore per {dim}", 40)
    s3 += table("03_profilo_error_vs_ok", f"Profilo {ctx.target_value} vs resto", 200) + "</section>"
    parts.append(s3)
    tree_txt = html.escape(m.get("tree_text", ""))
    parts.append(f"<section id='f4'><h2>Fase 4 - Pattern discovery</h2>{table('04_copertura_cumulativa', 'Problemi distinti in ordine di peso (copertura cumulativa)', 20, True)}"
                 f"{fig_html('04_pattern_scatter')}{table('04_association_rules', f'Pattern associati a {ctx.target_item} (conteggi esatti su tutte le righe)', 60, True)}"
                 f"{table('04_correlazioni', 'Correlazioni (Cramér V / MI)', 60)}{table('04_albero_regole', 'Regole albero decisionale', 60)}"
                 f"<details><summary>Albero decisionale (testo)</summary><pre style='font-size:11px;overflow:auto'>{tree_txt}</pre></details>"
                 f"{fig_html('04_importance')}{table('04_feature_importance', 'Feature importance (one-hot)', 60)}</section>")
    parts.append(f"<section id='f5'><h2>Fase 5 - Analisi temporale</h2>{fig_html('05_heatmap')}{fig_html('05_serie_errori')}{fig_html('05_serie_tasso')}{fig_html('05_serie_oraria')}"
                 f"{table('05_finestre_degrado', 'Finestre di degrado', 50, True)}{table('05_cambi_livello', 'Cambi di livello del tasso giornaliero', 10, True)}"
                 f"{table('05_tasso_per_ora', 'Tasso per ora', 24)}{table('05_tasso_per_giorno_sett', 'Tasso per giorno settimana', 7)}"
                 f"{table('05_serie_giornaliera', 'Serie giornaliera', 400)}</section>")
    bl = "".join(f"<li>{html.escape(b)}</li>" for b in m.get("behaviour_bullets", []))
    parts.append(f"<section id='f6'><h2>Fase 6 - Analisi funzionale {eng}</h2><p>Le righe {ev} risultano prevalentemente associate a:</p><ul>{bl or '<li>n/d</li>'}</ul>"
                 f"{table('06_presenza_campi', 'Campi assenti e caso analizzato', 40, True)}{table('06_relazione_sistemi', 'Relazione con gli altri sistemi', 40, True)}"
                 f"{table('06_detail_classificazione', f'Cause tecniche da {ctx.detail_col}', 20, True)}{table('06_detail_valori', f'Valori {ctx.detail_col}', 50)}"
                 f"{table('06_famiglie_valori', 'Famiglie di valori', 40)}{table('06_passaggi_facoltativi', 'Passaggi facoltativi', 40)}</section>")
    parts.append(f"<section id='f7'><h2>Fase 7 - Clustering errori</h2>{fig_html('07_cluster')}{table('07_cluster', 'Cluster', 20, True)}{table('07_cluster_assegnazioni', 'Assegnazioni (estratto)', 40)}</section>")
    parts.append(f"<section id='f8'><h2>Fase 8 - Anomaly detection</h2>{fig_html('08_pca')}{table('08_anomalie_sintesi', 'Sintesi', 10, True)}{table('08_anomalie_top', 'Top anomalie', 60)}</section>")
    fh = "".join(f"<div class='finding'><span class='tag'>{html.escape(f.fase)} · {html.escape(f.categoria)} · confidenza {f.confidenza}</span><br><strong>{html.escape(f.titolo)}</strong><br><small>{html.escape(f.evidenza)}</small>"
                 + (f"<br><em>{html.escape(f.raccomandazione)}</em>" if f.raccomandazione else "") + "</div>" for f in ctx.findings)
    parts.append(f"<section id='f9'><h2>Fase 9 - Findings e cause probabili</h2>{table('09_cause_probabili', 'Cause probabili', 10, True)}{fh}</section>")
    doc = f"<!doctype html><html lang='it'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{ev} Report</title><style>{HTML_CSS}</style></head><body>{nav}<div class='wrap'>{''.join(parts)}</div></body></html>"
    p.write_text(doc, encoding="utf-8")
    ctx.meta["report_html_path"] = str(p)
    LOG.info("HTML scritto: %s (%.1f MB)", p, p.stat().st_size / 1e6)
    return p


def write_csvs(ctx: AnalysisContext) -> None:
    o = ctx.out_dir
    pf = ctx.file_prefix

    def save(df: pd.DataFrame, name: str) -> None:
        # UTF-8 con BOM: Excel riconosce l'encoding e mostra correttamente le lettere accentate
        df.to_csv(o / f"{pf}_{name}.csv", index=False, sep=";", encoding="utf-8-sig")

    save(ctx.errors, "righe_target")
    pats = []
    r = ctx.results.get("04_association_rules", pd.DataFrame())
    if not r.empty:
        pats.append(r.assign(METODO="association_rules"))
    cov = ctx.results.get("04_copertura_cumulativa", pd.DataFrame())
    if not cov.empty:
        pats.append(cov.rename(columns={"N_RIGHE_NUOVE": "N_RIGHE", "N_ERRORI_NUOVI": "N_ERRORI", "TASSO_NELLE_NUOVE_PCT": "CONFIDENZA_PCT",
                                        "QUOTA_ERRORI_PCT": "COVERAGE_ERRORI_PCT"}).assign(METODO="copertura_cumulativa"))
    t = ctx.results.get("04_albero_regole", pd.DataFrame())
    if not t.empty:
        pats.append(t.rename(columns={"REGOLA": "PATTERN", "TASSO_ERRORE_PCT": "CONFIDENZA_PCT"}).assign(METODO="decision_tree"))
    save(pd.concat(pats, ignore_index=True) if pats else pd.DataFrame(), "patterns")
    save(ctx.results.get("07_cluster_assegnazioni", pd.DataFrame()), "clusters")
    save(ctx.results.get("07_cluster", pd.DataFrame()), "clusters_summary")
    save(ctx.results.get("08_anomalie_tutte", pd.DataFrame()), "anomalies")
    save(pd.DataFrame([f.as_row() for f in ctx.findings]), "findings")
    save(ctx.results.get("09_cause_probabili", pd.DataFrame()), "cause_probabili")
    save(ctx.results.get("02_dizionario", pd.DataFrame()), "dizionario_dati")
    LOG.info("CSV scritti in %s", o)


def write_all_reports(ctx: AnalysisContext, plotly_cdn: bool = True) -> None:
    png_fns = (png_error_rate_dims, png_patterns, png_importance, png_temporal, png_functional, png_clusters_anomalies)
    ctx.progress.set_total(1 + len(png_fns) + 4)
    ctx.progress.note("executive summary")
    build_executive_summary(ctx)
    summary_path = ctx.out_dir / f"{ctx.file_prefix}_executive_summary.md"
    summary_path.write_text(ctx.meta["executive_summary_text"], encoding="utf-8")
    ctx.meta["executive_summary_path"] = str(summary_path)
    ctx.progress.step("grafici PNG")
    for fn in png_fns:
        try:
            fn(ctx)
        except Exception as e:
            LOG.exception("PNG %s fallito: %s", fn.__name__, e)
        ctx.progress.step(fn.__name__.replace("png_", "PNG "))
    ctx.progress.note("figure plotly")
    try:
        build_plotly_figures(ctx)
    except Exception as e:
        LOG.exception("Figure plotly fallite: %s", e)
    ctx.progress.step("CSV")
    write_csvs(ctx)
    ctx.progress.step("Excel")
    try:
        write_excel(ctx)
    except Exception as e:
        LOG.exception("Excel fallito: %s", e)
    ctx.progress.step("HTML")
    try:
        write_html(ctx, plotly_cdn=plotly_cdn)
    except Exception as e:
        LOG.exception("HTML fallito: %s", e)
    LOG.info("Grafici PNG: %d", len(ctx.png_paths))
