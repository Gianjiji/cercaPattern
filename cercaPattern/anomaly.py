"""
FASE 8 - Anomaly detection sulle sole righe del caso target.

Tre algoritmi complementari sulla stessa matrice usata per il clustering:
- Isolation Forest (punti isolati nello spazio delle feature);
- Local Outlier Factor (densità locale);
- DBSCAN (punti rumore, eps stimato dalla curva delle k-distanze).

Gli algoritmi lavorano sui *profili* distinti delle righe target (vedi clustering): migliaia di
righe identiche non sono mille punti da confrontare tra loro ma un profilo molto frequente, che
per definizione non è anomalo. Oltre ``max_profiles`` profili LOF e DBSCAN vengono stimati su
un campione e applicati agli altri, così il costo non cresce col quadrato delle righe.

Si calcola un punteggio di consenso (quanti algoritmi concordano) e si
aggiungono due viste di priorità:
- ERRORE_UNICO: combinazione di valori delle dimensioni principali che compare una sola volta;
- ALTA_CRITICITA: riga target che gli altri sistemi segnalano con un esito critico, o con una
  misura molto alta (il caso analizzato si è verificato proprio dove contava di più).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AnalysisContext, CRITICAL_RX, Finding, LOG, contains_token, key_dimensions  # noqa: F401 (CRITICAL_RX riesportata)
from .clustering import error_profiles

SIGNATURE_DIMS = 8        # dimensioni che formano la "firma" di una riga target
HIGH_PRIORITY_VIEW = "Errori ad alta criticità (altri sistemi / misura)"


def _top_share(score: np.ndarray, counts: np.ndarray, share: float, eligible: np.ndarray = None) -> np.ndarray:
    """Profili con il punteggio più alto fino a coprire ``share`` delle righe (ogni profilo pesa
    per le sue righe): è la "contaminazione" degli algoritmi, misurata sulle righe e non sui profili."""
    flag = np.zeros(len(score), dtype=bool)
    idx = np.flatnonzero(eligible) if eligible is not None else np.arange(len(score))
    if len(idx) == 0:
        return flag
    order = idx[np.argsort(-score[idx], kind="stable")]
    cum = np.cumsum(counts[order])
    k = int(np.searchsorted(cum, share * counts.sum(), side="right"))
    flag[order[:k]] = True
    return flag


def run_anomaly(ctx: AnalysisContext, contamination: float = 0.05, top_n: int = 200, max_profiles: int = 20_000) -> pd.DataFrame:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
    from sklearn.cluster import DBSCAN

    n = ctx.n_errors
    if n < 20:
        LOG.warning("Troppo pochi errori (%d) per l'anomaly detection", n)
        return pd.DataFrame()
    prof = error_profiles(ctx)
    X, inverse, counts = prof.get("Xr", prof["X"]), prof["inverse"], prof["counts"]
    if X.shape[1] == 0:
        LOG.warning("Nessuna feature utilizzabile per l'anomaly detection")
        return pd.DataFrame()
    m = len(counts)
    err_mask = ctx.is_error.to_numpy()
    err = ctx.errors
    rs = np.random.RandomState(ctx.random_state)

    # Isolation Forest: addestrato sulle righe (un profilo frequente pesa di più), valutato sui profili
    fit_rows = inverse if n <= 50_000 else rs.choice(m, size=50_000, replace=True, p=counts / counts.sum())
    iso = IsolationForest(n_estimators=300, random_state=ctx.random_state, n_jobs=4).fit(X[fit_rows])
    if_score = -iso.score_samples(X)                 # più alto = più anomalo
    if_flag = _top_share(if_score, counts, contamination)

    # LOF e DBSCAN sui profili distinti (un campione pesato se sono troppi)
    sub = np.arange(m) if m <= max_profiles else np.sort(rs.choice(m, size=max_profiles, replace=False, p=counts / counts.sum()))
    rest = np.setdiff1d(np.arange(m), sub, assume_unique=True)
    Xs = X[sub]
    k = int(min(20, max(5, len(sub) // 50)))
    lof_score = np.ones(m)
    lof_flag = np.zeros(m, dtype=bool)
    db_label = np.full(m, -1, dtype=np.int64)
    db_flag = np.zeros(m, dtype=bool)
    eps = float("nan")
    if len(sub) > k + 1:
        lof = LocalOutlierFactor(n_neighbors=k, novelty=len(rest) > 0)
        if len(rest):
            lof.fit(Xs)
            lof_score[rest] = -lof.score_samples(X[rest])
        else:
            lof.fit_predict(Xs)
        lof_score[sub] = -lof.negative_outlier_factor_
        lof_score = np.nan_to_num(lof_score, nan=1.0, posinf=1e6)
        # un profilo con più di k righe identiche sta in una zona densa per definizione
        lof_flag = _top_share(lof_score, counts, contamination, eligible=counts <= k)
        # DBSCAN: eps = percentile della k-distanza
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(sub))).fit(Xs)
        dist, _ = nn.kneighbors(Xs)
        eps = float(np.percentile(dist[:, -1], 90)) or 0.5
        scale = counts.sum() / counts[sub].sum()      # il campione rappresenta tutte le righe
        db = DBSCAN(eps=eps, min_samples=max(3, k // 2)).fit(Xs, sample_weight=counts[sub] * scale)
        db_label[sub] = db.labels_
        if len(rest):                                 # gli altri profili: rumore se lontani da ogni punto "core"
            core = db.core_sample_indices_
            if len(core):
                d, j = NearestNeighbors(n_neighbors=1).fit(Xs[core]).kneighbors(X[rest])
                db_label[rest] = np.where(d[:, 0] <= eps, db.labels_[core][j[:, 0]], -1)
        db_flag = db_label == -1
    else:
        LOG.info("Solo %d profili distinti tra le righe target: LOF e DBSCAN non applicabili", m)

    out = pd.DataFrame(index=err.index)
    out["IF_SCORE"] = if_score[inverse]
    out["IF_FLAG"] = if_flag[inverse]
    out["LOF_FLAG"] = lof_flag[inverse]
    out["LOF_SCORE"] = lof_score[inverse]
    out["DBSCAN_CLUSTER"] = db_label[inverse]
    out["DBSCAN_FLAG"] = db_flag[inverse]
    out["CONSENSO"] = out[["IF_FLAG", "LOF_FLAG", "DBSCAN_FLAG"]].sum(axis=1)

    # errori unici (firma categorica che compare una sola volta): la firma è fatta dalle dimensioni
    # descrittive più associate al target
    model_cols = set(ctx.meta.get("model_columns", []))
    sig_cols = [c for c in key_dimensions(ctx) if c in model_cols][:SIGNATURE_DIMS]
    if sig_cols:
        sig = pd.util.hash_pandas_object(err[sig_cols].astype(object).fillna("(NULL)"), index=False)
        occ = sig.map(sig.value_counts()).to_numpy()
        out["FIRMA_N_OCCORRENZE"] = occ
        out["ERRORE_UNICO"] = occ == 1
    else:
        out["ERRORE_UNICO"] = False

    # alta criticità secondo gli altri sistemi / la misura principale
    critical_rx = ctx.critical_rx
    risk = np.zeros(n, dtype=np.int64)
    reasons = np.full(n, "", dtype=object)

    def add(hit: np.ndarray, why: str) -> None:
        nonlocal risk, reasons
        risk += hit.astype(np.int64)
        reasons = np.where(hit, reasons + why + "; ", reasons)

    for c in ctx.other_system_cols:    # il sistema analizzato non può "segnalare" i propri casi
        hit = contains_token(err[c], critical_rx).to_numpy()
        num = ctx.numeric(c)
        if num.notna().mean() > 0.8:                          # colonna di punteggio numerica: top 5%
            hit = hit | (num[err_mask] >= num.quantile(0.95)).to_numpy()
        add(hit, c)
    ra = ctx.decision_col
    strong = np.zeros(n, dtype=bool)
    if ra:
        strong = contains_token(err[ra], critical_rx).to_numpy()
        add(strong, ra)
    imp = ctx.measure_col
    if imp:
        v = ctx.numeric(imp)
        add((v[err_mask] >= v.quantile(0.99)).to_numpy(), f"{imp}>=p99")
    out["CRITICITA_SCORE"] = risk
    out["CRITICITA_MOTIVI"] = pd.Series(reasons, index=err.index).str.strip("; ")
    # alta criticità = decisione complessiva restrittiva (segnale forte) oppure almeno 2 segnali concordanti
    out["ALTA_CRITICITA"] = strong | (risk >= 2)

    clash = [c for c in err.columns if c in out.columns]
    full = pd.concat([out, err.rename(columns={c: f"{c}_ORIG" for c in clash})], axis=1)
    full = full.sort_values(["CONSENSO", "CRITICITA_SCORE", "IF_SCORE"], ascending=False)
    ctx.add_table("08_anomalie_tutte", full)
    ctx.add_table("08_anomalie_top", full.head(top_n))
    summary = pd.DataFrame([
        {"VISTA": "Isolation Forest", "N": int(out["IF_FLAG"].sum())},
        {"VISTA": "LOF", "N": int(out["LOF_FLAG"].sum())},
        {"VISTA": "DBSCAN (rumore)", "N": int(out["DBSCAN_FLAG"].sum())},
        {"VISTA": "Consenso >= 2 algoritmi", "N": int((out["CONSENSO"] >= 2).sum())},
        {"VISTA": "Consenso 3 algoritmi (punti estremi)", "N": int((out["CONSENSO"] == 3).sum())},
        {"VISTA": "Errori unici (firma singola)", "N": int(out["ERRORE_UNICO"].sum())},
        {"VISTA": HIGH_PRIORITY_VIEW, "N": int(out["ALTA_CRITICITA"].sum())},
    ])
    summary["PCT_ERRORI"] = (100 * summary["N"] / n).round(2)
    ctx.add_table("08_anomalie_sintesi", summary)
    n_hr = int(out["ALTA_CRITICITA"].sum())
    ctx.add_finding(Finding("FASE 8", "ANOMALIE", f"{int((out['CONSENSO']>=2).sum())} errori estremi (consenso >=2 algoritmi), {int(out['ERRORE_UNICO'].sum())} errori unici",
                            f"IF={int(out['IF_FLAG'].sum())}, LOF={int(out['LOF_FLAG'].sum())}, DBSCAN={int(out['DBSCAN_FLAG'].sum())}; eps DBSCAN={eps:.3f}; {m:,} profili distinti",
                            "MEDIA", "Gli errori estremi/unici vanno verificati manualmente: possono essere casi di test, flussi non censiti o richieste alterate."))
    if n_hr:
        what = f" o con {ctx.measure_col} >= p99" if ctx.measure_col else ""
        ctx.add_finding(Finding("FASE 8", "ERRORI AD ALTA CRITICITA'", f"{n_hr} righe {ctx.event_label} con esito critico negli altri sistemi{what}",
                                f"{100*n_hr/n:.1f}% degli errori; motivi più frequenti: " + "; ".join(f"[{k_}] x{v}" for k_, v in out.loc[out['ALTA_CRITICITA'], 'CRITICITA_MOTIVI'].value_counts().head(3).items()),
                                "ALTA", f"Priorità massima: su queste righe {ctx.system_label} non ha prodotto il proprio esito proprio dove gli altri segnali indicano una situazione critica."
                                if ctx.is_system_target else f"Righe {ctx.event_label} con segnali critici dagli altri sistemi o valori molto alti: da verificare per prime."))
    return full
