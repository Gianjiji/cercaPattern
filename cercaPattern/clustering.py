"""
FASE 7 - Clustering automatico delle righe del caso target.

Pipeline:
1. one-hot delle colonne categoriche (solo righe target), pesato per importanza della colonna;
2. riduzione dimensionale (TruncatedSVD) se le feature sono molte;
3. KMeans con k scelto via silhouette (k in 2..k_max);
4. caratterizzazione di ogni cluster: per ogni colonna, valore modale,
   quota nel cluster vs quota su tutti gli errori (lift) -> campi discriminanti;
5. etichetta automatica del cluster dai 3 campi più discriminanti.

Le righe target con gli stessi valori su tutte le colonne formano un unico *profilo*: i modelli
lavorano sui profili distinti pesati per il numero di righe. Il risultato è lo stesso che
lavorare riga per riga, ma tempo e memoria dipendono dai profili distinti e non dal numero di righe.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import AnalysisContext, Finding, LOG, categorical_columns
from .patterns import build_feature_frame, one_hot


def error_profiles(ctx: AnalysisContext) -> Dict[str, Any]:
    """Profili distinti delle righe target, calcolati una volta e condivisi tra clustering e anomalie.

    Chiavi: ``frame`` (frame categorico di tutte le righe target), ``X`` (matrice one-hot pesata, una
    riga per profilo), ``inverse`` (per ogni riga target l'indice del suo profilo), ``counts`` (righe
    per profilo), ``feat`` (nomi delle colonne di X)."""
    if "error_profiles" in ctx._cache:
        return ctx._cache["error_profiles"]
    cols = categorical_columns(ctx, exclude=[ctx.target_col])
    frame = build_feature_frame(ctx, cols)[ctx.is_error.to_numpy()]
    # dentro gli errori alcune colonne diventano costanti: non distinguono nulla
    frame = frame[[c for c in frame.columns if frame[c].nunique(dropna=False) > 1]]
    n = len(frame)
    if frame.shape[1] == 0:
        prof = {"frame": frame, "X": np.zeros((min(n, 1), 0), dtype=np.float32), "inverse": np.zeros(n, dtype=np.int64),
                "counts": np.array([n] if n else [], dtype=np.int64), "feat": []}
        ctx._cache["error_profiles"] = prof
        return prof
    key = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    inverse, _ = pd.factorize(key)                         # codici in ordine di prima comparsa
    _, first = np.unique(inverse, return_index=True)       # prima riga di ogni profilo
    counts = np.bincount(inverse)
    oh = one_hot(frame.iloc[first]).astype(np.float32)
    # pesi per colonna dall'importanza XGBoost/SHAP (se disponibile): il clustering avviene
    # nello spazio delle variabili che spiegano l'errore, non in quello delle più frequenti
    imp = ctx.results.get("04_importance_per_colonna")
    if imp is not None and not imp.empty:
        w = dict(zip(imp["COLONNA"], imp["QUOTA_PCT"]))
        weights = np.array([np.sqrt(max(w.get(c.split("=", 1)[0], 0.0), 1.0) / 100.0) for c in oh.columns], dtype=np.float32)
        oh = oh * weights
    # la misura principale NON entra nella matrice (dominerebbe le one-hot pesate): serve solo a descrivere i cluster
    prof = {"frame": frame, "X": oh.to_numpy(dtype=np.float32), "inverse": inverse, "counts": counts, "feat": list(oh.columns)}
    ctx._cache["error_profiles"] = prof
    return prof


def error_feature_matrix(ctx: AnalysisContext) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """Matrice numerica (una riga per riga target) + frame categorico usato per la descrizione."""
    prof = error_profiles(ctx)
    return prof["X"][prof["inverse"]] if prof["X"].shape[1] else np.zeros((len(prof["frame"]), 0), dtype=np.float32), prof["frame"], prof["feat"]


def run_clustering(ctx: AnalysisContext, k_max: int = 8) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics import silhouette_score

    n = ctx.n_errors
    if n < 10:
        LOG.warning("Troppo pochi errori (%d) per il clustering", n)
        return pd.DataFrame()
    prof = error_profiles(ctx)
    X, frame, inverse, counts = prof["X"], prof["frame"], prof["inverse"], prof["counts"]
    if X.shape[1] == 0:
        return pd.DataFrame()
    m = len(counts)                                   # profili distinti
    Xr = X
    n_comp = min(30, X.shape[1] - 1, m - 1)
    if X.shape[1] > 30 and n_comp >= 2:
        # componenti stimate come se ogni profilo comparisse `counts` volte
        svd = TruncatedSVD(n_components=n_comp, random_state=ctx.random_state).fit(X * np.sqrt(counts)[:, None].astype(np.float32))
        Xr = svd.transform(X)
    prof["Xr"] = Xr
    rs = np.random.RandomState(ctx.random_state)
    # silhouette su un campione di righe (ogni profilo pesa per le sue righe)
    sil_idx = inverse if n <= 5000 else rs.choice(m, size=5000, replace=True, p=counts / counts.sum())
    best_k, best_s, best_labels = None, -1.0, None
    for k in range(2, min(k_max, n // 5, m) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=ctx.random_state).fit(Xr, sample_weight=counts)
        if len(set(km.labels_[sil_idx])) < 2:
            continue
        s = silhouette_score(Xr[sil_idx], km.labels_[sil_idx])
        if s > best_s:
            best_k, best_s, best_labels = k, s, km.labels_
    if best_labels is None:
        best_k, best_labels, best_s = 1, np.zeros(m, dtype=int), np.nan
    LOG.info("Clustering errori: k=%s silhouette=%.3f (%s profili distinti su %s righe)", best_k, best_s, f"{m:,}", f"{n:,}")
    ctx.meta["cluster_k"], ctx.meta["cluster_silhouette"] = best_k, best_s
    labels = pd.Series(np.asarray(best_labels)[inverse], index=frame.index, name="CLUSTER")
    ctx.meta["cluster_labels"] = labels
    ctx.meta["cluster_X"] = Xr[inverse]

    # --- caratterizzazione --------------------------------------------------
    rows, assign_desc = [], {}
    glob = {c: frame[c].value_counts(normalize=True) for c in frame.columns}
    imp_col = ctx.measure_col
    measure = ctx.numeric(imp_col)[ctx.is_error.to_numpy()] if imp_col else None
    med_name = f"MEDIANA_{imp_col}" if imp_col else "MEDIANA_MISURA"
    for k in sorted(set(labels)):
        in_k = (labels == k).to_numpy()
        sub = frame[in_k]
        disc = []
        for c in frame.columns:
            vc = sub[c].value_counts(normalize=True)
            if vc.empty:
                continue
            val, share = vc.index[0], float(vc.iloc[0])
            g = float(glob[c].get(val, 0.0))
            lift = share / g if g > 0 else np.inf
            # un campo è discriminante se è molto omogeneo nel cluster e sovra-rappresentato
            disc.append((c, val, share, g, lift, share * (share - g)))
        disc.sort(key=lambda t: t[5], reverse=True)
        top = [t for t in disc[:3] if t[2] >= 0.5 and t[4] >= 1.2]
        label_txt = " | ".join(f"{c}={v}" for c, v, *_ in top) or "cluster eterogeneo"
        size = int(in_k.sum())
        imp_med = float(measure[in_k].median()) if measure is not None else np.nan
        rows.append({"CLUSTER": k, "ETICHETTA": f"Errore su {label_txt}", "DIMENSIONE": size,
                     "PCT_ERRORI": round(100 * size / n, 2), med_name: round(imp_med, 2) if not np.isnan(imp_med) else None,
                     "CAMPI_DISCRIMINANTI": "; ".join(f"{c}={v} ({100*s:.0f}% nel cluster vs {100*g:.0f}% globale, lift {l:.1f})" for c, v, s, g, l, _ in disc[:5]),
                     "CARATTERISTICHE": "; ".join(f"{c}={sub[c].value_counts().index[0]}" for c in frame.columns[:12] if not sub[c].empty)})
        assign_desc[k] = f"Errore su {label_txt}"
    clusters = pd.DataFrame(rows).sort_values("DIMENSIONE", ascending=False).reset_index(drop=True)
    ctx.add_table("07_cluster", clusters)
    assign = ctx.errors.copy()
    # se il tracciato ha già colonne con questi nomi, usiamo un prefisso per non collidere
    pre = "" if not {"CLUSTER", "CLUSTER_ETICHETTA"} & set(assign.columns) else "TARGET_"
    assign.insert(0, f"{pre}CLUSTER", labels.values)
    assign.insert(1, f"{pre}CLUSTER_ETICHETTA", labels.map(assign_desc).values)
    ctx.add_table("07_cluster_assegnazioni", assign)
    quality = "ALTA" if best_s >= 0.3 else "MEDIA"
    ctx.add_finding(Finding("FASE 7", "CLUSTER", f"{best_k} cluster di errore individuati (silhouette {best_s:.2f})",
                            "; ".join(f"C{r.CLUSTER}: {r.ETICHETTA} ({r.PCT_ERRORI}%)" for r in clusters.itertuples()),
                            quality, "Trattare ogni cluster come un filone di analisi separato."))
    return clusters
