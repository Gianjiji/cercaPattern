"""
FASE 5 - Analisi temporale.

Ricostruisce un timestamp da DATETIME (o DATA+TIME, o TIME_KEY) e calcola:
- tasso di errore per ora del giorno e per giorno della settimana;
- heatmap giorno-settimana x ora;
- serie storica giornaliera/oraria (volumi ed errori);
- finestre di degrado: ore in cui gli errori superano in modo statisticamente significativo
  quelli attesi dato il volume dell'ora e il tasso dei giorni vicini (test binomiale, FDR).
  Un'ora con 4 errori su 13 righe non è un degrado: è rumore;
- ambito di ogni finestra: trasversale (tutti i valori di tutte le dimensioni: tipico di un
  problema infrastrutturale) oppure localizzata su un valore (tipico di un flusso o di un rilascio);
- cambi di livello: giorni dai quali il tasso cambia in modo persistente (tipico di un
  rilascio o di un cambio di configurazione), con il valore che guida il cambio;
- fasce orarie ricorrenti: ore del giorno critiche anche fuori dalle finestre di degrado
  (un singolo incidente alle 10 non rende "critiche" le 10 di ogni giorno).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy

from .config import AnalysisContext, Finding, LOG, key_dimensions, plural
from .loader import parse_datetime
from .stats_utils import binom_excess_pvalue, fdr_bh

GIORNI = ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"]
BURST_Q = 0.01            # false discovery rate per le ore anomale
BURST_MIN_ERRORS = 5      # un'ora con meno errori non è mai un degrado
BURST_MIN_RATIO = 2.0     # ... e deve avere almeno il doppio degli errori attesi
BASELINE_HOURS = 84       # il tasso atteso di un'ora viene dai 3,5 giorni prima e dopo
MAX_BUCKETS = 200_000     # oltre (~23 anni di ore) la serie oraria non viene costruita
SHIFT_MIN_RATIO = 1.5     # un cambio di livello deve spostare il tasso di almeno il 50%
SCOPE_DIMS = 15           # dimensioni considerate per l'ambito di finestre e cambi di livello


# ---------------------------------------------------------------------------
# Timestamp
# ---------------------------------------------------------------------------
def build_timestamp(ctx: AnalysisContext) -> Optional[pd.Series]:
    """Ritorna una Series datetime (NaT dove non parsabile) o None se nessuna sorgente."""
    if "timestamp" in ctx._cache:
        return ctx._cache["timestamp"]
    ctx._cache["timestamp"] = ts = _build_timestamp(ctx)
    return ts


def _build_timestamp(ctx: AnalysisContext) -> Optional[pd.Series]:
    c = ctx.cols
    ts = None
    if c.has("datetime"):
        ts = ctx.datetime(c.get("datetime"))
        ctx.meta["ts_source"] = c.get("datetime")
    if (ts is None or ts.notna().mean() < 0.5) and c.has("date"):
        d = ctx.df[c.get("date")]
        if c.has("time"):
            comb = (d.astype(object) + " " + ctx.df[c.get("time")].astype(object)).where(d.notna() & ctx.df[c.get("time")].notna())
            ts2 = parse_datetime(comb)
            if ts2.notna().mean() >= 0.5:
                ts, ctx.meta["ts_source"] = ts2, f"{c.get('date')}+{c.get('time')}"
        if ts is None or ts.notna().mean() < 0.5:
            ts2 = ctx.datetime(c.get("date"))
            if ts2.notna().mean() >= 0.5:
                ts, ctx.meta["ts_source"] = ts2, c.get("date")
    if (ts is None or ts.notna().mean() < 0.5) and c.has("time_key"):
        ts2 = ctx.datetime(c.get("time_key"))
        if ts2.notna().mean() >= 0.5:
            ts, ctx.meta["ts_source"] = ts2, c.get("time_key")
    # se abbiamo solo l'ora (TIME) senza data: usiamo una data fittizia per la fascia oraria
    if (ts is None or ts.notna().mean() < 0.5) and c.has("time"):
        ts2 = ctx.datetime(c.get("time"))
        if ts2.notna().mean() >= 0.5:
            ts, ctx.meta["ts_source"] = ts2, f"{c.get('time')} (solo ora)"
            ctx.meta["ts_only_time"] = True
    if ts is None or ts.notna().mean() < 0.5:
        ctx.meta.pop("ts_source", None)
        LOG.warning("Nessuna colonna temporale parsabile: fase 5 limitata")
        return None
    ts = _drop_outlier_dates(ctx, ts)
    LOG.info("Timestamp da %s (%.1f%% parsati)", ctx.meta["ts_source"], 100 * ts.notna().mean())
    return ts


def _drop_outlier_dates(ctx: AnalysisContext, ts: pd.Series) -> pd.Series:
    """Date sentinella (1900-01-01, 2099-12-31...) allungherebbero la serie oraria di decenni:
    se pochi valori sono lontanissimi dal grosso dei dati vengono trattati come mancanti."""
    v = ts.dropna()
    if len(v) < 20 or ctx.meta.get("ts_only_time"):
        return ts
    lo, hi = v.quantile(0.005), v.quantile(0.995)
    margin = max(hi - lo, pd.Timedelta(days=1)) * 2
    out = (ts < lo - margin) | (ts > hi + margin)
    if out.any():
        n = int(out.sum())
        ctx.add_finding(Finding("FASE 5", "QUALITA' DATI", f"{n} date fuori dal periodo dei dati ignorate nell'analisi temporale",
                                f"periodo prevalente {lo:%Y-%m-%d} -> {hi:%Y-%m-%d}; estremi trovati {v.min():%Y-%m-%d} e {v.max():%Y-%m-%d}",
                                "MEDIA", "Verificare se sono date sentinella o errori di alimentazione."))
        ts = ts.mask(out)
    return ts


# ---------------------------------------------------------------------------
# Ore anomale e finestre di degrado
# ---------------------------------------------------------------------------
def detect_bursts(n: np.ndarray, e: np.ndarray, half_window: int = BASELINE_HOURS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ore con più errori di quanto il loro volume giustifichi.

    Per ogni ora gli errori attesi sono volume x tasso "di zona", cioè il tasso delle ore nei
    ``half_window`` bucket prima e dopo, escluse l'ora stessa e quelle già giudicate anomale.
    Il tasso di zona segue i cambi di livello duraturi (che non sono degradi episodici) e non
    viene gonfiato dal degrado stesso. Ritorna (anomala, errori attesi, q-value)."""
    n = np.asarray(n, dtype=float)
    e = np.asarray(e, dtype=float)
    flag = np.zeros(len(n), dtype=bool)
    expected = np.zeros(len(n))
    q = np.ones(len(n))
    win = 2 * half_window + 1
    for _ in range(6):
        nc, ec = np.where(flag, 0.0, n), np.where(flag, 0.0, e)
        tot_n, tot_e = nc.sum(), ec.sum()
        n_roll = pd.Series(nc).rolling(win, center=True, min_periods=1).sum().to_numpy() - nc
        e_roll = pd.Series(ec).rolling(win, center=True, min_periods=1).sum().to_numpy() - ec
        glob = (tot_e - ec) / np.maximum(tot_n - nc, 1)
        # con poche righe attorno il tasso di zona è instabile: si usa quello dell'intero periodo
        p0 = np.where(n_roll >= 200, e_roll / np.maximum(n_roll, 1), glob)
        p0 = np.clip(p0, 0.0, 1.0)
        expected = n * p0
        pv = np.where(n > 0, stats.binom.sf(e - 1, n, p0), 1.0)   # P(almeno e errori | volume, tasso di zona)
        q = fdr_bh(pv)
        new = (q <= BURST_Q) & (e >= BURST_MIN_ERRORS) & (e >= BURST_MIN_RATIO * expected) & (p0 < 0.5)
        if np.array_equal(new, flag):
            break
        flag = new
    return flag, expected, q


def _windows(hb: pd.DataFrame) -> List[Tuple[int, int]]:
    """Intervalli [inizio, fine] (indici di riga) di ore anomale contigue."""
    out, start = [], None
    flags = hb["BURST"].to_numpy()
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        if not f and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


def _scope_dims(ctx: AnalysisContext) -> List[str]:
    """Dimensioni su cui valutare l'ambito di una finestra o di un cambio di livello: le dimensioni
    descrittive del tracciato, a partire dalle più associate al target."""
    return key_dimensions(ctx, SCOPE_DIMS)


def _codes(ctx: AnalysisContext, col: str) -> Tuple[np.ndarray, np.ndarray]:
    """Colonna come codici interi + valori distinti (i nulli hanno un codice proprio). Finestre e
    cambi di livello interrogano più volte le stesse poche dimensioni: la codifica si fa una volta."""
    cache = ctx._cache.setdefault("scope_codes", {})
    if col not in cache:
        codes, uniques = pd.factorize(ctx.df[col].to_numpy(dtype=object), use_na_sentinel=False)
        cache[col] = (codes.astype(np.int32), uniques)
    return cache[col]


def _scope(ctx: AnalysisContext, rows: np.ndarray) -> str:
    """'trasversale' oppure 'COL=VAL (x% degli errori, y% del volume)' per le righe indicate
    (maschera booleana sull'intero dataset). Localizzata = un valore raccoglie quasi tutti gli
    errori della finestra senza essere quasi tutto il volume della finestra."""
    err = ctx.is_error.to_numpy() & rows
    n_rows, n_err = int(rows.sum()), int(err.sum())
    if n_err < 5:
        return "n/d"
    best = None
    for c in _scope_dims(ctx):
        codes, uniques = _codes(ctx, c)
        e = np.bincount(codes[err], minlength=len(uniques))
        v = np.bincount(codes[rows], minlength=len(uniques))
        i = int(e.argmax())
        se, sv = e[i] / n_err, v[i] / max(n_rows, 1)
        if se >= 0.8 and sv <= 0.6 and (best is None or se - sv > best[0]):
            best = (se - sv, f"{c}={_label(uniques[i])} ({100 * se:.0f}% degli errori, {100 * sv:.0f}% del volume)")
    return best[1] if best else "trasversale"


def _label(value: Any) -> str:
    return "(NULL)" if pd.isna(value) else str(value)


def hourly_profile(ctx: AnalysisContext) -> Optional[Dict[str, Any]]:
    """Serie oraria con ore anomale, finestre di degrado e appartenenza di ogni riga a una finestra.
    Calcolata una volta: serve alla fase 5 e, prima, al pattern mining (fase 4)."""
    if "hourly" in ctx._cache:
        return ctx._cache["hourly"]
    ctx._cache["hourly"] = None
    ts = build_timestamp(ctx)
    if ts is None or ctx.meta.get("ts_only_time"):
        return None
    valid = ts.notna().to_numpy()
    if not valid.any():
        return None
    bucket = ts.dt.floor("h")
    err = ctx.is_error.to_numpy()
    start, end = bucket.min(), bucket.max()
    n_buckets = int((end - start) / pd.Timedelta(hours=1)) + 1
    if n_buckets > MAX_BUCKETS:
        LOG.warning("Periodo troppo lungo per la serie oraria (%s ore): finestre di degrado non calcolate", f"{n_buckets:,}")
        return None
    pos = np.full(len(ts), -1, dtype=np.int64)
    pos[valid] = ((bucket[valid] - start) / pd.Timedelta(hours=1)).round().astype(np.int64).to_numpy()
    n = np.bincount(pos[valid], minlength=n_buckets)
    e = np.bincount(pos[valid & err], minlength=n_buckets)
    flag, expected, q = detect_bursts(n, e)
    hb = pd.DataFrame({"BUCKET_ORA": pd.date_range(start, periods=n_buckets, freq="h"), "N": n, "N_ERROR": e})
    hb["TASSO_ERRORE_PCT"] = (100 * hb["N_ERROR"] / hb["N"].replace(0, np.nan)).round(3)
    hb["ERRORI_ATTESI"] = np.round(expected, 2)
    hb["Q_VALUE"] = q
    hb["BURST"] = flag
    in_window = np.zeros(len(ts), dtype=bool)
    in_window[valid] = flag[pos[valid]]
    prof = {"hb": hb, "pos": pos, "valid": valid, "in_window": in_window, "windows": _windows(hb)}
    ctx._cache["hourly"] = prof
    return prof


def window_membership(ctx: AnalysisContext) -> Optional[np.ndarray]:
    """Per ogni riga: cade in una finestra di degrado? None se non ci sono finestre."""
    prof = hourly_profile(ctx)
    if prof is None or not prof["in_window"].any():
        return None
    return prof["in_window"]


# ---------------------------------------------------------------------------
# Cambi di livello
# ---------------------------------------------------------------------------
def _loglik(e: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Log-verosimiglianza binomiale al tasso di massima verosimiglianza e/n."""
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, e / np.maximum(n, 1), 0.0)
    return xlogy(e, p) + xlogy(n - e, 1 - p)


def _best_split(n: np.ndarray, e: np.ndarray, min_seg: int) -> Optional[Tuple[int, float]]:
    """Giorno che divide meglio la serie in due livelli: (indice del primo giorno del secondo
    segmento, p-value corretto per il numero di tagli provati)."""
    k = len(n)
    if k < 2 * min_seg:
        return None
    cn, ce = np.cumsum(n), np.cumsum(e)
    t = np.arange(min_seg, k - min_seg + 1)
    n1, e1 = cn[t - 1], ce[t - 1]
    n2, e2 = cn[-1] - n1, ce[-1] - e1
    ok = (n1 > 0) & (n2 > 0)
    if not ok.any():
        return None
    llr = _loglik(e1, n1) + _loglik(e2, n2) - _loglik(np.array(ce[-1]), np.array(cn[-1]))
    llr = np.where(ok, llr, -np.inf)
    i = int(llr.argmax())
    p = float(min(1.0, stats.chi2.sf(2 * max(llr[i], 0.0), 1) * len(t)))
    return int(t[i]), p


def detect_level_shifts(n: np.ndarray, e: np.ndarray, max_shifts: int = 3, alpha: float = 0.001) -> List[int]:
    """Indici dei giorni dai quali il tasso cambia livello in modo persistente (segmentazione binaria).

    Un taglio è accettato se è significativo, se il tasso complessivo dei due lati differisce di
    almeno ``SHIFT_MIN_RATIO`` volte e se lo stesso vale per la mediana dei tassi giornalieri:
    la mediana ignora il singolo giorno anomalo, che è un incidente e non un cambio di livello."""
    n = np.asarray(n, dtype=float)
    e = np.asarray(e, dtype=float)
    min_seg = 3 if len(n) >= 10 else 2
    cuts: List[int] = []

    def ratio(a: float, b: float) -> float:
        lo, hi = min(a, b), max(a, b)
        return np.inf if lo <= 0 < hi else (hi / lo if lo > 0 else 1.0)

    def split(lo: int, hi: int) -> None:
        if len(cuts) >= max_shifts:
            return
        found = _best_split(n[lo:hi], e[lo:hi], min_seg)
        if found is None:
            return
        t, p = found[0] + lo, found[1]
        r1, r2 = e[lo:t].sum() / max(n[lo:t].sum(), 1), e[t:hi].sum() / max(n[t:hi].sum(), 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            daily = np.where(n > 0, e / np.maximum(n, 1), np.nan)
        m1, m2 = np.nanmedian(daily[lo:t]), np.nanmedian(daily[t:hi])
        # almeno 5 errori di differenza rispetto all'atteso: evita "cambi" su tassi minuscoli
        moved = abs(e[t:hi].sum() - n[t:hi].sum() * r1) >= 5
        if p <= alpha and ratio(r1, r2) >= SHIFT_MIN_RATIO and ratio(m1, m2) >= SHIFT_MIN_RATIO and (r2 > r1) == (m2 > m1) and moved:
            cuts.append(t)
            split(lo, t)
            split(t, hi)

    split(0, len(n))
    return sorted(cuts)


def _shift_driver(ctx: AnalysisContext, before: np.ndarray, after: np.ndarray) -> str:
    """Valore che spiega la maggior parte della variazione di errori tra i due periodi
    (maschere booleane sull'intero dataset), oppure 'trasversale'."""
    err = ctx.is_error.to_numpy()
    r1 = err[before].sum() / max(before.sum(), 1)
    total = err[after].sum() - after.sum() * r1          # errori in più (o in meno) rispetto al tasso precedente
    if abs(total) < 5:
        return "n/d"
    best = None
    for c in _scope_dims(ctx):
        codes, uniques = _codes(ctx, c)
        k = len(uniques)
        n1, e1 = np.bincount(codes[before], minlength=k), np.bincount(codes[before & err], minlength=k)
        n2, e2 = np.bincount(codes[after], minlength=k), np.bincount(codes[after & err], minlength=k)
        # tasso precedente del valore; se il valore non esisteva prima, il tasso precedente globale
        prev = np.where(n1 > 0, e1 / np.maximum(n1, 1), r1)
        delta = (e2 - n2 * prev) / total
        i = int(delta.argmax())
        vol = n2[i] / max(after.sum(), 1)
        if delta[i] >= 0.7 and vol <= 0.6 and (best is None or delta[i] - vol > best[0]):
            new = " (valore assente prima del cambio)" if n1[i] == 0 else ""
            best = (delta[i] - vol, f"{c}={_label(uniques[i])} ({100 * min(delta[i], 1):.0f}% della variazione, {100 * vol:.0f}% del volume){new}")
    return best[1] if best else "trasversale"


# ---------------------------------------------------------------------------
# Fase
# ---------------------------------------------------------------------------
def run_temporal(ctx: AnalysisContext) -> None:
    ts = build_timestamp(ctx)
    if ts is None:
        return
    E = ctx.system_label
    base = ctx.error_rate
    prof = hourly_profile(ctx)
    in_window = prof["in_window"] if prof else np.zeros(len(ts), dtype=bool)
    d = pd.DataFrame({"ts": ts, "err": ctx.is_error.astype(int), "win": in_window}).dropna(subset=["ts"])
    d["ora"] = d["ts"].dt.hour
    d["dow"] = d["ts"].dt.day_of_week
    d["giorno"] = d["ts"].dt.floor("D")

    # --- fasce orarie ------------------------------------------------------
    hourly = d.groupby("ora")["err"].agg(N="count", N_ERROR="sum")
    hourly["TASSO_ERRORE_PCT"] = (100 * hourly["N_ERROR"] / hourly["N"]).round(3)
    hourly["LIFT"] = ((hourly["N_ERROR"] / hourly["N"]) / base).round(2) if base else np.nan
    # ricorrenza: stesso calcolo senza le finestre di degrado
    clean = d[~d["win"]].groupby("ora")["err"].agg(N_FUORI_FINESTRE="count", N_ERROR_FUORI_FINESTRE="sum")
    hourly = hourly.join(clean).fillna({"N_FUORI_FINESTRE": 0, "N_ERROR_FUORI_FINESTRE": 0})
    base_clean = hourly["N_ERROR_FUORI_FINESTRE"].sum() / max(hourly["N_FUORI_FINESTRE"].sum(), 1)
    rate_clean = hourly["N_ERROR_FUORI_FINESTRE"] / hourly["N_FUORI_FINESTRE"].replace(0, np.nan)
    hourly["TASSO_FUORI_FINESTRE_PCT"] = (100 * rate_clean).round(3)
    hourly["Q_VALUE"] = fdr_bh(binom_excess_pvalue(hourly["N_ERROR_FUORI_FINESTRE"].to_numpy(), hourly["N_FUORI_FINESTRE"].to_numpy(), base_clean))
    crit_mask = (hourly["Q_VALUE"] <= BURST_Q) & (rate_clean >= 1.5 * base_clean) & (hourly["N_ERROR_FUORI_FINESTRE"] >= 5)
    hourly["CRITICA"] = np.where(crit_mask, "SI", "NO")
    hourly = hourly.reset_index().rename(columns={"ora": "ORA"})
    ctx.add_table("05_tasso_per_ora", hourly)
    crit = hourly[hourly["CRITICA"] == "SI"]
    if not crit.empty:
        ctx.add_finding(Finding("FASE 5", "FASCE ORARIE", f"Fasce orarie ricorrenti critiche: {', '.join(f'{int(h)}h' for h in crit['ORA'])}",
                                "; ".join(f"{int(r.ORA)}h: tasso {r.TASSO_FUORI_FINESTRE_PCT}% fuori dalle finestre di degrado (media {100 * base_clean:.2f}%)" for r in crit.itertuples()),
                                "MEDIA", f"Correlare con finestre batch, deploy o picchi di carico sui servizi di {E}." if ctx.is_system_target
                                else "Correlare con finestre batch, deploy o picchi di carico."))
    if ctx.meta.get("ts_only_time"):
        return

    # --- heatmap dow x ora -------------------------------------------------
    hm = d.pivot_table(index="dow", columns="ora", values="err", aggfunc="mean").reindex(range(7)) * 100
    hm.index = [GIORNI[i] for i in hm.index]
    ctx.add_table("05_heatmap_dow_ora", hm.round(2).reset_index().rename(columns={"index": "GIORNO"}))
    cnt = d.pivot_table(index="dow", columns="ora", values="err", aggfunc="sum").reindex(range(7)).fillna(0)
    cnt.index = [GIORNI[i] for i in cnt.index]
    ctx.meta["heatmap_rate"], ctx.meta["heatmap_count"] = hm, cnt

    # --- serie giornaliera -------------------------------------------------
    daily = d.groupby("giorno")["err"].agg(N="count", N_ERROR="sum")
    daily["TASSO_ERRORE_PCT"] = (100 * daily["N_ERROR"] / daily["N"]).round(3)
    daily_clean = d[~d["win"]].groupby("giorno")["err"].agg(N="count", N_ERROR="sum").reindex(daily.index, fill_value=0)
    ctx.meta["period"] = (d["ts"].min(), d["ts"].max())

    # --- ore anomale e finestre di degrado ---------------------------------
    win = pd.DataFrame(columns=["INIZIO", "FINE", "ORE", "N", "N_ERROR", "TASSO_ERRORE_PCT", "ERRORI_ATTESI", "ECCESSO_ERRORI", "AMBITO"])
    if prof is not None:
        hb = prof["hb"]
        ctx.add_table("05_serie_oraria", hb)
        rows = []
        for a, b in prof["windows"]:
            seg = hb.iloc[a:b + 1]
            inside = prof["valid"] & (prof["pos"] >= a) & (prof["pos"] <= b)
            rows.append({"INIZIO": seg["BUCKET_ORA"].iloc[0], "FINE": seg["BUCKET_ORA"].iloc[-1], "ORE": len(seg), "N": int(seg["N"].sum()),
                         "N_ERROR": int(seg["N_ERROR"].sum()),
                         "TASSO_ERRORE_PCT": round(100 * seg["N_ERROR"].sum() / max(seg["N"].sum(), 1), 2),
                         "ERRORI_ATTESI": round(float(seg["ERRORI_ATTESI"].sum()), 1),
                         "ECCESSO_ERRORI": round(float(seg["N_ERROR"].sum() - seg["ERRORI_ATTESI"].sum()), 1),
                         "AMBITO": _scope(ctx, inside)})
        if rows:
            win = pd.DataFrame(rows).sort_values("N_ERROR", ascending=False).reset_index(drop=True)
    ctx.add_table("05_finestre_degrado", win)
    share = 100 * win["N_ERROR"].sum() / max(ctx.n_errors, 1) if not win.empty else 0.0
    cross = win[win["AMBITO"].isin(["trasversale", "n/d"])] if not win.empty else win
    ctx.meta["degrado_share"] = float(share)
    ctx.meta["degrado_share_trasversale"] = float(100 * cross["N_ERROR"].sum() / max(ctx.n_errors, 1)) if not win.empty else 0.0
    ctx.meta["degrado_localizzato"] = [a for a in win["AMBITO"].unique() if a not in ("trasversale", "n/d")] if not win.empty else []
    if not win.empty:
        w = win.iloc[0]
        local = ctx.meta["degrado_localizzato"]
        ctx.add_finding(Finding(
            "FASE 5", "FINESTRE DI DEGRADO",
            f"{plural(len(win), 'finestra', 'finestre')} di degrado: {plural(int(win['ORE'].sum()), 'ora', 'ore')} con errori ben oltre l'atteso",
            f"Le finestre contengono il {share:.1f}% degli errori ({win['ECCESSO_ERRORI'].sum():.0f} oltre l'atteso); la più grave: "
            f"{w['INIZIO']:%Y-%m-%d %H:00} -> {w['FINE']:%Y-%m-%d %H:59} ({int(w['N_ERROR'])} errori su {int(w['N'])} righe, attesi {w['ERRORI_ATTESI']:.0f}; ambito: {w['AMBITO']})",
            "ALTA" if share >= 30 else "MEDIA",
            ("Finestre trasversali a tutte le dimensioni: causa prevalentemente infrastrutturale (indisponibilità/timeout). Incrociare con incident, deploy e manutenzioni"
             + (f" di {E}" if ctx.is_system_target else "") + " nelle stesse ore.") if not local else
            f"Finestre localizzate su {'; '.join(local[:3])}: verificare rilasci o anomalie del singolo flusso in quelle ore, non l'infrastruttura comune."))

    # --- cambi di livello sulla serie giornaliera (senza le finestre di degrado) ---
    shifts = []
    dn, de = daily_clean["N"].to_numpy(), daily_clean["N_ERROR"].to_numpy()
    cuts = detect_level_shifts(dn, de) if len(daily) >= 4 else []
    bounds = [0] + cuts + [len(daily)]
    level = np.zeros(len(daily))
    for a, b in zip(bounds[:-1], bounds[1:]):
        level[a:b] = 100 * de[a:b].sum() / max(dn[a:b].sum(), 1)
    daily["LIVELLO_PCT"] = np.round(level, 3)
    days = daily.index
    day_of_row = ts.dt.floor("D")
    for j, t in enumerate(cuts):
        a, b = bounds[j], bounds[j + 2]
        r1, r2 = de[a:t].sum() / max(dn[a:t].sum(), 1), de[t:b].sum() / max(dn[t:b].sum(), 1)
        before = ((day_of_row >= days[a]) & (day_of_row < days[t])).to_numpy() & ~in_window
        after = ((day_of_row >= days[t]) & (day_of_row <= days[b - 1])).to_numpy() & ~in_window
        shifts.append({"DATA_CAMBIO": days[t], "DIREZIONE": "AUMENTO" if r2 > r1 else "CALO",
                       "TASSO_PRIMA_PCT": round(100 * r1, 3), "TASSO_DOPO_PCT": round(100 * r2, 3),
                       "RAPPORTO": round(r2 / r1, 2) if r1 > 0 else np.inf, "GIORNI_PRIMA": int(t - a), "GIORNI_DOPO": int(b - t),
                       "ERRORI_DOPO": int(de[t:b].sum()), "GUIDATO_DA": _shift_driver(ctx, before, after)})
    ctx.add_table("05_serie_giornaliera", daily.reset_index().rename(columns={"giorno": "GIORNO"}))
    sh = pd.DataFrame(shifts, columns=["DATA_CAMBIO", "DIREZIONE", "TASSO_PRIMA_PCT", "TASSO_DOPO_PCT", "RAPPORTO", "GIORNI_PRIMA",
                                       "GIORNI_DOPO", "ERRORI_DOPO", "GUIDATO_DA"])
    ctx.add_table("05_cambi_livello", sh)
    ctx.meta["level_shifts"] = shifts
    for s in shifts:
        up = s["DIREZIONE"] == "AUMENTO"
        drv = s["GUIDATO_DA"]
        ctx.add_finding(Finding(
            "FASE 5", "CAMBIO DI LIVELLO",
            f"Dal {s['DATA_CAMBIO']:%Y-%m-%d} il tasso {'sale' if up else 'scende'} dal {s['TASSO_PRIMA_PCT']:.2f}% al {s['TASSO_DOPO_PCT']:.2f}% in modo persistente",
            f"{s['GIORNI_PRIMA']} giorni prima, {s['GIORNI_DOPO']} giorni dopo; variazione x{s['RAPPORTO']}; guidato da: {drv}",
            "ALTA" if s["GIORNI_DOPO"] >= 3 and s["GIORNI_PRIMA"] >= 3 else "MEDIA",
            (f"Un salto persistente è tipico di un rilascio o di un cambio di configurazione: verificare cosa è cambiato il {s['DATA_CAMBIO']:%Y-%m-%d}"
             + (f" su {drv.split(' (')[0]}." if drv not in ("trasversale", "n/d") else " (componenti comuni a tutti i flussi).")) if up else
            f"Il tasso si è ridotto dal {s['DATA_CAMBIO']:%Y-%m-%d}: verificare quale intervento ha avuto effetto e se il problema è chiuso."))
    if win.empty and not shifts:
        ctx.add_finding(Finding("FASE 5", "DISTRIBUZIONE TEMPORALE", "Casi distribuiti nel tempo senza finestre di degrado né cambi di livello",
                                "Nessuna ora con errori significativamente oltre l'atteso e nessun salto persistente del tasso giornaliero", "MEDIA",
                                "Fenomeno di natura strutturale/funzionale (configurazione o integrazione), non episodica."))
    # giorno della settimana
    dow = d.groupby("dow")["err"].agg(N="count", N_ERROR="sum")
    dow["TASSO_ERRORE_PCT"] = (100 * dow["N_ERROR"] / dow["N"]).round(3)
    dow.index = [GIORNI[i] for i in dow.index]
    ctx.add_table("05_tasso_per_giorno_sett", dow.reset_index().rename(columns={"index": "GIORNO"}))
