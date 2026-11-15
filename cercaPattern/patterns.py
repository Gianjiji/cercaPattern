"""
FASE 4 - Pattern discovery.

Tecniche applicate (tutte su encoding "COLONNA=VALORE" delle dimensioni
categoriche, con item target ``<COLONNA>=<VALORE>``):

1. Pattern mining orientato al target. Le combinazioni candidate vengono cercate
   con FP-growth (mlxtend) sulle SOLE righe target: una regola ``A -> target`` può
   superare il supporto minimo solo se ``A`` è frequente tra le righe target, quindi
   cercare tra tutte le righe è lavoro sprecato (e molto lento). Per ogni candidato
   righe ed errori vengono poi contati in modo ESATTO su tutto il dataset con insiemi
   di bit: nessun campionamento, quindi confidenza e lift non sono distorti.
   Per ogni regola: supporto, confidenza (= tasso di errore nel pattern) con limite
   inferiore di Wilson, lift, coverage (= % di tutti gli errori spiegati), errori in
   eccesso rispetto all'atteso e significatività corretta per test multipli (FDR).
2. Copertura cumulativa: sequenza minima di pattern che, uno dopo l'altro, spiegano
   la quota più grande di righe target *non ancora spiegate* (weighted covering).
   Risponde a "quanti problemi distinti ci sono e quanto pesa ciascuno".
3. Correlation analysis: Cramér's V e Mutual Information colonna -> target.
4. Decision Tree explainability: estrazione delle regole (percorsi radice ->
   foglia) con purezza e copertura contate su tutto il dataset.
5. XGBoost + SHAP: feature importance globale aggregata per colonna, con AUC
   misurata su righe tenute fuori dall'addestramento (hold-out).

Tutte le tecniche lavorano sulle feature *disponibili*: se una libreria
manca o il modello fallisce, la fase registra l'anomalia e prosegue.
"""
from __future__ import annotations

import hashlib
import importlib.util
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .config import AnalysisContext, Finding, LOG, categorical_columns, derived_feature_frame, is_null_series
from .stats_utils import binom_excess_pvalue, fdr_bh, pack, popcount, wilson_interval

OTHER_LABEL = "(ALTRO)"
MIN_PATTERN_ROWS = 3          # un pattern deve riguardare almeno 3 righe target
TARGET_SHARE_FLOOR = 0.05     # se il target è raro la soglia scende al 5% delle righe target
TARGET_SHARE_MIN = 0.01       # ... ma un pattern che spiega meno dell'1% delle righe target non interessa
MAX_Q_VALUE = 0.05            # false discovery rate accettato
COVER_ALPHA = 0.001           # significatività richiesta a ogni passo della copertura cumulativa
Item = str
Itemset = Tuple[Item, ...]


# ---------------------------------------------------------------------------
# Encoding comune
# ---------------------------------------------------------------------------
def _sample(ctx: AnalysisContext, df: pd.DataFrame, is_err: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Campionamento stratificato per dataset grandi (mantiene tutti gli errori)."""
    if len(df) <= ctx.sample_rows:
        return df, is_err
    err_idx = df.index[is_err]
    ok_idx = df.index[~is_err]
    n_ok = max(ctx.sample_rows - len(err_idx), len(err_idx) * 5)
    ok_s = pd.Index(np.random.RandomState(ctx.random_state).choice(ok_idx, size=min(n_ok, len(ok_idx)), replace=False))
    keep = err_idx.union(ok_s)
    LOG.info("Campionamento stratificato per albero e XGBoost: %s -> %s righe", f"{len(df):,}", f"{len(keep):,}")
    return df.loc[keep], is_err.loc[keep]


def build_feature_frame(ctx: AnalysisContext, cols: List[str], max_levels: Optional[int] = None, other_label: str = OTHER_LABEL) -> pd.DataFrame:
    """DataFrame categorico (stringhe) con livelli rari raggruppati e nulli espliciti.
    Ogni colonna viene preparata una volta sola e riusata dalle fasi successive."""
    max_levels = max_levels or ctx.max_cardinality
    derived = derived_feature_frame(ctx)
    cache: Dict[str, pd.Series] = ctx._cache.setdefault(("feature", max_levels, other_label), {})
    data: Dict[str, pd.Series] = {}
    for c in list(cols) + list(derived.columns):
        if c not in cache:
            if c in ctx.df.columns:
                src, nul = ctx.df[c], ctx.isnull(c)
            else:
                src, nul = derived[c], is_null_series(derived[c])
            s = src.astype(object).where(~nul, "(NULL)").astype(str)
            vc = s.value_counts()
            if len(vc) > max_levels:
                keep = set(vc.index[:max_levels])
                s = s.where(s.isin(keep), other_label)
            cache[c] = s
        data[c] = cache[c]
    return pd.DataFrame(data, index=ctx.df.index)


def one_hot(frame: pd.DataFrame) -> pd.DataFrame:
    """One-hot booleano con nome colonna 'COL=VAL'."""
    if frame.shape[1] == 0:
        return pd.DataFrame(index=frame.index)
    oh = pd.get_dummies(frame, prefix_sep="=", dtype=bool)
    return oh


class ItemBits:
    """Per ogni item 'COL=VAL' l'insieme delle righe che lo soddisfano, su TUTTO il dataset,
    come bit impacchettati. Intersecare due insiemi è un AND su n/8 byte: contare righe ed
    errori di una combinazione resta veloce anche con milioni di righe."""

    def __init__(self, frame: pd.DataFrame, is_err: np.ndarray):
        err = np.asarray(is_err, dtype=bool)
        self.n = len(frame)
        self.n_err = int(err.sum())
        self.err = pack(err)
        self.all = pack(np.ones(self.n, dtype=bool))
        self._codes: Dict[str, np.ndarray] = {}
        self._where: Dict[Item, Tuple[str, int]] = {}
        self._bits: Dict[Item, np.ndarray] = {}
        self.rows: Dict[Item, int] = {}      # righe per item
        self.errs: Dict[Item, int] = {}      # righe target per item
        for c in frame.columns:
            codes, uniques = pd.factorize(frame[c].to_numpy(dtype=object))
            self._codes[c] = codes
            tot = np.bincount(codes, minlength=len(uniques))
            te = np.bincount(codes[err], minlength=len(uniques))
            for i, u in enumerate(uniques):
                item = f"{c}={u}"
                self._where[item] = (c, i)
                self.rows[item], self.errs[item] = int(tot[i]), int(te[i])

    def __contains__(self, item: Item) -> bool:
        return item in self._where

    def column(self, item: Item) -> str:
        return self._where[item][0]

    def get(self, item: Item) -> np.ndarray:
        b = self._bits.get(item)
        if b is None:
            c, i = self._where[item]
            b = self._bits[item] = pack(self._codes[c] == i)
        return b

    def rowset(self, items: Sequence[Item]) -> np.ndarray:
        """Righe che soddisfano tutti gli item."""
        b = self.all
        for it in items:
            b = b & self.get(it)
        return b

    def count(self, bits: np.ndarray) -> Tuple[int, int]:
        """(righe, righe target) di un insieme di righe."""
        return popcount(bits), popcount(bits & self.err)


# ---------------------------------------------------------------------------
# 1. Regole orientate al target
# ---------------------------------------------------------------------------
def _candidate_itemsets(ctx: AnalysisContext, frame_err: pd.DataFrame, min_rel: float, max_len: int,
                        skip: Sequence[Item] = (), max_rows: int = 60_000) -> List[Itemset]:
    """Combinazioni frequenti tra le righe target (FP-growth). Se le righe target sono moltissime
    i candidati si cercano su un campione: i conteggi veri vengono comunque rifatti su tutto."""
    from mlxtend.frequent_patterns import fpgrowth
    if len(frame_err) > max_rows:
        frame_err = frame_err.sample(max_rows, random_state=ctx.random_state)
        min_rel *= 0.9   # margine: un candidato al limite non va perso per effetto del campione
    oh = one_hot(frame_err)
    oh = oh.loc[:, oh.mean() >= min_rel]
    # valori rari raggruppati e item esclusi dal chiamante: non interpretabili come pattern
    oh = oh[[c for c in oh.columns if not c.endswith(f"={OTHER_LABEL}") and c not in set(skip)]]
    if oh.shape[1] == 0:
        return []
    LOG.info("FP-growth sulle righe target: %d item, %s righe (supporto minimo tra le righe target %.2f%%)",
             oh.shape[1], f"{len(oh):,}", 100 * min_rel)
    freq = fpgrowth(oh, min_support=min_rel, use_colnames=True, max_len=max_len)
    return [tuple(sorted(s)) for s in freq["itemsets"]]


def _count_itemsets(bits: ItemBits, itemsets: List[Itemset]) -> Tuple[np.ndarray, np.ndarray]:
    """Conteggio esatto (righe, righe target) di ogni combinazione su tutto il dataset. Le
    combinazioni sono già ordinate: quelle con lo stesso prefisso riusano l'intersezione fatta."""
    n_arr = np.zeros(len(itemsets), dtype=np.int64)
    e_arr = np.zeros(len(itemsets), dtype=np.int64)
    stack_items: List[Item] = []
    stack_bits: List[np.ndarray] = []
    for idx, its in enumerate(itemsets):
        if len(its) == 1:
            n_arr[idx], e_arr[idx] = bits.rows[its[0]], bits.errs[its[0]]
            continue
        k = 0
        while k < len(stack_items) and k < len(its) and stack_items[k] == its[k]:
            k += 1
        del stack_items[k:], stack_bits[k:]
        for it in its[k:]:
            b = bits.get(it)
            stack_bits.append(b if not stack_bits else stack_bits[-1] & b)
            stack_items.append(it)
        n_arr[idx], e_arr[idx] = bits.count(stack_bits[-1])
    return n_arr, e_arr


def _thresholds(n: int, n_err: int, min_support: float, min_confidence: float) -> Tuple[int, float]:
    """Soglie adattive: (righe target minime per pattern, confidenza minima).

    ``min_support`` è una quota di tutte le righe: se il caso target è più raro di così nessun
    pattern potrebbe mai superarla, quindi la soglia scende al 5% delle righe target. Allo stesso
    modo con un target raro una confidenza del 20% è irraggiungibile anche per pattern con lift
    molto alto: la soglia scende a 3 volte il tasso globale."""
    base = n_err / n
    min_count = min(int(np.ceil(min_support * n)), int(np.ceil(TARGET_SHARE_FLOOR * n_err)))
    min_count = max(MIN_PATTERN_ROWS, min_count, int(np.ceil(TARGET_SHARE_MIN * n_err)))
    min_conf = min(min_confidence, max(3 * base, 0.02))
    return min_count, min_conf


def mine_association_rules(ctx: AnalysisContext, frame: pd.DataFrame, bits: ItemBits,
                           min_support: float = 0.005, min_confidence: float = 0.2,
                           max_len: int = 3, top_n: int = 60, forced: Sequence[Item] = (),
                           skip: Sequence[Item] = ()) -> pd.DataFrame:
    """Regole ``antecedente -> target`` con al più ``max_len`` condizioni, contate su tutto il dataset.

    ``forced``: item valutati come pattern anche sotto il supporto minimo (es. "dentro una finestra
    di degrado": pochi errori ma concentrati, vanno pesati insieme agli altri pattern).
    ``skip``: item che non devono comparire nei pattern."""
    if importlib.util.find_spec("mlxtend") is None:
        LOG.warning("mlxtend non disponibile: association rules saltate")
        return pd.DataFrame()
    n, n_err = bits.n, bits.n_err
    if n_err == 0 or n_err == n or frame.shape[1] == 0:
        return pd.DataFrame()
    base = n_err / n
    min_count, min_conf = _thresholds(n, n_err, min_support, min_confidence)
    ctx.meta["pattern_min_count"], ctx.meta["pattern_min_conf"] = min_count, min_conf
    if min_count != int(np.ceil(min_support * n)) or min_conf != min_confidence:
        LOG.info("Soglie pattern adattate al target (%.2f%% delle righe): almeno %d righe target per pattern, confidenza >= %.1f%%",
                 100 * base, min_count, 100 * min_conf)
    tgt = ctx.target_item
    none = Finding("FASE 4", "PATTERN", f"Nessuna combinazione di valori è associata a {tgt} in modo significativo",
                   f"soglie: almeno {min_count} righe target, confidenza >= {100 * min_conf:.1f}%, lift > 1, FDR <= {MAX_Q_VALUE:.0%}",
                   "MEDIA", "Ridurre --min-support / --min-confidence oppure aggiungere colonne descrittive al tracciato.")
    if min_count > n_err:
        ctx.add_finding(none)
        return pd.DataFrame()
    err_mask = ctx.is_error.to_numpy()
    forced_sets = {(it,) for it in forced if it in bits}
    cands = set(_candidate_itemsets(ctx, frame[err_mask], min_count / n_err, max_len, skip=skip)) | forced_sets
    cands = sorted(cands)
    # tetto ai candidati: ogni conteggio costa ~n/8 byte; oltre il budget si tengono le combinazioni più corte
    cap = int(min(200_000, max(5_000, 2.5e10 / max(n / 8, 1))))
    if len(cands) > cap:
        LOG.info("%s combinazioni candidate: tenute le %s più corte", f"{len(cands):,}", f"{cap:,}")
        cands = sorted(sorted(cands, key=len)[:cap])
    if not cands:
        ctx.add_finding(none)
        return pd.DataFrame()
    n_arr, e_arr = _count_itemsets(bits, cands)
    conf = e_arr / np.maximum(n_arr, 1)
    lift = conf / base
    q = fdr_bh(binom_excess_pvalue(e_arr, n_arr, base))
    lo, _ = wilson_interval(e_arr, n_arr)
    enough = (e_arr >= min_count) | (np.array([c in forced_sets for c in cands]) & (e_arr >= MIN_PATTERN_ROWS))
    keep = enough & (conf >= min_conf) & (lift > 1) & (q <= MAX_Q_VALUE)
    ctx.meta["pattern_candidates"] = int(len(cands))
    if not keep.any():
        ctx.add_finding(none)
        return pd.DataFrame()
    idx = np.flatnonzero(keep)
    rules = pd.DataFrame({
        "ITEMS": [cands[i] for i in idx],
        "PATTERN": [" + ".join(cands[i]) for i in idx],
        "N_ITEM": [len(cands[i]) for i in idx],
        "N_RIGHE": n_arr[idx], "N_ERRORI": e_arr[idx],
        "CONFIDENZA_PCT": (100 * conf[idx]).round(2),                      # % errori dentro il pattern
        "CONF_MIN_95_PCT": (100 * lo[idx]).round(2),                       # limite inferiore di Wilson
        "COVERAGE_ERRORI_PCT": (100 * e_arr[idx] / n_err).round(2),        # % errori spiegati
        "LIFT": lift[idx].round(2),
        "SUPPORTO_PCT": (100 * e_arr[idx] / n).round(3),
        "ECCESSO_ERRORI": (e_arr[idx] - n_arr[idx] * base).round(1),       # errori oltre l'atteso al tasso globale
        "Q_VALUE": q[idx],
        # score composito: pattern precisi E che spiegano molti errori
        "SCORE": (conf[idx] * e_arr[idx] / n_err).round(4),
    })
    group_cols = set(ctx.target_group_cols)
    E = ctx.system_label
    rules["NOTA"] = rules["ITEMS"].apply(
        lambda its: f"contiene campi di {E}: verificare se sono dati in ingresso o in uscita (possibile conseguenza, non causa)"
        if any(bits.column(i).removesuffix("_PRESENTE") in group_cols for i in its) else "")
    rules = rules.sort_values(["SCORE", "LIFT"], ascending=False).reset_index(drop=True)
    rules = _merge_equivalent(_prune_redundant(rules), bits)
    # copertura cumulativa: quanti problemi distinti, e quanto pesa ciascuno
    try:
        cover = cumulative_cover(ctx, rules, bits)
        ctx.add_table("04_copertura_cumulativa", cover)
    except Exception as e:  # la copertura è un di più: le regole restano valide
        LOG.exception("Copertura cumulativa fallita: %s", e)
    out = rules.drop(columns=["ITEMS"]).head(top_n).reset_index(drop=True)
    out = out[["PATTERN", "N_ITEM", "N_RIGHE", "N_ERRORI", "CONFIDENZA_PCT", "CONF_MIN_95_PCT", "COVERAGE_ERRORI_PCT", "LIFT",
               "SUPPORTO_PCT", "ECCESSO_ERRORI", "Q_VALUE", "SCORE", "NOTA", "PATTERN_EQUIVALENTI"]]
    # deduzioni
    best = out.iloc[0]
    ctx.add_finding(Finding(
        "FASE 4", "PATTERN", f"Pattern principale: {best['PATTERN']}",
        f"{best['CONFIDENZA_PCT']:.1f}% delle righe con questo pattern sono {ctx.event_label} (almeno {best['CONF_MIN_95_PCT']:.1f}% al 95%); "
        f"spiega il {best['COVERAGE_ERRORI_PCT']:.1f}% dei casi (lift {best['LIFT']}, n={int(best['N_RIGHE'])})",
        "ALTA" if best["LIFT"] >= 3 else "MEDIA",
        f"Verificare l'integrazione con {ctx.system_label} per il pattern: {best['PATTERN']}." if ctx.is_system_target else f"Approfondire il pattern: {best['PATTERN']}."))
    strong = out[(out["CONF_MIN_95_PCT"] >= 80) & (out["N_RIGHE"] >= 20) & (out["LIFT"] >= 1.5)]
    if not strong.empty:
        ctx.add_finding(Finding(
            "FASE 4", "PATTERN DETERMINISTICO", f"{len(strong)} pattern con confidenza >= 80% (caso quasi sistematico)",
            "; ".join(strong["PATTERN"].head(3)), "ALTA",
            f"Pattern deterministici indicano flussi in cui {ctx.event_label} è sistematico: probabile mancata integrazione o instradamento errato." if ctx.is_system_target
            else f"Pattern deterministici: in questi flussi {ctx.event_label} è sistematico, va verificato se è atteso."))
    return out


def _prune_redundant(rules: pd.DataFrame, min_gain: float = 1.10, alpha: float = 0.001) -> pd.DataFrame:
    """Elimina le regole che aggiungono item a una regola già tenuta senza migliorarla davvero
    (es. 'A + B' scartata se 'A' ha confidenza simile). "Davvero" = confidenza più alta di almeno
    ``min_gain`` volte E miglioramento statisticamente significativo: senza il secondo requisito
    sopravvivono raffinamenti casuali come 'campo assente + PAESE=ES' (45,8% contro 41,6%)."""
    kept: Dict[frozenset, float] = {}
    keep_idx = []
    # dalle regole più semplici alle più complesse: una regola lunga sopravvive solo se migliora davvero
    order = rules.sort_values(["N_ITEM", "SCORE"], ascending=[True, False])
    for i, items, conf, n, e in zip(order.index, order["ITEMS"], order["CONFIDENZA_PCT"], order["N_RIGHE"], order["N_ERRORI"]):
        redundant = False
        for k in range(1, len(items)):
            for sub in combinations(items, k):
                parent = kept.get(frozenset(sub))
                if parent is not None and (conf <= parent * min_gain or float(binom_excess_pvalue(e, n, parent / 100)) > alpha):
                    redundant = True
                    break
            if redundant:
                break
        if not redundant:
            kept[frozenset(items)] = conf
            keep_idx.append(i)
    return rules.loc[keep_idx].sort_values(["SCORE", "LIFT"], ascending=False)


def _merge_equivalent(rules: pd.DataFrame, bits: ItemBits) -> pd.DataFrame:
    """Pattern equivalenti (selezionano ESATTAMENTE le stesse righe: es. APP=412 <=> CANALE=MOBILE)
    diventano una riga sola. Stessi conteggi non bastano: si confronta l'insieme delle righe."""
    rules = rules.copy()
    rules["PATTERN_EQUIVALENTI"] = ""
    drop: List[int] = []
    for _, grp in rules.groupby(["N_RIGHE", "N_ERRORI"], sort=False):
        if len(grp) < 2:
            continue
        by_rows: Dict[bytes, List[int]] = {}
        for i, its in grp["ITEMS"].items():
            digest = hashlib.blake2b(bits.rowset(its).tobytes(), digest_size=16).digest()
            by_rows.setdefault(digest, []).append(i)
        for same in by_rows.values():
            if len(same) > 1:
                rules.loc[same[0], "PATTERN_EQUIVALENTI"] = " ≡ ".join(rules.loc[same[1:], "PATTERN"])
                drop += same[1:]
    return rules.drop(index=drop)


def cumulative_cover(ctx: AnalysisContext, rules: pd.DataFrame, bits: ItemBits, max_steps: int = 10,
                     max_candidates: int = 250) -> pd.DataFrame:
    """Sequenza di pattern che, uno dopo l'altro, spiegano più righe target *non ancora spiegate*.

    A ogni passo si sceglie il pattern con più errori in eccesso tra le righe ancora scoperte,
    dove "atteso" è il tasso di errore delle righe scoperte: un pattern che è solo l'ombra di
    uno già scelto (es. APP=530 quando il vero problema è un campo mancante, già preso)
    non ha più eccesso e non viene scelto. L'eccesso deve essere statisticamente significativo:
    su qualche centinaio di righe scoperte un tasso un po' più alto del fondo capita per caso.
    Ci si ferma quando nessun pattern aggiunge almeno l'1% delle righe target. L'ultima riga
    misura ciò che resta fuori: il rumore di fondo."""
    n, n_err, base = bits.n, bits.n_err, bits.n_err / bits.n
    cands = rules.head(max_candidates)
    rowsets = {i: bits.rowset(its) for i, its in cands["ITEMS"].items()}
    uncovered = bits.all.copy()
    rows_left, err_left = n, n_err
    min_gain = max(MIN_PATTERN_ROWS, 0.01 * n_err)
    steps = []
    while len(steps) < max_steps and rowsets and err_left > 0:
        rate_left = err_left / max(rows_left, 1)
        best = None
        for i, rs in rowsets.items():
            new = rs & uncovered
            nr, ne = bits.count(new)
            if ne < min_gain or nr == 0 or ne / nr <= 1.5 * rate_left:
                continue
            gain = ne - nr * rate_left
            if best is not None and gain <= best[0]:
                continue
            if float(binom_excess_pvalue(ne, nr, rate_left)) > COVER_ALPHA:
                continue
            best = (gain, i, new, nr, ne)
        if best is None:
            break
        gain, i, new, nr, ne = best
        uncovered = uncovered & ~new
        rows_left, err_left = rows_left - nr, err_left - ne
        r = cands.loc[i]
        steps.append({"PASSO": len(steps) + 1, "PATTERN": r["PATTERN"], "N_RIGHE_NUOVE": nr, "N_ERRORI_NUOVI": ne,
                      "TASSO_NELLE_NUOVE_PCT": round(100 * ne / nr, 2), "QUOTA_ERRORI_PCT": round(100 * ne / n_err, 2),
                      "COPERTURA_CUMULATA_PCT": round(100 * (n_err - err_left) / n_err, 2),
                      "TASSO_RESIDUO_PCT": round(100 * err_left / max(rows_left, 1), 3)})
        del rowsets[i]
    if not steps:
        return pd.DataFrame()
    cover = pd.DataFrame(steps)
    explained = 100 * (n_err - err_left) / n_err
    resid = 100 * err_left / max(rows_left, 1)
    cover.loc[len(cover)] = {"PASSO": len(steps) + 1, "PATTERN": "(nessun pattern: rumore di fondo)", "N_RIGHE_NUOVE": rows_left,
                             "N_ERRORI_NUOVI": err_left, "TASSO_NELLE_NUOVE_PCT": round(resid, 2),
                             "QUOTA_ERRORI_PCT": round(100 * err_left / n_err, 2), "COPERTURA_CUMULATA_PCT": 100.0,
                             "TASSO_RESIDUO_PCT": round(resid, 3)}
    ctx.meta["cover_explained_pct"], ctx.meta["cover_residual_rate_pct"], ctx.meta["cover_steps"] = explained, resid, len(steps)
    ctx.add_finding(Finding(
        "FASE 4", "SPIEGAZIONE CUMULATIVA",
        f"{len(steps)} pattern {'distinto spiega' if len(steps) == 1 else 'distinti spiegano'} il {explained:.1f}% dei casi {ctx.event_label}",
        "; ".join(f"{s['PASSO']}) {s['PATTERN']}: +{s['QUOTA_ERRORI_PCT']}% (tasso {s['TASSO_NELLE_NUOVE_PCT']}%)" for s in steps[:5])
        + f". Fuori da questi pattern il tasso scende al {resid:.2f}% (globale {100 * base:.2f}%)",
        "ALTA" if explained >= 60 else "MEDIA",
        "Trattare i pattern in quest'ordine: ognuno è un problema distinto, il residuo è il tasso fisiologico atteso a problemi risolti."))
    return cover


# ---------------------------------------------------------------------------
# 2. Correlazioni
# ---------------------------------------------------------------------------
def correlation_analysis(ctx: AnalysisContext, frame: pd.DataFrame, is_err: pd.Series) -> pd.DataFrame:
    """Cramér's V, mutual information (nat) e p-value del chi-quadro di ogni colonna rispetto al target,
    dalla tabella di contingenza calcolata su tutte le righe."""
    y = np.asarray(is_err, dtype=bool).astype(np.int64)
    rows = []
    for c in frame.columns:
        codes, uniques = pd.factorize(frame[c].to_numpy(dtype=object))
        ct = np.bincount(codes * 2 + y, minlength=2 * len(uniques)).reshape(-1, 2).astype(float)
        ct = ct[ct.sum(axis=1) > 0]
        ct = ct[:, ct.sum(axis=0) > 0]
        v, p, mi = 0.0, np.nan, 0.0
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            chi2, p, _, expected = stats.chi2_contingency(ct)
            tot = ct.sum()
            v = float(np.sqrt(chi2 / (tot * (min(ct.shape) - 1))))
            nz = ct > 0
            mi = float((ct[nz] / tot * np.log(ct[nz] / expected[nz])).sum())
        rows.append({"COLONNA": c, "CRAMERS_V": round(v, 4), "MUTUAL_INFO": round(max(mi, 0.0), 5),
                     "CHI2_P_VALUE": p, "VALORI_DISTINTI": int(len(uniques))})
    out = pd.DataFrame(rows, columns=["COLONNA", "CRAMERS_V", "MUTUAL_INFO", "CHI2_P_VALUE", "VALORI_DISTINTI"])
    out = out.sort_values("CRAMERS_V", ascending=False).reset_index(drop=True)
    if not out.empty:
        top = out.iloc[0]
        ctx.add_finding(Finding("FASE 4", "CORRELAZIONE", f"Variabile più associata a {ctx.event_label}: {top['COLONNA']}",
                                f"Cramér's V={top['CRAMERS_V']}, MI={top['MUTUAL_INFO']}", "ALTA" if top["CRAMERS_V"] >= 0.3 else "MEDIA"))
    return out


# ---------------------------------------------------------------------------
# 3. Albero decisionale: regole esplicative
# ---------------------------------------------------------------------------
def decision_tree_rules(ctx: AnalysisContext, oh: pd.DataFrame, is_err: pd.Series, bits: Optional[ItemBits] = None,
                        max_depth: int = 4, min_leaf: Optional[int] = None) -> Tuple[pd.DataFrame, str]:
    """Albero poco profondo addestrato su ``oh`` (eventualmente un campione). Righe ed errori di ogni
    foglia vengono contati su tutto il dataset tramite ``bits`` quando disponibile."""
    from sklearn.tree import DecisionTreeClassifier, export_text
    X = oh.values.astype(np.uint8)
    y = is_err.astype(int).values
    min_leaf = min_leaf or max(30, len(y) // 500)
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf, class_weight="balanced", ccp_alpha=1e-4,
                                 random_state=ctx.random_state).fit(X, y)
    feat = list(oh.columns)
    exact = bits is not None and all(f in bits for f in feat)
    col_of = (lambda f: bits.column(f)) if exact else (lambda f: f.split("=", 1)[0])
    val_of = lambda f: f[len(col_of(f)) + 1:]
    levels: Dict[str, List[str]] = {}
    for f in feat:
        levels.setdefault(col_of(f), []).append(val_of(f))

    def cond_txt(c: int, t: bool) -> str:
        col, val = col_of(feat[c]), val_of(feat[c])
        if not t and len(levels.get(col, [])) == 2:   # NOT COL=A su colonna binaria -> COL=B
            other = [v for v in levels[col] if v != val][0]
            return f"{col}={other}"
        return feat[c] if t else f"NOT {feat[c]}"

    def leaf_counts(conds: List[Tuple[int, bool]]) -> Tuple[int, int]:
        if exact:
            b = bits.all
            for c, t in conds:
                ib = bits.get(feat[c])
                b = b & (ib if t else ~ib)
            return bits.count(b)
        idx = np.ones(len(X), dtype=bool)
        for c, t in conds:
            idx &= (X[:, c] == 1) if t else (X[:, c] == 0)
        return int(idx.sum()), int(y[idx].sum())

    tree = clf.tree_
    rows = []
    n_err_tot = max(bits.n_err if exact else int(y.sum()), 1)
    base = ctx.error_rate if exact else float(y.mean())

    def walk(node: int, conds: List[Tuple[int, bool]]):
        if tree.children_left[node] == -1:
            n, ne = leaf_counts(conds)
            # una condizione positiva COL=V rende ridondanti i NOT COL=W sulla stessa colonna
            pos_cols = {col_of(feat[c]) for c, t in conds if t}
            shown = [(c, t) for c, t in conds if t or col_of(feat[c]) not in pos_cols]
            desc = " AND ".join(cond_txt(c, t) for c, t in shown) or "(radice)"
            rows.append({"REGOLA": desc, "PROFONDITA": len(conds), "N_RIGHE": n, "N_ERRORI": ne,
                         "TASSO_ERRORE_PCT": round(100 * ne / n, 2) if n else 0,
                         "COVERAGE_ERRORI_PCT": round(100 * ne / n_err_tot, 2),
                         "LIFT": round((ne / n) / base, 2) if n and base else np.nan})
            return
        f = tree.feature[node]
        walk(tree.children_left[node], conds + [(f, False)])   # <= 0.5 -> feature assente
        walk(tree.children_right[node], conds + [(f, True)])   # > 0.5 -> feature presente

    walk(0, [])
    rules = pd.DataFrame(rows).sort_values(["TASSO_ERRORE_PCT", "N_ERRORI"], ascending=False).reset_index(drop=True)
    text = export_text(clf, feature_names=feat, max_depth=max_depth, show_weights=True)
    good = rules[(rules["N_ERRORI"] >= 10)]
    if not good.empty:
        b = good.iloc[0]
        ctx.add_finding(Finding("FASE 4", "ALBERO DECISIONALE", f"Regola più pura: {b['REGOLA']}",
                                f"tasso {b['TASSO_ERRORE_PCT']}% su {b['N_RIGHE']} righe (coverage {b['COVERAGE_ERRORI_PCT']}%)",
                                "ALTA" if b["TASSO_ERRORE_PCT"] >= 70 and b["LIFT"] >= 1.5 else "MEDIA"))
    return rules, text


# ---------------------------------------------------------------------------
# 4. XGBoost + SHAP
# ---------------------------------------------------------------------------
def xgb_shap_importance(ctx: AnalysisContext, oh: pd.DataFrame, is_err: pd.Series, use_shap: bool = True,
                        max_shap_rows: int = 20_000) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[object]]:
    """Ritorna (importanza per feature one-hot, importanza aggregata per colonna, explanation shap|None).

    Il modello viene addestrato su tre quarti delle righe e l'AUC misurata sul quarto restante
    (hold-out): un'AUC misurata sulle righe di addestramento è sempre ottimistica e non dice se
    le variabili spiegano davvero il caso target o lo stanno solo memorizzando."""
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    X = oh.values.astype(np.float32)
    y = is_err.astype(int).values
    pos = int(y.sum())
    neg = len(y) - pos
    rs = np.random.RandomState(ctx.random_state)
    train = np.ones(len(y), dtype=bool)
    if min(pos, neg) >= 40:   # abbastanza casi per tenerne fuori un quarto (stratificato)
        for cls in (0, 1):
            ix = np.flatnonzero(y == cls)
            train[rs.choice(ix, size=len(ix) // 4, replace=False)] = False
    pos_tr = max(int(y[train].sum()), 1)
    neg_tr = max(int(train.sum()) - pos_tr, 1)
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=neg_tr / pos_tr, eval_metric="logloss", random_state=ctx.random_state,
        n_jobs=4, tree_method="hist",
    )
    model.fit(X[train], y[train])
    feat = list(oh.columns)
    gain = model.get_booster().get_score(importance_type="gain")
    imp = pd.DataFrame({"FEATURE": feat, "GAIN": [gain.get(f"f{i}", 0.0) for i in range(len(feat))]})
    expl = None
    if use_shap:
        try:
            import shap
            idx = rs.choice(len(X), size=min(max_shap_rows, len(X)), replace=False)
            Xs = X[idx]
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(Xs)
            if isinstance(sv, list):
                sv = sv[-1]
            sv = np.asarray(sv)
            imp["MEAN_ABS_SHAP"] = np.abs(sv).mean(axis=0)
            # direzione: SHAP medio quando la feature è attiva
            on = Xs == 1
            n_on = on.sum(axis=0)
            imp["SHAP_QUANDO_ATTIVA"] = np.where(n_on > 0, (sv * on).sum(axis=0) / np.maximum(n_on, 1), 0.0)
            expl = (sv, Xs, feat)
        except Exception as e:  # pragma: no cover
            LOG.warning("SHAP non calcolabile (%s): uso gain XGBoost", e)
    key = "MEAN_ABS_SHAP" if "MEAN_ABS_SHAP" in imp else "GAIN"
    imp = imp.sort_values(key, ascending=False).reset_index(drop=True)
    imp["COLONNA"] = imp["FEATURE"].str.split("=", n=1).str[0]
    agg = imp.groupby("COLONNA")[[c for c in ("GAIN", "MEAN_ABS_SHAP") if c in imp]].sum().sort_values(key, ascending=False).reset_index()
    tot = agg[key].sum()
    agg["QUOTA_PCT"] = (100 * agg[key].astype(float) / float(tot)).round(2) if tot else 0.0
    for c in ("GAIN", "MEAN_ABS_SHAP", "SHAP_QUANDO_ATTIVA"):
        if c in imp:
            imp[c] = imp[c].astype(float).round(5)
        if c in agg:
            agg[c] = agg[c].astype(float).round(5)
    # capacità esplicativa: AUC su righe mai viste dal modello (in-sample solo se i casi sono troppo pochi)
    try:
        test = ~train
        if test.any() and len(np.unique(y[test])) == 2:
            ctx.meta["xgb_auc"], ctx.meta["xgb_auc_kind"] = float(roc_auc_score(y[test], model.predict_proba(X[test])[:, 1])), "hold-out"
        elif len(np.unique(y)) == 2:
            ctx.meta["xgb_auc"], ctx.meta["xgb_auc_kind"] = float(roc_auc_score(y, model.predict_proba(X)[:, 1])), "in-sample"
    except Exception as e:
        LOG.warning("AUC non calcolabile: %s", e)
    top = imp.iloc[0]
    pos_feats = imp[imp["SHAP_QUANDO_ATTIVA"] > 0].head(5)["FEATURE"].tolist() if "SHAP_QUANDO_ATTIVA" in imp else imp.head(5)["FEATURE"].tolist()
    auc = ctx.meta.get("xgb_auc")
    auc_txt = f"AUC {ctx.meta.get('xgb_auc_kind')}={auc:.3f}" if auc is not None else "AUC n/d"
    ctx.add_finding(Finding("FASE 4", "FEATURE IMPORTANCE", f"Feature più importante (XGBoost/SHAP): {top['FEATURE']}",
                            f"{auc_txt}; feature che spingono verso {ctx.target_value}: " + ", ".join(pos_feats),
                            "ALTA" if auc is None or auc >= 0.7 else "MEDIA",
                            "" if auc is None or auc >= 0.7 else
                            "Le colonne disponibili distinguono poco il caso target (AUC bassa): la causa è probabilmente in dati non presenti nel tracciato."))
    return imp, agg, expl


# ---------------------------------------------------------------------------
# Orchestrazione della fase
# ---------------------------------------------------------------------------
WINDOW_COL = "FINESTRA_DEGRADO"


def _with_degradation_windows(ctx: AnalysisContext, frame: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """Aggiunge al frame la colonna SI/NO "la riga cade in una finestra di degrado" (vedi fase 5).
    Ritorna il frame e il nome della colonna aggiunta (None se non ci sono finestre)."""
    try:
        from .temporal import window_membership
        inside = window_membership(ctx)
    except Exception as e:  # l'analisi temporale è accessoria per i pattern
        LOG.warning("Finestre di degrado non disponibili per il pattern mining: %s", e)
        inside = None
    if inside is None:
        return frame, None
    name = WINDOW_COL
    while name in frame.columns or name in ctx.df.columns:
        name += "_"
    return frame.assign(**{name: np.where(inside, "SI", "NO")}), name


def run_pattern_discovery(ctx: AnalysisContext, min_support: float = 0.005, min_confidence: float = 0.2,
                          use_shap: bool = True) -> None:
    exclude = ctx.target_group_cols  # evita leakage: le altre colonne dello stesso sistema descrivono l'esito stesso
    cols = categorical_columns(ctx, exclude=exclude)
    derived = derived_feature_frame(ctx)
    if not cols and derived.shape[1] == 0:
        LOG.warning("Nessuna colonna categorica utilizzabile per il pattern mining")
        return
    ctx.meta["model_columns"] = cols
    ctx.meta["derived_columns"] = list(derived.columns)
    LOG.info("Feature derivate per i modelli: %s", ctx.meta["derived_columns"])
    ctx.progress.set_total(5)
    ctx.progress.note("feature e insiemi di righe")
    frame_full = build_feature_frame(ctx, cols)
    # "dentro una finestra di degrado" entra tra le condizioni dei pattern (solo qui: nei modelli
    # sarebbe una variabile costruita guardando il target), così la copertura cumulativa pesa
    # insieme le cause legate agli attributi e quelle legate al tempo
    frame_rules, window_col = _with_degradation_windows(ctx, frame_full)
    bits = ItemBits(frame_rules, ctx.is_error.to_numpy())
    ctx.progress.step("regole orientate al target")
    # 1. regole + copertura cumulativa (conteggi esatti su tutte le righe)
    try:
        forced, skip = ([f"{window_col}=SI"], [f"{window_col}=NO"]) if window_col else ([], [])
        rules = mine_association_rules(ctx, frame_rules, bits, min_support, min_confidence, forced=forced, skip=skip)
        ctx.add_table("04_association_rules", rules)
    except Exception as e:
        LOG.exception("Association rules fallite: %s", e)
    ctx.progress.step("correlazioni (Cramer V, MI)")
    # 2. correlazioni (tutte le righe)
    try:
        corr = correlation_analysis(ctx, frame_full, ctx.is_error)
        ctx.add_table("04_correlazioni", corr)
    except Exception as e:
        LOG.exception("Correlazioni fallite: %s", e)
    ctx.progress.step("albero decisionale")
    # albero e XGBoost: su un campione stratificato se il dataset è molto grande
    df_s, err_s = _sample(ctx, ctx.df, ctx.is_error)
    oh = one_hot(frame_full.loc[df_s.index])
    # 3. albero
    try:
        rules_tree, text = decision_tree_rules(ctx, oh, err_s, bits)
        ctx.add_table("04_albero_regole", rules_tree)
        ctx.meta["tree_text"] = text
    except Exception as e:
        LOG.exception("Albero decisionale fallito: %s", e)
    ctx.progress.step("XGBoost + SHAP" if use_shap else "XGBoost")
    # 4. xgboost + shap
    try:
        imp, agg, expl = xgb_shap_importance(ctx, oh, err_s, use_shap=use_shap)
        ctx.add_table("04_feature_importance", imp.head(100))
        ctx.add_table("04_importance_per_colonna", agg)
        ctx.meta["shap_expl"] = expl
    except Exception as e:
        LOG.exception("XGBoost/SHAP fallito: %s", e)
