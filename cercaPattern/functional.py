"""
FASE 6 - Analisi funzionale del caso target.

Obiettivo: inferire *perché* si verifica il caso target osservando:
- campi mancanti: per i dati che vengono *raccolti* (identificativi, token, coordinate, testi, più
  sessione e regola applicata), quanto l'assenza è associata al caso;
- relazione con gli altri sistemi presenti nel tracciato: hanno elaborato regolarmente le
  stesse righe (problema isolato) oppure falliscono insieme (problema a monte)?
- classificazione della colonna di dettaglio del sistema target in cause tecniche (tassonomia a parole chiave);
- famiglie di valori: valori di una dimensione che iniziano con la stessa parola, oppure raggruppati
  da una regola dichiarata (``--family``), analizzati insieme;
- passaggi facoltativi: colonne in cui un valore dice "passaggio non eseguito".

Ogni blocco produce una tabella e, dove l'evidenza è forte, un Finding.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .config import (AnalysisContext, DETAIL_TAXONOMY, Finding, LOG, NEGATIVE_TOKENS, OUTCOME_WORDS, SESSION_RX, TIME_ROLES,
                     contains_token, derived_feature_frame, key_dimensions, name_tokens)

# esito di un sistema che indica mancata elaborazione. I token devono essere parole intere
# ("N/A", "KO", "GEO_ERR"), non pezzi di altre parole: "FINAL" contiene "NA", "TOKEN" contiene "KO".
# Niente lookaround: con stringhe pyarrow pandas usa RE2, che non li supporta.
FAILURE_RX = r"(?:^|[^A-Z0-9])(?:ERR[A-Z]*|FAIL[A-Z]*|KO|TIMEOUT|TIMED ?OUT|EXCEPTION|N/?A)(?:$|[^A-Z0-9])"

ROLE_SESSION, ROLE_RULE = "sessione / contesto", "regola applicata"
ROLE_SYSTEM, ROLE_DATA = "altro sistema", "dato raccolto"
SESSION_TITLE = "Identificativo di sessione"       # i finding con questo inizio alimentano l'ipotesi sulla sessione
MAX_PRESENCE_FINDINGS = 6
MAX_FAMILY_DIMS = 12


def _presence_stats(ctx: AnalysisContext, col: str, role: str) -> Dict[str, float]:
    """Tasso di errore quando il campo è presente vs assente + odds ratio."""
    present = ~ctx.isnull(col)
    e = ctx.is_error
    a, b = int((e & ~present).sum()), int((~e & ~present).sum())   # errori/non-errori con campo ASSENTE
    c, d = int((e & present).sum()), int((~e & present).sum())     # errori/non-errori con campo PRESENTE
    rate_abs = a / (a + b) if (a + b) else np.nan
    rate_pres = c / (c + d) if (c + d) else np.nan
    or_ = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
    try:
        _, p = stats.fisher_exact([[a, b], [c, d]]) if min(a, b, c, d) < 5 else stats.chi2_contingency([[a, b], [c, d]])[:2]
    except Exception:
        p = np.nan
    return {"CAMPO": col, "RUOLO": role,
            "PCT_ASSENTE_TRA_ERRORI": round(100 * a / max(ctx.n_errors, 1), 2),
            "PCT_ASSENTE_TRA_NON_ERRORI": round(100 * b / max(ctx.n_rows - ctx.n_errors, 1), 2),
            "TASSO_ERRORE_SE_ASSENTE_PCT": round(100 * rate_abs, 3) if not np.isnan(rate_abs) else np.nan,
            "TASSO_ERRORE_SE_PRESENTE_PCT": round(100 * rate_pres, 3) if not np.isnan(rate_pres) else np.nan,
            "ODDS_RATIO_ASSENZA": round(or_, 2), "P_VALUE": p,
            "N_ERRORI_ASSENTE": a, "N_ERRORI_PRESENTE": c}


def _mean(mask: pd.Series) -> float:
    """Media di una maschera, 0 se vuota (es. tutte le righe sono target)."""
    return float(mask.mean()) if len(mask) else 0.0


def analyze_presence(ctx: AnalysisContext) -> pd.DataFrame:
    """Quanto l'assenza di un dato è associata al caso target.

    Si guardano i dati che vengono *raccolti*: le colonne con molti valori distinti che non sono
    misure (identificativi, token, coordinate, testi: di questi non conta il valore ma se ci sono),
    più l'identificativo di sessione e la regola applicata. Le colonne con pochi valori sono già
    analizzate come dimensioni (il loro "(NULL)" è un valore come gli altri, fasi 3 e 4) e come
    passaggi facoltativi; nelle misure un vuoto è quasi sempre "non applicabile".

    Le colonne del sistema analizzato descrivono l'esito e restano fuori, tranne l'identificativo di
    sessione: è un dato che il sistema riceve, e la sua assenza è proprio ciò che l'ipotesi H5 cerca.
    Le colonne degli altri sistemi sono confrontate in :func:`analyze_systems` (lì un vuoto è un
    co-fallimento, non un dato mancante); entrano qui solo se il target non è l'esito di un sistema."""
    derived_feature_frame(ctx)                     # stabilisce quali colonne numeriche sono misure
    measures = set(ctx.meta.get("binned_cols", []))
    session, rule = ctx.session_col, ctx.cols.get("regola")
    skip = (set(ctx.target_group_cols) - {session}) | {ctx.target_col} | {ctx.cols.get(k) for k in TIME_ROLES}
    systems = set(ctx.other_system_cols)
    rows = []
    for c in ctx.df.columns:
        if c in skip:
            continue
        if c in systems:
            if ctx.is_system_target:
                continue
            role = ROLE_SESSION if re.search(SESSION_RX, c, re.IGNORECASE) else ROLE_SYSTEM
        elif c == session or re.search(SESSION_RX, c, re.IGNORECASE):
            role = ROLE_SESSION
        elif c == rule:
            role = ROLE_RULE
        elif ctx.nunique(c) > ctx.max_cardinality and c not in measures:
            role = ROLE_DATA
        else:
            continue
        share = float(ctx.isnull(c).mean())
        if not 0 < share < 1:                      # mai assente, oppure sempre: nessun confronto possibile
            continue
        rows.append(_presence_stats(ctx, c, role))
    out = pd.DataFrame(rows).sort_values("ODDS_RATIO_ASSENZA", ascending=False) if rows else pd.DataFrame()
    ctx.add_table("06_presenza_campi", out)
    E = ctx.system_label
    leak = set(ctx.meta.get("leakage_cols", []))
    n_found = 0
    for r in out.itertuples():
        if not (r.ODDS_RATIO_ASSENZA >= 3 and r.PCT_ASSENTE_TRA_ERRORI >= 20) or r.CAMPO in leak or n_found >= MAX_PRESENCE_FINDINGS:
            continue
        n_found += 1
        own = r.CAMPO in ctx.target_group_cols     # campo dello stesso sistema: l'assenza può essere causa o effetto
        label = (SESSION_TITLE if r.RUOLO == ROLE_SESSION else "Regola applicata" if r.RUOLO == ROLE_RULE
                 else "Esito di un altro sistema" if r.RUOLO == ROLE_SYSTEM else "Campo")
        ctx.add_finding(Finding("FASE 6", "ALIMENTAZIONE DATI",
                                f"{label} {r.CAMPO} mancante nel {r.PCT_ASSENTE_TRA_ERRORI}% degli errori vs {r.PCT_ASSENTE_TRA_NON_ERRORI}% delle altre righe",
                                f"odds ratio={r.ODDS_RATIO_ASSENZA}, tasso errore se assente={r.TASSO_ERRORE_SE_ASSENTE_PCT}% vs presente={r.TASSO_ERRORE_SE_PRESENTE_PCT}%",
                                "MEDIA" if own else "ALTA",
                                f"Verificare sui log se {r.CAMPO} manca PRIMA dell'elaborazione (sessione mai aperta o scaduta: causa) "
                                f"o se non viene valorizzato perché {E} è andato in errore (effetto)." if own else
                                f"Verificare la raccolta di {r.CAMPO} a monte: {E} non può elaborare richieste prive di questo dato." if ctx.is_system_target
                                else f"Verificare la raccolta di {r.CAMPO}: la sua assenza è fortemente associata a {ctx.event_label}."))
    return out


def analyze_systems(ctx: AnalysisContext) -> pd.DataFrame:
    """Relazione tra il caso target e gli esiti degli altri sistemi presenti nel tracciato."""
    rows = []
    E = ctx.system_label
    for system, cols in ctx.other_systems.items():
        for c in cols:
            s = ctx.df[c]
            null = ctx.isnull(c)
            sys_err = contains_token(s, FAILURE_RX)
            # tasso di nulli/errori del sistema tra le righe target vs resto
            r = {"SISTEMA": system, "COLONNA": c,
                 "PCT_NULL_TRA_TARGET": round(100 * null[ctx.is_error].mean(), 2),
                 "PCT_NULL_TRA_RESTO": round(100 * _mean(null[~ctx.is_error]), 2),
                 "PCT_ERR_SISTEMA_TRA_TARGET": round(100 * sys_err[ctx.is_error].mean(), 2),
                 "PCT_ERR_SISTEMA_TRA_RESTO": round(100 * _mean(sys_err[~ctx.is_error]), 2)}
            if ctx.nunique(c) <= ctx.max_cardinality:
                top_err = s[ctx.is_error].value_counts(normalize=True, dropna=False).head(3)
                r["TOP_VALORI_TRA_TARGET"] = "; ".join(f"{k}={100*v:.1f}%" for k, v in top_err.items())
                cs = pd.crosstab(s.where(~null, "(NULL)"), ctx.is_error)
                if cs.shape[0] > 1 and cs.shape[1] > 1:
                    chi2, p, _, _ = stats.chi2_contingency(cs)
                    r["CRAMERS_V"] = round(float(np.sqrt(chi2 / (cs.values.sum() * (min(cs.shape) - 1)))), 4)
            rows.append(r)
    out = pd.DataFrame(rows)
    ctx.add_table("06_relazione_sistemi", out)
    if out.empty:
        return out
    # Deduzione: co-fallimento (problema a monte) vs fallimento isolato (problema del sistema target)
    co_fail = out[(out["PCT_NULL_TRA_TARGET"] - out["PCT_NULL_TRA_RESTO"] >= 30) |
                  (out["PCT_ERR_SISTEMA_TRA_TARGET"] - out["PCT_ERR_SISTEMA_TRA_RESTO"] >= 30)]
    if not co_fail.empty:
        ctx.add_finding(Finding("FASE 6", "INTEGRAZIONE", f"Co-fallimento con altri sistemi: {', '.join(co_fail['SISTEMA'].unique())}",
                                "; ".join(f"{r.COLONNA}: null {r.PCT_NULL_TRA_TARGET}% vs {r.PCT_NULL_TRA_RESTO}%" for r in co_fail.itertuples()),
                                "ALTA", f"Anche gli altri sistemi non elaborano le stesse righe: la causa è a monte (orchestrazione / arricchimento dati), non specifica di {E}."))
    else:
        ctx.add_finding(Finding("FASE 6", "INTEGRAZIONE", f"Gli altri sistemi elaborano regolarmente le righe {ctx.event_label}",
                                "Nessun sistema mostra un incremento >=30 p.p. di nulli/errori sulle stesse righe",
                                "ALTA", f"Concentrare l'analisi sul canale verso {E} (libreria client, raccolta dati, API, configurazione): i componenti a monte e gli altri sistemi elaborano correttamente le stesse righe."
                                if ctx.is_system_target else f"Il caso {ctx.event_label} non è accompagnato da anomalie negli esiti degli altri sistemi."))
    return out


def classify_detail(ctx: AnalysisContext) -> pd.DataFrame:
    """Classifica la colonna di dettaglio del sistema target (solo righe target) in cause tecniche via tassonomia a parole chiave."""
    c = ctx.detail_col
    if not c:
        return pd.DataFrame()
    det = ctx.errors[c]

    def classify(v: str) -> str:
        up = str(v).upper()
        for name, kws in DETAIL_TAXONOMY.items():
            if any(k in up for k in kws):
                return name
        return "ALTRO / NON CLASSIFICATO"

    # si classificano i soli messaggi distinti (pochi rispetto alle righe), poi si riportano sulle righe
    mapping = {v: classify(v) for v in det.dropna().unique()}
    cat_s = det.map(mapping).astype(object).where(det.notna(), "(NULL)")
    ctx.meta["detail_category"] = cat_s
    tab = cat_s.value_counts().rename_axis("CAUSA_TECNICA_INFERITA").reset_index(name="N_ERRORI")
    tab["PCT"] = (100 * tab["N_ERRORI"] / max(ctx.n_errors, 1)).round(2)
    ctx.add_table("06_detail_classificazione", tab)
    vals = ctx.errors[c].value_counts(dropna=False).head(50).rename_axis(c).reset_index(name="N_ERRORI")
    vals["PCT"] = (100 * vals["N_ERRORI"] / max(ctx.n_errors, 1)).round(2)
    ctx.add_table("06_detail_valori", vals)
    if not tab.empty:
        top = tab.iloc[0]
        ctx.add_finding(Finding("FASE 6", "CAUSA TECNICA", f"Causa tecnica prevalente in {c}: {top['CAUSA_TECNICA_INFERITA']} ({top['PCT']}%)",
                                "; ".join(f"{r.CAUSA_TECNICA_INFERITA}={r.PCT}%" for r in tab.head(5).itertuples()),
                                "ALTA" if top["CAUSA_TECNICA_INFERITA"] not in ("(NULL)", "ALTRO / NON CLASSIFICATO") else "BASSA"))
        if top["CAUSA_TECNICA_INFERITA"] == "(NULL)" and top["PCT"] >= 50:
            ctx.add_finding(Finding("FASE 6", "OSSERVABILITA'", f"{c} vuoto nella maggioranza delle righe target",
                                    f"{top['PCT']}% degli errori senza dettaglio", "ALTA",
                                    f"Richiedere a chi gestisce {ctx.system_label} e la sua integrazione di propagare il codice errore nella risposta: senza dettaglio la root cause non è tracciabile."))
    return tab


def _family_key(value: object) -> str:
    """Prima parola del valore, in maiuscolo: 'ORDINE_ESPRESSO' -> 'ORDINE', 'web-mobile' -> 'WEB'.
    Un valore che inizia con un numero ('12.5', '2026-09') fa famiglia a sé."""
    text = str(value).strip().upper()
    m = re.match(r"[^\W_]+", text)
    return m.group(0) if m and not m.group(0).isdigit() else text


def _is_outcome_name(col: str) -> bool:
    """Il nome è quello di un esito (ESITO_FINALE, STATO, DECISIONE): non è una dimensione descrittiva."""
    return any(t in OUTCOME_WORDS for t in name_tokens(col))


def _declared_families(ctx: AnalysisContext, dims: List[str]) -> List[dict]:
    """Famiglie dichiarate con --family: per ogni dimensione (o per la sola colonna indicata) i valori
    che corrispondono alla regola, analizzati insieme. Il confronto ignora maiuscole e minuscole."""
    err = ctx.is_error.to_numpy()
    by_upper = {c.strip().upper(): c for c in ctx.df.columns}
    rows = []
    for col, name, rx_s in ctx.families:
        if col is not None and str(col).strip().upper() not in by_upper:
            LOG.warning("Famiglia '%s': colonna '%s' non presente nel CSV, ignorata", name, col)
            continue
        rx = re.compile(rx_s, re.IGNORECASE)
        for c in ([by_upper[str(col).strip().upper()]] if col is not None else dims):
            if c == ctx.target_col:
                continue
            s = ctx.df[c]
            members = [v for v in s.dropna().unique() if rx.search(str(v))]
            if not members:
                continue
            m = s.isin(members).to_numpy()
            n, e = int(m.sum()), int((m & err).sum())
            rate = e / n
            rows.append({"DIMENSIONE": c, "FAMIGLIA": name, "N_VALORI": len(members), "N": n, "N_ERROR": e,
                         "TASSO_ERRORE_PCT": round(100 * rate, 3),
                         "LIFT": round(rate / ctx.error_rate, 2) if ctx.error_rate else np.nan,
                         "QUOTA_ERRORI_PCT": round(100 * e / max(ctx.n_errors, 1), 2),
                         "VALORI": ", ".join(str(v) for v in members[:6]), "ORIGINE": "dichiarata"})
    return rows


def analyze_value_families(ctx: AnalysisContext) -> pd.DataFrame:
    """Famiglie di valori delle dimensioni descrittive.

    I valori di una dimensione che iniziano con la stessa parola (``ORDINE_STANDARD``,
    ``ORDINE_ESPRESSO``) formano una famiglia: analizzarli insieme fa emergere un problema che,
    diviso tra più valori, resterebbe sotto soglia. Altre famiglie si dichiarano con ``--family``
    (valori che corrispondono a una regola). La tabella riporta le famiglie dichiarate, quelle
    automatiche con più di un valore e i singoli valori in cui il caso target è sovra-rappresentato."""
    dims = [c for c in key_dimensions(ctx) if not _is_outcome_name(c)][:MAX_FAMILY_DIMS]
    err = ctx.is_error.to_numpy()
    rows = _declared_families(ctx, dims)
    declared = {(r["DIMENSIONE"], r["VALORI"]) for r in rows}
    for c in dims:
        s = ctx.df[c]
        codes, uniques = pd.factorize(s.to_numpy(dtype=object))       # i nulli hanno codice -1
        if len(uniques) < 2:
            continue
        fam_of = np.array([_family_key(u) for u in uniques], dtype=object)
        fam_codes, fams = pd.factorize(fam_of)
        if len(fams) < 2:                                              # una sola famiglia: è l'intera colonna
            continue
        valid = codes >= 0
        row_fam = fam_codes[codes[valid]]
        n = np.bincount(row_fam, minlength=len(fams))
        e = np.bincount(row_fam[err[valid]], minlength=len(fams))
        n_values = np.bincount(fam_codes, minlength=len(fams))
        for i, fam in enumerate(fams):
            if n[i] == 0:
                continue
            rate = e[i] / n[i]
            lift = rate / ctx.error_rate if ctx.error_rate else np.nan
            hot = bool(lift >= 2 and e[i] >= 10)
            if n_values[i] < 2 and not hot:                           # il singolo valore "normale" è già nella fase 3
                continue
            members = [str(u) for u in uniques[fam_codes == i]]
            if (c, ", ".join(members[:6])) in declared:               # stessi valori di una famiglia dichiarata
                continue
            rows.append({"DIMENSIONE": c, "FAMIGLIA": f"{fam}*" if n_values[i] > 1 else members[0], "N_VALORI": int(n_values[i]),
                         "N": int(n[i]), "N_ERROR": int(e[i]), "TASSO_ERRORE_PCT": round(100 * rate, 3),
                         "LIFT": round(lift, 2) if ctx.error_rate else np.nan,
                         "QUOTA_ERRORI_PCT": round(100 * e[i] / max(ctx.n_errors, 1), 2),
                         "VALORI": ", ".join(members[:6]), "ORIGINE": "automatica"})
    out = pd.DataFrame(rows).sort_values("N_ERROR", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()
    ctx.add_table("06_famiglie_valori", out)
    hot = out[(out["LIFT"] >= 2) & (out["N_ERROR"] >= 10)] if not out.empty else out
    if not hot.empty:
        names = [f"{r.DIMENSIONE}={r.FAMIGLIA}" for r in hot.head(6).itertuples()]
        ctx.add_finding(Finding("FASE 6", "CATEGORIE", "Valori e famiglie di valori con errore sovra-rappresentato: " + ", ".join(names),
                                "; ".join(f"{r.DIMENSIONE}={r.FAMIGLIA}: tasso {r.TASSO_ERRORE_PCT}% lift {r.LIFT} ({r.QUOTA_ERRORI_PCT}% degli errori)" for r in hot.head(6).itertuples()),
                                "ALTA", f"Categorie che seguono un percorso diverso da quello standard: spesso non sono integrate con {ctx.system_label}, non sono configurate o inviano dati incompleti."
                                if ctx.is_system_target else f"Categorie in cui {ctx.event_label} è sovra-rappresentato: verificare se il comportamento è atteso."))
    return out


def analyze_optional_steps(ctx: AnalysisContext) -> Optional[pd.DataFrame]:
    """Caso target con e senza un passaggio facoltativo.

    Una colonna categorica in cui una parte delle righe ha un valore assente o negativo (``NONE``,
    ``NO``, ``SKIPPED``...) descrive un passaggio che non sempre viene eseguito. Se il caso target
    si concentra dove il passaggio manca, l'esito può dipendere dal suo completamento."""
    skip = {ctx.target_col} | set(ctx.target_group_cols) | set(ctx.other_system_cols) | set(ctx.meta.get("leakage_cols", []))
    rows = []
    for c in ctx.df.columns:
        if c in skip or not 1 <= ctx.nunique(c) <= ctx.max_cardinality:
            continue
        s = ctx.df[c].astype("string").str.strip().str.upper()
        done = ~ctx.isnull(c) & ~s.isin(NEGATIVE_TOKENS).fillna(False).astype(bool)
        share = float(done.mean())
        if not 0.01 <= share <= 0.99:              # il passaggio c'è (o manca) praticamente sempre
            continue
        d, e = done.to_numpy(), ctx.is_error.to_numpy()
        for label, m in (("CON PASSAGGIO", d), ("SENZA PASSAGGIO", ~d)):
            n, ne = int(m.sum()), int((m & e).sum())
            rows.append({"COLONNA": c, "PASSAGGIO": label, "N": n, "N_ERROR": ne,
                         "TASSO_ERRORE_PCT": round(100 * ne / max(n, 1), 3),
                         "QUOTA_ERRORI_PCT": round(100 * ne / max(ctx.n_errors, 1), 2)})
    if not rows:
        return None
    g = pd.DataFrame(rows)
    ctx.add_table("06_passaggi_facoltativi", g)
    # colonne in cui il caso si concentra dove il passaggio manca: la più netta diventa un finding
    con = g[g["PASSAGGIO"] == "CON PASSAGGIO"].set_index("COLONNA")
    senza = g[g["PASSAGGIO"] == "SENZA PASSAGGIO"].set_index("COLONNA")
    hit = senza[(senza["TASSO_ERRORE_PCT"] >= 2 * con["TASSO_ERRORE_PCT"].clip(lower=0.01)) & (senza["QUOTA_ERRORI_PCT"] >= 50)]
    if not hit.empty:
        c = hit["QUOTA_ERRORI_PCT"].idxmax()
        also = [x for x in hit.index if x != c]
        ctx.add_finding(Finding("FASE 6", "PASSAGGIO FACOLTATIVO", f"Le righe senza {c} concentrano il {senza.loc[c, 'QUOTA_ERRORI_PCT']}% degli errori",
                                f"tasso senza {c} {senza.loc[c, 'TASSO_ERRORE_PCT']}% vs con {c} {con.loc[c, 'TASSO_ERRORE_PCT']}%"
                                + (f"; stesso comportamento per {', '.join(also[:3])}" if also else ""),
                                "MEDIA", f"L'elaborazione di {ctx.system_label} potrebbe concludersi solo al completamento del passaggio {c}: nei flussi che lo saltano l'esito non viene prodotto."
                                if ctx.is_system_target else f"{ctx.event_label} si concentra nei flussi senza il passaggio {c}."))
    return g


def behaviour_summary(ctx: AnalysisContext) -> List[str]:
    """Frasi conclusive 'Le righe del caso analizzato risultano prevalentemente associate a...'."""
    bullets: List[str] = []
    pres = ctx.results.get("06_presenza_campi", pd.DataFrame())
    leak = set(ctx.meta.get("leakage_cols", []))
    for r in pres.itertuples() if not pres.empty else []:
        if r.CAMPO in leak:
            continue
        if r.PCT_ASSENTE_TRA_ERRORI >= 50 and r.ODDS_RATIO_ASSENZA >= 2:
            bullets.append(f"{r.CAMPO} mancante ({r.PCT_ASSENTE_TRA_ERRORI}% degli errori)")
        elif r.ODDS_RATIO_ASSENZA >= 3 and r.PCT_ASSENTE_TRA_ERRORI >= 20 and r.N_ERRORI_PRESENTE > 0:
            bullets.append(f"{r.CAMPO} mancante in parte dei casi ({r.PCT_ASSENTE_TRA_ERRORI}% degli errori contro {r.PCT_ASSENTE_TRA_NON_ERRORI}% delle altre righe); "
                           f"i restanti {r.N_ERRORI_PRESENTE:,} errori hanno il campo valorizzato")
    fam = ctx.results.get("06_famiglie_valori", pd.DataFrame())
    hot = fam[(fam["LIFT"] >= 2) & (fam["N_ERROR"] >= 10)] if not fam.empty else fam
    for r in hot.head(5).itertuples():
        bullets.append(f"{r.DIMENSIONE} = {r.FAMIGLIA} (tasso {r.TASSO_ERRORE_PCT}%, lift {r.LIFT})")
    steps = ctx.results.get("06_passaggi_facoltativi", pd.DataFrame())
    if not steps.empty:
        # conta dove il caso è sovra-rappresentato: una colonna quasi sempre vuota raccoglie comunque quasi tutti i casi
        con = steps[steps["PASSAGGIO"] == "CON PASSAGGIO"].set_index("COLONNA")
        senza = steps[steps["PASSAGGIO"] == "SENZA PASSAGGIO"].set_index("COLONNA")
        s = senza[(senza["QUOTA_ERRORI_PCT"] >= 60) & (senza["TASSO_ERRORE_PCT"] >= 2 * con["TASSO_ERRORE_PCT"].clip(lower=0.01))]
        for c, r in s.sort_values("QUOTA_ERRORI_PCT", ascending=False).head(2).iterrows():
            bullets.append(f"Righe senza il passaggio {c} ({r['QUOTA_ERRORI_PCT']}% degli errori)")
    det = ctx.results.get("06_detail_classificazione", pd.DataFrame())
    for r in det.head(3).itertuples() if not det.empty else []:
        if r.CAUSA_TECNICA_INFERITA not in ("(NULL)", "ALTRO / NON CLASSIFICATO") and r.PCT >= 10:
            bullets.append(f"Dettaglio {ctx.system_label} riconducibile a '{r.CAUSA_TECNICA_INFERITA}' ({r.PCT}%)")
    dims = ctx.results.get("03_ranking_dimensioni", pd.DataFrame())
    for r in dims.head(4).itertuples() if not dims.empty else []:
        if r.QUOTA_ERRORI_TOP_PCT and r.QUOTA_ERRORI_TOP_PCT >= 40:
            bullets.append(f"{r.DIMENSIONE} = {r.VALORE_CON_PIU_ERRORI} ({r.QUOTA_ERRORI_TOP_PCT}% degli errori)")
    ctx.meta["behaviour_bullets"] = bullets
    return bullets


def run_functional(ctx: AnalysisContext) -> None:
    for fn in (analyze_presence, analyze_systems, classify_detail, analyze_value_families, analyze_optional_steps, behaviour_summary):
        try:
            fn(ctx)
        except Exception as e:
            LOG.exception("Blocco funzionale %s fallito: %s", fn.__name__, e)
