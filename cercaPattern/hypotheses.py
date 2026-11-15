"""
Motore deduttivo: dalle evidenze raccolte nelle fasi 1-8 alle *cause probabili*.

Ogni ipotesi è una classe di root-cause tipica dell'integrazione con un
sistema esterno che deve elaborare ogni riga; il punteggio (0..1) cresce con il numero e la
forza delle evidenze a supporto. Le regole sono esplicite e commentate così
che un analista possa verificarle e tararle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from .config import AnalysisContext, LOG
from .functional import ROLE_RULE, ROLE_SESSION, SESSION_TITLE


@dataclass
class Hypothesis:
    codice: str
    tipo: str              # INTEGRAZIONE APPLICATIVA / ALIMENTAZIONE DATI / INFRASTRUTTURA / CONFIGURAZIONE / FUNZIONALE / A MONTE
    causa: str
    score: float = 0.0
    evidenze: List[str] = field(default_factory=list)
    raccomandazione: str = ""

    def add(self, w: float, ev: str) -> None:
        self.score = min(1.0, self.score + w)
        self.evidenze.append(ev)

    @property
    def confidenza(self) -> str:
        return "ALTA" if self.score >= 0.7 else "MEDIA" if self.score >= 0.4 else "BASSA"


def _detail_share(ctx: AnalysisContext, *cats: str) -> float:
    t = ctx.results.get("06_detail_classificazione", pd.DataFrame())
    if t.empty:
        return 0.0
    return float(t[t["CAUSA_TECNICA_INFERITA"].isin(cats)]["PCT"].sum())


def _window_detail_share(ctx: AnalysisContext, *cats: str) -> Optional[float]:
    """% dei dettagli (classificati in fase 6) delle righe target cadute in una finestra di degrado
    che appartiene alle categorie indicate. None se mancano finestre o dettagli."""
    cat_s = ctx.meta.get("detail_category")
    prof = ctx._cache.get("hourly")
    if cat_s is None or not prof or not prof["in_window"].any():
        return None
    inside = pd.Series(prof["in_window"], index=ctx.df.index).reindex(cat_s.index)
    known = cat_s[inside.to_numpy() & (cat_s != "(NULL)").to_numpy()]
    if len(known) < 5:
        return None
    return float(100 * known.isin(cats).mean())


def build_hypotheses(ctx: AnalysisContext) -> pd.DataFrame:
    E = ctx.system_label
    H = {
        "INFRA": Hypothesis("H1", "INFRASTRUTTURA", f"Indisponibilità / timeout del servizio {E} in finestre temporali (degrado episodico)",
                            raccomandazione=f"Correlare le finestre con incident, deploy, saturazione delle risorse; verificare timeout e circuit breaker lato client {E}."),
        "FEED": Hypothesis("H2", "ALIMENTAZIONE DATI", f"Dati inviati a {E} incompleti: campi non raccolti o non propagati",
                           raccomandazione=f"Verificare la raccolta dei campi mancanti sui flussi impattati e la loro mappatura nella richiesta inviata a {E}."),
        "INTEG": Hypothesis("H3", "INTEGRAZIONE APPLICATIVA", f"Flussi specifici (valori di una o più dimensioni) non integrati correttamente con {E} (errore sistematico)",
                            raccomandazione="Censire i flussi con tasso di errore >= 50% e verificarne l'integrazione end-to-end (endpoint, versione API, formato della richiesta)."),
        "UPSTREAM": Hypothesis("H4", "A MONTE", "Problema in un componente a monte (orchestrazione / arricchimento dati) condiviso da più sistemi",
                               raccomandazione="Analizzare i log dei componenti a monte sulle stesse righe: anche gli altri sistemi non ricevono dati validi."),
        "SESSION": Hypothesis("H5", "FUNZIONALE", f"Sessione o contesto {E} non trovato/scaduto al momento dell'elaborazione",
                              raccomandazione=f"Verificare la durata della sessione {E}, i flussi con lunghi tempi di completamento e i riavvii del client."),
        "CONFIG": Hypothesis("H6", "CONFIGURAZIONE", f"Categoria non configurata o non supportata nelle regole di {E}",
                             raccomandazione=f"Confrontare l'elenco delle categorie con la configurazione delle regole di {E}, dimensione per dimensione."),
        "STEP": Hypothesis("H7", "FUNZIONALE", f"Elaborazione {E} non conclusa nei flussi che saltano un passaggio facoltativo (esito legato al suo completamento)",
                           raccomandazione="Verificare che la richiesta di elaborazione venga chiusa anche quando il passaggio facoltativo non è previsto."),
        "OBS": Hypothesis("H8", "OSSERVABILITA'", "Dettaglio errore non propagato: root cause non diagnosticabile dai dati",
                          raccomandazione=f"Far propagare codice e messaggio di errore {E} nella colonna di dettaglio."),
        "RELEASE": Hypothesis("H9", "RILASCIO / CAMBIO", "Regressione introdotta da un rilascio o da un cambio di configurazione (salto persistente del tasso da una certa data)",
                              raccomandazione="Ricostruire cosa è stato rilasciato o riconfigurato alla data del salto (applicazioni, librerie, regole, instradamento) e valutare il rollback."),
    }
    findings = ctx.findings
    cat = lambda c: [f for f in findings if f.categoria == c]

    # --- H1 infrastruttura ---------------------------------------------------
    # contano le finestre trasversali a tutte le dimensioni: un degrado confinato a un solo flusso
    # non è un problema dell'infrastruttura comune (va all'ipotesi H3)
    share = ctx.meta.get("degrado_share_trasversale", ctx.meta.get("degrado_share", 0.0))
    if share >= 30:
        H["INFRA"].add(0.6, f"Le finestre di degrado trasversali contengono il {share:.1f}% degli errori")
    elif share > 0:
        H["INFRA"].add(0.25, f"Finestre di degrado trasversali presenti ma contengono solo il {share:.1f}% degli errori")
    d = _detail_share(ctx, "TIMEOUT / LATENZA", "CONNETTIVITA / HTTP 5xx")
    if d >= 10:
        H["INFRA"].add(0.3 if d >= 40 else 0.15, f"{d:.1f}% dei dettagli classificati come timeout/connettività")
    w = _window_detail_share(ctx, "TIMEOUT / LATENZA", "CONNETTIVITA / HTTP 5xx")
    if w is not None and w >= 50:
        H["INFRA"].add(0.2, f"Dentro le finestre di degrado il {w:.0f}% dei dettagli è timeout/connettività")

    # --- H2 alimentazione dati ---------------------------------------------
    for f in cat("ALIMENTAZIONE DATI"):
        if not f.titolo.startswith(SESSION_TITLE):    # la sessione mancante è l'ipotesi H5
            H["FEED"].add(0.35, f.titolo)
    d = _detail_share(ctx, "DATI MANCANTI / PAYLOAD", "CLIENT / TELEMETRIA")
    if d >= 10:
        H["FEED"].add(0.3 if d >= 40 else 0.15, f"{d:.1f}% dei dettagli classificati come dati mancanti/raccolta lato client")

    # --- H3 integrazione applicativa ----------------------------------------
    for f in cat("TASSO ANOMALO"):
        H["INTEG"].add(0.3, f.titolo)
    for f in cat("PATTERN DETERMINISTICO"):
        H["INTEG"].add(0.3, f.titolo)
    # una concentrazione conta se riguarda una dimensione descrittiva (un flusso), non l'esito di un sistema
    flow_dims = set(ctx.df.columns) - set(ctx.other_system_cols) - set(ctx.target_group_cols)
    for f in cat("CONCENTRAZIONE"):
        if f.titolo.split("=")[0] in flow_dims:
            H["INTEG"].add(0.2, f.titolo)
    for f in cat("CATEGORIE"):
        H["INTEG"].add(0.15, f.titolo)
        H["CONFIG"].add(0.25, f.titolo)
    integ = [f for f in cat("INTEGRAZIONE") if "confinato" in f.titolo or "regolarmente" in f.titolo]
    if integ:
        H["INTEG"].add(0.2, integ[0].titolo)
    for scope in ctx.meta.get("degrado_localizzato", [])[:2]:
        H["INTEG"].add(0.2, f"Finestra di degrado localizzata su {scope}")

    # --- H9 rilascio / cambio di configurazione -------------------------------
    for s in ctx.meta.get("level_shifts", []):
        if s["DIREZIONE"] != "AUMENTO":
            continue
        H["RELEASE"].add(0.6 if s["RAPPORTO"] >= 2 else 0.4,
                         f"Dal {s['DATA_CAMBIO']:%Y-%m-%d} il tasso passa dal {s['TASSO_PRIMA_PCT']:.2f}% al {s['TASSO_DOPO_PCT']:.2f}% e resta alto")
        if s["GUIDATO_DA"] not in ("trasversale", "n/d"):
            H["RELEASE"].add(0.2, f"Il salto è guidato da {s['GUIDATO_DA']}")

    # --- H4 a monte ----------------------------------------------------------
    for f in cat("INTEGRAZIONE"):
        if "Co-fallimento" in f.titolo:
            H["UPSTREAM"].add(0.7, f.titolo)

    # --- H5 sessione ----------------------------------------------------------
    d = _detail_share(ctx, "SESSIONE / CONTESTO")
    if d >= 10:
        H["SESSION"].add(0.4 if d >= 30 else 0.2, f"{d:.1f}% dei dettagli riconducibili a sessione/contesto")
    pres = ctx.results.get("06_presenza_campi", pd.DataFrame())
    if not pres.empty and ctx.is_system_target:
        s = pres[pres["RUOLO"] == ROLE_SESSION]
        if not s.empty and s.iloc[0]["PCT_ASSENTE_TRA_ERRORI"] >= 30 and s.iloc[0]["ODDS_RATIO_ASSENZA"] >= 3:
            pct = s.iloc[0]["PCT_ASSENTE_TRA_ERRORI"]
            H["SESSION"].add(0.4 if pct >= 50 else 0.3, f"{SESSION_TITLE} {s.iloc[0]['CAMPO']} assente nel {pct}% delle righe target contro il {s.iloc[0]['PCT_ASSENTE_TRA_NON_ERRORI']}% delle altre")

    # --- H6 configurazione ----------------------------------------------------
    d = _detail_share(ctx, "CONFIGURAZIONE / REGOLA")
    if d >= 10:
        H["CONFIG"].add(0.4, f"{d:.1f}% dei dettagli riconducibili a configurazione/regola")
    if not pres.empty:
        r = pres[pres["RUOLO"] == ROLE_RULE]
        if not r.empty and r.iloc[0]["PCT_ASSENTE_TRA_ERRORI"] >= 80:
            H["CONFIG"].add(0.15, f"Nessuna regola valorizzata nel {r.iloc[0]['PCT_ASSENTE_TRA_ERRORI']}% degli errori")

    # --- H7 passaggio facoltativo ---------------------------------------------
    for f in cat("PASSAGGIO FACOLTATIVO"):
        H["STEP"].add(0.5, f.titolo)

    if not ctx.is_system_target:
        ctx.meta["hypotheses_note"] = (f"La colonna target {ctx.target_col} non è l'esito di un sistema riconosciuto nel tracciato: le ipotesi seguenti sono "
                                       f"formulate per l'esito di un'elaborazione e vanno lette come indicazioni generiche sul caso {ctx.event_label}.")
    # --- H8 osservabilità -----------------------------------------------------
    for f in cat("OSSERVABILITA'"):
        H["OBS"].add(0.8, f.titolo)

    rows = []
    for h in sorted(H.values(), key=lambda x: x.score, reverse=True):
        rows.append({"CODICE": h.codice, "TIPO": h.tipo, "CAUSA_PROBABILE": h.causa, "SCORE": round(h.score, 2),
                     "CONFIDENZA": h.confidenza, "EVIDENZE": " | ".join(h.evidenze) or "(nessuna evidenza diretta)",
                     "RACCOMANDAZIONE": h.raccomandazione})
    out = pd.DataFrame(rows)
    ctx.add_table("09_cause_probabili", out)
    LOG.info("Cause probabili: %s", "; ".join(f"{r.CODICE}={r.SCORE}" for r in out.itertuples()))
    return out
