#!/usr/bin/env python
"""
Genera un CSV sintetico per collaudare la pipeline senza dati reali: richieste a una piattaforma,
ciascuna valutata da più sistemi (ALFA, BETA, GAMMA, DELTA, EPSILON, ZETA) che scrivono il proprio
esito in colonne dedicate. Il caso da ritrovare è ``LIVELLO_ALFA = ERROR``, con cause "nascoste".

Pattern iniettati (da ritrovare nell'analisi):
- APP=412 + TIPO_RICHIESTA in {RESO, PRENOTAZIONE, BUONO_REGALO}: errore ~85% (flusso non integrato);
- TOKEN_CLIENT assente: errore ~40% (dati in ingresso incompleti);
- degrado infrastrutturale: 2 ore del 2026-09-14 con errore ~70% su tutto;
- ATTIVAZIONE_SERVIZIO senza PASSO_CONFERMA: errore ~30%.
"""
import argparse

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)


def main(n: int, out: str) -> None:
    start = pd.Timestamp("2026-09-01")
    ts = start + pd.to_timedelta(rng.integers(0, 30 * 24 * 3600, n), unit="s")
    # distribuzione oraria realistica
    hour_w = np.array([1, 1, 1, 1, 1, 2, 4, 7, 10, 12, 12, 11, 9, 9, 10, 10, 9, 8, 7, 6, 5, 4, 2, 1], float)
    hours = rng.choice(24, n, p=hour_w / hour_w.sum())
    ts = pd.DatetimeIndex(ts).normalize() + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(rng.integers(0, 3600, n), unit="s")

    tipi = ["ORDINE_STANDARD", "CONSEGNA_ESPRESSA", "RESO", "PRENOTAZIONE", "BUONO_REGALO", "ATTIVAZIONE_SERVIZIO", "RINNOVO_ABBONAMENTO",
            "DISDETTA", "PREVENTIVO", "ORDINE_ESTERO"]
    p_w = np.array([30, 15, 8, 4, 4, 3, 10, 8, 10, 8], float)
    tipo = rng.choice(tipi, n, p=p_w / p_w.sum())
    app = rng.choice(["101", "205", "412", "530"], n, p=[0.45, 0.25, 0.2, 0.1])
    canale = np.where(app == "101", "WEB", np.where(app == "205", "MOBILE_IOS", np.where(app == "412", "MOBILE_ANDROID", "TOTEM")))
    sede = rng.choice(["SEDE_A", "SEDE_B", "SEDE_C"], n, p=[0.6, 0.3, 0.1])
    segmento = rng.choice(["BASE", "PLUS", "PREMIUM", "PMI", "GRANDI_CLIENTI"], n, p=[0.55, 0.2, 0.05, 0.15, 0.05])
    importo = np.round(np.exp(rng.normal(5, 1.4, n)), 2)
    conferma = rng.choice(["SMS", "APP_MOBILE", "NONE"], n, p=[0.45, 0.35, 0.2])
    conferma = np.where(tipo == "ATTIVAZIONE_SERVIZIO", rng.choice(["NONE", "SMS"], n, p=[0.7, 0.3]), conferma)
    modalita = np.where(conferma == "NONE", "DIRETTA", "CON_CONFERMA")
    flusso = np.array([f"FLW_{t[:4]}_{a}" for t, a in zip(tipo, app)])
    tk_present = rng.random(n) > 0.08
    tk_present &= ~((canale == "TOTEM") & (rng.random(n) < 0.6))
    token = np.where(tk_present, [f"tk{x:012x}" for x in rng.integers(0, 2**48, n)], None)
    dispositivo = np.where(rng.random(n) > 0.05, [f"dev{x:010x}" for x in rng.integers(0, 2**40, n)], None)
    utente = rng.integers(1_000_000, 1_400_000, n).astype(str)
    lat = np.round(rng.normal(43.5, 2.5, n), 5)
    lon = np.round(rng.normal(11.5, 2.0, n), 5)
    paese = rng.choice(["IT", "IT", "IT", "IT", "DE", "FR", "ES", "RO", "GB"], n)
    geo_null = rng.random(n) < 0.1
    lat = np.where(geo_null, np.nan, lat)
    lon = np.where(geo_null, np.nan, lon)

    # probabilità che il sistema ALFA vada in errore
    p = np.full(n, 0.012)
    p = np.where((app == "412") & np.isin(tipo, ["RESO", "PRENOTAZIONE", "BUONO_REGALO"]), 0.85, p)
    p = np.where(~tk_present, np.maximum(p, 0.40), p)
    p = np.where((tipo == "ATTIVAZIONE_SERVIZIO") & (conferma == "NONE"), np.maximum(p, 0.30), p)
    burst = (ts >= pd.Timestamp("2026-09-14 10:00")) & (ts < pd.Timestamp("2026-09-14 12:00"))
    p = np.where(burst, 0.70, p)
    is_err = rng.random(n) < p

    # dettaglio di ALFA coerente con la causa
    detail = np.where(burst & is_err, "HTTP 504 Gateway Timeout from upstream service",
             np.where(~tk_present & is_err, "MISSING_CLIENT_TOKEN: client telemetry not received",
             np.where((app == "412") & is_err, rng.choice(["SESSION_NOT_FOUND", "INVALID_PAYLOAD: eventType unsupported", None], n, p=[0.5, 0.3, 0.2]),
             np.where(is_err, rng.choice(["INTERNAL_EXCEPTION NullPointerException", "SESSION_EXPIRED", None], n, p=[0.4, 0.3, 0.3]), None))))
    livello = np.where(is_err, "ERROR", rng.choice(["LOW", "MEDIUM", "HIGH"], n, p=[0.85, 0.12, 0.03]))
    score_alfa = np.where(is_err, np.nan, np.round(rng.beta(1.2, 8, n) * 100, 1))
    sessione = np.where(is_err & (rng.random(n) < 0.5), None, [f"s{x:016x}" for x in rng.integers(0, 2**60, n)])
    regola = np.where(livello == "HIGH", rng.choice(["R_NUOVO_DISPOSITIVO", "R_FREQUENZA", "R_ORARIO_INSOLITO", "R_IMPORTO_ELEVATO"], n),
                      np.where(livello == "MEDIUM", rng.choice(["R_NUOVO_DESTINATARIO", "R_PICCO_IMPORTO"], n), None))
    gamma_score = np.round(rng.beta(1.5, 6, n) * 1000).astype(int)
    gamma_azione = np.where(gamma_score > 850, "DENY", np.where(gamma_score > 600, "VERIFY", "ALLOW"))  # la soglia più alta va provata per prima
    zeta_score = np.round(rng.beta(1.3, 9, n), 4)
    zeta_dec = np.where(zeta_score > 0.6, "DECLINE", np.where(zeta_score > 0.3, "REVIEW", "APPROVE"))
    delta = np.where(geo_null, None, np.where(paese != "IT", rng.choice(["GEO_YELLOW", "GEO_RED"], n), "GEO_GREEN"))
    epsilon_score = np.round(rng.beta(1.2, 10, n), 4)
    epsilon_esito = np.where(epsilon_score > 0.5, "ALERT", "OK")
    beta = rng.choice(["OK", "WARN", "KO"], n, p=[0.9, 0.08, 0.02])
    decisione = np.where((gamma_azione == "DENY") | (zeta_dec == "DECLINE"), "DENY",
                np.where((gamma_azione == "VERIFY") | (livello == "HIGH") | (zeta_dec == "REVIEW"), "VERIFY", "ALLOW"))
    esito = np.where(decisione == "DENY", "RIFIUTATA", np.where(rng.random(n) < 0.97, "ESEGUITA", "ANNULLATA"))

    df = pd.DataFrame({
        "ID_RICHIESTA": [f"RQ{1_000_000 + i}" for i in range(n)],
        "DATETIME": ts.strftime("%Y-%m-%d %H:%M:%S"), "TIME": ts.strftime("%H:%M:%S"), "TIME_KEY": ts.strftime("%Y%m%d%H"),
        "ID_UTENTE": utente, "APP": app, "TIPO_RICHIESTA": tipo, "CANALE": canale, "SEDE": sede, "SEGMENTO_UTENTE": segmento,
        "IMPORTO": importo, "ESITO_FINALE": esito, "MODALITA": modalita, "PASSO_CONFERMA": conferma, "FLUSSO": flusso,
        "REGOLA": regola, "DECISIONE": decisione,
        "LIVELLO_ALFA": livello, "DETTAGLIO_ALFA": detail, "SCORE_ALFA": score_alfa, "SESSIONE_ALFA": sessione,
        "TOKEN_CLIENT": token, "ID_DISPOSITIVO": dispositivo, "LAT": lat, "LON": lon, "PAESE": paese,
        "ESITO_BETA": beta, "GAMMA_SCORE": gamma_score, "GAMMA_AZIONE": gamma_azione, "DELTA_ESITO": delta,
        "EPSILON_SCORE": epsilon_score, "EPSILON_ESITO": epsilon_esito, "ZETA_SCORE": zeta_score, "ZETA_DECISIONE": zeta_dec,
    })
    df.to_csv(out, index=False, sep=";")
    print(f"Scritto {out}: {len(df):,} righe, di cui LIVELLO_ALFA=ERROR {is_err.sum():,} ({is_err.mean():.2%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=40_000)
    ap.add_argument("--out", default="sample_dati.csv")
    a = ap.parse_args()
    main(a.rows, a.out)
