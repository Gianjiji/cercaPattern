"""Test end-to-end della pipeline su scenari di robustezza."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import PREFIX, assert_outputs


def test_baseline(base_df, run):
    ctx, out = run(base_df)
    assert ctx.n_errors > 0
    assert_outputs(out)
    assert len(ctx.png_paths) >= 10
    assert not ctx.results["04_association_rules"].empty
    assert not ctx.results["07_cluster"].empty
    assert "08_anomalie_sintesi" in ctx.results
    # conteggi dei pattern esatti su tutte le righe e copertura cumulativa coerente con il totale
    rules = ctx.results["04_association_rules"]
    assert (rules["N_ERRORI"] <= rules["N_RIGHE"]).all() and (rules["CONF_MIN_95_PCT"] <= rules["CONFIDENZA_PCT"]).all()
    cover = ctx.results["04_copertura_cumulativa"]
    assert cover["N_ERRORI_NUOVI"].sum() == ctx.n_errors and cover["N_RIGHE_NUOVE"].sum() == ctx.n_rows
    assert cover["COPERTURA_CUMULATA_PCT"].is_monotonic_increasing
    assert ctx.meta["xgb_auc_kind"] == "hold-out" and ctx.meta["xgb_auc"] > 0.7
    # il riepilogo apre con le conclusioni e conserva i 10 punti
    text = ctx.meta["executive_summary_text"]
    assert "## In breve" in text and "**Causa più probabile**" in text and "**Problemi distinti**" in text
    assert all(f"## {i}. " in text for i in range(1, 11))
    html = (out / f"{PREFIX}_report.html").read_text(encoding="utf-8")
    assert "copertura cumulativa" in html and ctx.meta["report_html_path"].endswith(f"{PREFIX}_report.html")
    assert {"H1", "H9"} <= set(ctx.results["09_cause_probabili"]["CODICE"])


def test_all_rows_target_exits(base_df, run):
    """Se tutte le righe sono nel caso target non c'è nulla con cui confrontarle: messaggio chiaro, non numeri senza senso."""
    df = base_df.copy()
    df["LIVELLO_ALFA"] = "ERROR"
    with pytest.raises(SystemExit, match="Tutte le"):
        run(df)


def test_level_shift_end_to_end(base_df, run):
    """Regressione dal giorno 20 su una sola APP: va riconosciuta come cambio di livello e attribuita a quell'APP."""
    df = base_df.copy()
    rng = np.random.default_rng(17)
    hit = (df["DATETIME"] >= "2026-09-20") & (df["APP"] == "205") & (rng.random(len(df)) < 0.7)
    df.loc[hit, "LIVELLO_ALFA"] = "ERROR"
    for c in ("DETTAGLIO_ALFA", "SCORE_ALFA", "REGOLA"):
        df.loc[hit, c] = ""
    ctx, out = run(df)
    sh = ctx.results["05_cambi_livello"]
    assert len(sh) == 1 and sh.iloc[0]["DIREZIONE"] == "AUMENTO"
    assert abs((sh.iloc[0]["DATA_CAMBIO"] - pd.Timestamp("2026-09-20")).days) <= 1
    assert sh.iloc[0]["GUIDATO_DA"].startswith(("APP=205", "CANALE=MOBILE_IOS"))
    hyp = ctx.results["09_cause_probabili"].set_index("CODICE")
    assert hyp.loc["H9", "SCORE"] >= 0.6
    text = ctx.meta["executive_summary_text"]
    assert "**Nel tempo**" in text and "il tasso sale" in text and "CAMBIO DI LIVELLO" in {f.categoria for f in ctx.findings}


def test_comma_sep_utf8_bom(base_df, run):
    ctx, out = run(base_df, sep=",", encoding="utf-8-sig")
    assert ctx.n_errors == int(base_df["LIVELLO_ALFA"].eq("ERROR").sum())


def test_tab_sep(base_df, run):
    ctx, out = run(base_df, sep="\t")
    assert ctx.n_errors > 0


def test_cp1252_accents(base_df, run):
    df = base_df.copy()
    df["NOTA"] = np.where(df.index % 3 == 0, "Richiesta città – più è", "Ordine à famiglia")
    ctx, out = run(df, encoding="cp1252")
    assert "NOTA" in ctx.df.columns
    assert ctx.n_errors > 0


def test_lowercase_columns(base_df, run):
    df = base_df.copy()
    df.columns = [c.lower() for c in df.columns]
    ctx, out = run(df)                              # il target è indicato in maiuscolo: il nome viene ritrovato lo stesso
    assert ctx.target_col == "livello_alfa"
    assert ctx.cols.get("datetime") == "datetime" and ctx.system_key == "ALFA"
    assert ctx.detail_col == "dettaglio_alfa" and ctx.measure_col == "importo"


def test_alias_columns(base_df, run):
    """Altri modi di scrivere gli stessi nomi: ruoli e sistemi si ritrovano, e il target viene proposto senza indicarlo."""
    df = base_df.rename(columns={"LIVELLO_ALFA": "ALFA_RISK_LEVEL", "DETTAGLIO_ALFA": "ALFA_DETAIL", "SESSIONE_ALFA": "alfaSessionId",
                                 "DATETIME": "TIMESTAMP", "IMPORTO": "AMOUNT", "DECISIONE": "FINAL_ACTION", "REGOLA": "RULE_NAME",
                                 "ESITO_BETA": "Beta Status"})
    ctx, out = run(df, target_col=None, target_value=None)
    c = ctx.cols
    assert (ctx.target_col, ctx.target_values) == ("ALFA_RISK_LEVEL", ["ERROR"]) and ctx.system_key == "ALFA"
    assert ctx.detail_col == "ALFA_DETAIL" and ctx.session_col == "alfaSessionId" and ctx.score_col == "SCORE_ALFA"
    assert c.get("datetime") == "TIMESTAMP" and c.get("regola") == "RULE_NAME"
    assert ctx.measure_col == "AMOUNT" and ctx.decision_col == "FINAL_ACTION"
    assert c.system_of("Beta Status") == "BETA" and c.system_of("ESITO_FINALE") is None


def test_minimal_columns(base_df, run):
    df = base_df[["ID_RICHIESTA", "TIPO_RICHIESTA", "APP", "LIVELLO_ALFA", "DETTAGLIO_ALFA"]]
    ctx, out = run(df)
    assert_outputs(out)


def test_only_target_and_id(base_df, run):
    df = base_df[["ID_RICHIESTA", "LIVELLO_ALFA"]]
    ctx, out = run(df)
    assert_outputs(out)


@pytest.mark.parametrize("fmt", ["%d/%m/%Y %H:%M", "%Y-%m-%dT%H:%M:%S", "%d-%m-%Y %H:%M:%S", "%Y/%m/%d %H:%M:%S"])
def test_datetime_formats(base_df, run, fmt):
    df = base_df.copy()
    df["DATETIME"] = pd.to_datetime(df["DATETIME"]).dt.strftime(fmt)
    ctx, out = run(df)
    assert "05_tasso_per_ora" in ctx.results
    assert "05_serie_giornaliera" in ctx.results


def test_datetime_with_timezone(base_df, run):
    """Date con offset (+00:00) come nei CSV esportati in UTC: non deve fallire il profiling."""
    df = base_df.copy()
    df["DATETIME"] = pd.to_datetime(df["DATETIME"]).dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    ctx, out = run(df)
    assert "05_tasso_per_ora" in ctx.results
    assert "05_serie_giornaliera" in ctx.results
    assert_outputs(out)


def test_datetime_with_mixed_timezones(base_df, run):
    """Offset misti (ora solare/legale) e formato ISO con Z nello stesso file."""
    df = base_df.copy()
    ts = pd.to_datetime(df["DATETIME"])
    df["DATETIME"] = [t.strftime("%Y-%m-%dT%H:%M:%SZ") if i % 3 == 0 else t.strftime("%Y-%m-%d %H:%M:%S") + f"+0{1 + i % 2}:00"
                      for i, t in enumerate(ts)]
    ctx, out = run(df)
    assert "05_tasso_per_ora" in ctx.results


def test_time_key_only(base_df, run):
    df = base_df.drop(columns=["DATETIME", "TIME"])
    ctx, out = run(df)
    assert ctx.meta.get("ts_source") == "TIME_KEY"
    assert "05_serie_giornaliera" in ctx.results


def test_time_only(base_df, run):
    df = base_df.drop(columns=["DATETIME", "TIME_KEY"])
    ctx, out = run(df)
    assert "05_tasso_per_ora" in ctx.results


def test_date_plus_time(base_df, run):
    df = base_df.drop(columns=["TIME_KEY"]).rename(columns={"DATETIME": "DATA"})
    df["DATA"] = pd.to_datetime(df["DATA"]).dt.strftime("%d/%m/%Y")
    ctx, out = run(df)
    assert "+" in ctx.meta.get("ts_source", "")


def test_no_time_columns(base_df, run):
    df = base_df.drop(columns=["DATETIME", "TIME", "TIME_KEY"])
    ctx, out = run(df)
    assert_outputs(out)


def test_single_day(base_df, run):
    df = base_df.copy()
    df["DATETIME"] = "2026-09-14 " + pd.to_datetime(df["DATETIME"]).dt.strftime("%H:%M:%S")
    df["TIME_KEY"] = "20260914" + pd.to_datetime(df["DATETIME"]).dt.strftime("%H")
    ctx, out = run(df)
    assert_outputs(out)


def test_few_errors(base_df, run):
    df = base_df.copy()
    df["LIVELLO_ALFA"] = "LOW"
    df.loc[df.index[:8], "LIVELLO_ALFA"] = "ERROR"
    ctx, out = run(df)
    assert ctx.n_errors == 8
    assert_outputs(out)


def test_single_error(base_df, run):
    df = base_df.copy()
    df["LIVELLO_ALFA"] = "LOW"
    df.loc[df.index[0], "LIVELLO_ALFA"] = "ERROR"
    ctx, out = run(df)
    assert ctx.n_errors == 1


def test_errors_between_10_and_20(base_df, run):
    df = base_df.copy()
    df["LIVELLO_ALFA"] = "LOW"
    df.loc[df.index[:14], "LIVELLO_ALFA"] = "ERROR"
    ctx, out = run(df)
    assert ctx.n_errors == 14


def test_high_error_rate(base_df, run):
    df = base_df.copy()
    df.loc[df.index[:2850], "LIVELLO_ALFA"] = "ERROR"
    ctx, out = run(df)
    assert ctx.error_rate > 0.9


def test_target_case_and_spaces(base_df, run):
    df = base_df.copy()
    df["LIVELLO_ALFA"] = df["LIVELLO_ALFA"].replace({"ERROR": " error "})
    ctx, out = run(df)
    assert ctx.n_errors == int(base_df["LIVELLO_ALFA"].eq("ERROR").sum())


def test_custom_target_value(base_df, run):
    df = base_df.copy()
    df["LIVELLO_ALFA"] = df["LIVELLO_ALFA"].replace({"ERROR": "KO"})
    ctx, out = run(df, target_value="ko")
    assert ctx.n_errors > 0


def test_custom_target_column(base_df, run):
    df = base_df.rename(columns={"LIVELLO_ALFA": "ESITO_SERVIZIO"})
    ctx, out = run(df, target_col="ESITO_SERVIZIO", target_value=None)      # il valore lo propone la pipeline
    assert ctx.target_col == "ESITO_SERVIZIO" and ctx.target_values == ["ERROR"]


def test_target_with_nulls(base_df, run):
    df = base_df.copy()
    df.loc[df.index[::7], "LIVELLO_ALFA"] = ""
    ctx, out = run(df)
    assert ctx.n_errors > 0


def test_no_errors_exits(base_df, run):
    df = base_df.copy()
    df["LIVELLO_ALFA"] = "LOW"
    with pytest.raises(SystemExit):
        run(df)


def test_missing_target_column_exits(base_df, run):
    df = base_df.drop(columns=["LIVELLO_ALFA"])
    with pytest.raises(SystemExit):
        run(df)


def test_decimal_comma_and_thousands(base_df, run):
    df = base_df.copy()
    v = pd.to_numeric(df["IMPORTO"])
    df["IMPORTO"] = v.apply(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    ctx, out = run(df)
    assert "IMPORTO_BIN" in ctx.meta.get("derived_columns", [])


def test_constant_and_all_null_columns(base_df, run):
    df = base_df.copy()
    df["COSTANTE"] = "X"
    df["VUOTA"] = ""
    df["SOLO_NULL_TOKEN"] = "NULL"
    ctx, out = run(df)
    prof = ctx.results["01_profiling"].set_index("CAMPO")
    assert prof.loc["COSTANTE", "COSTANTE"] == "SI"
    assert prof.loc["VUOTA", "PCT_NULL"] == 100


def test_tiny_dataset(base_df, run):
    df = base_df.head(60).copy()
    df.loc[df.index[:6], "LIVELLO_ALFA"] = "ERROR"
    ctx, out = run(df)
    assert_outputs(out)


def test_reserved_column_names(base_df, run):
    df = base_df.copy()
    df["CLUSTER"] = np.where(df.index % 2 == 0, "A", "B")
    df["N"] = "1"
    df["VALORE"] = df["TIPO_RICHIESTA"]
    df["index"] = df["ID_RICHIESTA"]
    ctx, out = run(df)
    assert_outputs(out)


def test_special_chars_in_column_names(base_df, run):
    df = base_df.rename(columns={"ESITO_FINALE": "Esito Finale", "GAMMA_AZIONE": "Gamma/Azione", "DELTA_ESITO": "Delta: esito [v2]",
                                 "ZETA_DECISIONE": "Zeta*Decisione?"})
    ctx, out = run(df)
    assert_outputs(out)
    assert {"GAMMA", "DELTA", "ZETA"} <= set(ctx.cols.systems) and ctx.cols.system_of("Esito Finale") is None


def test_long_colliding_column_names(base_df, run):
    df = base_df.copy()
    base = "UNA_COLONNA_CON_UN_NOME_MOLTO_LUNGO_"
    df[base + "ALPHA"] = np.where(df.index % 2 == 0, "A", "B")
    df[base + "BETA"] = np.where(df.index % 3 == 0, "C", "D")
    ctx, out = run(df)
    dims = ctx.meta["dims"]
    assert base + "ALPHA" in dims and base + "BETA" in dims
    full = ctx.results["03_tassi_errore_completo"]
    assert set(full["DIMENSIONE"]) >= {base + "ALPHA", base + "BETA"}
    import openpyxl
    names = openpyxl.load_workbook(out / f"{PREFIX}_report.xlsx", read_only=True).sheetnames
    assert len(names) == len(set(names))


def test_many_extra_columns(base_df, run):
    df = base_df.copy()
    rng = np.random.default_rng(1)
    for i in range(40):
        df[f"EXTRA_{i}"] = rng.choice(["a", "b", "c", "d"], len(df))
    ctx, out = run(df)
    assert_outputs(out)


def test_numeric_with_nulls_and_ints(base_df, run):
    df = base_df.copy()
    df["CONTATORE"] = np.where(df.index % 5 == 0, "", (df.index % 97).astype(str))
    ctx, out = run(df)
    assert "CONTATORE" in ctx.meta.get("binned_cols", []) or "CONTATORE" in ctx.meta.get("model_columns", [])


def test_options_no_shap_inline_dims_nrows(base_df, run):
    ctx, out = run(base_df, use_shap=False, plotly_cdn=False, extra_dims=["PAESE", "NON_ESISTE"], nrows=1500)
    assert ctx.n_rows == 1500
    html = (out / f"{PREFIX}_report.html").read_text(encoding="utf-8")
    assert 'src="https://cdn.plot.ly' not in html and "plotly" in html


def test_duplicate_rows_and_ids(base_df, run):
    df = pd.concat([base_df, base_df.head(500)], ignore_index=True)
    ctx, out = run(df)
    assert ctx.n_rows == len(df)


def test_all_errors_identical(base_df, run):
    """Tutti gli errori hanno lo stesso profilo: clustering/anomaly devono degradare senza eccezioni."""
    df = base_df.copy()
    df["LIVELLO_ALFA"] = "LOW"
    idx = df.index[:60]
    df.loc[idx, "LIVELLO_ALFA"] = "ERROR"
    for c in df.columns:
        if c not in ("ID_RICHIESTA", "LIVELLO_ALFA"):
            df.loc[idx, c] = df.loc[idx[0], c]
    ctx, out = run(df)
    assert_outputs(out)


# ---------------------------------------------------------------------------
# Target "cross": qualunque colonna/valore
# ---------------------------------------------------------------------------

def test_target_other_system(base_df, run):
    """Target su un altro sistema: BETA esce dal confronto, ALFA entra tra i sistemi confrontati."""
    ctx, out = run(base_df, target_col="ESITO_BETA", target_value="KO")
    assert ctx.target_col == "ESITO_BETA" and ctx.target_values == ["KO"]
    assert ctx.system_label == "BETA" and ctx.event_label == "BETA KO"
    assert_outputs(out, prefix="esito_beta_ko")
    rel = ctx.results["06_relazione_sistemi"]
    assert "BETA" not in set(rel["SISTEMA"]) and "ALFA" in set(rel["SISTEMA"])
    assert "BETA KO" in ctx.meta["executive_summary_text"]
    assert not any("ALFA ERROR" in f.titolo for f in ctx.findings)


def test_target_descriptive_column(base_df, run):
    """Target su una colonna descrittiva (non l'esito di un sistema): nessun riferimento ad ALFA nei testi, nota sulle ipotesi."""
    ctx, out = run(base_df, target_col="TIPO_RICHIESTA", target_value="RESO")
    assert not ctx.is_system_target and ctx.event_label == "TIPO_RICHIESTA = RESO"
    assert_outputs(out, prefix="tipo_richiesta_reso")
    assert "hypotheses_note" in ctx.meta
    text = ctx.meta["executive_summary_text"] + " ".join(f.titolo + f.raccomandazione for f in ctx.findings)
    assert "ALFA ERROR" not in text and "in ERROR" not in text
    assert "06_detail_classificazione" not in ctx.results  # nessuna colonna di dettaglio per un target che non è un sistema
    assert "Nota terminologica" in ctx.meta["executive_summary_text"]
    # FLUSSO codifica già il tipo di richiesta ("FLW_RESO_412"): il report lo segnala e dice come escluderlo
    assert "FLUSSO" in ctx.meta["perfect_separators"]
    assert any(f.categoria == "SEPARATORE PERFETTO" and "--exclude FLUSSO" in f.raccomandazione for f in ctx.findings)
    assert "**Causa più probabile**" not in ctx.meta["executive_summary_text"]     # le ipotesi H1..H9 valgono per gli esiti dei sistemi


def test_excluded_columns_stay_out_of_patterns_and_models(base_df, run):
    ctx, out = run(base_df, target_col="TIPO_RICHIESTA", target_value="RESO", exclude_cols=["FLUSSO"])
    assert ctx.meta["excluded_cols"] == ["FLUSSO"] and "FLUSSO" not in ctx.meta["model_columns"]
    assert not ctx.results["04_association_rules"]["PATTERN"].str.contains("FLUSSO").any()
    assert "FLUSSO" not in set(ctx.results["04_importance_per_colonna"]["COLONNA"])
    assert "03_err_FLUSSO" in ctx.results                 # resta visibile nelle tabelle descrittive
    assert_outputs(out, prefix="tipo_richiesta_reso")


def test_target_multiple_values(base_df, run):
    ctx, out = run(base_df, target_value="ERROR, high")
    assert ctx.target_values == ["ERROR", "HIGH"]
    assert ctx.n_errors == int(base_df["LIVELLO_ALFA"].str.upper().isin(["ERROR", "HIGH"]).sum())
    assert_outputs(out, prefix="livello_alfa_error_high")


def test_select_target_interactive(base_df):
    """Selezione guidata: numero di colonna, poi valori per numero e per nome."""
    from cercaPattern.select_target import select_target
    printed = []
    cols = list(base_df.columns)
    answers = iter([str(cols.index("TIPO_RICHIESTA") + 1), "xyz_non_esiste", "1, prenotazione"])
    col, vals = select_target(base_df, input_fn=lambda _p: next(answers), print_fn=printed.append)
    assert col == "TIPO_RICHIESTA"
    top = base_df["TIPO_RICHIESTA"].str.upper().value_counts().index[0]
    assert vals == [top, "PRENOTAZIONE"]
    assert any("COLONNE DISPONIBILI" in str(p) for p in printed) and any("non presente" in str(p) for p in printed)


def test_select_target_defaults_on_enter(base_df):
    from cercaPattern.select_target import select_target
    col, vals = select_target(base_df, input_fn=lambda _p: "", print_fn=lambda _s: None)
    assert (col, vals) == ("LIVELLO_ALFA", ["ERROR"])


def test_select_target_given_col_asks_only_value(base_df):
    from cercaPattern.select_target import select_target
    asked = []
    def ask(prompt):
        asked.append(prompt)
        return "KO"
    col, vals = select_target(base_df, target_col="esito_beta", input_fn=ask, print_fn=lambda _s: None)
    assert col == "ESITO_BETA" and vals == ["KO"] and len(asked) == 1
