"""Test mirati (veloci) su singole funzioni: conversioni del loader, statistica, struttura del tracciato,
ricerca dei pattern, analisi temporale, anomalie, scrittura dei report. Gli scenari end-to-end sono in test_pipeline.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cercaPattern import loader  # noqa: E402
from cercaPattern.config import contains_token  # noqa: E402
from cercaPattern.loader import coerce_numeric, detect_format, load_csv, number_convention, parse_datetime  # noqa: E402
from cercaPattern.pipeline import build_context  # noqa: E402
from cercaPattern.stats_utils import binom_excess_pvalue, fdr_bh, pack, popcount, wilson_interval  # noqa: E402


def make_ctx(df: pd.DataFrame, tmp_path: Path, target_col=None, target_value="ERROR", **kw):
    """Contesto di analisi costruito come fa la pipeline: CSV su disco, caricamento, target."""
    p = tmp_path / "in.csv"
    df.to_csv(p, sep=";", index=False)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    return build_context(load_csv(p), out, target_col, target_value, **kw)


# ---------------------------------------------------------------------------
# Loader: numeri
# ---------------------------------------------------------------------------
def test_decimals_with_three_digits_are_not_thousands():
    """"0.123" e "12.500" in una colonna di punteggi sono decimali (prima diventavano 123 e 12500)."""
    v = coerce_numeric(pd.Series(["0.123", "0.1234", "0.25", "12.500", "-3.5", None, "abc"]))
    assert v.dtype == np.float64
    assert v.tolist()[:5] == [0.123, 0.1234, 0.25, 12.5, -3.5]
    assert np.isnan(v.iloc[5]) and np.isnan(v.iloc[6])


@pytest.mark.parametrize("values, expected", [
    (["1.234,56", "12,5", "1.000.000,00"], [1234.56, 12.5, 1_000_000.0]),      # italiano
    (["1,234.56", "12.5", "1,000,000.00"], [1234.56, 12.5, 1_000_000.0]),      # anglosassone
    (["1234,5", "0,75", "10"], [1234.5, 0.75, 10.0]),                           # virgola decimale
    (["1.234.567", "12.500", "950"], [1_234_567.0, 12_500.0, 950.0]),           # interi con punto delle migliaia
    (["950", "1.234", "12.500", "125.000"], [950.0, 1234.0, 12_500.0, 125_000.0]),
    (["1,234,567", "12,500"], [1_234_567.0, 12_500.0]),                         # interi con virgola delle migliaia
    (["1 234,50", "2 000,25"], [1234.5, 2000.25]),                         # spazi (anche non separabili) come migliaia
])
def test_number_conventions(values, expected):
    assert coerce_numeric(pd.Series(values)).tolist() == expected


def test_number_convention_is_decided_on_the_whole_column():
    assert number_convention(pd.Series(["0.5", "1.250", "3.125"])) == (".", "")
    assert number_convention(pd.Series(["1.234,5", "7"])) == (",", ".")
    assert number_convention(pd.Series(["testo", None])) == (".", "")


def test_coerce_numeric_gives_nan_not_pandas_na():
    """Con NA di pandas un confronto dà NA e `astype(int)` fallisce: capitava nella fase anomalie."""
    v = coerce_numeric(pd.Series(["1", "x", None, "3"]))
    hit = pd.Series([True, False, False, False]) | (v >= 2)
    assert hit.astype(int).tolist() == [1, 0, 0, 1]


# ---------------------------------------------------------------------------
# Loader: date
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value, expected", [
    ("2026091410", "2026-09-14 10:00:00"),          # prima veniva letto come 01:00
    ("2026091423", "2026-09-14 23:00:00"),
    ("20260914", "2026-09-14 00:00:00"),            # prima veniva letto come 2026-09-01 04:00
    ("202609141030", "2026-09-14 10:30:00"),        # prima veniva letto come 10:03
    ("20260914103059", "2026-09-14 10:30:59"),
])
def test_compact_time_key_depends_on_length(value, expected):
    assert parse_datetime(pd.Series([value, value])).iloc[0] == pd.Timestamp(expected)


def test_mixed_length_time_keys():
    ts = parse_datetime(pd.Series(["2026091410", "20260915", "20260916120000", None]))
    assert ts.iloc[:3].tolist() == [pd.Timestamp("2026-09-14 10:00"), pd.Timestamp("2026-09-15"), pd.Timestamp("2026-09-16 12:00")]
    assert pd.isna(ts.iloc[3])


def test_epoch_seconds_and_milliseconds():
    assert parse_datetime(pd.Series(["1757841600", "1757845200"])).tolist() == [pd.Timestamp("2025-09-14 09:20:00"), pd.Timestamp("2025-09-14 10:20:00")]
    assert parse_datetime(pd.Series(["1757841600000", "1757841600500"])).iloc[0] == pd.Timestamp("2025-09-14 09:20:00")


def test_date_only_with_dashes_is_not_read_as_timezone():
    """"14-09-2026" finisce con "-2026", che sembrava un fuso orario: il parsing andava in errore."""
    ts = parse_datetime(pd.Series(["14-09-2026", "01-10-2026"]))
    assert ts.tolist() == [pd.Timestamp("2026-09-14"), pd.Timestamp("2026-10-01")]


def test_us_dates_are_read_consistently():
    """Con giorni > 12 il formato è mese/giorno per TUTTA la colonna, non valore per valore."""
    ts = parse_datetime(pd.Series(["09/14/2026 10:30:00", "09/30/2026 11:00:00", "10/02/2026 08:00:00"]))
    assert ts.dt.month.tolist() == [9, 9, 10] and ts.dt.day.tolist() == [14, 30, 2]


def test_italian_dates_stay_day_first():
    ts = parse_datetime(pd.Series(["02/10/2026 08:00:00", "14/09/2026 10:30:00"]))
    assert ts.dt.month.tolist() == [10, 9]


@pytest.mark.parametrize("value", ["2026-10-26 17:43:19+00:00", "2026-10-26T17:43:19Z", "2026-10-26 17:43:19 +0200", "2026-10-26 17:43:19.250+02:00"])
def test_timezone_is_dropped_keeping_wall_clock(value):
    ts = parse_datetime(pd.Series([value, value])).iloc[0]
    assert (ts.hour, ts.minute, ts.second) == (17, 43, 19)


def test_unparsable_dates_give_nat_without_raising():
    ts = parse_datetime(pd.Series(["non è una data", "??", None]))
    assert ts.isna().all()


# ---------------------------------------------------------------------------
# Loader: formato del file
# ---------------------------------------------------------------------------
def test_utf8_is_detected_when_sample_ends_inside_a_multibyte_char(tmp_path):
    """Il campione di 256 KB può tagliare a metà una lettera accentata: restava cp1252 e gli accenti si rompevano."""
    head = "A;B;C\n".encode("utf-8")
    row = "1;città;x\n".encode("utf-8")
    data = bytearray(head)
    while len(data) < loader.SNIFF_BYTES - len(row):
        data += row
    data += b"9;" + b"x" * (loader.SNIFF_BYTES - len(data) - 3)   # il prossimo carattere inizia all'ultimo byte del campione
    assert len(data) == loader.SNIFF_BYTES - 1
    data += "à".encode("utf-8") + b";fine\n" + row * 50
    p = tmp_path / "taglio.csv"
    p.write_bytes(bytes(data))
    enc, sep = detect_format(p)
    assert (enc, sep) == ("utf-8-sig", ";")
    assert "città" in set(load_csv(p)["B"])


def test_encoding_falls_back_when_only_the_body_is_not_utf8(tmp_path):
    """Testa tutta ASCII (sembra UTF-8), ma più avanti c'è un byte cp1252: prima il caricamento andava in errore."""
    p = tmp_path / "corpo_cp1252.csv"
    p.write_bytes(b"A;B\n" + b"1;x\n" * 80_000 + "2;città\n".encode("cp1252"))
    df = load_csv(p)
    assert len(df) == 80_001 and df["B"].iloc[-1] == "città"


def test_utf16_with_bom(tmp_path):
    p = tmp_path / "excel_unicode.csv"
    p.write_text("A\tB\n1\tcittà\n2\tx\n", encoding="utf-16")
    df = load_csv(p)
    assert list(df.columns) == ["A", "B"] and df["B"].tolist() == ["città", "x"]


def test_separator_inside_quotes_does_not_count(tmp_path):
    """Molte virgole dentro un campo tra virgolette non devono far scegliere la virgola come separatore."""
    p = tmp_path / "virgolette.csv"
    rows = ["ID;NOTE;ESITO"] + [f'{i};"uno, due, tre, quattro, cinque, sei";OK' for i in range(50)]
    p.write_text("\n".join(rows), encoding="utf-8")
    assert detect_format(p)[1] == ";"
    df = load_csv(p)
    assert list(df.columns) == ["ID", "NOTE", "ESITO"] and len(df) == 50


def test_multichar_separator(tmp_path):
    p = tmp_path / "multi.csv"
    p.write_text("A||B||C\n1||x||ERROR\n2||y||LOW\n", encoding="utf-8")
    df = load_csv(p, sep="||")
    assert list(df.columns) == ["A", "B", "C"] and df["C"].tolist() == ["ERROR", "LOW"]


def test_padded_and_duplicate_column_names(tmp_path):
    """' A' e 'A' dopo lo strip sono lo stesso nome: prima `df['A']` diventava una tabella e il caricamento falliva."""
    p = tmp_path / "dup.csv"
    p.write_text("A; A;B ;B\n1;2;3;4\n5;6;7;8\n", encoding="utf-8")
    df = load_csv(p)
    assert len(set(df.columns)) == 4 and list(df.columns)[:2] == ["A", "A_2"]
    assert all(isinstance(df[c], pd.Series) for c in df.columns)


def test_trailing_separator_column_is_dropped_and_nulls_normalised(tmp_path):
    p = tmp_path / "coda.csv"
    p.write_text("A;B;\n1; x ;\nNULL;n/a;\n3;-;\n", encoding="utf-8")
    df = load_csv(p)
    assert list(df.columns) == ["A", "B"]
    assert df["A"].isna().tolist() == [False, True, False] and df["B"].tolist()[0] == "x" and df["B"].isna().sum() == 2


def test_only_the_head_of_the_file_is_read_for_detection(tmp_path, monkeypatch):
    p = tmp_path / "grande.csv"
    p.write_text("A;B\n" + "1;x\n" * 200_000, encoding="utf-8")
    monkeypatch.setattr(Path, "read_bytes", lambda self: (_ for _ in ()).throw(AssertionError("letto l'intero file")))
    assert detect_format(p) == ("utf-8-sig", ";")


# ---------------------------------------------------------------------------
# Statistica
# ---------------------------------------------------------------------------
def test_wilson_interval_known_values():
    lo, hi = wilson_interval([5, 0, 10], [10, 10, 10])
    assert lo[0] == pytest.approx(0.2366, abs=1e-3) and hi[0] == pytest.approx(0.7634, abs=1e-3)
    assert lo[1] == 0 and hi[1] == pytest.approx(0.2775, abs=1e-3)
    assert hi[2] == pytest.approx(1.0) and lo[2] == pytest.approx(0.7225, abs=1e-3)
    lo, hi = wilson_interval([0], [0])
    assert (lo[0], hi[0]) == (0.0, 1.0)


def test_fdr_bh():
    assert fdr_bh([0.01, 0.02, 0.03, 0.04, 0.05]) == pytest.approx([0.05] * 5)
    q = fdr_bh([0.5, 0.001, np.nan])
    assert q[0] == pytest.approx(0.5) and q[1] == pytest.approx(0.002) and np.isnan(q[2])
    assert len(fdr_bh([])) == 0


def test_binomial_excess_pvalue():
    assert binom_excess_pvalue(0, 10, 0.1) == pytest.approx(1.0)
    assert binom_excess_pvalue(10, 10, 0.5) == pytest.approx(0.5 ** 10)
    assert binom_excess_pvalue(3, 0, 0.5) == pytest.approx(1.0)


def test_packed_bits_count_rows_exactly():
    rng = np.random.default_rng(0)
    for n in (1, 7, 8, 9, 1001):
        a, b = rng.random(n) < 0.4, rng.random(n) < 0.6
        assert popcount(pack(a)) == int(a.sum())
        assert popcount(pack(a) & pack(b)) == int((a & b).sum())
        universe = pack(np.ones(n, dtype=bool))
        assert popcount(universe & ~pack(a)) == int((~a).sum())      # i bit di riempimento non contano


# ---------------------------------------------------------------------------
# Ricerca dei pattern
# ---------------------------------------------------------------------------
def _pattern_df(n=6000, seed=3) -> pd.DataFrame:
    """Causa vera: A=x insieme a B=p (errore 90%); fondo 2%. C è rumore."""
    rng = np.random.default_rng(seed)
    a = rng.choice(["x", "y", "z"], n, p=[0.3, 0.4, 0.3])
    b = rng.choice(["p", "q"], n, p=[0.4, 0.6])
    c = rng.choice(["c1", "c2", "c3", "c4"], n)
    p_err = np.where((a == "x") & (b == "p"), 0.9, 0.02)
    err = rng.random(n) < p_err
    return pd.DataFrame({"ID": [f"T{i}" for i in range(n)], "A": a, "B": b, "C": c, "LIVELLO_ALFA": np.where(err, "ERROR", "LOW")})


def _mine(ctx, **kw):
    from cercaPattern.config import categorical_columns
    from cercaPattern.patterns import ItemBits, build_feature_frame, mine_association_rules
    frame = build_feature_frame(ctx, categorical_columns(ctx))
    bits = ItemBits(frame, ctx.is_error.to_numpy())
    return mine_association_rules(ctx, frame, bits, **kw), bits


def test_rules_find_the_true_cause_with_exact_counts(tmp_path):
    df = _pattern_df()
    ctx = make_ctx(df, tmp_path)
    rules, _ = _mine(ctx)
    best = rules.iloc[0]
    assert best["PATTERN"] == "A=x + B=p"
    cause = (df["A"] == "x") & (df["B"] == "p")
    assert best["N_RIGHE"] == int(cause.sum())
    assert best["N_ERRORI"] == int((cause & (df["LIVELLO_ALFA"] == "ERROR")).sum())
    assert best["CONF_MIN_95_PCT"] < best["CONFIDENZA_PCT"] <= 100
    # il rumore non produce pattern: nessuna regola contiene C
    assert not rules["PATTERN"].str.contains("C=").any()
    assert (rules["Q_VALUE"] <= 0.05).all() and (rules["LIFT"] > 1).all()


def test_cumulative_cover_isolates_distinct_problems(tmp_path):
    df = _pattern_df()
    ctx = make_ctx(df, tmp_path)
    _mine(ctx)
    cover = ctx.results["04_copertura_cumulativa"]
    assert cover.iloc[0]["PATTERN"] == "A=x + B=p"
    assert len(cover) == 2 and cover.iloc[-1]["PATTERN"].startswith("(nessun pattern")      # un solo problema, poi il fondo
    assert cover["N_ERRORI_NUOVI"].sum() == ctx.n_errors and cover["N_RIGHE_NUOVE"].sum() == ctx.n_rows
    assert ctx.meta["cover_residual_rate_pct"] < 3 < 100 * ctx.error_rate
    assert any(f.categoria == "SPIEGAZIONE CUMULATIVA" for f in ctx.findings)


def test_cover_skips_the_shadow_of_a_pattern_already_taken(tmp_path):
    """S accompagna quasi sempre la causa P: da solo ha un tasso alto, ma tolta P non spiega più nulla."""
    rng = np.random.default_rng(5)
    n = 8000
    p = rng.random(n) < 0.15
    s = np.where(p, rng.random(n) < 0.9, rng.random(n) < 0.05)       # ombra di P
    err = rng.random(n) < np.where(p, 0.6, 0.02)
    df = pd.DataFrame({"ID": range(n), "P": np.where(p, "si", "no"), "S": np.where(s, "si", "no"),
                       "R": rng.choice(["a", "b", "c"], n), "LIVELLO_ALFA": np.where(err, "ERROR", "LOW")})
    ctx = make_ctx(df, tmp_path)
    rules, _ = _mine(ctx)
    assert "S=si" in set(rules["PATTERN"])                           # preso da solo sembra un pattern...
    steps = ctx.results["04_copertura_cumulativa"]["PATTERN"].tolist()
    assert steps[0] == "P=si" and "S=si" not in steps                # ...ma non è un problema distinto


def test_equivalent_patterns_need_the_same_rows_not_just_the_same_counts(tmp_path):
    """G1 e G2 hanno gli stessi conteggi su righe diverse: restano due pattern. H è una copia di G1: una riga sola."""
    n = 3000
    g1 = np.zeros(n, dtype=bool); g1[:100] = True
    g2 = np.zeros(n, dtype=bool); g2[100:200] = True
    err = np.zeros(n, dtype=bool); err[:80] = True; err[100:180] = True
    err[np.arange(200, n, 100)] = True
    df = pd.DataFrame({"ID": range(n), "G1": np.where(g1, "1", "0"), "G2": np.where(g2, "1", "0"), "H": np.where(g1, "si", "no"),
                       "LIVELLO_ALFA": np.where(err, "ERROR", "LOW")})
    ctx = make_ctx(df, tmp_path)
    rules, _ = _mine(ctx)
    pats = rules.set_index("PATTERN")
    assert "G2=1" in pats.index                                       # stessi numeri di G1, ma altre righe
    kept = [p for p in ("G1=1", "H=si") if p in pats.index]
    assert len(kept) == 1                                             # G1 e H selezionano le stesse righe
    other = "H=si" if kept[0] == "G1=1" else "G1=1"
    assert other in pats.loc[kept[0], "PATTERN_EQUIVALENTI"]
    assert pats.loc["G2=1", "PATTERN_EQUIVALENTI"] == ""
    assert pats.loc["G2=1", "N_RIGHE"] == 100 and pats.loc["G2=1", "N_ERRORI"] == 80


def test_thresholds_adapt_to_a_rare_target(tmp_path):
    """Target allo 0,6%: col supporto del 2% su tutte le righe nessun pattern poteva emergere."""
    rng = np.random.default_rng(11)
    n = 10_000
    app = rng.choice(["101", "205", "412", "999"], n, p=[0.5, 0.3, 0.19, 0.01])
    err = rng.random(n) < np.where(app == "999", 0.5, 0.001)
    df = pd.DataFrame({"ID": range(n), "APP": app, "X": rng.choice(["a", "b"], n), "LIVELLO_ALFA": np.where(err, "ERROR", "LOW")})
    ctx = make_ctx(df, tmp_path)
    rules, _ = _mine(ctx, min_support=0.02, min_confidence=0.2)
    assert rules.iloc[0]["PATTERN"] == "APP=999"
    assert ctx.meta["pattern_min_count"] < 0.02 * n


def test_insignificant_refinements_are_pruned(tmp_path):
    """'P=si + R=a' non migliora davvero 'P=si' (R è rumore): non deve comparire."""
    rng = np.random.default_rng(8)
    n = 6000
    p = rng.random(n) < 0.2
    err = rng.random(n) < np.where(p, 0.45, 0.02)
    df = pd.DataFrame({"ID": range(n), "P": np.where(p, "si", "no"), "R": rng.choice(["a", "b", "c", "d", "e"], n),
                       "Q": rng.choice(["u", "v", "w"], n), "LIVELLO_ALFA": np.where(err, "ERROR", "LOW")})
    ctx = make_ctx(df, tmp_path)
    rules, _ = _mine(ctx)
    assert rules["PATTERN"].tolist() == ["P=si"]


def test_tree_and_rules_count_on_all_rows_even_when_models_are_sampled(tmp_path):
    """Con il campionamento attivo (dataset più grande di sample_rows) i conteggi restano quelli veri."""
    from cercaPattern.patterns import run_pattern_discovery
    df = _pattern_df(n=6000)
    ctx = make_ctx(df, tmp_path, sample_rows=1500)
    run_pattern_discovery(ctx, use_shap=False)
    cause = (df["A"] == "x") & (df["B"] == "p")
    best = ctx.results["04_association_rules"].iloc[0]
    assert (best["PATTERN"], best["N_RIGHE"]) == ("A=x + B=p", int(cause.sum()))
    tree = ctx.results["04_albero_regole"]
    assert tree["N_RIGHE"].sum() == len(df) and tree["N_ERRORI"].sum() == ctx.n_errors
    assert ctx.meta["xgb_auc_kind"] == "hold-out" and 0.5 < ctx.meta["xgb_auc"] <= 1
    corr = ctx.results["04_correlazioni"].set_index("COLONNA")
    assert corr.loc["C", "CRAMERS_V"] < 0.1 < corr.loc[["A", "B"], "CRAMERS_V"].min()


def test_no_pattern_when_nothing_explains_the_target(tmp_path):
    rng = np.random.default_rng(2)
    n = 4000
    df = pd.DataFrame({"ID": range(n), "A": rng.choice(["x", "y"], n), "B": rng.choice(["p", "q", "r"], n),
                       "LIVELLO_ALFA": np.where(rng.random(n) < 0.1, "ERROR", "LOW")})
    ctx = make_ctx(df, tmp_path)
    rules, _ = _mine(ctx)
    assert rules.empty and "04_copertura_cumulativa" not in ctx.results
    assert any(f.categoria == "PATTERN" and "Nessuna combinazione" in f.titolo for f in ctx.findings)


def test_alias_of_the_target_is_excluded_but_a_deterministic_cause_is_not(base_df, tmp_path):
    """COD_TIPO determina ogni valore di TIPO_RICHIESTA: alias. APP=412 che va sempre in errore è invece una causa da trovare."""
    from cercaPattern.config import categorical_columns
    codes = {p: f"P{i:02d}" for i, p in enumerate(sorted(base_df["TIPO_RICHIESTA"].unique()))}
    coded = base_df.assign(COD_TIPO=base_df["TIPO_RICHIESTA"].map(codes))
    ctx = make_ctx(coded, tmp_path, target_col="TIPO_RICHIESTA", target_value="RESO")
    assert ctx.meta["alias_cols"] == ["COD_TIPO"] and "COD_TIPO" not in categorical_columns(ctx)
    # FLUSSO separa RESO dal resto ma non determina ogni tipo (i due ORDINE_* hanno lo stesso prefisso):
    # potrebbe essere una causa, quindi resta nei modelli e il report avvisa
    assert "FLUSSO" in ctx.meta["perfect_separators"] and "FLUSSO" in categorical_columns(ctx)
    assert any(f.categoria == "SEPARATORE PERFETTO" and "--exclude FLUSSO" in f.raccomandazione for f in ctx.findings)
    # ... e l'analista può escluderlo (nome senza distinzione tra maiuscole e minuscole; i nomi inesistenti sono ignorati)
    ctx = make_ctx(coded, tmp_path, target_col="TIPO_RICHIESTA", target_value="RESO", exclude_cols=["flusso", "NON_ESISTE"])
    assert ctx.meta["excluded_cols"] == ["FLUSSO"] and "FLUSSO" not in categorical_columns(ctx)
    rules, _ = _mine(ctx)
    assert not rules.empty and not rules["PATTERN"].str.contains("FLUSSO|COD_TIPO").any()
    # stessa tabella, target LIVELLO_ALFA: nessuna colonna determina LOW/MEDIUM/HIGH/ERROR
    assert make_ctx(base_df, tmp_path).meta["alias_cols"] == []
    df = base_df.copy()
    df["LIVELLO_ALFA"] = np.where(df["APP"] == "412", "ERROR", df["LIVELLO_ALFA"].replace({"ERROR": "LOW"}))
    ctx = make_ctx(df, tmp_path)
    assert ctx.meta["alias_cols"] == [] and "APP" in categorical_columns(ctx) and "APP" in ctx.meta["perfect_separators"]
    rules, _ = _mine(ctx)
    assert rules.iloc[0]["CONFIDENZA_PCT"] == 100 and "412" in rules.iloc[0]["PATTERN"] + rules.iloc[0]["PATTERN_EQUIVALENTI"]
    # un target a due soli valori non ha alias: lì una colonna che lo determina è la spiegazione
    two = df.assign(ESITO=np.where(df["APP"] == "412", "KO", "OK"), COPIA=np.where(df["APP"] == "412", "k", "o"))
    assert make_ctx(two, tmp_path, target_col="ESITO", target_value="KO").meta["alias_cols"] == []


# ---------------------------------------------------------------------------
# Analisi errori (fase 3)
# ---------------------------------------------------------------------------
def test_error_tables_have_confidence_intervals_and_fdr(tmp_path):
    from cercaPattern.error_analysis import run_error_analysis
    ctx = make_ctx(_pattern_df(), tmp_path)
    run_error_analysis(ctx)
    t = ctx.results["03_err_A"].set_index("VALORE")
    assert {"TASSO_MIN_95_PCT", "TASSO_MAX_95_PCT", "Q_VALUE", "SIGNIFICATIVO"} <= set(t.columns)
    assert (t["TASSO_MIN_95_PCT"] <= t["TASSO_ERRORE_PCT"]).all() and (t["TASSO_ERRORE_PCT"] <= t["TASSO_MAX_95_PCT"]).all()
    assert t.loc["x", "SIGNIFICATIVO"] == "SI" and t.loc["y", "SIGNIFICATIVO"] == "NO"
    assert (ctx.results["03_err_C"]["SIGNIFICATIVO"] == "NO").all()      # il rumore non è mai significativo
    assert t["N"].sum() == ctx.n_rows and t["N_ERROR"].sum() == ctx.n_errors


def test_no_systematic_rate_findings_when_almost_everything_is_target(tmp_path):
    """Col 95% di righe target ogni valore supera il 50%: non sono 'tassi anomali', è il tasso normale."""
    from cercaPattern.error_analysis import run_error_analysis
    rng = np.random.default_rng(4)
    n = 3000
    df = pd.DataFrame({"ID": range(n), "APP": rng.choice(["101", "205", "412"], n), "TIPO": rng.choice(["A", "B"], n),
                       "LIVELLO_ALFA": np.where(rng.random(n) < 0.95, "ERROR", "LOW")})
    ctx = make_ctx(df, tmp_path)
    run_error_analysis(ctx)
    assert not [f for f in ctx.findings if f.categoria in ("TASSO ANOMALO", "CONCENTRAZIONE")]


# ---------------------------------------------------------------------------
# Analisi temporale (fase 5)
# ---------------------------------------------------------------------------
def test_burst_detection_ignores_small_noisy_hours():
    from cercaPattern.temporal import detect_bursts
    rng = np.random.default_rng(1)
    n = np.full(300, 100)
    e = rng.binomial(100, 0.05, 300)
    n[40], e[40] = 13, 4            # 31% su 13 righe: rumore, non degrado
    e[120], e[121] = 60, 55         # incidente vero
    flag, expected, q = detect_bursts(n, e)
    assert np.flatnonzero(flag).tolist() == [120, 121]
    assert expected[120] == pytest.approx(5, abs=1.5)       # l'atteso non è gonfiato dall'incidente stesso


def test_burst_baseline_follows_a_lasting_level_change():
    """Dopo un cambio di livello duraturo le ore 'nuove' non sono tutte degradi: lo decide il cambio di livello."""
    from cercaPattern.temporal import detect_bursts
    rng = np.random.default_rng(6)
    n = np.full(24 * 30, 100)
    e = np.concatenate([rng.binomial(100, 0.02, 24 * 15), rng.binomial(100, 0.06, 24 * 15)])
    flag, _, _ = detect_bursts(n, e)
    assert flag.sum() <= 3


def test_level_shift_detection():
    from cercaPattern.temporal import detect_level_shifts
    rng = np.random.default_rng(9)
    n = np.full(30, 2000)
    stable = rng.binomial(2000, 0.03, 30)
    assert detect_level_shifts(n, stable) == []
    spike = stable.copy(); spike[12] = 900                      # un giorno anomalo è un incidente, non un cambio di livello
    assert detect_level_shifts(n, spike) == []
    shifted = np.concatenate([rng.binomial(2000, 0.02, 18), rng.binomial(2000, 0.08, 12)])
    assert detect_level_shifts(n, shifted) == [18]
    up_and_down = np.concatenate([rng.binomial(2000, 0.02, 10), rng.binomial(2000, 0.09, 10), rng.binomial(2000, 0.02, 10)])
    assert detect_level_shifts(n, up_and_down) == [10, 20]


def _timeline(per_hour=60, days=30, seed=13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2026-09-01", periods=24 * days, freq="h")
    ts = np.repeat(hours, per_hour) + pd.to_timedelta(rng.integers(0, 3600, len(hours) * per_hour), unit="s")
    n = len(ts)
    return pd.DataFrame({"ID": [f"T{i}" for i in range(n)], "DATETIME": pd.DatetimeIndex(ts).strftime("%Y-%m-%d %H:%M:%S"),
                         "APP": rng.choice(["101", "205", "412"], n), "TIPO": rng.choice(["ORDINE", "RESO", "DISDETTA"], n),
                         "LIVELLO_ALFA": "LOW", "_u": rng.random(n)})


def _set_errors(df: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    df["LIVELLO_ALFA"] = np.where(df["_u"] < prob, "ERROR", "LOW")
    return df.drop(columns="_u")


def test_cross_cutting_window_points_to_infrastructure(tmp_path):
    from cercaPattern.hypotheses import build_hypotheses
    from cercaPattern.temporal import run_temporal
    df = _timeline()
    burst = df["DATETIME"].between("2026-09-10 14:00:00", "2026-09-10 15:59:59")
    ctx = make_ctx(_set_errors(df, np.where(burst, 0.7, 0.02)), tmp_path)
    run_temporal(ctx)
    win = ctx.results["05_finestre_degrado"]
    assert len(win) == 1 and win.iloc[0]["AMBITO"] == "trasversale" and win.iloc[0]["ORE"] == 2
    assert win.iloc[0]["INIZIO"] == pd.Timestamp("2026-09-10 14:00") and win.iloc[0]["ECCESSO_ERRORI"] > 50
    assert ctx.results["05_cambi_livello"].empty
    hyp = build_hypotheses(ctx).set_index("CODICE")
    assert hyp.loc["H1", "SCORE"] > 0 and hyp.loc["H9", "SCORE"] == 0


def test_localised_window_points_to_one_flow(tmp_path):
    from cercaPattern.hypotheses import build_hypotheses
    from cercaPattern.temporal import run_temporal
    df = _timeline()
    burst = df["DATETIME"].between("2026-09-10 14:00:00", "2026-09-10 15:59:59") & (df["APP"] == "412")
    ctx = make_ctx(_set_errors(df, np.where(burst, 0.9, 0.02)), tmp_path)
    run_temporal(ctx)
    win = ctx.results["05_finestre_degrado"]
    assert len(win) == 1 and win.iloc[0]["AMBITO"].startswith("APP=412")
    hyp = build_hypotheses(ctx).set_index("CODICE")
    assert hyp.loc["H1", "SCORE"] == 0                      # non è l'infrastruttura comune
    assert "localizzata su APP=412" in hyp.loc["H3", "EVIDENZE"]


def test_level_shift_is_attributed_to_its_driver(tmp_path):
    from cercaPattern.hypotheses import build_hypotheses
    from cercaPattern.temporal import run_temporal
    df = _timeline()
    after = (df["DATETIME"] >= "2026-09-20 00:00:00") & (df["APP"] == "205")
    ctx = make_ctx(_set_errors(df, np.where(after, 0.3, 0.02)), tmp_path)
    run_temporal(ctx)
    sh = ctx.results["05_cambi_livello"]
    assert len(sh) == 1
    s = sh.iloc[0]
    assert s["DATA_CAMBIO"] == pd.Timestamp("2026-09-20") and s["DIREZIONE"] == "AUMENTO" and s["GUIDATO_DA"].startswith("APP=205")
    assert s["TASSO_PRIMA_PCT"] < 3 < 8 < s["TASSO_DOPO_PCT"]
    assert ctx.results["05_finestre_degrado"].empty          # un cambio di livello non è una sequenza di degradi
    hyp = build_hypotheses(ctx)
    assert hyp.iloc[0]["CODICE"] == "H9" and hyp.iloc[0]["CONFIDENZA"] == "ALTA"
    daily = ctx.results["05_serie_giornaliera"]
    assert daily["LIVELLO_PCT"].nunique() == 2


def test_hour_of_single_incident_is_not_a_recurring_critical_hour(tmp_path):
    from cercaPattern.temporal import run_temporal
    df = _timeline()
    burst = df["DATETIME"].between("2026-09-10 14:00:00", "2026-09-10 14:59:59")
    ctx = make_ctx(_set_errors(df, np.where(burst, 0.9, 0.02)), tmp_path)
    run_temporal(ctx)
    hourly = ctx.results["05_tasso_per_ora"].set_index("ORA")
    assert hourly.loc[14, "LIFT"] > 1.5 and hourly.loc[14, "CRITICA"] == "NO"
    assert not [f for f in ctx.findings if f.categoria == "FASCE ORARIE"]


def test_time_key_hours_match_the_real_hours(base_df, tmp_path):
    from cercaPattern.temporal import build_timestamp
    ctx = make_ctx(base_df.drop(columns=["DATETIME", "TIME"]), tmp_path)
    ts = build_timestamp(ctx)
    real = pd.to_datetime(base_df["DATETIME"])
    assert (ts.dt.hour.to_numpy() == real.dt.hour.to_numpy()).all() and (ts.dt.day.to_numpy() == real.dt.day.to_numpy()).all()


def test_sentinel_dates_do_not_stretch_the_timeline(tmp_path):
    from cercaPattern.temporal import run_temporal
    df = _timeline(per_hour=10, days=10)
    df.loc[df.index[:3], "DATETIME"] = "1900-01-01 00:00:00"
    df.loc[df.index[3:5], "DATETIME"] = "2099-12-31 23:59:59"
    ctx = make_ctx(_set_errors(df, np.full(len(df), 0.05)), tmp_path)
    run_temporal(ctx)
    assert len(ctx.results["05_serie_oraria"]) == 24 * 10
    assert any(f.categoria == "QUALITA' DATI" and "fuori dal periodo" in f.titolo for f in ctx.findings)


# ---------------------------------------------------------------------------
# Funzionale, cluster, anomalie
# ---------------------------------------------------------------------------
def test_tokens_are_whole_words():
    from cercaPattern.anomaly import CRITICAL_RX
    from cercaPattern.functional import FAILURE_RX
    s = pd.Series(["FINAL", "TOKEN_OK", "GEO_ERR", "KO", "timeout", "ERROR", None, "ALTERNATIVE", "N/A", "FAILED"])
    assert contains_token(s, FAILURE_RX).tolist() == [False, False, True, True, True, True, False, False, True, True]
    r = pd.Series(["CREDIT", "GEO_RED", "DECLINED", "SCORED", "HIGH", "HIGHWAY", "deny", "BLOCKED", "OK", None])
    assert contains_token(r, CRITICAL_RX).tolist() == [False, True, True, False, True, False, True, True, False, False]


def test_optional_step_is_missing_when_empty_or_negative(base_df, tmp_path):
    """Un passaggio non eseguito è un campo vuoto ('NONE') oppure un valore negativo ('NO'); una colonna
    che ha sempre un valore pieno (MODALITA) non descrive un passaggio facoltativo."""
    from cercaPattern.functional import analyze_optional_steps
    skipped = base_df["PASSO_CONFERMA"] == "NONE"
    ctx = make_ctx(base_df.assign(CONFERMATA=np.where(skipped, "NO", "SI")), tmp_path)
    g = analyze_optional_steps(ctx)
    assert set(g["COLONNA"]) == {"PASSO_CONFERMA", "CONFERMATA"}
    for col in ("PASSO_CONFERMA", "CONFERMATA"):
        t = g[g["COLONNA"] == col].set_index("PASSAGGIO")
        assert t.loc["SENZA PASSAGGIO", "N"] == int(skipped.sum()) and t.loc["CON PASSAGGIO", "N"] == int((~skipped).sum())
        assert t["N_ERROR"].sum() == ctx.n_errors


def test_anomaly_phase_survives_dirty_score_columns(base_df, tmp_path):
    """Punteggi con qualche valore non numerico: il confronto dava NA e la fase 8 falliva."""
    from cercaPattern.anomaly import run_anomaly
    from cercaPattern.clustering import run_clustering
    df = base_df.copy()
    df.loc[df.index[::9], "ZETA_SCORE"] = "n.d."
    df.loc[df.index[::11], "GAMMA_SCORE"] = "errore"
    ctx = make_ctx(df, tmp_path)
    run_clustering(ctx)
    full = run_anomaly(ctx)
    assert len(full) == ctx.n_errors and full["ALTA_CRITICITA"].dtype == bool
    assert set(full["CONSENSO"].unique()) <= {0, 1, 2, 3}
    assert 0 < full["IF_FLAG"].mean() <= 0.06


def test_target_system_does_not_flag_its_own_rows_as_critical(base_df, tmp_path):
    """Con target ESITO_BETA=KO la colonna vale 'KO' su tutte le righe target: non è un segnale di un ALTRO sistema."""
    from cercaPattern.anomaly import run_anomaly
    from cercaPattern.clustering import run_clustering
    df = base_df.copy()
    ko = df.index[df["ESITO_BETA"] == "KO"]
    df.loc[ko[:5], "LIVELLO_ALFA"] = "HIGH"
    ctx = make_ctx(df, tmp_path, target_col="ESITO_BETA", target_value="KO")
    run_clustering(ctx)
    full = run_anomaly(ctx)
    assert not full["CRITICITA_MOTIVI"].str.contains("ESITO_BETA").any()
    assert full["CRITICITA_MOTIVI"].str.contains("LIVELLO_ALFA").any()     # ALFA = HIGH ora è il segnale di un altro sistema
    assert full["ALTA_CRITICITA"].mean() < 0.9


def test_identical_target_rows_form_one_profile(tmp_path):
    """Migliaia di righe target identiche sono pochi profili: stesso cluster, mai anomale."""
    from cercaPattern.anomaly import run_anomaly
    from cercaPattern.clustering import error_profiles, run_clustering
    rng = np.random.default_rng(21)
    n = 6000
    grp = rng.choice(["g1", "g2", "g3"], n, p=[0.5, 0.3, 0.2])
    err = rng.random(n) < 0.5
    df = pd.DataFrame({"ID": range(n), "APP": np.where(grp == "g1", "101", np.where(grp == "g2", "205", "412")),
                       "TIPO": np.where(grp == "g1", "ORDINE", np.where(grp == "g2", "RESO", "DISDETTA")),
                       "CANALE": np.where(grp == "g3", "TOTEM", "WEB"), "LIVELLO_ALFA": np.where(err, "ERROR", "LOW")})
    odd = df.index[err][:3]                                                 # tre righe target fuori dagli schemi
    df.loc[odd, ["APP", "TIPO", "CANALE"]] = [["999", "RARO", "FAX"], ["998", "RARO2", "FAX"], ["997", "RARO3", "TELEX"]]
    ctx = make_ctx(df, tmp_path)
    prof = error_profiles(ctx)
    assert len(prof["counts"]) == 6 and prof["counts"].sum() == ctx.n_errors
    clusters = run_clustering(ctx)
    labels = ctx.meta["cluster_labels"]
    key = df.loc[labels.index, ["APP", "TIPO", "CANALE"]].agg("|".join, axis=1)
    assert labels.groupby(key).nunique().max() == 1                         # righe identiche, stesso cluster
    assert clusters["DIMENSIONE"].sum() == ctx.n_errors
    full = run_anomaly(ctx)
    heavy = ~full.index.isin(odd)
    assert not full.loc[heavy, "LOF_FLAG"].any() and not full.loc[heavy, "DBSCAN_FLAG"].any()


# ---------------------------------------------------------------------------
# Struttura del tracciato: ruoli e sistemi
# ---------------------------------------------------------------------------
def test_systems_are_recognised_from_column_names():
    """Parola di esito + nome proprio = sistema; le parole generiche da sole non nominano nessun sistema."""
    from cercaPattern.config import ColumnMap, detect_systems, system_name
    cols = ["ID_RICHIESTA", "DATETIME", "TIME_KEY", "APP", "TIPO_RICHIESTA", "ESITO_FINALE", "DECISIONE", "MODALITA", "REGOLA",
            "LIVELLO_ALFA", "DETTAGLIO_ALFA", "SCORE_ALFA", "SESSIONE_ALFA", "ESITO_BETA", "GAMMA_SCORE", "GAMMA_AZIONE", "ID_DISPOSITIVO"]
    assert detect_systems(cols) == {"ALFA": ["LIVELLO_ALFA", "DETTAGLIO_ALFA", "SCORE_ALFA", "SESSIONE_ALFA"],
                                    "BETA": ["ESITO_BETA"], "GAMMA": ["GAMMA_SCORE", "GAMMA_AZIONE"]}
    assert [system_name(c) for c in ("Stato Delta", "delta_status_code", "resultCodeDelta", "DELTA: esito [v2]")] == ["DELTA"] * 4
    assert [system_name(c) for c in ("ESITO", "ESITO_FINALE", "STATUS_CODE", "ID_UTENTE", "TIME_KEY", "REGOLA", "TIPO_RICHIESTA", "N", "ESITO_2")] == [None] * 9
    # una colonna entra in un sistema solo se, tolto il nome, restano parole che descrivono un esito
    assert detect_systems(["ESITO_ALFA", "ALFA_CODE", "ALFA_VERSIONE", "ALFA", "TIPO_ALFA"]) == {"ALFA": ["ESITO_ALFA", "ALFA_CODE"]}
    assert detect_systems(["STATO_ALFA_BETA", "STATO_ALFA", "ALFA_BETA_MSG", "ALFA_MSG"]) == {"ALFA_BETA": ["STATO_ALFA_BETA", "ALFA_BETA_MSG"],
                                                                                           "ALFA": ["STATO_ALFA", "ALFA_MSG"]}
    m = ColumnMap(cols)
    assert m.get("datetime") == "DATETIME" and m.get("time_key") == "TIME_KEY" and m.get("regola") == "REGOLA"
    assert m.get("decisione") == "DECISIONE"                       # non MODALITA, non l'azione di un singolo sistema
    assert m.system_of("GAMMA_AZIONE") == "GAMMA" and m.system_of("ESITO_FINALE") is None


def test_declared_groups_take_precedence_over_names():
    from cercaPattern.config import detect_systems
    cols = ["ESITO_ALFA", "DETTAGLIO_ALFA", "K_OUT", "K_INFO", "ESITO_BETA"]
    assert detect_systems(cols, groups={"KAPPA": "^K_"}) == {"ALFA": ["ESITO_ALFA", "DETTAGLIO_ALFA"], "KAPPA": ["K_OUT", "K_INFO"], "BETA": ["ESITO_BETA"]}
    assert detect_systems(cols, groups={"KAPPA": "^K_"}, auto=False) == {"KAPPA": ["K_OUT", "K_INFO"]}
    assert detect_systems(cols, groups={"TUTTO": "alfa|beta"}) == {"TUTTO": ["ESITO_ALFA", "DETTAGLIO_ALFA", "ESITO_BETA"]}
    assert detect_systems(cols, auto=False) == {}


def test_groups_change_what_describes_the_target(base_df, tmp_path):
    """Senza gruppi il target non ha un sistema: nessuna colonna di dettaglio, nessun confronto con altri sistemi."""
    ctx = make_ctx(base_df, tmp_path)
    assert ctx.system_key == "ALFA" and ctx.detail_col == "DETTAGLIO_ALFA" and ctx.score_col == "SCORE_ALFA"
    assert ctx.target_group_cols == ["DETTAGLIO_ALFA", "SCORE_ALFA", "SESSIONE_ALFA"] and ctx.session_col == "SESSIONE_ALFA"
    assert set(ctx.other_systems) == {"BETA", "GAMMA", "DELTA", "EPSILON", "ZETA"}
    assert (ctx.measure_col, ctx.decision_col, ctx.id_col) == ("IMPORTO", "DECISIONE", "ID_RICHIESTA")
    assert any(f.categoria == "SISTEMA DEL TARGET" and "--no-auto-groups" in f.raccomandazione for f in ctx.findings)
    plain = make_ctx(base_df, tmp_path, auto_groups=False)
    assert not plain.is_system_target and plain.detail_col is None and plain.other_systems == {} and plain.event_label == "LIVELLO_ALFA = ERROR"
    named = make_ctx(base_df, tmp_path, auto_groups=False, groups=["PRIMO=_ALFA$", "SECONDO=^GAMMA_"])
    assert named.system_key == "PRIMO" and named.event_label == "PRIMO ERROR" and list(named.other_systems) == ["SECONDO"]
    with pytest.raises(SystemExit, match="NOME=REGEX"):
        make_ctx(base_df, tmp_path, groups=["senza_uguale"])


def test_missing_fields_look_at_collected_data(base_df, tmp_path):
    """Campi mancanti: identificativi, token, coordinate, sessione e regola. Non le colonne con pochi valori
    (già dimensioni e passaggi facoltativi), non le misure, non gli altri sistemi se il target è un sistema."""
    from cercaPattern.functional import ROLE_DATA, ROLE_RULE, ROLE_SESSION, ROLE_SYSTEM, analyze_presence
    df = base_df.assign(COMMISSIONE=np.where(base_df.index % 4 == 0, "", (base_df.index % 613 / 7).round(3).astype(str)))
    t = analyze_presence(make_ctx(df, tmp_path)).set_index("CAMPO")
    assert set(t.index) == {"TOKEN_CLIENT", "ID_DISPOSITIVO", "LAT", "LON", "SESSIONE_ALFA", "REGOLA"}
    assert t.loc["SESSIONE_ALFA", "RUOLO"] == ROLE_SESSION and t.loc["REGOLA", "RUOLO"] == ROLE_RULE and t.loc["TOKEN_CLIENT", "RUOLO"] == ROLE_DATA
    assert t.loc["TOKEN_CLIENT", "ODDS_RATIO_ASSENZA"] > 3 and t.loc["TOKEN_CLIENT", "N_ERRORI_ASSENTE"] == int(
        ((base_df["TOKEN_CLIENT"] == "") & (base_df["LIVELLO_ALFA"] == "ERROR")).sum())
    # target su una colonna descrittiva: nessun sistema è "quello del target", gli esiti dei sistemi sono dati come gli altri
    ctx = make_ctx(df, tmp_path, target_col="TIPO_RICHIESTA", target_value="RESO")
    t = analyze_presence(ctx).set_index("CAMPO")
    assert {"SCORE_ALFA", "DETTAGLIO_ALFA", "DELTA_ESITO"} <= set(t.index) and t.loc["SCORE_ALFA", "RUOLO"] == ROLE_SYSTEM
    assert t.loc["SESSIONE_ALFA", "RUOLO"] == ROLE_SESSION and "COMMISSIONE" not in t.index and "PASSO_CONFERMA" not in t.index


def test_declared_families_group_values_by_rule(base_df, tmp_path):
    """--family: i valori che corrispondono alla regola sono analizzati insieme, anche se non hanno un inizio comune."""
    from cercaPattern.functional import analyze_value_families
    tipo = base_df["TIPO_RICHIESTA"]
    ctx = make_ctx(base_df, tmp_path, families=["TIPO_RICHIESTA:CHIUSURE=reso|DISDETTA", "MOBILE=^MOBILE_", "tipo_richiesta:NESSUNA=xyz", "NON_ESISTE:X=a"])
    fam = analyze_value_families(ctx)
    d = fam[fam["ORIGINE"] == "dichiarata"].set_index("FAMIGLIA")
    assert set(d.index) == {"CHIUSURE", "MOBILE"}                                  # nessun valore, colonna assente: nessuna riga
    hit = tipo.isin(["RESO", "DISDETTA"])
    assert d.loc["CHIUSURE", "N"] == int(hit.sum()) and d.loc["CHIUSURE", "N_VALORI"] == 2
    assert d.loc["CHIUSURE", "N_ERROR"] == int((hit & (base_df["LIVELLO_ALFA"] == "ERROR")).sum())
    assert d.loc["MOBILE", "DIMENSIONE"] == "CANALE" and d.loc["MOBILE", "N"] == int(base_df["CANALE"].str.startswith("MOBILE_").sum())
    auto = fam[fam["ORIGINE"] == "automatica"]
    assert "ORDINE*" in set(auto["FAMIGLIA"])                                      # ORDINE_STANDARD + ORDINE_ESTERO
    assert not ((auto["DIMENSIONE"] == "CANALE") & (auto["FAMIGLIA"] == "MOBILE*")).any()   # già dichiarata: non compare due volte
    # la colonna target non è una dimensione di sé stessa
    own = make_ctx(base_df, tmp_path, target_col="TIPO_RICHIESTA", target_value="RESO", families=["TIPO_RICHIESTA:RESI=RESO"])
    assert "TIPO_RICHIESTA" not in set(analyze_value_families(own).get("DIMENSIONE", []))
    for bad in ("senza_uguale", "NOME=", "NOME=("):
        with pytest.raises(SystemExit, match="Famiglia"):
            make_ctx(base_df, tmp_path, families=[bad])


def test_numeric_codes_are_not_binned_like_measures(base_df, tmp_path):
    """Un identificativo numerico non è una misura: niente fasce quantiliche. Importi e punteggi sì."""
    from cercaPattern.config import derived_feature_frame
    df = base_df.assign(PROGRESSIVO=range(len(base_df)), CAP=(20100 + base_df.index % 700).astype(str))
    ctx = make_ctx(df, tmp_path)
    derived = set(derived_feature_frame(ctx).columns)
    assert {"IMPORTO_BIN", "GAMMA_SCORE_BIN", "ZETA_SCORE_BIN"} <= derived
    assert not {"ID_UTENTE_BIN", "PROGRESSIVO_BIN", "CAP_BIN", "TIME_KEY_BIN", "LAT_BIN", "LON_BIN"} & derived


def test_extra_critical_values(base_df, tmp_path):
    """--critical-values aggiunge esiti critici a quelli noti, sempre come parole intere."""
    from cercaPattern.config import CRITICAL_RX, critical_regex
    s = pd.Series(["SOSPESO", "IN_SOSPESO", "SOSPESOX", "DENY", "ok", None])
    assert contains_token(s, CRITICAL_RX).tolist() == [False, False, False, True, False, False]
    assert contains_token(s, critical_regex(["sospeso"])).tolist() == [True, True, False, True, False, False]
    assert contains_token(s, critical_regex(["sosp*"])).tolist() == [True, True, True, True, False, False]      # '*' = parole che iniziano così
    assert make_ctx(base_df, tmp_path, critical_values=["VERIFY"]).critical_rx == critical_regex(["VERIFY"])


# ---------------------------------------------------------------------------
# Selezione del target
# ---------------------------------------------------------------------------
def test_target_suggestions(base_df):
    from cercaPattern.select_target import suggest_targets
    s = suggest_targets(base_df)
    assert (s.iloc[0]["COLONNA"], s.iloc[0]["VALORE"]) == ("LIVELLO_ALFA", "ERROR")
    assert "ESITO_BETA" in set(s["COLONNA"]) and s["COLONNA"].is_unique
    assert "ID_RICHIESTA" not in set(s["COLONNA"]) and "IMPORTO" not in set(s["COLONNA"])


def test_missing_target_column_error_suggests_candidates(base_df, tmp_path):
    with pytest.raises(SystemExit) as e:
        make_ctx(base_df.drop(columns=["LIVELLO_ALFA"]), tmp_path)
    assert "Candidati plausibili" in str(e.value) and "ESITO_BETA=KO" in str(e.value)


def test_missing_parts_of_the_target_are_proposed_from_the_data(base_df, tmp_path):
    """Nessun target predefinito: ciò che manca è il candidato più plausibile; se non c'è, un messaggio chiaro."""
    from cercaPattern.select_target import default_target, suggest_targets
    assert default_target(base_df) == ("LIVELLO_ALFA", "ERROR")
    assert default_target(base_df, "ESITO_BETA") == ("ESITO_BETA", "KO")
    assert default_target(base_df, "APP") == ("APP", None)                         # nessun valore ha l'aspetto di un esito
    assert default_target(base_df, target_values=["ko"]) == ("ESITO_BETA", None)   # la colonna che contiene il valore chiesto
    assert default_target(base_df, target_values=["non_esiste"]) == (None, None)
    ctx = make_ctx(base_df, tmp_path, target_value=None)
    assert (ctx.target_col, ctx.target_values) == ("LIVELLO_ALFA", ["ERROR"])
    rest = base_df.drop(columns=["LIVELLO_ALFA"])                                  # senza quella colonna: il candidato successivo
    ctx = make_ctx(rest, tmp_path, target_value=None)
    assert (ctx.target_col, ctx.target_values[0]) == default_target(rest) and ctx.target_col in set(suggest_targets(base_df)["COLONNA"])
    ctx = make_ctx(base_df, tmp_path, target_value="ko")
    assert (ctx.target_col, ctx.target_values) == ("ESITO_BETA", ["KO"])
    with pytest.raises(SystemExit, match="--target-value"):
        make_ctx(base_df, tmp_path, target_col="APP", target_value=None)
    neutral = pd.DataFrame({"ID": range(50), "A": ["x", "y"] * 25, "B": ["p", "q", "r", "s", "t"] * 10})
    with pytest.raises(SystemExit, match="Colonna target non trovata"):
        make_ctx(neutral, tmp_path, target_value=None)


def test_interactive_listing_shows_candidates(base_df):
    from cercaPattern.select_target import select_target
    printed = []
    select_target(base_df, input_fn=lambda _p: "", print_fn=printed.append)
    assert any("Candidati plausibili" in str(p) and "LIVELLO_ALFA = ERROR" in str(p) for p in printed)


def test_target_values_ignore_case_spaces_and_duplicates(tmp_path):
    df = pd.DataFrame({"ID": range(6), "ESITO": [" ko ", "KO", "Ko", "ok", "OK", "high"]})
    ctx = make_ctx(df, tmp_path, target_col="esito", target_value="ko, KO ,high")
    assert ctx.target_values == ["KO", "HIGH"] and ctx.n_errors == 4 and ctx.target_col == "ESITO"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def test_excel_survives_control_characters_and_formula_like_text(tmp_path):
    """Un carattere di controllo nel CSV faceva fallire l'intero Excel; un testo che inizia per '=' diventava una formula."""
    import openpyxl
    from cercaPattern.reporting import write_excel
    df = pd.DataFrame({"ID": range(40), "NOTE": ["=1+1", "ok\x00\x07ko", "città"] + ["x"] * 37,
                       "LIVELLO_ALFA": ["ERROR"] * 10 + ["LOW"] * 30})
    ctx = make_ctx(df, tmp_path)
    ctx.meta["executive_summary_text"] = "# Titolo\nriga"
    ctx.add_table("99_prova", pd.DataFrame({"TESTO": ["=1+1", "ok\x00\x07ko", "città", None], "NUMERO": [1.5, np.nan, np.inf, 2],
                                            "DATA": [pd.Timestamp("2026-09-14 10:00"), pd.NaT, pd.NaT, pd.NaT], "LISTA": [(1, 2), None, None, None]}))
    ctx.add_table("07_cluster_assegnazioni", pd.DataFrame({"A": range(30)}))
    p = write_excel(ctx, row_level_max=12)
    wb = openpyxl.load_workbook(p)
    ws = wb["99_prova"]
    assert ws["A2"].value == "=1+1" and ws["A2"].data_type == "s"           # testo, non formula
    assert ws["A3"].value == "okko" and ws["A4"].value == "città" and ws["A5"].value is None
    assert ws["B2"].value == 1.5 and ws["B3"].value is None and ws["B4"].value == "inf"
    assert ws["C2"].value.hour == 10 and ws["D2"].value == "(1, 2)"
    assert wb["07_cluster_assegnazioni"].max_row == 13                      # intestazione + 12 righe: il resto è nel CSV
    assert wb.sheetnames[:2] == ["00_Executive_Summary", "00_Findings"] and ws.freeze_panes == "A2"


def test_markdown_to_html_escapes_and_formats():
    from cercaPattern.reporting import _md_to_html
    h = _md_to_html("# Titolo <b>\n- **A_B=1** e `codice`\n- _nota_\n\n| X | Y |\n|---|---|\n| 1 | <2 |\n1. primo\n---")
    assert "<h1>Titolo &lt;b&gt;</h1>" in h and "<strong>A_B=1</strong>" in h and "<code>codice</code>" in h and "<em>nota</em>" in h
    assert "<th>X</th>" in h and "<td>&lt;2</td>" in h and "<ol>" in h and "<hr>" in h and h.count("<ul>") == h.count("</ul>") == 1


def test_csv_outputs_open_cleanly_in_excel(base_df, tmp_path):
    from cercaPattern.reporting import write_csvs
    df = base_df.head(400).copy()
    df["NOTA"] = "città più è"
    ctx = make_ctx(df, tmp_path)
    write_csvs(ctx)
    raw = (ctx.out_dir / "livello_alfa_error_righe_target.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")                                  # BOM: Excel riconosce UTF-8
    back = load_csv(ctx.out_dir / "livello_alfa_error_righe_target.csv")
    assert list(back.columns) == list(ctx.df.columns) and back["NOTA"].iloc[0] == "città più è"
