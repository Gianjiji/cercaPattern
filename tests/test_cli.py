"""Test della CLI / drag & drop: percorsi di input e output, più file, pausa e apertura report in modalità eseguibile.

La pipeline vera è sostituita da un finto `run_pipeline` (veloce) tranne che nel test end-to-end finale.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_analysis  # noqa: E402
from conftest import assert_outputs  # noqa: E402


@pytest.fixture(autouse=True)
def no_local_options(monkeypatch, tmp_path_factory):
    """I test non devono dipendere da un eventuale file di opzioni locali accanto allo script."""
    empty = tmp_path_factory.mktemp("app")
    monkeypatch.setattr(run_analysis, "app_dir", lambda: empty)
    return empty


@pytest.fixture
def fake_pipeline(monkeypatch, tmp_path):
    """Sostituisce run_pipeline: registra le chiamate e crea un report HTML finto nella cartella di output."""
    calls = []

    def _fake(input_path, out_dir, **kw):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "livello_alfa_error_report.html").write_text("<html></html>")
        calls.append({"input": Path(input_path), "out": out_dir, **kw})
        return SimpleNamespace(meta={"executive_summary_text": f"summary {Path(input_path).name}"})

    monkeypatch.setattr(run_analysis, "run_pipeline", _fake)
    return calls


def _csv(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ID;ESITO\n1;ERROR\n2;LOW\n", encoding="utf-8")
    return path


def test_default_output_dir_next_to_csv(tmp_path):
    csv = tmp_path / "dati" / "richieste ottobre.csv"
    assert run_analysis.default_output_dir(csv) == (tmp_path / "dati" / "richieste ottobre_analisi").resolve()


def test_dragdrop_single_file_uses_csv_folder(fake_pipeline, tmp_path, capsys):
    csv = _csv(tmp_path / "cartella con spazi" / "richieste.csv")
    rc = run_analysis.main([str(csv), "--pause", "no", "--open-report", "no"])
    assert rc == 0
    assert len(fake_pipeline) == 1
    assert fake_pipeline[0]["input"] == csv
    assert fake_pipeline[0]["out"] == (csv.parent / "richieste_analisi").resolve()
    assert fake_pipeline[0]["interactive"] is False  # stdin non è un terminale sotto pytest
    out = capsys.readouterr().out
    assert "summary richieste.csv" in out and "Output in:" in out


def test_explicit_output_dir_wins(fake_pipeline, tmp_path):
    csv = _csv(tmp_path / "a.csv")
    rc = run_analysis.main([str(csv), "-o", str(tmp_path / "custom"), "--pause", "no", "--open-report", "no"])
    assert rc == 0 and fake_pipeline[0]["out"] == tmp_path / "custom"


def test_dragdrop_multiple_files_each_own_folder(fake_pipeline, tmp_path):
    a = _csv(tmp_path / "x" / "a.csv")
    b = _csv(tmp_path / "y" / "b.csv")
    rc = run_analysis.main([str(a), str(b), "--pause", "no", "--open-report", "no"])
    assert rc == 0
    assert [c["out"] for c in fake_pipeline] == [(a.parent / "a_analisi").resolve(), (b.parent / "b_analisi").resolve()]


def test_multiple_files_with_output_dir_get_subfolders(fake_pipeline, tmp_path):
    a = _csv(tmp_path / "a.csv")
    b = _csv(tmp_path / "b.csv")
    run_analysis.main([str(a), str(b), "-o", str(tmp_path / "out"), "--pause", "no", "--open-report", "no"])
    assert [c["out"] for c in fake_pipeline] == [tmp_path / "out" / "a_analisi", tmp_path / "out" / "b_analisi"]


def test_missing_file_returns_2_and_skips(fake_pipeline, tmp_path, capsys):
    rc = run_analysis.main([str(tmp_path / "non_esiste.csv"), "--pause", "no", "--open-report", "no"])
    assert rc == 2 and fake_pipeline == []
    assert "File non trovato" in capsys.readouterr().err


def test_missing_and_existing_files_mixed(fake_pipeline, tmp_path):
    a = _csv(tmp_path / "a.csv")
    rc = run_analysis.main([str(tmp_path / "manca.csv"), str(a), "--pause", "no", "--open-report", "no"])
    assert rc == 0 and len(fake_pipeline) == 1


def test_frozen_mode_pauses_and_opens_report(fake_pipeline, tmp_path, monkeypatch):
    """Come eseguibile (drag & drop): a fine analisi apre il report HTML e attende INVIO."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    csv = _csv(tmp_path / "a.csv")
    prompts, opened = [], []
    rc = run_analysis.main([str(csv)], input_fn=lambda p: prompts.append(p) or "", open_fn=lambda u: opened.append(u))
    assert rc == 0
    assert len(prompts) == 1 and "INVIO" in prompts[0]
    assert len(opened) == 1 and opened[0].startswith("file://") and opened[0].endswith("_report.html")


def test_not_frozen_no_pause_no_open(fake_pipeline, tmp_path, monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    csv = _csv(tmp_path / "a.csv")
    prompts, opened = [], []
    rc = run_analysis.main([str(csv)], input_fn=lambda p: prompts.append(p) or "", open_fn=lambda u: opened.append(u))
    assert rc == 0 and prompts == [] and opened == []


def test_pause_forced_yes_even_if_not_frozen(fake_pipeline, tmp_path, monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    csv = _csv(tmp_path / "a.csv")
    prompts = []
    run_analysis.main([str(csv), "--pause", "yes", "--open-report", "no"], input_fn=lambda p: prompts.append(p) or "")
    assert len(prompts) == 1


def test_pipeline_systemexit_is_reported_and_pauses(tmp_path, monkeypatch, capsys):
    """Errore atteso (es. target mancante) su un file: messaggio leggibile, si prosegue con gli altri, exit 1, pausa."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    calls = []

    def _fake(input_path, out_dir, **kw):
        calls.append(Path(input_path).name)
        if Path(input_path).name == "bad.csv":
            raise SystemExit("Colonna target non trovata")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(meta={"executive_summary_text": "ok"})

    monkeypatch.setattr(run_analysis, "run_pipeline", _fake)
    bad = _csv(tmp_path / "bad.csv")
    good = _csv(tmp_path / "good.csv")
    prompts = []
    rc = run_analysis.main([str(bad), str(good), "--open-report", "no"], input_fn=lambda p: prompts.append(p) or "")
    assert rc == 1 and calls == ["bad.csv", "good.csv"]
    assert "ANALISI INTERROTTA per bad.csv" in capsys.readouterr().err
    assert len(prompts) == 1


def test_unexpected_exception_is_logged_not_raised(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(run_analysis, "run_pipeline", lambda *a, **k: 1 / 0)
    csv = _csv(tmp_path / "a.csv")
    rc = run_analysis.main([str(csv), "--pause", "no", "--open-report", "no"])
    assert rc == 1 and "Errore inatteso" in caplog.text


def test_missing_file_in_frozen_mode_pauses(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    prompts = []
    rc = run_analysis.main([str(tmp_path / "manca.csv")], input_fn=lambda p: prompts.append(p) or "")
    assert rc == 2 and len(prompts) == 1


def test_opens_the_report_of_this_analysis_not_an_older_one(tmp_path, monkeypatch):
    """La cartella di output può contenere i report di altri target: va aperto quello appena prodotto."""
    csv = _csv(tmp_path / "a.csv")
    out = tmp_path / "out"
    out.mkdir()
    (out / "aaa_altro_target_report.html").write_text("<html>vecchio</html>")

    def _fake(input_path, out_dir, **kw):
        p = Path(out_dir) / "esito_beta_ko_report.html"
        p.write_text("<html>nuovo</html>")
        return SimpleNamespace(meta={"executive_summary_text": "ok", "report_html_path": str(p)})

    monkeypatch.setattr(run_analysis, "run_pipeline", _fake)
    opened = []
    rc = run_analysis.main([str(csv), "-o", str(out), "--pause", "no", "--open-report", "yes"], open_fn=opened.append)
    assert rc == 0 and len(opened) == 1 and opened[0].endswith("esito_beta_ko_report.html")


@pytest.mark.parametrize("args", [["--min-support", "2"], ["--min-support", "0"], ["--min-confidence", "abc"],
                                  ["--sample-rows", "0"], ["--max-cardinality", "-3"]])
def test_invalid_numeric_options_are_rejected(fake_pipeline, tmp_path, args, capsys):
    csv = _csv(tmp_path / "a.csv")
    with pytest.raises(SystemExit) as e:
        run_analysis.main([str(csv), "--pause", "no", "--open-report", "no", *args])
    assert e.value.code == 2 and fake_pipeline == []
    assert "error" in capsys.readouterr().err


def test_exclude_option_reaches_the_pipeline(fake_pipeline, tmp_path):
    csv = _csv(tmp_path / "a.csv")
    assert run_analysis.main([str(csv), "--pause", "no", "--open-report", "no", "--exclude", "FLUSSO", "COD TIPO"]) == 0
    assert fake_pipeline[0]["exclude_cols"] == ["FLUSSO", "COD TIPO"]


def test_group_and_critical_value_options_reach_the_pipeline(fake_pipeline, tmp_path):
    csv = _csv(tmp_path / "a.csv")
    args = ["--group", "ALFA=ALFA|ALF_", "--group", "BETA=^B_", "--no-auto-groups", "--critical-values", "SOSPESO", "FERMO"]
    assert run_analysis.main([str(csv), "--pause", "no", "--open-report", "no", *args]) == 0
    call = fake_pipeline[0]
    assert call["groups"] == ["ALFA=ALFA|ALF_", "BETA=^B_"] and call["auto_groups"] is False
    assert call["critical_values"] == ["SOSPESO", "FERMO"]


def test_systems_are_recognised_by_default(fake_pipeline, tmp_path):
    csv = _csv(tmp_path / "a.csv")
    assert run_analysis.main([str(csv), "--pause", "no", "--open-report", "no"]) == 0
    assert fake_pipeline[0]["groups"] is None and fake_pipeline[0]["auto_groups"] is True and fake_pipeline[0]["critical_values"] is None
    assert fake_pipeline[0]["families"] is None


def test_local_options_file_gives_defaults_and_command_line_wins(fake_pipeline, tmp_path, no_local_options, capsys):
    """Opzioni scritte nel file accanto allo script: valgono come default, le ripetibili si sommano a quelle date a mano."""
    (no_local_options / run_analysis.OPTIONS_FILE).write_text(
        "# opzioni proprie di questi dati\n\n--group 'ALFA=^A_\\d+'   # un commento in coda\n--critical-values 'SOSP*' FERMO\n"
        "--family 'TIPO:ORDINI=^ORDINE|CONSEGNA ESPRESSA'\n--target-col ESITO\n--min-support 0.01\n", encoding="utf-8")
    csv = _csv(tmp_path / "a.csv")
    base = [str(csv), "--pause", "no", "--open-report", "no"]
    assert run_analysis.main(base) == 0
    call = fake_pipeline[-1]
    assert call["groups"] == ["ALFA=^A_\\d+"] and call["critical_values"] == ["SOSP*", "FERMO"]
    assert call["families"] == ["TIPO:ORDINI=^ORDINE|CONSEGNA ESPRESSA"] and call["target_col"] == "ESITO" and call["min_support"] == 0.01
    assert "Opzioni lette da" in capsys.readouterr().out
    assert run_analysis.main(base + ["--group", "BETA=^B_", "--target-col", "ALTRA", "--critical-values", "X", "--min-support", "0.2"]) == 0
    call = fake_pipeline[-1]
    assert call["groups"] == ["ALFA=^A_\\d+", "BETA=^B_"] and call["target_col"] == "ALTRA"
    assert call["critical_values"] == ["X"] and call["min_support"] == 0.2
    assert run_analysis.main(base + ["--no-local-options"]) == 0
    call = fake_pipeline[-1]
    assert call["groups"] is None and call["families"] is None and call["target_col"] is None and call["min_support"] == 0.005


def test_explicit_options_file(fake_pipeline, tmp_path, capsys):
    csv = _csv(tmp_path / "a.csv")
    opts = tmp_path / "mie opzioni.txt"
    opts.write_text("--exclude FLUSSO 'COD TIPO'\n--family RESI=RESO\n--family 'CANALE:MOBILE=^MOBILE'\n", encoding="utf-8-sig")
    assert run_analysis.main([str(csv), "--pause", "no", "--open-report", "no", "--options", str(opts), "--family", "ALTRO=x"]) == 0
    call = fake_pipeline[0]
    assert call["exclude_cols"] == ["FLUSSO", "COD TIPO"] and call["families"] == ["RESI=RESO", "CANALE:MOBILE=^MOBILE", "ALTRO=x"]


@pytest.mark.parametrize("content", [None, "--opzione-che-non-esiste 3\n", "--family 'apice non chiuso\n", "altro.csv\n"])
def test_bad_options_file_stops_before_any_analysis(fake_pipeline, tmp_path, content, capsys):
    """File mancante, opzione sconosciuta, riga scritta male, un CSV al posto di un'opzione: nessuna analisi, codice 2."""
    csv = _csv(tmp_path / "a.csv")
    opts = tmp_path / "opzioni.txt"
    if content is not None:
        opts.write_text(content, encoding="utf-8")
    rc = run_analysis.main([str(csv), "--pause", "no", "--open-report", "no", "--options", str(opts)])
    assert rc == 2 and fake_pipeline == []
    assert "Opzioni su file non valide" in capsys.readouterr().err


def test_version_option(capsys):
    with pytest.raises(SystemExit) as e:
        run_analysis.main(["--version"])
    assert e.value.code == 0 and "cercaPattern 2." in capsys.readouterr().out


def test_decimal_comma_accepted_in_fractions(fake_pipeline, tmp_path):
    csv = _csv(tmp_path / "a.csv")
    assert run_analysis.main([str(csv), "--pause", "no", "--open-report", "no", "--min-support", "0,01"]) == 0
    assert fake_pipeline[0]["min_support"] == 0.01


def test_summary_with_unencodable_characters_does_not_crash(tmp_path, monkeypatch):
    """Console Windows in cp1252: un carattere fuori tabella nel riepilogo non deve far fallire la stampa finale."""
    import io
    csv = _csv(tmp_path / "a.csv")

    def _fake(input_path, out_dir, **kw):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(meta={"executive_summary_text": "pattern A ≡ B → 日本語 ✓"})

    monkeypatch.setattr(run_analysis, "run_pipeline", _fake)
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="cp1252"))
    rc = run_analysis.main([str(csv), "--pause", "no", "--open-report", "no"])
    sys.stdout.flush()
    assert rc == 0 and b"pattern A ? B" in raw.getvalue()


def test_dragdrop_end_to_end(base_df, tmp_path):
    """Pipeline vera lanciata come da drag & drop: solo il percorso del CSV, output accanto al file."""
    csv = tmp_path / "drop" / "richieste reali.csv"
    csv.parent.mkdir()
    base_df.to_csv(csv, sep=";", index=False)
    rc = run_analysis.main([str(csv), "--non-interactive", "--pause", "no", "--open-report", "no", "--min-support", "0.02"])
    assert rc == 0
    assert_outputs(csv.parent / "richieste reali_analisi")
