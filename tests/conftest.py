"""Fixture comuni: dataset sintetico piccolo + runner della pipeline che raccoglie i fallimenti nascosti."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate_sample_data import main as gen_main  # noqa: E402
from cercaPattern.pipeline import run_pipeline  # noqa: E402


@pytest.fixture(scope="session")
def base_df(tmp_path_factory) -> pd.DataFrame:
    p = tmp_path_factory.mktemp("data") / "base.csv"
    gen_main(3000, str(p))
    return pd.read_csv(p, sep=";", dtype=str, keep_default_na=False)


class _Collector(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.records = []

    def emit(self, record):
        self.records.append(self.format(record))


TARGET_COL, TARGET_VALUE = "LIVELLO_ALFA", "ERROR"     # il caso iniettato nel dataset sintetico


@pytest.fixture
def run(tmp_path):
    """Esegue la pipeline su un DataFrame e fallisce se una fase o un sotto-blocco ha sollevato eccezioni.
    Il target è quello del dataset sintetico, salvo indicazione diversa (None = lo sceglie la pipeline)."""
    def _run(df: pd.DataFrame, sep=";", encoding="utf-8", filename="in.csv", expect_fail=False, **kw):
        kw.setdefault("min_support", 0.02)
        kw.setdefault("target_col", TARGET_COL)
        kw.setdefault("target_value", TARGET_VALUE)
        p = tmp_path / filename
        df.to_csv(p, sep=sep, index=False, encoding=encoding)
        out = tmp_path / "out"
        col = _Collector()
        col.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        log = logging.getLogger("cercaPattern")
        log.addHandler(col)
        try:
            ctx = run_pipeline(p, out, **kw)
        finally:
            log.removeHandler(col)
        hidden = [f"{f.fase}: {f.titolo} | {f.evidenza}" for f in ctx.findings if f.categoria == "ERRORE ANALISI"]
        problems = hidden + col.records
        if problems and not expect_fail:
            pytest.fail("Fallimenti interni:\n" + "\n".join(problems))
        return ctx, out
    return _run


PREFIX = "livello_alfa_error"


def assert_outputs(out: Path, html=True, prefix: str = PREFIX):
    for f in (f"{prefix}_report.xlsx", f"{prefix}_righe_target.csv", f"{prefix}_patterns.csv", f"{prefix}_clusters.csv",
              f"{prefix}_anomalies.csv", f"{prefix}_findings.csv", f"{prefix}_executive_summary.md"):
        assert (out / f).exists(), f"manca {f}"
    if html:
        assert (out / f"{prefix}_report.html").exists()
