"""
Orchestrazione delle fasi. Ogni fase è isolata in try/except: un errore in
una fase viene loggato e registrato come Finding, senza bloccare le altre.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

from .anomaly import run_anomaly
from .clustering import run_clustering
from .config import (AnalysisContext, ColumnMap, Finding, LOG, detect_alias_columns, detect_leakage_columns,
                     detect_perfect_separators)
from .dictionary import build_dictionary
from .error_analysis import run_error_analysis
from .functional import run_functional
from .hypotheses import build_hypotheses
from .loader import load_csv
from .patterns import run_pattern_discovery
from .progress import Progress
from .profiling import run_profiling
from .reporting import write_all_reports
from .select_target import default_target, describe_suggestions, select_target
from .temporal import run_temporal


def parse_target_values(target_value: Optional[str]) -> List[str]:
    """'ERROR' -> ['ERROR']; 'error, high' -> ['ERROR', 'HIGH']; None -> []."""
    return [v.strip().upper() for v in str(target_value or "").split(",") if v.strip()]


def parse_groups(groups) -> Dict[str, str]:
    """Gruppi di colonne dichiarati dall'utente: dizionario nome -> regex, oppure elenco di 'NOME=REGEX'."""
    if not groups:
        return {}
    if isinstance(groups, dict):
        return {str(k): str(v) for k, v in groups.items()}
    out: Dict[str, str] = {}
    for item in groups:
        name, sep, rx = str(item).partition("=")
        if not sep or not name.strip() or not rx.strip():
            raise SystemExit(f"Gruppo non valido '{item}': la forma è NOME=REGEX (es. ALFA=ALFA|ALF_)")
        try:
            re.compile(rx)
        except re.error as e:
            raise SystemExit(f"Gruppo '{name}': espressione regolare non valida ({e})")
        out[name.strip()] = rx.strip()
    return out


def parse_families(families) -> List[tuple]:
    """Famiglie di valori dichiarate: elenco di 'NOME=REGEX' o 'COLONNA:NOME=REGEX' -> [(colonna o None, nome, regex)]."""
    out = []
    for item in families or []:
        if isinstance(item, (tuple, list)) and len(item) == 3:
            out.append(tuple(item))
            continue
        head, sep, rx = str(item).partition("=")
        col, colon, name = head.rpartition(":")
        if not sep or not name.strip() or not rx.strip():
            raise SystemExit(f"Famiglia non valida '{item}': la forma è NOME=REGEX oppure COLONNA:NOME=REGEX")
        try:
            re.compile(rx)
        except re.error as e:
            raise SystemExit(f"Famiglia '{name}': espressione regolare non valida ({e})")
        out.append((col.strip() if colon and col.strip() else None, name.strip(), rx.strip()))
    return out


def build_context(df: pd.DataFrame, out_dir: Path, target_col: Optional[str], target_value,
                  exclude_cols: Optional[List[str]] = None, groups=None, auto_groups: bool = True, families=None,
                  **kw) -> AnalysisContext:
    """target_value: stringa (anche 'A,B') o lista di valori. Il confronto è case-insensitive.
    Ciò che manca del target viene scelto tra i candidati plausibili (vedi ``select_target.default_target``).
    exclude_cols: colonne da tenere fuori da pattern e modelli (restano visibili nelle tabelle).
    groups: gruppi di colonne dichiarati (nome -> regex); auto_groups=False disattiva il riconoscimento automatico."""
    cols = ColumnMap(list(df.columns), groups=parse_groups(groups), auto_groups=auto_groups)
    values = [str(v).strip().upper() for v in target_value] if isinstance(target_value, (list, tuple)) else parse_target_values(target_value)
    values = list(dict.fromkeys(v for v in values if v))     # senza doppioni, ordine conservato
    tc = target_col
    if tc and tc not in df.columns:  # tolleranza su maiuscole/minuscole
        tc = next((c for c in df.columns if c.strip().upper() == str(tc).strip().upper()), tc)
    if not tc:                       # colonna non indicata: la più plausibile (che contenga i valori chiesti, se ci sono)
        tc, proposed = default_target(df, target_values=values)
        if tc:
            values = values or [proposed]
            LOG.info("Colonna target non indicata: scelta %s (il candidato più plausibile)", tc)
    if not tc or tc not in df.columns:
        hint = describe_suggestions(df)
        raise SystemExit(f"Colonna target non trovata (cercata: {target_col or 'nessuna indicata'}). Colonne: {list(df.columns)[:30]}"
                         + (f". Candidati plausibili per --target-col/--target-value: {hint}" if hint else ""))
    if not values:                   # valori non indicati: quello che nella colonna ha l'aspetto di un esito da spiegare
        proposed = default_target(df, tc)[1]
        if proposed is None:
            present = df[tc].value_counts(dropna=False).head(10).to_dict()
            raise SystemExit(f"Nessun valore target indicato per {tc} e nessuno ha l'aspetto di un esito da spiegare: "
                             f"indicarlo con --target-value. Valori presenti: {present}")
        values = [proposed]
        LOG.info("Valore target non indicato: scelto %s = %s", tc, proposed)
    # il confronto ignora maiuscole e spazi: si normalizzano i valori distinti, non ogni riga
    distinct = df[tc].dropna().unique()
    wanted = {v for v in distinct if str(v).strip().upper() in set(values)}
    is_err = df[tc].isin(wanted).fillna(False).astype(bool)
    ctx = AnalysisContext(df=df, out_dir=out_dir, target_col=tc, target_value="|".join(values), target_values=values,
                          cols=cols, is_error=is_err, families=parse_families(families), **kw)
    extra = [{"NOME_LOGICO": f"ruolo:{k}", "COLONNA_FISICA": v or "(non trovata)"}
             for k, v in (("misura", ctx.measure_col), ("sessione", ctx.session_col), ("identificativo", ctx.id_col))]
    summary = cols.summary()
    n_roles = int(summary["NOME_LOGICO"].str.startswith("ruolo:").sum())
    ctx.results["00_mappa_colonne"] = pd.concat([summary.iloc[:n_roles], pd.DataFrame(extra), summary.iloc[n_roles:]], ignore_index=True)
    leak = detect_leakage_columns(ctx)
    if leak:
        ctx.add_finding(Finding("SETUP", "LEAKAGE", f"{len(leak)} colonne escluse dai modelli perché conseguenza del caso target (vuote nel >=99% delle righe {ctx.event_label})",
                                ", ".join(leak), "ALTA"))
    # colonne escluse su richiesta (--exclude): nomi senza distinzione tra maiuscole e minuscole
    by_upper = {c.strip().upper(): c for c in df.columns}
    manual = []
    for name in exclude_cols or []:
        c = by_upper.get(str(name).strip().upper())
        if c is None:
            LOG.warning("Colonna da escludere '%s' non presente nel CSV: ignorata", name)
        elif c != tc and c not in manual:
            manual.append(c)
    ctx.meta["excluded_cols"] = manual
    if manual:
        ctx.meta["leakage_cols"] = leak = leak + [c for c in manual if c not in leak]
        ctx.add_finding(Finding("SETUP", "ESCLUSIONE", f"{len(manual)} colonne escluse dai modelli su richiesta", ", ".join(manual), "ALTA"))
    alias = detect_alias_columns(ctx)
    ctx.meta["alias_cols"] = alias
    if alias:
        ctx.meta["leakage_cols"] = leak = leak + alias
        ctx.add_finding(Finding("SETUP", "ALIAS DEL TARGET", f"{len(alias)} colonne escluse dai modelli perché determinano da sole il valore di {tc}",
                                ", ".join(alias), "ALTA",
                                f"A ogni valore di queste colonne corrisponde un solo valore di {tc}: sono una codifica dello stesso dato e spiegherebbero il caso solo in modo tautologico."))
    perfect = detect_perfect_separators(ctx)
    ctx.meta["perfect_separators"] = perfect
    if perfect:
        ctx.add_finding(Finding("SETUP", "SEPARATORE PERFETTO", f"{', '.join(perfect[:4])}: ogni valore sta tutto dentro o tutto fuori dal caso {ctx.event_label}",
                                f"{len(perfect)} colonne separano da sole il caso analizzato", "MEDIA",
                                f"Se è il fattore che determina il caso, è la spiegazione. Se invece è una codifica dello stesso dato della colonna {tc}, "
                                f"i pattern che la contengono sono tautologie: rilanciare con --exclude {' '.join(perfect[:4])}."))
    LOG.info("Target: %s in %s -> %s", tc, values, ctx.pretty_errors())
    if cols.systems:
        LOG.info("Sistemi riconosciuti nel tracciato: %s", "; ".join(f"{s} ({len(c)} colonne)" for s, c in cols.systems.items()))
    if ctx.is_system_target and ctx.target_group_cols:
        ctx.add_finding(Finding("SETUP", "SISTEMA DEL TARGET", f"{len(ctx.target_group_cols)} colonne dello stesso sistema del target ({ctx.system_key}) trattate come descrizione dell'esito",
                                ", ".join(ctx.target_group_cols), "MEDIA",
                                f"Le colonne categoriche di {ctx.system_key} restano fuori dai modelli. Se invece sono fattori indipendenti, "
                                f"rilanciare con --no-auto-groups oppure dichiarare i gruppi con --group."))
    if not ctx.is_system_target:
        LOG.info("La colonna target non è l'esito di un sistema riconosciuto: le ipotesi di causa (fase 9a) vanno lette come indicazioni generiche")
    return ctx


N_PHASES = 12  # caricamento + setup + 10 fasi


def _phase(ctx: AnalysisContext, name: str, fn: Callable, *a, **kw):
    t0 = time.time()
    LOG.info("==== %s ====", name)
    ctx.progress.begin(name)
    try:
        r = fn(ctx, *a, **kw)
        LOG.info("%s completata in %.1fs", name, time.time() - t0)
        ctx.progress.finish(ok=True)
        return r
    except Exception as e:  # la pipeline continua
        LOG.exception("%s FALLITA: %s", name, e)
        ctx.add_finding(Finding(name, "ERRORE ANALISI", f"Fase non completata: {type(e).__name__}", str(e)[:300], "BASSA"))
        ctx.progress.finish(ok=False, label=f"FALLITA: {type(e).__name__}")
        return None


def run_pipeline(input_path: Path, out_dir: Path, target_col: Optional[str] = None, target_value: Optional[str] = None,
                 sep: Optional[str] = None, encoding: Optional[str] = None, nrows: Optional[int] = None,
                 min_support: float = 0.005, min_confidence: float = 0.2, use_shap: bool = True,
                 max_cardinality: int = 40, sample_rows: int = 250_000, plotly_cdn: bool = True,
                 extra_dims: Optional[list] = None, progress: bool = True, interactive: bool = False,
                 exclude_cols: Optional[list] = None, groups=None, auto_groups: bool = True,
                 critical_values: Optional[list] = None, families: Optional[list] = None) -> AnalysisContext:
    """Esegue l'intera analisi sul target (colonna, valori).

    - `target_col`/`target_value` assenti e `interactive=True`: dopo il caricamento vengono chiesti a terminale.
    - altrimenti ciò che manca è il candidato più plausibile trovato nei dati. `target_value` accetta più valori separati da virgola.
    - `progress=True` mostra una barra per fase se stderr è un terminale.
    - `exclude_cols`: colonne da tenere fuori da pattern e modelli (codifiche del target, campi noti come conseguenza).
    - `groups`: gruppi di colonne dello stesso sistema dichiarati a mano ('NOME=REGEX' o dizionario); `auto_groups=False`
      disattiva il riconoscimento automatico dai nomi delle colonne.
    - `critical_values`: valori da trattare come esito critico degli altri sistemi, oltre a quelli già noti.
    - `families`: famiglie di valori da analizzare insieme ('NOME=REGEX' o 'COLONNA:NOME=REGEX').
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    prog = Progress(N_PHASES, enabled=progress)
    with prog.logging_context():
        try:
            return _run(input_path, out_dir, prog, target_col, target_value, sep, encoding, nrows, min_support,
                        min_confidence, use_shap, max_cardinality, sample_rows, plotly_cdn, extra_dims, interactive,
                        exclude_cols, groups, auto_groups, critical_values, families)
        finally:
            prog.close()


def _run(input_path: Path, out_dir: Path, prog: Progress, target_col, target_value, sep, encoding, nrows,
         min_support, min_confidence, use_shap, max_cardinality, sample_rows, plotly_cdn, extra_dims,
         interactive, exclude_cols=None, groups=None, auto_groups=True, critical_values=None, families=None) -> AnalysisContext:
    prog.begin("Caricamento CSV")
    df = load_csv(input_path, encoding=encoding, sep=sep, nrows=nrows)
    prog.finish(ok=True, label=f"{len(df):,} righe".replace(",", "."))
    if interactive and (not target_col or not target_value):
        target_col, target_value = select_target(df, target_col, target_value)
    prog.begin("Setup contesto e target")
    ctx = build_context(df, out_dir, target_col, target_value, exclude_cols=exclude_cols, groups=groups, auto_groups=auto_groups,
                        families=families, max_cardinality=max_cardinality, sample_rows=sample_rows, progress=prog,
                        critical_values=[str(v) for v in critical_values or []])
    ctx.meta["input_file"] = str(input_path)
    if ctx.n_errors == 0:
        prog.finish(ok=False, label="nessuna riga target")
        LOG.error("Nessuna riga con %s in %s: analisi non significativa", ctx.target_col, ctx.target_values)
        vals = df[ctx.target_col].value_counts(dropna=False).head(10)
        raise SystemExit(f"Nessuna riga target trovata. Valori presenti in {ctx.target_col}: {vals.to_dict()}")
    if ctx.n_errors == ctx.n_rows:
        prog.finish(ok=False, label="tutte le righe sono target")
        raise SystemExit(f"Tutte le {ctx.n_rows:,} righe hanno {ctx.target_col} in {ctx.target_values}: non c'è nulla con cui confrontarle. "
                         f"Scegliere un valore che riguardi solo una parte delle righe.")
    prog.finish(ok=True, label=ctx.pretty_errors())

    profile = _phase(ctx, "FASE 1 - Data profiling", run_profiling)
    if profile is not None:
        _phase(ctx, "FASE 2 - Dizionario dati", lambda c: c.add_table("02_dizionario", build_dictionary(c, profile)))
    else:
        prog.skip("FASE 2 - Dizionario dati", "saltata: profiling fallito")
    _phase(ctx, "FASE 3 - Analisi errori", run_error_analysis, extra_dims=extra_dims)
    _phase(ctx, "FASE 4 - Pattern discovery", run_pattern_discovery, min_support=min_support, min_confidence=min_confidence, use_shap=use_shap)
    _phase(ctx, "FASE 5 - Analisi temporale", run_temporal)
    _phase(ctx, "FASE 6 - Analisi funzionale", run_functional)
    _phase(ctx, "FASE 7 - Clustering", run_clustering)
    _phase(ctx, "FASE 8 - Anomaly detection", run_anomaly)
    _phase(ctx, "FASE 9a - Cause probabili", build_hypotheses)
    _phase(ctx, "FASE 9b - Report", write_all_reports, plotly_cdn=plotly_cdn)
    return ctx
