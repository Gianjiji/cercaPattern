#!/usr/bin/env python
"""
Entrypoint CLI: analisi "cross" di una colonna target del CSV (colonna e valori si scelgono a terminale
o con --target-col / --target-value; in mancanza si usa il candidato più plausibile trovato nei dati).

Uso da terminale:
    python run_analysis.py dati.csv                                   # output in <cartella csv>/<nome csv>_analisi
    python run_analysis.py dati.csv -o out --target-col ESITO_BETA --target-value KO
    python run_analysis.py a.csv b.csv --non-interactive              # più file, ciascuno nella propria cartella

Drag & drop (eseguibile py2exe/pyinstaller): trascinando uno o più CSV sull'eseguibile Windows li passa
come argomenti; lo script li analizza uno alla volta, scrive l'output accanto a ciascun CSV, apre il report
HTML e attende INVIO prima di chiudere la finestra, così eventuali errori restano leggibili.
"""
from __future__ import annotations

import argparse
import logging
import shlex
import sys
import webbrowser
from pathlib import Path
from typing import Callable, List, Optional

from cercaPattern import __version__
from cercaPattern.pipeline import run_pipeline

LOG = logging.getLogger("cercaPattern.cli")

OPTIONS_FILE = "cercaPattern.opzioni"      # opzioni locali, lette da sole se il file sta accanto allo script o all'eseguibile


def is_frozen() -> bool:
    """True quando gira come eseguibile impacchettato (py2exe, pyinstaller, cx_Freeze)."""
    return bool(getattr(sys, "frozen", False))


def stdin_is_tty() -> bool:
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def default_output_dir(csv_path: Path) -> Path:
    """Cartella di default: accanto al CSV, '<nome csv>_analisi' (es. C:\\dati\\richieste_analisi)."""
    csv_path = Path(csv_path).resolve()
    return csv_path.parent / f"{csv_path.stem}_analisi"


def resolve_output_dirs(inputs: List[Path], output_dir: Optional[Path]) -> List[Path]:
    """Un solo CSV con -o: usa -o. Più CSV con -o: sottocartelle per nome file. Senza -o: accanto a ogni CSV."""
    if output_dir is None:
        return [default_output_dir(p) for p in inputs]
    if len(inputs) == 1:
        return [Path(output_dir)]
    return [Path(output_dir) / f"{Path(p).stem}_analisi" for p in inputs]


def _fraction(text: str) -> float:
    """Argomento tra 0 e 1 (supporto, confidenza)."""
    try:
        v = float(text.replace(",", "."))
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' non è un numero")
    if not 0 < v <= 1:
        raise argparse.ArgumentTypeError(f"{text}: deve essere una quota tra 0 e 1 (es. 0.005 = 0,5%)")
    return v


def _positive_int(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' non è un numero intero")
    if v <= 0:
        raise argparse.ArgumentTypeError(f"{text}: deve essere maggiore di zero")
    return v


def _safe_console() -> None:
    """Su Windows la console (o un file di redirezione) usa cp1252/cp850: un carattere fuori tabella
    in un valore del CSV farebbe fallire la stampa del riepilogo a fine analisi. Meglio un '?'."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def _report_html(ctx, out_dir: Path) -> Optional[Path]:
    """Report HTML dell'analisi appena conclusa. La cartella può contenere i report di altri target:
    si usa il percorso registrato dalla pipeline e, in mancanza, il più recente."""
    p = getattr(ctx, "meta", {}).get("report_html_path")
    if p and Path(p).exists():
        return Path(p)
    found = sorted(out_dir.glob("*_report.html"), key=lambda f: f.stat().st_mtime, reverse=True)
    return found[0] if found else None


def app_dir() -> Path:
    """Cartella dello script, o dell'eseguibile quando è impacchettato."""
    return Path(sys.executable if is_frozen() else __file__).resolve().parent


def read_options_file(path: Path) -> List[str]:
    """Opzioni scritte in un file come sulla riga di comando, una o più per riga. Le righe vuote e
    ciò che segue un '#' sono ignorati; un valore con spazi o barre rovesciate va tra apici singoli."""
    tokens: List[str] = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        try:
            tokens += shlex.split(line, comments=True)
        except ValueError as e:
            raise SystemExit(f"{path}, riga {n}: {e}")
    return tokens


def options_from_files(explicit: Optional[Path] = None, use_local: bool = True) -> dict:
    """Opzioni lette dai file di opzioni, da usare come default: la riga di comando ha l'ultima parola
    (le opzioni ripetibili, come --group e --family, si sommano). Si leggono, nell'ordine, il file
    locale accanto allo script (se esiste e non è escluso con --no-local-options) e quello indicato
    con --options."""
    files = []
    local = app_dir() / OPTIONS_FILE
    if use_local and local.is_file():
        files.append(local)
    if explicit is not None:
        if not Path(explicit).is_file():
            raise SystemExit(f"File di opzioni non trovato: {explicit}")
        files.append(Path(explicit))
    if not files:
        return {}
    tokens = [t for f in files for t in read_options_file(f)]
    file_parser = build_parser(with_input=False)
    ns = file_parser.parse_args(tokens)
    print("Opzioni lette da: " + ", ".join(str(f) for f in files))
    return {k: v for k, v in vars(ns).items() if v != file_parser.get_default(k) and k not in ("options", "no_local_options")}


def build_parser(with_input: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analisi forense di una colonna target del CSV (pattern, cause probabili, perimetro). "
                                            "Senza --target-col/--target-value la scelta viene chiesta a terminale dopo il caricamento. "
                                            "Accetta più CSV (anche trascinati sull'eseguibile). Le opzioni si possono scrivere in un file "
                                            f"(--options); un file '{OPTIONS_FILE}' accanto allo script viene letto da solo.")
    if with_input:
        p.add_argument("input", type=Path, nargs="+", help="uno o più CSV di input")
    p.add_argument("--output-dir", "-o", type=Path, default=None,
                   help="cartella di output (default: <cartella del csv>/<nome csv>_analisi; con più CSV una sottocartella ciascuno)")
    p.add_argument("--sep", default=None, help="separatore (default: auto)")
    p.add_argument("--encoding", default=None, help="encoding (default: auto)")
    p.add_argument("--target-col", default=None, help="colonna oggetto di analisi (se assente viene chiesta; senza terminale: il candidato più plausibile)")
    p.add_argument("--target-value", default=None, help="valore/i della colonna che definiscono il caso da analizzare, separati da virgola "
                                                        "(se assente viene chiesto; senza terminale: il valore più plausibile della colonna)")
    p.add_argument("--non-interactive", "-y", action="store_true", help="non chiedere nulla a terminale: per ciò che manca usa il candidato più plausibile")
    p.add_argument("--nrows", type=_positive_int, default=None, help="limita le righe lette (debug)")
    p.add_argument("--min-support", type=_fraction, default=0.005,
                   help="quota minima di righe che un pattern deve coprire tra le righe target (default 0.005; si adatta se il target è raro)")
    p.add_argument("--min-confidence", type=_fraction, default=0.2,
                   help="confidenza minima dei pattern (default 0.2; si adatta se il target è raro)")
    p.add_argument("--max-cardinality", type=_positive_int, default=40, help="max valori distinti per colonna nei modelli")
    p.add_argument("--sample-rows", type=_positive_int, default=250_000,
                   help="righe max per albero e XGBoost (campionamento stratificato; i pattern sono sempre contati su tutte le righe)")
    p.add_argument("--dims", nargs="*", default=None, help="dimensioni aggiuntive per il tasso di errore")
    p.add_argument("--exclude", nargs="*", default=None, metavar="COL",
                   help="colonne da tenere fuori da pattern e modelli (es. una codifica del target o un campo noto come conseguenza)")
    p.add_argument("--group", action="append", default=None, metavar="NOME=REGEX",
                   help="dichiara un sistema: le colonne il cui nome corrisponde alla regex descrivono l'esito dello stesso componente "
                        "(ripetibile; ha la precedenza sul riconoscimento automatico)")
    p.add_argument("--no-auto-groups", action="store_true", help="non riconoscere i sistemi dai nomi delle colonne (restano quelli dichiarati con --group)")
    p.add_argument("--critical-values", nargs="*", default=None, metavar="VAL",
                   help="valori da considerare esito critico degli altri sistemi, oltre a quelli già noti (DENY, BLOCK, HIGH, KO, ...); "
                        "con '*' finale vale per ogni parola che inizia così")
    p.add_argument("--family", action="append", default=None, metavar="[COLONNA:]NOME=REGEX",
                   help="dichiara una famiglia di valori da analizzare insieme: i valori (di ogni dimensione, o della sola colonna "
                        "indicata) che corrispondono alla regex (ripetibile)")
    p.add_argument("--options", type=Path, default=None, metavar="FILE",
                   help="file con altre opzioni, scritte come sulla riga di comando (una o più per riga, '#' per i commenti)")
    p.add_argument("--no-local-options", action="store_true", help=f"ignora il file '{OPTIONS_FILE}' accanto allo script")
    p.add_argument("--no-shap", action="store_true", help="salta SHAP (usa solo gain XGBoost)")
    p.add_argument("--plotly-inline", action="store_true", help="HTML autonomo (plotly.js inline, ~4MB) invece del CDN")
    p.add_argument("--no-progress", action="store_true", help="disattiva la barra di avanzamento per fase (auto-off fuori da un terminale)")
    p.add_argument("--pause", choices=["auto", "yes", "no"], default="auto",
                   help="attende INVIO prima di chiudere (auto: solo quando gira come eseguibile, es. drag & drop)")
    p.add_argument("--open-report", choices=["auto", "yes", "no"], default="auto",
                   help="apre il report HTML nel browser a fine analisi (auto: solo quando gira come eseguibile)")
    p.add_argument("--log-level", default="INFO", help="livello di log (DEBUG, INFO, WARNING, ...); con WARNING resta visibile solo la barra")
    p.add_argument("--version", action="version", version=f"cercaPattern {__version__}")
    return p


def _flag(value: str) -> bool:
    return value == "yes" or (value == "auto" and is_frozen())


def _pause(input_fn: Callable[[str], str]) -> None:
    try:
        input_fn("\nPremi INVIO per chiudere... ")
    except (EOFError, KeyboardInterrupt):
        pass


def main(argv: Optional[List[str]] = None, input_fn: Callable[[str], str] = input,
         open_fn: Callable[[str], object] = webbrowser.open) -> int:
    """Ritorna 0 se tutti i CSV sono stati analizzati, 1 se almeno uno è fallito, 2 se nessun file esiste."""
    argv = list(sys.argv[1:] if argv is None else argv)
    _safe_console()
    parser = build_parser()
    a = parser.parse_args(argv)
    try:                                   # opzioni scritte su file: valgono come default, poi si rilegge la riga di comando
        from_files = options_from_files(a.options, use_local=not a.no_local_options)
    except SystemExit as e:
        print(f"Opzioni su file non valide{': ' + e.code if isinstance(e.code, str) else ''}", file=sys.stderr)
        if _flag(a.pause):
            _pause(input_fn)
        return 2
    if from_files:
        parser.set_defaults(**from_files)
        a = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, a.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("matplotlib", "shap", "numba", "fontTools"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    pause = _flag(a.pause)
    open_report = _flag(a.open_report)

    inputs = [Path(p) for p in a.input]
    missing = [p for p in inputs if not p.exists()]
    for p in missing:
        print(f"File non trovato: {p}", file=sys.stderr)
    inputs = [p for p in inputs if p.exists()]
    if not inputs:
        if pause:
            _pause(input_fn)
        return 2
    out_dirs = resolve_output_dirs(inputs, a.output_dir)
    interactive = not a.non_interactive and stdin_is_tty()

    failed = 0
    for csv_path, out_dir in zip(inputs, out_dirs):
        print(f"\n>>> Analisi di {csv_path.resolve()}\n    output: {out_dir.resolve()}")
        try:
            ctx = run_pipeline(csv_path, out_dir, target_col=a.target_col, target_value=a.target_value, sep=a.sep,
                               encoding=a.encoding, nrows=a.nrows, min_support=a.min_support, min_confidence=a.min_confidence,
                               use_shap=not a.no_shap, max_cardinality=a.max_cardinality, sample_rows=a.sample_rows,
                               plotly_cdn=not a.plotly_inline, extra_dims=a.dims, progress=not a.no_progress,
                               interactive=interactive, exclude_cols=a.exclude, groups=a.group, auto_groups=not a.no_auto_groups,
                               critical_values=a.critical_values, families=a.family)
        except SystemExit as e:  # errori "attesi" della pipeline (target mancante, zero righe target, ...)
            failed += 1
            print(f"\nANALISI INTERROTTA per {csv_path.name}: {e}", file=sys.stderr)
            continue
        except KeyboardInterrupt:
            print("\nInterrotto dall'utente.", file=sys.stderr)
            failed += 1
            break
        except Exception as e:  # errore inatteso: traccia completa, ma si prosegue con gli altri file
            failed += 1
            LOG.exception("Errore inatteso su %s: %s", csv_path.name, e)
            continue
        print("\n" + ctx.meta.get("executive_summary_text", "")[:4000])
        print(f"\nOutput in: {out_dir.resolve()}")
        html = _report_html(ctx, out_dir)
        if open_report and html is not None:
            try:
                open_fn(html.resolve().as_uri())
            except Exception as e:  # il browser non è essenziale
                LOG.warning("Impossibile aprire il report nel browser: %s", e)
    if pause:
        _pause(input_fn)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
