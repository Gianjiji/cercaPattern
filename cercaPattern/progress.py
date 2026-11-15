"""
Barra di avanzamento per fase (tqdm).

Una barra per ogni fase della pipeline, lasciata a video al termine con il
tempo impiegato ed esito (verde = OK, rosso = fallita, giallo = saltata).
Le fasi lunghe dichiarano i propri sotto-passi (`set_total`) e li avanzano
con `step(label)`; `note(label)` aggiorna solo l'etichetta di cosa sta
girando. I log vengono scritti sopra le barre tramite `tqdm.write`.

Se disabilitata (stderr non è un terminale, `--no-progress`, test) ogni
metodo è un no-op, quindi le fasi possono chiamarla senza controlli.
"""
from __future__ import annotations

import contextlib
import shutil
import sys
import time
from typing import Optional

try:
    from tqdm import tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm
except ImportError:  # pragma: no cover - tqdm è in requirements, ma non blocchiamo l'analisi
    tqdm = None
    logging_redirect_tqdm = None

BAR_FORMAT = "{desc:<36} {percentage:3.0f}%|{bar:25}| {n_fmt}/{total_fmt} [{elapsed}] {unit}"


class Progress:
    def __init__(self, n_phases: int, enabled: bool = True):
        self.n_phases = n_phases
        try:
            tty = sys.stderr is not None and sys.stderr.isatty()
        except (AttributeError, ValueError):  # eseguibile senza console
            tty = False
        self.enabled = bool(enabled and tqdm is not None and tty)
        self._i = 0
        self._bar = None
        self._t0 = 0.0

    # -- ciclo di vita della fase ------------------------------------------
    def begin(self, name: str, total: int = 1) -> None:
        self._i += 1
        self._t0 = time.time()
        if not self.enabled:
            return
        # larghezza letta a ogni fase: terminali senza dimensione (alcune CI/pty) restituiscono 0 -> fallback
        ncols = shutil.get_terminal_size((0, 0)).columns or 110
        self._bar = tqdm(total=max(1, total), desc=f"[{self._i:>2}/{self.n_phases}] {name}"[:36],
                         bar_format=BAR_FORMAT, leave=True, ncols=min(ncols, 130), file=sys.stderr, colour="blue", unit="")

    def finish(self, ok: bool = True, label: Optional[str] = None) -> None:
        if not self._bar:
            return
        elapsed = time.time() - self._t0
        if ok:
            self._bar.n = self._bar.total
            self._bar.colour = "green"
            self._label(label or f"OK {elapsed:.1f}s")
        else:
            self._bar.colour = "red"
            self._label(label or "FALLITA")
        self._bar.close()
        self._bar = None

    def skip(self, name: str, reason: str = "saltata") -> None:
        self.begin(name)
        if self._bar:
            self._bar.colour = "yellow"
            self._label(reason)
            self._bar.close()
            self._bar = None

    # -- sotto-passi della fase corrente ------------------------------------
    def set_total(self, total: int) -> None:
        if self._bar:
            self._bar.total = max(1, int(total))
            self._bar.refresh()

    def note(self, label: str) -> None:
        if self._bar:
            self._label(label)

    def step(self, label: Optional[str] = None, n: int = 1) -> None:
        if self._bar:
            if label:
                self._label(label, refresh=False)
            self._bar.update(n)

    def _label(self, text: str, refresh: bool = True) -> None:
        # `unit` è un campo libero di tqdm: evita la virgola che tqdm antepone a {postfix}
        self._bar.unit = str(text)[:45]
        if refresh:
            self._bar.refresh()

    def close(self) -> None:
        if self._bar:
            self.finish(ok=False, label="interrotta")

    def logging_context(self):
        """Reindirizza i log sopra le barre (no-op se disabilitata)."""
        if self.enabled and logging_redirect_tqdm is not None:
            return logging_redirect_tqdm()
        return contextlib.nullcontext()
