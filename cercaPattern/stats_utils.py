"""
Strumenti statistici condivisi dalle fasi.

Con decine di dimensioni e centinaia di valori si eseguono centinaia di confronti: senza
intervalli di confidenza e senza correzione per test multipli il report si riempie di
"tassi anomali" che sono solo rumore su pochi casi. Qui ci sono:

- intervallo di Wilson per una proporzione (affidabile anche con pochi casi e tassi estremi);
- test binomiale unilaterale (il tasso osservato supera quello di riferimento?);
- correzione di Benjamini-Hochberg (false discovery rate) sui p-value;
- insiemi di righe come bit impacchettati, per contare in modo esatto le righe che soddisfano
  una combinazione di condizioni anche su milioni di righe.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from scipy import stats

_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def wilson_interval(k, n, z: float = 1.96) -> Tuple[np.ndarray, np.ndarray]:
    """Estremi (inferiore, superiore) dell'intervallo di Wilson per k successi su n prove."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, k / n, 0.0)
        den = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / den
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    lo = np.where(n > 0, np.clip(centre - half, 0, 1), 0.0)
    hi = np.where(n > 0, np.clip(centre + half, 0, 1), 1.0)
    return lo, hi


def binom_excess_pvalue(k, n, p0: float) -> np.ndarray:
    """P(X >= k) con X ~ Binomiale(n, p0): probabilità di osservare per caso almeno k casi
    se il tasso vero fosse quello di riferimento p0."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    p0 = float(min(max(p0, 0.0), 1.0))
    return np.where(n > 0, stats.binom.sf(k - 1, n, p0), 1.0)


def poisson_excess_pvalue(k, expected) -> np.ndarray:
    """P(X >= k) con X ~ Poisson(expected)."""
    k = np.asarray(k, dtype=float)
    expected = np.asarray(expected, dtype=float)
    return np.where(expected > 0, stats.poisson.sf(k - 1, expected), np.where(k > 0, 0.0, 1.0))


def fdr_bh(pvalues: Iterable[float]) -> np.ndarray:
    """q-value di Benjamini-Hochberg (stesso ordine dell'input; i NaN restano NaN)."""
    p = np.asarray(list(pvalues) if not isinstance(pvalues, np.ndarray) else pvalues, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return q
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


# ---------------------------------------------------------------------------
# Insiemi di righe come bit impacchettati
# ---------------------------------------------------------------------------
def pack(mask: np.ndarray) -> np.ndarray:
    """Maschera booleana -> bit impacchettati (1/8 della memoria, AND e conteggi molto più veloci)."""
    return np.packbits(np.asarray(mask, dtype=bool))


def popcount(bits: np.ndarray) -> int:
    """Numero di bit a 1 (= righe nell'insieme)."""
    if hasattr(np, "bitwise_count"):          # numpy >= 2
        return int(np.bitwise_count(bits).sum(dtype=np.int64))
    return int(_POP8[bits].sum(dtype=np.int64))
