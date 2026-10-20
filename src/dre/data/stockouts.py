"""Stock-out detection: recorded sales are censored demand.

A run of zero-sales days is flagged as a likely stock-out when it is improbable given how often the
item sold before it: with p0 = the (smoothed) share of zero days in the preceding window, a run of
L zeros has probability p0**L. Runs with probability below `alpha` are flagged.

This is distribution-free (it works for intermittent and fast-moving items alike) and only uses
data up to the start of the run, so the flag is available in production on the day after the run.
Flagged days are excluded from training targets and from the "unconstrained" demand features, so
the model learns demand rather than "demand when it happened to be on the shelf".
"""

from __future__ import annotations

import numpy as np


def detect(sales: np.ndarray, window: int = 56, alpha: float = 1e-3, min_run: int = 3, max_run: int | None = None) -> np.ndarray:
    """Flag improbable zero runs. Runs longer than `max_run` are left unflagged: a product
    missing for months has usually been delisted or replaced, which is not lost demand."""
    y = np.nan_to_num(sales, nan=-1.0)
    n, t = y.shape
    zero = (y == 0).astype(np.float64)
    obs = (y >= 0).astype(np.float64)
    cz = np.concatenate([np.zeros((n, 1)), np.cumsum(zero, 1)], 1)
    co = np.concatenate([np.zeros((n, 1)), np.cumsum(obs, 1)], 1)
    started = np.maximum.accumulate((y > 0), axis=1)  # before the first sale the item is not listed
    out = np.zeros((n, t), bool)
    for i in range(n):
        row = y[i]
        z = np.flatnonzero(row == 0)
        if z.size == 0:
            continue
        breaks = np.flatnonzero(np.diff(z) > 1)
        starts = np.concatenate([[z[0]], z[breaks + 1]])
        ends = np.concatenate([z[breaks], [z[-1]]])
        for s, e in zip(starts, ends, strict=True):
            run = e - s + 1
            if run < min_run or (max_run is not None and run > max_run) or s < window or not started[i, s - 1]:
                continue
            nz = cz[i, s] - cz[i, s - window]
            no = co[i, s] - co[i, s - window]
            p0 = (nz + 1) / (no + 2)
            if run * np.log(p0) < np.log(alpha):
                out[i, s : e + 1] = True
    return out
