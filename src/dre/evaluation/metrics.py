"""Accuracy metrics, defined once for every squad.

* WRMSSE — the M5 accuracy metric: RMSE scaled by each series' naive one-step error, computed at
  12 aggregation levels (total, state, store, category, department, ... item × store) and weighted
  by recent dollar sales. Errors that cancel at item level but bias a store's total are caught.
* WSPL — scaled pinball loss of the M5 quantiles at item × store level, dollar-weighted: the
  quality of the distribution the replenishment optimiser actually consumes.
* Coverage of the 50% and 95% intervals, bias (forecast / actual − 1) and WAPE.

Restricting to one category (FOODS) leaves levels 1 and 4 identical; they are kept so the
structure and weights match the M5 definition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from ..models.base import QUANTILES, Forecast

LEVELS = [
    ("total", []),
    ("state", ["state_id"]),
    ("store", ["store_id"]),
    ("category", ["cat_id"]),
    ("department", ["dept_id"]),
    ("state_category", ["state_id", "cat_id"]),
    ("state_department", ["state_id", "dept_id"]),
    ("store_category", ["store_id", "cat_id"]),
    ("store_department", ["store_id", "dept_id"]),
    ("item", ["item_id"]),
    ("item_state", ["item_id", "state_id"]),
    ("item_store", ["item_id", "store_id"]),
]


def aggregation(ids: pd.DataFrame) -> list[tuple[str, sparse.csr_matrix, list[str]]]:
    out = []
    n = len(ids)
    for name, cols in LEVELS:
        if not cols:
            key = pd.Series(["total"] * n)
        else:
            key = ids[cols].astype(str).agg("|".join, axis=1)
        codes, uniq = pd.factorize(key)
        m = sparse.csr_matrix((np.ones(n), (codes, np.arange(n))), shape=(len(uniq), n))
        out.append((name, m, list(uniq)))
    return out


def _scale(hist: np.ndarray, squared: bool) -> np.ndarray:
    """Mean squared (or absolute) one-step naive error from each series' first non-zero day."""
    started = np.maximum.accumulate(hist > 0, axis=1)
    d = np.diff(hist, axis=1)
    use = started[:, :-1]
    e = d**2 if squared else np.abs(d)
    s = np.where(use, e, 0).sum(1) / np.maximum(use.sum(1), 1)
    return np.where(s > 0, s, np.nan)


def dollar_weights(sales: np.ndarray, prices: np.ndarray, origin: int) -> np.ndarray:
    sl = slice(origin - 27, origin + 1)
    return np.nansum(sales[:, sl] * np.nan_to_num(prices[:, sl]), 1)


def wrmsse(ids: pd.DataFrame, sales: np.ndarray, prices: np.ndarray, origin: int, pred: np.ndarray, aggs=None) -> dict:
    """pred: (N, H) means for days origin+1..origin+H. Returns the overall score, the score per
    level and the per-series RMSSE at item × store level (used by the release gate)."""
    h = pred.shape[1]
    hist = np.nan_to_num(sales[:, : origin + 1])
    act = sales[:, origin + 1 : origin + 1 + h]
    w = dollar_weights(sales, prices, origin)
    aggs = aggs or aggregation(ids)
    by_level, total = {}, 0.0
    item_store = None
    for name, m, _ in aggs:
        hs, a, f, wg = m @ hist, m @ act, m @ pred, m @ w
        sc = _scale(hs, True)
        r = np.sqrt(((a - f) ** 2).mean(1) / sc)
        wn = wg / wg.sum()
        ok = ~np.isnan(r)
        s = float((wn[ok] * r[ok]).sum())
        by_level[name] = round(s, 4)
        total += s / len(aggs)
        if name == "item_store":
            item_store = (r, wn)
    return {"wrmsse": round(total, 4), "levels": by_level, "series_rmsse": item_store[0], "series_weight": item_store[1]}


def pinball(ids, sales, prices, origin: int, fc: Forecast, levels=QUANTILES) -> dict:
    h = fc.mean.shape[1]
    act = sales[:, origin + 1 : origin + 1 + h]
    q = fc.quantiles(levels)
    sc = _scale(np.nan_to_num(sales[:, : origin + 1]), False)
    w = dollar_weights(sales, prices, origin)
    w = w / w.sum()
    loss = np.zeros(len(act))
    for j, tau in enumerate(levels):
        d = act - q[..., j]
        loss += np.mean(np.maximum(tau * d, (tau - 1) * d), 1)
    spl = loss / len(levels) / sc
    ok = ~np.isnan(spl)
    lo50, hi50 = levels.index(0.25), levels.index(0.75)
    lo95, hi95 = levels.index(0.025), levels.index(0.975)
    cov50 = ((act >= q[..., lo50]) & (act <= q[..., hi50])).mean()
    cov95 = ((act >= q[..., lo95]) & (act <= q[..., hi95])).mean()
    return {
        "wspl": round(float((w[ok] * spl[ok]).sum()), 4),
        "coverage_50": round(float(cov50), 4),
        "coverage_95": round(float(cov95), 4),
        "series_spl": spl,
    }


def point(sales, origin: int, pred: np.ndarray) -> dict:
    h = pred.shape[1]
    act = sales[:, origin + 1 : origin + 1 + h]
    return {
        "bias": round(float(pred.sum() / act.sum() - 1), 4),
        "wape": round(float(np.abs(pred - act).sum() / act.sum()), 4),
        "wape_day1_7": round(float(np.abs(pred[:, :7] - act[:, :7]).sum() / act[:, :7].sum()), 4),
    }
