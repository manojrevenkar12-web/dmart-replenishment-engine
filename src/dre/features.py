"""Feature rows for a direct multi-horizon forecaster.

One row per (series, origin, horizon h): everything is computed from data up to and including the
origin day, plus what is known in advance about the target day (calendar, events, SNAP, the
planned price). The horizon is a feature, so one global model serves day 1 to day 28 and day 1
uses yesterday's sales — what a daily replenishment run needs.

All statistics come from cumulative sums over the (series × day) matrix, so building millions of
rows is a few hundred vectorised operations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data.m5 import Panel

WINDOWS = (7, 14, 28, 56, 112, 364)
CATEGORICAL = ["item", "dept", "store", "state", "wday", "event"]


def level(X: np.ndarray, columns: list[str]) -> np.ndarray:
    """The series' recent demand level: the offset both global models predict a multiplier on.
    Learning relative effects lets tree models follow level shifts they never saw in training."""
    idx = {c: i for i, c in enumerate(columns)}
    m28 = np.clip(X[:, idx["mean_28"]], 0, None)
    u28 = X[:, idx["unconstrained_mean_28"]]
    u28 = np.where(np.isnan(u28), m28, np.clip(u28, 0, None))
    return (0.5 * u28 + 0.5 * m28 + 0.05).astype(np.float64)


@dataclass
class Rows:
    X: np.ndarray  # (rows, features) float32
    y: np.ndarray  # (rows,) target units (NaN when unknown)
    series: np.ndarray  # (rows,) series index
    origin: np.ndarray  # (rows,) origin day index
    h: np.ndarray  # (rows,) horizon
    valid: np.ndarray  # (rows,) listed on the target day (price known and launched)
    censored: np.ndarray  # (rows,) target day flagged as a stock-out
    columns: list[str]

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.X, columns=self.columns)


class FeatureBuilder:
    def __init__(self, panel: Panel, stockouts: np.ndarray | None = None):
        self.p = panel
        n, t = panel.sales.shape
        y = np.nan_to_num(panel.sales).astype(np.float64)
        self.y = y
        self.so = stockouts if stockouts is not None else np.zeros((n, t), bool)
        ok = ~self.so
        self.c = np.concatenate([np.zeros((n, 1)), np.cumsum(y, 1)], 1)
        self.c2 = np.concatenate([np.zeros((n, 1)), np.cumsum(y * y, 1)], 1)
        self.cz = np.concatenate([np.zeros((n, 1)), np.cumsum(y == 0, 1)], 1)
        self.cu = np.concatenate([np.zeros((n, 1)), np.cumsum(y * ok, 1)], 1)
        self.cn = np.concatenate([np.zeros((n, 1)), np.cumsum(ok, 1)], 1)
        self.cs = np.concatenate([np.zeros((n, 1)), np.cumsum(self.so, 1)], 1)
        self.first = panel.first_sale()
        last_sale = np.where(y > 0, np.arange(t)[None, :], -1)
        self.last_sale = np.maximum.accumulate(last_sale, axis=1)
        pr = panel.prices
        self.price_max = np.fmax.accumulate(np.nan_to_num(pr, nan=0.0), axis=1)
        self.snap = panel.snap()
        cal = panel.calendar
        self.wday, self.month, self.mday, self.event = (cal[c].to_numpy() for c in ("wday", "month", "mday", "event_type"))
        ev = np.flatnonzero(self.event > 0)
        nxt = np.searchsorted(ev, np.arange(t))
        self.days_to_event = np.where(nxt < len(ev), ev[np.minimum(nxt, len(ev) - 1)] - np.arange(t), 99).clip(0, 30)
        ids = panel.ids
        self.codes = {
            "item": pd.factorize(ids.item_id, sort=True)[0],
            "dept": pd.factorize(ids.dept_id, sort=True)[0],
            "store": pd.factorize(ids.store_id, sort=True)[0],
            "state": pd.factorize(ids.state_id, sort=True)[0],
        }

    @staticmethod
    def _mean(cum: np.ndarray, s: np.ndarray, o: int, w: int) -> np.ndarray:
        a = max(0, o + 1 - w)
        return (cum[s, o + 1] - cum[s, a]) / (o + 1 - a)

    def block(self, o: int, h: int, series: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Features of all (or the given) series for origin o and horizon h (target day o + h)."""
        s = np.arange(self.p.n) if series is None else series
        t = o + h
        f: dict[str, np.ndarray] = {"h": np.full(len(s), h)}
        for k in ("item", "dept", "store", "state"):
            f[k] = self.codes[k][s]
        f["wday"] = np.full(len(s), self.wday[t])
        f["month"] = np.full(len(s), self.month[t])
        f["mday"] = np.full(len(s), self.mday[t])
        f["event"] = np.full(len(s), self.event[t])
        f["days_to_event"] = np.full(len(s), self.days_to_event[t])
        f["snap"] = self.snap[s, t]
        pt, po = self.p.prices[s, t], self.p.prices[s, o]
        f["price"] = pt
        f["price_vs_origin"] = pt / po
        f["price_vs_max"] = pt / np.where(self.price_max[s, o] > 0, self.price_max[s, o], np.nan)
        y, c = self.y, self.c
        f["last"] = y[s, o]
        for w in WINDOWS:
            f[f"mean_{w}"] = self._mean(c, s, o, w)
        m28 = f["mean_28"]
        f["std_28"] = np.sqrt(np.maximum(self._mean(self.c2, s, o, 28) - m28**2, 0))
        f["zero_share_28"] = self._mean(self.cz, s, o, 28)
        a = max(0, o - 27)
        n_ok = self.cn[s, o + 1] - self.cn[s, a]
        f["unconstrained_mean_28"] = np.where(n_ok > 0, (self.cu[s, o + 1] - self.cu[s, a]) / np.maximum(n_ok, 1), np.nan)
        f["stockout_share_28"] = self._mean(self.cs, s, o, 28)
        k0 = -(-h // 7)  # most recent same weekday at or before the origin
        lags = [t - 7 * k for k in range(k0, k0 + 4)]
        f["same_wday_4"] = y[np.ix_(s, lags)].mean(1)
        f["lag_364"] = y[s, t - 364] if t >= 364 else np.full(len(s), np.nan)
        f["days_since_sale"] = np.where(self.last_sale[s, o] >= 0, o - self.last_sale[s, o], 999).clip(0, 365)
        f["age"] = np.where(self.first[s] <= o, o - self.first[s], -1).clip(-1, 2000)  # launch date only once it has happened
        return f

    def rows(self, origins: list[int], horizon: int, series: np.ndarray | None = None, fraction: float = 1.0, seed: int = 0) -> Rows:
        """Feature rows for every (origin, horizon, series). `fraction` < 1 keeps a random share of
        series per block (for training on long histories within memory); rows are written into one
        preallocated float32 array."""
        s_all = np.arange(self.p.n) if series is None else np.asarray(series)
        tmax = self.p.t - 1
        meta = [(o, h) for o in origins for h in range(1, horizon + 1) if o + h <= tmax]
        rng = np.random.default_rng(seed)
        per = len(s_all) if fraction >= 1 else max(1, int(round(fraction * len(s_all))))
        cols = list(self.block(meta[0][0], meta[0][1], s_all[:1]))
        X = np.empty((len(meta) * per, len(cols)), np.float32)
        series_idx = np.empty(len(meta) * per, np.int64)
        for k, (o, h) in enumerate(meta):
            s = s_all if per == len(s_all) else np.sort(rng.choice(s_all, per, replace=False))
            b = self.block(o, h, s)
            sl = slice(k * per, (k + 1) * per)
            for j, c in enumerate(cols):
                X[sl, j] = b[c]
            series_idx[sl] = s
        origin = np.repeat([m[0] for m in meta], per)
        hh = np.repeat([m[1] for m in meta], per)
        tgt = origin + hh
        y = self.p.sales[series_idx, tgt]
        valid = ~np.isnan(self.p.prices[series_idx, tgt]) & (tgt >= self.first[series_idx])
        censored = self.so[series_idx, tgt]
        return Rows(X, y, series_idx, origin, hh, valid, censored, cols)
