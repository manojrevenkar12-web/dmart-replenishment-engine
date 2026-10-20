"""Benchmarks every model must beat before anyone looks at it: seasonal naive (same weekday last
week), a 28-day moving average with a weekday profile, and TSB (Teunter-Syntetos-Babai) for
intermittent demand. Dispersion comes from each method's own one-step errors."""

from __future__ import annotations

import numpy as np

from ..features import FeatureBuilder
from .base import Forecast, nb_alpha

RESID_DAYS = 56


def _alpha(fb: FeatureBuilder, origin: int, one_step: np.ndarray) -> np.ndarray:
    """one_step: (N, RESID_DAYS) in-sample one-step predictions for days origin-55..origin."""
    y = fb.y[:, origin - RESID_DAYS + 1 : origin + 1]
    return nb_alpha(y, one_step, fb.codes["dept"] * 100 + fb.codes["store"])


class SeasonalNaive:
    name = "seasonal_naive"

    def fit(self, fb: FeatureBuilder, origin: int) -> None:
        pass

    def predict(self, fb: FeatureBuilder, origin: int, horizon: int) -> Forecast:
        y = fb.y
        mean = np.stack([y[:, origin + h - 7 * (-(-h // 7))] for h in range(1, horizon + 1)], 1)
        one = y[:, origin - RESID_DAYS + 1 - 7 : origin + 1 - 7]
        a = _alpha(fb, origin, one)
        return Forecast(mean, np.repeat(a[:, None], horizon, 1), origin, self.name)


class MovingAverage:
    """28-day level times a weekday profile estimated per store × department over 8 weeks."""

    name = "moving_average"

    def fit(self, fb: FeatureBuilder, origin: int) -> None:
        pass

    def _profile(self, fb: FeatureBuilder, origin: int) -> np.ndarray:
        g = fb.codes["dept"] * 100 + fb.codes["store"]
        days = np.arange(origin - 55, origin + 1)
        prof = np.ones((fb.p.n, 7))
        for k in np.unique(g):
            sel = g == k
            tot = fb.y[np.ix_(sel, days)].sum(0)
            by = np.array([tot[fb.wday[days] == d].mean() for d in range(7)])
            prof[sel] = by / by.mean() if by.mean() > 0 else 1.0
        return prof

    def predict(self, fb: FeatureBuilder, origin: int, horizon: int) -> Forecast:
        level = (fb.c[:, origin + 1] - fb.c[:, origin - 27]) / 28
        prof = self._profile(fb, origin)
        mean = np.stack([level * prof[:, fb.wday[origin + h]] for h in range(1, horizon + 1)], 1)
        days = np.arange(origin - RESID_DAYS + 1, origin + 1)
        one = np.stack([(fb.c[:, d] - fb.c[:, d - 28]) / 28 * prof[:, fb.wday[d]] for d in days], 1)
        a = _alpha(fb, origin, one)
        return Forecast(mean, np.repeat(a[:, None], horizon, 1), origin, self.name)


class TSB:
    """Teunter-Syntetos-Babai: smooths demand probability and demand size separately, so the
    forecast decays towards zero during long gaps (unlike Croston)."""

    name = "tsb"

    def __init__(self, alpha_p: float = 0.1, alpha_z: float = 0.1, start: int = 365):
        self.ap, self.az, self.start = alpha_p, alpha_z, start

    def fit(self, fb: FeatureBuilder, origin: int) -> None:
        pass

    def predict(self, fb: FeatureBuilder, origin: int, horizon: int) -> Forecast:
        y = fb.y
        s0 = origin - self.start
        prob = (y[:, s0 - 56 : s0] > 0).mean(1)
        nz = np.where(y[:, s0 - 56 : s0] > 0, y[:, s0 - 56 : s0], np.nan)
        cnt = (~np.isnan(nz)).sum(1)
        size = np.where(cnt > 0, np.nansum(nz, 1) / np.maximum(cnt, 1), 1.0)
        hist = np.zeros((fb.p.n, RESID_DAYS))
        for d in range(s0, origin + 1):
            if d > origin - RESID_DAYS:
                hist[:, d - (origin - RESID_DAYS + 1)] = prob * size
            occ = y[:, d] > 0
            prob = prob + self.ap * (occ - prob)
            size = np.where(occ, size + self.az * (y[:, d] - size), size)
        mean = np.repeat((prob * size)[:, None], horizon, 1)
        a = _alpha(fb, origin, hist)
        return Forecast(mean, np.repeat(a[:, None], horizon, 1), origin, self.name)
