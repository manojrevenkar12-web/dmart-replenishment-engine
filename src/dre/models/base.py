"""The contract every forecaster in the domain implements.

A squad's model plugs into the backtest, the release gate, the replenishment optimiser and the API
by implementing `fit` and `predict` and returning a `Forecast`: a mean and a negative-binomial
dispersion per series and day. Probabilistic output is mandatory, because ordering decisions need
quantiles, not point forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy import stats

from ..features import FeatureBuilder

QUANTILES = (0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995)  # the M5 uncertainty levels


@dataclass
class Forecast:
    mean: np.ndarray  # (N, H)
    alpha: np.ndarray  # (N, H) NB2 dispersion: Var = mu + alpha * mu^2 (0 = Poisson)
    origin: int
    model: str

    def _nb(self):
        mu = np.maximum(self.mean, 1e-6)
        a = np.maximum(self.alpha, 1e-6)
        r = 1.0 / a
        return r, r / (r + mu)

    def quantiles(self, levels=QUANTILES) -> np.ndarray:
        """(N, H, Q) integer quantiles."""
        r, p = self._nb()
        q = np.stack([stats.nbinom.ppf(level, r, p) for level in levels], -1)
        return np.where(self.mean[..., None] <= 1e-6, 0.0, q)

    def cdf(self, k: np.ndarray) -> np.ndarray:
        r, p = self._nb()
        return stats.nbinom.cdf(k, r, p)

    def sum_over(self, days: slice) -> tuple[np.ndarray, np.ndarray]:
        """Mean and dispersion of demand summed over a range of days (moment matching, days
        treated as independent) — the protection-period demand an order must cover."""
        m = self.mean[:, days].sum(1)
        v = (self.mean[:, days] + self.alpha[:, days] * self.mean[:, days] ** 2).sum(1)
        a = np.where(m > 1e-9, np.maximum(v - m, 0) / np.maximum(m, 1e-9) ** 2, 0.0)
        return m, a


class Forecaster(Protocol):
    name: str

    def fit(self, fb: FeatureBuilder, origin: int) -> None: ...

    def predict(self, fb: FeatureBuilder, origin: int, horizon: int) -> Forecast: ...


def nb_alpha(y: np.ndarray, mu: np.ndarray, groups: np.ndarray | None = None, prior_weight: float = 30.0, level_bins: int = 6) -> np.ndarray:
    """Method-of-moments NB2 dispersion per series from residuals (rows = series, cols = days),
    shrunk towards the median of similar series so short or flat series get a sensible value.

    "Similar" means the same group (e.g. store × department) *and* the same sales-rate band: NB2
    dispersion depends strongly on the level, and pooling a fast mover with slow intermittent
    items would give it their (much larger) dispersion and absurd safety stock."""
    ok = ~np.isnan(y) & ~np.isnan(mu)
    r2 = np.where(ok, (y - mu) ** 2 - mu, 0.0)
    m2 = np.where(ok, mu**2, 0.0)
    n = ok.sum(1)
    a = np.where(m2.sum(1) > 0, r2.sum(1) / np.maximum(m2.sum(1), 1e-9), 0.0).clip(0, 50)
    rate = np.where(ok, mu, 0.0).sum(1) / np.maximum(n, 1)
    edges = np.quantile(np.log1p(rate), np.linspace(0, 1, level_bins + 1)[1:-1]) if len(rate) > level_bins else np.array([])
    band = np.digitize(np.log1p(rate), edges)
    key = band if groups is None else np.asarray(groups) * 100 + band
    prior = np.zeros_like(a)
    for g in np.unique(key):
        sel = key == g
        has = sel & (n > 0)
        prior[sel] = np.median(a[has]) if has.any() else np.median(a[n > 0]) if (n > 0).any() else 0.1
    return (n * a + prior_weight * prior) / (n + prior_weight)
