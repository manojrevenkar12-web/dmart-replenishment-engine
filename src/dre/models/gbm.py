"""Global LightGBM forecaster: one Tweedie model for all series and horizons 1..H.

Tweedie loss suits retail units (many zeros, a long right tail). Training rows come from a set of
past origins, so the model sees every horizon in the same situation it will be used in. With
`censored="mask"` days flagged as stock-outs are dropped from the targets: the model then learns
demand, not sales that were capped by an empty shelf.
"""

from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np

from ..features import CATEGORICAL, FeatureBuilder, level
from .base import Forecast, nb_alpha


class LightGBMForecaster:
    def __init__(
        self,
        horizon: int = 28,
        n_origins: int = 16,
        origin_step: int = 14,
        censored: str = "keep",
        fraction: float = 1.0,
        params: dict | None = None,
        name: str | None = None,
        seed: int = 7,
    ):
        assert censored in ("keep", "mask")
        self.horizon, self.n_origins, self.step, self.censored = horizon, n_origins, origin_step, censored
        self.fraction, self.seed = fraction, seed
        self.name = name or ("lightgbm" if censored == "keep" else "lightgbm_demand")
        self.params = {
            "objective": "tweedie",
            "tweedie_variance_power": 1.1,
            "learning_rate": 0.05,
            "n_estimators": 600,
            "num_leaves": 255,
            "min_child_samples": 100,
            "subsample": 0.7,
            "subsample_freq": 1,
            "colsample_bytree": 0.7,
            "reg_lambda": 1.0,
            "max_bin": 127,
            "verbose": -1,
            "random_state": seed,
        } | (params or {})
        self.model: lgb.LGBMRegressor | None = None
        self.alpha_: np.ndarray | None = None
        self.info: dict = {}

    def training_origins(self, origin: int) -> list[int]:
        return [origin - self.horizon - self.step * k for k in range(self.n_origins)]

    def fit(self, fb: FeatureBuilder, origin: int) -> None:
        t0 = time.time()
        r = fb.rows(self.training_origins(origin), self.horizon, fraction=self.fraction, seed=self.seed)
        keep = r.valid & ~np.isnan(r.y)
        if self.censored == "mask":
            keep &= ~r.censored
        cols = r.columns
        self.columns = cols
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(
            r.X[keep],
            r.y[keep],
            init_score=np.log(level(r.X[keep], cols)),  # predict a multiplier on the recent level (log link)
            feature_name=cols,
            categorical_feature=[c for c in CATEGORICAL if c in cols],
        )
        # in-sample dispersion from the most recent training origin; replaced by out-of-sample
        # dispersion when the backtest calibrates the model on the previous fold
        last = r.origin == max(r.origin)
        pred = self._mu(r.X[last])
        y = np.full((fb.p.n, self.horizon), np.nan)
        mu = np.full((fb.p.n, self.horizon), np.nan)
        y[r.series[last], r.h[last] - 1] = r.y[last]
        mu[r.series[last], r.h[last] - 1] = pred
        self.alpha_ = nb_alpha(y, mu, fb.codes["dept"] * 100 + fb.codes["store"])
        self.info = {
            "train_rows": int(keep.sum()),
            "train_seconds": round(time.time() - t0, 1),
            "censored_rows_dropped": int((r.censored & r.valid).sum()) if self.censored == "mask" else 0,
        }

    def predict(self, fb: FeatureBuilder, origin: int, horizon: int) -> Forecast:
        r = fb.rows([origin], horizon)
        mu = self._mu(r.X)
        mean = np.zeros((fb.p.n, horizon))
        mean[r.series, r.h - 1] = mu
        listed = ~np.isnan(fb.p.prices[:, origin + 1 : origin + horizon + 1])  # not listed that day: no demand
        mean = np.where(listed, mean, 0.0)
        return Forecast(mean, np.repeat(self.alpha_[:, None], horizon, 1), origin, self.name)

    def _mu(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.model.booster_.predict(X, raw_score=True) + np.log(level(X, self.columns)))

    def importance(self) -> dict[str, float]:
        g = self.model.booster_.feature_importance("gain")
        return dict(sorted(zip(self.columns, (g / g.sum()).round(4).tolist(), strict=True), key=lambda kv: -kv[1]))
