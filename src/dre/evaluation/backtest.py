"""Rolling-origin backtest: the one evaluation procedure every model in the domain goes through.

For each fold origin the model is refitted on data up to the origin and forecasts the next H
days. A warm-up fold runs first and is not scored: its out-of-sample errors calibrate the
dispersion of the first scored fold, and every fold's errors calibrate the next one. That is how
the model would run in production (last month's errors set this month's uncertainty), it is the
same for every model, and it never uses the period being scored.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..features import FeatureBuilder
from ..models.base import Forecast, nb_alpha
from . import metrics


@dataclass
class FoldResult:
    origin: int
    wrmsse: float
    levels: dict
    wspl: float
    coverage_50: float
    coverage_95: float
    bias: float
    wape: float
    wape_day1_7: float
    fit_seconds: float
    series_rmsse: np.ndarray = field(repr=False)
    series_weight: np.ndarray = field(repr=False)
    series_spl: np.ndarray = field(repr=False)

    def summary(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("series_")}


@dataclass
class ModelResult:
    name: str
    folds: list[FoldResult]
    last_forecast: Forecast
    last_model: object
    info: dict

    def mean(self, key: str) -> float:
        return round(float(np.mean([getattr(f, key) for f in self.folds])), 4)


def calibrate(fc: Forecast, prev: Forecast, actual: np.ndarray, groups: np.ndarray) -> Forecast:
    """Replace the dispersion with the one implied by the previous fold's out-of-sample errors.

    Errors are measured around the previous forecast after removing its per-series bias: when an
    item's demand level shifts (a new price, a listing change), last month's forecast was biased,
    but the new forecast has already re-learned the level from recent data. Counting that bias as
    noise would give the item absurd safety stock for a month."""
    ratio = np.clip(np.nansum(actual, 1) / np.maximum(prev.mean.sum(1), 1e-9), 0.25, 4.0)
    a = nb_alpha(actual, prev.mean * ratio[:, None], groups)
    return Forecast(fc.mean, np.repeat(a[:, None], fc.mean.shape[1], 1), fc.origin, fc.model)


def run(fb_for: Callable[[int], FeatureBuilder], factory: Callable[[], object], folds: list[int], warmup: int, horizon: int, log=print) -> ModelResult:
    """fb_for(origin) returns the feature builder with stock-out flags computed from data up to
    that origin only (a zero run's length is only known once it has happened)."""
    prev: Forecast | None = None
    out: list[FoldResult] = []
    name, model, fc, info = "", None, None, {}
    for o in [warmup, *folds]:
        model = factory()
        name = model.name
        t0 = time.time()
        fb = fb_for(o)
        p = fb.p
        if o == warmup:
            aggs = metrics.aggregation(p.ids)
            groups = fb.codes["dept"] * 100 + fb.codes["store"]
        model.fit(fb, o)
        fc = model.predict(fb, o, horizon)
        secs = round(time.time() - t0, 1)
        del fb
        if prev is not None:
            fc = calibrate(fc, prev, p.sales[:, prev.origin + 1 : prev.origin + 1 + horizon], groups)
        prev = fc
        if o == warmup:
            continue
        w = metrics.wrmsse(p.ids, p.sales, p.prices, o, fc.mean, aggs)
        q = metrics.pinball(p.ids, p.sales, p.prices, o, fc)
        pt = metrics.point(p.sales, o, fc.mean)
        out.append(
            FoldResult(
                o,
                w["wrmsse"],
                w["levels"],
                q["wspl"],
                q["coverage_50"],
                q["coverage_95"],
                pt["bias"],
                pt["wape"],
                pt["wape_day1_7"],
                secs,
                w["series_rmsse"],
                w["series_weight"],
                q["series_spl"],
            )
        )
        info = getattr(model, "info", {})
        log(f"  {name} origin d_{o + 1}: WRMSSE {w['wrmsse']:.4f}  WSPL {q['wspl']:.4f}  bias {pt['bias']:+.3f}  ({secs}s)")
    return ModelResult(name, out, fc, model, info)
