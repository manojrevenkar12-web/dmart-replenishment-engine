"""Production monitoring for forecasts: what a squad checks every day after go-live.

* Input data: missing or negative sales, items without a price, share of days flagged as
  stock-outs (a jump points to a supply problem or a broken feed, not a model problem).
* Accuracy once actuals arrive, by store and department: WAPE and bias with alert thresholds
  relative to the backtest, because a forecast that is fine in total can be 15% low in one store.
* Drift: population stability (PSI) of the key inputs between training and scoring rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import FeatureBuilder
from ..models.base import Forecast

DRIFT_FEATURES = ["mean_28", "zero_share_28", "stockout_share_28", "price", "price_vs_origin", "days_since_sale"]


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + 1e-4
    c = np.histogram(cur, edges)[0] / len(cur) + 1e-4
    return float(((c - r) * np.log(c / r)).sum())


def data_checks(fb: FeatureBuilder, origin: int) -> dict:
    s = fb.p.sales[:, origin - 27 : origin + 1]
    return {
        "missing_sales_values": int(np.isnan(s).sum()),
        "negative_sales_values": int((s < 0).sum()),
        "items_listed_without_price": int(np.isnan(fb.p.prices[:, origin]).sum()),
        "stockout_flag_share_last_28d": round(float(fb.so[:, origin - 27 : origin + 1].mean()), 4),
        "stockout_flag_share_prior_year": round(float(fb.so[:, origin - 392 : origin - 27].mean()), 4),
    }


def accuracy(fc: Forecast, panel, backtest_wape: float, rules: dict) -> tuple[pd.DataFrame, list[dict]]:
    o, h = fc.origin, fc.mean.shape[1]
    act = np.nan_to_num(panel.sales[:, o + 1 : o + 1 + h])
    rows, alerts = [], []
    for key in ("store_id", "dept_id"):
        for g, idx in panel.ids.groupby(key).groups.items():
            f, a = fc.mean[idx].sum(), act[idx].sum()
            wape = float(np.abs(fc.mean[idx] - act[idx]).sum() / max(a, 1))
            bias = float(f / max(a, 1) - 1)
            rows.append(
                {
                    "group": key.replace("_id", ""),
                    "value": g,
                    "actual_units": int(a),
                    "forecast_units": int(round(f)),
                    "bias": round(bias, 4),
                    "wape": round(wape, 4),
                }
            )
            if abs(bias) > rules["bias_alert"]:
                alerts.append({"group": g, "kind": "bias", "value": round(bias, 4), "threshold": rules["bias_alert"]})
            if wape > backtest_wape * (1 + rules["wape_degradation_alert"]):
                alerts.append(
                    {"group": g, "kind": "wape", "value": round(wape, 4), "threshold": round(backtest_wape * (1 + rules["wape_degradation_alert"]), 4)}
                )
    return pd.DataFrame(rows), alerts


def drift(fb: FeatureBuilder, train_origins: list[int], origin: int, horizon: int) -> dict:
    ref = fb.rows(train_origins[:4], horizon, fraction=0.25, seed=1).frame()
    cur = fb.rows([origin], horizon).frame()
    out = {}
    for f in DRIFT_FEATURES:
        v = psi(ref[f].to_numpy(), cur[f].to_numpy())
        out[f] = {"psi": round(v, 4), "status": "red" if v >= 0.25 else "amber" if v >= 0.1 else "green"}
    return out
