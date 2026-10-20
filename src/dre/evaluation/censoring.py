"""Does correcting for stock-outs matter? A controlled experiment on real data.

Real stock-outs in M5 are unlabelled, so their effect cannot be measured directly. Here stock-out
episodes (3–7 days of zero sales) are planted into the training history of the faster-moving items
of one store, where the true sales are known. Then:

* detection: the share of planted days the detector flags;
* bias: the same model trained on the damaged history with censored days kept vs masked,
  compared with one trained on the clean history, on the untouched test month.
"""

from __future__ import annotations

import numpy as np

from ..data import stockouts
from ..data.m5 import Panel
from ..features import FeatureBuilder
from ..models.gbm import LightGBMForecaster


def plant(sales: np.ndarray, eligible: np.ndarray, start: int, end: int, share: float, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = sales.copy()
    planted = np.zeros_like(sales, bool)
    days = end - start
    for i in np.flatnonzero(eligible):
        target = int(share * days)
        hit = 0
        while hit < target:
            length = int(rng.integers(3, 8))
            s = int(rng.integers(start, end - length))
            if planted[i, max(start, s - 7) : s + length + 7].any():
                continue
            planted[i, s : s + length] = True
            hit += length
    out[planted] = 0.0
    return out, planted


def run(panel: Panel, cfg: dict, horizon: int, origin: int, log=print) -> dict:
    c = cfg["censoring_experiment"]
    so_cfg = cfg["forecast"]["stockouts"]
    p = panel.subset(np.flatnonzero((panel.ids.store_id == c["store"]).to_numpy()))
    y = np.nan_to_num(p.sales)
    rate = y[:, origin - 364 : origin + 1].mean(1)
    eligible = rate >= c["min_rate"]
    lgb_cfg = {"n_origins": c["n_origins"], "params": c["params"]}
    first_train_day = origin - horizon - 14 * (c["n_origins"] - 1) - 120
    damaged, planted = plant(p.sales, eligible, first_train_day, origin + 1, c["share_of_days"])
    pd_ = Panel(p.ids, damaged, p.prices, p.calendar, p.meta)
    flags = stockouts.detect(damaged[:, : origin + 1], so_cfg["window"], so_cfg["alpha"], so_cfg["min_run"], so_cfg.get("max_run"))
    flags = np.pad(flags, ((0, 0), (0, p.t - flags.shape[1])))
    pl = planted[:, : origin + 1]
    real_sales_lost = float((p.sales[:, : origin + 1][pl]).sum() / np.nansum(p.sales[eligible, first_train_day : origin + 1]))
    detection = {
        "planted_days": int(pl.sum()),
        "planted_days_flagged": round(float(flags[:, : origin + 1][pl].mean()), 4),
        "planted_episodes_with_any_flag": _episode_recall(pl, flags[:, : origin + 1]),
        "share_of_eligible_sales_removed": round(real_sales_lost, 4),
    }
    act = p.sales[:, origin + 1 : origin + 1 + horizon]
    results = {}
    for name, panel_, mask in (("clean history", p, "keep"), ("damaged, censored days kept", pd_, "keep"), ("damaged, censored days masked", pd_, "mask")):
        so_ = stockouts.detect(
            np.nan_to_num(panel_.sales[:, : origin + 1], nan=-1), so_cfg["window"], so_cfg["alpha"], so_cfg["min_run"], so_cfg.get("max_run")
        )
        so_ = np.pad(so_, ((0, 0), (0, p.t - so_.shape[1])))
        fb = FeatureBuilder(panel_, so_)
        m = LightGBMForecaster(horizon=horizon, censored=mask, **lgb_cfg)
        m.fit(fb, origin)
        f = m.predict(fb, origin, horizon).mean
        e = eligible
        results[name] = {
            "bias_eligible_items": round(float(f[e].sum() / act[e].sum() - 1), 4),
            "wape_eligible_items": round(float(np.abs(f[e] - act[e]).sum() / act[e].sum()), 4),
        }
        log(f"  censoring experiment — {name}: bias {results[name]['bias_eligible_items']:+.3f}")
    return {"store": c["store"], "eligible_items": int(eligible.sum()), "detection": detection, "forecast": results}


def _episode_recall(planted: np.ndarray, flags: np.ndarray) -> float:
    hit = tot = 0
    for i in np.flatnonzero(planted.any(1)):
        d = np.diff(np.r_[0, planted[i].astype(int), 0])
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True):
            tot += 1
            hit += bool(flags[i, s:e].any())
    return round(hit / max(tot, 1), 4)
