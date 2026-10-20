"""Release gate: when may a challenger replace the champion?

The rules are written down once (config `gate:`) and applied the same way to every squad's model:

1. Accuracy: the challenger's WRMSSE, averaged over the backtest folds, is lower.
2. Significance: the dollar-weighted per-series difference in RMSSE at item × store level, pooled
   over folds, is below zero with 95% confidence (paired bootstrap over series). A gain inside the
   noise does not justify a change in production.
3. Uncertainty: WSPL is not worse by more than `max_wspl_regression`.
4. Bias: the absolute bias is not worse by more than `max_bias_regression` — a biased forecast
   empties or overfills shelves however good its RMSE.
5. Consistency: the challenger wins in at least `min_fold_wins` of the folds.
"""

from __future__ import annotations

import numpy as np

from .backtest import ModelResult


def paired_bootstrap(champ: ModelResult, chall: ModelResult, n_boot: int = 2000, seed: int = 0) -> dict:
    d, w = [], []
    for fc, fh in zip(champ.folds, chall.folds, strict=True):
        ok = ~np.isnan(fc.series_rmsse) & ~np.isnan(fh.series_rmsse)
        d.append((fh.series_rmsse - fc.series_rmsse)[ok])
        w.append(fc.series_weight[ok])
    d, w = np.concatenate(d), np.concatenate(w)
    rng = np.random.default_rng(seed)
    est = float((w * d).sum() / w.sum())
    boots = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, len(d), len(d))
        boots[b] = (w[i] * d[i]).sum() / w[i].sum()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"weighted_rmsse_diff": round(est, 5), "ci95": [round(float(lo), 5), round(float(hi), 5)], "series": int(len(d))}


def decide(champ: ModelResult, chall: ModelResult, rules: dict) -> dict:
    boot = paired_bootstrap(champ, chall)
    wins = sum(fh.wrmsse < fc.wrmsse for fc, fh in zip(champ.folds, chall.folds, strict=True))
    checks = {
        "lower_wrmsse": chall.mean("wrmsse") < champ.mean("wrmsse"),
        "significant": boot["ci95"][1] < 0,
        "wspl_ok": chall.mean("wspl") <= champ.mean("wspl") * (1 + rules["max_wspl_regression"]),
        "bias_ok": abs(chall.mean("bias")) <= abs(champ.mean("bias")) + rules["max_bias_regression"],
        "consistent": wins >= rules["min_fold_wins"],
    }
    return {
        "champion": champ.name,
        "challenger": chall.name,
        "wrmsse": {champ.name: champ.mean("wrmsse"), chall.name: chall.mean("wrmsse")},
        "wspl": {champ.name: champ.mean("wspl"), chall.name: chall.mean("wspl")},
        "bias": {champ.name: champ.mean("bias"), chall.name: chall.mean("bias")},
        "fold_wins": f"{wins}/{len(champ.folds)}",
        "bootstrap": boot,
        "checks": checks,
        "promote": all(checks.values()),
    }
