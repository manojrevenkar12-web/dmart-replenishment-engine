import numpy as np
import pytest

from dre.evaluation import gate, metrics
from dre.evaluation.backtest import FoldResult, ModelResult
from dre.models.base import Forecast, nb_alpha
from dre.models.baselines import TSB, MovingAverage, SeasonalNaive
from dre.models.gbm import LightGBMForecaster


def test_wrmsse_zero_for_perfect_forecast(panel):
    o = 480
    actual = panel.sales[:, o + 1 : o + 29]
    r = metrics.wrmsse(panel.ids, panel.sales, panel.prices, o, actual)
    assert r["wrmsse"] == 0
    assert len(r["levels"]) == 12


def test_wrmsse_penalises_bias_at_aggregate_levels(panel):
    """Two forecasts with the same item-level absolute error: the one whose errors all point the
    same way must score worse, because the errors add up at store and total level."""
    o = 480
    a = np.nan_to_num(panel.sales[:, o + 1 : o + 29])
    signs = np.where(np.arange(panel.n) % 2 == 0, 1.0, -1.0)[:, None]
    cancel = metrics.wrmsse(panel.ids, panel.sales, panel.prices, o, a + signs)["wrmsse"]
    biased = metrics.wrmsse(panel.ids, panel.sales, panel.prices, o, a + 1.0)["wrmsse"]
    assert biased > cancel


def test_pinball_coverage_of_well_specified_distribution(rng):
    """Poisson data scored with its own distribution: the 95% interval covers ~95%."""
    n, h, o = 400, 28, 100
    mu = rng.uniform(2, 10, n)
    sales = rng.poisson(mu[:, None], (n, o + 1 + h)).astype(np.float32)
    import pandas as pd

    ids = pd.DataFrame({"item_id": [f"i{i}" for i in range(n)], "dept_id": "D", "cat_id": "C", "store_id": "S", "state_id": "X"})
    prices = np.ones_like(sales)
    fc = Forecast(np.repeat(mu[:, None], h, 1), np.full((n, h), 1e-6), o, "true")
    p = metrics.pinball(ids, sales, prices, o, fc)
    assert 0.93 < p["coverage_95"] <= 0.995


def test_nb_alpha_recovers_overdispersion(rng):
    mu = np.full((300, 60), 5.0)
    alpha = 0.5
    y = rng.negative_binomial(1 / alpha, 1 / (1 + alpha * mu))
    est = nb_alpha(y.astype(float), mu)
    assert abs(np.median(est) - alpha) < 0.12


@pytest.mark.parametrize("model", [SeasonalNaive(), MovingAverage(), TSB()])
def test_baselines_shapes_and_non_negative(fb, model):
    f = model.predict(fb, 450, 28)
    assert f.mean.shape == (60, 28) and (f.mean >= 0).all()
    assert (f.alpha >= 0).all()
    q = f.quantiles()
    assert q.shape == (60, 28, 9) and (np.diff(q, axis=-1) >= 0).all()


def test_lightgbm_beats_moving_average_on_synthetic(panel, fb):
    o = 470
    lgb = LightGBMForecaster(horizon=28, n_origins=8, params={"n_estimators": 120, "num_leaves": 31, "min_child_samples": 20})
    lgb.fit(fb, o)
    f_lgb = lgb.predict(fb, o, 28)
    f_ma = MovingAverage().predict(fb, o, 28)
    w_lgb = metrics.wrmsse(panel.ids, panel.sales, panel.prices, o, f_lgb.mean)["wrmsse"]
    w_ma = metrics.wrmsse(panel.ids, panel.sales, panel.prices, o, f_ma.mean)["wrmsse"]
    assert w_lgb < w_ma
    assert lgb.importance()


def test_torch_model_trains_and_predicts(fb):
    pytest.importorskip("torch")
    from dre.models.nn import TorchNB

    m = TorchNB(horizon=14, n_origins=4, epochs=1, batch=512)
    m.fit(fb, 470)
    f = m.predict(fb, 470, 14)
    assert f.mean.shape == (60, 14) and np.isfinite(f.mean).all() and (f.alpha > 0).all()


def _result(name, rmsse, wr):
    folds = [
        FoldResult(o, wr, {}, 0.3, 0.5, 0.95, 0.0, 0.5, 0.5, 1.0, rmsse + 0.001 * k, np.ones_like(rmsse), np.ones_like(rmsse)) for k, o in enumerate((1, 2, 3))
    ]
    return ModelResult(name, folds, None, None, {})


def test_gate_rejects_noise_and_promotes_real_gain(rng):
    rules = {"max_wspl_regression": 0.01, "max_bias_regression": 0.01, "min_fold_wins": 2}
    base = rng.uniform(0.5, 1.5, 2000)
    champ = _result("champ", base, 0.80)
    noise = _result("noise", base + rng.normal(0, 0.05, 2000), 0.799)
    better = _result("better", base - 0.05 + rng.normal(0, 0.02, 2000), 0.75)
    assert not gate.decide(champ, noise, rules)["promote"]
    d = gate.decide(champ, better, rules)
    assert d["promote"] and d["bootstrap"]["ci95"][1] < 0
