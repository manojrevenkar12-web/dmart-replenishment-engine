import numpy as np

from dre.data import m5, stockouts
from dre.features import FeatureBuilder


def test_synthetic_panel_shape(panel):
    assert panel.sales.shape == panel.prices.shape == (60, 520)
    assert set(panel.ids.columns) == {"item_id", "dept_id", "cat_id", "store_id", "state_id"}
    assert panel.snap().shape == panel.sales.shape


def test_stockout_detector_flags_improbable_runs_only():
    rng = np.random.default_rng(1)
    fast = rng.poisson(6, 300).astype(np.float32)
    fast[200:206] = 0  # six zero days for an item selling ~6 a day: a stock-out
    slow = (rng.random(300) < 0.15).astype(np.float32)  # sells on 15% of days: long zero runs are normal
    flags = stockouts.detect(np.stack([fast, slow]))
    assert flags[0, 200:206].all() and flags[0, :200].sum() == 0
    assert flags[1].sum() == 0


def test_stockout_detector_ignores_days_before_launch():
    y = np.zeros((1, 200), np.float32)
    y[0, 120:] = 5
    assert stockouts.detect(y).sum() == 0


def test_features_use_no_future_data(panel):
    """Changing anything after the origin must not change the features of that origin (except
    the planned price, which is known in advance and allowed)."""
    o = 450
    changed = m5.Panel(panel.ids, panel.sales.copy(), panel.prices, panel.calendar)
    changed.sales[:, o + 1 :] = 999
    fb2 = FeatureBuilder(changed, np.pad(stockouts.detect(changed.sales[:, : o + 1]), ((0, 0), (0, panel.t - o - 1))))
    fb1 = FeatureBuilder(panel, np.pad(stockouts.detect(panel.sales[:, : o + 1]), ((0, 0), (0, panel.t - o - 1))))
    r1, r2 = fb1.rows([o], 28), fb2.rows([o], 28)
    np.testing.assert_array_equal(np.nan_to_num(r1.X, nan=-7), np.nan_to_num(r2.X, nan=-7))


def test_feature_values_by_hand(panel, fb):
    o, h = 400, 3
    b = fb.block(o, h)
    y = panel.sales
    np.testing.assert_allclose(b["mean_7"], y[:, o - 6 : o + 1].mean(1), rtol=1e-6)
    np.testing.assert_allclose(b["last"], y[:, o])
    np.testing.assert_allclose(b["same_wday_4"], y[:, [o + h - 7, o + h - 14, o + h - 21, o + h - 28]].mean(1))
    assert (b["h"] == 3).all()


def test_rows_subsampling_is_reproducible(fb):
    a = fb.rows([300, 314], 7, fraction=0.5, seed=4)
    b = fb.rows([300, 314], 7, fraction=0.5, seed=4)
    assert len(a.y) == 2 * 7 * 30
    np.testing.assert_array_equal(a.series, b.series)
