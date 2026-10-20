"""M5 (Walmart) daily unit sales as a store × item panel.

The M5 competition data covers 3,049 products in 10 stores over 1,969 days (2011-01-29 to
2016-06-19) with prices, SNAP days and events. FOODS (1,437 items × 10 stores = 14,370 series) is
the grocery part and the closest public analogue to a quick-commerce dark-store assortment: many
intermittent series, a few very fast movers, weekly price changes and calendar effects.

Everything downstream works on dense matrices (series × days), which keeps feature building and
policy simulation vectorised and the memory footprint predictable.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://raw.githubusercontent.com/Nixtla/m5-forecasts/main/datasets/m5.zip"
ID_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
EVENT_TYPES = ["Sporting", "Cultural", "National", "Religious"]


@dataclass
class Panel:
    ids: pd.DataFrame  # one row per series: item_id, dept_id, cat_id, store_id, state_id
    sales: np.ndarray  # (N, T) float32 units; NaN where the day is not observed
    prices: np.ndarray  # (N, T) float32 sell price; NaN before the item was listed in the store
    calendar: pd.DataFrame  # T rows: date, wday (0=Mon), month, mday, event_type, snap_{state}
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.sales.shape[0]

    @property
    def t(self) -> int:
        return self.sales.shape[1]

    def subset(self, rows) -> Panel:
        rows = np.asarray(rows)
        return Panel(self.ids.iloc[rows].reset_index(drop=True), self.sales[rows], self.prices[rows], self.calendar, dict(self.meta))

    def snap(self) -> np.ndarray:
        """(N, T) SNAP flag of each series' state."""
        out = np.zeros((self.n, self.t), np.float32)
        for st in self.ids.state_id.unique():
            out[(self.ids.state_id == st).to_numpy()] = self.calendar[f"snap_{st}"].to_numpy(np.float32)
        return out

    def first_sale(self) -> np.ndarray:
        """Index of the first day with a sale (series are 'not launched' before it)."""
        s = np.nan_to_num(self.sales) > 0
        return np.where(s.any(1), s.argmax(1), self.t)


def download(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if (root / "calendar.csv").exists():
        return root
    with urllib.request.urlopen(URL, timeout=300) as r:  # noqa: S310 — fixed public URL
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(root)
    return root


def _calendar(root: Path) -> pd.DataFrame:
    c = pd.read_csv(root / "calendar.csv", parse_dates=["date"])
    out = pd.DataFrame({"date": c.date, "wm_yr_wk": c.wm_yr_wk})
    out["wday"] = c.date.dt.weekday
    out["month"] = c.date.dt.month
    out["mday"] = c.date.dt.day
    ev = c.event_type_1.fillna(c.event_type_2)
    out["event_type"] = ev.map({e: i + 1 for i, e in enumerate(EVENT_TYPES)}).fillna(0).astype(int)
    out["event_name"] = c.event_name_1.fillna("")
    for st in ("CA", "TX", "WI"):
        out[f"snap_{st}"] = c[f"snap_{st}"]
    return out


def load(root: Path, category: str | None = "FOODS", stores: list[str] | None = None, cache: Path | None = None) -> Panel:
    """Load the panel; observed days are d_1..d_1941 plus d_1942..d_1969 when the evaluation truth
    file is present. Results are cached as .npz/.parquet for fast re-runs."""
    key = f"{category or 'ALL'}_{'-'.join(stores) if stores else 'all'}"
    if cache is not None and (cache / f"{key}.npz").exists():
        z = np.load(cache / f"{key}.npz")
        return Panel(pd.read_parquet(cache / f"{key}_ids.parquet"), z["sales"], z["prices"], pd.read_parquet(cache / "calendar.parquet"), {"key": key})
    cal = _calendar(root)
    tr = pd.read_csv(root / "sales_train_evaluation.csv")
    if category:
        tr = tr[tr.cat_id == category]
    if stores:
        tr = tr[tr.store_id.isin(stores)]
    tr = tr.reset_index(drop=True)
    day_cols = [c for c in tr.columns if c.startswith("d_")]
    sales = np.full((len(tr), len(cal)), np.nan, np.float32)
    sales[:, : len(day_cols)] = tr[day_cols].to_numpy(np.float32)
    test = root / "sales_test_evaluation.csv"
    if test.exists():
        te = pd.read_csv(test).merge(tr[ID_COLS].reset_index(), on=ID_COLS).sort_values("index")
        tcols = [c for c in te.columns if c.startswith("d_")]
        first = int(tcols[0][2:]) - 1
        sales[te["index"].to_numpy(), first : first + len(tcols)] = te[tcols].to_numpy(np.float32)
    pr = pd.read_csv(root / "sell_prices.csv")
    pr = pr.merge(tr[["store_id", "item_id"]].reset_index(), on=["store_id", "item_id"])
    weeks = {w: i for i, w in enumerate(sorted(cal.wm_yr_wk.unique()))}
    wk = np.full((len(tr), len(weeks)), np.nan, np.float32)
    wk[pr["index"].to_numpy(), pr.wm_yr_wk.map(weeks).to_numpy()] = pr.sell_price.to_numpy(np.float32)
    prices = wk[:, cal.wm_yr_wk.map(weeks).to_numpy()]
    ids = tr[ID_COLS].copy()
    cal = cal.drop(columns=["wm_yr_wk"])
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache / f"{key}.npz", sales=sales, prices=prices)
        ids.to_parquet(cache / f"{key}_ids.parquet")
        cal.to_parquet(cache / "calendar.parquet")
    return Panel(ids, sales, prices, cal, {"key": key})


def synthetic(n_items: int = 40, stores: tuple[str, ...] = ("CA_1", "TX_1"), days: int = 500, seed: int = 0) -> Panel:
    """A small panel in M5 shape for tests and quick runs: weekly seasonality, intermittent and
    fast-moving items, price promotions and some planted stock-out runs (zeros)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-03", periods=days, freq="D")
    cal = pd.DataFrame({"date": dates, "wday": dates.weekday, "month": dates.month, "mday": dates.day})
    cal["event_type"] = np.where((dates.month == 12) & (dates.day == 25), 3, 0)
    cal["event_name"] = np.where(cal.event_type > 0, "Christmas", "")
    for st in ("CA", "TX", "WI"):
        cal[f"snap_{st}"] = (dates.day <= 10).astype(int)
    rows, sales, prices = [], [], []
    week = np.array([1.0, 0.9, 0.9, 0.95, 1.1, 1.3, 1.25])
    for s in stores:
        for i in range(n_items):
            dept = f"FOODS_{1 + i % 3}"
            base = rng.lognormal(0.0, 1.0) * (3 if i % 5 == 0 else 1)
            price = float(np.round(rng.uniform(1, 8), 2))
            p = np.full(days, price, np.float32)
            promo = rng.random(-(-days // 7)) < 0.08
            p[np.repeat(promo, 7)[:days]] *= 0.8
            lam = base * week[cal.wday] * (p / price) ** -2.0 * np.where(cal.event_type > 0, 0.2, 1.0)
            y = rng.poisson(lam).astype(np.float32)
            launch = rng.integers(0, days // 4) if i % 7 == 0 else 0
            y[:launch] = 0
            p[:launch] = np.nan
            if base > 2 and rng.random() < 0.5:  # planted stock-out
                start = rng.integers(launch + 60, days - 60)
                y[start : start + rng.integers(4, 10)] = 0
            rows.append({"item_id": f"{dept}_{i:03d}", "dept_id": dept, "cat_id": "FOODS", "store_id": s, "state_id": s[:2]})
            sales.append(y)
            prices.append(p)
    return Panel(pd.DataFrame(rows), np.array(sales), np.array(prices), cal, {"key": "synthetic"})
