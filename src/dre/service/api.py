"""Order-advice API for store managers and vendor tools.

Reads the champion's forecast and item economics from the registry (written by the nightly batch)
and answers in milliseconds; no model runs at request time.

GET  /health
GET  /v1/model                                   champion metadata (version, backtest, gate)
GET  /v1/forecast/{store_id}/{item_id}           daily mean and 10/50/90% quantiles
POST /v1/orders                                  order advice for a store from its current stock
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .. import registry
from ..explain import llm
from ..models.base import Forecast
from ..replenishment import advisor
from ..replenishment.economics import Economics

app = FastAPI(title="Dmart replenishment engine", version="1.0.0")


class StockLine(BaseModel):
    item_id: str
    units: float = Field(ge=0)
    remaining_life_days: int = Field(ge=1, description="days the units can still be sold, today included")


class OrderRequest(BaseModel):
    store_id: str
    date: str = Field(description="order date (YYYY-MM-DD); delivery the next morning")
    stock: list[StockLine] = []
    explain: bool = True
    service_overrides: dict[str, float] | None = None


class Store:
    def __init__(self, path: Path):
        self.meta = json.loads((path / "metadata.json").read_text())
        fc = pd.read_parquet(path / "forecast.parquet")
        self.dates = sorted(fc.date.unique())
        self.h = len(self.dates)
        eco = pd.read_parquet(path / "economics.parquet")
        self.ids = eco[["item_id", "store_id", "dept_id"]].reset_index(drop=True)
        n = len(eco)
        self.mean = fc["mean"].to_numpy().reshape(n, self.h)
        self.alpha = fc["alpha"].to_numpy().reshape(n, self.h)
        price = eco[[f"price_{k}" for k in range(self.h + 1)]].to_numpy()
        cost = eco[[f"cost_{k}" for k in range(self.h + 1)]].to_numpy()
        delivery = np.zeros((n, 7), bool)
        for i, s in enumerate(eco.delivery_weekdays):
            delivery[i, [int(x) for x in s.split(",") if x]] = True
        self.eco = Economics(
            price,
            cost,
            eco.shelf_life_days.to_numpy(),
            eco.pack.to_numpy(),
            eco.capacity.to_numpy(),
            delivery,
            self.meta["holding_rate_per_day"],
            eco.store_id.to_numpy(),
        )
        self.index = {(s, i): k for k, (s, i) in enumerate(zip(self.ids.store_id, self.ids.item_id, strict=True))}


@lru_cache(maxsize=1)
def store() -> Store:
    root = Path(os.environ.get("DRE_REGISTRY", "out/registry"))
    return Store(registry.champion(root))


@app.get("/health")
def health() -> dict:
    s = store()
    return {"status": "ok", "model": s.meta["model"], "version": s.meta["version"], "forecast_days": [s.dates[0], s.dates[-1]]}


@app.get("/v1/model")
def model() -> dict:
    return store().meta


@app.get("/v1/forecast/{store_id}/{item_id}")
def forecast(store_id: str, item_id: str) -> dict:
    s = store()
    k = s.index.get((store_id, item_id))
    if k is None:
        raise HTTPException(404, f"unknown series {store_id}/{item_id}")
    f = Forecast(s.mean[k : k + 1], s.alpha[k : k + 1], -1, s.meta["model"])
    q = f.quantiles((0.1, 0.5, 0.9))[0]
    return {
        "store_id": store_id,
        "item_id": item_id,
        "model": s.meta["model"],
        "days": [
            {"date": d, "mean": round(float(m), 2), "p10": float(a), "p50": float(b), "p90": float(c)}
            for d, m, (a, b, c) in zip(s.dates, f.mean[0], q, strict=True)
        ],
    }


@app.post("/v1/orders")
def orders(req: OrderRequest) -> dict:
    s = store()
    if req.date not in s.dates[:-1] and req.date != _day_before(s.dates[0]):
        raise HTTPException(422, f"date must be between {_day_before(s.dates[0])} and {s.dates[-2]} (forecast window)")
    rows = np.flatnonzero(s.ids.store_id.to_numpy() == req.store_id)
    if len(rows) == 0:
        raise HTTPException(404, f"unknown store {req.store_id}")
    t = s.dates.index(req.date) + 1 if req.date in s.dates else 0  # day index within the stored window
    lmax = int(s.eco.life.max())
    inv = np.zeros((len(rows), lmax + 1))
    pos = {s.ids.item_id.iloc[r]: j for j, r in enumerate(rows)}
    for line in req.stock:
        if line.item_id not in pos:
            raise HTTPException(422, f"unknown item {line.item_id} in {req.store_id}")
        j = pos[line.item_id]
        left = min(line.remaining_life_days, int(s.eco.life[rows[j]])) - 1  # sellable days left from tomorrow
        if left >= 1:  # units on their last day today expire tonight and are not stock for tomorrow
            inv[j, left] += line.units
    # local day frame: 0 = the forecast origin, 1..H = the forecast dates; t = the order day
    fc = Forecast(s.mean, s.alpha, 0, s.meta["model"])
    weekday = (pd.Timestamp(req.date) + pd.Timedelta(days=1)).weekday()
    cfg = {"stockout_penalty_share": 0.2, "min_service": 0.5, "max_service": 0.98} | (req.service_overrides or {})
    advice = advisor.advise(fc, s.eco, t, rows, inv, weekday, s.ids, cfg)
    out = []
    for a in advice:
        if a["order_units"] <= 0:
            continue
        item = dict(a)
        if req.explain:
            e = llm.explain(a)
            item["explanation"] = e.text
            item["explanation_source"] = e.source
        out.append(item)
    return {
        "store_id": req.store_id,
        "order_date": req.date,
        "model": s.meta["model"],
        "version": s.meta["version"],
        "orders": out,
        "items_considered": len(advice),
    }


def _day_before(d: str) -> str:
    return (pd.Timestamp(d) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
