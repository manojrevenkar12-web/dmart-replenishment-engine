"""Item economics and supply constraints for the replenishment simulation.

M5 has prices but no costs, shelf lives, pack sizes or delivery schedules, so these come from
`config/dre.yaml` (documented assumptions, the same for every policy). Departments stand in for
temperature zones: FOODS_1 as fresh, FOODS_2 as chilled, FOODS_3 as ambient grocery.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.m5 import Panel

PACKS = np.array([1, 2, 4, 6, 12, 24])


@dataclass
class Economics:
    price: np.ndarray  # (N, T) sell price (NaN = not listed)
    cost: np.ndarray  # (N, T) unit purchase cost
    life: np.ndarray  # (N,) shelf life in days once received
    pack: np.ndarray  # (N,) case pack size
    capacity: np.ndarray  # (N,) shelf capacity in units
    delivery: np.ndarray  # (N, 7) bool: supplier delivers on weekday
    holding_rate: float  # daily holding cost as share of unit cost
    store: np.ndarray  # (N,) store code


def build(panel: Panel, origin: int, cfg: dict) -> Economics:
    e = cfg["economics"]
    dept = panel.ids.dept_id.to_numpy()
    zone_of = {d: z for z, ds in e["zone_of_department"].items() for d in ds}
    life = np.array([e["zones"][zone_of[d]]["shelf_life_days"] for d in dept])
    delivery = np.zeros((panel.n, 7), bool)
    for i, d in enumerate(dept):
        delivery[i, e["zones"][zone_of[d]]["delivery_weekdays"]] = True
    rate = np.nan_to_num(panel.sales[:, origin - 27 : origin + 1]).mean(1)
    # smallest pack that is at least half a day's sales: realistic, and keeps fast movers in cases
    pack = PACKS[np.minimum(np.searchsorted(PACKS, 0.5 * rate), len(PACKS) - 1)]
    capacity = np.maximum(e["capacity_min_packs"] * pack, np.ceil(e["capacity_days_of_sales"] * rate)).astype(int)
    price = panel.prices
    cost = price * (1 - e["gross_margin"])
    return Economics(price, cost, life, pack, capacity, delivery, e["holding_rate_per_day"], panel.ids.store_id.to_numpy())
