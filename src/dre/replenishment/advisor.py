"""Order advice for one store and day, with the drivers behind each number (used by the API and
by the explanation layer)."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ..models.base import Forecast
from .economics import Economics
from .simulate import Context, Newsvendor, daily_path, expected_waste_existing, protection_demand


def protection_days(eco: Economics, rows: np.ndarray, weekday_tomorrow: int) -> np.ndarray:
    p = np.ones(len(rows), int)
    for k in range(1, 8):
        later = eco.delivery[rows, (weekday_tomorrow + k) % 7]
        p = np.where((p == k) & ~later, k + 1, p)
    return p


def advise(
    fc: Forecast,
    eco: Economics,
    t: int,
    rows: np.ndarray,
    inv: np.ndarray,
    weekday_tomorrow: int,
    ids,
    cfg: dict,
    calendar=None,
) -> list[dict]:
    """rows: series indices of one store; inv: (len(rows), L+1) stock by remaining life tomorrow."""
    sub = Economics(
        eco.price[rows],
        eco.cost[rows],
        eco.life[rows],
        eco.pack[rows],
        eco.capacity[rows],
        eco.delivery[rows],
        eco.holding_rate,
        eco.store[rows],
    )
    f = Forecast(fc.mean[rows], fc.alpha[rows], fc.origin, fc.model)
    protect = protection_days(eco, rows, weekday_tomorrow)
    on_hand = inv.sum(1)
    order = sub.delivery[:, weekday_tomorrow] & ~np.isnan(sub.price[:, t + 1])
    ctx = Context(t, order, protect, inv, on_hand, sub, np.zeros((len(rows), t + 2)))
    m, a = protection_demand(f, t, protect)
    policy = Newsvendor(lambda _t: f, cfg, True, "perishable")
    expiring = expected_waste_existing(inv, daily_path(f, t, int(protect.max())), protect)
    target = policy.target(ctx, f, on_hand - expiring)
    r = 1 / np.maximum(a, 1e-6)
    tau = np.where(m > 1e-9, stats.nbinom.cdf(target, r, r / (r + np.maximum(m, 1e-9))), 1.0)  # implied service level
    q = np.where(order, policy(ctx), 0)
    raw = np.ceil(np.maximum(target - (on_hand - expiring), 0) / sub.pack - 1e-9) * sub.pack
    out = []
    for j, i in enumerate(rows):
        if not order[j]:
            continue
        p_now, p_next = sub.price[j, t], sub.price[j, t + 1]
        ev = ""
        if calendar is not None:
            names = [n for n in calendar.event_name.iloc[t + 1 : t + 1 + protect[j]] if n]
            ev = ", ".join(names)
        out.append(
            {
                "item_id": ids.item_id.iloc[i],
                "store_id": ids.store_id.iloc[i],
                "order_units": float(q[j]),
                "order_packs": float(q[j] / sub.pack[j]),
                "pack": float(sub.pack[j]),
                "protection_days": float(protect[j]),
                "forecast_units": round(float(m[j]), 1),
                "target_units": float(target[j]),
                "service_level": round(float(tau[j]), 2),
                "on_hand_units": float(on_hand[j]),
                "expiring_units": round(float(expiring[j]), 1),
                "price_change_pct": round(float((p_next / p_now - 1) * 100), 0) if p_now and not np.isnan(p_now) and p_next != p_now else 0.0,
                "event": ev,
                "capped_by_shelf": bool(q[j] < raw[j]),
                "shelf_life_days": float(sub.life[j]),
            }
        )
    return out
