"""Day-by-day replenishment simulation on real demand.

Each simulated day: the morning delivery arrives with full shelf life; demand (the actual M5
sales of that day) is served first-expiry-first-out; unsold units at the end of their life are
written off; in the evening each item whose supplier delivers tomorrow gets an order, rounded to
case packs and capped by shelf capacity. Lost sales are not back-ordered.

Policies differ only in how they set the order. Forecasts are refreshed every evening from the
model trained at the start of the month (models are retrained monthly, scored daily — the usual
production pattern), so model-based policies see the same information as the heuristic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import optimize, sparse, stats

from ..models.base import Forecast
from .economics import Economics

ForecastFn = Callable[[int], Forecast]


@dataclass
class Context:
    t: int  # evening of day t; the order arrives on the morning of t + 1
    order: np.ndarray  # (N,) bool: delivery tomorrow
    protect: np.ndarray  # (N,) days the order must cover (until the next delivery)
    inv: np.ndarray  # (N, L+1) units by remaining life tomorrow morning (column r = sellable for r more days)
    on_hand: np.ndarray  # (N,)
    eco: Economics
    sales: np.ndarray  # (N, T) realised demand; policies may only read columns <= t


# ---------------------------------------------------------------- shared building blocks
def protection_demand(fc: Forecast, t: int, protect: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and NB dispersion of demand over days t+1 .. t+P (per item P), from a forecast issued at
    or before t. Days beyond the forecast horizon continue at the average of its last week."""
    pmax = int(protect.max()) if len(protect) else 0
    m = np.zeros(len(protect))
    v = np.zeros(len(protect))
    width = fc.mean.shape[1]
    tail_mu = fc.mean[:, max(0, width - 7) :].mean(1)
    tail_a = fc.alpha[:, max(0, width - 7) :].mean(1)
    for k in range(1, pmax + 1):
        c = t + k - fc.origin - 1
        mu, al = (fc.mean[:, c], fc.alpha[:, c]) if c < width else (tail_mu, tail_a)
        use = protect >= k
        m += np.where(use, mu, 0)
        v += np.where(use, mu + al * mu**2, 0)
    a = np.where(m > 1e-9, np.maximum(v - m, 0) / np.maximum(m, 1e-9) ** 2, 1e-6)
    return m, a


def _sf(y: np.ndarray, m: np.ndarray, a: np.ndarray) -> np.ndarray:
    """P(D >= y) for a grid y (1, Y) and NB parameters per item (n,)."""
    r = 1 / np.maximum(a, 1e-6)
    p = r / (r + np.maximum(m, 1e-9))
    out = stats.nbinom.sf(y - 1, r[:, None], p[:, None])
    return np.where(m[:, None] > 1e-9, out, (y <= 0).astype(float))


def marginal_values(ctx: Context, fc: Forecast, usable: np.ndarray, cfg: dict, items: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expected profit of the y-th unit of stock (y = 1..Y) for perishable items under FEFO.

    Buying one more unit costs c. It sells within the protection period with probability
    P(D_P >= y) and then earns the price plus the avoided stock-out penalty; otherwise it is carried
    and sells before it expires with probability P(D_life >= y) - P(D_P >= y), saving a future
    purchase (worth c, less holding); otherwise it is written off. Returns the (n, Y) marginal
    values and the grid of y."""
    e, t = ctx.eco, ctx.t
    p = np.nan_to_num(e.price[items, t + 1])
    c = np.nan_to_num(e.cost[items, t + 1])
    s_pen = cfg["stockout_penalty_share"] * p
    sub = Forecast(fc.mean[items], fc.alpha[items], fc.origin, fc.model)
    mp, ap = protection_demand(sub, t, ctx.protect[items])
    ml, al = protection_demand(sub, t, np.maximum(e.life[items], ctx.protect[items]))
    hi = nb_ppf(np.full(len(items), 0.999), mp, ap)
    ymax = int(max(1, (hi + usable[items] + e.pack[items]).max())) + 1
    y = np.arange(1, ymax + 1)[None, :]
    sp, sl = _sf(y, mp, ap), _sf(y, ml, al)
    hold = e.holding_rate * c * ctx.protect[items]
    mv = (p + s_pen)[:, None] * sp + (c - hold)[:, None] * np.maximum(sl - sp, 0) - c[:, None] - hold[:, None] * 0.5
    return mv, y[0]


def nb_ppf(q: np.ndarray, m: np.ndarray, a: np.ndarray) -> np.ndarray:
    r = 1 / np.maximum(a, 1e-6)
    return np.where(m > 1e-9, stats.nbinom.ppf(q, r, r / (r + np.maximum(m, 1e-9))), 0.0)


def expected_waste_existing(inv: np.ndarray, daily: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Units of today's stock expected to expire unsold within the protection period, given expected
    daily demand (N, P) and FEFO issuing."""
    x = inv.astype(float).copy()
    waste = np.zeros(len(x))
    for k in range(daily.shape[1]):
        use = days > k
        need = np.where(use, daily[:, k], 0.0)
        for r in range(1, x.shape[1]):
            take = np.minimum(x[:, r], need)
            x[:, r] -= take
            need -= take
        waste += np.where(use, x[:, 1], 0.0)
        x[:, 1] = np.where(use, 0.0, x[:, 1])
        x[:, 1:-1] = np.where(use[:, None], x[:, 2:], x[:, 1:-1])
        x[:, -1] = np.where(use, 0.0, x[:, -1])
    return waste


def daily_path(fc: Forecast, t: int, pmax: int) -> np.ndarray:
    cols = [min(t + k - fc.origin - 1, fc.mean.shape[1] - 1) for k in range(1, pmax + 1)]
    return fc.mean[:, cols]


def to_packs(units: np.ndarray, pack: np.ndarray) -> np.ndarray:
    return np.ceil(np.maximum(units, 0) / pack - 1e-9) * pack


def cap(q: np.ndarray, ctx: Context) -> np.ndarray:
    room = np.maximum(ctx.eco.capacity - ctx.on_hand, 0)
    return np.minimum(q, np.floor(room / ctx.eco.pack) * ctx.eco.pack)


def critical_ratio(ctx: Context, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Textbook newsvendor: lost margin plus stock-out penalty vs holding cost (no perishability)."""
    p = np.nan_to_num(ctx.eco.price[:, ctx.t + 1])
    c = np.nan_to_num(ctx.eco.cost[:, ctx.t + 1])
    under = p - c + cfg["stockout_penalty_share"] * p
    over = c * ctx.eco.holding_rate * ctx.protect
    tau = np.clip(under / np.maximum(under + over, 1e-9), cfg["min_service"], cfg["max_service"])
    return tau, under, over


# ---------------------------------------------------------------- policies
class CoverDays:
    """The common heuristic: order up to N days of recent average sales, counting all stock on hand."""

    def __init__(self, safety_days: float = 2.0):
        self.name = "cover_days"
        self.safety = safety_days

    def __call__(self, ctx: Context) -> np.ndarray:
        rate = ctx.sales[:, ctx.t - 27 : ctx.t + 1].mean(1)
        target = rate * (ctx.protect + self.safety)
        return cap(to_packs(target - ctx.on_hand, ctx.eco.pack), ctx)


class Newsvendor:
    """`perishable=False`: the textbook newsvendor — order up to the critical-ratio quantile of
    protection-period demand, all stock on hand counts, leftovers only cost holding.
    `perishable=True`: counts only stock expected to sell before it expires and stocks each unit
    whose expected marginal profit is positive given the item's shelf life (see marginal_values)."""

    def __init__(self, forecast: ForecastFn, cfg: dict, perishable: bool, name: str):
        self.forecast, self.cfg, self.perishable, self.name = forecast, cfg, perishable, name

    def usable(self, ctx: Context, fc: Forecast) -> np.ndarray:
        if not self.perishable:
            return ctx.on_hand
        pmax = int(ctx.protect.max())
        return ctx.on_hand - expected_waste_existing(ctx.inv, daily_path(fc, ctx.t, pmax), ctx.protect)

    def target(self, ctx: Context, fc: Forecast, usable: np.ndarray) -> np.ndarray:
        m, a = protection_demand(fc, ctx.t, ctx.protect)
        tau, _, _ = critical_ratio(ctx, self.cfg)
        if not self.perishable:
            return nb_ppf(tau, m, a)
        target = np.zeros(len(m))
        items = np.flatnonzero(ctx.order)
        if len(items):
            mv, _ = marginal_values(ctx, fc, usable, self.cfg, items)
            target[items] = (mv >= 0).sum(1)  # marginal value falls with y: stock every unit that still pays
        lo = nb_ppf(np.full(len(m), self.cfg["min_service"]), m, a)
        hi = nb_ppf(np.full(len(m), self.cfg["max_service"]), m, a)
        return np.clip(target, lo, hi)

    def __call__(self, ctx: Context) -> np.ndarray:
        fc = self.forecast(ctx.t)
        usable = self.usable(ctx, fc)
        return cap(to_packs(self.target(ctx, fc, usable) - usable, ctx.eco.pack), ctx)


class Oracle:
    """Knows tomorrow's demand: the upper bound any forecast can reach under the same constraints."""

    name = "oracle"

    def __call__(self, ctx: Context) -> np.ndarray:
        pmax = int(ctx.protect.max())
        future = np.stack([ctx.sales[:, min(ctx.t + k, ctx.sales.shape[1] - 1)] for k in range(1, pmax + 1)], 1)
        need = np.where(np.arange(1, pmax + 1)[None, :] <= ctx.protect[:, None], future, 0).sum(1)
        usable = ctx.on_hand - expected_waste_existing(ctx.inv, future, ctx.protect)
        return cap(to_packs(need - usable, ctx.eco.pack), ctx)


class CapitalConstrained(Newsvendor):
    """Perishable-aware newsvendor under a store-level cap on inventory capital: a multiple-choice
    knapsack (MILP, HiGHS) picks each item's order — 0, 1, 2 … packs — to maximise expected
    margin minus expected waste and holding cost, so scarce capital goes where it earns most."""

    def __init__(self, forecast: ForecastFn, cfg: dict, cap_days: float, daily_cogs: dict[str, float], name: str | None = None, method: str = "milp"):
        super().__init__(forecast, cfg, True, name or f"capital_cap_{cap_days:g}d")
        assert method in ("milp", "proportional")
        self.cap_days, self.daily_cogs, self.method = cap_days, daily_cogs, method
        self.solves, self.solve_seconds = 0, 0.0

    def __call__(self, ctx: Context) -> np.ndarray:
        import time

        fc = self.forecast(ctx.t)
        usable = np.maximum(self.usable(ctx, fc), 0)
        unconstrained = cap(to_packs(self.target(ctx, fc, usable) - usable, ctx.eco.pack), ctx)
        q = np.zeros(len(usable))
        c = np.nan_to_num(ctx.eco.cost[:, ctx.t + 1])
        for store in np.unique(ctx.eco.store):
            s = ctx.eco.store == store
            budget = self.cap_days * self.daily_cogs[store] - float((c * ctx.on_hand)[s].sum())
            items = np.flatnonzero(s & ctx.order & (unconstrained > 0))
            if budget <= 0 or len(items) == 0:
                continue
            if float((c * unconstrained)[items].sum()) <= budget:
                q[items] = unconstrained[items]
                continue
            if self.method == "proportional":  # the usual workaround: cut every order by the same share
                share = budget / float((c * unconstrained)[items].sum())
                q[items] = np.floor(unconstrained[items] * share / ctx.eco.pack[items]) * ctx.eco.pack[items]
                continue
            t0 = time.time()
            mv, _ = marginal_values(ctx, fc, usable, self.cfg, items)
            q[items] = self._knapsack(items, unconstrained, usable, mv, c, ctx.eco.pack, budget)
            self.solves += 1
            self.solve_seconds += time.time() - t0
        return q

    @staticmethod
    def _knapsack(items, unconstrained, usable, mv, c, pack, budget) -> np.ndarray:
        """Multiple-choice knapsack: for each item pick one option (0, 1, … k packs) whose value is
        the summed marginal profit of the extra units; total purchase cost within the budget."""
        k = (unconstrained[items] / pack[items]).astype(int)
        cum = np.concatenate([np.zeros((len(items), 1)), np.cumsum(mv, 1)], 1)  # value of stocking y units
        ymax = cum.shape[1] - 1
        rows, val, cost_opt, owner = [], [], [], []
        for j, i in enumerate(items):
            u = int(min(max(usable[i], 0), ymax))
            for n in range(k[j] + 1):
                qq = n * pack[i]
                val.append(cum[j, int(min(u + qq, ymax))] - cum[j, u])
                cost_opt.append(c[i] * qq)
                owner.append(j)
                rows.append(qq)
        val, cost_opt, owner, rows = map(np.asarray, (val, cost_opt, owner, rows))
        nvar = len(val)
        pick = sparse.csr_matrix((np.ones(nvar), (owner, np.arange(nvar))), shape=(len(items), nvar))
        cons = [optimize.LinearConstraint(pick, 1, 1), optimize.LinearConstraint(cost_opt[None, :], 0, budget)]
        res = optimize.milp(-val, constraints=cons, integrality=np.ones(nvar), bounds=optimize.Bounds(0, 1), options={"time_limit": 10})
        x = np.round(res.x) if res.x is not None else (rows == 0).astype(float)
        out = np.zeros(len(items))
        np.add.at(out, owner, x * rows)
        return out


# ---------------------------------------------------------------- the simulation
def simulate(policy, eco: Economics, sales: np.ndarray, start: int, days: int, start_weekday: int, init_days: float = 2.0) -> dict:
    """Run `days` days from the morning of day `start` (weekday `start_weekday`, 0 = Monday).
    `sales` holds realised demand."""
    n = len(eco.life)
    lmax = int(eco.life.max())
    inv = np.zeros((n, lmax + 1))
    rate = np.nan_to_num(sales[:, start - 28 : start]).mean(1)
    listed0 = ~np.isnan(eco.price[:, start])
    inv[np.arange(n), eco.life] = np.where(listed0, np.ceil(rate * init_days), 0)
    pending = np.zeros(n)
    y = np.nan_to_num(sales)

    def weekday(day: int) -> int:
        return (start_weekday + day - start) % 7

    acc = {k: np.zeros(n) for k in ("demand", "sold", "waste", "revenue", "cogs", "waste_cost", "capital", "stockout_days", "ordered", "orders")}
    for d in range(start, start + days):
        # morning: delivery
        inv[np.arange(n), eco.life] += pending
        pending = np.zeros(n)
        listed = ~np.isnan(eco.price[:, d])
        dem = np.where(listed, y[:, d], 0.0)
        need = dem.copy()
        for r in range(1, lmax + 1):  # FEFO
            take = np.minimum(inv[:, r], need)
            inv[:, r] -= take
            need -= take
        sold = dem - need
        price = np.nan_to_num(eco.price[:, d])
        cost = np.nan_to_num(eco.cost[:, d])
        acc["demand"] += dem
        acc["sold"] += sold
        acc["revenue"] += sold * price
        acc["cogs"] += sold * cost
        acc["stockout_days"] += (need > 0) & listed
        # evening: expiry, ageing
        acc["waste"] += inv[:, 1]
        acc["waste_cost"] += inv[:, 1] * cost
        inv[:, 1:-1] = inv[:, 2:]
        inv[:, -1] = 0
        on_hand = inv.sum(1)
        acc["capital"] += on_hand * cost
        if d + 1 >= start + days:
            break
        order = eco.delivery[:, weekday(d + 1)] & ~np.isnan(eco.price[:, d + 1])
        protect = np.ones(n, int)
        for k in range(2, 9):
            later = eco.delivery[:, weekday(d + k)]
            protect = np.where((protect == k - 1) & ~later, k, protect)
        protect = np.minimum(protect, start + days - (d + 1))
        ctx = Context(d, order, protect, inv, on_hand, eco, y)
        q = np.where(order, policy(ctx), 0.0)
        q = np.maximum(q, 0)
        pending = q
        acc["ordered"] += q
        acc["orders"] += q > 0
    return acc


def summarise(acc: dict, eco: Economics, groups: dict[str, np.ndarray] | None = None, days: int = 28) -> dict:
    def block(sel):
        dem = acc["demand"][sel].sum()
        margin = acc["revenue"][sel].sum() - acc["cogs"][sel].sum() - acc["waste_cost"][sel].sum()
        return {
            "fill_rate": round(float(acc["sold"][sel].sum() / max(dem, 1)), 4),
            "lost_units": int(round(dem - acc["sold"][sel].sum())),
            "waste_units": int(round(acc["waste"][sel].sum())),
            "waste_cost": round(float(acc["waste_cost"][sel].sum()), 0),
            "waste_share_of_cogs": round(float(acc["waste_cost"][sel].sum() / max(acc["cogs"][sel].sum(), 1)), 4),
            "avg_inventory_capital": round(float(acc["capital"][sel].sum() / days), 0),
            "margin_after_waste": round(float(margin), 0),
            "stockout_item_days_share": round(float(acc["stockout_days"][sel].sum() / (sel.sum() * days)), 4),
            "inventory_turn_days": round(float(acc["capital"][sel].sum() / days / max(acc["cogs"][sel].sum() / days, 1e-9)), 2),
        }

    out = {"all": block(np.ones(len(eco.life), bool))}
    for k, sel in (groups or {}).items():
        out[k] = block(sel)
    return out
