import numpy as np
import pytest

from dre.models.base import Forecast
from dre.replenishment import economics, simulate


def _eco(n=1, life=3, pack=1, capacity=1000, price=10.0, days=40, delivery=None):
    price_m = np.full((n, days), price)
    return economics.Economics(
        price=price_m,
        cost=price_m * 0.7,
        life=np.full(n, life),
        pack=np.full(n, pack),
        capacity=np.full(n, capacity),
        delivery=np.ones((n, 7), bool) if delivery is None else delivery,
        holding_rate=0.001,
        store=np.array(["S"] * n),
    )


class Fixed:
    name = "fixed"

    def __init__(self, q):
        self.q = q

    def __call__(self, ctx):
        return np.full(len(ctx.on_hand), self.q, float)


def test_fefo_and_expiry_by_hand():
    """Life 3 days, 10 units arrive every morning, demand 4 a day: each day's 6 leftover units are
    sold first the next day (FEFO); in steady state 10 arrive, 4 sell from old stock... waste
    appears exactly when units reach the end of their life."""
    eco = _eco(life=3)
    sales = np.full((1, 40), 4.0)
    acc = simulate.simulate(Fixed(10), eco, sales, start=30, days=6, start_weekday=0, init_days=0)
    assert acc["sold"].sum() == pytest.approx(4 * 5)  # day 1 has no stock (init 0, first order arrives day 2)
    assert acc["demand"].sum() == 24
    # in by day 7: 50 units; sold 20; on hand + waste = 30
    assert acc["waste"].sum() > 0


def test_no_waste_when_stock_matches_demand():
    eco = _eco(life=2)
    sales = np.full((1, 40), 5.0)
    acc = simulate.simulate(Fixed(5), eco, sales, start=30, days=8, start_weekday=0, init_days=1)
    assert acc["waste"].sum() == 0
    assert acc["sold"].sum() == 40


def test_expected_waste_of_existing_stock():
    inv = np.zeros((1, 5))
    inv[0, 1] = 10  # 10 units that can only be sold tomorrow
    inv[0, 4] = 10
    w = simulate.expected_waste_existing(inv, np.array([[3.0, 3.0, 3.0]]), np.array([3]))
    assert w[0] == pytest.approx(7.0)


def test_protection_demand_sums_days():
    fc = Forecast(np.array([[1.0, 2.0, 3.0, 4.0]]), np.zeros((1, 4)), 10, "x")
    m, a = simulate.protection_demand(fc, 10, np.array([3]))
    assert m[0] == pytest.approx(6.0) and a[0] == pytest.approx(0, abs=1e-6)


def test_perishable_policy_does_not_count_stock_that_will_expire():
    """12 units that can only be sold tomorrow, demand 8 a day: 4 will expire. The textbook policy
    counts all 12 as stock; the perishable-aware policy counts 8."""
    eco = _eco(n=1, life=3)
    fc = Forecast(np.full((1, 10), 8.0), np.full((1, 10), 0.1), 20, "x")
    inv = np.zeros((1, 4))
    inv[0, 1] = 12
    ctx = simulate.Context(20, np.array([True]), np.array([2]), inv, inv.sum(1), eco, np.zeros((1, 40)))
    cfg = {"stockout_penalty_share": 0.2, "min_service": 0.5, "max_service": 0.98}
    assert simulate.Newsvendor(lambda t: fc, cfg, False, "t").usable(ctx, fc)[0] == 12
    assert simulate.Newsvendor(lambda t: fc, cfg, True, "p").usable(ctx, fc)[0] == pytest.approx(8)


@pytest.mark.parametrize("life, expect", [(1, "lower"), (60, "same")])
def test_marginal_analysis_limits(life, expect):
    """A product that must sell tomorrow or be thrown away gets less stock than the textbook
    newsvendor; a long-life product gets the same (capped) service level."""
    eco = _eco(n=1, life=life)
    fc = Forecast(np.full((1, 10), 20.0), np.full((1, 10), 0.2), 20, "x")
    inv = np.zeros((1, life + 1))
    ctx = simulate.Context(20, np.array([True]), np.array([1]), inv, inv.sum(1), eco, np.zeros((1, 40)))
    cfg = {"stockout_penalty_share": 0.2, "min_service": 0.5, "max_service": 0.98}
    textbook = simulate.Newsvendor(lambda t: fc, cfg, False, "t").target(ctx, fc, np.zeros(1))[0]
    aware = simulate.Newsvendor(lambda t: fc, cfg, True, "p").target(ctx, fc, np.zeros(1))[0]
    if expect == "lower":
        assert aware < textbook
    else:
        assert aware == textbook


def test_capital_constrained_respects_budget_and_prefers_margin():
    n = 4
    eco = _eco(n=n, life=60)
    eco.price[2:] *= 3  # items 2, 3 earn three times the margin per unit
    eco.cost[2:] *= 3
    fc = Forecast(np.full((n, 10), 10.0), np.full((n, 10), 0.05), 20, "x")
    inv = np.zeros((n, 61))
    ctx = simulate.Context(20, np.ones(n, bool), np.full(n, 2), inv, inv.sum(1), eco, np.zeros((n, 40)))
    cfg = {"stockout_penalty_share": 0.0, "min_service": 0.5, "max_service": 0.98}
    pol = simulate.CapitalConstrained(lambda t: fc, cfg, cap_days=1.0, daily_cogs={"S": 150.0})
    q = pol(ctx)
    c = eco.cost[:, 21]
    assert (c * q).sum() <= 150.0 + 1e-6
    assert pol.solves == 1


def test_economics_from_config(panel, cfg):
    eco = economics.build(panel, 400, cfg)
    assert set(np.unique(eco.life)) <= {4, 10, 60}
    assert (eco.capacity >= 2 * eco.pack).all()
    assert eco.delivery.any(1).all()


def test_milp_allocates_capital_better_than_proportional_cuts():
    """Same budget: the knapsack puts capital where the marginal profit is highest, so its
    expected value is at least that of cutting every order by the same share."""
    n = 6
    eco = _eco(n=n, life=60, pack=1)
    eco.price[:3] *= 4  # half the items earn four times the margin
    eco.cost[:3] *= 4
    fc = Forecast(np.full((n, 10), 10.0), np.full((n, 10), 0.05), 20, "x")
    inv = np.zeros((n, 61))
    ctx = simulate.Context(20, np.ones(n, bool), np.full(n, 2), inv, inv.sum(1), eco, np.zeros((n, 40)))
    cfg = {"stockout_penalty_share": 0.0, "min_service": 0.5, "max_service": 0.98}
    budget = {"S": 200.0}
    q_milp = simulate.CapitalConstrained(lambda t: fc, cfg, 1.0, budget, method="milp")(ctx)
    q_prop = simulate.CapitalConstrained(lambda t: fc, cfg, 1.0, budget, method="proportional")(ctx)
    mv, _ = simulate.marginal_values(ctx, fc, np.zeros(n), cfg, np.arange(n))
    cum = np.concatenate([np.zeros((n, 1)), np.cumsum(mv, 1)], 1)
    value = lambda q: sum(cum[i, int(q[i])] for i in range(n))  # noqa: E731
    c = eco.cost[:, 21]
    assert (c * q_milp).sum() <= 200 + 1e-6 and (c * q_prop).sum() <= 200 + 1e-6
    assert value(q_milp) >= value(q_prop)
