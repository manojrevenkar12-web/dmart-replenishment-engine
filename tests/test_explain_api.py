import numpy as np
import pytest

from dre.explain import llm

DRIVERS = {
    "item_id": "FOODS_3_090",
    "store_id": "CA_1",
    "order_units": 24.0,
    "order_packs": 2.0,
    "pack": 12.0,
    "protection_days": 3.0,
    "forecast_units": 31.4,
    "target_units": 38.0,
    "service_level": 0.95,
    "on_hand_units": 14.0,
    "expiring_units": 0.0,
    "price_change_pct": 0.0,
    "event": "",
    "capped_by_shelf": False,
}


class FakeClient:
    def __init__(self, text=None, fail=False):
        self.text, self.fail = text, fail
        self.messages = self

    def create(self, **kw):
        if self.fail:
            raise TimeoutError("down")

        class B:
            text = self.text

        class M:
            content = [B()]

        return M()


def test_template_contains_the_decision():
    t = llm.template(DRIVERS)
    assert "24 units" in t and "FOODS_3_090" in t and "95%" in t


def test_llm_answer_with_driver_numbers_is_used():
    e = llm.explain(
        DRIVERS, client=FakeClient("We order 24 units, 2 cases of 12, because about 31.4 units should sell in the next 3 days and we aim for 95% availability.")
    )
    assert e.source == "llm"


def test_llm_answer_with_invented_number_is_rejected():
    e = llm.explain(DRIVERS, client=FakeClient("We order 24 units because sales rose 40% last week."))
    assert e.source == "template" and "40" in e.rejected_reason


def test_llm_failure_falls_back_to_template():
    e = llm.explain(DRIVERS, client=FakeClient(fail=True))
    assert e.source == "template" and e.rejected_reason.startswith("llm error")


@pytest.fixture(scope="module")
def client(tmp_path_factory, panel, cfg):
    from fastapi.testclient import TestClient

    from dre import registry
    from dre.models.baselines import MovingAverage
    from dre.pipeline import feature_builder
    from dre.replenishment import economics

    o = 480
    fb = feature_builder(panel, o, cfg["forecast"]["stockouts"])
    m = MovingAverage()
    fc = m.predict(fb, o, 28)
    eco = economics.build(panel, o, cfg)
    root = tmp_path_factory.mktemp("registry")
    registry.save(root, m, fc, panel, eco, {"backtest": {"wrmsse": 0.9}})
    import os

    os.environ["DRE_REGISTRY"] = str(root)
    from dre.service import api

    api.store.cache_clear()
    return TestClient(api.app)


def test_api_health_and_forecast(client, panel):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model"] == "moving_average"
    item = panel.ids.item_id.iloc[0]
    f = client.get(f"/v1/forecast/CA_1/{item}").json()
    assert len(f["days"]) == 28 and all(d["p10"] <= d["p50"] <= d["p90"] for d in f["days"])
    assert client.get("/v1/forecast/CA_1/NOPE").status_code == 404


def test_api_orders(client, panel):
    h = client.get("/health").json()
    first = h["forecast_days"][0]
    import pandas as pd

    day_before = (pd.Timestamp(first) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    items = panel.ids[panel.ids.store_id == "CA_1"].item_id.tolist()
    body = {"store_id": "CA_1", "date": day_before, "stock": [{"item_id": items[0], "units": 3, "remaining_life_days": 2}]}
    r = client.post("/v1/orders", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["items_considered"] > 0
    for o in out["orders"]:
        assert o["order_units"] % o["pack"] == 0 and o["explanation"]
    assert client.post("/v1/orders", json=body | {"date": "1999-01-01"}).status_code == 422
    assert client.post("/v1/orders", json=body | {"stock": [{"item_id": "NOPE", "units": 1, "remaining_life_days": 2}]}).status_code == 422
    assert np.isfinite([o["forecast_units"] for o in out["orders"]]).all()
