"""End-to-end run: data → stock-out detection → backtest of every model → release gate →
replenishment simulation on the champion → censored-demand experiment → monitoring → registry →
reports. `dre run` calls this; CI runs it on one store with lighter settings."""

from __future__ import annotations

import gc
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import registry
from .data import m5, stockouts
from .evaluation import backtest, censoring, gate
from .explain import llm
from .features import FeatureBuilder
from .models.base import Forecast
from .models.baselines import TSB, MovingAverage, SeasonalNaive
from .models.gbm import LightGBMForecaster
from .monitoring import report as monitoring
from .replenishment import advisor, economics, simulate


def load_config(path: Path, fast: bool = False, stores: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if stores:
        cfg["data"]["stores"] = stores
    if fast:  # CI / laptop smoke run: same code paths, lighter models
        for spec in cfg["forecast"]["models"].values():
            if spec["kind"] in ("lightgbm", "torch"):
                spec.update({"n_origins": 6, "fraction": 1.0})
                spec.setdefault("params", {})["n_estimators"] = 80
                spec["epochs"] = 1
        cfg["forecast"]["folds"] = cfg["forecast"]["folds"][-2:]
        cfg["forecast"]["warmup_origin"] = cfg["forecast"]["folds"][0] - cfg["forecast"]["horizon"]
        cfg["censoring_experiment"].update({"n_origins": 4, "params": {"n_estimators": 60}})
        cfg["censoring_experiment"]["store"] = (stores or [cfg["censoring_experiment"]["store"]])[0]
    return cfg


def make(spec: dict, name: str, horizon: int):
    k = spec["kind"]
    if k == "seasonal_naive":
        return SeasonalNaive()
    if k == "moving_average":
        return MovingAverage()
    if k == "tsb":
        return TSB()
    args = {a: spec[a] for a in ("n_origins", "origin_step", "fraction", "censored") if a in spec}
    if k == "lightgbm":
        return LightGBMForecaster(horizon=horizon, params=spec.get("params"), name=name, **args)
    if k == "torch":
        from .models.nn import TorchNB

        return TorchNB(horizon=horizon, epochs=spec.get("epochs", 2), name=name, **args)
    raise ValueError(k)


def feature_builder(panel: m5.Panel, origin: int, so_cfg: dict) -> FeatureBuilder:
    """Stock-out flags from data up to `origin` only."""
    flags = stockouts.detect(panel.sales[:, : origin + 1], so_cfg["window"], so_cfg["alpha"], so_cfg["min_run"], so_cfg.get("max_run"))
    return FeatureBuilder(panel, np.pad(flags, ((0, 0), (0, panel.t - flags.shape[1]))))


def run(config: Path, out: Path, fast: bool = False, stores: list[str] | None = None, skip_torch: bool = False, log=print) -> dict:
    t_start = time.time()
    cfg = load_config(config, fast, stores)
    fc_cfg, H = cfg["forecast"], cfg["forecast"]["horizon"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(exist_ok=True)
    root = Path(cfg["data"]["root"])
    m5.download(root)
    panel = m5.load(root, cfg["data"]["category"], cfg["data"]["stores"], cache=Path(cfg["data"]["cache"]))
    log(f"data: {panel.n:,} series × {panel.t:,} days ({panel.ids.store_id.nunique()} stores, {panel.ids.item_id.nunique():,} items)")
    so_cfg = fc_cfg["stockouts"]
    fb_for = lambda o: feature_builder(panel, o, so_cfg)  # noqa: E731

    # ------------------------------------------------------------ backtest
    results: dict[str, backtest.ModelResult] = {}
    for name, spec in fc_cfg["models"].items():
        if skip_torch and spec["kind"] == "torch":
            continue
        log(f"backtest {name}")
        results[name] = backtest.run(fb_for, lambda spec=spec, name=name: make(spec, name, H), fc_cfg["folds"], fc_cfg["warmup_origin"], H, log)
        gc.collect()

    # ------------------------------------------------------------ release gate
    champ = fc_cfg["champion"]
    decisions = []
    for name in sorted((n for n in results if n != champ), key=lambda n: results[n].mean("wrmsse")):
        d = gate.decide(results[champ], results[name], cfg["gate"])
        decisions.append(d)
        log(f"gate {name} vs {champ}: promote={d['promote']} {d['checks']}")
        if d["promote"]:
            champ = name
    log(f"champion: {champ}")
    final = results[champ]

    # ------------------------------------------------------------ replenishment simulation
    rp = cfg["replenishment"] | {"economics": cfg["economics"]}
    start = rp["start_origin"]
    eco = economics.build(panel, start, cfg)
    alpha_cal = {n: r.last_forecast.alpha[:, :1] for n, r in results.items()}
    cache: dict[tuple[str, int], Forecast] = {}

    def forecaster(name: str):
        model = results[name].last_model

        def at(t: int) -> Forecast:
            if (name, t) not in cache:
                fb = fb_for(t)
                h = min(H, panel.t - 1 - t)
                f = model.predict(fb, t, h)
                cache[(name, t)] = Forecast(f.mean, np.repeat(alpha_cal[name], h, 1), t, name)
                del fb
            return cache[(name, t)]

        return at

    days = rp["days"]
    sim_start = start + 1
    weekday0 = int(panel.calendar.wday.iloc[sim_start])
    sales = panel.sales
    cogs = (
        pd.Series(np.nansum(sales[:, start - 27 : start + 1] * np.nan_to_num(eco.cost[:, start - 27 : start + 1]), 1) / 28)
        .groupby(panel.ids.store_id.to_numpy())
        .sum()
        .to_dict()
    )
    zone_of = {d: z for z, ds in cfg["economics"]["zone_of_department"].items() for d in ds}
    zones = panel.ids.dept_id.map(zone_of).to_numpy()
    groups = {z: zones == z for z in cfg["economics"]["zone_of_department"]}
    base = "moving_average" if "moving_average" in results else champ
    policies = [
        simulate.CoverDays(rp["cover_days_safety"]),
        simulate.Newsvendor(forecaster(base), rp, False, f"newsvendor[{base}]"),
        *([simulate.Newsvendor(forecaster(champ), rp, False, f"newsvendor[{champ}]")] if champ != base else []),
        *([simulate.Newsvendor(forecaster(base), rp, True, f"perishable_newsvendor[{base}]")] if champ != base else []),
        simulate.Newsvendor(forecaster(champ), rp, True, f"perishable_newsvendor[{champ}]"),
        *[
            simulate.Newsvendor(forecaster(champ), rp | {"stockout_penalty_share": pen}, True, f"perishable_newsvendor_penalty_{pen:g}[{champ}]")
            for pen in rp.get("penalty_sensitivity", [])
        ],
        *[
            simulate.CapitalConstrained(forecaster(champ), rp, c, cogs, f"capital_cap_{c:g}d_{how}[{champ}]", how)
            for c in rp["capital_caps_days"]
            for how in ("proportional", "milp")
        ],
        simulate.Oracle(),
    ]
    policy_results = {}
    for pol in policies:
        t0 = time.time()
        acc = simulate.simulate(pol, eco, sales, sim_start, days, weekday0, rp["init_days_of_stock"])
        policy_results[pol.name] = simulate.summarise(acc, eco, groups, days)
        extra = f", {pol.solves} MILP solves in {pol.solve_seconds:.1f}s" if isinstance(pol, simulate.CapitalConstrained) else ""
        a = policy_results[pol.name]["all"]
        log(
            f"policy {pol.name}: fill {a['fill_rate']:.3f} waste {a['waste_share_of_cogs']:.3f} capital {a['avg_inventory_capital']:,.0f} margin {a['margin_after_waste']:,.0f} ({time.time() - t0:.0f}s{extra})"
        )
        if isinstance(pol, simulate.CapitalConstrained):
            policy_results[pol.name]["milp"] = {"solves": pol.solves, "seconds": round(pol.solve_seconds, 1)}

    # ------------------------------------------------------------ order explanations (examples)
    f0 = forecaster(champ)(start)
    store0 = panel.ids.store_id.iloc[0]
    rows0 = np.flatnonzero(panel.ids.store_id.to_numpy() == store0)
    inv0 = np.zeros((len(rows0), int(eco.life.max()) + 1))
    rate0 = np.nan_to_num(sales[rows0, start - 27 : start + 1]).mean(1)
    inv0[np.arange(len(rows0)), np.minimum(eco.life[rows0], 2)] = np.round(rate0 * 1.5)
    advice = advisor.advise(f0, eco, start, rows0, inv0, int(panel.calendar.wday.iloc[start + 1]), panel.ids, rp, panel.calendar)
    advice = sorted(advice, key=lambda a: -a["order_units"])
    examples = [{"drivers": a, "explanation": llm.explain(a).__dict__} for a in advice[:3] + [a for a in advice if a["expiring_units"] > 0][:2]]

    # ------------------------------------------------------------ censored-demand experiment
    log("censored-demand experiment")
    cens = censoring.run(panel, cfg, H, start, log)

    # ------------------------------------------------------------ monitoring of the champion's last fold
    fb = fb_for(start)
    last_model = final.last_model
    mon_acc, alerts = monitoring.accuracy(final.last_forecast, panel, final.mean("wape"), {"bias_alert": 0.05, "wape_degradation_alert": 0.2})
    mon = {
        "data_checks": monitoring.data_checks(fb, start),
        "drift": monitoring.drift(fb, getattr(last_model, "training_origins", lambda o: [o - H - 14 * k for k in range(8)])(start), start, H),
        "alerts": alerts,
    }
    importance = last_model.importance() if hasattr(last_model, "importance") else {}
    del fb

    # ------------------------------------------------------------ registry and reports
    backtest_rows = [{"model": n, **f.summary()} for n, r in results.items() for f in r.folds]
    bt = pd.DataFrame(backtest_rows)
    summary = {
        n: {k: r.mean(k) for k in ("wrmsse", "wspl", "coverage_50", "coverage_95", "bias", "wape", "wape_day1_7", "fit_seconds")} | {"train": r.info}
        for n, r in results.items()
    }
    meta = {"backtest": summary[champ], "gate": [d for d in decisions if d["challenger"] == champ], "config_sha": registry.sha(Path(config).read_bytes())}
    reg = registry.save(out / "registry", last_model, final.last_forecast, panel, eco, meta)
    report = {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "data": {
            "series": panel.n,
            "stores": int(panel.ids.store_id.nunique()),
            "items": int(panel.ids.item_id.nunique()),
            "days": panel.t,
            "units_sold": int(np.nansum(panel.sales)),
            "first_day": str(panel.calendar.date.iloc[0].date()),
            "last_day": str(panel.calendar.date.iloc[-1].date()),
        },
        "config": {"folds": fc_cfg["folds"], "warmup": fc_cfg["warmup_origin"], "horizon": H, "fast": fast},
        "backtest": summary,
        "levels": {n: {lv: round(float(np.mean([f.levels[lv] for f in r.folds])), 4) for lv in r.folds[0].levels} for n, r in results.items()},
        "gate": decisions,
        "champion": champ,
        "importance": dict(list(importance.items())[:12]),
        "replenishment": {
            "start_date": str(panel.calendar.date.iloc[sim_start].date()),
            "days": days,
            "policies": policy_results,
            "assumptions": cfg["economics"]
            | {k: rp[k] for k in ("stockout_penalty_share", "min_service", "max_service", "cover_days_safety", "capital_caps_days")},
        },
        "censoring_experiment": cens,
        "monitoring": mon,
        "explanations": examples,
        "registry": str(reg),
        "runtime_minutes": round((time.time() - t_start) / 60, 1),
    }
    bt.to_csv(out / "reports" / "backtest.csv", index=False)
    mon_acc.to_csv(out / "reports" / "monitoring_accuracy.csv", index=False)
    pd.DataFrame([{"policy": p, "zone": z, **v} for p, r in policy_results.items() for z, v in r.items() if z != "milp"]).to_csv(
        out / "reports" / "replenishment.csv", index=False
    )
    (out / "reports" / "report.json").write_text(json.dumps(report, indent=2, default=_json))
    from .reporting import dashboard, model_card

    model_card.write(report, cfg, out / "reports" / "model_card.md")
    dashboard.write(report, bt, out / "reports" / "dashboard.html")
    log(f"done in {report['runtime_minutes']} min → {out / 'reports'}")
    return report


def _json(o):
    if isinstance(o, np.integer | np.floating):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)
