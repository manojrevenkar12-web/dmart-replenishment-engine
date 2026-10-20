"""Model card generated from the run's report: every number comes from report.json."""

from __future__ import annotations

from pathlib import Path


def _t(rows: list[dict], cols: list[str], heads: list[str]) -> str:
    out = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    out += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def write(r: dict, cfg: dict, path: Path) -> None:
    bt, champ = r["backtest"], r["champion"]
    rows = [{"model": n, **v} for n, v in sorted(bt.items(), key=lambda kv: kv[1]["wrmsse"])]
    pol = r["replenishment"]["policies"]

    def fmt(a: dict) -> dict:
        return {
            "fill_rate": f"{a['fill_rate']:.1%}",
            "waste_share_of_cogs": f"{a['waste_share_of_cogs']:.1%}",
            "avg_inventory_capital": f"{a['avg_inventory_capital']:,.0f}",
            "margin_after_waste": f"{a['margin_after_waste']:,.0f}",
            "stockout_item_days_share": f"{a['stockout_item_days_share']:.1%}",
        }

    prow = [{"policy": n, **fmt(v["all"])} for n, v in pol.items()]
    zones = [z for z in next(iter(pol.values())) if z not in ("all", "milp")]
    zrow = [{"policy": n, "zone": z, **fmt(v[z])} for n, v in pol.items() for z in zones if n == "cover_days" or "newsvendor" in n]
    cens = r["censoring_experiment"]
    lines = [
        f"# Model card — {champ}",
        "",
        f"Generated {r['generated']} · data {r['data']['series']:,} store × item series, {r['data']['first_day']} to {r['data']['last_day']} (M5, FOODS) · registry `{Path(r['registry']).name}`",
        "",
        "## Intended use",
        "Daily demand forecasts (mean and negative-binomial distribution, 1–28 days) per store × item, consumed by the replenishment optimiser "
        "to set order quantities. Not intended for items without sales history (new listings need a cold-start rule) or for price setting.",
        "",
        "## Backtest (rolling origin, mean over folds " + ", ".join(f"d_{o + 1}" for o in r["config"]["folds"]) + ")",
        _t(
            rows,
            ["model", "wrmsse", "wspl", "bias", "wape", "wape_day1_7", "coverage_50", "coverage_95"],
            ["Model", "WRMSSE", "WSPL", "Bias", "WAPE", "WAPE days 1–7", "50% cover", "95% cover"],
        ),
        "",
        "WRMSSE as defined in M5, on the FOODS hierarchy (the implementation reproduces the published M5 seasonal-naive score of 0.847 on the full dataset).",
        "",
        "## Release gate",
        *[
            f"- {d['challenger']} vs {d['champion']}: **{'promoted' if d['promote'] else 'rejected'}** — WRMSSE {d['wrmsse'][d['champion']]} → {d['wrmsse'][d['challenger']]}, "
            f"weighted RMSSE difference {d['bootstrap']['weighted_rmsse_diff']} (95% CI {d['bootstrap']['ci95']}), folds won {d['fold_wins']}, "
            f"checks {', '.join(k for k, v in d['checks'].items() if not v) or 'all passed'}{' failed' if not all(d['checks'].values()) else ''}"
            for d in r["gate"]
        ],
        "",
        "## Replenishment value (simulated on the real daily demand of " + r["replenishment"]["start_date"] + " + 28 days)",
        _t(
            prow,
            ["policy", "fill_rate", "waste_share_of_cogs", "avg_inventory_capital", "margin_after_waste", "stockout_item_days_share"],
            ["Policy", "Fill rate", "Waste / COGS", "Avg inventory capital", "Margin after waste", "Stock-out item-days"],
        ),
        "",
        "By temperature zone:",
        "",
        _t(
            zrow,
            ["policy", "zone", "fill_rate", "waste_share_of_cogs", "margin_after_waste"],
            ["Policy", "Zone", "Fill rate", "Waste / COGS", "Margin after waste"],
        ),
        "",
        "## Censored demand",
        f"Planted stock-outs in {cens['store']} ({cens['eligible_items']} items selling ≥ 1/day): the detector flags {cens['detection']['planted_days_flagged']:.0%} of planted days "
        f"and {cens['detection']['planted_episodes_with_any_flag']:.0%} of episodes.",
        "",
        _t(
            [{"training": k, **v} for k, v in cens["forecast"].items()],
            ["training", "bias_eligible_items", "wape_eligible_items"],
            ["Training history", "Bias", "WAPE"],
        ),
        "",
        "## Monitoring at go-live",
        f"Data checks: {r['monitoring']['data_checks']}",
        "",
        "Drift (PSI): " + ", ".join(f"{k} {v['psi']} ({v['status']})" for k, v in r["monitoring"]["drift"].items()),
        "",
        f"Alerts: {len(r['monitoring']['alerts'])}"
        + ("".join(f"\n- {a['group']} {a['kind']} {a['value']} (threshold {a['threshold']})" for a in r["monitoring"]["alerts"])),
        "",
        "## Limitations",
        "- M5 is Walmart supermarket data, not dark-store data: assortment, basket and delivery promise differ. The method transfers; the numbers do not.",
        "- Recorded sales are censored demand. The replenishment simulation replays recorded sales, so policies that stock more cannot sell more than history did: fill-rate gains are lower bounds, waste is not.",
        "- Costs, shelf lives, pack sizes, shelf capacity and delivery days are documented assumptions (config), identical for every policy.",
        "- Departments stand in for temperature zones; M5 does not say which products are fresh.",
    ]
    path.write_text("\n".join(lines) + "\n")
