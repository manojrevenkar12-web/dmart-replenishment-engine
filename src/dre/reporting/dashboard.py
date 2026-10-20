"""Self-contained HTML report of a run (charts as inline SVG)."""

from __future__ import annotations

import html
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

INK, MUTED, GRID, BG = "#1f2328", "#667085", "#e6e8ec", "#fbfaf7"
ACCENT, ACCENT2, GOOD, BAD = "#d6452f", "#2f6f73", "#2e8b57", "#b54708"
SERIES = ["#2f6f73", "#d6452f", "#7b61a8", "#c08a1e", "#4a7fb5", "#8a8f98", "#3f8f5a"]


def _svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    s = buf.getvalue()
    return s[s.index("<svg") :]


def _ax(title: str, w=6.2, h=2.8):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def short(name: str, champ: str) -> str:
    """Policy label: the policy, plus the forecast it uses when that is not the champion."""
    if "[" not in name:
        return name
    pol, fc = name[:-1].split("[")
    return pol if fc == champ else f"{pol} · {fc}"


def write(r: dict, bt: pd.DataFrame, path: Path) -> None:
    champ = r["champion"]
    order = sorted(r["backtest"], key=lambda n: r["backtest"][n]["wrmsse"])
    charts = []
    fig, ax = _ax("WRMSSE by model (lower is better), mean of folds")
    vals = [r["backtest"][n]["wrmsse"] for n in order]
    ax.barh(order[::-1], vals[::-1], color=[ACCENT if n == champ else "#b8c4c6" for n in order[::-1]], height=0.6)
    for y, v in enumerate(vals[::-1]):
        ax.text(v, y, f" {v:.3f}", va="center", fontsize=8, color=INK)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    charts.append(_svg(fig))

    base = "moving_average" if "moving_average" in r["levels"] else order[-1]
    fig, ax = _ax(f"Error by aggregation level: {champ} vs {base}")
    lv = list(r["levels"][champ])
    x = range(len(lv))
    ax.plot(x, [r["levels"][base][k] for k in lv], marker="o", ms=3, color="#8a8f98", label=base)
    ax.plot(x, [r["levels"][champ][k] for k in lv], marker="o", ms=3, color=ACCENT, label=champ)
    ax.set_xticks(list(x), [k.replace("_", "×") for k in lv], rotation=40, ha="right", fontsize=7)
    ax.legend(frameon=False, fontsize=7)
    charts.append(_svg(fig))

    pol = r["replenishment"]["policies"]
    fig, ax = _ax("Replenishment policies: capital tied up vs margin after waste")
    for i, (n, v) in enumerate(pol.items()):
        a = v["all"]
        ax.scatter(a["avg_inventory_capital"] / 1000, a["margin_after_waste"] / 1000, s=40, color=SERIES[i % len(SERIES)], zorder=3)
        if "penalty" in n or (n.startswith("perishable") and not n.endswith(f"[{champ}]")):
            continue  # close to the champion's perishable point; listed in the table below
        ax.annotate(
            short(n, champ),
            (a["avg_inventory_capital"] / 1000, a["margin_after_waste"] / 1000),
            fontsize=7,
            color=INK,
            xytext=(4, 3 if i % 2 else -9),
            textcoords="offset points",
        )
    ax.set_xlabel("average inventory capital (k$)", fontsize=8, color=MUTED)
    ax.set_ylabel("margin after waste (k$)", fontsize=8, color=MUTED)
    charts.append(_svg(fig))

    fig, ax = _ax("Fill rate and waste by policy")
    names = list(pol)
    fr = [pol[n]["all"]["fill_rate"] * 100 for n in names]
    ws = [pol[n]["all"]["waste_share_of_cogs"] * 100 for n in names]
    y = range(len(names))
    ax.barh([i + 0.2 for i in y], fr, height=0.38, color=ACCENT2, label="fill rate %")
    ax.barh([i - 0.2 for i in y], ws, height=0.38, color=BAD, label="waste % of COGS")
    ax.set_yticks(list(y), [short(n, champ) for n in names], fontsize=7)
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    charts.append(_svg(fig))

    c = r["censoring_experiment"]
    fig, ax = _ax("Planted stock-outs: forecast bias on the true test month", h=2.2)
    ks = list(c["forecast"])
    bv = [c["forecast"][k]["bias_eligible_items"] * 100 for k in ks]
    ax.barh(ks[::-1], bv[::-1], color=["#8a8f98", BAD, ACCENT2][: len(ks)][::-1], height=0.55)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("bias %", fontsize=8, color=MUTED)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    charts.append(_svg(fig))

    cd = pol.get("cover_days", {}).get("all")
    best = max(pol, key=lambda n: pol[n]["all"]["margin_after_waste"] if n != "oracle" else -1e18)
    b = pol[best]["all"]
    tiles = [
        (
            f"{r['backtest'][champ]['wrmsse']:.3f}",
            f"WRMSSE of {champ} (moving average {r['backtest'].get('moving_average', {}).get('wrmsse', float('nan')):.3f})",
        ),
        (f"{r['backtest'][champ]['wspl']:.3f}", "scaled pinball loss of the forecast distribution"),
    ]
    if cd:
        tiles += [
            (f"{(b['waste_cost'] / max(cd['waste_cost'], 1) - 1):+.0%}", f"waste vs the days-of-cover rule ({short(best, champ)})"),
            (f"{(b['margin_after_waste'] / cd['margin_after_waste'] - 1):+.1%}", "margin after waste vs the days-of-cover rule"),
        ]
    tile_html = "".join(f'<div class="tile"><b>{a}</b><span>{html.escape(t)}</span></div>' for a, t in tiles)
    gate_rows = "".join(
        f"<tr><td>{html.escape(d['challenger'])}</td><td>{d['wrmsse'][d['challenger']]}</td><td>{d['bootstrap']['weighted_rmsse_diff']} [{d['bootstrap']['ci95'][0]}, {d['bootstrap']['ci95'][1]}]</td>"
        f"<td>{d['fold_wins']}</td><td>{'promoted' if d['promote'] else 'rejected: ' + ', '.join(k for k, v in d['checks'].items() if not v)}</td></tr>"
        for d in r["gate"]
    )
    zones = [z for z in next(iter(pol.values())) if z not in ("all", "milp")]
    key_pols = [n for n in pol if n in ("cover_days", "oracle") or n.startswith(("newsvendor[", "perishable_newsvendor["))]
    zone_rows = "".join(
        f"<tr><td>{html.escape(short(n, champ))}</td>"
        + "".join(
            f"<td>{pol[n][z]['fill_rate']:.1%} · {pol[n][z]['waste_share_of_cogs']:.1%} · {pol[n][z]['margin_after_waste'] / 1000:,.1f}k</td>" for z in zones
        )
        + "</tr>"
        for n in key_pols
    )
    pol_rows = "".join(
        f"<tr><td>{html.escape(short(n, champ))}</td><td>{v['all']['fill_rate']:.1%}</td><td>{v['all']['waste_share_of_cogs']:.1%}</td><td>{v['all']['avg_inventory_capital']:,.0f}</td>"
        f"<td>{v['all']['margin_after_waste']:,.0f}</td><td>{v['all']['inventory_turn_days']}</td></tr>"
        for n, v in pol.items()
    )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dmart replenishment run</title>
<style>
body{{margin:0;font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:{INK};background:{BG}}}
main{{max-width:1120px;margin:0 auto;padding:24px 16px 48px}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:15px;margin:28px 0 10px}}
p.sub{{color:{MUTED};margin:0 0 18px;font-size:13px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}}
.tile{{background:#fff;border:1px solid {GRID};border-radius:6px;padding:14px;border-top:3px solid {ACCENT}}} .tile b{{display:block;font-size:24px}} .tile span{{color:{MUTED};font-size:12px}}
.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(470px,1fr));gap:12px}}
.card{{background:#fff;border:1px solid {GRID};border-radius:6px;padding:12px;overflow-x:auto}} .card svg{{max-width:100%;height:auto}}
table{{border-collapse:collapse;width:100%;font-size:12px;background:#fff}} th,td{{border-bottom:1px solid {GRID};padding:6px 8px;text-align:left;vertical-align:top}}
th{{color:{MUTED};font-weight:600}} .wrap{{overflow-x:auto}}
</style></head><body><main>
<h1>Dmart replenishment engine — run report</h1>
<p class="sub">M5 FOODS · {r["data"]["series"]:,} store × item series · champion {html.escape(champ)} · replenishment simulated on {r["replenishment"]["start_date"]} + {r["replenishment"]["days"]} days · generated {r["generated"]}</p>
<div class="tiles">{tile_html}</div>
<h2>Forecasting and replenishment</h2><div class="charts">{"".join(f'<div class="card">{c}</div>' for c in charts)}</div>
<h2>Release gate</h2><div class="wrap"><table><tr><th>Challenger</th><th>WRMSSE</th><th>Weighted RMSSE diff [95% CI]</th><th>Folds won</th><th>Decision</th></tr>{gate_rows}</table></div>
<h2>By temperature zone <span style="font-weight:400;color:{MUTED};font-size:12px">(fill rate · waste / COGS · margin after waste)</span></h2><div class="wrap"><table><tr><th>Policy</th>{"".join(f"<th>{z}</th>" for z in zones)}</tr>{zone_rows}</table></div>
<h2>Policies</h2><div class="wrap"><table><tr><th>Policy</th><th>Fill rate</th><th>Waste / COGS</th><th>Avg inventory capital</th><th>Margin after waste</th><th>Days of inventory</th></tr>{pol_rows}</table></div>
</main></body></html>"""
    path.write_text(page)
