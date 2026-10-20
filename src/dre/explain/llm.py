"""Order explanations for store managers and vendors, with an LLM kept on a short leash.

A suggested order is only useful if the people who receive it trust it. Each order comes with its
drivers (forecast, stock that will expire, protection period, service level, case pack …). A
deterministic template always produces an explanation; when an LLM is configured, it may rephrase
the drivers in plain language, under two controls:

* it only receives the drivers, never raw data, and is told not to add facts;
* every number in its answer must match a driver value (after rounding), otherwise the answer is
  rejected and the template is used. The decision and the reason are returned for audit.

Without ANTHROPIC_API_KEY the service runs on templates only; nothing else depends on the LLM.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

SYSTEM = (
    "You explain replenishment orders for a quick-commerce dark store to a store manager. "
    "Use only the facts given. Do not add numbers that are not in the facts. Two or three short sentences, plain language."
)


@dataclass
class Explanation:
    text: str
    source: str  # "template" | "llm"
    rejected_reason: str | None = None


def template(d: dict) -> str:
    parts = [f"Order {d['order_units']:g} units ({d['order_packs']:g} × {d['pack']:g}) of {d['item_id']} for {d['store_id']}."]
    parts.append(
        f"Demand until the next delivery ({d['protection_days']:g} days) is forecast at {d['forecast_units']:.1f} units; "
        f"we stock to {d['target_units']:g} units to meet {d['service_level']:.0%} of demand."
    )
    if d.get("expiring_units", 0) > 0:
        parts.append(f"{d['expiring_units']:.1f} of the {d['on_hand_units']:g} units on hand are expected to expire before they sell, so they are not counted.")
    else:
        parts.append(f"{d['on_hand_units']:g} units are on hand.")
    if d.get("price_change_pct"):
        parts.append(f"The price changes by {d['price_change_pct']:+.0f}% in this period.")
    if d.get("event"):
        parts.append(f"Calendar: {d['event']}.")
    if d.get("capped_by_shelf"):
        parts.append("The order is limited by shelf capacity.")
    return " ".join(parts)


_NUM = re.compile(r"(?<![A-Za-z_])-?\d+(?:[.,]\d+)?")


def _allowed_numbers(d: dict) -> set[str]:
    out: set[str] = set()
    for v in d.values():
        if isinstance(v, bool) or not isinstance(v, int | float):
            if isinstance(v, str):
                out.update(_NUM.findall(v))
            continue
        for x in (v, v * 100):
            for nd in (0, 1, 2):
                out.add(f"{round(x, nd):g}")
                out.add(f"{round(x, nd):.{nd}f}")
    return out


def validate(text: str, d: dict) -> str | None:
    """None if every number in the text is a driver value; otherwise the offending number."""
    allowed = _allowed_numbers(d)
    for n in _NUM.findall(text):
        n = n.replace(",", ".")
        if n not in allowed and f"{float(n):g}" not in allowed:
            return f"number {n} is not in the drivers"
    return None


def explain(d: dict, client=None, model: str | None = None) -> Explanation:
    base = template(d)
    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return Explanation(base, "template")
        import anthropic

        client = anthropic.Anthropic()
    facts = "\n".join(f"- {k}: {v}" for k, v in d.items())
    try:
        msg = client.messages.create(
            model=model or os.environ.get("DRE_LLM_MODEL", "claude-haiku-5-5"),
            max_tokens=200,
            system=SYSTEM,
            messages=[{"role": "user", "content": f"Facts:\n{facts}\n\nExplain this order."}],
        )
        text = "".join(getattr(b, "text", "") for b in msg.content).strip()
    except Exception as e:  # the LLM is optional: never fail the order because of it
        return Explanation(base, "template", f"llm error: {type(e).__name__}")
    bad = validate(text, d)
    if bad or not text:
        return Explanation(base, "template", bad or "empty answer")
    return Explanation(text, "llm")
