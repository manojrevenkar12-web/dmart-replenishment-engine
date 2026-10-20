"""A file-based model registry: every promoted model is stored with what is needed to reproduce,
audit and serve it — artifact, data and config hashes, code version, backtest metrics and the
gate decision that promoted it. Swappable for MLflow or Vertex AI Model Registry; the contract
(artifact + metadata JSON) stays the same."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .models.base import Forecast


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip() or None
    except Exception:
        return None


def save(root: Path, model, forecast: Forecast, panel, economics, meta: dict) -> Path:
    """Store the champion and the forecast the service reads."""
    v = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    d = root / f"{model.name}-{v}"
    d.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "model") and hasattr(model.model, "booster_"):
        model.model.booster_.save_model(str(d / "model.txt"))
    elif hasattr(model, "net"):
        import torch

        torch.save(model.net.state_dict(), d / "model.pt")
    o = forecast.origin
    h = forecast.mean.shape[1]
    dates = panel.calendar.date.iloc[o + 1 : o + 1 + h].dt.strftime("%Y-%m-%d").to_numpy()
    fc = pd.DataFrame(
        {
            "item_id": np.repeat(panel.ids.item_id.to_numpy(), h),
            "store_id": np.repeat(panel.ids.store_id.to_numpy(), h),
            "date": np.tile(dates, panel.n),
            "mean": forecast.mean.ravel().astype(np.float32),
            "alpha": forecast.alpha.ravel().astype(np.float32),
        }
    )
    fc.to_parquet(d / "forecast.parquet")
    eco = pd.DataFrame(
        {
            "item_id": panel.ids.item_id,
            "store_id": panel.ids.store_id,
            "dept_id": panel.ids.dept_id,
            "shelf_life_days": economics.life,
            "pack": economics.pack,
            "capacity": economics.capacity,
            "delivery_weekdays": [",".join(str(k) for k in np.flatnonzero(r)) for r in economics.delivery],
        }
    )
    for k in range(h + 1):
        eco[f"price_{k}"] = economics.price[:, o + k]
        eco[f"cost_{k}"] = economics.cost[:, o + k]
    eco.to_parquet(d / "economics.parquet")
    meta = meta | {
        "version": v,
        "model": model.name,
        "trained_until_day_index": o,
        "forecast_start": str(dates[0]),
        "horizon": h,
        "holding_rate_per_day": economics.holding_rate,
        "git": git_sha(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (d / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    (root / "CHAMPION").write_text(d.name)
    return d


def champion(root: Path) -> Path:
    return root / (root / "CHAMPION").read_text().strip()
