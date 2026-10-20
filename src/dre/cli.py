"""`dre run` (full pipeline), `dre serve` (order API) and `dre download`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dre")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="backtest, gate, replenishment simulation, experiment, monitoring, registry and reports")
    r.add_argument("--config", default="config/dre.yaml")
    r.add_argument("--out", default="out")
    r.add_argument("--stores", nargs="*", help="limit to some stores, e.g. CA_1")
    r.add_argument("--fast", action="store_true", help="lighter models and two folds (CI smoke run)")
    r.add_argument("--skip-torch", action="store_true")
    s = sub.add_parser("serve", help="serve forecasts and order advice from the registry champion")
    s.add_argument("--registry", default="out/registry")
    s.add_argument("--port", type=int, default=8080)
    d = sub.add_parser("download", help="fetch the M5 data")
    d.add_argument("--root", default="data/m5")
    a = p.parse_args(argv)
    if a.cmd == "run":
        from .pipeline import run

        rep = run(Path(a.config), Path(a.out), a.fast, a.stores, a.skip_torch)
        print(json.dumps({"champion": rep["champion"], "backtest": rep["backtest"][rep["champion"]]}, indent=2, default=str))
    elif a.cmd == "serve":
        import os

        import uvicorn

        os.environ["DRE_REGISTRY"] = a.registry
        uvicorn.run("dre.service.api:app", host="0.0.0.0", port=a.port)  # noqa: S104
    elif a.cmd == "download":
        from .data import m5

        print(m5.download(Path(a.root)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
