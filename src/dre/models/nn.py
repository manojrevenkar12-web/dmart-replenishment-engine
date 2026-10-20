"""Global neural forecaster in PyTorch: a negative-binomial output layer on the same feature rows.

Entity embeddings for item, store, department, weekday and event; an MLP predicts the NB mean as a
multiplier on the series' recent level (so the network learns relative effects and trains fast on
CPU) and the NB dispersion. Trained by maximum likelihood, so it outputs a full distribution.
"""

from __future__ import annotations

import time

import numpy as np

from ..features import FeatureBuilder, level
from .base import Forecast, nb_alpha

EMB = {"item": 16, "store": 4, "dept": 2, "state": 2, "wday": 3, "event": 2}
LEVEL_COLS = ("last", "mean_7", "mean_14", "mean_28", "mean_56", "mean_112", "mean_364", "std_28", "unconstrained_mean_28", "same_wday_4", "lag_364")


class TorchNB:
    def __init__(
        self,
        horizon: int = 28,
        n_origins: int = 12,
        origin_step: int = 14,
        censored: str = "keep",
        fraction: float = 1.0,
        epochs: int = 3,
        batch: int = 4096,
        lr: float = 2e-3,
        seed: int = 7,
        name: str | None = None,
    ):
        self.horizon, self.n_origins, self.step, self.censored = horizon, n_origins, origin_step, censored
        self.fraction = fraction
        self.epochs, self.batch, self.lr, self.seed = epochs, batch, lr, seed
        self.name = name or "torch_nb"
        self.info: dict = {}

    def training_origins(self, origin: int) -> list[int]:
        return [origin - self.horizon - self.step * k for k in range(self.n_origins)]

    # ---------------------------------------------------------------- preprocessing
    def _prep(self, X: np.ndarray, cols: list[str], fit: bool = False):
        import torch

        idx = {c: i for i, c in enumerate(cols)}
        cat = np.stack([X[:, idx[c]] for c in EMB], 1).astype(np.int64)
        num_cols = [c for c in cols if c not in EMB]
        num = X[:, [idx[c] for c in num_cols]].astype(np.float32).copy()
        for j, c in enumerate(num_cols):
            if c in LEVEL_COLS or c == "price":
                num[:, j] = np.log1p(np.clip(num[:, j], 0, None))
        num = np.nan_to_num(num, nan=0.0, posinf=0.0, neginf=0.0)
        if fit:
            self.mu_, self.sd_ = num.mean(0), num.std(0) + 1e-6
            self.card_ = {c: int(cat[:, k].max()) + 2 for k, c in enumerate(EMB)}
        num = (num - self.mu_) / self.sd_
        base = level(X, cols)
        cat = np.minimum(cat, np.array([self.card_[c] - 1 for c in EMB]))
        return torch.from_numpy(cat), torch.from_numpy(num), torch.from_numpy(np.log(base).astype(np.float32))

    def _net(self, n_num: int):
        import torch
        from torch import nn

        card = self.card_

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.ModuleList([nn.Embedding(card[c], d) for c, d in EMB.items()])
                width = n_num + sum(EMB.values())
                self.mlp = nn.Sequential(nn.Linear(width, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 2))

            def forward(self, cat, num, log_base):
                z = torch.cat([e(cat[:, k]) for k, e in enumerate(self.emb)] + [num], 1)
                out = self.mlp(z)
                log_mu = log_base + out[:, 0]
                alpha = nn.functional.softplus(out[:, 1]) + 1e-3
                return log_mu, alpha

        return Net()

    @staticmethod
    def _nll(y, log_mu, alpha):
        import torch

        r = 1.0 / alpha
        mu = torch.exp(log_mu)
        return -(torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1) + r * (torch.log(r) - torch.log(r + mu)) + y * (log_mu - torch.log(r + mu))).mean()

    # ---------------------------------------------------------------- API
    def fit(self, fb: FeatureBuilder, origin: int) -> None:
        import torch

        torch.manual_seed(self.seed)
        torch.set_num_threads(max(1, torch.get_num_threads()))
        t0 = time.time()
        r = fb.rows(self.training_origins(origin), self.horizon, fraction=self.fraction, seed=self.seed)
        keep = r.valid & ~np.isnan(r.y)
        if self.censored == "mask":
            keep &= ~r.censored
        self.columns = r.columns
        cat, num, lb = self._prep(r.X[keep], r.columns, fit=True)
        y = torch.from_numpy(r.y[keep].astype(np.float32))
        self.net = self._net(num.shape[1])
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        steps = self.epochs * int(np.ceil(len(y) / self.batch))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps)
        g = torch.Generator().manual_seed(self.seed)
        self.net.train()
        losses = []
        for _ in range(self.epochs):
            perm = torch.randperm(len(y), generator=g)
            tot = 0.0
            for i in range(0, len(y), self.batch):
                b = perm[i : i + self.batch]
                log_mu, alpha = self.net(cat[b], num[b], lb[b])
                loss = self._nll(y[b], log_mu, alpha)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                tot += float(loss.detach()) * len(b)
            losses.append(round(tot / len(y), 4))
        self.net.eval()
        last = r.origin == max(r.origin)
        mu, _ = self._infer(r.X[last])
        yy = np.full((fb.p.n, self.horizon), np.nan)
        mm = np.full((fb.p.n, self.horizon), np.nan)
        yy[r.series[last], r.h[last] - 1] = r.y[last]
        mm[r.series[last], r.h[last] - 1] = mu
        self.alpha_ = nb_alpha(yy, mm, fb.codes["dept"] * 100 + fb.codes["store"])
        self.info = {"train_rows": int(keep.sum()), "train_seconds": round(time.time() - t0, 1), "epoch_nll": losses}

    def _infer(self, X: np.ndarray):
        import torch

        cat, num, lb = self._prep(X, self.columns)
        out_mu, out_a = [], []
        with torch.no_grad():
            for i in range(0, len(lb), 65536):
                lm, a = self.net(cat[i : i + 65536], num[i : i + 65536], lb[i : i + 65536])
                out_mu.append(torch.exp(lm).numpy())
                out_a.append(a.numpy())
        return np.concatenate(out_mu), np.concatenate(out_a)

    def predict(self, fb: FeatureBuilder, origin: int, horizon: int) -> Forecast:
        r = fb.rows([origin], horizon)
        mu, a = self._infer(r.X)
        mean = np.zeros((fb.p.n, horizon))
        alpha = np.zeros((fb.p.n, horizon))
        mean[r.series, r.h - 1] = mu
        alpha[r.series, r.h - 1] = a
        listed = ~np.isnan(fb.p.prices[:, origin + 1 : origin + horizon + 1])
        return Forecast(np.where(listed, mean, 0.0), alpha, origin, self.name)
