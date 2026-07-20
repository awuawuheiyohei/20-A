"""
Step 5 (cont.) — Deployment: scheduled retraining + monitoring dashboard.

  1. RetrainScheduler: periodic retrain of the model on fresh data
     (cron-like; can be driven by an external scheduler or the loop below).
  2. MonitoringDashboard: plots equity curve, drawdown, rolling Sharpe,
     prediction-vs-actual drift — to catch model degradation in production.

Install:  pip install matplotlib pandas numpy

Usage:
    from deploy import RetrainScheduler, MonitoringDashboard
    sched = RetrainScheduler(data_loader=load_fresh_features, every_days=30)
    sched.start()                          # blocking loop (or wire to cron)
    MonitoringDashboard().render(results, "dashboard.png")
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TRADING_DAYS = 252


# ====================================================================== #
#  Scheduled retraining
# ====================================================================== #
@dataclass
class RetrainScheduler:
    """
    Periodically retrains a model on freshly loaded data and persists it.

    Parameters
    ----------
    data_loader : callable -> (X_tab, y, X_seq, dates)
        Pulls the latest feature matrix each cycle.
    train_fn : callable (X_tab, y) -> fitted_model
        Defaults to a Random Forest via ModelEngine.
    every_days : int
        Wall-clock cadence in days (simulated with sleep in the loop;
        in production, drive this from cron / Airflow / systemd timer).
    """

    data_loader: Callable
    train_fn: Callable | None = None
    every_days: int = 30
    max_runs: int = 3            # safety cap for the demo loop
    model: object = field(default=None, repr=False)

    def __post_init__(self):
        if self.train_fn is None:
            self.train_fn = self._default_train_fn

    @staticmethod
    def _default_train_fn(X_tab, y):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=200, max_depth=12,
                                     n_jobs=-1, random_state=42).fit(X_tab, y)

    def retrain_once(self):
        """One retrain cycle: load fresh data, fit, evaluate, persist."""
        X_tab, y, X_seq, dates = self.data_loader()
        n_test = max(1, int(0.2 * len(y)))
        tr, te = np.arange(0, len(y) - n_test), np.arange(len(y) - n_test, len(y))
        self.model = self.train_fn(X_tab[tr], y[tr])
        preds = self.model.predict(X_tab[te])
        rmse = float(np.sqrt(np.mean((preds - y[te]) ** 2)))
        da = float(np.mean(np.sign(preds) == np.sign(y[te])) * 100)
        print(f"[retrain] samples={len(y)}  oof_rmse={rmse:.5f}  dir_acc={da:.1f}%")
        self._persist(self.model)
        return {"rmse": rmse, "dir_acc": da}

    @staticmethod
    def _persist(model):
        import os, joblib
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/latest_model.joblib")

    def start(self):
        """Blocking demo loop. In prod, replace with a real scheduler."""
        print(f"[scheduler] retraining every {self.every_days} days "
              f"(max {self.max_runs} runs for demo)")
        for run in range(self.max_runs):
            try:
                self.retrain_once()
            except Exception as exc:
                print(f"[retrain][error] {exc}")
            if run < self.max_runs - 1:
                time.sleep(self.every_days)   # placeholder cadence
        print("[scheduler] done")


# ====================================================================== #
#  Monitoring dashboard
# ====================================================================== #
@dataclass
class MonitoringDashboard:
    """Render a PNG snapshot of strategy health + model drift for monitoring."""

    @staticmethod
    def _rolling_sharpe(returns: pd.Series, window: int = 63) -> pd.Series:
        mu = returns.rolling(window, min_periods=window).mean() * TRADING_DAYS
        sd = returns.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(TRADING_DAYS)
        return mu / sd.replace(0, np.nan)

    def render(self, backtest_res: pd.DataFrame, oof: pd.DataFrame | None = None,
               path: str = "dashboard.png"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = 3 if oof is not None else 2
        fig, axes = plt.subplots(rows, 1, figsize=(11, 3 * rows))

        # 1. Equity vs buy-and-hold
        ax = axes[0]
        ax.plot(backtest_res.index, backtest_res["equity"], label="Strategy")
        ax.plot(backtest_res.index, backtest_res["buyhold_equity"],
                label="Buy & Hold", alpha=0.7)
        ax.set_title("Equity curve (net of costs)")
        ax.legend(); ax.grid(alpha=0.3)

        # 2. Drawdown + rolling Sharpe
        ax = axes[1]
        eq = backtest_res["equity"]
        dd = eq / eq.cummax() - 1
        ax.fill_between(dd.index, dd, 0, color="red", alpha=0.4, label="Drawdown")
        ax.set_title("Drawdown")
        ax.legend(); ax.grid(alpha=0.3)

        if oof is not None:
            # 3. Rolling prediction directional accuracy (model drift)
            ax = axes[2]
            correct = (np.sign(oof["y_pred"]) == np.sign(oof["y_true"])).astype(float)
            roll_da = correct.rolling(63, min_periods=20).mean() * 100
            ax.plot(roll_da.index, roll_da, label="Rolling Dir-Acc (%)")
            ax.axhline(50, ls="--", color="gray", label="Random (50%)")
            ax.set_title("Model drift — rolling directional accuracy")
            ax.legend(); ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(path, dpi=110)
        print(f"[dashboard] saved -> {path}")
        return path


# ---------------------------------------------------------------------- #
#  Self-test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import os
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2020-01-02", "2024-06-28")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, len(idx)))),
                      index=idx)

    # ---- build a fake backtest result + oof frame for the dashboard ----
    from backtest import Strategy, Backtester
    fwd = close.pct_change().shift(-1).fillna(0).values
    y_pred = fwd + rng.normal(0, 0.01, len(fwd))
    pos = Strategy().positions(y_pred)
    res = Backtester().run(close, pos)
    oof = pd.DataFrame({"y_true": fwd, "y_pred": y_pred}, index=idx)

    MonitoringDashboard().render(res, oof, path="dashboard.png")

    # ---- retrain scheduler (1 quick run, no real wait) ----
    def loader():
        feats = pd.DataFrame({"close": close.values}, index=idx)
        from model_train import ModelEngine
        s = ModelEngine(sequence_len=20)
        X_tab, y, X_seq, dates = s.prepare(feats)
        return X_tab, y, X_seq, dates

    sched = RetrainScheduler(data_loader=loader, every_days=1, max_runs=1)
    sched.retrain_once()
    print("DONE")
