"""
Step 5 — Strategy & Deployment for A-share stock price prediction.

  1. Trading strategy: convert model predictions to probabilities and trade
     when P(up) crosses a threshold (e.g. > 60%).
  2. Backtest with transaction costs (commission + slippage, A-share style).
  3. Deployment scaffolding: scheduled retraining + monitoring dashboard.

Install:  pip install matplotlib pandas numpy

Usage:
    from backtest import Strategy, Backtester
    from deploy import RetrainScheduler, MonitoringDashboard
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TRADING_DAYS = 252


# ====================================================================== #
#  Trading strategy
# ====================================================================== #
@dataclass
class Strategy:
    """
    Convert predicted returns into P(up) probabilities and positions.

    Position rule (long-only, no shorting for A-shares):
        P(up) > buy_thr  -> long (1.0)
        P(up) < sell_thr -> flat (0.0)
        else             -> hold previous position
    """
    buy_thr: float = 0.60       # buy if P(up) above this
    sell_thr: float = 0.40      # exit if P(up) below this
    prob_smoothing: int = 5     # rolling-window mean of predicted sign

    def probabilities(self, y_pred: np.ndarray) -> np.ndarray:
        """Map predicted returns -> P(up) via a rolling share of positive preds."""
        s = pd.Series(y_pred)
        p_up = s.rolling(self.prob_smoothing, min_periods=1).apply(
            lambda w: np.mean(w > 0), raw=True
        )
        return p_up.fillna(0.5).values

    def positions(self, y_pred: np.ndarray) -> np.ndarray:
        p_up = self.probabilities(y_pred)
        pos = np.zeros(len(p_up))
        current = 0.0
        for i, p in enumerate(p_up):
            if p > self.buy_thr:
                current = 1.0
            elif p < self.sell_thr:
                current = 0.0
            # else hold
            pos[i] = current
        return pos


# ====================================================================== #
#  Backtester with transaction costs
# ====================================================================== #
@dataclass
class Backtester:
    """
    Event-light backtester with A-share-style costs.

    Costs per turnover:
        commission :双边佣金 (e.g. 0.025% each way -> 0.05% round-trip)
        stamp_tax  :印花税, sell-side only (0.05%)
        slippage   :滑点 per trade
    """
    commission: float = 0.00025   # per side
    stamp_tax: float = 0.0005     # sell-side only
    slippage: float = 0.0005      # per side

    def run(self, prices: pd.Series, positions: np.ndarray) -> pd.DataFrame:
        """
        Returns a per-day equity/results frame including cost-adjusted returns,
        cumulative equity, and strategy vs buy-and-hold.
        """
        prices = pd.Series(prices).astype(float)
        pos = pd.Series(positions, index=prices.index, dtype=float)

        # Tomorrow's position is decided today -> shift to avoid look-ahead.
        held = pos.shift(1).fillna(0.0)
        turnover = held.diff().abs().fillna(0.0)            # shares traded

        asset_ret = prices.pct_change().fillna(0.0)
        gross = held * asset_ret

        # cost: commission+slippage on both sides, stamp tax on sells only
        buy_cost = turnover.clip(lower=0) * (self.commission + self.slippage)
        sell_cost = (-turnover).clip(lower=0) * (self.commission + self.slippage + self.stamp_tax)
        cost = buy_cost + sell_cost

        net_ret = gross - cost
        equity = (1 + net_ret).cumprod()
        bh_equity = (1 + asset_ret).cumprod()

        return pd.DataFrame({
            "price": prices, "position": held, "turnover": turnover,
            "gross_ret": gross, "cost": cost, "net_ret": net_ret,
            "equity": equity, "buyhold_equity": bh_equity,
        }, index=prices.index)

    # ---- performance summary -----------------------------------------
    @staticmethod
    def summarize(res: pd.DataFrame) -> dict:
        r = res["net_ret"]
        ann_ret = r.mean() * TRADING_DAYS
        ann_vol = r.std(ddof=0) * np.sqrt(TRADING_DAYS)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        eq = res["equity"]
        peak = eq.cummax()
        mdd = (eq / peak - 1).min()
        bh_ann = res["buyhold_equity"].iloc[-1] ** (TRADING_DAYS / len(r)) - 1 \
            if len(r) else np.nan
        return {
            "total_return": float(eq.iloc[-1] - 1),
            "annual_return": float(ann_ret),
            "annual_vol": float(ann_vol),
            "sharpe": float(sharpe),
            "max_drawdown": float(mdd),
            "n_trades": int((res["turnover"] > 0).sum()),
            "buyhold_annual_return": float(bh_ann),
        }


# ---------------------------------------------------------------------- #
#  Self-test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import os
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2020-01-02", "2024-06-28")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, len(idx)))),
                      index=idx)

    # fake predictions: slightly correlated with next-day sign
    fwd = close.pct_change().shift(-1).fillna(0).values
    y_pred = fwd + rng.normal(0, 0.01, len(fwd))

    strat = Strategy(buy_thr=0.60, sell_thr=0.40)
    pos = strat.positions(y_pred)

    bt = Backtester()
    res = bt.run(close, pos)
    summ = bt.summarize(res)
    print("=== Backtest summary ===")
    for k, v in summ.items():
        print(f"  {k:24s}: {v:.4f}" if isinstance(v, float) else f"  {k:24s}: {v}")
    print("\nlast 3 rows:")
    print(res[["position", "net_ret", "equity", "buyhold_equity"]].tail(3).round(4))
