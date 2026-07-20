"""
Step 4 — Risk & Validation for A-share stock price prediction.

  1. Walk-forward validation (expanding/rolling window) to simulate real trading.
  2. Regime-robustness testing across bull / bear / sideways markets.
  3. Limitations documentation — prediction is probabilistic, not guaranteed.

Usage:
    from validation import WalkForward, RegimeTester
    from model_train import ModelEngine, evaluate

    suite = ModelEngine(sequence_len=20)
    X_tab, y, X_seq, dates = suite.prepare(features_df)
    wf = WalkForward(n_folds=5, train_size=252)
    oof = wf.run(suite, X_tab, y, dates)         # out-of-fold predictions
    regimes = RegimeTester().by_regime(oof)       # metrics per regime
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TRADING_DAYS = 252


# ====================================================================== #
#  Walk-forward validation
# ====================================================================== #
@dataclass
class WalkForward:
    """
    Time-series walk-forward validation.

    Each fold trains on a window and tests on the immediately following block,
    then the window rolls forward — exactly how a model would be re-deployed in
    live trading. No shuffling, no look-ahead.
    """

    n_folds: int = 5
    train_size: int = 252        # ~1 year of training data per fold
    test_size: int = 63          # ~1 quarter of out-of-sample testing per fold
    step: int | None = None      # roll size; defaults to test_size

    def __post_init__(self):
        if self.step is None:
            self.step = self.test_size

    def folds(self, n: int):
        """Yield (train_idx, test_idx) index arrays over n samples."""
        start = 0
        for k in range(self.n_folds):
            tr_end = start + self.train_size
            te_end = tr_end + self.test_size
            if te_end > n:
                break
            yield (np.arange(start, tr_end), np.arange(tr_end, te_end))
            start += self.step

    def run(self, suite, X_tab, y, dates) -> pd.DataFrame:
        """
        Run walk-forward with the suite's Random Forest baseline (cheap, stable).
        Returns out-of-fold predictions concatenated into one frame.
        """
        preds, actuals, test_dates, fold_id = [], [], [], []
        for k, (tr, te) in enumerate(self.folds(len(y))):
            p = suite.train_random_forest(X_tab[tr], y[tr], X_tab[te])
            preds.extend(p)
            actuals.extend(y[te])
            test_dates.extend(dates[te])
            fold_id.extend([k] * len(te))
            print(f"  fold {k}: train[{tr[0]}:{tr[-1]}] test[{te[0]}:{te[-1]}] "
                  f"rmse={np.sqrt(np.mean((p - y[te]) ** 2)):.5f}")

        return pd.DataFrame({
            "date": pd.DatetimeIndex(test_dates),
            "y_true": actuals, "y_pred": preds, "fold": fold_id,
        }).set_index("date").sort_index()


# ====================================================================== #
#  Market-regime testing
# ====================================================================== #
@dataclass
class RegimeTester:
    """
    Classify each period as bull / bear / sideways from realized returns,
    then report model metrics per regime.
    """
    window: int = 63            # ~quarterly rolling window for regime label
    bull_thr: float = 0.10      # annualized return above this  => bull
    bear_thr: float = -0.10     # below this                       => bear

    @staticmethod
    def label_regime(df: pd.DataFrame, price_col: str = "close",
                     window: int = 63, bull_thr: float = 0.10,
                     bear_thr: float = -0.10) -> pd.Series:
        """Return a 'bull'/'bear'/'sideways' label per date."""
        ret = df[price_col].pct_change()
        ann = ret.rolling(window, min_periods=window).mean() * TRADING_DAYS
        label = pd.Series("sideways", index=df.index)
        label[ann > bull_thr] = "bull"
        label[ann < bear_thr] = "bear"
        return label

    def by_regime(self, oof: pd.DataFrame, features: pd.DataFrame | None = None,
                  price_col: str = "close") -> pd.DataFrame:
        """
        Compute RMSE / MAPE / directional accuracy per regime.

        If ``features`` (with a price column) is supplied, regimes are derived
        from it; otherwise the sign of y_true is used as a coarse proxy.
        """
        from model_train import evaluate

        if features is not None and price_col in features.columns:
            regime = self.label_regime(features, price_col, self.window,
                                       self.bull_thr, self.bear_thr)
            oof = oof.join(regime.rename("regime"), how="inner")
        else:
            oof = oof.copy()
            oof["regime"] = np.where(oof["y_true"] > 0, "bull", "bear")

        rows = []
        for r, g in oof.groupby("regime"):
            m = evaluate(g["y_true"].values, g["y_pred"].values)
            m["regime"] = r
            m["n_days"] = len(g)
            rows.append(m)
        return pd.DataFrame(rows).set_index("regime")


# ====================================================================== #
#  Limitations documentation
# ====================================================================== #
LIMITATIONS = """
=====================================================================
  LIMITATIONS — READ BEFORE TRADING
=====================================================================
1. Prediction is PROBABILISTIC, not guaranteed. A directional-accuracy
   of 55% still means the model is wrong 45% of the time on any given day.
2. Past performance does not predict future results. Market regimes shift,
   and a model fit on a bull market can fail catastrophically in a crash.
3. Walk-forward backtests still suffer from survivorship and look-ahead
   biases if data cleaning is not time-consistent (never fill future NaNs).
4. Transaction costs, slippage, and liquidity (limit-up/down, trading
   halts on A-shares) are approximated; real fills differ.
5. Macro data is published with a lag — using the release date (not the
   reporting period) is essential to avoid look-ahead bias.
6. Overfitting: with 60+ features and limited samples, in-sample metrics
   always look better than out-of-sample. Trust the walk-forward numbers.
7. This is research code, NOT investment advice. Do not deploy capital
   on the strength of these models alone.
=====================================================================
"""


# ---------------------------------------------------------------------- #
#  Self-test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import os
    from model_train import ModelEngine

    path = "data_cache/features.parquet"
    feats = pd.read_parquet(path) if os.path.exists(path) else None
    if feats is None:
        rng = np.random.default_rng(0)
        idx = pd.bdate_range("2019-01-02", "2024-06-28")
        n = len(idx)
        close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.013, n)))
        feats = pd.DataFrame({"close": close, "volume": rng.integers(1e6, 5e6, n),
                              "ma5": close, "ma20": close, "rsi": 50.0,
                              "macd": 0.0, "bb_width": 0.02}, index=idx)
        feats.index.name = "date"

    suite = ModelEngine(sequence_len=20)
    X_tab, y, X_seq, dates = suite.prepare(feats)

    print("=== Walk-forward validation ===")
    wf = WalkForward(n_folds=5, train_size=252, test_size=63)
    oof = wf.run(suite, X_tab, y, dates)
    from model_train import evaluate
    print("overall:", {k: round(v, 4) for k, v in evaluate(oof.y_true, oof.y_pred).items()})

    print("\n=== Regime robustness ===")
    rt = RegimeTester()
    regimes = rt.by_regime(oof, feats, price_col="close")
    print(regimes[["n_days", "rmse", "mape", "directional_acc"]].round(4))

    print(LIMITATIONS)
