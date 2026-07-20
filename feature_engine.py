"""
Step 2 — Feature Engineering for A-share stock price prediction.

Builds on the aligned panel produced by ``data_prep.py`` (Step 1). It adds:
  1. Technical indicators: moving averages, RSI, MACD, Bollinger Bands.
  2. Lagged features and rolling statistics for momentum & volatility.
  3. Encoded categorical events: earnings reports & policy announcements.

Usage:
    from data_prep import DataPrep
    from feature_engine import FeatureEngine

    panel = DataPrep(...).align_time_series(stock, macro)   # Step 1 output
    fe = FeatureEngine()
    features = fe.build(panel)                               # full feature matrix
    fe.save(features, "features.parquet")

    # Or build incrementally:
    df = fe.add_technical(panel)
    df = fe.add_lags_and_rolling(df)
    df = fe.add_events(df, earnings_dates=[...], policy_dates=[...])
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngine:
    """Transforms an aligned OHLCV+macro panel into a model-ready feature set."""

    # ---- technical-indicator windows (all configurable) ---------------
    ma_windows: tuple[int, ...] = (5, 10, 20, 60)
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_window: int = 20
    bb_n_std: float = 2.0

    # ---- lagged / rolling windows ------------------------------------
    lag_periods: tuple[int, ...] = (1, 2, 3, 5, 10, 20)
    rolling_windows: tuple[int, ...] = (5, 10, 20, 60)

    # ================================================================== #
    #  1. TECHNICAL INDICATORS
    # ================================================================== #
    def add_technical(self, df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
        """Add MA, RSI, MACD, Bollinger Bands."""
        df = df.copy()
        close = df[price_col]

        # --- Moving averages + price/MA ratio ---------------------------
        for w in self.ma_windows:
            df[f"ma{w}"] = close.rolling(w, min_periods=w).mean()
            df[f"close_over_ma{w}"] = close / df[f"ma{w}"] - 1.0

        # --- RSI (Wilder's smoothing) -----------------------------------
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        # Wilder uses an exponential moving average equivalent to alpha=1/window.
        avg_gain = gain.ewm(alpha=1 / self.rsi_window, adjust=False, min_periods=self.rsi_window).mean()
        avg_loss = loss.ewm(alpha=1 / self.rsi_window, adjust=False, min_periods=self.rsi_window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - 100 / (1 + rs)
        df["rsi"] = df["rsi"].fillna(50.0)            # neutral until enough data

        # --- MACD -------------------------------------------------------
        ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=self.macd_signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # --- Bollinger Bands --------------------------------------------
        mid = close.rolling(self.bb_window, min_periods=self.bb_window).mean()
        sd = close.rolling(self.bb_window, min_periods=self.bb_window).std(ddof=0)
        df["bb_mid"] = mid
        df["bb_upper"] = mid + self.bb_n_std * sd
        df["bb_lower"] = mid - self.bb_n_std * sd
        # %B: position of price within the band (0=lower, 1=upper)
        df["bb_pctb"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
        # Bandwidth: volatility proxy
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / mid.replace(0, np.nan)

        return df

    # ================================================================== #
    #  2. LAGGED FEATURES & ROLLING STATISTICS
    # ================================================================== #
    def add_lags_and_rolling(self, df: pd.DataFrame,
                             price_col: str = "close",
                             ret_col: str | None = None) -> pd.DataFrame:
        """
        Lagged returns + rolling momentum/volatility stats.

        Creates a daily-return column (if absent) and, for each rolling
        window, mean / std / min / max / skew — capturing momentum and
        volatility regimes.
        """
        df = df.copy()
        if ret_col is None:
            ret_col = "ret_1d"
        if ret_col not in df.columns:
            df[ret_col] = df[price_col].pct_change()

        # --- Lagged returns (momentum at multiple horizons) -------------
        for lag in self.lag_periods:
            df[f"ret_lag{lag}"] = df[ret_col].shift(lag)

        # --- Lagged price levels ----------------------------------------
        for lag in (1, 5, 20):
            df[f"close_lag{lag}"] = df[price_col].shift(lag)

        # --- Rolling statistics on daily returns ------------------------
        for w in self.rolling_windows:
            roll = df[ret_col].rolling(w, min_periods=w)
            df[f"ret_mean{w}"] = roll.mean()            # momentum
            df[f"ret_std{w}"] = roll.std(ddof=0)        # volatility
            df[f"ret_min{w}"] = roll.min()              # downside risk
            df[f"ret_max{w}"] = roll.max()              # upside
            df[f"ret_skew{w}"] = roll.skew()            # asymmetry

        # --- Rolling volume statistics (liquidity) ----------------------
        if "volume" in df.columns:
            for w in (5, 20):
                df[f"vol_mean{w}"] = df["volume"].rolling(w, min_periods=w).mean()
                df[f"vol_std{w}"] = df["volume"].rolling(w, min_periods=w).std(ddof=0)
            df["vol_ratio5_20"] = df["vol_mean5"] / df["vol_mean20"].replace(0, np.nan)

        # --- Realized volatility (annualized, √252 trading days) -------
        df["realized_vol20"] = df[ret_col].rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252)

        return df

    # ================================================================== #
    #  3. CATEGORICAL EVENT ENCODING
    # ================================================================== #
    @staticmethod
    def add_events(df: pd.DataFrame,
                   earnings_dates: list[str] | pd.DatetimeIndex | None = None,
                   policy_dates: list[str] | pd.DatetimeIndex | None = None,
                   event_windows: tuple[int, int] = (-3, 3)) -> pd.DataFrame:
        """
        Encode earnings reports and policy announcements.

        Produces:
          * one-hot flags for the event day itself
          * event-window dummies spanning ``event_windows`` trading days
            around each event (e.g. ±3) to capture pre/post drift
          * a categorical ``event_type`` column (none/earnings/policy)
            integer-encoded via pandas codes

        Parameters
        ----------
        earnings_dates, policy_dates : date-like
            Dates of earnings releases and policy announcements.
        event_windows : (pre, post)
            Trading-day offsets around each event to flag, inclusive.
        """
        df = df.copy()
        dates = df.index

        def _flag(event_dates, prefix):
            """Return DataFrame with on-day and window one-hots for one event type."""
            flag = pd.Series(0, index=dates, dtype="int8")
            cols = {f"{prefix}_day": flag.copy()}
            pre, post = event_windows
            window = pd.Series(0, index=dates, dtype="int8")
            ed = pd.to_datetime(pd.Series(event_dates)).dropna().dt.normalize()
            for d in ed:
                # one-hot on the exact day
                if d in dates:
                    cols[f"{prefix}_day"].loc[d] = 1
                # window flag: any trading day within [d+pre, d+post]
                lo = d + pd.Timedelta(days=pre)   # offsets handled below per-row
                hi = d + pd.Timedelta(days=post)
                mask = (dates >= d) & (dates <= d)  # placeholder, replaced
                # Use trading-day offsets instead of calendar offsets:
                pos = dates.get_indexer([d], method="nearest") if d in dates else np.array([-1])
            # Build window flags using trading-day index positions for accuracy
            window[:] = 0
            ed_in_range = [d for d in ed]
            idx_list = list(dates)
            for d in ed:
                # find nearest trading day position
                if d in idx_list:
                    p = idx_list.index(d)
                else:
                    # snap to first available trading day >= d
                    after = [i for i, x in enumerate(idx_list) if x >= d]
                    if not after:
                        continue
                    p = after[0]
                for k in range(pre, post + 1):
                    j = p + k
                    if 0 <= j < len(idx_list):
                        window.iloc[j] = 1
            cols[f"{prefix}_window"] = window
            return pd.DataFrame(cols, index=dates)

        if earnings_dates is not None and len(earnings_dates):
            ed_df = _flag(earnings_dates, "earnings")
            df = pd.concat([df, ed_df], axis=1)
            earnings_on = ed_df["earnings_day"].astype(bool)
        else:
            earnings_on = pd.Series(False, index=dates)

        if policy_dates is not None and len(policy_dates):
            po_df = _flag(policy_dates, "policy")
            df = pd.concat([df, po_df], axis=1)
            policy_on = po_df["policy_day"].astype(bool)
        else:
            policy_on = pd.Series(False, index=dates)

        # Categorical event_type: 0=none, 1=earnings, 2=policy
        etype = np.where(earnings_on.values, 1, np.where(policy_on.values, 2, 0))
        df["event_type"] = pd.Categorical(
            pd.Series(etype, index=dates).map({0: "none", 1: "earnings", 2: "policy"})
        )
        # Integer-encoded version for tree/linear models
        df["event_type_code"] = df["event_type"].cat.codes.astype("int8")

        return df

    # ================================================================== #
    #  Orchestration
    # ================================================================== #
    def build(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Run the full feature pipeline. Optional kwargs pass through to add_events."""
        df = self.add_technical(df)
        df = self.add_lags_and_rolling(df)
        df = self.add_events(df, **kwargs)
        return df

    def save(self, df: pd.DataFrame, path: str) -> None:
        import os
        os.makedirs("./data_cache", exist_ok=True)
        full = path if "/" in path else os.path.join("./data_cache", path)
        df.to_parquet(full) if full.endswith(".parquet") else df.to_csv(full)
        print(f"[ok] saved {df.shape} -> {full}")


# ---------------------------------------------------------------------- #
#  Demo / self-test (uses synthetic data so it runs offline)
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2022-01-03", "2024-06-28")
    n = len(idx)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    volume = rng.integers(1_000_000, 5_000_000, n)
    panel = pd.DataFrame({
        "open": close * rng.uniform(0.99, 1.01, n),
        "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": volume,
        "gdp_yoy": 5.0, "cpi_yoy": 2.0, "lpr_1y": 3.45,
        "symbol": "TEST",
    }, index=idx)
    panel.index.name = "date"

    fe = FeatureEngine()
    feats = fe.build(
        panel,
        earnings_dates=["2022-04-28", "2022-08-25", "2023-04-27"],
        policy_dates=["2022-10-20", "2023-07-24"],
    )
    print("feature matrix:", feats.shape)
    tech = ["ma5", "ma20", "rsi", "macd", "macd_hist", "bb_pctb", "bb_width"]
    lag = ["ret_lag1", "ret_lag5", "ret_std20", "realized_vol20", "vol_ratio5_20"]
    evt = ["earnings_day", "earnings_window", "policy_day", "event_type", "event_type_code"]
    print(feats[tech + lag + evt].tail(5).round(3))
    fe.save(feats, "features.parquet")
    print("DONE")
