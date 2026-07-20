"""
Step 1 — Data Preparation for A-share stock price prediction.

Covers:
  1. Download historical A-share OHLC + volume data.
  2. Collect macroeconomic indicators (GDP growth, CPI, interest rates).
  3. Clean and normalize data, handle missing values.
  4. Align time series across stock and macro data.

Data source: akshare (https://akshare.akfamily.xyz) — free, no API token required.
Install:  pip install akshare pandas numpy

Usage:
    from data_prep import DataPrep
    dp = DataPrep(start_date="2015-01-01", end_date="2024-12-31")
    stock = dp.get_stock_data("000001")        # 平安银行 (Shenzhen)
    macro = dp.get_macro_data()                # GDP, CPI, interest rates
    aligned = dp.align_time_series(stock, macro)
    dp.save(aligned, "aligned_panel.parquet")
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _retry(fn, *args, retries: int = 4, backoff: float = 2.0, **kw):
    """Retry a network-bound call with exponential backoff."""
    last_exc = None
    for attempt in range(retries):
        try:
            return fn(*args, **kw)
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
    raise last_exc


# akshare is imported lazily inside the class so the module can still be
# imported (and inspected) on machines where it is not yet installed.


@dataclass
class DataPrep:
    """Orchestrates A-share + macro data collection, cleaning, and alignment."""

    start_date: str = "2015-01-01"
    end_date: str = "2024-12-31"
    adjust: str = "qfq"          # qfq=前复权(forward-adjusted), hfq=后复权, ""=不复权
    macro_freq: str = "ME"        # resample macro to month-end before alignment
    cache_dir: str = "./data_cache"
    _ak: object = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------ #
    #  Lazy akshare loader
    # ------------------------------------------------------------------ #
    @property
    def ak(self):
        if self._ak is None:
            try:
                import akshare as ak
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "akshare is required. Install with: pip install akshare"
                ) from e
            self._ak = ak
        return self._ak

    # ================================================================== #
    #  1. STOCK DATA  (OHLC + volume)
    # ================================================================== #
    def get_stock_data(self, symbol: str, adjust: str | None = None) -> pd.DataFrame:
        """
        Download daily OHLCV for one A-share.

        Parameters
        ----------
        symbol : str
            6-digit ticker, e.g. "000001" (平安银行), "600519" (贵州茅台).
        adjust : str
            "" / "qfq" / "hfq". Falls back to self.adjust.
        """
        adjust = self.adjust if adjust is None else adjust
        df = _retry(
            self.ak.stock_zh_a_hist,
            symbol=symbol,
            period="daily",
            start_date=self.start_date.replace("-", ""),
            end_date=self.end_date.replace("-", ""),
            adjust=adjust,
        )
        # akshare returns Chinese column names; rename to canonical English.
        rename = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_change",
            "涨跌额": "change",
            "换手率": "turnover",
        }
        df = df.rename(columns=rename)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df["symbol"] = symbol
        return df

    def get_multiple_stocks(self, symbols: list[str]) -> pd.DataFrame:
        """Download OHLCV for many tickers, concatenated into a long panel."""
        frames = []
        for sym in symbols:
            try:
                frames.append(self.get_stock_data(sym))
            except Exception as exc:  # network / delisting errors
                print(f"[warn] {sym}: {exc}")
        if not frames:
            return pd.DataFrame()
        panel = pd.concat(frames)
        return panel.sort_index()

    # ================================================================== #
    #  2. MACRO DATA  (GDP growth, CPI, interest rates)
    # ================================================================== #
    def get_macro_data(self) -> pd.DataFrame:
        """
        Collect China macro indicators. Each series is fetched independently
        and merged into one wide DataFrame indexed by month.
        """
        pieces = {}

        # --- GDP growth (季度同比, %) -------------------------------------
        try:
            gdp = _retry(self.ak.macro_china_gdp)
            col = "国内生产总值-同比增长"
            if col in gdp.columns:
                gdp = gdp[["季度", col]].rename(columns={"季度": "date", col: "gdp_yoy"})
                # '2024年第1季度' -> quarter-end timestamp via Period
                qstr = gdp["date"].astype(str).str.extract(r"(\d{4})年第?(\d)季度?")[0]
                yr = gdp["date"].astype(str).str.extract(r"(\d{4})年第?(\d)季度?")[1]
                period = pd.PeriodIndex(yr + "Q" + qstr, freq="Q")
                gdp["date"] = period.to_timestamp(how="end").normalize()
                pieces["gdp_yoy"] = gdp.set_index("date")["gdp_yoy"]
        except Exception as exc:
            print(f"[warn] GDP: {exc}")

        # --- CPI (月度, %) — uses 日期 + 今值 -----------------------------
        try:
            cpi = _retry(self.ak.macro_china_cpi_monthly)
            col_date, col_val = "日期", "今值"
            if col_date in cpi.columns and col_val in cpi.columns:
                cpi = cpi[[col_date, col_val]].rename(
                    columns={col_date: "date", col_val: "cpi_yoy"}
                )
                cpi["date"] = pd.to_datetime(cpi["date"], errors="coerce")
                cpi = cpi.dropna(subset=["date"]).set_index("date").sort_index()
                pieces["cpi_yoy"] = cpi["cpi_yoy"]
        except Exception as exc:
            print(f"[warn] CPI: {exc}")

        # --- Interest rates: 1-year LPR (1年期贷款市场报价利率) ------------
        try:
            rate = _retry(self.ak.macro_china_lpr)
            rate["date"] = pd.to_datetime(rate["TRADE_DATE"], errors="coerce")
            rate = rate.dropna(subset=["date"]).set_index("date").sort_index()
            # LPR1Y only has values since 2019; RATE_1 (1Y deposit) covers earlier history.
            series = rate["LPR1Y"].fillna(rate.get("RATE_1"))
            pieces["lpr_1y"] = series
        except Exception as exc:
            print(f"[warn] interest rate: {exc}")

        if not pieces:
            return pd.DataFrame()

        macro = pd.concat(pieces, axis=1)
        macro.index.name = "date"
        return macro.sort_index()

    # ================================================================== #
    #  3. CLEAN & NORMALIZE
    # ================================================================== #
    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values and drop duplicates."""
        df = df.copy()
        df = df[~df.index.duplicated(keep="last")]      # 去重日期
        df = df.replace([np.inf, -np.inf], np.nan)
        # numeric columns: forward-fill then back-fill (time-series friendly)
        num = df.select_dtypes(include=[np.number]).columns
        df[num] = df[num].ffill().bfill()
        # any still-missing numeric -> column median
        df[num] = df[num].fillna(df[num].median())
        # categorical/object: fill with mode
        obj = df.select_dtypes(include=["object"]).columns
        for c in obj:
            df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "NA")
        return df

    @staticmethod
    def normalize(df: pd.DataFrame, method: str = "zscore",
                  exclude: list[str] | None = None) -> pd.DataFrame:
        """
        Normalize numeric columns.

        method:
            'zscore'  -> (x - mean) / std
            'minmax'  -> (x - min) / (max - min)
            'robust'  -> (x - median) / IQR
        """
        df = df.copy()
        exclude = set(exclude or [])
        num = [c for c in df.select_dtypes(include=[np.number]).columns
               if c not in exclude]
        if not num:
            return df
        X = df[num]
        if method == "zscore":
            df[num] = (X - X.mean()) / (X.std(ddof=0).replace(0, np.nan))
        elif method == "minmax":
            df[num] = (X - X.min()) / ((X.max() - X.min()).replace(0, np.nan))
        elif method == "robust":
            q1, q3 = X.quantile(0.25), X.quantile(0.75)
            df[num] = (X - X.median()) / ((q3 - q1).replace(0, np.nan))
        else:
            raise ValueError(f"unknown method: {method}")
        df[num] = df[num].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return df

    # ================================================================== #
    #  4. ALIGN TIME SERIES
    # ================================================================== #
    def align_time_series(self, stock: pd.DataFrame, macro: pd.DataFrame,
                          how: str = "left") -> pd.DataFrame:
        """
        Align daily stock data with (lower-frequency) macro data.

        Macro is resampled to ``self.macro_freq`` then forward-filled onto the
        stock's daily timeline via a merge-as-of, so each trading day carries
        the most recently published macro figure.
        """
        stock = stock.copy().sort_index()
        macro = macro.copy().sort_index()

        # Resample macro to a clean monthly/monthly grid for stable join keys.
        macro = macro.resample(self.macro_freq).last()
        macro = macro.ffill().bfill()

        # merge_asof: every stock row picks up the latest macro value <= its date.
        stock_reset = stock.reset_index()
        macro_reset = macro.reset_index()
        aligned = pd.merge_asof(
            stock_reset.sort_values("date"),
            macro_reset.sort_values("date"),
            on="date",
            direction="backward",
        )
        aligned = aligned.set_index("date").sort_index()

        # Drop the warm-up rows that precede the first macro publication, so
        # no leading NaNs remain from indicators that simply didn't exist yet.
        first_macro = macro.index.min()
        aligned = aligned.loc[aligned.index >= first_macro]
        return aligned

    # ================================================================== #
    #  IO helpers
    # ================================================================== #
    def save(self, df: pd.DataFrame, path: str) -> None:
        import os
        os.makedirs(self.cache_dir, exist_ok=True)
        full = os.path.join(self.cache_dir, path) if "/" not in path else path
        if full.endswith(".parquet"):
            df.to_parquet(full)
        else:
            df.to_csv(full)
        print(f"[ok] saved {df.shape} -> {full}")


# ---------------------------------------------------------------------- #
#  Demo entry point
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    dp = DataPrep(start_date="2018-01-01", end_date="2024-12-31")

    print("→ Downloading A-share OHLCV …")
    stock = dp.get_stock_data("600519")          # 贵州茅台
    print(stock.tail())

    print("→ Downloading macro indicators …")
    macro = dp.get_macro_data()
    print(macro.tail())

    print("→ Cleaning …")
    stock_clean = dp.clean(stock)
    macro_clean = dp.clean(macro)

    print("→ Aligning time series …")
    panel = dp.align_time_series(stock_clean, macro_clean)
    print(panel[["close", "volume", "gdp_yoy", "cpi_yoy", "lpr_1y"]].tail())

    dp.save(panel, "aligned_panel.parquet")
