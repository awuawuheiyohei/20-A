"""
技术面 + 资金面 数据层（institutional-grade 分析数据）。

提供：
  * fetch_daily_k(symbol, n=120)        —— 日 K 线历史（DataFrame）
  * calc_ma(close, periods)             —— 简单移动平均
  * calc_macd(close)                    —— MACD (12,26,9)
  * calc_rsi(close, period=14)          —— RSI
  * calc_boll(close, n=20, k=2)         —— 布林带
  * compute_indicators(symbol)          —— 一站式：拉 K + 算所有指标
  * fetch_capital_flow(symbol)          —— 个股资金流（Eastmoney HTTP）
  * fetch_capital_flow_rank(top_n=20)   —— 全市场主力净流入榜

数据源策略：
  * 日 K 线：Tencent `fqkline`（稳定，60-2000 行）
  * 资金流：Eastmoney `push2.eastmoney.com/api/qt/stock/get` HTTP 直调（绕开 akshare 反爬）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import requests

from data_prep import _retry
from intraday import _tencent_session, _tencent_symbol

log = logging.getLogger("indicators")


# ====================================================================== #
#  1. 日 K 线（Tencent fqkline）
# ====================================================================== #
def fetch_daily_k(symbol: str, n: int = 120,
                  fq: str = "qfq") -> pd.DataFrame:
    """
    拉个股日 K 线（默认前复权 qfq）。

    Tencent fqkline 返回列：[date, open, close, high, low, volume, amount, ...]

    Returns
    -------
    DataFrame: date (index), open, close, high, low, volume, amount
    """
    t_sym = _tencent_symbol(symbol)
    sess = _tencent_session()

    def _do():
        r = sess.get(
            "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
            params={"param": f"{t_sym},day,,,{n},{fq}"},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()

    data = _retry(_do, retries=2, backoff=2.0)
    raw_key = "qfqday" if fq == "qfq" else ("hfqday" if fq == "hfq" else "day")
    raw = (
        data.get("data", {})
        .get(t_sym, {})
        .get(raw_key, [])
    )
    if not raw:
        return pd.DataFrame()

    rows = []
    for line in raw:
        if len(line) < 6:
            continue
        try:
            rows.append({
                "date": pd.Timestamp(line[0]),
                "open": float(line[1]),
                "close": float(line[2]),
                "high": float(line[3]),
                "low": float(line[4]),
                "volume": float(line[5]),  # 单位：手 (1 手 = 100 股)
            })
        except (ValueError, IndexError):
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("date").sort_index()
    # 添加 amount（估算 = (high+low+close)/3 * volume * 100）— 如果 fqkline 没给
    if "amount" not in df.columns:
        df["amount"] = (df["high"] + df["low"] + df["close"]) / 3 * df["volume"] * 100
    return df


# ====================================================================== #
#  2. 技术指标
# ====================================================================== #
def calc_ma(close: pd.Series, periods: list[int] | None = None) -> pd.DataFrame:
    """
    简单移动平均线。返回 DataFrame，每列一个周期。

    默认周期：[5, 10, 20, 60]
    """
    if periods is None:
        periods = [5, 10, 20, 60]
    return pd.DataFrame({f"MA{p}": close.rolling(p).mean() for p in periods})


def calc_macd(close: pd.Series, fast: int = 12, slow: int = 26,
              signal: int = 9) -> pd.DataFrame:
    """
    MACD 指标。

    Returns
    -------
    DataFrame with columns: DIF, DEA, MACD (柱)
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd = (dif - dea) * 2
    return pd.DataFrame({"DIF": dif, "DEA": dea, "MACD": macd})


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI 指标（相对强弱指数）。"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)  # 初始无数据时给中性 50


def calc_boll(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    """
    布林带 (Bollinger Bands)。

    Returns
    -------
    DataFrame: MID (中轨=MA20), UPPER (上轨), LOWER (下轨)
    """
    mid = close.rolling(n).mean()
    std = close.rolling(n).std()
    return pd.DataFrame({
        "BOLL_MID": mid,
        "BOLL_UPPER": mid + k * std,
        "BOLL_LOWER": mid - k * std,
    })


# ====================================================================== #
#  3. 一站式：拉 K + 算指标
# ====================================================================== #
@dataclass
class Indicators:
    """某只股票的完整技术指标快照（最近一日）。"""
    symbol: str
    last: float                # 最新收盘价
    prev_close: float          # 昨日收盘
    pct_change: float          # 涨跌幅（%）

    # MA
    ma5: float | None = None
    ma10: float | None = None
    ma20: float | None = None
    ma60: float | None = None
    ma_status: str = ""        # 多头排列 / 空头排列 / 缠绕

    # MACD
    dif: float | None = None
    dea: float | None = None
    macd_bar: float | None = None
    macd_status: str = ""      # 金叉 / 死叉 / 多头 / 空头

    # RSI
    rsi14: float | None = None
    rsi_status: str = ""       # 超买 / 超卖 / 中性

    # 布林带
    boll_upper: float | None = None
    boll_mid: float | None = None
    boll_lower: float | None = None
    boll_status: str = ""      # 突破上轨 / 跌破下轨 / 中轨附近

    # 趋势
    trend: str = ""            # 上升 / 下降 / 震荡 / 拐点

    # K 线形态（最近 5 日）
    recent_pattern: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "last": self.last,
            "prev_close": self.prev_close,
            "pct_change": self.pct_change,
            "ma5": self.ma5, "ma10": self.ma10, "ma20": self.ma20, "ma60": self.ma60,
            "ma_status": self.ma_status,
            "dif": self.dif, "dea": self.dea, "macd_bar": self.macd_bar,
            "macd_status": self.macd_status,
            "rsi14": self.rsi14, "rsi_status": self.rsi_status,
            "boll_upper": self.boll_upper, "boll_mid": self.boll_mid,
            "boll_lower": self.boll_lower, "boll_status": self.boll_status,
            "trend": self.trend,
            "recent_pattern": self.recent_pattern,
        }


def _round(x: float | None, ndigits: int = 2) -> float | None:
    return round(x, ndigits) if x is not None and not pd.isna(x) else None


def compute_indicators(symbol: str, k_n: int = 120) -> Indicators | None:
    """
    拉日 K + 计算所有技术指标，返回 Indicators 快照（最新一日）。
    """
    df = fetch_daily_k(symbol, n=k_n)
    if df.empty or len(df) < 20:
        log.warning(f"{symbol} K线数据不足 ({len(df)} 行)")
        return None

    close = df["close"]
    ma = calc_ma(close)
    macd = calc_macd(close)
    rsi = calc_rsi(close)
    boll = calc_boll(close)

    last_close = close.iloc[-1]
    prev_close = close.iloc[-2]
    pct_change = (last_close - prev_close) / prev_close * 100

    ind = Indicators(
        symbol=symbol,
        last=_round(last_close),
        prev_close=_round(prev_close),
        pct_change=_round(pct_change),
        ma5=_round(ma["MA5"].iloc[-1]),
        ma10=_round(ma["MA10"].iloc[-1]),
        ma20=_round(ma["MA20"].iloc[-1]),
        ma60=_round(ma["MA60"].iloc[-1]),
        dif=_round(macd["DIF"].iloc[-1], 4),
        dea=_round(macd["DEA"].iloc[-1], 4),
        macd_bar=_round(macd["MACD"].iloc[-1], 4),
        rsi14=_round(rsi.iloc[-1]),
        boll_upper=_round(boll["BOLL_UPPER"].iloc[-1]),
        boll_mid=_round(boll["BOLL_MID"].iloc[-1]),
        boll_lower=_round(boll["BOLL_LOWER"].iloc[-1]),
    )

    # === MA 状态 ===
    if ind.ma5 and ind.ma10 and ind.ma20 and ind.ma60:
        if ind.ma5 > ind.ma10 > ind.ma20 > ind.ma60:
            ind.ma_status = "多头排列（强势）"
        elif ind.ma5 < ind.ma10 < ind.ma20 < ind.ma60:
            ind.ma_status = "空头排列（弱势）"
        else:
            ind.ma_status = "缠绕（震荡）"

    # === MACD 状态 ===
    if ind.dif is not None and ind.dea is not None:
        prev_dif = macd["DIF"].iloc[-2]
        prev_dea = macd["DEA"].iloc[-2]
        if prev_dif <= prev_dea and ind.dif > ind.dea:
            ind.macd_status = "金叉（买入信号）"
        elif prev_dif >= prev_dea and ind.dif < ind.dea:
            ind.macd_status = "死叉（卖出信号）"
        elif ind.dif > ind.dea:
            ind.macd_status = "多头延续"
        else:
            ind.macd_status = "空头延续"

    # === RSI 状态 ===
    if ind.rsi14 is not None:
        if ind.rsi14 >= 70:
            ind.rsi_status = "超买（>70，警惕回调）"
        elif ind.rsi14 <= 30:
            ind.rsi_status = "超卖（<30，可能反弹）"
        elif ind.rsi14 >= 60:
            ind.rsi_status = "偏强"
        elif ind.rsi14 <= 40:
            ind.rsi_status = "偏弱"
        else:
            ind.rsi_status = "中性"

    # === 布林带状态 ===
    if ind.boll_upper and ind.boll_lower and ind.boll_mid:
        if last_close >= ind.boll_upper:
            ind.boll_status = "突破上轨（超买）"
        elif last_close <= ind.boll_lower:
            ind.boll_status = "跌破下轨（超卖）"
        elif last_close > ind.boll_mid:
            ind.boll_status = "中轨上方（偏强）"
        else:
            ind.boll_status = "中轨下方（偏弱）"

    # === 整体趋势 ===
    if ind.ma5 and ind.ma20:
        if ind.ma5 > ind.ma20 and pct_change > 0:
            ind.trend = "上升趋势"
        elif ind.ma5 < ind.ma20 and pct_change < 0:
            ind.trend = "下降趋势"
        else:
            ind.trend = "震荡/拐点"

    # === 最近 5 日形态 ===
    if len(df) >= 5:
        recent = df["close"].iloc[-5:].tolist()
        opens = df["open"].iloc[-5:].tolist()
        # 简单描述
        ups = sum(1 for i in range(1, 5) if recent[i] > recent[i-1])
        if ups >= 4:
            ind.recent_pattern = f"近5日{ups}阳线（强势反弹）"
        elif ups <= 1:
            ind.recent_pattern = f"近5日{5-ups}阴线（弱势回调）"
        else:
            ind.recent_pattern = f"近5日{ups}阳{5-ups}阴（震荡整理）"

    return ind


# ====================================================================== #
#  4. 资金流（Eastmoney HTTP 直调，绕开 akshare 反爬）
# ====================================================================== #
# Eastmoney push2 接口字段映射（实测）：
#   f43 = 最新价
#   f44 = 最高
#   f45 = 最低
#   f46 = 今开
#   f47 = 成交量(手)
#   f48 = 成交额(元)
#   f60 = 昨收
#   f84 = 特大单净额(元)    100万以上
#   f85 = 大单净额(元)      20-100万
#   f167 = 主力净占比(%)     (特大+大单)净占比
#   f168 = 特大单净占比(%)
#   f169 = 大单净占比(%)
#   f170 = 主力净流入率(%)
_FUND_FLOW_FIELDS = (
    "f43,f44,f45,f46,f47,f48,f60,"   # 价格
    "f84,f85,"                        # 资金流(净额)
    "f167,f168,f169,f170"             # 资金流(占比)
)


def fetch_capital_flow(symbol: str) -> dict[str, Any] | None:
    """
    拉个股资金流（实时）。

    数据源：Eastmoney push2 接口（HTTP 直调，绕开 akshare 反爬）。

    Returns
    -------
    {
        "symbol": str,
        "last": float,                  # 最新价
        "prev_close": float,            # 昨收
        "pct_change": float,            # 涨跌幅(%)
        "high": float, "low": float, "open": float,
        "volume_lots": float,           # 成交量(手)
        "amount_cny": float,            # 成交额(元)
        "super_large_net": float,       # 特大单净额(元)
        "large_net": float,             # 大单净额(元)
        "main_net": float,              # 主力净额 = 特大单 + 大单
        "main_pct": float,              # 主力净占比(%)
        "super_large_pct": float,       # 特大单净占比(%)
        "large_pct": float,             # 大单净占比(%)
    }
    """
    market = "0" if symbol.startswith(("00", "30", "20")) else "1"
    secid = f"{market}.{symbol}"

    sess = _tencent_session()
    sess.headers["Referer"] = "https://quote.eastmoney.com/"

    def _do():
        r = sess.get(
            "https://push2.eastmoney.com/api/qt/stock/get",
            params={
                "secid": secid,
                "fields": _FUND_FLOW_FIELDS,
                "invt": 2,
                "fltt": 2,
            },
            timeout=10,
        )
        r.raise_for_status()
        return r.json()

    try:
        data = _retry(_do, retries=2, backoff=2.0)
    except Exception as exc:
        log.warning(f"资金流拉取失败 {symbol}: {exc}")
        return None

    d = data.get("data")
    if not d or d.get("f43") is None:
        return None

    super_large = d.get("f84") or 0.0
    large = d.get("f85") or 0.0

    return {
        "symbol": symbol,
        "last": d.get("f43") or 0.0,
        "prev_close": d.get("f60") or 0.0,
        "pct_change": d.get("f170") or 0.0,        # 涨跌幅(%)直接是数值
        "high": d.get("f44") or 0.0,
        "low": d.get("f45") or 0.0,
        "open": d.get("f46") or 0.0,
        "volume_lots": d.get("f47") or 0.0,
        "amount_cny": d.get("f48") or 0.0,
        "super_large_net": super_large,
        "large_net": large,
        "main_net": super_large + large,
        "main_pct": d.get("f167") or 0.0,
        "super_large_pct": d.get("f168") or 0.0,
        "large_pct": d.get("f169") or 0.0,
    }


# ====================================================================== #
#  Self-test
# ====================================================================== #
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    sym = "002409"
    print(f"=== {sym} 技术指标 ===")
    ind = compute_indicators(sym)
    if ind:
        for k, v in ind.to_dict().items():
            print(f"  {k}: {v}")

    print()
    print(f"=== {sym} 资金流 ===")
    flow = fetch_capital_flow(sym)
    if flow:
        for k, v in flow.items():
            print(f"  {k}: {v}")
    else:
        print("资金流拉取失败")