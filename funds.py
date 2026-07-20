"""
基金数据层 —— 给 LLM 提供基金问答所需的数据。

提供：
  * is_fund_query(text)              —— 检测 query 是否是基金问题
  * parse_fund_query(text)           —— 提取基金名 / 代码
  * FUND_NAME_TO_CODE                —— 常见基金名 → 代码字典（80+ 只）
  * fetch_fund_basic(code)           —— 基本信息（类型/规模/经理/成立日）
  * fetch_fund_nav(code, days)       —— 单位净值历史
  * fetch_fund_holdings(code)        —— 最新季度前 10 大持仓
  * fetch_fund_status(code)          —— 申赎状态（开放/限购/暂停）
  * compute_fund_metrics(code)       —— 业绩指标（夏普/回撤/年化）
  * build_fund_snapshot(text, cfg)   —— 给 LLM 的聚合字典

数据源：
  * 基金基本信息：akshare fund_individual_basic_info_xq（雪球）
  * 净值历史：akshare fund_open_fund_info_em（东方财富）
  * 季度持仓：akshare fund_portfolio_hold_em（东方财富）
  * 申赎状态：akshare fund_etf_fund_info_em（东方财富）
  * 业绩指标：基于 NAV 历史自计算（避免 fund_individual_achievement_xq 的 KeyError）
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from data_prep import _retry

log = logging.getLogger("funds")


# ====================================================================== #
#  1. 基金名 → 代码字典（常见 80+ 只）
# ====================================================================== #
FUND_NAME_TO_CODE: dict[str, str] = {
    # === 平安 ===
    "平安港股通": "021046",            # 平安港股通红利精选混合发起式 A（用户提到的）
    "平安策略先锋": "700003",
    "平安优势产业": "008949",
    "平安医疗健康": "003032",
    "平安价值成长": "005895",
    "平安新鑫先锋": "000739",

    # === 永赢 ===
    "永赢高端装备智选": "015789",      # 永赢高端装备智选混合发起 A（用户提到的）
    "永赢高端装备智选C": "015790",
    "永赢智能领先": "006266",
    "永赢消费主题": "006252",
    "永赢乾元": "015789",
    "永赢高端制造": "007113",
    "永赢科技智选": "011204",
    "永赢新能源智选": "015997",

    # === 招商 ===
    "招商中证白酒": "161725",
    "招商国证生物医药": "161726",
    "招商中证煤炭": "161032",
    "招商核心竞争力": "014412",
    "招商瑞文": "009423",
    "招商安泰平衡": "217002",

    # === 易方达 ===
    "易方达蓝筹精选": "005827",
    "易方达消费行业": "110022",
    "易方达中小盘": "110011",
    "易方达上证50": "110003",
    "易方达沪深300": "110020",
    "易方达消费精选": "011102",

    # === 华夏 ===
    "华夏成长": "000001",
    "华夏国证半导体芯片": "008887",
    "华夏新能源车": "011041",
    "华夏沪深300": "000051",
    "华夏复兴": "000031",
    "华夏大盘精选": "000011",

    # === 嘉实 ===
    "嘉实沪深300": "160706",
    "嘉实新兴产业": "000751",
    "嘉实增长": "070002",
    "嘉实优化红利": "070032",

    # === 兴全 ===
    "兴全合润": "163406",
    "兴全趋势": "163402",
    "兴全合宜": "163417",
    "兴全商业模式": "163415",

    # === 富国 ===
    "富国天惠": "161005",
    "富国消费主题": "519915",
    "富国天合稳健": "100026",
    "富国医疗保健": "161035",

    # === 汇添富 ===
    "汇添富中证主要消费": "000248",
    "汇添富医药保健": "470006",
    "汇添富优势精选": "519008",
    "汇添富均衡增长": "519018",
    "汇添富中证新能源车": "011037",

    # === 广发 ===
    "广发纳指": "270042",
    "广发双引擎": "162703",
    "广发稳健增长": "270002",
    "广发行业领先": "270025",

    # === 睿远 ===
    "睿远成长价值": "007119",
    "睿远均衡价值": "008969",
    "睿远稳进配置": "013351",

    # === 张坤 ===
    "张坤易方达蓝筹": "005827",
    "张坤易方达精选": "110009",

    # === 刘格菘 ===
    "刘格菘广发小盘": "162703",
    "刘格菘广发创新": "002939",

    # === 谢治宇 ===
    "谢治宇兴全合润": "163406",

    # === 行业 ETF（场内基金）===
    "半导体ETF": "159995",
    "芯片ETF": "159995",
    "医药ETF": "159929",
    "酒ETF": "512690",
    "消费ETF": "159928",
    "新能源车ETF": "515030",
    "光伏ETF": "515790",
    "军工ETF": "512660",
    "恒生ETF": "159920",
    "纳指ETF": "513100",
    "沪深300ETF": "510310",
    "中证500ETF": "510500",
    "科创50ETF": "588000",
    "创业板ETF": "159915",
    "红利ETF": "510880",
}


# 基金相关关键词（用于 is_fund_query 检测）
FUND_KEYWORDS = [
    "基金", "净值", "申赎", "申购", "赎回", "定投", "夏普",
    "最大回撤", "基金经理", "基金类型", "基金规模", "基金评级",
    "近1周", "近1月", "近3月", "近6月", "近1年", "近2年", "近3年",
    "今年来", "成立来", "同类", "排行", "份额",
    "持有期", "定投好", "一次性",
]

# 基金公司名字（出现即高概率是基金问题）
FUND_COMPANIES = [
    # 国内大型
    "易方达", "华夏", "招商", "嘉实", "兴全", "富国", "汇添富", "广发",
    "南方", "博时", "国泰", "银华", "华安", "景顺长城", "长城", "银河",
    "中欧", "天弘", "前海开源", "国投瑞银", "申万菱信", "光大保德信",
    "中海", "诺安", "民生加银", "华泰柏瑞", "国寿安保", "泰康",
    "东方", "工银", "农银", "中银", "交银", "建信",
    # 平安/永赢系列
    "平安", "永赢",
    # 知名外资
    "路博迈", "摩根", "贝莱德", "高盛", "先锋领航", "富兰克林",
    "联博", "安联", "施罗德", "瑞银", "汇丰", "德意志",
    # 顶流基金经理
    "张坤", "刘格菘", "谢治宇", "葛兰", "刘彦春", "朱少醒",
    "周蔚文", "傅鹏博", "董承非", "曹名长",
]


def is_fund_query(text: str) -> bool:
    """
    检测 query 是否是基金问题。

    判定规则（满足任一即可）：
    1. 含基金关键词
    2. 含已知基金名（FUND_NAME_TO_CODE）
    3. 含基金公司名（FUND_COMPANIES，如"路博迈"/"平安"等）
    4. 含基金代码前缀（15/16/50/51/52/58）
    5. 含通用基金词尾（混合/股票/债券/指数等 + A/C 类）
    """
    # 1. 关键词
    for kw in FUND_KEYWORDS:
        if kw in text:
            return True

    # 2. 已知基金名（按长度倒序，避免"永赢"先于"永赢高端装备智选"匹配）
    for name in sorted(FUND_NAME_TO_CODE.keys(), key=len, reverse=True):
        if name in text:
            return True

    # 3. 基金公司名
    for company in FUND_COMPANIES:
        if company in text:
            return True

    # 4. 基金代码特征
    fund_code_prefixes = ("15", "16", "50", "51", "52", "58")
    for m in re.finditer(r"\b(\d{6})\b", text):
        code = m.group(1)
        if code.startswith(fund_code_prefixes):
            return True

    # 5. 通用基金词尾（"xx混合"/"xx股票"/"xx债券"）
    #    即使 query 里只有"xx混合A"也能识别
    fund_suffixes = ("混合", "股票基金", "债券", "指数", "ETF", "联接",
                     "发起式", "发起", "LOF", "FOF", "QDII")
    for suffix in fund_suffixes:
        if suffix in text:
            return True

    return False


def parse_fund_query(text: str) -> dict[str, str]:
    """
    提取 query 中的基金名 / 代码。

    Returns
    -------
    {"name": "...", "code": "..."}  （name 可能为空）

    Examples
    --------
    >>> parse_fund_query("永赢高端装备智选怎么样")
    {'name': '永赢高端装备智选', 'code': '015789'}
    >>> parse_fund_query("015789 这只基金")
    {'name': '', 'code': '015789'}
    >>> parse_fund_query("路博迈港股通科技股")  # 不在硬编码字典，fuzzy match
    {'name': '路博迈港股通科技股票A', 'code': '026173'}
    """
    # 1. 已知基金名（按长度倒序优先匹配长名）
    for name in sorted(FUND_NAME_TO_CODE.keys(), key=len, reverse=True):
        if name in text:
            return {"name": name, "code": FUND_NAME_TO_CODE[name]}

    # 2. 6 位数字代码
    m = re.search(r"\b(\d{6})\b", text)
    if m:
        return {"name": "", "code": m.group(1)}

    # 3. Fuzzy match：到 akshare 全基金表（27000+）里找最长子串匹配
    fuzzy = fuzzy_match_fund(text)
    if fuzzy:
        return fuzzy

    return {"name": "", "code": ""}


# ====================================================================== #
#  Fuzzy match：akshare 全基金表（27000+）
# ====================================================================== #
_FUND_TABLE_CACHE: Any = None


def _load_fund_table() -> Any:
    """加载并缓存 akshare 全基金表（首次调用约 1-2s）。"""
    global _FUND_TABLE_CACHE
    if _FUND_TABLE_CACHE is None:
        import akshare as ak
        _FUND_TABLE_CACHE = ak.fund_name_em()
    return _FUND_TABLE_CACHE


def fuzzy_match_fund(text: str) -> dict[str, str] | None:
    """
    在 akshare 全基金表里模糊匹配 query。

    策略：
    1. 清洗 query（去掉标点 + 助词），避免 "C值得买吗？" 这种尾随问号破坏类后缀判断
    2. 枚举 query 中所有长度 >= 3 的子串，用 | 拼成 regex 一次性搜索
    3. 在匹配结果里，按"匹配子串长度 × 类后缀权重"排序
    4. 类后缀优先：A/B/C/D → query 含 "C" 优先匹配 C

    Examples
    --------
    >>> fuzzy_match_fund("路博迈港股通科技股")
    {'name': '路博迈港股通科技股票A', 'code': '026173'}
    >>> fuzzy_match_fund("平安新能源精选混合C")
    {'name': '平安新能源精选混合发起式C', 'code': '025924'}
    >>> fuzzy_match_fund("平安新能源精选混合C值得买吗？")
    {'name': '平安新能源精选混合发起式C', 'code': '025924'}
    """
    df = _load_fund_table()
    if df is None or df.empty:
        return None

    # 清洗 query：去掉中文/英文标点 + 句末语气词
    cleaned = re.sub(r"[，。？！；：、,.?!;:]+", "", text).strip()

    # 提取所有长度 >= 3 的子串
    L = len(cleaned)
    if L < 3:
        return None
    substrings = set()
    for length in range(min(15, L), 2, -1):
        for start in range(L - length + 1):
            s = cleaned[start:start + length]
            if s.isdigit():
                continue
            substrings.add(s)

    if not substrings:
        return None

    # 一次性 regex 搜索
    pattern = "|".join(re.escape(s) for s in substrings)
    try:
        matches = df[df["基金简称"].str.contains(pattern, regex=True, na=False)]
    except Exception:
        return None

    if matches.empty:
        return None

    # 检测 query 含哪个类后缀（A/B/C/D/E/F...）
    # 用清洗后的 query，避免"？"破坏判断
    # 用 regex 找单独的 A-F 字母（前后都是非字母），而不是看末尾字符
    class_letter = None
    m = re.search(r"[A-F](?![A-Za-z])", cleaned)
    if m:
        class_letter = m.group(0)
    else:
        # 也支持 "C类" 这种形式（虽然 regex 已覆盖）
        pass

    substr_list = sorted(substrings, key=len, reverse=True)

    def _match_score(row: dict) -> tuple[int, int, int, str]:
        """
        返回排序键: (-匹配长度, -类后缀, 简称长度, 简称)
        """
        name = row["基金简称"]
        match_len = 0
        for s in substr_list:
            if s in name:
                match_len = len(s)
                break
        class_bonus = 0
        if class_letter and name.endswith(class_letter):
            class_bonus = 1
        return (-match_len, -class_bonus, len(name), name)

    matches = matches.copy()
    matches["_score"] = matches.apply(_match_score, axis=1)
    matches = matches.sort_values(by="_score")

    row = matches.iloc[0]
    return {
        "name": str(row["基金简称"]),
        "code": str(row["基金代码"]),
    }


# ====================================================================== #
#  2. 基金基本信息
# ====================================================================== #
def fetch_fund_basic(code: str) -> dict[str, Any] | None:
    """
    基金基本信息（来源：雪球）。

    Returns dict with keys:
        code, name, full_name, type, manager, inception_date, scale, custodian,
        investment_strategy, benchmark, rating_agency, rating
    """
    import akshare as ak

    def _do():
        return ak.fund_individual_basic_info_xq(code)

    try:
        df = _retry(_do, retries=2, backoff=2.0)
    except Exception as exc:
        log.warning(f"基金基本信息拉取失败 {code}: {exc}")
        return None

    if df is None or df.empty:
        return None

    info = {row["item"]: row["value"] for _, row in df.iterrows() if row.get("item")}

    return {
        "code": code,
        "name": str(info.get("基金名称", "")),
        "full_name": str(info.get("基金全称", "")),
        "type": str(info.get("基金类型", "")),
        "manager": str(info.get("基金经理", "")),
        "custodian": str(info.get("托管银行", "")),
        "inception_date": str(info.get("成立时间", "")),
        "scale": str(info.get("最新规模", "")),
        "investment_strategy": str(info.get("投资策略", ""))[:500],
        "benchmark": str(info.get("业绩比较基准", "")),
        "rating": str(info.get("基金评级", "")),
    }


# ====================================================================== #
#  3. 单位净值历史
# ====================================================================== #
def fetch_fund_nav(code: str, days: int = 365) -> pd.DataFrame:
    """
    基金单位净值历史（数据源：东方财富 fund_open_fund_info_em）。

    默认拉近 1 年数据。返回 DataFrame: date, nav, daily_growth
    """
    import akshare as ak

    # period 选最接近 days 的
    if days <= 31:
        period = "1月"
    elif days <= 90:
        period = "3月"
    elif days <= 180:
        period = "6月"
    elif days <= 365:
        period = "1年"
    else:
        period = "成立来"

    def _do():
        return ak.fund_open_fund_info_em(
            symbol=code, indicator="单位净值走势", period=period,
        )

    try:
        df = _retry(_do, retries=2, backoff=2.0)
    except Exception as exc:
        log.warning(f"基金净值拉取失败 {code}: {exc}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # 列名：净值日期 / 单位净值 / 日增长率
    df = df.rename(columns={
        "净值日期": "date",
        "单位净值": "nav",
        "日增长率": "daily_growth",
    })
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df["daily_growth"] = pd.to_numeric(df["daily_growth"], errors="coerce")
    df = df.dropna(subset=["date", "nav"]).sort_values("date").reset_index(drop=True)
    return df


# ====================================================================== #
#  4. 申赎状态（来自 fund_etf_fund_info_em 的额外列）
# ====================================================================== #
def fetch_fund_status(code: str) -> dict[str, str] | None:
    """
    申赎状态：开放 / 限购 / 暂停

    数据源：fund_etf_fund_info_em（东方财富），最后一行的申购状态/赎回状态列。
    """
    import akshare as ak

    def _do():
        return ak.fund_etf_fund_info_em(fund=code, start_date="20250101", end_date="20260101")

    try:
        df = _retry(_do, retries=2, backoff=2.0)
    except Exception as exc:
        log.warning(f"基金申赎状态拉取失败 {code}: {exc}")
        return None

    if df is None or df.empty or "申购状态" not in df.columns:
        return None

    # 取最近一行（最新状态）
    latest = df.iloc[0]
    return {
        "purchase_status": str(latest.get("申购状态", "未知")),
        "redemption_status": str(latest.get("赎回状态", "未知")),
    }


# ====================================================================== #
#  5. 前 10 大持仓（最新季度）
# ====================================================================== #
def fetch_fund_holdings(code: str) -> list[dict[str, Any]]:
    """
    基金最新季度前 10 大持仓。

    数据源：fund_portfolio_hold_em（东方财富），返回所有季度的全部持仓，
    过滤前 10 大 + 最新季度。
    """
    import akshare as ak

    def _do():
        return ak.fund_portfolio_hold_em(symbol=code)

    try:
        df = _retry(_do, retries=2, backoff=2.0)
    except Exception as exc:
        log.warning(f"基金持仓拉取失败 {code}: {exc}")
        return []

    if df is None or df.empty or "季度" not in df.columns:
        return []

    # 取最新季度（季度列按字典序，新的在后面）
    quarters = df["季度"].dropna().unique()
    if len(quarters) == 0:
        return []
    latest_quarter = sorted(quarters)[-1]
    sub = df[df["季度"] == latest_quarter].copy()

    # 按占净值比例降序，取前 10
    sub = sub.sort_values("占净值比例", ascending=False).head(10)

    holdings = []
    for rank, (_, row) in enumerate(sub.iterrows(), start=1):
        holdings.append({
            "rank": rank,
            "symbol": str(row.get("股票代码", "")),
            "name": str(row.get("股票名称", "")),
            "pct_of_nav": float(row.get("占净值比例", 0) or 0),
            "market_value": float(row.get("持仓市值", 0) or 0),
        })
    return holdings


# ====================================================================== #
#  6. 业绩指标（基于 NAV 自计算）
# ====================================================================== #
@dataclass
class FundMetrics:
    """基金的业绩指标。"""
    latest_nav: float
    latest_date: str
    perf_1m: float = 0.0      # 近 1 月收益（%）
    perf_3m: float = 0.0      # 近 3 月收益（%）
    perf_6m: float = 0.0      # 近 6 月收益（%）
    perf_1y: float = 0.0      # 近 1 年收益（%）
    perf_ytd: float = 0.0     # 今年来收益（%）
    perf_since_inception: float = 0.0   # 成立来总收益（%）
    annualized: float = 0.0   # 年化收益（%）
    max_drawdown: float = 0.0  # 最大回撤（%，负数）
    sharpe: float = 0.0       # 夏普比率（简化版）
    volatility: float = 0.0   # 年化波动率（%）

    def to_dict(self) -> dict[str, Any]:
        return {
            "latest_nav": round(self.latest_nav, 4),
            "latest_date": self.latest_date,
            "perf_1m": round(self.perf_1m, 2),
            "perf_3m": round(self.perf_3m, 2),
            "perf_6m": round(self.perf_6m, 2),
            "perf_1y": round(self.perf_1y, 2),
            "perf_ytd": round(self.perf_ytd, 2),
            "perf_since_inception": round(self.perf_since_inception, 2),
            "annualized": round(self.annualized, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "sharpe": round(self.sharpe, 2),
            "volatility": round(self.volatility, 2),
        }


def compute_fund_metrics(code: str, days: int = 365) -> FundMetrics | None:
    """
    基于 NAV 历史计算业绩指标。

    计算方法：
    - 阶段收益：(nav_t / nav_t-k) - 1
    - 最大回撤：peak-to-trough 最大跌幅
    - 年化：(nav_t / nav_0) ^ (365 / 天数) - 1
    - 夏普：(年化 - 2.5%) / 年化波动率 （无风险利率近似 2.5%）
    - 波动率：日收益 std × √252
    """
    df = fetch_fund_nav(code, days=days)
    if df.empty or len(df) < 30:
        return None

    nav = df["nav"].astype(float)
    latest_nav = float(nav.iloc[-1])
    latest_date = str(df["date"].iloc[-1].date())

    def _perf(window: int) -> float:
        if len(nav) <= window:
            return 0.0
        return (float(nav.iloc[-1]) / float(nav.iloc[-window]) - 1) * 100

    perf_1m = _perf(min(22, len(nav) - 1))   # ~22 交易日 = 1 月
    perf_3m = _perf(min(65, len(nav) - 1))
    perf_6m = _perf(min(130, len(nav) - 1))
    perf_1y = _perf(min(245, len(nav) - 1))

    # YTD：从今年 1 月 1 日起
    import datetime as _dt
    this_year_start = _dt.date(df["date"].iloc[-1].year, 1, 1)
    ytd_navs = df[df["date"].dt.date >= this_year_start]["nav"]
    perf_ytd = ((latest_nav / float(ytd_navs.iloc[0]) - 1) * 100) if len(ytd_navs) >= 2 else 0.0

    # 成立来
    perf_since = (latest_nav / float(nav.iloc[0]) - 1) * 100
    total_days = (df["date"].iloc[-1] - df["date"].iloc[0]).days
    annualized = ((latest_nav / float(nav.iloc[0])) ** (365 / max(total_days, 1)) - 1) * 100

    # 最大回撤
    cum_max = nav.cummax()
    drawdown = (nav - cum_max) / cum_max * 100
    max_dd = float(drawdown.min())

    # 波动率 + 夏普
    daily_returns = nav.pct_change().dropna()
    if len(daily_returns) >= 20:
        vol = float(daily_returns.std()) * np.sqrt(252) * 100
        # 简化夏普：(年化收益 - 2.5%) / 年化波动率
        sharpe = (annualized - 2.5) / vol if vol > 0 else 0.0
    else:
        vol = 0.0
        sharpe = 0.0

    return FundMetrics(
        latest_nav=latest_nav,
        latest_date=latest_date,
        perf_1m=perf_1m,
        perf_3m=perf_3m,
        perf_6m=perf_6m,
        perf_1y=perf_1y,
        perf_ytd=perf_ytd,
        perf_since_inception=perf_since,
        annualized=annualized,
        max_drawdown=max_dd,
        sharpe=sharpe,
        volatility=vol,
    )


# ====================================================================== #
#  7. 一站式：构建 LLM 用的基金快照
# ====================================================================== #
def build_fund_snapshot(text: str, cfg: dict | None = None) -> dict[str, Any] | None:
    """
    构建 LLM 用的基金快照。

    Parameters
    ----------
    text : 用户 query
    cfg : 用户配置（可选，用于 portfolio / risk_profile）

    Returns
    -------
    {
        "as_of": str,
        "query": str,
        "fund": {basic info},
        "metrics": {performance},
        "holdings": [list of top 10],
        "status": {申购/赎回},
        "concentration_top10_pct": float,
    }
    """
    parsed = parse_fund_query(text)
    code = parsed["code"]
    name = parsed["name"]
    if not code:
        return None

    import datetime as _dt
    snap: dict[str, Any] = {
        "as_of": _dt.datetime.now().isoformat(timespec="minutes"),
        "query": text,
        "parsed_name": name,
        "parsed_code": code,
        "fund": None,
        "metrics": None,
        "holdings": [],
        "status": None,
        "concentration_top10_pct": 0.0,
    }

    basic = fetch_fund_basic(code)
    if basic:
        snap["fund"] = basic
        # 优先用数据库的正式名覆盖用户输入的简称
        if not name and basic.get("name"):
            snap["parsed_name"] = basic["name"]

    metrics = compute_fund_metrics(code, days=365)
    if metrics:
        snap["metrics"] = metrics.to_dict()

    holdings = fetch_fund_holdings(code)
    if holdings:
        snap["holdings"] = holdings
        snap["concentration_top10_pct"] = round(
            sum(h["pct_of_nav"] for h in holdings), 2,
        )

    status = fetch_fund_status(code)
    if status:
        snap["status"] = status

    return snap


# ====================================================================== #
#  Self-test
# ====================================================================== #
if __name__ == "__main__":
    import sys, json
    sys.path.insert(0, ".")

    for query in ["永赢高端装备智选怎么样", "平安港股通能买吗", "021046 这只基金"]:
        print(f"\n========== query: {query!r} ==========")
        print(f"is_fund_query: {is_fund_query(query)}")
        snap = build_fund_snapshot(query)
        if snap:
            print(f"fund: {snap['fund']['name'] if snap.get('fund') else 'NO BASIC'}")
            print(f"  code: {snap['parsed_code']}")
            if snap.get('metrics'):
                print(f"  latest NAV: {snap['metrics']['latest_nav']}")
                print(f"  perf 1m/3m/1y: {snap['metrics']['perf_1m']:.2f}/{snap['metrics']['perf_3m']:.2f}/{snap['metrics']['perf_1y']:.2f}%")
                print(f"  annualized: {snap['metrics']['annualized']:.2f}%, max DD: {snap['metrics']['max_drawdown']:.2f}%")
                print(f"  sharpe: {snap['metrics']['sharpe']:.2f}, volatility: {snap['metrics']['volatility']:.2f}%")
            if snap.get('holdings'):
                print(f"  top 3 holdings:")
                for h in snap['holdings'][:3]:
                    print(f"    {h['rank']}. {h['name']}({h['symbol']}) {h['pct_of_nav']:.2f}%")
            if snap.get('status'):
                print(f"  status: {snap['status']}")
        else:
            print("No snapshot (fund not detected)")