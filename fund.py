"""
基金模块 —— 排行 + 详情 + 推荐筛选。

数据源：akshare 的 `fund_open_fund_rank_em`（东方财富接口，**慢但能用**）。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from data_prep import _retry

log = logging.getLogger("fund")


# ====================================================================== #
#  数据源
# ====================================================================== #
def fetch_fund_rank(category: str = "股票型", indicator: str = "近1年",
                    top_n: int = 30) -> pd.DataFrame:
    """
    拉取某类基金的排行。

    category: 股票型 / 混合型 / 债券型 / 指数型 / QDII / FOF
    indicator: 近1周 / 近1月 / 近3月 / 近6月 / 近1年 / 近2年 / 近3年 / 今年来 / 成立来
    top_n: 返回前 N 条

    返回字段：基金代码、基金简称、单位净值、日增长率、近1周/1月/.../今年来/成立来
    """
    valid_categories = {"股票型", "混合型", "债券型", "指数型", "QDII", "FOF"}
    if category not in valid_categories:
        raise ValueError(f"category 必须是 {valid_categories} 之一")
    if top_n < 1 or top_n > 200:
        raise ValueError("top_n 范围 1-200")

    import akshare as ak

    def _do():
        return ak.fund_open_fund_rank_em(symbol=category)

    df = _retry(_do)

    # 解析 indicator 对应的列
    indicator_col_map = {
        "近1周": "近1周", "近1月": "近1月", "近3月": "近3月",
        "近6月": "近6月", "近1年": "近1年", "近2年": "近2年",
        "近3年": "近3年", "今年来": "今年来", "成立来": "成立来",
    }
    if indicator not in indicator_col_map:
        raise ValueError(f"indicator 必须是 {list(indicator_col_map)} 之一")
    sort_col = indicator_col_map[indicator]
    if sort_col not in df.columns:
        raise RuntimeError(f"列 {sort_col} 不在返回结果中")

    # 排序 + 取 top
    df[sort_col] = pd.to_numeric(df[sort_col], errors="coerce")
    df = df.sort_values(sort_col, ascending=False, na_position="last").head(top_n)

    # 格式化
    rename = {
        "基金代码": "code", "基金简称": "name", "日期": "date",
        "单位净值": "nav", "累计净值": "cum_nav", "日增长率": "daily_growth",
        "近1周": "week_1", "近1月": "month_1", "近3月": "month_3",
        "近6月": "month_6", "近1年": "year_1", "近2年": "year_2",
        "近3年": "year_3", "今年来": "ytd", "成立来": "since_inception",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    return df.reset_index(drop=True)


# ====================================================================== #
#  筛选器（基于规则）
# ====================================================================== #
@dataclass
class FundFilter:
    """基金筛选器"""
    min_year_1: float | None = None        # 最低 1 年收益（%）
    min_year_3: float | None = None        # 最低 3 年收益（%）
    max_drawdown: float | None = None      # 最高回撤（%，负数表示最大亏损）
    category: str | None = None            # 基金类型
    top_n: int = 10

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.min_year_1 is not None and "year_1" in out.columns:
            out = out[pd.to_numeric(out["year_1"], errors="coerce") >= self.min_year_1]
        if self.min_year_3 is not None and "year_3" in out.columns:
            out = out[pd.to_numeric(out["year_3"], errors="coerce") >= self.min_year_3]
        if self.category:
            out = out[out.get("category") == self.category] if "category" in out.columns else out
        return out.head(self.top_n)


# ====================================================================== #
#  文本格式化（给 LLM / 用户看）
# ====================================================================== #
def format_fund_table(df: pd.DataFrame, sort_by: str = "year_1",
                     max_rows: int = 10) -> str:
    """把基金排行 DataFrame 格式化成 markdown 文本。"""
    if df.empty:
        return "📭 暂无符合条件的基金"

    keep = ["code", "name", "year_1", "year_3", "ytd"]
    keep = [c for c in keep if c in df.columns]
    sub = df[keep].head(max_rows).copy()

    # 格式化百分比
    for col in ("year_1", "year_3", "ytd"):
        if col in sub.columns:
            sub[col] = sub[col].apply(
                lambda x: f"{x:.1f}%" if pd.notna(x) else "-"
            )

    # 重命名列头
    headers = {
        "code": "代码", "name": "名称",
        "year_1": "近1年", "year_3": "近3年", "ytd": "今年来",
    }
    sub = sub.rename(columns=headers)

    # markdown 表格
    lines = ["| " + " | ".join(sub.columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(sub.columns)) + " |")
    for _, row in sub.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")

    return "\n".join(lines)


# ====================================================================== #
#  推荐（综合打分）
# ====================================================================== #
def recommend_funds(category: str = "股票型", indicator: str = "近1年",
                   top_n: int = 10) -> str:
    """
    拉取某类基金按指定周期排行，返回 top_n 的 markdown 表格。
    """
    try:
        df = fetch_fund_rank(category=category, indicator=indicator, top_n=top_n)
    except Exception as exc:
        log.exception("fund rank failed")
        return f"❌ 拉取基金排行失败：{exc}"

    title = f"📊 **{category}基金 {indicator} 排行 TOP {top_n}**\n"
    return title + format_fund_table(df)


def parse_fund_query(text: str) -> dict:
    """解析用户的基金 query，提取类型和时间窗口。

    例：
        "股票型基金近1年排行" -> {category: "股票型", indicator: "近1年"}
        "近3年混合型基金前10"  -> {category: "混合型", indicator: "近3年"}
        "基金排行"             -> {category: "股票型", indicator: "近1年"}（默认）
    """
    valid_categories = ["股票型", "混合型", "债券型", "指数型", "QDII", "FOF"]
    valid_indicators = ["近1周", "近1月", "近3月", "近6月", "近1年", "近2年", "近3年", "今年来"]

    category = next((c for c in valid_categories if c in text), "股票型")
    indicator = next((i for i in valid_indicators if i in text), "近1年")

    # 提取 top_n（"前10"、"top 5"、"前20"）
    top_n = 10
    m = re.search(r"(?:前|top)\s*(\d+)", text, re.IGNORECASE)
    if m:
        top_n = min(int(m.group(1)), 30)

    return {"category": category, "indicator": indicator, "top_n": top_n}


# ---------------------------------------------------------------------- #
#  Self-test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=== 股票型基金近 1 年排行 ===")
    print(recommend_funds("股票型", "近1年", 5))
    print()
    print("=== 指数型基金今年来 ===")
    print(recommend_funds("指数型", "今年来", 5))