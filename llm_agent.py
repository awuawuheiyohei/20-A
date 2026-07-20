"""
LLM 推理层 —— 调用 DeepSeek，组装 prompt，生成结构化建议。

输入：
  * 11:30 / 14:30 触发时的市场快照（来自 intraday.py）
  * 用户持仓 + 关注股（来自 config.yaml）
  * 用户场景/问题（interactive.py 触发时传入）
  * 现有 ML pipeline 的预测信号（可选）

输出：
  * 结构化的中文建议文本（直接发到企业微信）
  * JSON 形式的结构化字段（signal, confidence, reasoning, risks）
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from openai import OpenAI

log = logging.getLogger("llm_agent")

# ====================================================================== #
#  模型配置
# ====================================================================== #
# DeepSeek V4 模型表 + V3/R1 已弃用（2026/07/24 EOL）。
# V4-flash / V4-pro 都支持"思考/非思考"两种模式。
MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "deepseek-v4-flash": {
        "temperature": 0.4,
        "max_tokens": 2000,
        "thinking": False,
        "reasoning_effort": "medium",   # low/medium/high
        "description": "V4-flash — 快速、低成本，支持思考/非思考模式",
        "deprecated": False,
    },
    "deepseek-v4-pro": {
        "temperature": 0.4,
        "max_tokens": 4000,
        "thinking": False,
        "reasoning_effort": "high",
        "description": "V4-pro — 高质量，支持思考/非思考模式",
        "deprecated": False,
    },
    # ===== 已弃用：2026/07/24 EOL，V3 = V4-flash 非思考模式；R1 = V4-flash 思考模式 =====
    "deepseek-chat": {
        "temperature": 0.4,
        "max_tokens": 1500,
        "thinking": False,
        "description": "V3 — 已弃用，请改用 deepseek-v4-flash",
        "deprecated": True,
    },
    "deepseek-reasoner": {
        "temperature": 0.6,
        "max_tokens": 4000,
        "thinking": True,
        "description": "R1 — 已弃用，请改用 deepseek-v4-flash (思考模式)",
        "deprecated": True,
    },
}

DEFAULT_MODEL = "deepseek-v4-flash"

# 别名 → (model, thinking) 映射
MODEL_ALIASES: dict[str, tuple[str, bool]] = {
    # === V4 完整名（直接指定）===
    "deepseek-v4-flash": ("deepseek-v4-flash", False),
    "deepseek-v4-pro": ("deepseek-v4-pro", False),
    # === V4 + 思考模式（-thinking 后缀）===
    "deepseek-v4-flash-thinking": ("deepseek-v4-flash", True),
    "deepseek-v4-pro-thinking": ("deepseek-v4-pro", True),
    # === 简短别名 ===
    "v4-flash": ("deepseek-v4-flash", False),
    "v4-pro": ("deepseek-v4-pro", False),
    "v4": ("deepseek-v4-flash", False),
    "v4-thinking": ("deepseek-v4-flash", True),
    "v4-flash-thinking": ("deepseek-v4-flash", True),
    "v4-pro-thinking": ("deepseek-v4-pro", True),
    # === 旧名兼容（v3/r1 仍常用口语）===
    "v3": ("deepseek-v4-flash", False),
    "chat": ("deepseek-v4-flash", False),
    "r1": ("deepseek-v4-flash", True),
    "reasoner": ("deepseek-v4-flash", True),
    "reasoning": ("deepseek-v4-flash", True),
    # === 旧模型名（保留兼容，已弃用）===
    "deepseek-chat": ("deepseek-v4-flash", False),
    "deepseek-reasoner": ("deepseek-v4-flash", True),
}


@dataclass
class ResolvedModel:
    """`resolve_model` 的返回类型。"""
    name: str              # 真实 API 模型名（用于 chat.completions.create）
    thinking: bool         # 是否启用思考模式
    deprecated: bool       # 是否已弃用


def resolve_model(name: str | None) -> ResolvedModel:
    """
    规范化模型名（含别名 + 思考模式）。

    Returns
    -------
    ResolvedModel(name, thinking, deprecated)

    Examples
    --------
    >>> resolve_model(None)                    # 默认
    ResolvedModel('deepseek-v4-flash', False, False)
    >>> resolve_model('v4-pro')                # V4-pro 非思考
    ResolvedModel('deepseek-v4-pro', False, False)
    >>> resolve_model('r1')                    # R1 别名 → V4-flash 思考
    ResolvedModel('deepseek-v4-flash', True, False)
    >>> resolve_model('deepseek-chat')         # 旧名，标记 deprecated
    ResolvedModel('deepseek-v4-flash', False, True)
    """
    if not name:
        name = DEFAULT_MODEL
    name_lower = name.strip().lower()
    if name_lower in MODEL_ALIASES:
        model, thinking = MODEL_ALIASES[name_lower]
        # deprecated 标志：用原名查（不是 alias 目标）
        # 例：deepseek-chat → V4-flash，但 deprecated=True 应来自 deepseek-chat
        deprecated = MODEL_DEFAULTS.get(name_lower, {}).get("deprecated", False)
        return ResolvedModel(model, thinking, deprecated)
    log.warning(f"未知模型 {name!r}，退回 {DEFAULT_MODEL}")
    return ResolvedModel(DEFAULT_MODEL, False, False)


SYSTEM_PROMPT_BASE = """你是用户的 A 股投资决策辅助助手。

【用户投资风格】
- 保守型，不追高
- 偏好"回调到位 + 关键支撑 + 缩量" 的低吸机会
- 偏好"短线 T"（日内或 1-3 天的小波段）
- 严格止损
- 资金利用率优先：宁可空仓也不乱开仓

【硬性约束 — 违反任意一条就是错的】
1. **绝对禁止编造数据**。如果某只股票在"当前市场数据"里没有给出具体数字（现价、涨跌幅等），你必须说"暂无该股实时数据"，并引导用户提供或自行查询；**不准用"假设"、"约"、"大约"等词编造具体数字**。
2. **不准编造技术位**（支撑位、压力位、前低、前高、年线等）——除非"用户提及的个股"或"板块数据"里有明确数值，否则不写。
3. 任何关于个股的判断，**必须基于"当前市场数据"里给出的具体数字**。没有数据 → 不写判断。
4. 所有结论必须以"非投资建议"提示结尾。
5. 优先考虑用户的"持仓成本、当前仓位、风险偏好"，不要推荐超出其风险承受度的操作。
6. **不要追涨**：刚涨停、放大量拉升、涨幅 > 7% 的股不主动推荐。
7. **宁缺毋滥**：个股推荐不超过 2-3 只，没把握就空仓。
8. 涉及个股时必须说明：方向、理由、具体入场位、止损位、持有周期。
9. 用简洁、结构化的中文，长度控制在用户能在 30 秒内读完。
"""


PROMPT_TEMPLATES = {
    "morning_close": """\
现在是 {date} 上午收盘（11:30）。

【任务】基于下方数据，给出"上午复盘 + 下午预判 + 适时空仓机会"。

【用户当前状态】
- 总仓位：{position_pct}%  |  现金：{cash_pct}%
- 持仓：{holdings_summary}
- 关注股：{watchlist_summary}
- 风险偏好：单只最大 {max_single}%，止损 {stop_loss}%，止盈 {take_profit}%

【上午市场快照】
{snapshot_json}

【输出要求】
按下面格式输出（每节简短）：

📊 **上午复盘**
- 三大指数表现 + 成交量（缩量/放量）
- 资金情绪：强势 / 观望 / 谨慎
- 领涨/领跌板块

🔮 **下午预判**
- 倾向方向（强势 / 震荡 / 转弱）
- 关键观察点（支撑/压力位）

🎯 **持仓建议**（对每只持仓股给一句话）
- [股票名]：[继续持有/做T减仓/止损/观望] | 理由 + 操作要点

💡 **下午关注**（以下只在你认为**真正有把握**时给，最多 2 只）
- **回调中的低吸机会**（不是追涨）：
  - 形态：缩量回踩关键支撑 + 不破前低
  - 入场位 + 止损位 + 持有周期（1-3 天）
  - 没说就别写这节
- **避免追高**（点名今天涨幅 > 7% 但你已经想买的）
  - 如果用户关注的股在涨停潮里，明确告诉他"别追"

⚠️ 风险提示：以上内容不构成投资建议。

""",

    "afternoon_close": """\
现在是 {date} 尾盘（14:30）。

【任务】基于下方数据，给出"今日复盘 + 尾盘决策 + 次日低吸布局"。

【用户当前状态】
- 总仓位：{position_pct}%  |  现金：{cash_pct}%
- 持仓：{holdings_summary}
- 关注股：{watchlist_summary}
- 风险偏好：单只最大 {max_single}%，止损 {stop_loss}%，止盈 {take_profit}%

【今日市场快照】
{snapshot_json}

【输出要求】
按下面格式输出：

🌅 **今日总结**
- 指数方向 + 成交（缩量/放量）+ 板块轮动
- 资金情绪：强势 / 观望 / 谨慎

🎯 **尾盘操作**（对每只持仓股给具体动作）
- [股票名]：[持有/做T减仓/止损] | 价位 + 理由
- 重点：尾盘该不该动？为什么？

💡 **次日低吸机会**（最多 2 只，**只在真正有把握时给**）
- 必须是**回调到位**的，不是当天刚拉起的
- 形态：缩量 + 回踩关键位 + 不破前低
- 写明：入场位、止损位、预期持有周期（1-3 天）
- **没把握就空仓**

📋 **明日观察**
- 关键支撑/压力位（指数 + 你的持仓）
- 关注板块轮动方向

⚠️ 风险提示：以上内容不构成投资建议。

""",

    "interactive": """\
用户提了一个关于 A 股的问题，请你结合【技术面 + 资金面 + 板块】给出完整分析。

【用户问题】
{user_query}

【用户状态】
- 总仓位：{position_pct}%
- 持仓：{holdings_summary}
- 关注股：{watchlist_summary}

【用户风格】保守，不追高，偏好低吸和短线 T，严格止损。

【当前市场数据】
{snapshot_json}

【⚠️ 硬性约束 — 违反任一条就是错的】

1. **绝对禁止编造数据**。如果"当前市场数据"中**没有**某只股票的具体数字（现价、涨跌幅、技术位、资金流），**必须跳过该段或写"暂无该股 X 数据"**，**不准用"假设"、"约"、"大约"、"前高"、"MA60"等占位话术假装有数据**。
2. **不准编造数字**。所有具体价格、涨跌幅、成交量等必须从"当前市场数据"中读取；**不在数据里的数字 = 不准写**。
3. **不准给万能场景化建议**。不要说"若股价放量突破近期压力位（如前高或 MA60）"这种没有具体数字的废话。要么给具体价位，要么不写。
4. **数据源说明**：如果有数据，标"数据来源：腾讯财经 / 东方财富"；如果是基于知识库的定性分析（没有实时数据），明确标"⚠️ 知识库分析（非实时）"。
5. 所有结论必须以"非投资建议"提示结尾。

【输出格式要求】（**严格按以下 6 段输出**）

### 1️⃣ 现价 + 关键位
- **当前价、涨跌幅、今日开高低**（从"当前市场数据"读取，没数据写"暂无该股实时数据"）
- 关键支撑位（**具体价位**，从 MA60 / 布林下轨 / 近期低点 取，最多 2 个）
- 关键压力位（**具体价位**，从近期高点 / 布林上轨 取，最多 2 个）
- ⚠️ 没有数据时直接写"暂无数据"，不要编价位

### 2️⃣ 技术面
- **均线状态**：MA5/10/20/60 排列 → 多头 / 空头 / 缠绕
- **MACD**：金叉 / 死叉 / 多头延续 / 空头延续
- **RSI**：超买 / 超卖 / 中性（>70 警惕，<30 反弹）
- **布林带**：现价在中轨 / 上轨 / 下轨哪个位置
- **整体趋势**：上升 / 下降 / 震荡
- ⚠️ 没有数据时直接说"暂无该股技术面数据"，**不准用"前高"、"MA60"等占位符**

### 3️⃣ 资金面
- **主力净流入**（元 → 亿元）：流入 / 流出 / 观望
- **主力占比**（%）：正 / 负
- **特大单 + 大单** 各自净额
- 一句话态度：强势流入 / 弱势流出 / 散户接盘 / 主力出货
- ⚠️ 没有数据时直接说"暂无该股资金面数据"

### 4️⃣ 基本面（如"市场数据"里有行业 / 板块信息）
- 所属行业 / 板块今日表现
- 板块是否在主线 / 弱势 / 退潮
- ⚠️ 若"市场数据"里没有该股票所属行业信息，跳过此节或标"暂无板块数据"

### 5️⃣ 正反因素对比表
| 利好因素 | 利空因素 |
|---|---|
| ... | ... |
每条 1 行、具体不抽象。**没数据就跳过整节**。

### 6️⃣ 3 种交易场景（**必须有具体数字，不能用占位符**）
- 🟢 **突破 / 走强场景**：具体入场位（数字）+ 仓位（X%）+ 持有周期
- 🟡 **震荡 / 观望场景**：具体操作（不操作 / 减仓 / 做 T）
- 🔴 **破位 / 走弱场景**：具体止损位（数字）+ 离场 / 减仓
- ⚠️ **若没有具体技术位数据，写"暂无具体数据，不给场景化建议（避免误导）"**

【仓位管理】
- 当前仓位 {position_pct}%，本次操作建议不超过 X%
- 单只最大 {max_single}%
- 止损 {stop_loss}%
- 止盈 {take_profit}%

【结尾】
⚠️ 风险提示：以上内容**不构成投资建议**，仅基于公开数据 + 技术面分析，请结合自身风险承受度决策。市场有风险，投资需谨慎。

【📚 知识库分析模式（备用）】

如果"当前市场数据"中**完全没有**用户问到的股票信息，但你**在训练知识中**知道这家公司：
- 切换到"知识库分析模式"，开头加：
  ```
  📚 **知识库分析（非实时）**
  ⚠️ 我的实时数据源未覆盖此股票。以下基于公开知识 + 我的训练数据，**不包含当前价/涨跌幅**。
  ```
- 只输出**定性分析**：
  - 公司主营业务 / 行业地位
  - 行业 / 板块定性看法
  - 投资该股票需要关注什么风险点
  - 类似的已上市公司（如有）
- **不准编造任何具体数字**（价格、估值、市值、营收等）
- 明确建议用户**自行查询实时数据**后做决策

如果**训练知识里也没有这家公司**：
- 直接说"❌ 我的数据源 + 知识库都没有这只股票的信息。请提供 6 位股票代码或在东方财富 / 同花顺 / 雪球查询后分享数据。"
- **不准瞎编**

""",

    "fund": """\
用户提了一个关于**公募基金**的问题，请你按 7 段格式给出完整分析。

【用户问题】
{user_query}

【用户状态】
- 总仓位：{position_pct}%
- 持仓：{holdings_summary}

【用户风格】保守，不追高，偏好低吸和短线 T，严格止损。

【当前基金数据】
{snapshot_json}

【⚠️ 硬性约束 — 违反任一条就是错的】

1. **绝对禁止编造基金数据**。如果"当前基金数据"里没有给出某只基金的【基金代码 / 名称 / 类型 / 规模 / 经理 / 业绩 / 持仓 / 申赎状态】中的任何一项，必须明确说"暂无该基金 X 数据"，**不准用"假设"、"约"、"大概是"等词编造**。
2. **不准把股票当基金分析**。如果用户问的是基金，但"当前基金数据"显示"未找到该基金"，必须直接回复：
   ```
   ❌ 未找到这只基金（可能的代码：xxx）。
   请提供准确的 6 位基金代码（可在天天基金网 / 东方财富查）。
   ```
   **不准分析任何股票、ETF、债券、用户持仓等其他产品**。
3. **不准编造经理名字、业绩数字、持仓股票**。所有数据必须基于"当前基金数据"中给出的具体内容。
4. 涉及具体数字时，写明【数据截至日期】和【数据来源】（如"最新净值 1.1895，截至 2026-06-25，天天基金网"）。
5. **不准类比 / 推断**。不要写"类似于 X 基金"、"应该是 X 策略"、"可能投向科技板块"这种话——除非有具体数据支持。
6. 数据滞后是真实的：季报持仓滞后 1-2 月，净值 T-1（盘后才更新）。

【输出格式要求】（**严格按以下 7 段输出**，缺数据就跳过该段，不编造）

### 1️⃣ 基金概况
- 基金代码 / 全名 / 类型（股票型/混合型/债券型/指数型/QDII/FOF）
- 成立时间 / 最新规模 / 基金经理 / 托管银行 / 业绩比较基准
- 投资策略（一句话总结）
- 基金评级（如有：晨星 / 济安金信 / 上海证券）

### 2️⃣ 业绩分析
- 阶段收益（近1月 / 3月 / 6月 / 1年 / 今年来 / 成立来）vs 同类平均 vs 基准
- **年化收益**（成立以来年化）
- **阶段排名**（同类前 X% / 后 X%，如有）
- **业绩稳定性**：连续 3 月跑赢 / 跑输基准？

### 3️⃣ 风险指标
- **最大回撤**（成立以来最深跌幅）
- **年化波动率**
- **夏普比率**（越高越好，>1 优秀，>2 极好）
- 风险等级（R1-R5）

### 4️⃣ 持仓风格
- **前 10 大重仓股**（占净值 %）
- **持仓集中度**（前10 占净值 X%）
- **行业偏好**（从重仓股推断）
- **大小盘风格**（大盘 / 中盘 / 小盘）—— 从重仓股市值推断
- **成长 vs 价值**：从持仓行业推断（如半导体=成长、银行=价值）

### 5️⃣ 申购赎回状态
- 申购状态：开放 / 限购 X 万 / 暂停
- 赎回状态：开放 / 暂停
- 提示：基金数据通常有 1-2 月滞后，季报持仓数据更滞后（季度后 ~2 月公布）

### 6️⃣ 适合谁
- 适合的投资者类型（保守 / 稳健 / 激进）
- 推荐持有期（半年 / 1 年 / 3 年+）
- 适合定投还是一次性买入
- 当前市场环境下是否适合入场

### 7️⃣ 配置建议
- **是否值得配置**：强推 / 谨慎 / 不建议（理由）
- **配置仓位**：占总投资 X%（考虑用户当前仓位 + 风险偏好）
- **入场方式**：一次性 vs 定投（推荐哪种，金额分几批）
- **持有周期**：6 月 / 1 年 / 3 年
- **触发止盈 / 加仓 / 止损 / 减仓的条件**：
  - 止盈：年化 > X% 或 持有 > X 月
  - 加仓：回调 > X% 且 夏普 > X
  - 止损：跌破 X 日均线 或 最大回撤 > X%
  - 减仓 / 离场：基金经理更换 / 规模急剧缩水 / 风格漂移

### 8️⃣ 操作清单（5 条以内）
- 每条都是具体动作，不写"可以考虑"这种模糊话
- 必须含数字：金额 / 比例 / 触发价 / 时间窗口
- 例：
  - ❌ 模糊："可以适当配置"
  - ✅ 具体："下周二定投 5,000 元，分 6 个月完成，目标仓位 5%"
- 触发条件要可验证：跌破 0.85 元 OR 持有满 12 个月

【仓位管理】
- 当前仓位 {position_pct}%，本次操作建议仓位 X%
- 单只基金最大 {max_single}%
- 止损 {stop_loss}%
- 止盈 {take_profit}%

【结尾】
⚠️ **风险提示**：以上内容**不构成投资建议**。基金过往业绩不代表未来表现，基金投资有风险，过往最大回撤不代表未来最大回撤。基金持仓数据有 1-2 月滞后。市场有风险，投资需谨慎。

""",
}


@dataclass
class LLMAgent:
    """
    LLM 推理入口。

    使用：
        agent = LLMAgent()
        text = agent.analyze_morning_close(snapshot, user_context)

    切换模型（默认 `deepseek-v4-flash` 非思考模式）：
        1. 构造参数：
            LLMAgent(model="v4-pro")                       # V4-pro 非思考
            LLMAgent(model="v4-flash-thinking")            # V4-flash 思考模式
            LLMAgent(model="r1")                            # R1 别名 → V4-flash 思考
        2. 环境变量：`.env` 里 `DEEPSEEK_MODEL=deepseek-v4-pro`
        3. 别名支持：v3 / v4 / v4-flash / v4-pro / r1 / chat / reasoner ...

    控制思考模式（V4-flash / V4-pro 通用）：
        LLMAgent(model="v4-pro", thinking=True, reasoning_effort="high")

    常用组合：
        - 日常快速分析：model="v4-flash"          （默认）
        - 严谨多空推演：model="v4-pro-thinking", reasoning_effort="high"
        - 11:30 / 14:30 推送：model="v4-flash"  （短文本 + 低成本）
    """

    api_key: str | None = None
    base_url: str = "https://api.deepseek.com"
    model: str = DEFAULT_MODEL
    temperature: float | None = None          # None = 用 MODEL_DEFAULTS
    max_tokens: int | None = None
    thinking: bool | None = None              # None = 用 MODEL_DEFAULTS；True/False 强制覆盖
    reasoning_effort: str | None = None       # "low" / "medium" / "high"，None = 用 MODEL_DEFAULTS

    def __post_init__(self):
        # 加载 .env（如果还没加载）
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass

        # 1. 优先级：构造参数 > 环境变量
        if self.api_key is None:
            self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        if self.base_url == "https://api.deepseek.com":  # 默认值时尝试从 env 读
            env_base = os.environ.get("DEEPSEEK_BASE_URL")
            if env_base:
                self.base_url = env_base

        # 2. 规范化模型名（构造参数优先）
        env_model = os.environ.get("DEEPSEEK_MODEL")
        raw = self.model if self.model != DEFAULT_MODEL or not env_model else env_model
        resolved = resolve_model(raw)
        self.model = resolved.name

        # 3. 弃用警告
        if resolved.deprecated:
            log.warning(
                f"⚠️  模型 {raw!r} 已弃用（2026/07/24 EOL），已自动映射到 {resolved.name}"
                f"{' (思考模式)' if resolved.thinking else ''}。"
                f"建议改用 'v4-flash' 或 'v4-pro'。"
            )

        # 4. 思考模式：用户没显式传，就用 resolved 的默认值
        if self.thinking is None:
            self.thinking = resolved.thinking

        # 5. 用模型默认值填充缺失参数
        defaults = MODEL_DEFAULTS.get(self.model, MODEL_DEFAULTS[DEFAULT_MODEL])
        if self.temperature is None:
            self.temperature = defaults["temperature"]
        if self.max_tokens is None:
            self.max_tokens = defaults["max_tokens"]
        if self.reasoning_effort is None:
            self.reasoning_effort = defaults.get("reasoning_effort", "medium")

        if not self.api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY not found. Set it in .env or env vars."
            )

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        log.info(
            f"LLMAgent ready: model={self.model}, "
            f"thinking={self.thinking}, reasoning_effort={self.reasoning_effort}, "
            f"temp={self.temperature}, max_tokens={self.max_tokens}"
        )

    # ================================================================== #
    #  三个入口：上午收盘 / 尾盘 / 交互
    # ================================================================== #
    def analyze_morning_close(self, snapshot: dict[str, Any],
                              user_ctx: dict[str, Any]) -> str:
        """11:30 触发：上午复盘 + 下午预判。"""
        prompt = PROMPT_TEMPLATES["morning_close"].format(
            date=datetime.now().strftime("%Y-%m-%d"),
            snapshot_json=_fmt_snapshot(snapshot),
            **_fmt_user_ctx(user_ctx),
        )
        return self._call(prompt)

    def analyze_afternoon_close(self, snapshot: dict[str, Any],
                                user_ctx: dict[str, Any]) -> str:
        """14:30 触发：尾盘决策 + 次日布局。"""
        prompt = PROMPT_TEMPLATES["afternoon_close"].format(
            date=datetime.now().strftime("%Y-%m-%d"),
            snapshot_json=_fmt_snapshot(snapshot),
            **_fmt_user_ctx(user_ctx),
        )
        return self._call(prompt)

    def analyze_interactive(self, user_query: str, snapshot: dict[str, Any],
                            user_ctx: dict[str, Any]) -> str:
        """交互 query：用户问啥答啥。"""
        prompt = PROMPT_TEMPLATES["interactive"].format(
            user_query=user_query,
            snapshot_json=_fmt_snapshot(snapshot),
            **_fmt_user_ctx(user_ctx),
        )
        return self._call(prompt)

    def chat_turn(self, user_input: str, history: list[dict[str, str]],
                  snapshot: dict[str, Any],
                  user_ctx: dict[str, Any]) -> str:
        """
        多轮对话：传 history，每轮都注入最新行情。

        history 格式：[{"role": "user"/"assistant", "content": "..."}, ...]
        """
        user_vars = _fmt_user_ctx(user_ctx)
        snapshot_text = _fmt_snapshot(snapshot)
        current_user_msg = (
            f"【当前市场数据】\n{snapshot_text}\n\n"
            f"【用户状态】\n"
            f"总仓位：{user_vars['position_pct']}%\n"
            f"持仓：{user_vars['holdings_summary']}\n"
            f"关注股：{user_vars['watchlist_summary']}\n\n"
            f"【用户当前问题】\n{user_input}"
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT_BASE}]
        # 控制 history 长度，避免超 token 上限
        max_history = 20
        if len(history) > max_history:
            history = history[-max_history:]
        messages.extend(history)
        messages.append({"role": "user", "content": current_user_msg})

        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False,
                **self._extra_body_kwargs(),
            )
            content, reasoning = _extract_content(resp)
            if reasoning:
                log.debug(f"[thinking] {len(reasoning)} chars")
            return content
        except Exception as exc:
            return f"❌ LLM 调用失败：{type(exc).__name__}: {exc}"

    # ================================================================== #
    #  内部：实际调用
    # ================================================================== #
    def _call(self, user_prompt: str) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_BASE},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False,
                **self._extra_body_kwargs(),
            )
            content, reasoning = _extract_content(resp)
            if reasoning:
                log.debug(f"[thinking] {len(reasoning)} chars")
            return content
        except Exception as exc:
            return f"❌ LLM 调用失败：{type(exc).__name__}: {exc}"

    def _extra_body_kwargs(self) -> dict[str, Any]:
        """
        构造 OpenAI SDK 的 extra_body 参数（DeepSeek V4 思考模式）。

        V4 文档示例：
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high"

        非 V4 模型（V3 / R1 旧名）忽略这个参数。
        """
        # 只对 V4 传 thinking 参数
        if not self.model.startswith("deepseek-v4"):
            return {}
        return {
            "extra_body": {
                "thinking": {"type": "enabled" if self.thinking else "disabled"},
                **({"reasoning_effort": self.reasoning_effort}
                   if self.thinking and self.reasoning_effort else {}),
            }
        }


def _extract_content(resp: Any) -> tuple[str, str]:
    """
    从 DeepSeek 响应里抽出最终答案 + 思维链。

    V4 (R1 思考模式) 响应：
        message.content           = 最终答案
        message.reasoning_content = 思维链

    V3 / V4 非思考模式：只有 message.content

    Returns
    -------
    (content, reasoning_content) — content 已 strip，reasoning 可能为空字符串。
    边界情况：V4 有时 content 为空但 reasoning_content 有总结（罕见），做兜底。
    """
    msg = resp.choices[0].message
    content = (getattr(msg, "content", None) or "").strip()
    reasoning = (getattr(msg, "reasoning_content", None) or "").strip()
    if not content and reasoning:
        content = reasoning
    if not content:
        content = "(空响应)"
    return content, reasoning


# ====================================================================== #
#  格式化辅助
# ====================================================================== #
def _fmt_snapshot(snapshot: dict[str, Any]) -> str:
    """把市场快照字典压成结构化文本。"""
    if not snapshot:
        return "(无数据)"

    # ====== 基金快照分支 ======
    if snapshot.get("fund") or snapshot.get("parsed_code"):
        return _fmt_fund_snapshot(snapshot)

    lines = []

    # 指数
    if snapshot.get("indices"):
        lines.append("【三大指数】")
        for code, info in snapshot["indices"].items():
            arrow = "↑" if info["pct_change"] >= 0 else "↓"
            lines.append(
                f"- {info['name']}（{code}）：{info['last']} "
                f"{arrow}{abs(info['pct_change']):.2f}% "
                f"[开 {info['open']} | 高 {info['high']} | 低 {info['low']}]"
            )

    # 持仓
    if snapshot.get("holdings"):
        lines.append("\n【持仓个股】")
        for sym, info in snapshot["holdings"].items():
            arrow = "↑" if info["pct_change"] >= 0 else "↓"
            lines.append(
                f"- {info['name']}（{sym}）：{info['last']} "
                f"{arrow}{abs(info['pct_change']):.2f}% "
                f"[开 {info['open']} | 高 {info['high']} | 低 {info['low']} | "
                f"成交 {info['amount_wan']:.0f}万]"
            )

    # 关注股
    if snapshot.get("watchlist"):
        lines.append("\n【关注个股】")
        for sym, info in snapshot["watchlist"].items():
            arrow = "↑" if info["pct_change"] >= 0 else "↓"
            lines.append(
                f"- {info['name']}（{sym}）：{info['last']} "
                f"{arrow}{abs(info['pct_change']):.2f}%"
            )

    # 用户在对话中提到的股票
    if snapshot.get("mentioned"):
        lines.append("\n【用户提到的个股】")
        for sym, info in snapshot["mentioned"].items():
            arrow = "↑" if info["pct_change"] >= 0 else "↓"
            lines.append(
                f"- {info['name']}（{sym}）：{info['last']} "
                f"{arrow}{abs(info['pct_change']):.2f}% "
                f"[开 {info['open']} | 高 {info['high']} | 低 {info['low']} | "
                f"成交 {info['amount_wan']:.0f}万]"
            )

    # === 用户提到股票的技术指标 + 资金流（enrich）===
    enriched = snapshot.get("mentioned_enriched", {})
    for sym, payload in enriched.items():
        name = payload.get("name", sym)

        # 技术指标
        ind = payload.get("indicators")
        if ind:
            lines.append(f"\n【{name}（{sym}）技术面】")
            lines.append(
                f"- 现价 {ind['last']}（{ind['pct_change']:+.2f}%），昨收 {ind['prev_close']}"
            )
            lines.append(
                f"- 均线：MA5={ind.get('ma5', '-')} / "
                f"MA10={ind.get('ma10', '-')} / "
                f"MA20={ind.get('ma20', '-')} / "
                f"MA60={ind.get('ma60', '-')}"
            )
            lines.append(f"- MA 排列：{ind.get('ma_status', '-')}")
            lines.append(
                f"- MACD：DIF={ind.get('dif', '-')} / "
                f"DEA={ind.get('dea', '-')} / "
                f"BAR={ind.get('macd_bar', '-')} → {ind.get('macd_status', '-')}"
            )
            lines.append(f"- RSI14：{ind.get('rsi14', '-')} → {ind.get('rsi_status', '-')}")
            lines.append(
                f"- 布林带：上 {ind.get('boll_upper', '-')} / "
                f"中 {ind.get('boll_mid', '-')} / "
                f"下 {ind.get('boll_lower', '-')} → {ind.get('boll_status', '-')}"
            )
            lines.append(f"- 整体趋势：{ind.get('trend', '-')}")
            lines.append(f"- K 线形态：{ind.get('recent_pattern', '-')}")

        # 资金流
        flow = payload.get("capital_flow")
        if flow:
            main_net = flow.get("main_net", 0)
            main_pct = flow.get("main_pct", 0)
            attitude = "强势流入" if main_net > 0 else ("弱势流出" if main_net < 0 else "观望")
            lines.append(f"\n【{name}（{sym}）资金面】")
            # 主力净额（元 → 亿元）
            main_yi = main_net / 1e8
            sl_yi = flow.get("super_large_net", 0) / 1e8
            lg_yi = flow.get("large_net", 0) / 1e8
            lines.append(
                f"- 主力净流入：{main_yi:+.2f} 亿（占比 {main_pct:+.2f}%） → {attitude}"
            )
            lines.append(
                f"- 特大单：{sl_yi:+.2f} 亿（{flow.get('super_large_pct', 0):+.2f}%）"
                f"  /  大单：{lg_yi:+.2f} 亿（{flow.get('large_pct', 0):+.2f}%）"
            )
            amt_yi = flow.get("amount_cny", 0) / 1e8
            lines.append(f"- 当日成交额：{amt_yi:.2f} 亿")

    # 板块
    if snapshot.get("sector_top"):
        lines.append("\n【领涨板块】")
        for s in snapshot["sector_top"][:5]:
            name = s.get("name", "?")
            pct = s.get("pct_change", 0)
            leader = s.get("leader", "")
            lines.append(f"- {name}（+{pct:.2f}%）领涨股：{leader}")

    if snapshot.get("sector_bottom"):
        lines.append("\n【领跌板块】")
        for s in snapshot["sector_bottom"][:5]:
            name = s.get("name", "?")
            pct = s.get("pct_change", 0)
            leader = s.get("leader", "")
            lines.append(f"- {name}（{pct:.2f}%）领跌股：{leader}")

    return "\n".join(lines)


def _fmt_fund_snapshot(snap: dict[str, Any]) -> str:
    """把基金快照字典压成结构化文本。"""
    lines = []
    name = snap.get("parsed_name") or "(未知基金)"
    code = snap.get("parsed_code") or "(未知代码)"

    # 1. 基本信息
    f = snap.get("fund") or {}
    lines.append(f"【基金基本信息】")
    lines.append(f"- 基金代码 / 名称：{code} / {f.get('name') or name}")
    lines.append(f"- 基金全称：{f.get('full_name', '-')}")
    lines.append(f"- 类型：{f.get('type', '-')}")
    lines.append(f"- 成立时间：{f.get('inception_date', '-')}")
    lines.append(f"- 最新规模：{f.get('scale', '-')}")
    lines.append(f"- 基金经理：{f.get('manager', '-')}")
    lines.append(f"- 托管银行：{f.get('custodian', '-')}")
    lines.append(f"- 业绩比较基准：{f.get('benchmark', '-')}")
    if f.get("rating"):
        lines.append(f"- 基金评级：{f.get('rating')}")
    if f.get("investment_strategy"):
        lines.append(f"- 投资策略：{f.get('investment_strategy')[:200]}")

    # 2. 业绩
    m = snap.get("metrics") or {}
    if m:
        lines.append(f"\n【业绩指标】（数据截至 {m.get('latest_date', '-')}）")
        lines.append(f"- 最新单位净值：{m.get('latest_nav', '-')}")
        lines.append(
            f"- 阶段收益（近1月 / 3月 / 6月 / 1年）："
            f"{m.get('perf_1m', 0):+.2f}% / {m.get('perf_3m', 0):+.2f}% / "
            f"{m.get('perf_6m', 0):+.2f}% / {m.get('perf_1y', 0):+.2f}%"
        )
        lines.append(
            f"- 今年来 / 成立来 / 年化："
            f"{m.get('perf_ytd', 0):+.2f}% / {m.get('perf_since_inception', 0):+.2f}% / "
            f"{m.get('annualized', 0):+.2f}%"
        )

    # 3. 风险
    if m:
        lines.append(f"\n【风险指标】")
        lines.append(f"- 最大回撤：{m.get('max_drawdown', 0):.2f}%")
        lines.append(f"- 年化波动率：{m.get('volatility', 0):.2f}%")
        lines.append(f"- 夏普比率：{m.get('sharpe', 0):.2f}")
        # 夏普解读
        sharpe = m.get('sharpe', 0)
        if sharpe >= 2:
            lines.append(f"  → 解读：极好（>2）")
        elif sharpe >= 1:
            lines.append(f"  → 解读：优秀（1-2）")
        elif sharpe >= 0:
            lines.append(f"  → 解读：一般（0-1）")
        else:
            lines.append(f"  → 解读：差（<0，跑输无风险）")

    # 4. 持仓
    holdings = snap.get("holdings") or []
    if holdings:
        lines.append(f"\n【最新季度前 10 大持仓】（集中度 {snap.get('concentration_top10_pct', 0):.2f}%）")
        for h in holdings:
            lines.append(
                f"- #{h['rank']} {h['name']}（{h['symbol']}）："
                f"占净值 {h['pct_of_nav']:.2f}% / 持仓市值 {h['market_value']:.0f}万"
            )

    # 5. 申赎状态
    status = snap.get("status") or {}
    if status:
        lines.append(f"\n【申赎状态】")
        lines.append(f"- 申购：{status.get('purchase_status', '未知')}")
        lines.append(f"- 赎回：{status.get('redemption_status', '未知')}")

    # 6. 数据时效提示
    lines.append(f"\n【数据时效】")
    lines.append(f"- 净值数据：T-1（场外基金当日 22:00 后更新）")
    lines.append(f"- 持仓数据：最新季度，约滞后 1-2 个月")
    lines.append(f"- 申赎状态：实时")

    return "\n".join(lines)


def _fmt_user_ctx(user_ctx: dict[str, Any]) -> dict[str, str]:
    """把用户上下文压成 prompt 变量。"""
    pos = user_ctx.get("portfolio", {})
    holdings = pos.get("holdings", [])
    risk = user_ctx.get("risk_profile", {})

    holdings_summary_parts = []
    for h in holdings:
        sym = h.get("symbol", "?")
        cost = h.get("cost", 0)
        shares = h.get("shares", 0)
        holdings_summary_parts.append(f"{sym}（成本 {cost}×{shares}股）")
    holdings_summary = "，".join(holdings_summary_parts) or "(无持仓)"

    watchlist = user_ctx.get("watchlist", [])
    watchlist_summary = "，".join(watchlist) or "(无)"

    return {
        "position_pct": pos.get("total_position_pct", "?"),
        "cash_pct": pos.get("cash_pct", "?"),
        "holdings_summary": holdings_summary,
        "watchlist_summary": watchlist_summary,
        "max_single": risk.get("max_single_position_pct", "?"),
        "stop_loss": risk.get("stop_loss_pct", "?"),
        "take_profit": risk.get("take_profit_pct", "?"),
    }


# ---------------------------------------------------------------------- #
#  Self-test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    from intraday import IntradayClient
    ic = IntradayClient()
    snap = ic.build_market_snapshot(
        watchlist=["600276", "601318", "002594"],
        holdings=["600519", "000001", "300750"],
    )

    user_ctx = {
        "portfolio": {
            "total_position_pct": 35,
            "cash_pct": 65,
            "holdings": [
                {"symbol": "600519", "cost": 1680, "shares": 100},
                {"symbol": "000001", "cost": 12.5, "shares": 5000},
            ],
        },
        "watchlist": ["600276", "601318", "002594"],
        "risk_profile": {
            "max_single_position_pct": 20,
            "stop_loss_pct": 8,
            "take_profit_pct": 20,
        },
    }

    agent = LLMAgent()
    print("=" * 60)
    print("测试 morning_close 分析：")
    print("=" * 60)
    text = agent.analyze_morning_close(snap, user_ctx)
    print(text)