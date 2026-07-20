"""
企业微信智能机器人（长连接模式）—— 主动连出去，无需公网 URL。

架构：
    群里 @智能机器人
       ↓ (WebSocket)
    本程序接收事件
       ↓
    命令路由 / LLM 推理
       ↓
    WebSocket 主动 send_message
       ↓
    群里看到 bot 回复

相比自建应用的好处：
- 不用 ngrok / cloudflared
- 不用公网 URL
- 不用 IP 白名单
- 不用 URL 回调配置
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any

import yaml
from aibot import WSClient, WSClientOptions

from intraday import IntradayClient, extract_symbols, fetch_stock_snapshot, format_movers_table, format_potential_table, score_potential_stocks, fetch_top_movers
from fund import parse_fund_query, recommend_funds
from funds import is_fund_query, build_fund_snapshot
from llm_agent import LLMAgent, PROMPT_TEMPLATES

# Meta-query 检测：用户问"为什么没有数据"型的问题
META_QUERY_KEYWORDS = [
    "为什么", "怎么又", "怎么没", "查不到", "找不到",
    "覆盖", "支持", "数据源", "哪些", "不工作", "不支持",
    "你认识", "你能查", "你能", "你会",
]


def is_meta_query(text: str) -> bool:
    """
    检测 query 是否是关于"数据覆盖 / bot 能力"的元问题。
    这种问题不应触发股票/基金分析。
    """
    # 含元问题关键词
    for kw in META_QUERY_KEYWORDS:
        if kw in text:
            return True
    return False


META_QUERY_RESPONSE = """\
📊 **数据覆盖说明**

我的实时数据源主要覆盖：

**股票（A 股 / 港股通）**：
- 实时行情：腾讯财经（首选）+ Sina 兜底（约 5,500+ 只 A 股覆盖）
- 历史 K 线：腾讯 `fqkline`（每只 120 天日 K，供 MA/MACD/RSI/布林 计算）
- 资金流：东方财富 HTTP 直调（绕开反爬），含特大单/大单净额
- 基本面：股票名模糊匹配（akshare 全 A 股表 5,500+ 只）

**基金**：
- 净值历史：东方财富（27,000+ 只开放式基金）
- 持仓：东方财富（最新季度前 10 大）
- 业绩指标：基于 NAV 自计算（夏普/回撤/年化）
- 申赎状态：东方财富
- 基金名模糊匹配：akshare `fund_name_em` 全表

**板块**：
- 行业板块：同花顺（akshare `stock_board_industry_summary_ths`）

**典型能查的**：
- 任何在沪深京交易所上市的 A 股（提供名称 / 6 位代码即可）
- 任何有公开净值的公募基金

**可能查不到的**：
- 新上市 < 1 周的股票 / 基金
- 数据源临时不可用
- 极小众/已退市产品
- 用户说"某个名字"但实际不存在（会说"未找到"并提示）

**遇到问题请这样**：
- 报具体 6 位代码（如 `600237` / `026173`）→ 一定能查到
- 报完整名（如"路博迈港股通科技股票A"）→ 模糊匹配
- 实在查不到 → 我会明确告诉你"未找到"，不会瞎分析
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] smart_bot: %(message)s",
)
log = logging.getLogger("smart_bot")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


# ====================================================================== #
#  Config helpers
# ====================================================================== #
def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ====================================================================== #
#  命令实现
# ====================================================================== #
def cmd_portfolio_text(cfg: dict) -> str:
    portfolio = cfg.get("portfolio", {})
    holdings = portfolio.get("holdings", [])
    pos_pct = portfolio.get("total_position_pct", "?")
    cash_pct = portfolio.get("cash_pct", "?")

    if not holdings:
        return f"📭 当前无持仓\n\n总仓位：{pos_pct}% | 现金：{cash_pct}%\n\n回复 `添加 600519 1680 100` 加第一只"

    symbols = [h["symbol"] for h in holdings]
    try:
        snap = fetch_stock_snapshot(symbols)
    except Exception:
        snap = {}

    lines = [f"💼 **当前持仓**（总仓位 {pos_pct}% / 现金 {cash_pct}%）", ""]
    total_cost = 0.0
    total_mv = 0.0
    for h in holdings:
        sym = h.get("symbol", "?")
        cost = float(h.get("cost", 0))
        shares = int(h.get("shares", 0))
        info = snap.get(sym, {})
        name = info.get("name", sym)
        last = float(info.get("last", 0)) if info else 0
        if not last:
            lines.append(f"- {name}（{sym}）：成本 {cost} × {shares}股（无实时价）")
            continue
        mv = last * shares
        cb = cost * shares
        pnl = mv - cb
        pnl_pct = (pnl / cb * 100) if cb else 0
        total_cost += cb
        total_mv += mv
        arrow = "🟢" if pnl >= 0 else "🔴"
        lines.append(
            f"- {name}（{sym}）：成本 {cost} × {shares}股\n"
            f"  现价 {last} | 市值 {mv:,.0f} | "
            f"{arrow} 浮盈浮亏 {pnl:+,.0f}（{pnl_pct:+.2f}%）"
        )
    if total_cost:
        tp = total_mv - total_cost
        tpct = (tp / total_cost * 100) if total_cost else 0
        arrow = "🟢" if tp >= 0 else "🔴"
        lines.append("")
        lines.append(
            f"📊 **合计**：成本 {total_cost:,.0f} | 市值 {total_mv:,.0f} | "
            f"{arrow} {tp:+,.0f}（{tpct:+.2f}%）"
        )
    return "\n".join(lines)


def cmd_risk_text(cfg: dict) -> str:
    risk = cfg.get("risk_profile", {})
    if not risk:
        return "📭 未配置风险偏好"
    return (
        "⚖️ **当前风险偏好**\n"
        f"- 单只最大仓位：{risk.get('max_single_position_pct', '?')}%\n"
        f"- 止损线：{risk.get('stop_loss_pct', '?')}%\n"
        f"- 止盈线：{risk.get('take_profit_pct', '?')}%"
    )


def cmd_watchlist_text(cfg: dict) -> str:
    wl = cfg.get("watchlist", [])
    if not wl:
        return "📭 关注列表为空"
    return f"👀 **关注列表**：{', '.join(wl)}"


def cmd_help_text() -> str:
    return (
        "🤖 **可用命令**\n"
        "```\n"
        "持仓                    显示持仓 + 浮盈浮亏\n"
        "风险                    显示风险偏好\n"
        "关注                    显示关注列表\n"
        "基金 股票型 近1年 前10   基金排行\n"
        "潜力股                  量化打分潜力个股\n"
        "涨跌榜                  涨跌幅榜\n"
        "帮助                    这条\n"
        "其它任意文本             LLM 实时分析（自动拉行情）\n"
        "```"
    )


def cmd_fund_text(text: str, _cfg: dict) -> str:
    params = parse_fund_query(text)
    return recommend_funds(**params)


def cmd_potential_text(_text: str, _cfg: dict) -> str:
    """量化打分潜力个股 TOP 10（动量+量能+形态三维评分）。"""
    try:
        df = score_potential_stocks(top_n=10)
    except Exception as exc:
        return f"❌ 拉取数据失败：{exc}"
    return "🎯 **潜力个股 TOP 10**\n\n" + format_potential_table(df)


def cmd_movers_text(_text: str, _cfg: dict) -> str:
    """全市场涨幅 / 跌幅 TOP 10（已排除 ST）。"""
    try:
        df = fetch_top_movers(top_n=10)
    except Exception as exc:
        return f"❌ 拉取数据失败：{exc}"
    return format_movers_table(df)


def cmd_llm_text(text: str, cfg: dict) -> str:
    """任意自然语言 query：自动识别 fund / stock + LLM 实时分析。"""
    portfolio = cfg.get("portfolio", {})
    user_ctx = {
        "portfolio": portfolio,
        "watchlist": cfg.get("watchlist", []),
        "risk_profile": cfg.get("risk_profile", {}),
    }
    holdings = [h["symbol"] for h in portfolio.get("holdings", [])]

    # ===== Meta query 短路（用户问"为什么没有数据"型问题）=====
    if is_meta_query(text):
        return META_QUERY_RESPONSE

    # ===== 自动识别：fund 还是 stock =====
    if is_fund_query(text):
        return _handle_fund_query(text, user_ctx)
    return _handle_stock_query(text, user_ctx, holdings)


def _handle_stock_query(text: str, user_ctx: dict, holdings: list[str]) -> str:
    """股票 query：拉最新行情 + LLM 实时分析。"""
    mentioned = extract_symbols(text)

    try:
        snapshot = IntradayClient().build_market_snapshot(
            watchlist=user_ctx["watchlist"],
            holdings=holdings,
            mentioned=mentioned,
        )
    except Exception as exc:
        log.exception("snapshot failed")
        return f"❌ 拉取行情失败：{exc}"

    return LLMAgent().analyze_interactive(text, snapshot, user_ctx)


def _handle_fund_query(text: str, user_ctx: dict) -> str:
    """基金 query：拉基金数据 + 用 fund prompt。

    找不到基金时直接报错，不调 LLM（避免瞎答）。
    """
    try:
        snapshot = build_fund_snapshot(text, cfg=user_ctx)
    except Exception as exc:
        log.exception("fund snapshot failed")
        return f"❌ 拉取基金数据失败：{exc}"

    if snapshot is None:
        return (
            "❌ 未识别到这只基金。\n\n"
            "**请提供**：\n"
            "- 6 位基金代码（如 `026173`、`025924`），可在天天基金网 / 东方财富 / 支付宝搜基金名查到\n"
            "- 或完整基金简称（如「平安港股通红利精选混合发起式 A」）\n\n"
            "我目前识别范围：\n"
            "- 80+ 常见基金名（平安/永赢/招商/易方达/华夏/嘉实/兴全/富国/汇添富/广发/睿远 + 行业 ETF）\n"
            "- akshare 全基金表（27000+）做模糊匹配\n"
            "- 50+ 基金公司名（路博迈/平安/摩根/贝莱德/高盛 等）"
        )

    # 关键检查：fund 数据真的拿到了吗？
    if not snapshot.get("fund") and not snapshot.get("metrics"):
        code = snapshot.get("parsed_code", "?")
        return (
            f"❌ 识别到基金代码 {code}，但拉不到数据。\n\n"
            f"可能原因：\n"
            f"- 基金代码错误（请核对天天基金网）\n"
            f"- 基金太新（akshare 还未收录）\n"
            f"- 数据源暂时不可用（稍后重试）"
        )

    # 用 fund 格式 prompt
    from llm_agent import _fmt_snapshot, _fmt_user_ctx

    prompt = PROMPT_TEMPLATES["fund"].format(
        user_query=text,
        snapshot_json=_fmt_snapshot(snapshot),
        **_fmt_user_ctx(user_ctx),
    )

    # 直接调 LLM（用 fund prompt，不走 analyze_interactive 那个股票模板）
    agent = LLMAgent()
    try:
        resp = agent._call(prompt)
        return resp
    except Exception as exc:
        log.exception("fund LLM call failed")
        return f"❌ 基金分析失败：{exc}"


COMMANDS = {
    "持仓": cmd_portfolio_text,
    "我的持仓": cmd_portfolio_text,
    "风险": cmd_risk_text,
    "关注": cmd_watchlist_text,
    "基金": cmd_fund_text,
    "fund": cmd_fund_text,
    "潜力股": cmd_potential_text,
    "涨跌榜": cmd_movers_text,
    "帮助": lambda c: cmd_help_text(),
    "help": lambda c: cmd_help_text(),
    "?": lambda c: cmd_help_text(),
}


def route_command(text: str, cfg: dict) -> str:
    """
    路由用户输入到命令。

    - 第一个 token 是已知命令词 → 调对应函数
    - 否则 → 当作自然语言 query 走 LLM
    """
    parts = text.strip().split()
    if parts and parts[0] in COMMANDS:
        return COMMANDS[parts[0]](cfg)
    return cmd_llm_text(text, cfg)


# ====================================================================== #
#  智能机器人
# ====================================================================== #
class SmartBot:
    def __init__(self, bot_id: str, secret: str):
        self.bot_id = bot_id
        self.secret = secret
        self.client: WSClient | None = None
        self._ready = asyncio.Event()

    async def start(self):
        log.info(f"connecting as bot_id={self.bot_id[:8]}...")
        self.client = WSClient(WSClientOptions(
            bot_id=self.bot_id,
            secret=self.secret,
        ))

        # 注册事件（async 回调）
        self.client.on("authenticated", self._on_authenticated)
        self.client.on("message.text", self._on_message_text)
        self.client.on("event.enter_chat", self._on_enter_chat)
        # 兜底
        self.client.on("event", self._on_event)

        try:
            await self.client.connect()
            log.info("connected, waiting for messages...")
            await self._ready.wait()
            # 保持运行
            while True:
                await asyncio.sleep(1)
        except Exception as exc:
            log.exception(f"connection failed: {exc}")

    async def _on_authenticated(self, *_):
        log.info("✅ 鉴权成功，智能机器人在线")
        self._ready.set()

    async def _on_event(self, frame):
        """兜底事件处理。"""
        cmd = frame.get("cmd", "")
        log.debug(f"event: {cmd}")

    async def _on_enter_chat(self, frame):
        log.info("user entered chat")
        try:
            await self.client.reply_welcome(frame, {
                "msgtype": "text",
                "text": {
                    "content": "👋 你好！我是 A 股分析助手。\n\n"
                               "回复 `帮助` 看命令。\n"
                               "或者直接问问题，比如「京东方现在能加仓吗？」"
                }
            })
        except Exception as exc:
            log.error(f"reply_welcome failed: {exc}")

    async def _on_message_text(self, frame):
        try:
            body = frame.get("body", {}) or {}
            content = body.get("text", {}).get("content", "")
            user_id = body.get("from", {}).get("userid", "")
            chatid = body.get("chatid", "")
            chattype = body.get("chattype", "")
            msgid = body.get("msgid", "")

            content = re.sub(r"^@[^\s]+\s*", "", content).strip()
            log.info(f"user ({user_id}) in {chattype} chat: {content!r}")

            cfg = load_config()
            reply = route_command(content, cfg)
            log.info(f"reply ({len(reply)} chars)")

            # 优先 markdown，失败降级到 text
            try:
                await self.client.send_message(
                    chatid=chatid,
                    body={
                        "msgtype": "markdown",
                        "markdown": {"content": reply},
                    },
                )
            except Exception as exc:
                log.warning(f"markdown send failed: {exc}, falling back to text")
                # 截断 text（最大 2048 字节）
                await self.client.send_message(
                    chatid=chatid,
                    body={
                        "msgtype": "text",
                        "text": {"content": reply[:2000]},
                    },
                )
        except Exception as exc:
            log.exception(f"handle text message failed: {exc}")


# ====================================================================== #
#  入口
# ====================================================================== #
def main():
    import argparse
    parser = argparse.ArgumentParser(description="A 股分析 - 智能机器人长连接模式")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    bot_id = os.environ.get("WECOM_SMART_BOT_ID", "")
    secret = os.environ.get("WECOM_SMART_BOT_SECRET", "")

    if not bot_id or not secret:
        log.error("WECOM_SMART_BOT_ID / WECOM_SMART_BOT_SECRET 未配置")
        log.error("在 .env 里加：")
        log.error("  WECOM_SMART_BOT_ID=你的BotID")
        log.error("  WECOM_SMART_BOT_SECRET=你的Secret")
        return

    log.info(f"starting smart bot: {bot_id[:12]}...")
    asyncio.run(SmartBot(bot_id, secret).start())


if __name__ == "__main__":
    main()