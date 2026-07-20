"""
企业微信双向对话 + 持仓管理。

接收：自建应用回调（无需 IP 白名单）
发送：群机器人 webhook（你已配好）

支持命令（@bot 后输入）：
    持仓                 —— 列出当前持仓 + 浮盈浮亏
    添加 600519 1680 100  —— 加持仓（代码 成本 股数）
    删除 600519           —— 删持仓
    改成 50%              —— 改总仓位
    风险                 —— 显示当前风险偏好
    帮助                 —— 命令列表
    其它                 —— LLM 实时分析
"""

from __future__ import annotations

import logging
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from threading import Lock
from typing import Any

import yaml
from flask import Flask, request, abort

from intraday import IntradayClient, extract_symbols
from fund import parse_fund_query, recommend_funds, format_movers_table, format_potential_table, score_potential_stocks
from llm_agent import LLMAgent
from wechat_crypto import WeComCrypto
from wechat_push import Pusher


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] webhook: %(message)s",
)
log = logging.getLogger("webhook")

CONFIG_PATH = Path(__file__).parent / "config.yaml"
_config_lock = Lock()


# ====================================================================== #
#  Config helpers
# ====================================================================== #
def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_config(cfg: dict) -> None:
    with _config_lock:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


# ====================================================================== #
#  命令路由
# ====================================================================== #
def cmd_portfolio(_args: list[str], cfg: dict) -> str:
    """显示当前持仓 + 浮盈浮亏。"""
    portfolio = cfg.get("portfolio", {})
    holdings = portfolio.get("holdings", [])
    pos_pct = portfolio.get("total_position_pct", "?")
    cash_pct = portfolio.get("cash_pct", "?")

    if not holdings:
        return f"📭 当前无持仓\n\n总仓位：{pos_pct}% | 现金：{cash_pct}%\n\n回复 `添加 600519 1680 100` 加第一只持仓"

    # 拉实时数据算浮盈浮亏
    symbols = [h["symbol"] for h in holdings]
    snap = IntradayClient().fetch_stock_snapshot(symbols) if hasattr(
        IntradayClient, 'fetch_stock_snapshot'
    ) else {}

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
        cost_basis = cost * shares
        pnl = mv - cost_basis
        pnl_pct = (pnl / cost_basis * 100) if cost_basis else 0
        total_cost += cost_basis
        total_mv += mv
        arrow = "🟢" if pnl >= 0 else "🔴"
        lines.append(
            f"- {name}（{sym}）：成本 {cost} × {shares}股\n"
            f"  现价 {last} | 市值 {mv:,.0f} | "
            f"{arrow} 浮盈浮亏 {pnl:+,.0f}（{pnl_pct:+.2f}%）"
        )

    if total_cost:
        total_pnl = total_mv - total_cost
        total_pct = (total_pnl / total_cost * 100) if total_cost else 0
        arrow = "🟢" if total_pnl >= 0 else "🔴"
        lines.append("")
        lines.append(
            f"📊 **合计**：成本 {total_cost:,.0f} | 市值 {total_mv:,.0f} | "
            f"{arrow} {total_pnl:+,.0f}（{total_pct:+.2f}%）"
        )

    return "\n".join(lines)


def cmd_add(args: list[str], cfg: dict) -> str:
    """添加/更新持仓：添加 <code> <cost> <shares>"""
    if len(args) < 3:
        return "❌ 格式：`添加 600519 1680 100`（代码 成本 股数）"

    code = args[0]
    if not re.match(r"^\d{6}$", code):
        return f"❌ 股票代码格式错：{code}（需要 6 位数字）"

    try:
        cost = float(args[1])
        shares = int(args[2])
    except ValueError:
        return f"❌ 成本/股数格式错：{args[1]} {args[2]}"

    if cost <= 0 or shares <= 0:
        return "❌ 成本和股数必须 > 0"

    # 找股票名（用 snapshot 查一下）
    try:
        from intraday import fetch_stock_snapshot
        snap = fetch_stock_snapshot([code])
        name = snap.get(code, {}).get("name", code)
    except Exception:
        name = code

    with _config_lock:
        portfolio = cfg.setdefault("portfolio", {})
        holdings = portfolio.setdefault("holdings", [])

        # 覆盖已有的
        for i, h in enumerate(holdings):
            if h.get("symbol") == code:
                holdings[i] = {"symbol": code, "cost": cost, "shares": shares}
                save_config(cfg)
                return f"✅ 已更新：{name}（{code}）成本 {cost} × {shares}股"

        holdings.append({"symbol": code, "cost": cost, "shares": shares})
        save_config(cfg)

    return f"✅ 已添加：{name}（{code}）成本 {cost} × {shares}股"


def cmd_remove(args: list[str], cfg: dict) -> str:
    """删除持仓：删除 <code>"""
    if len(args) < 1:
        return "❌ 格式：`删除 600519`"

    code = args[0]
    portfolio = cfg.get("portfolio", {})
    holdings = portfolio.get("holdings", [])

    new_holdings = [h for h in holdings if h.get("symbol") != code]
    if len(new_holdings) == len(holdings):
        return f"📭 持仓里没有 {code}"

    with _config_lock:
        cfg["portfolio"]["holdings"] = new_holdings
        save_config(cfg)

    return f"✅ 已删除 {code}（剩余 {len(new_holdings)} 只持仓）"


def cmd_set_position(args: list[str], cfg: dict) -> str:
    """改总仓位：改成 50%"""
    if len(args) < 1:
        return "❌ 格式：`改成 50%` 或 `改成 50`"

    m = re.search(r"(\d+(?:\.\d+)?)", args[0])
    if not m:
        return f"❌ 解析不了：{args[0]}"
    pct = float(m.group(1))
    if pct < 0 or pct > 100:
        return "❌ 仓位百分比要在 0-100"

    with _config_lock:
        cfg.setdefault("portfolio", {})["total_position_pct"] = pct
        cfg["portfolio"]["cash_pct"] = round(100 - pct, 2)
        save_config(cfg)

    return f"✅ 总仓位改为 {pct}%，现金 {100 - pct}%"


def cmd_risk(_args: list[str], cfg: dict) -> str:
    """显示风险偏好。"""
    risk = cfg.get("risk_profile", {})
    if not risk:
        return "📭 未配置风险偏好（在 config.yaml 里）"
    return (
        "⚖️ **当前风险偏好**\n"
        f"- 单只最大仓位：{risk.get('max_single_position_pct', '?')}%\n"
        f"- 止损线：{risk.get('stop_loss_pct', '?')}%\n"
        f"- 止盈线：{risk.get('take_profit_pct', '?')}%"
    )


def cmd_watchlist(_args: list[str], cfg: dict) -> str:
    """显示关注列表。"""
    wl = cfg.get("watchlist", [])
    if not wl:
        return "📭 关注列表为空\n回复 `关注 600519` 添加（功能待加）"
    return f"👀 **关注列表**：{', '.join(wl)}"


def cmd_help(_args: list[str], _cfg: dict) -> str:
    return (
        "🤖 **可用命令**\n"
        "```\n"
        "持仓                    显示持仓 + 浮盈浮亏\n"
        "添加 600519 1680 100    加/更新持仓\n"
        "删除 600519              删持仓\n"
        "改成 50%                改总仓位\n"
        "风险                    显示风险偏好\n"
        "关注                    显示关注列表\n"
        "基金 股票型 近1年 前10   基金排行\n"
        "潜力股                  量化打分潜力个股\n"
        "涨跌榜                  涨跌幅榜\n"
        "帮助                    这条\n"
        "其它任意文本             LLM 实时分析（自动拉行情）\n"
        "```"
    )


def cmd_llm(text: str, cfg: dict) -> str:
    """其他文本走 LLM 分析。"""
    portfolio = cfg.get("portfolio", {})
    user_ctx = {
        "portfolio": portfolio,
        "watchlist": cfg.get("watchlist", []),
        "risk_profile": cfg.get("risk_profile", {}),
    }
    holdings = [h["symbol"] for h in portfolio.get("holdings", [])]
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


# 命令分发
COMMANDS = {
    "持仓": cmd_portfolio,
    "我的持仓": cmd_portfolio,
    "portfolio": cmd_portfolio,
    "添加": cmd_add,
    "加仓": cmd_add,
    "add": cmd_add,
    "删除": cmd_remove,
    "清仓": cmd_remove,
    "remove": cmd_remove,
    "改成": cmd_set_position,
    "set": cmd_set_position,
    "风险": cmd_risk,
    "关注": cmd_watchlist,
    "watchlist": cmd_watchlist,
    "帮助": cmd_help,
    "help": cmd_help,
    "?": cmd_help,
}


def cmd_fund(args: list[str], cfg: dict) -> str:
    """基金排行：fund 股票型 近1年 前10"""
    # 把整段 text 传过去给 parser
    full_text = " ".join(["fund"] + args) if args else "fund 股票型 近1年 前10"
    params = parse_fund_query(full_text)
    return recommend_funds(**params)


def cmd_potential(_args: list[str], _cfg: dict) -> str:
    """量化打分潜力个股 TOP 10"""
    try:
        df = score_potential_stocks(top_n=10)
    except Exception as exc:
        return f"❌ 拉取数据失败：{exc}"
    title = "🎯 **潜力个股 TOP 10**（动量+量价+形态综合打分）\n\n"
    return title + format_potential_table(df)


def cmd_movers(_args: list[str], _cfg: dict) -> str:
    """涨跌幅榜"""
    from intraday import fetch_top_movers
    try:
        df = fetch_top_movers(top_n=10)
    except Exception as exc:
        return f"❌ 拉取数据失败：{exc}"
    return format_movers_table(df)


# 添加新命令
COMMANDS["基金"] = cmd_fund
COMMANDS["fund"] = cmd_fund
COMMANDS["潜力股"] = cmd_potential
COMMANDS["涨跌榜"] = cmd_movers


def route_command(text: str) -> str:
    """根据文本路由到对应命令。"""
    cfg = load_config()
    stripped = text.strip()
    parts = stripped.split()

    # 精确匹配命令
    if parts and parts[0] in COMMANDS:
        return COMMANDS[parts[0]](parts[1:], cfg)

    # 默认走 LLM
    return cmd_llm(text, cfg)


# ====================================================================== #
#  消息解析
# ====================================================================== #
def parse_message(xml_str: str) -> dict[str, str]:
    """解析企业微信回调消息 XML。"""
    root = ET.fromstring(xml_str)
    return {child.tag: (child.text or "") for child in root}


def extract_user_text(msg: dict[str, str]) -> str:
    """从消息里取出用户实际发的文本（去掉 @ 部分）。"""
    if msg.get("MsgType") != "text":
        return ""
    content = msg.get("Content", "").strip()
    # 去掉开头的 @bot 部分（如果有）
    return re.sub(r"^@[^\s]+\s*", "", content).strip()


# ====================================================================== #
#  Flask app
# ====================================================================== #
def create_app() -> Flask:
    app = Flask(__name__)

    # 关键：先把 .env 加载到 os.environ
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception as exc:
        log.warning(f"load_dotenv failed: {exc}")

    # 加载配置
    token = os.environ.get("WECOM_TOKEN", "")
    aes_key = os.environ.get("WECOM_AES_KEY", "")
    receive_id = os.environ.get("WECOM_CORP_ID", "")

    # 详细诊断：哪个值缺失/异常
    problems = []
    if not token:
        problems.append("WECOM_TOKEN 缺失")
    if not aes_key:
        problems.append("WECOM_AES_KEY 缺失")
    elif len(aes_key) != 43:
        problems.append(f"WECOM_AES_KEY 长度 {len(aes_key)} ≠ 43（多了/少了字符）")
    if not receive_id:
        problems.append("WECOM_CORP_ID 缺失")

    if problems:
        log.warning("WeCom 凭证有问题：")
        for p in problems:
            log.warning(f"  - {p}")
        log.warning("  启动后只能本地测试，企业微信回调会失败。")
        crypto = None
    else:
        try:
            crypto = WeComCrypto(token, aes_key, receive_id)
            log.info("WeCom crypto ready")
        except Exception as exc:
            log.error(f"WeComCrypto init failed: {exc}")
            crypto = None

    pusher = Pusher()

    @app.route("/callback", methods=["GET", "POST"])
    def callback():
        if not crypto:
            return "crypto not configured", 500

        msg_signature = request.args.get("msg_signature", "")
        timestamp = request.args.get("timestamp", "")
        nonce = request.args.get("nonce", "")

        if request.method == "GET":
            # URL 验证：返回解密后的 echostr
            echostr = request.args.get("echostr", "")
            log.info(f"GET /callback: msg_signature={msg_signature[:20]}... ts={timestamp} nonce={nonce} echostr_len={len(echostr)}")

            if not msg_signature or not timestamp or not nonce or not echostr:
                log.warning(f"GET 缺少参数: signature={bool(msg_signature)} ts={bool(timestamp)} nonce={bool(nonce)} echostr={bool(echostr)}")
                return "missing params", 400

            if not crypto.verify_signature(msg_signature, timestamp, nonce, echostr):
                log.warning(f"GET 签名验证失败 - 检查 .env 的 WECOM_TOKEN 是否和企业微信后台「设置 API 接收」里的 Token 一致")
                return "signature invalid", 403

            try:
                # 兼容老接口
                if hasattr(crypto, "decrypt_echo"):
                    decrypted = crypto.decrypt_echo(echostr)
                else:
                    decrypted = crypto.decrypt(echostr)[0]
            except Exception as exc:
                log.error(f"GET 解密失败: {exc}")
                return "decrypt failed", 500

            log.info(f"URL verification OK, returning {len(decrypted)} chars")
            return decrypted

        # POST: 业务消息
        body = request.data.decode("utf-8", errors="ignore")
        log.info(f"received POST, body len={len(body)}")

        # 从 body 里抓 <Encrypt> 字段
        m = re.search(r"<Encrypt><!\[CDATA\[(.*?)\]\]></Encrypt>", body)
        if not m:
            log.warning("no <Encrypt> in body")
            return "success", 200
        encrypted = m.group(1)

        if not crypto.verify_signature(msg_signature, timestamp, nonce, encrypted):
            log.warning("POST signature invalid")
            return "success", 200  # 静默忽略

        try:
            xml_msg, _ = crypto.decrypt(encrypted)
        except Exception as exc:
            log.error(f"decrypt failed: {exc}")
            return "success", 200

        log.info(f"decrypted: {xml_msg[:200]}")

        # 解析消息
        try:
            msg = parse_message(xml_msg)
        except Exception as exc:
            log.error(f"parse failed: {exc}")
            return "success", 200

        user_text = extract_user_text(msg)
        if not user_text:
            log.info(f"non-text message from {msg.get('FromUserName')}: type={msg.get('MsgType')}")
            return "success", 200

        log.info(f"user ({msg.get('FromUserName')}): {user_text}")

        # 路由命令
        try:
            reply = route_command(user_text)
        except Exception as exc:
            log.exception("route_command failed")
            reply = f"❌ 处理失败：{type(exc).__name__}: {exc}"

        # 推到群（通过 webhook，不走加密）
        result = pusher.push_markdown(reply, title="回复")
        log.info(f"push result: {result.get('status')} channel={result.get('channel')}")

        return "success", 200

    @app.route("/healthz")
    def healthz():
        return "ok", 200

    @app.route("/home")
    def home():
        """企业微信应用主页（在 Workspace 里展示）。"""
        cfg = load_config()
        portfolio = cfg.get("portfolio", {})
        holdings = portfolio.get("holdings", [])
        risk = cfg.get("risk_profile", {})

        holdings_html = ""
        if holdings:
            items = "".join(
                f"<li><code>{h.get('symbol', '?')}</code> "
                f"成本 {h.get('cost', 0)} × {int(h.get('shares', 0))} 股</li>"
                for h in holdings
            )
            holdings_html = f"<ul>{items}</ul>"
        else:
            holdings_html = "<p><em>暂无持仓</em></p>"

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>A 股分析助手</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif;
    max-width: 720px;
    margin: 40px auto;
    padding: 0 20px;
    color: #1a1a1a;
    line-height: 1.6;
  }}
  h1 {{ color: #06ad56; border-bottom: 2px solid #06ad56; padding-bottom: 8px; }}
  h2 {{ color: #333; margin-top: 32px; }}
  code {{
    background: #f4f4f4;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: "SF Mono", Menlo, monospace;
    font-size: 0.9em;
  }}
  pre {{
    background: #f8f8f8;
    border: 1px solid #e1e1e1;
    border-radius: 6px;
    padding: 12px 16px;
    overflow-x: auto;
    font-size: 0.9em;
  }}
  .meta {{
    color: #888;
    font-size: 0.9em;
  }}
  .warning {{
    background: #fff7e6;
    border-left: 4px solid #faad14;
    padding: 12px 16px;
    margin: 20px 0;
    border-radius: 4px;
  }}
  ul li {{ margin: 4px 0; }}
</style>
</head>
<body>
<h1>📈 A 股分析助手</h1>
<p class="meta">DeepSeek + 实时行情数据 · 每天 11:30 / 14:30 自动推送</p>

<h2>在群里这样用</h2>
<p>在企业微信群里 <code>@本应用名</code>，后接你的问题或命令。例：</p>
<pre>@本应用名 京东方现在能加仓吗？
@本应用名 持仓
@本应用名 添加 000725 7.0 1000</pre>

<h2>支持命令</h2>
<pre>持仓              显示当前持仓 + 浮盈浮亏
添加 600519 1680 100   加/更新持仓
删除 600519        删持仓
改成 50%           改总仓位
风险              显示风险偏好
关注              显示关注列表
帮助              命令列表
其它任意文本       LLM 实时分析（自动拉行情）</pre>

<h2>当前状态</h2>
<p><strong>总仓位：</strong> {portfolio.get('total_position_pct', '?')}% &nbsp;|&nbsp;
   <strong>现金：</strong> {portfolio.get('cash_pct', '?')}%</p>
<p><strong>持仓：</strong></p>
{holdings_html}

<h2>风险偏好</h2>
<p>单只最大仓位 {risk.get('max_single_position_pct', '?')}% &nbsp;|&nbsp;
   止损 {risk.get('stop_loss_pct', '?')}% &nbsp;|&nbsp;
   止盈 {risk.get('take_profit_pct', '?')}%</p>

<div class="warning">
⚠️ <strong>风险提示：</strong>本工具输出仅供个人参考，<strong>不构成投资建议</strong>。投资有风险，决策需谨慎。
</div>

<p class="meta">API：DeepSeek · 行情：腾讯财经 + 同花顺 · 推送：企业微信群机器人</p>
</body>
</html>"""
        return html

    return app


# ====================================================================== #
#  入口
# ====================================================================== #
def main():
    import argparse
    parser = argparse.ArgumentParser(description="A 股分析助手 - 企业微信回调服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    app = create_app()
    log.info(f"webhook server starting on {args.host}:{args.port}")
    log.info(f"  callback URL: http://{args.host}:{args.port}/callback")
    log.info("  （用 ngrok 暴露后，填到企业微信后台「接收消息」）")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()