"""
主入口 —— 把所有模块串起来。

Usage:
    # 启动调度器（每天 11:30 / 14:30 自动跑）
    python main.py serve

    # 手动立刻跑一次上午复盘
    python main.py morning

    # 手动立刻跑一次尾盘决策
    python main.py afternoon

    # 手动 query
    python main.py ask "现在半导体板块怎么看？"

    # 立刻跑一次并推到微信（手动验证完整 pipeline）
    python main.py test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

from intraday import IntradayClient, extract_symbols
from llm_agent import LLMAgent
from triggers import TradingDayScheduler
from wechat_push import WeComPusher


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("main")


CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        log.warning(f"config.yaml 不存在 ({CONFIG_PATH})，使用空配置")
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_user_ctx(cfg: dict) -> dict:
    """从 config.yaml 构建 agent 用的上下文。"""
    portfolio = cfg.get("portfolio", {})
    holdings_symbols = [h["symbol"] for h in portfolio.get("holdings", [])]
    return {
        "portfolio": portfolio,
        "watchlist": cfg.get("watchlist", []),
        "risk_profile": cfg.get("risk_profile", {}),
        "_holdings_symbols": holdings_symbols,
    }


def run_morning_close() -> str:
    """上午收盘触发。"""
    log.info("=" * 50)
    log.info("执行：上午复盘")
    log.info("=" * 50)

    cfg = load_config()
    user_ctx = build_user_ctx(cfg)
    holdings = user_ctx.pop("_holdings_symbols", [])
    watchlist = user_ctx["watchlist"]

    snapshot = IntradayClient().build_market_snapshot(
        watchlist=watchlist,
        holdings=holdings,
    )
    text = LLMAgent().analyze_morning_close(snapshot, user_ctx)
    log.info(f"LLM 输出 ({len(text)} 字)")

    pusher = WeComPusher()
    result = pusher.push_markdown(text, title="上午复盘")
    log.info(f"推送结果：{result['status']}")
    return text


def run_afternoon_close() -> str:
    """尾盘触发。"""
    log.info("=" * 50)
    log.info("执行：尾盘决策")
    log.info("=" * 50)

    cfg = load_config()
    user_ctx = build_user_ctx(cfg)
    holdings = user_ctx.pop("_holdings_symbols", [])
    watchlist = user_ctx["watchlist"]

    snapshot = IntradayClient().build_market_snapshot(
        watchlist=watchlist,
        holdings=holdings,
    )
    text = LLMAgent().analyze_afternoon_close(snapshot, user_ctx)
    log.info(f"LLM 输出 ({len(text)} 字)")

    pusher = WeComPusher()
    result = pusher.push_markdown(text, title="尾盘决策")
    log.info(f"推送结果：{result['status']}")
    return text


def run_ask(query: str) -> str:
    cfg = load_config()
    user_ctx = build_user_ctx(cfg)
    holdings = user_ctx.pop("_holdings_symbols", [])
    watchlist = user_ctx["watchlist"]

    snapshot = IntradayClient().build_market_snapshot(
        watchlist=watchlist,
        holdings=holdings,
    )
    text = LLMAgent().analyze_interactive(query, snapshot, user_ctx)
    print(text)
    return text


def cmd_serve(args) -> None:
    """启动调度器，阻塞运行。"""
    cfg = load_config()
    triggers_cfg = cfg.get("triggers", {})
    morning = triggers_cfg.get("morning_close", "11:30")
    afternoon = triggers_cfg.get("afternoon_close", "14:30")

    sched = TradingDayScheduler(
        morning_time=morning,
        afternoon_time=afternoon,
    )
    sched.add_morning_close(run_morning_close)
    sched.add_afternoon_close(run_afternoon_close)

    log.info("调度器启动：")
    for j in sched.list_jobs():
        log.info(f"  - {j['name']}")
    log.info("按 Ctrl+C 退出。")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("调度器退出。")


def cmd_morning(args) -> None:
    text = run_morning_close()
    print()
    print("=" * 50)
    print(text)


def cmd_afternoon(args) -> None:
    text = run_afternoon_close()
    print()
    print("=" * 50)
    print(text)


def cmd_ask(args) -> None:
    query = " ".join(args.query)
    if not query:
        print("请提供问题，例如：python main.py ask \"600519 现在能加仓吗？\"")
        sys.exit(1)
    run_ask(query)


def cmd_test(args) -> None:
    """立刻跑 morning + afternoon + 显示推送状态。"""
    log.info("=== 测试完整 pipeline ===")
    morning_text = run_morning_close()
    print(morning_text)
    print()
    afternoon_text = run_afternoon_close()
    print(afternoon_text)


def cmd_chat(args) -> None:
    """多轮对话 REPL。"""
    print("=" * 50)
    print("  A 股分析助手 - 交互对话模式")
    print("  - 输入 'exit' / 'quit' / Ctrl+C 退出")
    print("  - 输入 'clear' 清空对话历史")
    print("  - 输入 'history' 看对话历史条数")
    print("=" * 50)
    print()

    cfg = load_config()
    user_ctx = build_user_ctx(cfg)
    holdings = user_ctx.pop("_holdings_symbols", [])
    watchlist = user_ctx["watchlist"]

    agent = LLMAgent()
    history: list[dict[str, str]] = []

    while True:
        try:
            user_input = input("你 > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见。")
            break

        if not user_input:
            continue
        # 命令识别（先剥离空白，宽松匹配）
        normalized = user_input.strip().lower()
        if normalized in ("exit", "quit", "q", "退出", "再见"):
            print("再见。")
            break
        if normalized in ("clear", "清空", "清空历史"):
            history = []
            print("(历史已清空)")
            continue
        if normalized in ("history", "历史"):
            n_user = sum(1 for h in history if h["role"] == "user")
            n_assist = sum(1 for h in history if h["role"] == "assistant")
            print(f"(已记录 {n_user} 轮 user / {n_assist} 轮 assistant)")
            continue

        # 拉最新行情（包含用户问题中提到的股票）
        try:
            mentioned = extract_symbols(user_input)
            snapshot = IntradayClient().build_market_snapshot(
                watchlist=watchlist,
                holdings=holdings,
                mentioned=mentioned,
            )
            if mentioned:
                print(f"[*] 自动补充数据: {mentioned}")
        except Exception as exc:
            print(f"[warn] 拉取数据失败: {exc}")
            snapshot = {}

        # 调 LLM
        print("[*] 思考中 ...", end="\r")
        text = agent.chat_turn(user_input, history, snapshot, user_ctx)

        # 更新历史
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": text})

        # 输出
        print(" " * 20, end="\r")  # 清掉"思考中"
        print()
        print(text)
        print()


def cmd_webhook(args) -> None:
    """启动 Flask 回调服务（企业微信双向对话）。"""
    from webhook_server import main as webhook_main
    # 直接转给 webhook_server.main 处理参数
    import sys
    sys.argv = ["webhook_server", "--host", args.host, "--port", str(args.port)]
    webhook_main()


def cmd_all(args) -> None:
    """
    一个进程同时跑：Flask webhook（双向对话）+ APScheduler（11:30/14:30 推送）。
    比单独开 'serve' + 'webhook' 省一个终端。
    注意：cloudflared 还是要单独跑（外部进程）。
    """
    from threading import Thread
    from webhook_server import create_app

    # 1. 启动 Flask（后台线程）
    app = create_app()
    flask_thread = Thread(
        target=lambda: app.run(
            host=args.host, port=args.port,
            debug=False, use_reloader=False,
        ),
        daemon=True,
        name="flask-webhook",
    )
    flask_thread.start()
    log.info(f"Flask webhook started on http://{args.host}:{args.port} (后台线程)")

    # 2. 启动调度器（主线程，阻塞）
    cfg = load_config()
    triggers_cfg = cfg.get("triggers", {})
    morning = triggers_cfg.get("morning_close", "11:30")
    afternoon = triggers_cfg.get("afternoon_close", "14:30")

    sched = TradingDayScheduler(
        morning_time=morning,
        afternoon_time=afternoon,
    )
    sched.add_morning_close(run_morning_close)
    sched.add_afternoon_close(run_afternoon_close)

    log.info("调度器启动：")
    for j in sched.list_jobs():
        log.info(f"  - {j['name']}")
    log.info(f"下次触发：11:30 / 14:30（工作日）")
    log.info("按 Ctrl+C 停止所有服务。")

    try:
        sched.start()  # 阻塞
    except (KeyboardInterrupt, SystemExit):
        log.info("调度器退出，Flask 也会跟着关。")


def cmd_bot(args) -> None:
    """智能机器人长连接模式（推荐）。"""
    from smart_bot import main as bot_main
    import sys
    sys.argv = ["smart_bot"]
    bot_main()


def cmd_all_bot(args) -> None:
    """智能机器人 + 调度器，一个进程搞定。"""
    import threading
    import asyncio
    from smart_bot import SmartBot
    from dotenv import load_dotenv
    load_dotenv()

    # 1. 启动智能机器人（后台线程，跑 asyncio 事件循环）
    bot_id = os.environ.get("WECOM_SMART_BOT_ID", "")
    secret = os.environ.get("WECOM_SMART_BOT_SECRET", "")
    if not bot_id or not secret:
        log.error("WECOM_SMART_BOT_ID / WECOM_SMART_BOT_SECRET 未配置")
        return

    def run_bot():
        asyncio.run(SmartBot(bot_id, secret).start())

    bot_thread = threading.Thread(target=run_bot, daemon=True, name="smart-bot")
    bot_thread.start()
    log.info(f"智能机器人已启动：{bot_id[:12]}...")

    # 2. 启动调度器（主线程，阻塞）
    cfg = load_config()
    triggers_cfg = cfg.get("triggers", {})
    morning = triggers_cfg.get("morning_close", "11:30")
    afternoon = triggers_cfg.get("afternoon_close", "14:30")

    sched = TradingDayScheduler(
        morning_time=morning,
        afternoon_time=afternoon,
    )
    sched.add_morning_close(run_morning_close)
    sched.add_afternoon_close(run_afternoon_close)

    log.info("调度器启动：")
    for j in sched.list_jobs():
        log.info(f"  - {j['name']}")
    log.info("按 Ctrl+C 停止所有服务。")

    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("调度器退出，智能机器人也会跟着关。")


def main():
    parser = argparse.ArgumentParser(description="A 股实时分析助手")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("serve", help="启动调度器（阻塞）")
    sub.add_parser("morning", help="手动跑一次上午复盘")
    sub.add_parser("afternoon", help="手动跑一次尾盘决策")
    sub.add_parser("test", help="测试：跑两次并显示输出")
    sub.add_parser("chat", help="多轮对话 REPL")

    webhook_p = sub.add_parser("webhook", help="启动 Flask 回调服务（企业微信双向对话）")
    webhook_p.add_argument("--host", default="0.0.0.0")
    webhook_p.add_argument("--port", type=int, default=5050, help="默认 5050（macOS AirPlay 占 5000）")

    all_p = sub.add_parser("all", help="一个进程同时跑 webhook + 调度器（推荐）")
    all_p.add_argument("--host", default="0.0.0.0")
    all_p.add_argument("--port", type=int, default=5050)

    bot_p = sub.add_parser("bot", help="智能机器人长连接模式（无需公网 URL）")

    all_bot_p = sub.add_parser("all-bot", help="智能机器人 + 调度器（推荐）")

    ask_p = sub.add_parser("ask", help="手动 query（一次性）")
    ask_p.add_argument("query", nargs="+", help="问题文本")

    args = parser.parse_args()

    dispatch = {
        "serve": cmd_serve,
        "morning": cmd_morning,
        "afternoon": cmd_afternoon,
        "test": cmd_test,
        "ask": cmd_ask,
        "chat": cmd_chat,
        "webhook": cmd_webhook,
        "all": cmd_all,
        "bot": cmd_bot,
        "all-bot": cmd_all_bot,
    }
    handler = dispatch.get(args.cmd)
    if not handler:
        parser.print_help()
        sys.exit(1)
    handler(args)


if __name__ == "__main__":
    main()