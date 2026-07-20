"""
交互式 CLI —— 用户随时手动提问。

Usage:
    python interactive.py "现在半导体板块怎么看？"
    python interactive.py "600519 现在能加仓吗？"
    python interactive.py "今天尾盘要不要减仓平安银行？"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from intraday import IntradayClient
from llm_agent import LLMAgent
from wechat_push import WeComPusher


CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_user_ctx() -> dict:
    """从 config.yaml 加载用户上下文。"""
    if not CONFIG_PATH.exists():
        return {"portfolio": {}, "watchlist": [], "risk_profile": {}}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {
        "portfolio": cfg.get("portfolio", {}),
        "watchlist": cfg.get("watchlist", []),
        "risk_profile": cfg.get("risk_profile", {}),
    }


def build_user_ctx_for_agent(cfg: dict) -> dict:
    """转换 config 格式给 agent 用。"""
    holdings_symbols = [
        h["symbol"] for h in cfg["portfolio"].get("holdings", [])
    ]
    return {
        "portfolio": cfg["portfolio"],
        "watchlist": cfg.get("watchlist", []),
        "risk_profile": cfg.get("risk_profile", {}),
        "_holdings_symbols": holdings_symbols,
    }


def handle_query(user_query: str, push: bool = False) -> str:
    cfg = load_user_ctx()
    user_ctx = build_user_ctx_for_agent(cfg)

    holdings = user_ctx.pop("_holdings_symbols", [])
    watchlist = user_ctx["watchlist"]

    print(f"[*] 拉取实时市场数据 ...", file=sys.stderr)
    ic = IntradayClient()
    snapshot = ic.build_market_snapshot(
        watchlist=watchlist,
        holdings=holdings,
    )

    print(f"[*] 调用 LLM 推理 ...", file=sys.stderr)
    agent = LLMAgent()
    text = agent.analyze_interactive(user_query, snapshot, user_ctx)

    print(text)
    print()

    if push:
        pusher = WeComPusher()
        result = pusher.push_markdown(text, title=f"问答_{user_query[:20]}")
        print(f"\n[push result] {result['status']}", file=sys.stderr)

    return text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python interactive.py <你的问题>")
        print('示例：python interactive.py "现在半导体板块怎么看？"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    handle_query(query, push=False)