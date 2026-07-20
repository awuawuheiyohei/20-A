"""
多渠道消息推送 —— 支持 5 种通道，按优先级自动选用：

  1. PushPlus       —— 推个人微信（公众号），200 条/天免费
  2. WxPusher       —— 推个人微信（公众号），无条数限制
  3. Server酱       —— 推个人微信（公众号），5 条/天免费
  4. 企业微信群机器人 webhook
  5. 企业微信自建应用 API（需 IP 白名单）

任选一个配到 .env 即可。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests

# API 端点
PUSHPLUS_URL = "https://www.pushplus.plus/send"
WXPUSHER_URL = "https://wxpusher.zjiecode.com/api/send/message"
SERVERCHAN_URL = "https://sct.ftqq.com"
WEBHOOK_URL_PATTERN = "qyapi.weixin.qq.com/cgi-bin/webhook/send"
ACCESS_TOKEN_URL = "https://qyapi.weixin.qq.com/cgi-bin/gettoken"
SEND_MSG_URL = "https://qyapi.weixin.qq.com/cgi-bin/message/send"


@dataclass
class Pusher:
    """
    消息推送器。

    优先级（自动按顺序选用）：
        1. PushPlus       — PUSHPLUS_TOKEN
        2. WxPusher       — WXPUSHER_APP_TOKEN + WXPUSHER_UID
        3. Server酱       — SERVERCHAN_SENDKEY
        4. 企业微信 webhook — WECOM_WEBHOOK_URL
        5. 企业微信 app    — WECOM_CORP_ID / WECOM_CORP_SECRET / WECOM_AGENT_ID

    全部未配时降级为本地存档。
    """

    pushplus_token: str | None = None
    wxpusher_app_token: str | None = None
    wxpusher_uid: str | None = None
    serverchan_sendkey: str | None = None
    webhook_url: str | None = None
    corp_id: str | None = None
    corp_secret: str | None = None
    agent_id: str | None = None
    to_user: str = "@all"
    save_dir: str = str(Path(__file__).parent / "reports")
    _token: str | None = None
    _token_expires_at: float = 0.0

    def __post_init__(self):
        env_keys = {
            "pushplus_token": "PUSHPLUS_TOKEN",
            "wxpusher_app_token": "WXPUSHER_APP_TOKEN",
            "wxpusher_uid": "WXPUSHER_UID",
            "serverchan_sendkey": "SERVERCHAN_SENDKEY",
            "webhook_url": "WECOM_WEBHOOK_URL",
            "corp_id": "WECOM_CORP_ID",
            "corp_secret": "WECOM_CORP_SECRET",
            "agent_id": "WECOM_AGENT_ID",
            "to_user": "WECOM_TO_USER",
        }
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass
        for attr, env in env_keys.items():
            if not getattr(self, attr):
                setattr(self, attr, os.environ.get(env))

    # ================================================================== #
    #  配置探测
    # ================================================================== #
    @property
    def pushplus_configured(self) -> bool:
        return bool(self.pushplus_token)

    @property
    def wxpusher_configured(self) -> bool:
        return bool(self.wxpusher_app_token and self.wxpusher_uid)

    @property
    def serverchan_configured(self) -> bool:
        return bool(self.serverchan_sendkey)

    @property
    def webhook_configured(self) -> bool:
        return bool(self.webhook_url) and WEBHOOK_URL_PATTERN in (self.webhook_url or "")

    @property
    def app_configured(self) -> bool:
        return bool(self.corp_id and self.corp_secret and self.agent_id)

    @property
    def is_configured(self) -> bool:
        return any([
            self.pushplus_configured,
            self.wxpusher_configured,
            self.serverchan_configured,
            self.webhook_configured,
            self.app_configured,
        ])

    # ================================================================== #
    #  主入口
    # ================================================================== #
    def push_markdown(self, content: str, title: str = "A股分析") -> dict:
        """推送 Markdown。"""
        saved = self._save_locally(title, content)

        if self.pushplus_configured:
            return self._push_pushplus(title, content, saved)
        if self.wxpusher_configured:
            return self._push_wxpusher(title, content, saved)
        if self.serverchan_configured:
            return self._push_serverchan(title, content, saved)
        if self.webhook_configured:
            return self._push_webhook(content, saved)
        if self.app_configured:
            return self._push_app(content, saved)
        print(f"[push] 未配置任何推送通道，仅本地保存：{saved}")
        return {"status": "saved_only", "path": saved}

    def push_text(self, content: str, title: str = "A股分析") -> dict:
        saved = self._save_locally(title, content)
        if self.pushplus_configured:
            return self._push_pushplus_text(content, saved, title)
        if self.wxpusher_configured:
            return self._push_wxpusher_text(content, saved, title)
        if self.serverchan_configured:
            return self._push_serverchan_text(content, saved, title)
        if self.webhook_configured:
            return self._push_webhook_text(content, saved)
        if self.app_configured:
            return self._push_app_text(content, saved)
        return {"status": "saved_only", "path": saved}

    # ================================================================== #
    #  通道 1：PushPlus
    # ================================================================== #
    def _push_pushplus(self, title: str, content: str, saved: str) -> dict:
        try:
            payload = {
                "token": self.pushplus_token,
                "title": title[:100],
                "content": content,
                "template": "markdown",
            }
            r = requests.post(PUSHPLUS_URL, json=payload, timeout=10)
            resp = r.json()
            if resp.get("code") == 200:
                return {"status": "sent", "channel": "pushplus", "path": saved}
            return {
                "status": "failed", "channel": "pushplus",
                "error": resp.get("msg", resp.get("message", "?")),
                "path": saved,
            }
        except Exception as exc:
            return {"status": "failed", "channel": "pushplus",
                    "error": f"{type(exc).__name__}: {exc}", "path": saved}

    def _push_pushplus_text(self, content: str, saved: str, title: str) -> dict:
        try:
            payload = {
                "token": self.pushplus_token,
                "title": title[:100],
                "content": content,
                "template": "txt",
            }
            r = requests.post(PUSHPLUS_URL, json=payload, timeout=10)
            resp = r.json()
            if resp.get("code") == 200:
                return {"status": "sent", "channel": "pushplus", "path": saved}
            return {"status": "failed", "channel": "pushplus",
                    "error": resp.get("msg", "?"), "path": saved}
        except Exception as exc:
            return {"status": "failed", "channel": "pushplus",
                    "error": str(exc), "path": saved}

    # ================================================================== #
    #  通道 2：WxPusher
    # ================================================================== #
    def _push_wxpusher(self, title: str, content: str, saved: str) -> dict:
        try:
            payload = {
                "appToken": self.wxpusher_app_token,
                "content": content,
                "summary": title[:100],
                "contentType": 3,   # 3 = Markdown
                "uids": [self.wxpusher_uid],
            }
            r = requests.post(WXPUSHER_URL, json=payload, timeout=10)
            resp = r.json()
            if resp.get("code") == 1000:
                return {"status": "sent", "channel": "wxpusher", "path": saved}
            return {
                "status": "failed", "channel": "wxpusher",
                "errcode": resp.get("code"),
                "error": resp.get("msg", "?"),
                "path": saved,
            }
        except Exception as exc:
            return {"status": "failed", "channel": "wxpusher",
                    "error": f"{type(exc).__name__}: {exc}", "path": saved}

    def _push_wxpusher_text(self, content: str, saved: str, title: str) -> dict:
        try:
            payload = {
                "appToken": self.wxpusher_app_token,
                "content": content,
                "summary": title[:100],
                "contentType": 1,   # 1 = 纯文本
                "uids": [self.wxpusher_uid],
            }
            r = requests.post(WXPUSHER_URL, json=payload, timeout=10)
            resp = r.json()
            if resp.get("code") == 1000:
                return {"status": "sent", "channel": "wxpusher", "path": saved}
            return {"status": "failed", "channel": "wxpusher",
                    "error": resp.get("msg", "?"), "path": saved}
        except Exception as exc:
            return {"status": "failed", "channel": "wxpusher",
                    "error": str(exc), "path": saved}

    # ================================================================== #
    #  通道 3：Server酱
    # ================================================================== #
    def _push_serverchan(self, title: str, content: str, saved: str) -> dict:
        try:
            url = f"{SERVERCHAN_URL}/{self.serverchan_sendkey}.send"
            payload = {"title": title[:100], "desp": content}
            r = requests.post(url, data=payload, timeout=10)
            # 先看状态码和响应头，区分"网络/鉴权/限流"错误
            if r.status_code != 200:
                return {
                    "status": "failed", "channel": "serverchan",
                    "error": f"HTTP {r.status_code}: {r.text[:200]}",
                    "path": saved,
                }
            try:
                resp = r.json()
            except ValueError:
                return {
                    "status": "failed", "channel": "serverchan",
                    "error": f"非 JSON 响应: {r.text[:200]}",
                    "path": saved,
                }
            if resp.get("code") == 0:
                return {"status": "sent", "channel": "serverchan", "path": saved}
            return {"status": "failed", "channel": "serverchan",
                    "error": resp.get("message", "?"), "path": saved}
        except Exception as exc:
            return {"status": "failed", "channel": "serverchan",
                    "error": f"{type(exc).__name__}: {exc}", "path": saved}

    def _push_serverchan_text(self, content: str, saved: str, title: str) -> dict:
        return self._push_serverchan(title, content, saved)

    # ================================================================== #
    #  通道 4：企业微信群机器人 webhook
    # ================================================================== #
    def _push_webhook(self, content: str, saved: str) -> dict:
        try:
            payload = {"msgtype": "markdown", "markdown": {"content": content}}
            r = requests.post(self.webhook_url, json=payload, timeout=10)
            resp = r.json()
            if resp.get("errcode") == 0:
                return {"status": "sent", "channel": "webhook", "path": saved}
            return {"status": "failed", "channel": "webhook",
                    "error": resp.get("errmsg", "?"), "path": saved}
        except Exception as exc:
            return {"status": "failed", "channel": "webhook",
                    "error": str(exc), "path": saved}

    def _push_webhook_text(self, content: str, saved: str) -> dict:
        try:
            payload = {"msgtype": "text", "text": {"content": content}}
            r = requests.post(self.webhook_url, json=payload, timeout=10)
            resp = r.json()
            if resp.get("errcode") == 0:
                return {"status": "sent", "channel": "webhook", "path": saved}
            return {"status": "failed", "channel": "webhook",
                    "error": resp.get("errmsg", "?"), "path": saved}
        except Exception as exc:
            return {"status": "failed", "channel": "webhook",
                    "error": str(exc), "path": saved}

    # ================================================================== #
    #  通道 5：企业微信自建应用 API
    # ================================================================== #
    def _get_token(self) -> str:
        if self._token and time.time() < self._token_expires_at - 60:
            return self._token
        r = requests.get(
            ACCESS_TOKEN_URL,
            params={"corpid": self.corp_id, "corpsecret": self.corp_secret},
            timeout=10,
        )
        data = r.json()
        if data.get("errcode") != 0:
            raise RuntimeError(f"WeCom token error: {data}")
        self._token = data["access_token"]
        self._token_expires_at = time.time() + data["expires_in"]
        return self._token

    def _push_app(self, content: str, saved: str) -> dict:
        try:
            token = self._get_token()
            payload = {
                "touser": self.to_user,
                "msgtype": "markdown",
                "agentid": int(self.agent_id),
                "markdown": {"content": content},
            }
            r = requests.post(SEND_MSG_URL, params={"access_token": token},
                              json=payload, timeout=10)
            resp = r.json()
            if resp.get("errcode") == 0:
                return {"status": "sent", "channel": "app", "path": saved, "response": resp}
            errcode = resp.get("errcode")
            errmsg = resp.get("errmsg", "unknown")
            if errcode == 60020:
                errmsg += "\n  → IP 不在白名单。改用 PushPlus/WxPusher/Server酱 绕过"
            return {"status": "failed", "channel": "app", "errcode": errcode,
                    "error": errmsg, "path": saved}
        except Exception as exc:
            return {"status": "failed", "channel": "app",
                    "error": f"{type(exc).__name__}: {exc}", "path": saved}

    def _push_app_text(self, content: str, saved: str) -> dict:
        try:
            token = self._get_token()
            payload = {
                "touser": self.to_user,
                "msgtype": "text",
                "agentid": int(self.agent_id),
                "text": {"content": content},
            }
            r = requests.post(SEND_MSG_URL, params={"access_token": token},
                              json=payload, timeout=10)
            resp = r.json()
            if resp.get("errcode") == 0:
                return {"status": "sent", "channel": "app", "path": saved, "response": resp}
            return {"status": "failed", "channel": "app",
                    "error": resp.get("errmsg", "?"), "path": saved}
        except Exception as exc:
            return {"status": "failed", "channel": "app",
                    "error": str(exc), "path": saved}

    # ================================================================== #
    #  本地存档（始终执行）
    # ================================================================== #
    def _save_locally(self, title: str, content: str) -> str:
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(self.save_dir) / f"{ts}_{title}.md"
        path.write_text(content, encoding="utf-8")
        return str(path)


# 兼容旧名
WeComPusher = Pusher


# ---------------------------------------------------------------------- #
#  Self-test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    p = Pusher()
    print("=" * 50)
    print("已配置的推送通道：")
    print(f"  PushPlus:    {'✓' if p.pushplus_configured else '✗'}")
    print(f"  WxPusher:    {'✓' if p.wxpusher_configured else '✗'}")
    print(f"  Server酱:    {'✓' if p.serverchan_configured else '✗'}")
    print(f"  Webhook:     {'✓' if p.webhook_configured else '✗'}")
    print(f"  App API:     {'✓' if p.app_configured else '✗'}")
    print("=" * 50)

    if p.is_configured:
        result = p.push_markdown(
            "# 测试推送\n\n这是一条来自 A 股分析助手的测试消息。\n\n收到请回复 OK",
            title="测试"
        )
        print("推送结果：")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("未配置任何推送通道。")
        print()
        print("推荐三个个人微信通道，按【易用度】排序：")
        print()
        print("  A. Server酱（5 条/天，5 分钟配好）")
        print("     1. 打开 https://sct.ftqq.com/")
        print("     2. 微信扫码登录，拿到 SendKey")
        print("     3. 关注公众号「方糖」")
        print("     4. .env: SERVERCHAN_SENDKEY=你的key")
        print()
        print("  B. WxPusher（无条数限制）")
        print("     1. 打开 https://wxpusher.zjiecode.com/")
        print("     2. 微信扫码登录 → 创建应用 → 拿 appToken")
        print("     3. 用户管理 → 复制自己的 uid")
        print("     4. 关注公众号「WxPusher 服务通知」")
        print("     5. .env: WXPUSHER_APP_TOKEN + WXPUSHER_UID")
        print()
        print("  C. PushPlus（200 条/天）")
        print("     1. 打开 https://www.pushplus.plus/")
        print("     2. 微信扫码登录，复制 token")
        print("     3. 关注公众号「pushplus 推送加」")
        print("     4. .env: PUSHPLUS_TOKEN=你的token")
        print()
        p.push_markdown("# 测试\n\n未配置推送通道", title="测试")