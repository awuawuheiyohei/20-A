"""
小工具：检测当前公网 IP，用于企业微信 IP 白名单配置。

用法：
    python check_ip.py            # 打印当前 IP
    python check_ip.py --check    # 和 .env 里 WECOM_REGISTERED_IP 对比

IP 变化时只需要：
  1. 把这个脚本输出的 IP 复制
  2. 粘到企业微信后台「应用管理 → 自建应用 → 企业可信 IP」
  3. 重新跑 python wechat_push.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests

ENV_PATH = Path(__file__).parent / ".env"

# 多个公网 IP 查询服务，挑一个能用的
IP_CHECK_SERVICES = [
    "https://api.ipify.org?format=json",
    "https://ifconfig.me/all.json",
    "https://ipinfo.io/json",
]


def get_public_ip() -> str | None:
    """查询当前公网 IP。多个服务都试一遍。"""
    for url in IP_CHECK_SERVICES:
        try:
            r = requests.get(url, timeout=8)
            r.raise_for_status()
            data = r.json()
            for key in ("ip", "origin", "ip_addr"):
                if key in data:
                    return data[key].strip()
        except Exception:
            continue
    return None


def read_registered_ip() -> str | None:
    """从 .env 读 WECOM_REGISTERED_IP（你自己记下来的上次加的 IP）。"""
    if not ENV_PATH.exists():
        return None
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("WECOM_REGISTERED_IP="):
            return line.split("=", 1)[1].strip()
    return None


def main():
    parser = argparse.ArgumentParser(description="检测公网 IP")
    parser.add_argument("--check", action="store_true",
                        help="和 .env 中记录的 IP 对比")
    args = parser.parse_args()

    print("[*] 查询当前公网 IP ...")
    ip = get_public_ip()

    if not ip:
        print("❌ 所有 IP 查询服务都失败", file=sys.stderr)
        sys.exit(1)

    print(f"[ok] 当前公网 IP：{ip}")

    if args.check:
        registered = read_registered_ip()
        if not registered:
            print("[!] .env 里没记 WECOM_REGISTERED_IP（建议加上方便对比）")
        elif registered == ip:
            print(f"[ok] IP 没变（{registered}），白名单应该还是有效的")
        else:
            print(f"[!] IP 变了！之前是 {registered}，现在是 {ip}")
            print()
            print("    需要更新企业微信 IP 白名单：")
            print("    1. 登录 https://work.weixin.qq.com/wework_admin/")
            print("    2. 应用管理 → 自建应用 → 你的应用")
            print("    3. 找到「企业可信 IP」配置")
            print(f"    4. 把 {ip} 加进去（旧的 {registered} 可以删）")
            print()
            print("    也可以把新 IP 记到 .env，下次方便对比：")
            print(f'    WECOM_REGISTERED_IP={ip}')


if __name__ == "__main__":
    main()