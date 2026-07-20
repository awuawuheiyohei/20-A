"""
企业微信回调消息加解密（基于官方 AES-256-CBC + PKCS#7 协议）。

文档：https://developer.work.weixin.qq.com/document/path/90968

只做加解密和签名验证，**不**做 XML 解析（让上层处理）。
"""

from __future__ import annotations

import base64
import hashlib
import struct
from typing import Tuple

from Crypto.Cipher import AES


class WeComCrypto:
    """
    企业微信回调消息加解密器。

    三要素：
        token          —— 在企业微信后台「接收消息 → 设置 API 接收」配置
        encoding_aes_key —— 同上，43 位字符串
        receive_id     —— 一般是 corp_id（自建应用回调场景）
    """

    BLOCK_SIZE = 32

    def __init__(self, token: str, encoding_aes_key: str, receive_id: str):
        if not token:
            raise ValueError("token is required")
        if not encoding_aes_key or len(encoding_aes_key) != 43:
            raise ValueError("encoding_aes_key must be 43 chars")
        if not receive_id:
            raise ValueError("receive_id is required")

        self.token = token
        self.receive_id = receive_id
        # Key = base64decode(aes_key + "=")
        self.key = base64.b64decode(encoding_aes_key + "=")
        if len(self.key) != 32:
            raise ValueError("decoded AES key must be 32 bytes")

    # ================================================================== #
    #  签名验证
    # ================================================================== #
    def verify_signature(self, signature: str, timestamp: str,
                         nonce: str, encrypt: str) -> bool:
        """验证 msg_signature。

        算法：sha1(sort([token, timestamp, nonce, encrypt]).join(''))
        """
        items = sorted([self.token, timestamp, nonce, encrypt])
        expected = hashlib.sha1("".join(items).encode("utf-8")).hexdigest()
        return expected == signature

    # ================================================================== #
    #  解密（echostr 或 业务消息）
    # ================================================================== #
    def decrypt(self, encrypted_b64: str) -> Tuple[str, str]:
        """
        解密 base64 编码的 AES 密文。

        Returns
        -------
        (msg, receive_id)
            msg: 业务消息 XML 字符串
            receive_id: 校验用的 receive_id（应该等于 self.receive_id）
        """
        try:
            iv = self.key[:16]
            cipher = AES.new(self.key, AES.MODE_CBC, iv)
            ciphertext = base64.b64decode(encrypted_b64)
            plaintext = cipher.decrypt(ciphertext)

            # 去 PKCS#7 padding
            pad = plaintext[-1]
            if isinstance(pad, str):
                pad = ord(pad)
            if pad < 1 or pad > self.BLOCK_SIZE:
                raise ValueError(f"invalid padding: {pad}")
            plaintext = plaintext[:-pad]

            # 结构：16 字节随机 + 4 字节 msg_len + msg + receive_id
            if len(plaintext) < 20:
                raise ValueError("plaintext too short")
            msg_len = struct.unpack(">I", plaintext[16:20])[0]
            if len(plaintext) < 20 + msg_len:
                raise ValueError(f"msg_len {msg_len} exceeds remaining data")
            msg = plaintext[20:20 + msg_len].decode("utf-8")
            receive_id = plaintext[20 + msg_len:].decode("utf-8")
            return msg, receive_id
        except Exception as exc:
            raise ValueError(f"decrypt failed: {exc}") from exc

    # ================================================================== #
    #  加密（用于主动发送回复，可选）
    # ================================================================== #
    def encrypt(self, reply_msg: str, timestamp: str | None = None,
                nonce: str | None = None) -> Tuple[str, str]:
        """
        加密回复消息（标准协议格式）。

        Returns
        -------
        (encrypted_b64, signature)
        """
        import os
        import time

        if timestamp is None:
            timestamp = str(int(time.time()))
        if nonce is None:
            nonce = os.urandom(8).hex()

        # 拼接：16 随机 + 4 长度 + msg + receive_id
        rand_bytes = os.urandom(16)
        msg_bytes = reply_msg.encode("utf-8")
        receive_id_bytes = self.receive_id.encode("utf-8")
        msg_len = struct.pack(">I", len(msg_bytes))
        raw = rand_bytes + msg_len + msg_bytes + receive_id_bytes

        # PKCS#7 padding 到 32 字节倍数
        pad_len = self.BLOCK_SIZE - (len(raw) % self.BLOCK_SIZE)
        if pad_len == 0:
            pad_len = self.BLOCK_SIZE
        raw_padded = raw + bytes([pad_len]) * pad_len

        # AES-CBC 加密
        iv = self.key[:16]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        encrypted = cipher.encrypt(raw_padded)
        encrypted_b64 = base64.b64encode(encrypted).decode("utf-8")

        # 签名
        items = sorted([self.token, timestamp, nonce, encrypted_b64])
        signature = hashlib.sha1("".join(items).encode("utf-8")).hexdigest()

        return encrypted_b64, signature


# ---------------------------------------------------------------------- #
#  Self-test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import os
    import time

    # 模拟一遍
    token = "test_token"
    aes_key = "A" * 43  # 43 chars
    receive_id = "ww1234567890"

    c = WeComCrypto(token, aes_key, receive_id)

    # 加密
    original = "<xml><Content>hello</Content></xml>"
    ts = str(int(time.time()))
    nonce = "abc"
    encrypted, sig = c.encrypt(original, ts, nonce)
    print(f"encrypted: {encrypted[:60]}...")
    print(f"signature: {sig}")

    # 验签
    assert c.verify_signature(sig, ts, nonce, encrypted), "signature mismatch"

    # 解密
    msg, rid = c.decrypt(encrypted)
    print(f"decrypted msg: {msg}")
    print(f"decrypted receive_id: {rid}")
    assert msg == original
    assert rid == receive_id
    print("OK")