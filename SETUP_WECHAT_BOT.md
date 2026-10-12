# 企业微信双向对话 部署指南

> 让你在企业微信群里 **@机器人** 直接对话、查持仓、让 AI 分析。

## 架构

```
你在企业微信群里说：@机器人 京东方怎么看？
       ↓ (企业微信发加密消息)
ngrok 公网 URL → 你的 Mac localhost:5000
       ↓ (Flask 接收并解密)
LLM 分析
       ↓ (通过群机器人 webhook 推送)
你看到机器人在群里回复
```

**关键**：
- **接收**走自建应用回调（WeCom 主动调你，无 IP 限制）
- **发送**走群机器人 webhook（你已配好的那个，URL 形态，无 IP 限制）
- **完全绕开企业微信的 IP 白名单**

## 一次性配置（10 分钟）

### 第 1 步：企业微信后台拿回调凭证

1. 登录 https://work.weixin.qq.com/wework_admin/
2. **「应用管理」→「自建应用」** → 点开你的应用
3. 找到 **「接收消息」** 区域（一般在底部或「开发者接口」分组）
4. 点 **「设置 API 接收消息」**：
   - **Token**：你自己起一个字符串（比如 `ashare123`，记下来）
   - **EncodingAESKey**：点「随机生成」会得到 43 位字符串（**记下来，只显示一次**）
5. **先不要点保存**（要先启动 ngrok + 服务拿到公网 URL 再回来填）

### 第 2 步：装 ngrok

Mac 上：
```bash
brew install ngrok
# 或者直接下载：https://ngrok.com/download
```

注册 ngrok 账号（免费），拿到 authtoken：
```bash
ngrok config add-authtoken 你的authtoken
```

### 第 3 步：填 .env

```bash
WECOM_WEBHOOK_URL=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=你的key  # 已配
WECOM_CORP_ID=ww4e577a97b82c3653                                              # 已配
WECOM_AGENT_ID=1000002                                                          # 已配
WECOM_TOKEN=你刚才设的token                                                     # 新填
WECOM_AES_KEY=43位的EncodingAESKey                                              # 新填
```

### 第 4 步：启动 Flask 服务（推荐用 `all` 一条命令同时跑 webhook + 调度器）

**方式 A（推荐）**：`all` 一个进程同时跑 Flask + 11:30/14:30 调度器

```bash
cd /Users/jiangwenrui/Downloads/mass/20-A
.venv/bin/python main.py all
```

预期看到：
```
[INFO] webhook: WeCom crypto ready
[INFO] werkzeug: Running on http://127.0.0.1:5050
[INFO] main: 调度器启动：
[INFO] main:   - Morning close @ 11:30
[INFO] main:   - Afternoon close @ 14:30
[INFO] main: 按 Ctrl+C 停止所有服务。
```

**方式 B（分开）**：`webhook` 只跑 Flask，调度器另开终端跑 `serve`

```bash
# 终端 1
.venv/bin/python main.py webhook

# 终端 2
.venv/bin/python main.py serve
```

**这个终端别关。**

### 第 5 步：启动 ngrok（新开一个终端）

```bash
ngrok http 5000
```

预期看到：
```
Session Status   online
Forwarding       https://xxxx-xxx-xxx-xxx-xxx.ngrok-free.app -> http://localhost:5000
```

**复制那个 `https://xxxx.ngrok-free.app` URL**。

### 第 6 步：回企业微信后台填 URL

回到「设置 API 接收消息」页面：

- **URL**：填 `https://xxxx.ngrok-free.app/callback`
- **Token** / **EncodingAESKey**：已经填过，不用改
- 点 **「保存」**

企业微信会发一个 GET 请求验证 URL，我们的 Flask 会自动通过。

**预期看到「保存成功」**。如果失败，看 ngrok 那边的请求日志。

### 第 7 步：群里试一下

在企业微信群（**自建应用和群机器人都在的那个群**）发：

```
@你的应用名 持仓
```

几秒内，机器人（群机器人）会回复你的持仓列表。

试试这些：
```
@你的应用名 帮助
@你的应用名 京东方现在能加仓吗？
@你的应用名 添加 600519 1680 100
@你的应用名 持仓
@你的应用名 改成 50%
```

## 日常使用

- 每天开机后：
  ```bash
  # 终端 1
  cd /Users/jiangwenrui/Downloads/mass/20-A
  .venv/bin/python main.py webhook

  # 终端 2
  ngrok http 5000
  ```
- ngrok 免费版 URL 每次启动会变。**如果 URL 变了**，要回企业微信后台改 URL 并重新点保存。
- 想 URL 稳定：升级 ngrok 付费版（$8/月），或者把服务部署到云服务器（IP 固定）。

## 常见问题

**Q: ngrok URL 每次都变，烦死了？**
A: 两个办法：
1. ngrok 付费版固定子域名（$8/月）
2. 部署到云服务器（推荐，阿里云 ¥10/月，IP 固定，不需要 ngrok）

**Q: 能用其他反向代理吗？**
A: 可以。Cloudflare Tunnel、frp、Ngrok、localhost.run 都行，只要能暴露公网 URL 就行。

**Q: 群机器人发的消息前面有 "[机器人名字]"，能去掉吗？**
A: 不能，企业微信强制加 bot 名字作为标识。

**Q: 怎么停掉？**
A: 两个终端都 Ctrl+C。Flask 停了，群里 @bot 就不会回复了。