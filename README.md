# A 股实时分析助手

基于 DeepSeek + 腾讯财经/同花顺数据的 A 股智能分析系统。

每天 11:30 / 14:30 自动跑两次，输出到企业微信（未配置时本地保存 + 终端打印）。
其他时间可以手动 query。

## 项目结构

```
A/
├── 原有 ML pipeline（保留作为 LLM 的工具）
│   ├── data_prep.py         # 日线 + 宏观数据
│   ├── feature_engine.py    # 技术指标
│   ├── model_train.py       # 线性 / RF / LSTM / Transformer
│   ├── validation.py        # walk-forward + 监管测试
│   ├── backtest.py          # 策略回测
│   └── deploy.py            # 重训调度 + 监控面板
│
├── 新增模块
│   ├── intraday.py          # 盘中数据（分钟线 + 实时快照 + 板块 + 股票名识别）
│   ├── llm_agent.py         # DeepSeek 推理层
│   ├── wechat_push.py       # 多通道推送（webhook / 自建应用 / 第三方）
│   ├── wechat_crypto.py     # 企业微信回调消息 AES 加解密
│   ├── webhook_server.py    # 双向对话 Flask 服务（命令路由 + LLM）
│   ├── triggers.py          # 11:30 / 14:30 调度
│   ├── interactive.py       # 手动 query CLI
│   ├── check_ip.py          # 检测公网 IP（IP 白名单辅助）
│   └── main.py              # 主入口
│
├── 配置
│   ├── .env                 # API key（不要 commit）
│   ├── config.yaml          # 持仓 / 关注股 / 风险偏好 / 触发时点
│   └── .gitignore
│
├── 文档
│   ├── README.md
│   └── SETUP_WECHAT_BOT.md  # 企业微信双向对话 部署指南
│
└── reports/                 # 历史报告（自动生成）
```

## 快速开始

### 1. 安装依赖

```bash
source .venv/bin/activate
pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple openai python-dotenv pyyaml apscheduler requests flask pycryptodome
```

### 2. 配置 `.env`

```bash
DEEPSEEK_API_KEY=sk-xxxxx
WECOM_WEBHOOK_URL=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxxx
WECOM_CORP_ID=wwxxxx
WECOM_AGENT_ID=1000002
WECOM_TOKEN=xxxx              # 双向对话用
WECOM_AES_KEY=43位字符         # 双向对话用
```

### 3. 配置 `config.yaml`

填入你的持仓、关注股、风险偏好、触发时点。

### 4. 跑命令

```bash
# 手动跑一次上午复盘（立即执行）
python main.py morning

# 手动跑一次尾盘决策
python main.py afternoon

# 手动 query
python main.py ask "宁德时代现在要不要止盈？"

# 多轮对话（终端里和模型聊天）
python main.py chat

# 启动调度器（每天 11:30 / 14:30 自动跑）
python main.py serve

# 启动企业微信双向对话（需要 ngrok，详见 SETUP_WECHAT_BOT.md）
python main.py webhook

# 一个进程同时跑 webhook + 调度器（推荐）
python main.py all
```

## 数据源

| 数据 | 来源 | 接口 |
|---|---|---|
| 个股分钟线 | 腾讯财经 | `web.ifzq.gtimg.cn` |
| 指数分钟线 | 腾讯财经 | `web.ifzq.gtimg.cn` |
| 实时快照 | 腾讯财经 | `qt.gtimg.cn` |
| 板块排行 | 同花顺（akshare） | `stock_board_industry_summary_ths` |
| 日线/财报 | akshare | 历史 pipeline |
| LLM 推理 | DeepSeek | OpenAI 兼容 API |

注：东方财富（akshare 默认）部分接口对 Python 客户端有限制，已切到腾讯和同花顺。

## 企业微信配置（可选）

未配置时系统把报告写到 `reports/` 目录 + 终端打印。

要让消息出现在个人微信：

1. 注册企业微信（个人免费）：https://work.weixin.qq.com
2. 「应用管理 → 自建应用」创建一个应用
3. 记下 AgentId 和 Secret
4. 「我的企业 → 企业信息」记下 CorpId
5. 把这三个值填到 `.env`
6. 「微信插件」中扫码绑定（消息才会推送到个人微信）

## 输出示例

### 11:30 上午复盘

```
📊 **上午复盘**
- 三大指数低开高走，上证微涨0.11%，深成指、创业板分别涨1.24%、1.41%
- 领涨：能源金属（+4.27%）、半导体（+3.80%）
- 领跌：化学制药（-0.89%）、生物制品（-1.03%）

🔮 **下午预判**
- 倾向方向：震荡偏强
- 关键观察点：上证能否站稳4110点，半导体持续性

🎯 **操作建议**
- 贵州茅台（600519）：观望 | 跌破成本1680，缩量下跌...
- 宁德时代（300750）：看多 | 浮盈超100%，可持有...
- ...

⚠️ 风险提示：以上内容不构成投资建议。
```

### 14:30 尾盘决策

```
🌅 **今日总结**
- 指数方向 + 板块轮动 + 资金情绪

🎯 **尾盘 / 次日操作建议**
- [持仓] 持有/止盈/止损 建议
- [关注] 试多/观望 建议

📋 **明日关注**
- 关键事件 + 数据 + 板块

⚠️ 风险提示：以上内容不构成投资建议。
```

## 后续可扩展

- 接现有 ML pipeline 的预测（让 LLM 把模型信号作为输入之一）
- 加新闻/舆情数据（财联社、东方财富快讯）
- 微信双向消息（用户能在微信里直接问）
- 策略回测增强（自动评估 LLM 建议的胜率）
- 多用户支持