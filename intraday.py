"""
盘中数据层 —— 为 11:30 / 14:30 触发器提供分钟级行情。

数据源策略（避免被 Eastmoney 反爬切断）：
  * 分钟 K 线：腾讯财经 `web.ifzq.gtimg.cn`（稳定、不限频）
  * 实时快照：腾讯 `qt.gtimg.cn`（更全的字段）
  * 板块排行：akshare `stock_board_industry_summary_ths`（同花顺，稳定）

提供：
  * fetch_stock_minutes(symbol)         —— 个股当日分钟 K
  * fetch_index_minutes(code)           —— 指数当日分钟 K
  * fetch_stock_snapshot(symbols)       —— 多只股票实时快照
  * fetch_index_snapshot(codes)         —— 指数实时快照
  * fetch_industry_sectors()            —— 行业板块涨跌幅榜
  * IntradayClient.build_market_snapshot —— 聚合成 LLM 可用的字典
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, time as dtime
from typing import Any

import pandas as pd
import requests

from data_prep import _retry


# ====================================================================== #
#  股票名 / 代码互查（覆盖常见 100+ 只；不够用可自己加）
# ====================================================================== #
# 注：000001 既是"平安银行"也是"上证指数"，按股票匹配（指数走专用逻辑）
STOCK_NAME_TO_CODE = {
    # 显示器件
    "京东方": "000725", "boe": "000725", "京东方a": "000725",
    "tcl科技": "000100", "tcl": "000100",
    # 白酒 / 消费
    "茅台": "600519", "贵州茅台": "600519",
    "五粮液": "000858",
    "汾酒": "600809", "山西汾酒": "600809",
    "泸州老窖": "000568",
    "伊利": "600887", "伊利股份": "600887",
    "双汇": "000895",
    "海天": "603288", "海天味业": "603288",
    # 银行 / 保险
    "平安银行": "000001", "平安": "000001",
    "招行": "600036", "招商银行": "600036",
    "工行": "601398",
    "建行": "601939",
    "农行": "601288",
    "中行": "601988",
    "中国平安": "601318",
    "太保": "601601", "中国太保": "601601",
    "新华保险": "601336",
    # 新能源 / 车
    "宁德": "300750", "宁德时代": "300750", "catl": "300750",
    "比亚迪": "002594", "byd": "002594",
    "长城": "601633", "长城汽车": "601633",
    "理想": "2015",  # 港股
    # 半导体 / 科技
    "中芯国际": "688981", "中芯": "688981", "smic": "688981",
    "海光": "688041", "海光信息": "688041",
    "寒武纪": "688256",
    "北方华创": "002371",
    "韦尔": "603501", "韦尔股份": "603501",
    "海康": "002415", "海康威视": "002415",
    "立讯": "002475", "立讯精密": "002475",
    "工业富联": "601138",
    "晶方科技": "603005",
    "歌尔": "002241", "歌尔股份": "002241",
    "蓝思": "300433", "蓝思科技": "300433",
    "传音": "688036", "传音控股": "688036",
    "澜起": "688008", "澜起科技": "688008",
    "兆易": "603986", "兆易创新": "603986",
    "长电": "600584", "长电科技": "600584",
    "通富": "002156", "通富微电": "002156",
    "华工": "000988", "华工科技": "000988",
    # 医药
    "恒瑞": "600276", "恒瑞医药": "600276",
    "药明": "603259", "药明康德": "603259",
    "迈瑞": "300760", "迈瑞医疗": "300760",
    "爱尔": "300015", "爱尔眼科": "300015",
    "片仔癀": "600436",
    "智飞": "300122", "智飞生物": "300122",
    "长春高新": "000661",
    # 光伏 / 新能源
    "隆基": "601012", "隆基绿能": "601012",
    "通威": "600438",
    "阳光电源": "300274",
    # 制造
    "美的": "000333", "美的集团": "000333",
    "格力": "000651", "格力电器": "000651",
    "海尔": "600690", "海尔智家": "600690",
    "三一": "600031", "三一重工": "600031",
    "大族激光": "002008",  # 用户持仓
    "汇川": "300124", "汇川技术": "300124",
    # 资源 / 周期
    "紫金": "601899", "紫金矿业": "601899",
    "中国神华": "601088",
    "陕西煤业": "601225",
    "海螺": "600585", "海螺水泥": "600585",
    "万华": "600309", "万华化学": "600309",
    "宝钢": "600019",
    # 金融
    "中信证券": "600030",
    "东方财富": "300059",
    "宁波银行": "002142",
    # 其他常见
    "长江电力": "600900",
    "中国核电": "601985",
    "上海机场": "600009",
    "顺丰": "002352", "顺丰控股": "002352",

    # ===== 化学品 / 半导体材料 =====
    "雅克科技": "002409",                   # 用户关注
    "安集科技": "688019",
    "彤程新材": "603650",
    "南大光电": "300346",
    "晶瑞电材": "300655", "晶瑞": "300655",
    "容大感光": "300576",
    "上海新阳": "300236",
    "江化微": "603078",
    "飞凯材料": "300398",
    "强力新材": "300429",
    "金宏气体": "688106",
    "华特气体": "688268",
    "凯美特气": "002549",
    "昊华科技": "600378",
    "巨化股份": "600160",
    "永和股份": "605020",
    "三美股份": "603379",
    "新宙邦": "300037",
    "天赐材料": "002709",
    "多氟多": "002407",
    "石大胜华": "603026",
    "杉杉股份": "600884",
    "璞泰来": "603659",
    "恩捷股份": "002812",
    "星源材质": "300568",

    # ===== 半导体设备 / 设计 =====
    "北方华创": "002371",
    "中微公司": "688012",
    "拓荆科技": "688072",
    "华海清科": "688120",
    "芯源微": "688037",
    "长川科技": "300604",
    "精测电子": "300567",
    "华峰测控": "688200",
    "概伦电子": "688206",
    "紫光国微": "002049",
    "兆易创新": "603986",
    "韦尔股份": "603501",
    "卓胜微": "300782",
    "圣邦股份": "300661",
    "思瑞浦": "688536",
    "纳芯微": "688052",
    "东微半导": "688261",
    "士兰微": "600460",
    "华润微": "688396",
    "扬杰科技": "300373",
    "新洁能": "605111",
    "时代电气": "688187",
    "斯达半导": "603290",

    # ===== PCB / 封装 =====
    "生益科技": "600183",
    "深南电路": "002916",
    "沪电股份": "002463",
    "景旺电子": "603228",
    "胜宏科技": "300476",
    "鹏鼎控股": "002938",
    "东山精密": "002384",
    "兴森科技": "002436",
    "江丰电子": "300666",

    # ===== AI / 算力 =====
    "海光信息": "688041",
    "寒武纪": "688256",
    "中科曙光": "603019",
    "浪潮信息": "000977",
    "紫光股份": "000938",
    "工业富联": "601138",

    # ===== 电子元器件 / 电容 / 电阻 =====
    "铜峰电子": "600237",                # 用户问过
    "风华高科": "000636",
    "顺络电子": "002138",
    "三环集团": "300408",
    "法拉电子": "600563",
    "艾华集团": "603989",
    "江海股份": "002484",
    "火炬电子": "603678",
    "鸿远电子": "603267",
    "宏达电子": "300726",
    "振华科技": "000733",
    "北方导航": "600435",
    "中国卫星": "600118",
    "航天电器": "002025",
    "航天电子": "600879",

    # ===== 通信 / 5G =====
    "中兴通讯": "000063",
    "烽火通信": "600498",
    "光迅科技": "002281",
    "中际旭创": "300308",
    "新易盛": "300502",
    "天孚通信": "300394",
    "信维通信": "300136",

    # ===== 工业制造 / 高端装备 =====
    "三一重工": "600031",
    "徐工机械": "000425",
    "中联重科": "000157",
    "柳工": "000528",
    "杭叉": "603298",
    "安徽合力": "600761",
    "中国一重": "601106",
    "二重重装": "601268",
    "东方电气": "600875",
    "上海电气": "601727",
    "哈尔滨电气": "1133",  # 港股

    # ===== 电力 / 公用事业 =====
    "长江电力": "600900",
    "中国核电": "601985",
    "华能水电": "600025",
    "国投电力": "600886",
    "川投能源": "600674",
    "华能国际": "600011",
    "大唐发电": "601991",
    "华电国际": "600027",
    "国电电力": "600795",
    "中国神华": "601088",
    "陕西煤业": "601225",
    "兖矿能源": "600188",
    "中煤能源": "601898",

    # ===== 传媒 / 游戏 =====
    "分众传媒": "002027",
    "三七互娱": "002555",
    "完美世界": "002624",
    "吉比特": "603444",
    "芒果超媒": "300413",
    "光线传媒": "300251",
    "万达电影": "002739",
    "中国电影": "600977",

    # ===== 医药商业 / 中药 =====
    "上海医药": "601607",
    "国药一致": "000028",
    "国药股份": "600511",
    "白云山": "600332",
    "云南白药": "000538",
    "同仁堂": "600085",
    "片仔癀": "600436",
    "东阿阿胶": "000423",
    "华润三九": "000999",
    "以岭药业": "002603",

    # ===== 券商 =====
    "中信证券": "600030",
    "华泰证券": "601688",
    "国泰君安": "601211",
    "海通证券": "600837",
    "招商证券": "600999",
    "广发证券": "000776",
    "申万宏源": "000166",
    "中信建投": "601066",
    "东方财富": "300059",

    # ===== 保险 =====
    "中国平安": "601318",
    "中国人寿": "601628",
    "中国太保": "601601",
    "新华保险": "601336",
    "中国人保": "601319",
    "天茂集团": "000627",
    "西水股份": "600291",
}


def extract_symbols(text: str) -> list[str]:
    """
    从用户问题里识别股票代码。

    三级匹配：
    1. 6 位数字（排除已知指数）
    2. 硬编码 STOCK_NAME_TO_CODE（80+ 常见股）
    3. Fuzzy match 到 akshare 全 A 股表（5500+，缓存）
    """
    found: set[str] = set()

    # 6 位数字（要求不是 399001/399006 这种指数代码——指数另行处理）
    for m in re.finditer(r"\b(\d{6})\b", text):
        code = m.group(1)
        # 排除明显的指数代码（399xxx 和 0xxxxx 中已知是指数的）
        if code not in ("399001", "399006", "399905", "000300", "000688"):
            found.add(code)

    # 硬编码字典（按长度倒序）
    for name in sorted(STOCK_NAME_TO_CODE.keys(), key=len, reverse=True):
        if name in text:
            found.add(STOCK_NAME_TO_CODE[name])

    # Fuzzy match 兜底（akshare 全 A 股表）
    # 性能考虑：只在硬编码没匹配到时跑
    if not found:
        for code in fuzzy_match_stock_all(text):
            found.add(code)

    return list(found)


# ====================================================================== #
#  股票 Fuzzy match（akshare 全 A 股表 5500+）
# ====================================================================== #
_STOCK_TABLE_CACHE: Any = None


def _load_stock_table() -> Any:
    """加载并缓存 akshare 全 A 股名表（首次调用约 5-7s，缓存后 <10ms）。"""
    global _STOCK_TABLE_CACHE
    if _STOCK_TABLE_CACHE is None:
        import akshare as ak
        _STOCK_TABLE_CACHE = ak.stock_info_a_code_name()
    return _STOCK_TABLE_CACHE


def fuzzy_match_stock_all(text: str) -> list[str]:
    """
    Fuzzy match query 中的 A 股名 → 代码列表。

    策略：按"最长子串优先"分层搜索。
    1. 先试 length = L 的整个 query（4 字符以上）
    2. 没匹配 → 试 length = L-1
    3. 一直到 length = 2

    每一层内：
    - 找到该层所有匹配的股票
    - 在 matches 里找"含最长 query 子串 + 简称短"的最优

    Returns
    -------
    代码列表（去重，按匹配度排序，最多 5 个）
    """
    df = _load_stock_table()
    if df is None or df.empty:
        return []

    # 清洗 query
    cleaned = re.sub(r"[，。？！；：、,.?!;:]+", "", text).strip()
    L = len(cleaned)
    if L < 2:
        return []

    # 按长度倒序枚举子串（用 list 保持顺序）
    all_substrings: list[str] = []
    for length in range(min(10, L), 1, -1):
        for start in range(L - length + 1):
            s = cleaned[start:start + length]
            if s.isdigit():
                continue
            all_substrings.append(s)

    if not all_substrings:
        return []

    # 单次 regex 搜索（union 所有子串）
    pattern = "|".join(re.escape(s) for s in all_substrings)
    try:
        matches = df[df["name"].str.contains(pattern, regex=True, na=False)]
    except Exception:
        return []

    if matches.empty:
        return []

    # 评分：在 matches 里，名字中含最长 query 子串的优先
    substr_by_len = sorted(all_substrings, key=len, reverse=True)
    # query 前 2 字符作为强信号（如 "中航电子" → 优先含"中航"的）
    first_two = cleaned[:2] if len(cleaned) >= 2 else cleaned

    def _match_score(row):
        name = row["name"]
        # 找名字中含的"最长 query 子串"
        best_len = 0
        for s in substr_by_len:
            if s in name:
                best_len = len(s)
                break
        # 前缀 bonus：query 前 2 字符在 name 里 → 加 1（让"中航电子"优先"电子城"）
        prefix_bonus = 1 if first_two in name and first_two != cleaned else 0
        return (-best_len, -prefix_bonus, len(name), name)

    matches = matches.copy()
    matches["_score"] = matches.apply(_match_score, axis=1)
    matches = matches.sort_values(by="_score")

    # 只返回 top 1（避免污染 mentioned list）
    # 如果用户想要多只，他们应该明确说"几只股票"或分别问
    return matches.head(1)["code"].tolist()


# ====================================================================== #
#  公共：HTTP session
# ====================================================================== #
def _tencent_session() -> requests.Session:
    """构造带浏览器 UA 的 requests session。"""
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://gu.qq.com/",
    })
    return s


def _tencent_symbol(symbol: str) -> str:
    """6 位股票代码 -> 腾讯带前缀代码：sh600519 / sz000001。"""
    if symbol.startswith(("60", "68", "9", "5")):
        return f"sh{symbol}"
    return f"sz{symbol}"


def _tencent_index_symbol(code: str) -> str:
    """
    指数代码 -> 腾讯带前缀代码。

    注：腾讯把指数和股票混在同一个 q 接口里，必须用真实存在的"指数 secid"。
    上证系列（含 000001 = 上证指数） -> sh 前缀
    深证系列（399xxx） -> sz 前缀
    """
    if code.startswith("0"):
        return f"sh{code}"
    return f"sz{code}"


# ====================================================================== #
#  1. 个股当日分钟 K（腾讯）
# ====================================================================== #
def fetch_stock_minutes(symbol: str) -> pd.DataFrame:
    """
    拉取某只股票当日分钟 K（1 分钟粒度）。
    返回列：datetime, close, volume, amount, open, high, low, avg_price
    """
    t_symbol = _tencent_symbol(symbol)
    sess = _tencent_session()

    def _do():
        r = sess.get(
            "https://web.ifzq.gtimg.cn/appstock/app/minute/query",
            params={"code": t_symbol},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()

    data = _retry(_do, retries=3, backoff=2.0)
    raw = (
        data.get("data", {})
        .get(t_symbol, {})
        .get("data", {})
        .get("data", [])
    )
    if not raw:
        return pd.DataFrame()

    today = datetime.now().date()
    rows = []
    for line in raw:
        parts = line.split()
        if len(parts) < 4:
            continue
        hhmm = parts[0]
        if not re.match(r"^\d{4}$", hhmm):
            continue
        try:
            dt = pd.Timestamp(today) + pd.Timedelta(
                hours=int(hhmm[:2]), minutes=int(hhmm[2:])
            )
            rows.append({
                "datetime": dt,
                "close": float(parts[1]),
                "volume": float(parts[2]),
                "amount": float(parts[3]),
            })
        except (ValueError, IndexError):
            continue

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("datetime").sort_index()
    if not df.empty:
        df["open"] = df["close"].iloc[0]
        df["high"] = df["close"].cummax()
        df["low"] = df["close"].cummin()
        df["avg_price"] = (df["amount"] / df["volume"].replace(0, pd.NA)).fillna(0.0)
    df["symbol"] = symbol
    return df


def fetch_multiple_stocks_minutes(symbols: list[str]) -> pd.DataFrame:
    """批量拉多只股票的当日分钟 K；任一失败不阻塞整体，返回合并后的 DataFrame。"""
    frames = []
    for sym in symbols:
        try:
            frames.append(fetch_stock_minutes(sym))
        except Exception as exc:
            print(f"[warn] {sym} intraday: {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()


# ====================================================================== #
#  2. 指数当日分钟 K（腾讯）
# ====================================================================== #
INDEX_NAMES = {
    "000001": "上证指数",
    "399001": "深证成指",
    "399006": "创业板指",
    "000300": "沪深300",
    "000688": "科创50",
    "399905": "中证500",
}


def fetch_index_minutes(code: str) -> pd.DataFrame:
    t_symbol = _tencent_index_symbol(code)
    sess = _tencent_session()

    def _do():
        r = sess.get(
            "https://web.ifzq.gtimg.cn/appstock/app/minute/query",
            params={"code": t_symbol},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()

    data = _retry(_do, retries=3, backoff=2.0)
    raw = (
        data.get("data", {})
        .get(t_symbol, {})
        .get("data", {})
        .get("data", [])
    )
    if not raw:
        return pd.DataFrame()

    today = datetime.now().date()
    rows = []
    for line in raw:
        parts = line.split()
        if len(parts) < 4:
            continue
        hhmm = parts[0]
        if not re.match(r"^\d{4}$", hhmm):
            continue
        try:
            dt = pd.Timestamp(today) + pd.Timedelta(
                hours=int(hhmm[:2]), minutes=int(hhmm[2:])
            )
            rows.append({
                "datetime": dt,
                "close": float(parts[1]),
                "volume": float(parts[2]),
                "amount": float(parts[3]),
            })
        except (ValueError, IndexError):
            continue

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("datetime").sort_index()
    if not df.empty:
        df["open"] = df["close"].iloc[0]
        df["high"] = df["close"].cummax()
        df["low"] = df["close"].cummin()
    df["symbol"] = code
    df["name"] = INDEX_NAMES.get(code, code)
    return df


def fetch_multiple_indices(codes: list[str]) -> pd.DataFrame:
    """批量拉多个指数的当日分钟 K。"""
    frames = []
    for c in codes:
        try:
            frames.append(fetch_index_minutes(c))
        except Exception as exc:
            print(f"[warn] index {c}: {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()


# ====================================================================== #
#  3. 实时快照（腾讯）
# ====================================================================== #
def fetch_stock_snapshot(symbols: list[str]) -> dict[str, dict[str, float]]:
    """
    实时行情快照（腾讯 v_qt 格式 + Sina 兜底）。

    返回：{ symbol: { name, last, open, high, low, prev_close,
                     volume_lots, amount_wan, change, pct_change } }

    策略：先批量调 Tencent，缺失的 symbol 逐个走 Sina 兜底。
    """
    if not symbols:
        return {}

    # 1. Tencent 主源（批量）
    t_symbols = [_tencent_symbol(s) for s in symbols]
    sess = _tencent_session()

    def _do_tencent():
        r = sess.get(
            "https://qt.gtimg.cn/q=" + ",".join(t_symbols),
            timeout=10,
        )
        r.raise_for_status()
        return r.content.decode("gbk", errors="ignore")

    try:
        text = _retry(_do_tencent, retries=2, backoff=2.0)
    except Exception as exc:
        print(f"[warn] tencent snapshot: {exc}")
        text = ""

    result: dict[str, dict[str, float]] = {}
    for line in text.strip().split("\n"):
        m = re.match(r'v_(\w+)="([^"]+)"', line)
        if not m:
            continue
        t_sym, payload = m.groups()
        parts = payload.split("~")
        if len(parts) < 50:
            continue
        try:
            symbol = t_sym[2:]
            result[symbol] = {
                "name": parts[1],
                "last": float(parts[3]) if parts[3] else 0.0,
                "prev_close": float(parts[4]) if parts[4] else 0.0,
                "open": float(parts[5]) if parts[5] else 0.0,
                "volume_lots": float(parts[6]) if parts[6] else 0.0,
                "high": float(parts[33]) if parts[33] else 0.0,
                "low": float(parts[34]) if parts[34] else 0.0,
                "amount_wan": float(parts[37]) if parts[37] else 0.0,
                "change": float(parts[31]) if parts[31] else 0.0,
                "pct_change": float(parts[32]) if parts[32] else 0.0,
            }
        except (ValueError, IndexError):
            continue

    # 2. Sina 兜底：缺失的 symbol 逐个调
    missing = [s for s in symbols if s not in result]
    if missing:
        sina_data = fetch_stock_snapshot_sina(missing)
        result.update(sina_data)

    return result


# ====================================================================== #
#  3.1 Sina 行情快照（兜底源，覆盖更多小票）
# ====================================================================== #
def fetch_stock_snapshot_sina(symbols: list[str]) -> dict[str, dict[str, float]]:
    """
    Sina 行情快照（hq.sinajs.cn）。覆盖 Tencent 拿不到的小票/冷门股。

    Sina 返回格式：
      var hq_str_sz002409="雅克科技,180.000,181.030,185.390,191.880,178.200,
                           185.280,185.400,24380386,4553938723.180,...";

    字段位置：
      [0]  名称
      [1]  今开
      [2]  昨收
      [3]  现价
      [4]  最高
      [5]  最低
      [8]  成交量(股)  → 除 100 转手
      [9]  成交额(元)  → 除 10000 转万
    """
    if not symbols:
        return {}

    # Sina 编码：深圳 sz、上海 sh、北京 bj
    def _sina_code(s: str) -> str:
        if s.startswith(("60", "68", "9", "5")):
            return f"sh{s}"
        return f"sz{s}"

    sess = requests.Session()
    sess.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://finance.sina.com.cn/",
    })

    result: dict[str, dict[str, float]] = {}
    for sym in symbols:
        try:
            r = sess.get(
                f"https://hq.sinajs.cn/list={_sina_code(sym)}",
                timeout=10,
            )
            r.raise_for_status()
            text = r.content.decode("gbk", errors="ignore")
            m = re.search(r'"([^"]+)"', text)
            if not m:
                continue
            parts = m.group(1).split(",")
            if len(parts) < 10:
                continue
            try:
                result[sym] = {
                    "name": parts[0],
                    "open": float(parts[1]),
                    "prev_close": float(parts[2]),
                    "last": float(parts[3]),
                    "high": float(parts[4]),
                    "low": float(parts[5]),
                    "volume_lots": float(parts[8]) / 100,   # 股 → 手
                    "amount_wan": float(parts[9]) / 10000,  # 元 → 万元
                    "change": float(parts[3]) - float(parts[2]),
                    "pct_change": (float(parts[3]) - float(parts[2])) / float(parts[2]) * 100,
                }
            except (ValueError, IndexError):
                continue
        except Exception as exc:
            print(f"[warn] sina {sym}: {exc}")
    return result


def fetch_index_snapshot(codes: list[str]) -> dict[str, dict[str, float]]:
    """指数实时快照。"""
    if not codes:
        return {}
    t_symbols = [_tencent_index_symbol(c) for c in codes]
    sess = _tencent_session()

    def _do():
        r = sess.get(
            "https://qt.gtimg.cn/q=" + ",".join(t_symbols),
            timeout=10,
        )
        r.raise_for_status()
        return r.content.decode("gbk", errors="ignore")

    text = _retry(_do, retries=3, backoff=2.0)
    result: dict[str, dict[str, float]] = {}
    for line in text.strip().split("\n"):
        m = re.match(r'v_(\w+)="([^"]+)"', line)
        if not m:
            continue
        t_sym, payload = m.groups()
        parts = payload.split("~")
        if len(parts) < 35:
            continue
        try:
            code = t_sym[2:]
            result[code] = {
                "name": parts[1],
                "last": float(parts[3]) if parts[3] else 0.0,
                "prev_close": float(parts[4]) if parts[4] else 0.0,
                "open": float(parts[5]) if parts[5] else 0.0,
                "high": float(parts[33]) if parts[33] else 0.0,
                "low": float(parts[34]) if parts[34] else 0.0,
                "change": float(parts[31]) if parts[31] else 0.0,
                "pct_change": float(parts[32]) if parts[32] else 0.0,
            }
        except (ValueError, IndexError):
            continue
    return result


# ====================================================================== #
#  4. 行业板块涨跌幅（同花顺接口，akshare 走的就是这个）
# ====================================================================== #
def fetch_industry_sectors(top_n: int = 32) -> pd.DataFrame:
    """
    行业板块涨跌幅榜。
    数据源：akshare.stock_board_industry_summary_ths（同花顺，最稳定）。

    字段：name, pct_change, leader, leader_pct, advance_count, decline_count
    """
    try:
        import akshare as ak
        df = _retry(ak.stock_board_industry_summary_ths, retries=2, backoff=2.0)
    except Exception as exc:
        print(f"[warn] industry sectors: {exc}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    rename = {
        "板块": "name",
        "涨跌幅": "pct_change",
        "领涨股": "leader",
        "领涨股-涨跌幅": "leader_pct",
        "上涨家数": "advance_count",
        "下跌家数": "decline_count",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    keep = [c for c in ("name", "pct_change", "leader", "leader_pct",
                        "advance_count", "decline_count") if c in df.columns]
    df = df[keep].copy()
    if "pct_change" in df.columns:
        df["pct_change"] = pd.to_numeric(df["pct_change"], errors="coerce")
        df = df.sort_values("pct_change", ascending=False, na_position="last")
    return df.head(top_n)


# ====================================================================== #
#  全市场行情 + 涨跌幅榜 + 量化打分
# ====================================================================== #
_FULL_MARKET_CACHE: dict[str, Any] = {"ts": 0.0, "df": None}
_FULL_MARKET_TTL = 60  # 秒


def fetch_full_market_spot(force_refresh: bool = False) -> pd.DataFrame:
    """
    拉全 A 股实时行情（~5500 只，14 列）。
    数据源：akshare.stock_zh_a_spot（东方财富接口，~20s，慢）。
    加了 60s 内存缓存。

    字段：代码、名称、最新价、涨跌额、涨跌幅、买入、卖出、昨收、今开、最高、最低、成交量、成交额、时间戳
    """
    now = time.time()
    if not force_refresh and _FULL_MARKET_CACHE["df"] is not None:
        if now - _FULL_MARKET_CACHE["ts"] < _FULL_MARKET_TTL:
            return _FULL_MARKET_CACHE["df"]

    try:
        import akshare as ak
        df = _retry(ak.stock_zh_a_spot, retries=2, backoff=2.0)
    except Exception as exc:
        print(f"[warn] full market spot: {exc}")
        return _FULL_MARKET_CACHE.get("df") or pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # 重命名
    rename = {
        "代码": "symbol", "名称": "name", "最新价": "last",
        "涨跌额": "change", "涨跌幅": "pct_change",
        "买入": "bid", "卖出": "ask", "昨收": "prev_close",
        "今开": "open", "最高": "high", "最低": "low",
        "成交量": "volume", "成交额": "amount", "时间戳": "ts",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # 清理 symbol（去掉前缀如 "bj920000" → "bj920000"，深沪前缀保留）
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str)

    # 数值化
    for col in ("last", "change", "pct_change", "open", "high", "low", "volume", "amount"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    _FULL_MARKET_CACHE["ts"] = now
    _FULL_MARKET_CACHE["df"] = df
    return df


def fetch_top_movers(top_n: int = 30, exclude_st: bool = True) -> pd.DataFrame:
    """
    全市场涨跌幅榜：涨幅 + 跌幅 各 top_n。

    Returns
    -------
    DataFrame with columns: symbol, name, last, pct_change, change, volume, amount
    """
    df = fetch_full_market_spot()
    if df.empty:
        return df

    if exclude_st and "name" in df.columns:
        # 排除 ST / *ST
        df = df[~df["name"].str.contains("ST", na=False)]

    # 涨幅榜
    top_up = df.nlargest(top_n, "pct_change")[
        ["symbol", "name", "last", "pct_change", "change", "volume", "amount"]
    ].copy()
    top_up["direction"] = "up"

    # 跌幅榜
    top_down = df.nsmallest(top_n, "pct_change")[
        ["symbol", "name", "last", "pct_change", "change", "volume", "amount"]
    ].copy()
    top_down["direction"] = "down"

    return pd.concat([top_up, top_down], ignore_index=True)


def score_potential_stocks(top_n: int = 10, exclude_st: bool = True,
                          exclude_limit_up: bool = True,
                          exclude_bj: bool = True) -> pd.DataFrame:
    """
    量化打分潜力个股。

    评分维度（满分 100）：
        - 动量 (40 分)：当日涨跌幅
            * -5% ~ +5% 线性给分
            * > 5% 减分（追高风险）
            * < -5% 大幅减分（弱势）
        - 量能 (35 分)：成交额 / 价格（粗略的"换手"指标）
            * 适中为佳（流动性好但不疯狂）
        - 形态 (25 分)：振幅（high-low）/ last
            * 3-7% 最佳（盘中有承接）
            * > 10% 太混乱
            * < 1% 太沉闷

    Returns
    -------
    DataFrame with columns: symbol, name, last, pct_change, score, components
    """
    df = fetch_full_market_spot()
    if df.empty:
        return df

    # 预处理
    if exclude_st and "name" in df.columns:
        df = df[~df["name"].str.contains("ST", na=False)]
    if exclude_bj and "symbol" in df.columns:
        # 排除北交所（bj 前缀，流动性差）
        df = df[~df["symbol"].str.startswith("bj", na=False)]
    if exclude_limit_up:
        # 排除涨跌停股
        df = df[(df["pct_change"] < 9.5) & (df["pct_change"] > -9.5)]

    # 排除无成交
    df = df[(df["volume"] > 0) & (df["last"] > 0)]
    if df.empty:
        return df

    # === 动量得分 (40 分封顶) ===
    pct = df["pct_change"]
    # 在 [-5%, +5%] 区间线性给分，斜率 4（+5%=20, +1%=4）
    momentum = (pct.clip(-5, 5) * 4 + 20).clip(0, 40)
    # > 5% 减分：6% 给 19，7% 给 17，8% 给 14
    overbought_mask = pct > 5
    momentum = momentum.mask(
        overbought_mask,
        (40 - (pct - 5) * 4).clip(0, 40)
    )
    # < -5% 大幅减分
    weak_mask = pct < -5
    momentum = momentum.mask(
        weak_mask,
        (20 - (5 - pct) * 4).clip(0, 40)
    )

    # === 量能得分 (35 分封顶) ===
    # 用 amount / last 估算"相对活跃度"，用 percentile 排名
    if "amount" in df.columns:
        activity = df["amount"] / df["last"].replace(0, pd.NA)
        # 用全市场的分位数归一化
        rank = activity.rank(pct=True, na_option="bottom")
        activity_score = rank * 35  # 0-35 分
        activity_score = activity_score.fillna(0).clip(0, 35)
    else:
        activity_score = pd.Series([17.5] * len(df))

    # === 形态得分 (25 分封顶) ===
    if {"high", "low", "last"}.issubset(df.columns):
        amplitude = (df["high"] - df["low"]) / df["last"].replace(0, pd.NA) * 100
        # 3-7% 最优 (给 25 分)
        # 偏离这个区间扣分
        amp_score = 25 - (amplitude - 5).abs() * 2
        amp_score = amp_score.clip(0, 25).fillna(0)
    else:
        amp_score = pd.Series([12.5] * len(df))

    df = df.assign(
        momentum_score=momentum.round(1),
        activity_score=activity_score.round(1),
        amplitude_score=amp_score.round(1),
    )
    df["score"] = (momentum + activity_score + amp_score).round(1)

    return df.nlargest(top_n, "score")[
        ["symbol", "name", "last", "pct_change", "change", "score",
         "momentum_score", "activity_score", "amplitude_score"]
    ]


def format_potential_table(df: pd.DataFrame) -> str:
    """把潜力股 DataFrame 格式化成 markdown 文本。"""
    if df.empty:
        return "📭 暂无符合条件的潜力股（可能全市场都在跌，或数据源不可用）"

    lines = ["| 代码 | 名称 | 现价 | 涨跌幅 | 评分 | 动量 | 量能 | 形态 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for _, r in df.iterrows():
        arrow = "🟢" if r["pct_change"] >= 0 else "🔴"
        lines.append(
            f"| {r['symbol']} | {r['name']} | {r['last']:.2f} | "
            f"{arrow}{abs(r['pct_change']):.2f}% | {r['score']:.0f} | "
            f"{r['momentum_score']:.0f} | {r['activity_score']:.0f} | "
            f"{r['amplitude_score']:.0f} |"
        )
    return "\n".join(lines)


def format_movers_table(df: pd.DataFrame, top_n: int = 10) -> str:
    """把涨跌幅榜格式化成 markdown 文本。"""
    if df.empty:
        return "📭 暂无数据"

    up = df[df["direction"] == "up"].head(top_n)
    down = df[df["direction"] == "down"].head(top_n)

    text = "**📈 涨幅 TOP**\n"
    text += "| 代码 | 名称 | 现价 | 涨跌幅 |\n| --- | --- | --- | --- |\n"
    for _, r in up.iterrows():
        text += f"| {r['symbol']} | {r['name']} | {r['last']:.2f} | 🟢+{r['pct_change']:.2f}% |\n"

    text += "\n**📉 跌幅 TOP**\n"
    text += "| 代码 | 名称 | 现价 | 涨跌幅 |\n| --- | --- | --- | --- |\n"
    for _, r in down.iterrows():
        text += f"| {r['symbol']} | {r['name']} | {r['last']:.2f} | 🔴{r['pct_change']:.2f}% |\n"
    return text


# ====================================================================== #
#  5. 行情快照聚合（给 LLM 用）
# ====================================================================== #
@dataclass
class IntradayClient:
    """聚合生成"市场快照" —— 供 LLM agent 直接使用的结构化字典。"""

    cache_dir: str = "./data_cache/intraday"
    indices: tuple[str, ...] = ("000001", "399001", "399006")

    def build_market_snapshot(self, watchlist: list[str],
                              holdings: list[str],
                              mentioned: list[str] | None = None,
                              enrich_mentioned: bool = True) -> dict[str, Any]:
        """
        构建 LLM 用的市场快照。

        Parameters
        ----------
        watchlist : 自选股列表
        holdings  : 持仓列表
        mentioned : 用户在对话中提到的股票
        enrich_mentioned : 是否为 mentioned 的股票拉技术指标 + 资金流
                           （每个股票 +2-3s；只在用户明确问到时启用）
        """
        now = datetime.now()
        session_phase = _infer_phase(now)
        mentioned = mentioned or []

        snapshot: dict[str, Any] = {
            "as_of": now.isoformat(timespec="minutes"),
            "session_phase": session_phase,
            "indices": {},
            "watchlist": {},
            "holdings": {},
            "mentioned": {},
            "mentioned_enriched": {},   # 技术指标 + 资金流（仅 mentioned）
            "sector_top": [],
            "sector_bottom": [],
        }

        # ---- 指数 ----
        idx_snap = fetch_index_snapshot(list(self.indices))
        for code in self.indices:
            if code in idx_snap:
                info = idx_snap[code]
                snapshot["indices"][code] = {
                    "name": info["name"],
                    "last": round(info["last"], 2),
                    "pct_change": round(info["pct_change"], 2),
                    "open": round(info["open"], 2),
                    "high": round(info["high"], 2),
                    "low": round(info["low"], 2),
                }

        # ---- 自选股 + 持仓 + 用户提及 ----
        all_symbols = list(dict.fromkeys(holdings + watchlist + mentioned))
        if all_symbols:
            stk_snap = fetch_stock_snapshot(all_symbols)
            for sym in all_symbols:
                if sym not in stk_snap:
                    continue
                info = stk_snap[sym]
                summary = {
                    "name": info["name"],
                    "last": round(info["last"], 2),
                    "pct_change": round(info["pct_change"], 2),
                    "open": round(info["open"], 2),
                    "high": round(info["high"], 2),
                    "low": round(info["low"], 2),
                    "volume_lots": round(info["volume_lots"] / 100, 0),
                    "amount_wan": round(info["amount_wan"], 0),
                }
                if sym in holdings:
                    snapshot["holdings"][sym] = summary
                if sym in watchlist:
                    snapshot["watchlist"][sym] = summary
                if sym in mentioned and sym not in holdings and sym not in watchlist:
                    snapshot["mentioned"][sym] = summary

        # ---- mentioned 股票的技术指标 + 资金流（enrich）----
        if enrich_mentioned and mentioned:
            try:
                from indicators import compute_indicators, fetch_capital_flow
            except ImportError:
                log_snapshot = None
                compute_indicators = None
                fetch_capital_flow = None

            for sym in mentioned:
                enriched: dict[str, Any] = {}

                # 技术指标
                if compute_indicators is not None:
                    try:
                        ind = compute_indicators(sym)
                        if ind is not None:
                            enriched["indicators"] = ind.to_dict()
                    except Exception as exc:
                        print(f"[warn] indicators {sym}: {exc}")

                # 资金流
                if fetch_capital_flow is not None:
                    try:
                        flow = fetch_capital_flow(sym)
                        if flow is not None:
                            enriched["capital_flow"] = flow
                    except Exception as exc:
                        print(f"[warn] capital flow {sym}: {exc}")

                if enriched:
                    # 加上股票名（从 mentioned summary 取）
                    if sym in snapshot["mentioned"]:
                        enriched["name"] = snapshot["mentioned"][sym].get("name", sym)
                    elif sym in snapshot["holdings"]:
                        enriched["name"] = snapshot["holdings"][sym].get("name", sym)
                    elif sym in snapshot["watchlist"]:
                        enriched["name"] = snapshot["watchlist"][sym].get("name", sym)
                    snapshot["mentioned_enriched"][sym] = enriched

        # ---- 板块 ----
        sectors = fetch_industry_sectors()
        if not sectors.empty and "pct_change" in sectors.columns:
            keep_cols = [c for c in ("name", "pct_change", "leader", "leader_pct")
                         if c in sectors.columns]
            snapshot["sector_top"] = sectors.head(5)[keep_cols].to_dict("records")
            snapshot["sector_bottom"] = sectors.tail(5)[keep_cols].to_dict("records")

        return snapshot


def _infer_phase(now: datetime) -> str:
    """
    根据当前时间推断交易阶段。

    Returns
    -------
    "morning_session"    9:30-11:30 上午盘
    "morning_close"      11:30-13:00 午休（11:30 触发点）
    "afternoon_session"  13:00-14:30 下午盘
    "afternoon_close"    14:30-15:00 尾盘（14:30 触发点）
    "closed"             其它时段
    """
    t = now.time()
    if dtime(9, 30) <= t < dtime(11, 30):
        return "morning_session"
    if dtime(11, 30) <= t < dtime(13, 0):
        return "morning_close"
    if dtime(13, 0) <= t < dtime(14, 30):
        return "afternoon_session"
    if dtime(14, 30) <= t <= dtime(15, 0):
        return "afternoon_close"
    return "closed"


# ---------------------------------------------------------------------- #
#  Self-test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    ic = IntradayClient()
    snap = ic.build_market_snapshot(
        watchlist=["600276", "601318", "002594"],
        holdings=["600519", "000001", "300750"],
    )
    print(json.dumps(snap, ensure_ascii=False, indent=2, default=str))