"""觀察池定義。

台股與美股會在執行時動態取得(依成交值挑流動性最好的標的),
取得失敗時才使用下面的備用清單。其他資產類別為固定清單。
"""
from __future__ import annotations

import io
import logging

import pandas as pd
import requests

log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0 (market-scan; +https://github.com/orangeai/market-scan)"}

# ---- 固定清單 -------------------------------------------------------------
INDICES = {
    "^TWII": "台灣加權指數",
    "^GSPC": "標普500指數",
    "^IXIC": "那斯達克指數",
    "^SOX": "費城半導體指數",
    "^DJI": "道瓊工業指數",
}

FUTURES = {
    "NQ=F": "那斯達克100期貨(小那同步)",
    "ES=F": "標普500期貨",
    "YM=F": "道瓊期貨",
    "GC=F": "黃金期貨",
    "SI=F": "白銀期貨",
    "HG=F": "銅期貨",
    "PL=F": "鉑金期貨",
    "CL=F": "WTI原油期貨",
    "BZ=F": "布蘭特原油期貨",
    "NG=F": "天然氣期貨",
    "ZS=F": "黃豆期貨",
    "ZC=F": "玉米期貨",
    "ZW=F": "小麥期貨",
    "KC=F": "咖啡期貨",
    "SB=F": "糖期貨",
    "CC=F": "可可期貨",
}

FX = {
    "EURUSD=X": "EUR/USD",
    "JPY=X": "USD/JPY",
    "GBPUSD=X": "GBP/USD",
    "AUDUSD=X": "AUD/USD",
    "TWD=X": "USD/TWD",
    "DX-Y.NYB": "美元指數",
}

CRYPTO = {
    "BTC-USD": "比特幣",
    "ETH-USD": "以太幣",
    "SOL-USD": "Solana",
    "XRP-USD": "XRP",
    "BNB-USD": "BNB",
    "DOGE-USD": "狗狗幣",
    "ADA-USD": "Cardano",
    "TRX-USD": "TRON",
    "AVAX-USD": "Avalanche",
    "LINK-USD": "Chainlink",
}

US_ETF = {
    "SPY": "SPDR 標普500 ETF",
    "QQQ": "Invesco 那斯達克100 ETF",
    "SMH": "VanEck 半導體 ETF",
    "SOXX": "iShares 半導體 ETF",
    "IWM": "iShares 羅素2000 ETF",
    "DIA": "SPDR 道瓊 ETF",
    "XLK": "科技類股 ETF",
    "XLF": "金融類股 ETF",
    "XLE": "能源類股 ETF",
    "XLV": "醫療類股 ETF",
    "XLI": "工業類股 ETF",
    "TLT": "20年期以上美債 ETF",
}

# 盤前任務也會用到的關鍵指標，一律輸出 K 線檔
KEY_SYMBOLS = ["^TWII", "^GSPC", "^IXIC", "^SOX", "NQ=F", "GC=F", "CL=F", "BZ=F", "TWD=X", "BTC-USD"]

# ---- 備用清單(動態取得失敗時使用) -----------------------------------------
TW_FALLBACK = [
    "2330", "2317", "2454", "2308", "2382", "2412", "2881", "2882", "2891", "2303",
    "3711", "2886", "2884", "1301", "1303", "2002", "3008", "2357", "2395", "3034",
    "2379", "6669", "3231", "2345", "2603", "2609", "2615", "5880", "2892", "2880",
    "2885", "2887", "2890", "1216", "2207", "2912", "4904", "3045", "2327", "3037",
    "2408", "2344", "3443", "3661", "2301", "2376", "2377", "3017", "2368", "1519",
]
US_FALLBACK = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "BRK-B", "JPM",
    "LLY", "V", "UNH", "XOM", "MA", "COST", "HD", "PG", "JNJ", "NFLX",
    "AMD", "ORCL", "CRM", "BAC", "ABBV", "KO", "PEP", "WMT", "MRK", "CVX",
    "ADBE", "TMO", "CSCO", "ACN", "MCD", "LIN", "ABT", "INTC", "QCOM", "TXN",
    "AMAT", "MU", "LRCX", "KLAC", "PLTR", "UBER", "GE", "CAT", "IBM", "NOW",
]


def _get_json(url: str):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def _num(x) -> float:
    try:
        return float(str(x).replace(",", "").strip())
    except (TypeError, ValueError):
        return 0.0


def tw_listed(top_n: int = 200) -> tuple[dict[str, str], str]:
    """上市(TWSE)依當日成交值排序，回傳 {yahoo代號: 名稱}。"""
    try:
        rows = _get_json("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL")
        out = []
        for r in rows:
            code = str(r.get("Code", "")).strip()
            # 只收一般股票：4 碼數字且不是 00 開頭的 ETF
            if len(code) == 4 and code.isdigit() and not code.startswith("0"):
                out.append((code, str(r.get("Name", "")).strip(), _num(r.get("TradeValue"))))
        out.sort(key=lambda t: t[2], reverse=True)
        if len(out) < 50:
            raise ValueError(f"only {len(out)} rows")
        return {f"{c}.TW": n for c, n, _ in out[:top_n]}, "TWSE OpenAPI STOCK_DAY_ALL"
    except Exception as e:  # noqa: BLE001
        log.warning("TWSE list failed: %s; using fallback", e)
        return {f"{c}.TW": "" for c in TW_FALLBACK}, f"備用清單(TWSE 取得失敗: {e})"


def tw_otc(top_n: int = 60) -> tuple[dict[str, str], str]:
    """上櫃(TPEx)依當日成交金額排序。"""
    try:
        rows = _get_json("https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes")
        out = []
        for r in rows:
            keys = {k.lower(): k for k in r}
            code_k = next((keys[k] for k in keys if "code" in k), None)
            name_k = next((keys[k] for k in keys if "name" in k), None)
            amt_k = next((keys[k] for k in keys if "amount" in k or "value" in k), None)
            if not code_k or not amt_k:
                continue
            code = str(r[code_k]).strip()
            if len(code) == 4 and code.isdigit() and not code.startswith("0"):
                out.append((code, str(r.get(name_k, "")).strip(), _num(r[amt_k])))
        out.sort(key=lambda t: t[2], reverse=True)
        if len(out) < 20:
            raise ValueError(f"only {len(out)} rows")
        return {f"{c}.TWO": n for c, n, _ in out[:top_n]}, "TPEx OpenAPI mainboard_quotes"
    except Exception as e:  # noqa: BLE001
        log.warning("TPEx list failed: %s", e)
        return {}, f"略過(TPEx 取得失敗: {e})"


def us_sp500() -> tuple[dict[str, str], str]:
    """標普500成分股(維基百科)。"""
    try:
        html = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=HEADERS, timeout=30
        ).text
        df = pd.read_html(io.StringIO(html))[0]
        syms = {str(s).replace(".", "-").strip(): str(n) for s, n in zip(df["Symbol"], df["Security"])}
        if len(syms) < 400:
            raise ValueError(f"only {len(syms)} rows")
        return syms, "Wikipedia S&P 500 成分股"
    except Exception as e:  # noqa: BLE001
        log.warning("S&P500 list failed: %s; using fallback", e)
        return {s: "" for s in US_FALLBACK}, f"備用清單(S&P500 取得失敗: {e})"


def build_universe() -> tuple[pd.DataFrame, dict]:
    """回傳 DataFrame[symbol, name, group] 與來源說明。"""
    sources = {}
    rows = []

    tw, sources["tw_listed"] = tw_listed()
    otc, sources["tw_otc"] = tw_otc()
    us, sources["us"] = us_sp500()

    for s, n in tw.items():
        rows.append((s, n, "台股"))
    for s, n in otc.items():
        rows.append((s, n, "台股"))
    for s, n in us.items():
        rows.append((s, n, "美股"))
    for s, n in US_ETF.items():
        rows.append((s, n, "美股ETF"))
    for s, n in INDICES.items():
        rows.append((s, n, "指數"))
    for s, n in FUTURES.items():
        rows.append((s, n, "期貨商品"))
    for s, n in FX.items():
        rows.append((s, n, "外匯"))
    for s, n in CRYPTO.items():
        rows.append((s, n, "加密貨幣"))

    df = pd.DataFrame(rows, columns=["symbol", "name", "group"]).drop_duplicates("symbol")
    return df.reset_index(drop=True), sources
