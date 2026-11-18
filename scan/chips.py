"""台股籌碼：三大法人買賣超、融資融券餘額(證交所 / 櫃買中心，每日全市場一次抓)。

資料公布時間：法人約 15:00–16:30、融資融券約 21:00 後，因此由獨立排程(台北 21:40)更新
data/chips/tw_chips.csv;14:30 的掃描使用到「前一交易日」為止的籌碼。

用法：
  python -m scan.chips --update                # 補到今天(從檔案最後日期往後)
  python -m scan.chips --backfill-days 500     # 往回補抓(只抓檔案裡沒有的日期)
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .universe import HEADERS

log = logging.getLogger(__name__)
TPE = ZoneInfo("Asia/Taipei")
ROOT = Path(__file__).resolve().parents[1]
CHIPS_CSV = ROOT / "data" / "chips" / "tw_chips.csv"
COLS = ["date", "symbol", "foreign", "trust", "dealer", "total", "margin_bal", "margin_chg", "short_bal", "short_chg"]
TWSE_SLEEP = 4.0   # 證交所請求間隔(秒),太快會被暫時封鎖
TPEX_SLEEP = 1.5


def _n(x) -> float | None:
    try:
        return float(str(x).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _get(url: str, retries: int = 3, sleep: float = 2.0):
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.headers.get("content-type", "").startswith("text/html"):
                raise RuntimeError("HTML 回應(可能被限流)")
            return r.json()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(sleep * (i + 2) * 3)
    raise RuntimeError(f"{url}: {last}")


def _idx(fields: list[str], *keys: str) -> int | None:
    """依欄位名稱找索引(全部關鍵字都要出現)。"""
    for i, f in enumerate(fields):
        if all(k in f for k in keys):
            return i
    return None


def twse_inst(d: date) -> pd.DataFrame:
    j = _get(f"https://www.twse.com.tw/rwd/zh/fund/T86?date={d:%Y%m%d}&selectType=ALLBUT0999&response=json")
    if j.get("stat") != "OK" or not j.get("data"):
        return pd.DataFrame()
    f = j["fields"]
    i_code = 0
    i_fx = _idx(f, "外陸資買賣超", "不含")
    i_fd = _idx(f, "外資自營商買賣超")
    i_tr = _idx(f, "投信買賣超")
    i_dl = next(i for i, x in enumerate(f) if x.startswith("自營商買賣超"))
    i_tot = _idx(f, "三大法人買賣超")
    rows = []
    for r in j["data"]:
        fx = (_n(r[i_fx]) or 0) + ((_n(r[i_fd]) or 0) if i_fd is not None else 0)
        rows.append({"symbol": f"{r[i_code].strip()}.TW", "foreign": fx, "trust": _n(r[i_tr]),
                     "dealer": _n(r[i_dl]), "total": _n(r[i_tot])})
    return pd.DataFrame(rows)


def twse_margin(d: date) -> pd.DataFrame:
    j = _get(f"https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?date={d:%Y%m%d}&selectType=ALL&response=json")
    if j.get("stat") != "OK":
        return pd.DataFrame()
    t = next((t for t in j.get("tables", []) if t.get("fields") and t["fields"][0] == "代號"), None)
    if not t or not t.get("data"):
        return pd.DataFrame()
    # 欄位：代號 名稱 | 融資 買進 賣出 現金償還 前日餘額 今日餘額 限額 | 融券 買進 賣出 現券償還 前日餘額 今日餘額 限額 | 資券互抵
    rows = []
    for r in t["data"]:
        mp, mb, sp, sb = _n(r[5]), _n(r[6]), _n(r[11]), _n(r[12])
        rows.append({"symbol": f"{r[0].strip()}.TW", "margin_bal": mb, "margin_chg": (mb - mp) if None not in (mb, mp) else None,
                     "short_bal": sb, "short_chg": (sb - sp) if None not in (sb, sp) else None})
    return pd.DataFrame(rows)


def tpex_inst(d: date) -> pd.DataFrame:
    j = _get(f"https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade?type=Daily&sect=EW&date={d:%Y/%m/%d}&response=json")
    t = (j.get("tables") or [{}])[0]
    if j.get("stat") != "ok" or not t.get("data"):
        return pd.DataFrame()
    # 代號 名稱 | 外資(不含自營) 買 賣 超 | 外資自營 買 賣 超 | 外資合計 買 賣 超 | 投信 買 賣 超 |
    # 自營(自行) 買 賣 超 | 自營(避險) 買 賣 超 | 自營合計 買 賣 超 | 三大法人合計
    rows = [{"symbol": f"{r[0].strip()}.TWO", "foreign": _n(r[10]), "trust": _n(r[13]), "dealer": _n(r[22]),
             "total": _n(r[23])} for r in t["data"] if len(r) >= 24]
    return pd.DataFrame(rows)


def tpex_margin(d: date) -> pd.DataFrame:
    j = _get(f"https://www.tpex.org.tw/www/zh-tw/margin/balance?date={d:%Y/%m/%d}&response=json")
    t = (j.get("tables") or [{}])[0]
    if j.get("stat") != "ok" or not t.get("data"):
        return pd.DataFrame()
    # 代號 名稱 前資餘額 資買 資賣 現償 資餘額 資屬證金 資使用率 資限額 前券餘額 券賣 券買 券償 券餘額 ...
    rows = []
    for r in t["data"]:
        mp, mb, sp, sb = _n(r[2]), _n(r[6]), _n(r[10]), _n(r[14])
        rows.append({"symbol": f"{r[0].strip()}.TWO", "margin_bal": mb, "margin_chg": (mb - mp) if None not in (mb, mp) else None,
                     "short_bal": sb, "short_chg": (sb - sp) if None not in (sb, sp) else None})
    return pd.DataFrame(rows)


def fetch_day(d: date, keep: set[str] | None = None) -> pd.DataFrame | None:
    """抓一天全市場籌碼;非交易日回傳 None。keep 指定只保留的代號。"""
    parts = []
    ti = twse_inst(d); time.sleep(TWSE_SLEEP)
    tm = twse_margin(d); time.sleep(TWSE_SLEEP)
    oi = tpex_inst(d); time.sleep(TPEX_SLEEP)
    om = tpex_margin(d); time.sleep(TPEX_SLEEP)
    if ti.empty and oi.empty:
        return None
    for inst, marg in ((ti, tm), (oi, om)):
        if inst.empty and marg.empty:
            continue
        if inst.empty:
            parts.append(marg)
        elif marg.empty:
            parts.append(inst)
        else:
            parts.append(inst.merge(marg, on="symbol", how="outer"))
    df = pd.concat(parts, ignore_index=True)
    if keep is not None:
        df = df[df.symbol.isin(keep)]
    df.insert(0, "date", d.isoformat())
    return df.reindex(columns=COLS)


def load(path: Path = CHIPS_CSV) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLS)
    return pd.read_csv(path, dtype={"symbol": str})


def save(df: pd.DataFrame, path: Path = CHIPS_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.drop_duplicates(["date", "symbol"], keep="last").sort_values(["date", "symbol"])
    df.to_csv(path, index=False, float_format="%.0f")


def pool_symbols() -> set[str]:
    """籌碼保留範圍：目前觀察池裡的台股 + 檔案裡已有的代號。"""
    syms: set[str] = set()
    sa = ROOT / "data" / "latest" / "screen_all.csv"
    if sa.exists():
        s = pd.read_csv(sa, usecols=["symbol"]).symbol
        syms |= {x for x in s if str(x).endswith((".TW", ".TWO"))}
    if not syms:
        from . import universe
        uni, _ = universe.build_universe()
        syms |= {x for x in uni.symbol if str(x).endswith((".TW", ".TWO"))}
    return syms


def update(days: list[date], keep: set[str], max_minutes: float = 300) -> int:
    hist = load()
    have = set(hist.date.unique())
    new, t0 = [], time.time()
    for d in days:
        if d.weekday() >= 5 or d.isoformat() in have:
            continue
        if (time.time() - t0) / 60 > max_minutes:
            log.warning("time budget reached at %s", d)
            break
        try:
            df = fetch_day(d, keep | set(hist.symbol.unique()))
        except Exception as e:  # noqa: BLE001
            log.warning("chips %s failed: %s", d, e)
            continue
        if df is None:
            log.info("chips %s: 非交易日", d)
            continue
        if df.margin_bal.isna().all() and d >= datetime.now(TPE).date() - timedelta(days=1):
            log.info("chips %s: 融資融券尚未公布，稍後再補", d)
            continue
        log.info("chips %s: %d 檔", d, len(df))
        new.append(df)
        if len(new) % 20 == 0:  # 長時間補抓時定期存檔
            save(pd.concat([hist] + new, ignore_index=True))
    if new:
        save(pd.concat([hist] + new, ignore_index=True))
    return len(new)


def features(hist: pd.DataFrame, symbol: str, avg_vol_shares: float | None, asof: str | None = None) -> dict | None:
    """單一代號的籌碼摘要(張 = 1000 股)。asof:只用此日期(含)以前的資料。"""
    h = hist[hist.symbol == symbol]
    if asof:
        h = h[h.date <= asof]
    if h.empty:
        return None
    h = h.sort_values("date")
    last = h.iloc[-1]
    t5 = h.tail(5)

    def streak(col):
        s, sign = 0, 0
        for v in reversed(h[col].fillna(0).tolist()):
            cur = 1 if v > 0 else -1 if v < 0 else 0
            if cur == 0 or (sign and cur != sign):
                break
            sign = cur; s += 1
        return s * sign

    def pct_vol(x):
        return round(x / (avg_vol_shares * len(t5)) * 100, 2) if avg_vol_shares else None

    mb = h.margin_bal.dropna()
    m5 = round((mb.iloc[-1] / mb.iloc[-6] - 1) * 100, 2) if len(mb) >= 6 and mb.iloc[-6] else None
    return {
        "date": last.date,
        "foreign_lots": round((last.foreign or 0) / 1000), "trust_lots": round((last.trust or 0) / 1000),
        "dealer_lots": round((last.dealer or 0) / 1000),
        "foreign_streak": streak("foreign"), "trust_streak": streak("trust"),
        "foreign_5d_pct_vol": pct_vol(t5.foreign.fillna(0).sum()), "trust_5d_pct_vol": pct_vol(t5.trust.fillna(0).sum()),
        "inst_5d_pct_vol": pct_vol(t5.total.fillna(0).sum()),
        "margin_bal": None if pd.isna(last.margin_bal) else int(last.margin_bal),
        "margin_5d_chg_pct": m5,
        "short_margin_ratio_pct": round(last.short_bal / last.margin_bal * 100, 2)
        if last.margin_bal and not pd.isna(last.short_bal) else None,
    }


def apply_score(r: dict, df: pd.DataFrame | None = None) -> None:
    """依 backtest/chips_study 的結果調整多方分數(2024-11~2026-10 台股 176 檔，見 docs/backtest_chips_2026-11.md)。
    只採用在 A突破/B蓄勢/D Stage2 隨機日或四個季度大多同向的訊號;其餘只標記。"""
    c = r.get("chips")
    if not c or r.get("direction") != "多":
        return
    bo = (r.get("breakout") or {}).get("found")
    f5, ts = c.get("foreign_5d_pct_vol"), c.get("trust_streak") or 0
    # 突破 + 外資 5 日買超 ≥ 5% 均量:+0.148R(t=1.69),四季中三季為正
    if bo and f5 is not None and f5 >= 5:
        r["score"] += 1
        r["reasons"].append("突破+外資5日買超≥5%量")
    # 投信連買 ≥ 3 日：後續偏弱(突破 -0.20R t=-2.0、蓄勢 -0.13R、Stage2 隨機日 -0.14R t=-3.0)
    if ts >= 3:
        r["score"] -= 1
        r["reasons"].append(f"投信連買{ts}日(回測偏弱)")
    # 蓄勢 + 融資 5 日增 > 5% 而股價 5 日漲 < 2%:樣本少(N=63),只標記
    m5 = c.get("margin_5d_chg_pct")
    if r.get("stage") == "蓄勢" and m5 is not None and m5 > 5 and df is not None and len(df) > 5:
        px5 = float(df["Close"].iloc[-1] / df["Close"].iloc[-6] - 1)
        if px5 < 0.02:
            r["reasons"].append("融資增價不漲(未計分)")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true", help="從檔案最後日期補到今天(最多 30 天)")
    ap.add_argument("--backfill-days", type=int, default=0, help="往回補抓的日曆天數")
    ap.add_argument("--max-minutes", type=float, default=300)
    a = ap.parse_args()
    today = datetime.now(TPE).date()
    keep = pool_symbols()
    hist = load()
    if a.backfill_days:
        days = [today - timedelta(days=i) for i in range(a.backfill_days, -1, -1)]
    else:
        start = (date.fromisoformat(hist.date.max()) + timedelta(days=1)) if len(hist) else today - timedelta(days=10)
        start = max(start, today - timedelta(days=30))
        days = [start + timedelta(days=i) for i in range((today - start).days + 1)]
    n = update(days, keep, a.max_minutes)
    log.info("added %d trading days; pool %d symbols", n, len(keep))


if __name__ == "__main__":
    main()
