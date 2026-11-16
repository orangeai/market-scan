"""每日掃描主程式：python -m scan.run [--out data]"""
from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import fetch, indicators, screens, universe

log = logging.getLogger("scan")
TPE = ZoneInfo("Asia/Taipei")
PARTIAL_GROUPS = {"期貨商品", "外匯", "加密貨幣"}  # 24 小時或跨時區交易，最後一根可能未收完
STOCK_GROUPS = {"台股", "美股"}
BENCH = {"台股": "^TWII", "美股": "^GSPC", "美股ETF": "^GSPC"}
TOP_N = 15
MAX_PER_GROUP = 6
MIN_SCORE = 6

DAILY_COLS = ["Open", "High", "Low", "Close", "Volume", "MA5", "MA10", "MA20", "MA50", "MA60", "MA150",
              "MA200", "RSI14", "MACD_DIF", "MACD_DEA", "MACD_HIST", "K9", "D9", "ATR14", "BB_UP", "BB_LOW",
              "KC_UP", "KC_LOW", "SQUEEZE_ON", "VOL_MA50"]


def _jd(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1, default=_jd)


def safe(sym: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", sym)


def write_csv(df: pd.DataFrame, path: Path, tail: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.tail(tail).round(4).to_csv(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    now_utc = datetime.now(timezone.utc)
    now_tpe = now_utc.astimezone(TPE)
    out = Path(args.out)
    latest = out / "latest"
    latest.mkdir(parents=True, exist_ok=True)

    uni, sources = universe.build_universe()
    log.info("universe: %d symbols", len(uni))
    meta_of = uni.set_index("symbol").to_dict("index")

    daily, failed = fetch.download(uni["symbol"].tolist(), period="2y", interval="1d")
    log.info("daily ok=%d failed=%d", len(daily), len(failed))

    # ---- 移除未收完的最後一根 -------------------------------------------------
    partial_info = {}
    today_utc = now_utc.date()
    for sym, df in list(daily.items()):
        g = meta_of[sym]["group"]
        if g in PARTIAL_GROUPS and len(df) and df.index[-1].date() >= today_utc:
            partial_info[sym] = {"date": str(df.index[-1].date()), "price": float(df["Close"].iloc[-1])}
            daily[sym] = df.iloc[:-1]

    # ---- 流動性篩選(股票依 20 日平均成交值) ----------------------------------
    turnover = {s: float((d["Close"] * d["Volume"]).tail(20).mean()) for s, d in daily.items()}
    keep = set(daily)
    for g, n in (("台股", 180), ("美股", 150)):
        syms = [s for s in daily if meta_of[s]["group"] == g]
        syms.sort(key=lambda s: turnover.get(s, 0), reverse=True)
        keep -= set(syms[n:])

    # ---- 指標與篩選 -----------------------------------------------------------
    enriched, results = {}, {}
    for sym in sorted(keep):
        df = daily[sym]
        if len(df) < 80:
            continue
        try:
            e = indicators.add_all(df)
            enriched[sym] = e
            results[sym] = screens.analyse(e)
        except Exception as ex:  # noqa: BLE001
            log.warning("analyse %s failed: %s", sym, ex)

    # ---- 相對強度(63 日報酬相對基準，於同類中排百分位) ---------------------
    def ret(sym, n=63):
        d = daily.get(sym)
        return float(d["Close"].iloc[-1] / d["Close"].iloc[-n - 1] - 1) if d is not None and len(d) > n else np.nan

    rs_raw = {}
    for sym in results:
        g = meta_of[sym]["group"]
        b = BENCH.get(g)
        rs_raw[sym] = ret(sym) - (ret(b) if b else 0.0)
    for g in set(meta_of[s]["group"] for s in results):
        syms = [s for s in results if meta_of[s]["group"] == g and not np.isnan(rs_raw[s])]
        ranks = pd.Series({s: rs_raw[s] for s in syms}).rank(pct=True) * 100
        for s in syms:
            pct = round(float(ranks[s]), 1)
            r = results[s]
            r["rs_pct_in_group"] = pct
            r["ret63_pct"] = round(ret(s) * 100, 2)
            if r["direction"] == "多":
                bonus = 2 if pct >= 80 else 1 if pct >= 60 else 0
            elif r["direction"] == "空":
                bonus = 2 if pct <= 20 else 1 if pct <= 40 else 0
            else:
                bonus = 0
            if bonus:
                r["score"] += bonus
                r["reasons"].append(f"相對強度百分位 {pct}")

    # ---- 候選名單 -------------------------------------------------------------
    pool = [s for s, r in results.items() if r["direction"] != "無" and r["stage"] and r["score"] >= MIN_SCORE]
    pool.sort(key=lambda s: (results[s]["score"], {"蓄勢": 2, "剛發動": 1}.get(results[s]["stage"], 0)),
              reverse=True)
    picked, per_group = [], {}
    for s in pool:
        g = meta_of[s]["group"]
        if per_group.get(g, 0) >= MAX_PER_GROUP:
            continue
        picked.append(s)
        per_group[g] = per_group.get(g, 0) + 1
        if len(picked) >= TOP_N:
            break

    # ---- 60 分 K(候選 + 關鍵指標) ---------------------------------------------
    bar_syms = list(dict.fromkeys(picked + [s for s in universe.KEY_SYMBOLS if s in enriched]))
    intraday, failed_60m = fetch.download(bar_syms, period="60d", interval="60m", chunk=20)

    # ---- 輸出 -----------------------------------------------------------------
    bars_dir = latest / "bars"
    if bars_dir.exists():
        for f in bars_dir.glob("*.csv"):
            f.unlink()
    for s in bar_syms:
        e = enriched[s]
        write_csv(e[DAILY_COLS], bars_dir / f"{safe(s)}_D.csv", 260)
        write_csv(indicators.to_weekly(daily[s]), bars_dir / f"{safe(s)}_W.csv", 104)
        if s in intraday:
            h = intraday[s]
            idx = pd.to_datetime(h.index)
            if idx.tz is not None:
                idx = idx.tz_convert(TPE).tz_localize(None)
            h.index = idx
            h.index.name = "DateTime_Taipei"
            write_csv(indicators.add_intraday(h), bars_dir / f"{safe(s)}_60m.csv", 300)

    def card(s):
        m = meta_of[s]
        d = {"symbol": s, "name": m["name"], "group": m["group"], **results[s]}
        d["files"] = {"daily": f"bars/{safe(s)}_D.csv", "weekly": f"bars/{safe(s)}_W.csv",
                      "h60": f"bars/{safe(s)}_60m.csv" if s in intraday else None}
        if s in partial_info:
            d["partial_latest"] = partial_info[s]
        return d

    candidates = [card(s) for s in picked]
    (latest / "candidates.json").write_text(dumps(candidates))

    key = []
    for s in universe.KEY_SYMBOLS:
        if s in daily and len(daily[s]) > 1:
            d = daily[s]
            key.append({"symbol": s, "name": meta_of[s]["name"], "date": str(d.index[-1].date()),
                        "close": round(float(d["Close"].iloc[-1]), 4),
                        "chg_pct": round(float(d["Close"].iloc[-1] / d["Close"].iloc[-2] - 1) * 100, 2),
                        "partial_latest": partial_info.get(s)})
    (latest / "key_symbols.json").write_text(dumps(key))

    flat = []
    for s, r in results.items():
        flat.append({"symbol": s, "name": meta_of[s]["name"], "group": meta_of[s]["group"], "date": r["date"],
                     "close": r["close"], "direction": r["direction"], "stage": r["stage"], "score": r["score"],
                     "pivot": r["pivot"], "dist_to_pivot_pct": r["dist_to_pivot_pct"],
                     "trend_long": f"{r['trend_long']['passed']}/{r['trend_long']['total']}",
                     "trend_short": r["trend_short"]["ok"], "vcp": r["vcp"].get("ok"),
                     "squeeze_on": r["squeeze_on"], "squeeze_fired": r["squeeze_fired"],
                     "ma_tangle": r["ma_tangle"], "pocket_pivot": r["pocket_pivot"]["recent"],
                     "rsi14": r["rsi14"], "k9": r["kd"]["k"], "d9": r["kd"]["d"],
                     "macd_hist": r["macd"]["hist"], "pct_from_52w_high": r["pct_from_52w_high"],
                     "rs_pct_in_group": r.get("rs_pct_in_group"), "bars": r["bars"]})
    pd.DataFrame(flat).sort_values("score", ascending=False).to_csv(latest / "screen_all.csv", index=False)

    last_dates = {}
    for g in sorted(set(meta_of[s]["group"] for s in daily)):
        ds = [daily[s].index[-1].date() for s in daily if meta_of[s]["group"] == g and len(daily[s])]
        if ds:
            last_dates[g] = str(pd.Series(ds).mode().iloc[0])
    meta = {
        "generated_at_taipei": now_tpe.strftime("%Y-%m-%d %H:%M"),
        "generated_at_utc": now_utc.strftime("%Y-%m-%d %H:%M"),
        "data_source": "Yahoo Finance(yfinance),日K 為分割調整、未還原除權息",
        "universe_sources": sources,
        "universe_count": len(uni),
        "downloaded": len(daily),
        "analysed": len(results),
        "failed_daily": failed,
        "failed_60m": failed_60m,
        "last_bar_date_by_group": last_dates,
        "partial_bars_removed": partial_info,
        "notes": [
            "期貨、外匯、加密貨幣若最後一根日K尚未收完，已移除不納入判斷，最新價見 partial_bars_removed。",
            "週K 由日K 合併(週五為一週結束);週一至週四執行時，最後一根週K 為本週未完成的週K。",
            "外匯無真實成交量，量能相關訊號一律不適用;期貨成交量受換月影響，僅供參考。",
            "60分K 時間為台北時間。",
        ],
        "params": {"near_pivot_pct": 3, "fresh_bars": 3, "fresh_max_pct": 5, "breakout_vol_x": 1.5,
                   "min_score": MIN_SCORE, "top_n": TOP_N, "max_per_group": MAX_PER_GROUP},
    }
    (latest / "meta.json").write_text(dumps(meta))

    lines = [f"# 蓄勢候選清單 {meta['generated_at_taipei']}(台北時間)", "",
             "| # | 商品 | 類別 | 多/空 | 階段 | 分數 | 收盤 | 關鍵價 | 距關鍵價% | 依據 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for i, c in enumerate(candidates, 1):
        lines.append(f"| {i} | {c['symbol']} {c['name']} | {c['group']} | {c['direction']} | {c['stage']} | "
                     f"{c['score']} | {c['close']} | {c['pivot']} | {c['dist_to_pivot_pct']} | "
                     f"{'、'.join(c['reasons'])} |")
    lines += ["", "資料日期(各類別最後一根日K):", ""] + [f"- {g}: {d}" for g, d in last_dates.items()]
    (latest / "candidates.md").write_text("\n".join(lines) + "\n")

    hist = out / "history" / now_tpe.strftime("%Y-%m-%d")
    hist.mkdir(parents=True, exist_ok=True)
    (hist / "candidates.json").write_text(dumps(candidates))
    (hist / "key_symbols.json").write_text(dumps(key))
    (hist / "meta.json").write_text(dumps(meta))
    dates = sorted(p.name for p in (out / "history").iterdir() if p.is_dir())
    (out / "history" / "index.json").write_text(json.dumps(dates, indent=1))
    log.info("done: %d candidates", len(candidates))


if __name__ == "__main__":
    main()
