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

from . import fetch, indicators, screens, twse, universe

log = logging.getLogger("scan")
TPE = ZoneInfo("Asia/Taipei")
NY = ZoneInfo("America/New_York")
US_INDEX = {"^GSPC", "^IXIC", "^SOX", "^DJI"}
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

    # ---- 台股：用證交所/櫃買官方資料補上 Yahoo 尚未更新的最新一日 ------------------
    tw_daily = {s: d for s, d in daily.items() if s.endswith((".TW", ".TWO"))}
    official, official_note = twse.latest_official_bars()
    patched = twse.patch(tw_daily, official)
    daily.update(tw_daily)
    log.info("official TW patch: %s; patched %d", official_note, len(patched))

    # ---- 移除未收完的最後一根 -------------------------------------------------
    partial_info = {}
    now_ny = now_utc.astimezone(NY)
    for sym, df in list(daily.items()):
        if not len(df):
            continue
        g = meta_of[sym]["group"]
        last = df.index[-1].date()
        if g in PARTIAL_GROUPS:
            is_partial = last >= now_utc.date()
        elif g == "台股" or sym == "^TWII":
            is_partial = last == now_tpe.date() and (now_tpe.hour, now_tpe.minute) < (13, 45)
        elif g in ("美股", "美股ETF") or sym in US_INDEX:
            is_partial = last == now_ny.date() and (now_ny.hour, now_ny.minute) < (16, 15)
        else:
            is_partial = False
        if is_partial:
            partial_info[sym] = {"date": str(last), "price": float(df["Close"].iloc[-1])}
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
            b = BENCH.get(meta_of[sym]["group"])
            bench = daily[b]["Close"] if b and b in daily and b != sym else None
            results[sym] = screens.analyse(e, bench)
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

    # ---- 極端軌道提醒(S06):最新一根觸及極端下軌(跌過頭)或上軌(漲過頭) --------------
    # 回測(docs/backtest_2026-11.md):空頭股觸及下軌後 20 日平均 +5.6%(隨機日 +2.7%),反彈風險高;
    # Stage 2 股觸及下軌 5 日平均 +1.6%(隨機 +0.9%),可留意拉回買點。
    def trend_label(r):
        if r["trend_long"]["ok"]:
            return "長多"
        if r["trend_short"]["ok"]:
            return "空頭"
        return "盤整/其他"

    alerts = []
    for s, r in results.items():
        rails = r.get("extreme_rails") or {}
        for side, key_, rail in (("跌過頭", "touch_lower", "lower"), ("漲過頭", "touch_upper", "upper")):
            if rails.get(key_) and rails.get(rail):
                t = trend_label(r)
                if side == "跌過頭":
                    note = ("空頭超跌：反彈風險高，空單勿追、可考慮回補" if t == "空頭"
                            else "長多拉回到極端下軌：留意止跌後的買點" if t == "長多" else "超跌：觀察是否止跌")
                else:
                    note = ("長多加速：已偏離均線，勿追高、可考慮分批停利" if t == "長多"
                            else "空頭急彈到極端上軌：留意反彈結束" if t == "空頭" else "急漲：留意拉回")
                alerts.append({"symbol": s, "name": meta_of[s]["name"], "group": meta_of[s]["group"],
                               "type": side, "trend": t, "close": r["close"], "rail": rails[rail],
                               "m23": rails.get("m23"), "dist_pct": round((r["close"] / rails[rail] - 1) * 100, 2),
                               "note": note})
    alerts.sort(key=lambda a: (a["type"] != "跌過頭", a["dist_pct"] if a["type"] == "跌過頭" else -a["dist_pct"]))
    (latest / "alerts.json").write_text(dumps(alerts))

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
                     "rs_pct_in_group": r.get("rs_pct_in_group"), "rs_lead_recent": r.get("rs_lead_recent"),
                     "breakout_failed": r.get("breakout_failed"),
                     "rail_upper": r["extreme_rails"]["upper"], "rail_lower": r["extreme_rails"]["lower"],
                     "bars": r["bars"]})
    pd.DataFrame(flat).sort_values("score", ascending=False).to_csv(latest / "screen_all.csv", index=False)

    last_dates = {}
    for g in sorted(set(meta_of[s]["group"] for s in daily)):
        ds = [daily[s].index[-1].date() for s in daily if meta_of[s]["group"] == g and len(daily[s])]
        if ds:
            last_dates[g] = str(pd.Series(ds).mode().iloc[0])
    def expected(now_local, close_hm):
        d = now_local.date()
        if now_local.weekday() >= 5 or (now_local.hour, now_local.minute) < close_hm:
            d = d - pd.Timedelta(days=1)
        while d.weekday() >= 5:
            d = d - pd.Timedelta(days=1)
        return str(d)

    exp = {"台股": expected(now_tpe, (13, 45)), "美股": expected(now_ny, (16, 15))}
    stale = {g: f"最後日期 {last_dates.get(g)} 早於預期 {e}(可能是假日，或資料源尚未更新)"
             for g, e in exp.items() if last_dates.get(g) and last_dates[g] < e}
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
        "expected_last_date": exp,
        "stale_warning": stale,
        "tw_official_patch": {"note": official_note, "patched_count": len(patched)},
        "partial_bars_removed": partial_info,
        "notes": [
            "期貨、外匯、加密貨幣若最後一根日K尚未收完，已移除不納入判斷，最新價見 partial_bars_removed。",
            "台股最新一日若 Yahoo 未更新，會以證交所/櫃買中心 OpenAPI 官方收盤資料補上(tw_official_patch)。",
            "週K 由日K 合併(週五為一週結束);週一至週四執行時，最後一根週K 為本週未完成的週K。",
            "外匯無真實成交量，量能相關訊號一律不適用;期貨成交量受換月影響，僅供參考。",
            "60分K 時間為台北時間。",
            "extreme_rails:23 日線 ± 過去 60 日最大偏離(S06 極端軌道),可作延伸目標/超跌參考;觸及下軌的空方候選扣 2 分;全部觸及上下軌的標的另列於 alerts.json 與 candidates.md 的「極端軌道提醒」。",
            "rs_lead_recent:近 5 根內出現 RS 領先(S02);突破前 10 根內出現者加 1 分。breakout_failed:突破後收盤曾跌回突破點或 EMA20(S03),僅標記不計分(回測差異不顯著)。",
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
    for side in ("跌過頭", "漲過頭"):
        items = [a for a in alerts if a["type"] == side][:15]
        lines += ["", f"## 極端軌道提醒：{side}(最新一根觸及{'下' if side == '跌過頭' else '上'}軌,共 "
                  f"{sum(a['type'] == side for a in alerts)} 檔，列前 15)", ""]
        if not items:
            lines.append("今日無。")
            continue
        lines += ["| 商品 | 類別 | 趨勢 | 收盤 | 極端軌道 | 收盤距軌道% | 23日線 | 提醒 |", "|---|---|---|---|---|---|---|---|"]
        lines += [f"| {a['symbol']} {a['name']} | {a['group']} | {a['trend']} | {a['close']} | {a['rail']} | "
                  f"{a['dist_pct']} | {a['m23']} | {a['note']} |" for a in items]
    lines += ["", "資料日期(各類別最後一根日K):", ""] + [f"- {g}: {d}" for g, d in last_dates.items()]
    (latest / "candidates.md").write_text("\n".join(lines) + "\n")

    hist = out / "history" / now_tpe.strftime("%Y-%m-%d")
    hist.mkdir(parents=True, exist_ok=True)
    (hist / "candidates.json").write_text(dumps(candidates))
    (hist / "alerts.json").write_text(dumps(alerts))
    (hist / "key_symbols.json").write_text(dumps(key))
    (hist / "meta.json").write_text(dumps(meta))
    dates = sorted(p.name for p in (out / "history").iterdir() if p.is_dir())
    (out / "history" / "index.json").write_text(json.dumps(dates, indent=1))
    log.info("done: %d candidates", len(candidates))


if __name__ == "__main__":
    main()
