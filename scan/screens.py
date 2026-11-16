"""型態篩選：趨勢模板、VCP、Squeeze、均線糾結、量縮、Pocket Pivot、突破/跌破、階段分級與評分。

所有判斷只用已收盤的 K 棒(呼叫端負責移除未完成的最後一根)。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

NEAR_PIVOT = 0.03      # 蓄勢：距關鍵價位 3% 以內
FRESH_BARS = 3         # 剛發動：突破 3 根 K 棒內
FRESH_MAX = 0.05       # 剛發動：距突破點 5% 以內
BREAK_VOL = 1.5        # 放量：成交量 >= 50 日均量 1.5 倍
LOOKBACK_PIVOT = 20    # 突破/跌破參考：前 20 日高/低


def _f(x):
    """轉成可 JSON 化的數值。"""
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else round(v, 4)


def has_volume(df: pd.DataFrame) -> bool:
    v = df["Volume"].tail(50)
    return len(v) >= 20 and (v > 0).mean() >= 0.8


def swings(df: pd.DataFrame, k: int = 3, lookback: int = 120):
    """回傳近期波段高/低點 [(位置, 價格, 'H'|'L')],依時間排序，高低交替。"""
    d = df.tail(lookback)
    h, l = d["High"].to_numpy(), d["Low"].to_numpy()
    pts = []
    for i in range(k, len(d) - k):
        if h[i] == h[i - k:i + k + 1].max():
            pts.append((i, h[i], "H"))
        if l[i] == l[i - k:i + k + 1].min():
            pts.append((i, l[i], "L"))
    pts.sort(key=lambda p: p[0])
    alt = []
    for p in pts:  # 合併連續同類點，保留較極端者
        if alt and alt[-1][2] == p[2]:
            if (p[2] == "H" and p[1] >= alt[-1][1]) or (p[2] == "L" and p[1] <= alt[-1][1]):
                alt[-1] = p
        else:
            alt.append(p)
    offset = len(df) - len(d)
    return [(i + offset, v, t) for i, v, t in alt]


def trend_long(df: pd.DataFrame) -> dict:
    r = df.iloc[-1]
    n = len(df)
    if n >= 220 and not pd.isna(r["MA200"]):
        ma200_prev = df["MA200"].iloc[-21]
        checks = {
            "收盤>150日線且>200日線": r["Close"] > r["MA150"] and r["Close"] > r["MA200"],
            "150日線>200日線": r["MA150"] > r["MA200"],
            "200日線上揚(對比20日前)": r["MA200"] > ma200_prev,
            "50日線>150日線且>200日線": r["MA50"] > r["MA150"] and r["MA50"] > r["MA200"],
            "收盤>50日線": r["Close"] > r["MA50"],
            "高於52週低點30%以上": r["Close"] >= 1.3 * r["LOW_52W"],
            "距52週高點25%以內": r["Close"] >= 0.75 * r["HIGH_52W"],
        }
        passed = sum(bool(v) for v in checks.values())
        return {"mode": "完整趨勢模板", "passed": passed, "total": 7, "ok": passed == 7,
                "partial": passed >= 5, "checks": {k: bool(v) for k, v in checks.items()}}
    if n >= 80 and not pd.isna(r["MA60"]):
        checks = {
            "收盤>20日線": r["Close"] > r["MA20"],
            "收盤>60日線": r["Close"] > r["MA60"],
            "60日線上揚(對比10日前)": r["MA60"] > df["MA60"].iloc[-11],
        }
        passed = sum(bool(v) for v in checks.values())
        return {"mode": "簡化版(K棒不足200根)", "passed": passed, "total": 3, "ok": passed == 3,
                "partial": False, "checks": {k: bool(v) for k, v in checks.items()}}
    return {"mode": "資料不足", "passed": 0, "total": 0, "ok": False, "partial": False, "checks": {}}


def trend_short(df: pd.DataFrame, sw) -> dict:
    r = df.iloc[-1]
    if len(df) < 80 or pd.isna(r["MA50"]):
        return {"ok": False, "fresh": False, "checks": {}}
    long_ma = "MA200" if not pd.isna(r["MA200"]) else "MA60"
    highs = [v for _, v, t in sw if t == "H"]
    checks = {
        "收盤<50日線": r["Close"] < r["MA50"],
        "50日線下彎(對比10日前)": r["MA50"] < df["MA50"].iloc[-11],
        f"收盤<{long_ma.replace('MA', '')}日線": r["Close"] < r[long_ma],
        "高點降低(lower high)": len(highs) >= 2 and highs[-1] < highs[-2],
    }
    ok = all(checks.values())
    # 剛轉空：近 30 根內 50日線跌破長均線，或近 20 根內收盤跌破長均線
    ma_mid = "MA150" if not pd.isna(r["MA150"]) else "MA60"
    cross_ma = (df["MA50"] < df[ma_mid]) & (df["MA50"].shift() >= df[ma_mid].shift())
    cross_px = (df["Close"] < df[long_ma]) & (df["Close"].shift() >= df[long_ma].shift())
    fresh = bool(cross_ma.tail(30).any() or cross_px.tail(20).any())
    return {"ok": bool(ok), "fresh": fresh, "checks": {k: bool(v) for k, v in checks.items()}}


def vcp(df: pd.DataFrame, sw, vol_ok: bool) -> dict:
    """簡化 VCP：基底內回檔幅度逐次縮小、低點不破、最後一次收縮量縮。"""
    pts = [p for p in sw if p[0] >= len(df) - 90]
    if not pts:
        return {"ok": False}
    hi_pos = max((p for p in pts if p[2] == "H"), key=lambda p: p[1], default=None)
    if hi_pos is None:
        return {"ok": False}
    seq = [p for p in pts if p[0] >= hi_pos[0]]
    pulls = []  # (高點位置, 高點, 低點位置, 低點, 回檔幅度)
    for a, b in zip(seq, seq[1:]):
        if a[2] == "H" and b[2] == "L":
            pulls.append((a[0], a[1], b[0], b[1], (a[1] - b[1]) / a[1]))
    if len(pulls) < 2:
        return {"ok": False, "contractions": [round(p[4] * 100, 1) for p in pulls]}
    depths = [p[4] for p in pulls][-4:]
    contracting = all(d2 < d1 * 0.9 for d1, d2 in zip(depths, depths[1:]))
    higher_lows = pulls[-1][3] >= pulls[-2][3]
    last = pulls[-1]
    dry = None
    if vol_ok:
        seg = df["Volume"].iloc[last[0]:last[2] + 1]
        base = df["VOL_MA50"].iloc[last[2]]
        dry = bool(len(seg) and base and seg.mean() < 0.8 * base)
    ok = contracting and higher_lows and depths[0] <= 0.40 and depths[-1] <= 0.15 and (dry is not False)
    pivot = float(last[1])
    return {
        "ok": bool(ok),
        "contractions": [round(d * 100, 1) for d in depths],
        "pivot": _f(pivot),
        "last_low": _f(last[3]),
        "volume_dry_up": dry,
        "pivot_date": str(df.index[last[0]].date()),
    }


def pocket_pivot(df: pd.DataFrame, vol_ok: bool) -> dict:
    if not vol_ok or len(df) < 60:
        return {"recent": False, "dates": []}
    c, v, ma50, ma10 = df["Close"], df["Volume"], df["MA50"], df["MA10"]
    hits = []
    for i in range(len(df) - FRESH_BARS, len(df)):
        if c.iloc[i] <= c.iloc[i - 1]:
            continue
        win = df.iloc[i - 10:i]
        prev_close = df["Close"].iloc[i - 11:i - 1].to_numpy()
        downs = [win["Volume"].iloc[j] for j in range(len(win)) if win["Close"].iloc[j] < prev_close[j]]
        mx = max(downs) if downs else 0
        if v.iloc[i] > mx and c.iloc[i] > ma50.iloc[i] and c.iloc[i] >= ma10.iloc[i]:
            hits.append(str(df.index[i].date()))
    return {"recent": bool(hits), "dates": hits}


def breakout(df: pd.DataFrame, vol_ok: bool, direction: str) -> dict:
    """近 3 根內是否放量突破前 20 日高(做多)或跌破前 20 日低(做空)。"""
    for back in range(1, FRESH_BARS + 1):
        i = len(df) - back
        if i - LOOKBACK_PIVOT < 0:
            break
        prior = df.iloc[i - LOOKBACK_PIVOT:i]
        row = df.iloc[i]
        vol_cond = (not vol_ok) or (row["Volume"] >= BREAK_VOL * (df["VOL_MA50"].iloc[i] or np.inf))
        if direction == "long" and row["Close"] > prior["High"].max() and vol_cond:
            return {"found": True, "date": str(df.index[i].date()), "bars_ago": back - 1,
                    "level": _f(prior["High"].max()), "volume_confirmed": vol_ok}
        if direction == "short" and row["Close"] < prior["Low"].min() and vol_cond:
            return {"found": True, "date": str(df.index[i].date()), "bars_ago": back - 1,
                    "level": _f(prior["Low"].min()), "volume_confirmed": vol_ok}
    return {"found": False}


def analyse(df: pd.DataFrame) -> dict:
    """單一商品的完整篩選結果(df 需已含 indicators.add_all 欄位)。"""
    r = df.iloc[-1]
    vol_ok = has_volume(df)
    sw = swings(df)
    tl, ts = trend_long(df), trend_short(df, sw)

    sq_on = bool(r["SQUEEZE_ON"])
    sq_len = 0
    for v in df["SQUEEZE_ON"].iloc[::-1]:
        if not v:
            break
        sq_len += 1
    sq_fired = (not sq_on) and bool(df["SQUEEZE_ON"].iloc[-2]) if len(df) > 2 else False
    ma3 = [r["MA5"], r["MA10"], r["MA20"]]
    tangle = (not any(pd.isna(ma3))) and (max(ma3) - min(ma3)) / r["Close"] <= 0.03
    dry = vol_ok and r["VOL_MA50"] and df["Volume"].tail(5).mean() <= 0.6 * r["VOL_MA50"]
    rng20 = (df["High"].tail(20).max() - df["Low"].tail(20).min()) / r["Close"]

    v = vcp(df, sw, vol_ok)
    pp = pocket_pivot(df, vol_ok)

    hh20 = df["High"].iloc[-LOOKBACK_PIVOT - 1:-1].max()
    ll20 = df["Low"].iloc[-LOOKBACK_PIVOT - 1:-1].min()

    if tl["ok"] or (tl["partial"] and not ts["ok"]):
        direction = "long"
    elif ts["ok"]:
        direction = "short"
    else:
        direction = None

    res = {
        "date": str(df.index[-1].date()),
        "close": _f(r["Close"]),
        "bars": len(df),
        "has_volume": vol_ok,
        "direction": {"long": "多", "short": "空"}.get(direction, "無"),
        "trend_long": tl,
        "trend_short": ts,
        "squeeze_on": sq_on,
        "squeeze_bars": sq_len,
        "squeeze_fired": bool(sq_fired),
        "ma_tangle": bool(tangle),
        "volume_dry_up": bool(dry) if vol_ok else None,
        "range20_pct": _f(rng20 * 100),
        "vcp": v,
        "pocket_pivot": pp,
        "high20": _f(hh20),
        "low20": _f(ll20),
        "high52w": _f(r["HIGH_52W"]),
        "low52w": _f(r["LOW_52W"]),
        "pct_from_52w_high": _f((r["Close"] / r["HIGH_52W"] - 1) * 100) if r["HIGH_52W"] else None,
        "ma": {k: _f(r[k]) for k in ("MA5", "MA10", "MA20", "MA50", "MA60", "MA150", "MA200")},
        "rsi14": _f(r["RSI14"]),
        "macd": {"dif": _f(r["MACD_DIF"]), "dea": _f(r["MACD_DEA"]), "hist": _f(r["MACD_HIST"])},
        "kd": {"k": _f(r["K9"]), "d": _f(r["D9"])},
        "atr14": _f(r["ATR14"]),
        "vol_ratio": _f(r["Volume"] / r["VOL_MA50"]) if vol_ok and r["VOL_MA50"] else None,
        "swing_points": [{"date": str(df.index[i].date()), "price": _f(p), "type": t} for i, p, t in sw[-8:]],
    }

    stage, pivot, score, reasons = None, None, 0.0, []
    if direction == "long":
        bo = breakout(df, vol_ok, "long")
        pivot = v.get("pivot") if v.get("ok") else _f(hh20)
        if bo["found"]:
            lvl = bo["level"]
            ext = r["Close"] / lvl - 1
            stage = "剛發動" if ext <= FRESH_MAX else "已延伸"
            pivot = lvl
        else:
            dist = (pivot - r["Close"]) / r["Close"] if pivot else None
            if dist is not None and 0 <= dist <= NEAR_PIVOT:
                stage = "蓄勢"
            elif dist is not None and dist < 0:
                stage = "已延伸" if r["Close"] > pivot * (1 + FRESH_MAX) else "剛發動"
        if not pd.isna(r["MA50"]) and r["Close"] > r["MA50"] * 1.25:
            stage = "已延伸"
            reasons.append("距50日線超過25%")
        res["breakout"] = bo
        score += 3 if tl["ok"] and tl["total"] == 7 else 2 if tl["ok"] else 1 if tl["partial"] else 0
        reasons.append(f"趨勢{tl['passed']}/{tl['total']}({tl['mode']})")
        if v.get("ok"):
            score += 3; reasons.append(f"VCP {v['contractions']}%")
        if pp["recent"]:
            score += 2; reasons.append("Pocket Pivot")
        if bo["found"]:
            score += 2; reasons.append("放量突破" if vol_ok else "突破(無量能資料)")
        prox = r["Close"] / r["HIGH_52W"] if r["HIGH_52W"] else 0
        score += 2 if prox >= 0.95 else 1 if prox >= 0.85 else 0
    elif direction == "short":
        bo = breakout(df, vol_ok, "short")
        pivot = _f(ll20)
        if bo["found"]:
            lvl = bo["level"]
            ext = 1 - r["Close"] / lvl
            stage = "剛發動" if ext <= FRESH_MAX else "已延伸"
            pivot = lvl
        else:
            dist = (r["Close"] - pivot) / r["Close"] if pivot else None
            if dist is not None and 0 <= dist <= NEAR_PIVOT:
                stage = "蓄勢"
            elif dist is not None and dist < 0:
                stage = "已延伸" if r["Close"] < pivot * (1 - FRESH_MAX) else "剛發動"
        if not pd.isna(r["MA50"]) and r["Close"] < r["MA50"] * 0.8:
            stage = "已延伸"
            reasons.append("低於50日線超過20%")
        res["breakout"] = bo
        score += 3
        reasons.append("空頭結構")
        if ts["fresh"]:
            score += 2; reasons.append("剛轉空")
        if bo["found"]:
            score += 2; reasons.append("放量跌破" if vol_ok else "跌破(無量能資料)")
        prox = r["LOW_52W"] / r["Close"] if r["Close"] and r["LOW_52W"] else 0
        score += 1 if prox >= 0.9 else 0

    if direction:
        if sq_on:
            score += 2; reasons.append(f"Squeeze 中({sq_len}根)")
        elif sq_fired:
            score += 2; reasons.append("Squeeze 剛發射")
        if tangle:
            score += 1; reasons.append("均線糾結")
        if dry:
            score += 1; reasons.append("量縮")
        score += {"蓄勢": 3, "剛發動": 2, "已延伸": -3}.get(stage, 0)

    res["stage"] = stage
    res["pivot"] = _f(pivot)
    res["dist_to_pivot_pct"] = _f((pivot / r["Close"] - 1) * 100) if pivot else None
    res["score"] = round(score, 2)
    res["reasons"] = reasons
    return res
