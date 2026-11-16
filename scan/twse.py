"""用證交所 / 櫃買中心官方 OpenAPI 補上 Yahoo 尚未更新的台股最新一日 K 棒。"""
from __future__ import annotations

import logging
import re
from datetime import date

import pandas as pd
import requests

from .universe import HEADERS

log = logging.getLogger(__name__)


def parse_date(s) -> date | None:
    s = str(s or "").strip()
    digits = re.sub(r"\D", "", s)
    try:
        if len(digits) == 8:  # 20261116
            return date(int(digits[:4]), int(digits[4:6]), int(digits[6:]))
        if len(digits) == 7:  # 民國 1151116
            return date(int(digits[:3]) + 1911, int(digits[3:5]), int(digits[5:]))
        if "/" in s:  # 115/11/16
            y, m, d = (int(x) for x in s.split("/"))
            return date(y + 1911 if y < 1000 else y, m, d)
    except ValueError:
        return None
    return None


def _num(x):
    try:
        v = float(str(x).replace(",", "").strip())
        return v
    except (TypeError, ValueError):
        return None


def _key(row: dict, *cands: str):
    low = {k.lower(): k for k in row}
    for c in cands:
        for lk, k in low.items():
            if lk == c.lower():
                return k
    for c in cands:
        for lk, k in low.items():
            if c.lower() in lk:
                return k
    return None


def _market_date() -> date | None:
    """證交所每日市場成交資訊的最後一筆日期(STOCK_DAY_ALL 沒附日期時用)。"""
    try:
        rows = requests.get("https://openapi.twse.com.tw/v1/exchangeReport/FMTQIK", headers=HEADERS,
                            timeout=30).json()
        k = _key(rows[-1], "Date")
        return parse_date(rows[-1][k]) if k else None
    except Exception as e:  # noqa: BLE001
        log.warning("FMTQIK failed: %s", e)
        return None


def _rows_to_bars(rows, suffix: str, fallback_date: date | None) -> dict[str, tuple[date, dict]]:
    out = {}
    if not rows:
        return out
    r0 = rows[0]
    kc = _key(r0, "Code", "SecuritiesCompanyCode")
    ko, kh, kl = _key(r0, "OpeningPrice", "Open"), _key(r0, "HighestPrice", "High"), _key(r0, "LowestPrice", "Low")
    kcl = _key(r0, "ClosingPrice", "Close")
    kv = _key(r0, "TradeVolume", "TradingShares")
    kd = _key(r0, "Date")
    if not all([kc, ko, kh, kl, kcl, kv]):
        log.warning("unexpected fields: %s", list(r0))
        return out
    for r in rows:
        d = parse_date(r.get(kd)) if kd else fallback_date
        if d is None:
            continue
        o, h, l, c, v = (_num(r.get(k)) for k in (ko, kh, kl, kcl, kv))
        if None in (o, h, l, c, v) or c <= 0:
            continue  # 當日無成交
        out[f"{str(r[kc]).strip()}{suffix}"] = (d, {"Open": o, "High": h, "Low": l, "Close": c, "Volume": v})
    return out


def latest_official_bars() -> tuple[dict[str, tuple[date, dict]], str]:
    bars, notes = {}, []
    try:
        rows = requests.get("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL", headers=HEADERS,
                            timeout=30).json()
        need_date = rows and not _key(rows[0], "Date")
        b = _rows_to_bars(rows, ".TW", _market_date() if need_date else None)
        bars.update(b)
        notes.append(f"TWSE {len(b)} 檔")
    except Exception as e:  # noqa: BLE001
        notes.append(f"TWSE 失敗: {e}")
    try:
        rows = requests.get("https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes", headers=HEADERS,
                            timeout=30).json()
        b = _rows_to_bars(rows, ".TWO", None)
        bars.update(b)
        notes.append(f"TPEx {len(b)} 檔")
    except Exception as e:  # noqa: BLE001
        notes.append(f"TPEx 失敗: {e}")
    return bars, ";".join(notes)


def patch(daily: dict[str, pd.DataFrame], bars) -> list[str]:
    """若官方資料日期比 Yahoo 最後一根新，就補上(或覆蓋同日不完整的 K 棒)。回傳被補的代號。"""
    patched = []
    for sym, df in daily.items():
        if sym not in bars:
            continue
        d, ohlcv = bars[sym]
        ts = pd.Timestamp(d)
        last = df.index[-1].normalize()
        if ts > last:
            daily[sym] = pd.concat([df, pd.DataFrame([ohlcv], index=pd.DatetimeIndex([ts], name="Date"))])
            patched.append(sym)
        elif ts == last:
            df.loc[df.index[-1], ["Open", "High", "Low", "Close", "Volume"]] = [ohlcv[k] for k in
                                                                              ("Open", "High", "Low", "Close", "Volume")]
    return patched
