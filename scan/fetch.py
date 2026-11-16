"""從 Yahoo Finance(yfinance)抓取 K 線。"""
from __future__ import annotations

import logging
import time

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

COLS = ["Open", "High", "Low", "Close", "Volume"]


def _split(df: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    if df is None or df.empty:
        return out
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = set(df.columns.get_level_values(0))
        for t in tickers:
            if t in lvl0:
                sub = df[t]
                if all(c in sub.columns for c in COLS):
                    sub = sub[COLS].dropna(subset=["Close"])
                    if not sub.empty:
                        out[t] = sub
    else:  # 單一代號
        if all(c in df.columns for c in COLS) and len(tickers) == 1:
            sub = df[COLS].dropna(subset=["Close"])
            if not sub.empty:
                out[tickers[0]] = sub
    return out


def download(tickers: list[str], period: str = "2y", interval: str = "1d", chunk: int = 80,
             retries: int = 3) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """分批下載，回傳 ({代號: DataFrame}, 失敗代號清單)。"""
    data: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        pending = list(batch)
        for attempt in range(retries):
            if not pending:
                break
            try:
                df = yf.download(pending, period=period, interval=interval, group_by="ticker",
                                 auto_adjust=False, actions=False, threads=True, progress=False)
                got = _split(df, pending)
                data.update(got)
                pending = [t for t in pending if t not in got]
            except Exception as e:  # noqa: BLE001
                log.warning("download batch failed (attempt %d): %s", attempt + 1, e)
            if pending:
                time.sleep(5 * (attempt + 1))
        time.sleep(1)
    failed = [t for t in tickers if t not in data]
    for t, df in data.items():
        idx = pd.to_datetime(df.index)
        if getattr(idx, "tz", None) is not None and interval == "1d":
            idx = idx.tz_localize(None)
        df.index = idx
        df.index.name = "Date"
    return data, failed
