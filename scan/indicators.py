"""技術指標(全部用標準公式，台股 KD 採 9,3,3 慣用算法)。"""
from __future__ import annotations

import numpy as np
import pandas as pd

SMA_PERIODS = (5, 10, 20, 50, 60, 150, 200)


def wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = wilder(d.clip(lower=0), n)
    dn = wilder(-d.clip(upper=0), n)
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(dn != 0, 100.0)


def macd(close: pd.Series, fast=12, slow=26, sig=9):
    dif = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    dea = dif.ewm(span=sig, adjust=False).mean()
    return dif, dea, dif - dea


def kd(df: pd.DataFrame, n=9, k_s=3, d_s=3):
    ll = df["Low"].rolling(n, min_periods=n).min()
    hh = df["High"].rolling(n, min_periods=n).max()
    rng = (hh - ll).replace(0, np.nan)
    rsv = ((df["Close"] - ll) / rng * 100).fillna(50)
    k_vals, d_vals = [], []
    k_prev = d_prev = 50.0
    for i, v in enumerate(rsv.to_numpy()):
        if i < n - 1:
            k_vals.append(np.nan)
            d_vals.append(np.nan)
            continue
        k_prev = (k_s - 1) / k_s * k_prev + v / k_s
        d_prev = (d_s - 1) / d_s * d_prev + k_prev / d_s
        k_vals.append(k_prev)
        d_vals.append(d_prev)
    return pd.Series(k_vals, index=df.index), pd.Series(d_vals, index=df.index)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return wilder(tr, n)


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]
    for n in SMA_PERIODS:
        df[f"MA{n}"] = c.rolling(n, min_periods=n).mean()
    df["RSI14"] = rsi(c)
    df["MACD_DIF"], df["MACD_DEA"], df["MACD_HIST"] = macd(c)
    df["K9"], df["D9"] = kd(df)
    df["ATR14"] = atr(df, 14)
    mid = c.rolling(20, min_periods=20).mean()
    sd = c.rolling(20, min_periods=20).std(ddof=0)
    df["BB_UP"], df["BB_MID"], df["BB_LOW"] = mid + 2 * sd, mid, mid - 2 * sd
    ema20 = c.ewm(span=20, adjust=False).mean()
    a20 = atr(df, 20)
    df["KC_UP"], df["KC_LOW"] = ema20 + 1.5 * a20, ema20 - 1.5 * a20
    df["SQUEEZE_ON"] = (df["BB_UP"] < df["KC_UP"]) & (df["BB_LOW"] > df["KC_LOW"])
    df["VOL_MA50"] = df["Volume"].rolling(50, min_periods=20).mean()
    df["HIGH_52W"] = df["High"].rolling(252, min_periods=120).max()
    df["LOW_52W"] = df["Low"].rolling(252, min_periods=120).min()
    return df


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    w = df[["Open", "High", "Low", "Close", "Volume"]].resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna(subset=["Close"])
    w["MA10W"] = w["Close"].rolling(10, min_periods=10).mean()
    w["MA20W"] = w["Close"].rolling(20, min_periods=20).mean()
    w["MA40W"] = w["Close"].rolling(40, min_periods=40).mean()
    w["RSI14"] = rsi(w["Close"])
    w["MACD_DIF"], w["MACD_DEA"], w["MACD_HIST"] = macd(w["Close"])
    w["K9"], w["D9"] = kd(w)
    return w


def add_intraday(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]
    for n in (20, 60, 120):
        df[f"MA{n}"] = c.rolling(n, min_periods=n).mean()
    df["RSI14"] = rsi(c)
    df["MACD_DIF"], df["MACD_DEA"], df["MACD_HIST"] = macd(c)
    df["K9"], df["D9"] = kd(df)
    return df
