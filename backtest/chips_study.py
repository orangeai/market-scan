"""台股籌碼訊號回測(三大法人、融資融券)。

需要先有 data/chips/tw_chips.csv(由 tw-chips workflow 的 backfill 建立)與 5 年日K 快取
(python -m backtest.feature_study --download 會建立 backtest/cache/daily5y.pkl)。

用法：python -m backtest.chips_study [--chips data/chips/tw_chips.csv]

事件(只取籌碼資料涵蓋的期間，同一檔 5 日內只取第一次):
  A 突破：趨勢模板 7/7 + 收盤突破前 20 日高 + 量 ≥ 1.5 倍 50 日均量
  B 蓄勢：趨勢模板 7/7 + 收盤距前 20 日高 3% 內未突破
  C 籌碼先篩：Stage 2 且「投信連買第 3 天」(不看型態)
籌碼一律用事件當天(含)以前的資料;進場為隔日開盤(當晚籌碼已公布，無前視)。
交易規則與 feature_study 相同：停損 2ATR、目標 4ATR(=2R)、最多 20 日。
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.feature_study import CACHE, _diff, _st, trade  # noqa: E402
from scan import fetch, indicators  # noqa: E402


def _streak(x: pd.Series) -> pd.Series:
    """連續買超(正)/賣超(負)天數。"""
    sign = np.sign(x.fillna(0)).astype(int)
    grp = (sign != sign.shift()).cumsum()
    return sign * sign.groupby(grp).cumcount().add(1)


def chip_frame(ch: pd.DataFrame, idx: pd.DatetimeIndex, vol: pd.Series) -> pd.DataFrame:
    c = ch.set_index(pd.to_datetime(ch.date)).sort_index().reindex(idx)
    covered = c.total.notna()
    out = pd.DataFrame(index=idx)
    v20 = vol.rolling(20).mean()
    for k in ("foreign", "trust", "total"):
        s = c[k]
        out[f"{k}_streak"] = _streak(s).where(covered)
        out[f"{k}_5d_pct"] = (s.fillna(0).rolling(5).sum() / (v20 * 5) * 100).where(covered)
    mb = c.margin_bal
    out["margin_5d_pct"] = (mb / mb.shift(5) - 1) * 100
    out["sm_ratio"] = c.short_bal / mb * 100
    out["covered"] = covered
    return out


def study(chips_csv: Path):
    ch_all = pd.read_csv(chips_csv, dtype={"symbol": str})
    syms = sorted(ch_all.symbol.unique())
    cache = CACHE / "daily5y.pkl"
    if cache.exists():
        o = pickle.load(open(cache, "rb"))
        D = o["daily"]
    else:
        D = {}
    need = [s for s in syms + ["^TWII"] if s not in D]
    if need:
        d2, _ = fetch.download(need, period="5y", interval="1d")
        D.update(d2)
    rows = []
    by = dict(tuple(ch_all.groupby("symbol")))
    for s in syms:
        if s not in D or len(D[s]) < 300:
            continue
        df = indicators.add_all(D[s])
        c, h, v = df.Close, df.High, df.Volume
        f = chip_frame(by[s], df.index, v)
        ma200p = df.MA200.shift(20)
        stg2 = (c > df.MA150) & (c > df.MA200) & (df.MA150 > df.MA200) & (df.MA200 > ma200p) & (df.MA50 > df.MA150)
        t7 = stg2 & (df.MA50 > df.MA200) & (c > df.MA50) & (c >= 1.3 * df.LOW_52W) & (c >= 0.75 * df.HIGH_52W)
        hh20 = h.shift(1).rolling(20).max()
        bo = (c > hh20) & (v >= 1.5 * df.VOL_MA50)
        base = t7 & (c <= hh20) & (c >= hh20 * 0.97)
        chip_first = stg2 & (f.trust_streak == 3)
        px5 = c / c.shift(5) - 1
        fwd20 = c.shift(-20) / c - 1
        for kind, sig in (("A突破", t7 & bo), ("B蓄勢", base), ("C籌碼先篩", chip_first), ("D Stage2隨機", stg2)):
            last = -99
            pos = np.where((sig & f.covered).to_numpy())[0]
            if kind.startswith("D"):
                pos = pos[::7]  # 對照組：每 7 根取一根
            for i in pos:
                if i < 220 or i - last < 5 or i + 21 >= len(df):
                    continue
                last = i
                r = f.iloc[i]
                rows.append(dict(kind=kind, sym=s, date=df.index[i], R=trade(df, i, 1, df.ATR14.iat[i]),
                                 f20=fwd20.iat[i], px5=px5.iat[i], **r.to_dict()))
    E = pd.DataFrame(rows)
    E.to_pickle(CACHE / "chip_events.pkl")
    return E


def report(E: pd.DataFrame):
    print("期間", E.date.min().date(), "~", E.date.max().date(), "標的", E.sym.nunique())
    for k, d in E.groupby("kind"):
        print(f"{k}: R {_st(d.R)} | 20日 {_st(d.f20)}")
    tests = {
        "投信連買≥3日": lambda d: d.trust_streak >= 3,
        "投信5日買超≥2%量": lambda d: d.trust_5d_pct >= 2,
        "外資5日買超≥5%量": lambda d: d.foreign_5d_pct >= 5,
        "外資+投信5日同買": lambda d: (d.foreign_5d_pct > 0) & (d.trust_5d_pct > 0),
        "法人5日合計賣超": lambda d: d.total_5d_pct < 0,
        "融資5日增>5%且價5日<2%": lambda d: (d.margin_5d_pct > 5) & (d.px5 < 0.02),
        "融資5日減且價5日≥0": lambda d: (d.margin_5d_pct < 0) & (d.px5 >= 0),
        "券資比≥20%": lambda d: d.sm_ratio >= 20,
    }
    for kind in ("A突破", "B蓄勢"):
        d = E[E.kind == kind]
        print(f"\n===== {kind}(N={len(d)})")
        mid = d.date.min() + (d.date.max() - d.date.min()) / 2
        for name, fn in tests.items():
            m = fn(d).fillna(False).astype(bool)
            if m.sum() < 15:
                print(f"[{name}] 樣本 {m.sum()} 太少")
                continue
            e = d.date < mid
            print(f"[{name}] 有 {_st(d[m].R)} | 無 {_st(d[~m].R)} | R {_diff(d[m].R, d[~m].R)} | "
                  f"20日 {_diff(d[m].f20, d[~m].f20)} | 前半 {d[m & e].R.mean():+.3f}/{d[~m & e].R.mean():+.3f} "
                  f"後半 {d[m & ~e].R.mean():+.3f}/{d[~m & ~e].R.mean():+.3f}")
    c, r = E[E.kind == "C籌碼先篩"], E[E.kind == "D Stage2隨機"]
    print(f"\n[C 籌碼先篩 vs Stage2 隨機日] R {_diff(c.R, r.R)} | 20日 {_diff(c.f20, r.f20)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", default=str(ROOT / "data/chips/tw_chips.csv"))
    a = ap.parse_args()
    report(study(Path(a.chips)))
