"""特徵事件回測(S02 RS 領先 / S03 突破失敗 / S06 極端軌道)。

用法：python -m backtest.feature_study [--download]
  --download:以目前觀察池(data/latest/screen_all.csv + 備用清單)抓 5 年日K,存到 backtest/cache/daily5y.pkl
注意：觀察池是「今天」的流動性名單，有倖存者偏差;結果只用來比較「有/沒有某特徵」的相對差異。

交易規則(全部特徵共用):訊號隔日開盤進場、停損 2ATR、目標 4ATR(=2R)、最多持有 20 日。
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
from scan import fetch, indicators, universe  # noqa: E402

CACHE = Path(__file__).resolve().parent / "cache"
BENCH = {"台股": "^TWII", "美股": "^GSPC", "美股ETF": "^GSPC"}


def download():
    CACHE.mkdir(exist_ok=True)
    uni, _ = universe.build_universe()
    grp = dict(zip(uni.symbol, uni.group))
    sa = pd.read_csv(ROOT / "data/latest/screen_all.csv")
    grp.update(dict(zip(sa.symbol, sa.group)))
    daily, failed = fetch.download(sorted(grp), period="5y", interval="1d")
    print("ok", len(daily), "failed", failed)
    pickle.dump({"daily": daily, "group": grp}, open(CACHE / "daily5y.pkl", "wb"))


def trade(df, i, side, a):
    n = len(df)
    if i + 1 >= n:
        return np.nan
    e = df.Open.iat[i + 1]
    st, tg = (e - 2 * a, e + 4 * a) if side > 0 else (e + 2 * a, e - 4 * a)
    end = min(i + 21, n)
    for j in range(i + 1, end):
        if side > 0:
            if df.Low.iat[j] <= st:
                return -1.0
            if df.High.iat[j] >= tg:
                return 2.0
        else:
            if df.High.iat[j] >= st:
                return -1.0
            if df.Low.iat[j] <= tg:
                return 2.0
    if end - 1 - i < 20:
        return np.nan  # 未滿 20 日不計
    return side * (df.Close.iat[end - 1] - e) / (2 * a)


def study():
    o = pickle.load(open(CACHE / "daily5y.pkl", "rb"))
    D, G = o["daily"], o["group"]
    L, S, F, B = [], [], [], []
    for s, raw in D.items():
        g = G.get(s)
        if g not in BENCH or len(raw) < 400 or BENCH[g] not in D:
            continue
        df = indicators.add_all(raw)
        b = D[BENCH[g]]["Close"].reindex(df.index).ffill()
        c, h, l, v = df.Close, df.High, df.Low, df.Volume
        if (v.tail(100) > 0).mean() < 0.8:
            continue
        ma200p = df.MA200.shift(20)
        stg2 = (c > df.MA150) & (c > df.MA200) & (df.MA150 > df.MA200) & (df.MA200 > ma200p) & (df.MA50 > df.MA150)
        t7 = stg2 & (df.MA50 > df.MA200) & (c > df.MA50) & (c >= 1.3 * df.LOW_52W) & (c >= 0.75 * df.HIGH_52W)
        hh20, ll20 = h.shift(1).rolling(20).max(), l.shift(1).rolling(20).min()
        volx = v >= 1.5 * df.VOL_MA50
        bo, bd = (c > hh20) & volx, (c < ll20) & volx
        rs = c / b
        rs_nh, rs_nl = rs >= rs.rolling(20).max(), rs <= rs.rolling(20).min()
        stress = ~(b < b.shift(5)) | (c > c.shift(5) * 0.98)
        rs_lead = rs_nh & ~(h >= hh20) & stg2 & stress
        resil = (b < b.shift(5) * 0.98) & (c >= c.shift(5))
        m23 = c.rolling(23).mean()
        sell = m23 + (h - m23).rolling(60).max().shift(1)
        buy = m23 - (m23 - l).rolling(60).max().shift(1)
        em20 = c.ewm(span=20, adjust=False).mean()
        lead10 = rs_lead.shift(1).rolling(10).max().fillna(0).astype(bool)
        resil10 = resil.rolling(10).max().fillna(0).astype(bool)
        short_tr = (c < df.MA50) & (df.MA50 < df.MA50.shift(10)) & (c < df.MA200)
        fwd20 = c.shift(-20) / c - 1

        last = -99
        for i in np.where((t7 & bo).to_numpy())[0]:
            if i < 260 or i - last < 5:
                continue
            last = i
            L.append(dict(sym=s, g=g, date=df.index[i], R=trade(df, i, 1, df.ATR14.iat[i]), f20=fwd20.iat[i],
                          rs_nh=bool(rs_nh.iat[i]), lead10=bool(lead10.iat[i]), resil10=bool(resil10.iat[i]),
                          ext_up=bool(h.iat[i] > sell.iat[i]), dist50=c.iat[i] / df.MA50.iat[i] - 1))
            lvl = hh20.iat[i]
            for k in (1, 2, 3):
                if i + k + 20 >= len(df):
                    break
                seg = slice(i + 1, i + k + 1)
                fail = bool(((c.iloc[seg] < lvl) | (c.iloc[seg] < em20.iloc[seg])).any())
                F.append(dict(sym=s, g=g, k=k, fail=fail, f20=c.iat[i + k + 20] / c.iat[i + k] - 1,
                              R=trade(df, i + k, 1, df.ATR14.iat[i + k])))
        last = -99
        for i in np.where((short_tr & bd).to_numpy())[0]:
            if i < 260 or i - last < 5:
                continue
            last = i
            S.append(dict(sym=s, g=g, date=df.index[i], R=trade(df, i, -1, df.ATR14.iat[i]), f20=-fwd20.iat[i],
                          rs_nl=bool(rs_nl.iat[i]), ext_dn=bool(l.iat[i] < buy.iat[i])))
        base = t7 & (c <= hh20) & (c >= hh20 * 0.97)
        nb = (c > hh20).shift(-1).rolling(10).max().shift(-9)
        last = -99
        for i in np.where(base.to_numpy())[0]:
            if i < 260 or i - last < 5 or np.isnan(fwd20.iat[i]):
                continue
            last = i
            B.append(dict(sym=s, g=g, date=df.index[i], f20=fwd20.iat[i], brk10=nb.iat[i],
                          lead_now=bool(rs_lead.iloc[max(0, i - 4):i + 1].any()), resil10=bool(resil10.iat[i]),
                          R=trade(df, i, 1, df.ATR14.iat[i])))
    E = {k: pd.DataFrame(v) for k, v in dict(L=L, S=S, F=F, B=B).items()}
    pd.to_pickle(E, CACHE / "events.pkl")
    return E


def _st(x):
    x = x.dropna()
    if not len(x):
        return "N=0"
    se = x.std() / np.sqrt(len(x))
    return f"N={len(x):4d} 勝率={(x > 0).mean() * 100:5.1f}% 平均={x.mean():+.3f}(±{1.96 * se:.3f})"


def _diff(a, b):
    a, b = a.dropna(), b.dropna()
    d = a.mean() - b.mean()
    return f"差={d:+.4f} t={d / np.sqrt(a.var() / len(a) + b.var() / len(b)):+.2f}"


def report(E):
    L, S, F, B = E["L"], E["S"], E["F"], E["B"]
    print("期間", L.date.min().date(), "~", L.date.max().date())
    print("多方突破基準 R:", _st(L.R))
    for name, f in (("S02 突破前10日 RS 領先", "lead10"), ("突破當日 RS 新高", "rs_nh"),
                    ("抗跌(大盤5日跌>2%)", "resil10"), ("S06 觸及極端上軌", "ext_up")):
        print(f"\n[{name}]\n  有 {_st(L[L[f]].R)}\n  無 {_st(L[~L[f]].R)}\n  R {_diff(L[L[f]].R, L[~L[f]].R)}")
        for g, d in L.groupby("g"):
            print(f"   {g}: 有 {d[d[f]].R.mean():+.3f}(N={d[f].sum()}) 無 {d[~d[f]].R.mean():+.3f}")
        early = L.date < "2024-05-01"
        for nm, d in (("前半", L[early]), ("後半", L[~early])):
            print(f"   {nm}: 有 {d[d[f]].R.mean():+.3f} 無 {d[~d[f]].R.mean():+.3f}")
    print("\n[S03 突破後 k 日內跌回突破點/EMA20]")
    for k, d in F.groupby("k"):
        print(f"  k={k} 失敗 {_st(d[d.fail].f20)} | 未失敗 {_st(d[~d.fail].f20)} | {_diff(d[d.fail].f20, d[~d.fail].f20)}")
    print("\n空方跌破基準 R:", _st(S.R))
    for name, f in (("S06 觸及極端下軌", "ext_dn"), ("RS 創 20 日新低", "rs_nl")):
        print(f"[{name}] 有 {_st(S[S[f]].R)} | 無 {_st(S[~S[f]].R)} | {_diff(S[S[f]].R, S[~S[f]].R)}")
    print("\n蓄勢基準 f20:", _st(B.f20), f"10日內突破率 {B.brk10.mean() * 100:.1f}%")
    for f in ("lead_now", "resil10"):
        print(f"[{f}] 有 {_st(B[B[f]].f20)} 突破率 {B[B[f]].brk10.mean() * 100:.1f}% | "
              f"無 {_st(B[~B[f]].f20)} 突破率 {B[~B[f]].brk10.mean() * 100:.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args()
    if a.download or not (CACHE / "daily5y.pkl").exists():
        download()
    report(study())
