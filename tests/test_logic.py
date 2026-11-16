"""用合成數據測試程式邏輯能正確執行(只測程式，不產生任何市場數據)。"""
import numpy as np
import pandas as pd

from scan import indicators, screens


def make(n=400, drift=0.001, seed=0, vol=True):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-06-01", periods=n)
    c = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.015, n)))
    h = c * (1 + rng.uniform(0, 0.01, n))
    l = c * (1 - rng.uniform(0, 0.01, n))
    o = (h + l) / 2
    v = rng.integers(1e5, 5e5, n).astype(float) if vol else np.zeros(n)
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}, index=idx)


def test_indicators_ranges():
    e = indicators.add_all(make())
    assert e["RSI14"].dropna().between(0, 100).all()
    assert e["K9"].dropna().between(0, 100).all()
    assert e["D9"].dropna().between(0, 100).all()
    w = indicators.to_weekly(make())
    assert (w.index.dayofweek == 4).all()


def test_rsi_known():
    # 單調上漲 → RSI 100
    s = pd.Series(np.arange(1, 50, dtype=float))
    assert indicators.rsi(s).iloc[-1] == 100


def test_analyse_runs_all_regimes():
    for drift, vol in ((0.002, True), (-0.002, True), (0.0, False)):
        r = screens.analyse(indicators.add_all(make(drift=drift, vol=vol)))
        assert r["direction"] in ("多", "空", "無")
        assert "score" in r


def test_short_history():
    r = screens.analyse(indicators.add_all(make(n=100)))
    assert r["trend_long"]["mode"].startswith("簡化版")


def test_twse_parse_and_patch():
    from datetime import date
    from scan import twse
    assert twse.parse_date("1151116") == date(2026, 11, 16)
    assert twse.parse_date("115/11/16") == date(2026, 11, 16)
    assert twse.parse_date("20261116") == date(2026, 11, 16)
    df = make(n=100)
    nxt = df.index[-1] + pd.Timedelta(days=3)
    d = {"2330.TW": df}
    patched = twse.patch(d, {"2330.TW": (nxt.date(), {"Open": 1, "High": 2, "Low": 0.5, "Close": 1.5, "Volume": 9})})
    assert patched == ["2330.TW"] and d["2330.TW"]["Close"].iloc[-1] == 1.5 and len(d["2330.TW"]) == 101
