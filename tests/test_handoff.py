"""引き継ぎ用の判断関数が、検証で使った zoo の判断と一致することを確かめる。"""

import numpy as np
import pandas as pd
import pytest

from tradebot import data as dt
from tradebot import handoff as ho
from tradebot import zoo


def test_lev3_matches_zoo_on_real_spy():
    spy = dt.load("SPY")
    mkt = spy
    sig = zoo.c_sma(200, "L3")({"MKT": mkt}, mkt)
    days = spy.index[300::97]  # 実データの色々な日で照合
    for d in days:
        want = "UPRO" if sig["L3"]["entry"].loc[d] else "BIL"
        assert ho.target_lev3(spy["close"].loc[:d]) == want, d


def test_voltarget_matches_zoo_on_real_spy():
    spy = dt.load("SPY")
    data = {"MKT": spy, "L2": spy, "L3": spy, "CASH": spy}
    sig = zoo.c_voltarget_monthly_lev(200, 0.15, 0.25)(data, spy)
    name = {"L3": "UPRO", "L2": "SSO", "MKT": "SPYM", "CASH": "BIL"}
    for d in spy.index[300::89]:
        want = [name[k] for k in ("L3", "L2", "MKT", "CASH") if sig[k]["entry"].loc[d]]
        assert [ho.target_voltarget(spy["close"].loc[:d])] == want, d


def test_tqqq_and_errors():
    up = pd.Series(np.linspace(100, 200, 250))
    down = pd.Series(np.r_[np.linspace(100, 200, 200), np.linspace(200, 100, 50)])
    assert ho.target_tqqq(up) == "TQQQ" and ho.target_tqqq(down) == "BIL"
    with pytest.raises(ValueError):
        ho.target_lev3(up.iloc[:150])


def test_momentum_6_1_ranks_by_skip_month_return():
    idx = pd.bdate_range("2025-01-01", periods=130)
    closes = pd.DataFrame({
        "AAA": np.r_[np.full(3, 100.0), np.linspace(100, 200, 106), np.full(21, 50.0)],  # 直近21日の急落は無視
        "BBB": np.linspace(100, 150, 130),
        "CCC": np.linspace(100, 300, 130),  # 構成銘柄でない
    }, index=idx)
    assert ho.momentum_6_1(closes, ["AAA", "BBB"], top_n=2) == ["AAA", "BBB"]
