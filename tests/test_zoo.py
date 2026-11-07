import numpy as np
import pandas as pd

from tradebot import backtest as bt
from tradebot import longhist as lh
from tradebot import zoo


def frame(close, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(close))
    c = pd.Series(close, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 0, "div_yield": 0.0})


def test_targets_to_signals_switches_once():
    idx = pd.bdate_range("2020-01-01", periods=6)
    tgt = pd.Series(["A", "A", "B", "B", None, "A"], index=idx)
    sig = zoo.targets_to_signals(tgt, pd.Series(True, index=idx), ["A", "B"])
    assert list(sig["A"]["entry"]) == [True, True, False, False, False, True]
    assert list(sig["B"]["exit"]) == [True, True, False, False, True, True]


def test_sma_strategy_holds_lev_above_and_cash_below():
    up = list(np.linspace(100, 200, 260)) + list(np.linspace(200, 100, 100))
    mkt = frame(up)
    cash = frame(100 * 1.0001 ** np.arange(len(up)))
    data = {"MKT": mkt, "CASH": cash, "L3": lh.leveraged(mkt, cash, 3)}
    sig = zoo.c_sma(200, "L3")(data, mkt)
    res = bt.run({"L3": data["L3"], "CASH": cash}, sig,
                 bt.Config(initial_cash=1000, fractional=True, max_positions=1,
                           costs=bt.Costs(commission_rate=0, slippage=0)))
    syms = [t.symbol for t in res.trades]
    assert syms[0] == "L3" and "CASH" in syms  # 上昇中は3倍、下落で200日線を割ったら現金


def test_leveraged_matches_formula():
    mkt = frame([100, 110, 99])
    cash = frame([100, 100, 100])
    l3 = lh.leveraged(mkt, cash, 3, spread=0, expense=0)["close"]
    assert np.allclose(l3.pct_change().dropna(), [0.30, -0.30])


def test_voltarget_picks_leverage_by_volatility():
    rng = np.random.default_rng(0)
    calm = 100 * np.cumprod(1 + rng.normal(0.0005, 0.003, 300))  # 年率5%前後の値動き
    mkt = frame(calm)
    cash = frame(np.full(300, 100.0))
    data = {"MKT": mkt, "CASH": cash, "L2": mkt, "L3": mkt}
    sig = zoo.c_voltarget(None, 0.10, 0.20)(data, mkt)
    assert sig["L3"]["entry"].iloc[-1] and not sig["MKT"]["entry"].iloc[-1]
