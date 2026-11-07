import numpy as np
import pandas as pd
import pytest

from tradebot import backtest as bt
from tradebot import handoff as ho
from tradebot import weights as wt


def _df(closes, start="2019-01-01"):
    idx = pd.bdate_range(start, periods=len(closes))
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1})


def _two_assets(n=800, seed=1):
    rng = np.random.default_rng(seed)
    a = _df(100 * np.cumprod(1 + rng.normal(0.0008, 0.02, n)))
    b = _df(50 * np.cumprod(1 + rng.normal(0.0001, 0.001, n)))
    return {"A": a, "B": b}


def test_one_hot_weights_match_single_position_engine():
    data = _two_assets()
    idx = data["A"].index
    hold_a = pd.Series((np.arange(len(idx)) // 60) % 2 == 0, index=idx)  # 60日ごとに A と B を入れ替える
    sig = {"A": pd.DataFrame({"entry": hold_a, "exit": ~hold_a, "score": 0.0}),
           "B": pd.DataFrame({"entry": ~hold_a, "exit": hold_a, "score": 0.0})}
    cfg = bt.Config(initial_cash=2000, fractional=True, max_positions=1, tax=bt.TAXABLE)
    r1 = bt.run(data, sig, cfg)
    w = pd.DataFrame({"A": hold_a.astype(float), "B": (~hold_a).astype(float)})
    r2 = wt.run_weights(data, w, cfg)
    assert r2.equity.iloc[-1] == pytest.approx(r1.equity.iloc[-1], rel=1e-6)
    assert r2.tax_paid == pytest.approx(r1.tax_paid, rel=1e-6)


def test_partial_sell_realizes_only_that_part():
    data = {"A": _df([100, 100, 200, 200, 200]), "B": _df([10, 10, 10, 10, 10])}
    no_cost = bt.Costs(commission_rate=0, slippage=0)
    cfg = bt.Config(initial_cash=1000, fractional=True, costs=no_cost, tax=bt.TAXABLE)
    w = pd.DataFrame({"A": [1, 1, 0.5, 0.5, 0.5], "B": [0, 0, 0.5, 0.5, 0.5]}, index=data["A"].index, dtype=float)
    res = wt.run_weights(data, w, cfg)
    # 100 で10株 → 200 になった翌日に半分の価値（1000ドル＝5株）を売る。利益は 5株×100 = 500
    # 最終日に残りを売る分（500）と合わせて 1000 の利益。同じ年なので税金は 1000×20.315%
    assert res.tax_paid == pytest.approx(1000 * 0.20315)
    assert res.equity.iloc[-1] == pytest.approx(2000 - 1000 * 0.20315)


def test_target_weights_follow_voltarget():
    idx = pd.bdate_range("2023-01-02", periods=400)
    c = pd.Series(100 * 1.001 ** np.arange(400), index=idx)  # 値動きのない上昇 → UPRO
    assert ho.target_voltarget(c) == "UPRO"
    assert ho.target_weights(c) == {"UPRO": 1.0, "BIL": 0.0}  # 値動きが小さいので上限の 100%
    down = pd.concat([c, pd.Series(c.iloc[-1] * 0.97 ** np.arange(1, 30), index=pd.bdate_range(idx[-1], periods=30)[1:])])
    assert ho.target_weights(down) == {"UPRO": 0.0, "BIL": 1.0}


def test_plan_rebalance_sells_part_then_buys():
    p = {"UPRO": 128.0, "BIL": 91.5}
    orders = ho.plan_rebalance({"UPRO": 15}, 40, {"UPRO": 2 / 3, "BIL": 1 / 3}, p)
    assert orders == [{"side": "SELL", "symbol": "UPRO", "qty": 5}, {"side": "BUY", "symbol": "BIL", "qty": 7}]
    # 逃げるときは全部売る
    orders = ho.plan_rebalance({"UPRO": 10, "BIL": 7}, 30, {"UPRO": 0.0, "BIL": 1.0}, p)
    assert orders[0] == {"side": "SELL", "symbol": "UPRO", "qty": 10}
    assert orders[1]["side"] == "BUY" and orders[1]["symbol"] == "BIL"
    # 使うお金は手元の現金と売った分を超えない
    cash_after = 30 + 10 * 128 - orders[1]["qty"] * 91.5
    assert cash_after >= 0
