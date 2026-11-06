import numpy as np
import pandas as pd
import pytest

from tradebot import live


def ledger():
    return live.new_ledger("2026-11-05")


def test_new_ledger_and_benchmark():
    lg = ledger()
    assert lg["strategies"]["momentum"]["cash"] == 2000
    assert lg["benchmark"]["qty"] is None
    live.apply_fill(lg, "trend", "BUY", "SPYM", 21, 87.32, 2.4, "2026-11-05")
    live.set_benchmark(lg, live.first_fill_price(lg, "SPYM"))
    assert lg["benchmark"]["qty"] == 22  # 2000 / (87.32 * 1.00132) の整数部分
    assert lg["benchmark"]["cash"] == pytest.approx(2000 - 22 * 87.32 * 1.00132)
    live.set_benchmark(lg, 999.0)  # 2回目は変えない
    assert lg["benchmark"]["qty"] == 22


def test_fill_bookkeeping_and_oversell_guard():
    lg = ledger()
    live.apply_fill(lg, "trend", "BUY", "SPYM", 20, 87.0, 1.0, "2026-11-05")
    book = lg["strategies"]["trend"]
    assert book["cash"] == pytest.approx(2000 - 20 * 87 - 1)
    assert book["positions"]["SPYM"]["qty"] == 20
    with pytest.raises(ValueError):
        live.apply_fill(lg, "trend", "SELL", "SPYM", 21, 90.0, 0, "2026-12-01")
    live.apply_fill(lg, "trend", "SELL", "SPYM", 20, 90.0, 1.0, "2026-12-01")
    assert "SPYM" not in book["positions"]
    assert book["cash"] == pytest.approx(2000 - 20 * 87 - 1 + 20 * 90 - 1)


def test_plan_trend_switches_and_uses_whole_shares():
    book = {"cash": 10.0, "positions": {"SPYM": {"qty": 22, "cost": 87.0}}}
    prices = {"SPYM": 90.0, "IEF": 92.0}
    orders = live.plan_trend(book, "IEF", prices)
    assert [(o.side, o.symbol, o.qty) for o in orders] == [("SELL", "SPYM", 22), ("BUY", "IEF", 21)]
    assert live.plan_trend(book, "SPYM", prices) == []  # 既に SPYM なら何もしない（現金10ドルでは買えない）


def test_plan_momentum_sells_dropouts_and_skips_unaffordable():
    book = {"cash": 0.0, "positions": {"OLD": {"qty": 10, "cost": 200.0}}}  # 評価額 2000
    prices = {"OLD": 200.0, "BIG": 900.0, "A": 50.0, "B": 100.0, "C": 30.0, "D": 10.0, "E": 20.0}
    ranking = ["BIG", "A", "B", "C", "D", "E"]
    orders = live.plan_momentum(book, ranking, prices, top_n=5)
    assert orders[0].side == "SELL" and orders[0].symbol == "OLD"
    buys = [o.symbol for o in orders if o.side == "BUY"]
    assert "BIG" not in buys  # 1枠 400 ドルでは 900 ドルの株は買えない
    assert buys == ["A", "B", "C", "D", "E"]
    spent = sum(o.qty * o.ref_price * live.BUY_BUFFER for o in orders if o.side == "BUY")
    assert spent <= 2000


def test_plan_momentum_keeps_holdings_in_top():
    book = {"cash": 400.0, "positions": {"A": {"qty": 8, "cost": 50.0}}}
    prices = {"A": 50.0, "B": 100.0}
    orders = live.plan_momentum(book, ["A", "B"], prices, top_n=2)
    assert [(o.side, o.symbol) for o in orders] == [("BUY", "B")]


def test_last_month_end():
    idx = pd.bdate_range("2026-10-01", "2026-11-05")
    assert live.last_month_end(idx, pd.Timestamp("2026-11-05")) == pd.Timestamp("2026-10-30")


def test_momentum_ranking_uses_12_1_and_membership():
    idx = pd.bdate_range("2025-01-01", periods=300)
    rng = np.random.default_rng(0)
    data = {
        "AAPL": pd.DataFrame({"close": np.linspace(100, 200, 300)}, index=idx),  # 加盟中、上昇
        "MSFT": pd.DataFrame({"close": np.linspace(100, 120, 300)}, index=idx),  # 加盟中、小幅上昇
        "ZZZZ": pd.DataFrame({"close": np.linspace(100, 400, 300)}, index=idx),  # 加盟していない
    }
    r = live.momentum_ranking(data, idx[-1])
    assert r[:2] == ["AAPL", "MSFT"] and "ZZZZ" not in r


def test_plan_swing_stop_exit_and_entries():
    book = {"cash": 1000.0, "positions": {
        "LOSS": {"qty": 4, "cost": 100.0},   # 終値 89 → 損切り
        "EXIT": {"qty": 2, "cost": 100.0},   # 20日安値割れ
        "KEEP": {"qty": 2, "cost": 100.0},
    }}
    prices = {"LOSS": 89.0, "EXIT": 105.0, "KEEP": 110.0, "NEW1": 50.0, "BIG": 5000.0, "NEW2": 30.0}
    orders = live.plan_swing(book, ["BIG", "NEW1", "NEW2", "KEEP"], {"EXIT"}, prices, max_pos=3)
    sells = {o.symbol: o.reason for o in orders if o.side == "SELL"}
    assert set(sells) == {"LOSS", "EXIT"} and "損切り" in sells["LOSS"]
    buys = [o.symbol for o in orders if o.side == "BUY"]
    assert buys == ["NEW1", "NEW2"]  # BIG は買えない、KEEP は保有中。枠は 3 まで
    assert orders.index(next(o for o in orders if o.side == "BUY")) > 1  # 売りが先


def test_swing_signals_breakout_and_exit():
    idx = pd.bdate_range("2025-01-01", periods=300)
    up = np.r_[np.linspace(100, 150, 299), 200.0]           # 最終日に高値更新
    down = np.r_[np.linspace(100, 150, 290), np.full(10, 90)]  # 最終日には20日安値割れ済み
    stocks = {"AAPL": pd.DataFrame({"close": up}, index=idx), "MSFT": pd.DataFrame({"close": down}, index=idx)}
    spy = pd.DataFrame({"close": np.linspace(100, 200, 300)}, index=idx)
    entries, exits = live.swing_signals(stocks, spy, idx[-1])
    assert entries == ["AAPL"]
    entries2, exits2 = live.swing_signals(stocks, spy, idx[-10])
    assert "MSFT" in exits2  # 急落した日に売りシグナル


def test_ensure_books_added_without_touching_existing():
    lg = ledger()
    lg["strategies"]["swing"]["cash"] = 123.0
    live.ensure_swing_book(lg)
    assert lg["strategies"]["swing"]["cash"] == 123.0
    assert lg["strategies"]["lev3"]["cash"] == live.INITIAL_CASH
    assert lg["strategies"]["swing_aggr"]["positions"] == {}


def test_lev3_target_and_plan():
    idx = pd.bdate_range("2025-01-01", periods=260)
    up = pd.DataFrame({"close": np.linspace(100, 200, 260)}, index=idx)
    down = pd.DataFrame({"close": np.r_[np.linspace(100, 200, 200), np.linspace(200, 120, 60)]}, index=idx)
    assert live.lev3_target(up, idx[-1]) == "UPRO"
    assert live.lev3_target(down, idx[-1]) == "BIL"
    book = {"cash": 2000.0, "positions": {}}
    orders = live.plan_trend(book, "UPRO", {"UPRO": 128.0, "BIL": 91.0}, strategy="lev3", on_symbol="UPRO")
    assert [(o.strategy, o.side, o.symbol, o.qty) for o in orders] == [("lev3", "BUY", "UPRO", 15)]
