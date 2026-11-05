import numpy as np
import pandas as pd
import pytest

from tradebot import backtest as bt
from tradebot import indicators as ind
from tradebot import strategies as st


def make_df(closes, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(closes))
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1})


def sig(df, entry_days=(), exit_days=()):
    e = pd.Series(False, index=df.index)
    x = pd.Series(False, index=df.index)
    e.iloc[list(entry_days)] = True
    x.iloc[list(exit_days)] = True
    return pd.DataFrame({"entry": e, "exit": x, "score": 0.0})


NO_COST = bt.Costs(commission_rate=0, slippage=0)


def test_fills_next_open_not_signal_close():
    df = make_df([10, 11, 12, 13, 14])
    df["open"] = [10, 20, 12, 13, 14]  # 2日目の始値だけ高い
    cfg = bt.Config(initial_cash=1000, max_positions=1, fractional=True, costs=NO_COST)
    res = bt.run({"A": df}, {"A": sig(df, entry_days=[0], exit_days=[3])}, cfg)
    t = res.trades[0]
    assert t.entry_date == df.index[1] and t.entry_price == 20
    assert t.exit_date == df.index[4] and t.exit_price == 14


def test_hold_forever_matches_buy_and_hold():
    rng = np.random.default_rng(0)
    df = make_df(100 * np.cumprod(1 + rng.normal(0, 0.01, 300)))
    cfg = bt.Config(initial_cash=10_000, max_positions=1, fractional=True)
    # 初日の前日に買いシグナルを出したことにして、初日の始値で買わせる
    pre = df.index[0] - pd.Timedelta(days=1)
    df2 = pd.concat([df.iloc[:1].set_axis([pre]), df])
    s = sig(df2, entry_days=[0])
    res = bt.run({"A": df2}, {"A": s}, cfg, start=str(pre.date()))
    bh = bt.buy_and_hold(df, cfg)
    # 最終日は手仕舞いの手数料とスリッページの分だけ違う
    assert res.equity.iloc[-2] == pytest.approx(bh.iloc[-2], rel=1e-9)


def test_costs_reduce_equity():
    df = make_df([10] * 20)
    s = sig(df, entry_days=[0, 5, 10], exit_days=[3, 8, 13])
    free = bt.run({"A": df}, {"A": s}, bt.Config(initial_cash=1000, max_positions=1, fractional=True, costs=NO_COST))
    paid = bt.run({"A": df}, {"A": s}, bt.Config(initial_cash=1000, max_positions=1, fractional=True))
    assert free.equity.iloc[-1] == pytest.approx(1000)
    assert paid.equity.iloc[-1] < 1000
    assert all(t.ret < 0 for t in paid.trades)  # 値動きゼロなら手数料の分だけ負ける


def test_stop_loss_gap_fills_at_open():
    df = make_df([100, 100, 100, 80, 80])
    df.loc[df.index[3], "open"] = 85  # 損切り値 90 より下に窓を開けて始まる
    cfg = bt.Config(initial_cash=1000, max_positions=1, fractional=True, stop_loss=0.10, costs=NO_COST)
    res = bt.run({"A": df}, {"A": sig(df, entry_days=[0])}, cfg)
    t = res.trades[0]
    assert t.reason == "stop" and t.exit_price == 85


def test_whole_shares_and_max_positions():
    a, b, c = make_df([300] * 5), make_df([50] * 5), make_df([50] * 5)
    cfg = bt.Config(initial_cash=500, max_positions=2, fractional=False, costs=NO_COST)
    sigs = {k: sig(d, entry_days=[0]) for k, d in {"A": a, "B": b, "C": c}.items()}
    sigs["B"]["score"] = 1.0
    res = bt.run({"A": a, "B": b, "C": c}, sigs, cfg)
    syms = sorted(t.symbol for t in res.trades)
    assert len(syms) <= 2
    for t in res.trades:
        assert t.shares == int(t.shares)
    assert "B" in syms  # スコアが高いものが優先される


def test_signals_do_not_look_ahead():
    """未来のデータを書き換えても、過去のシグナルは変わらない。"""
    rng = np.random.default_rng(1)
    closes = 100 * np.cumprod(1 + rng.normal(0.0005, 0.015, 600))
    data = {s: make_df(closes * (1 + i * 0.1)) for i, s in enumerate(["A", "B", "C"])}
    bench = data["A"]
    cut = 450
    for name, fn in st.STRATEGIES.items():
        full = fn(data, bench)
        trunc_data = {s: d.iloc[:cut] for s, d in data.items()}
        part = fn(trunc_data, trunc_data["A"])
        for s in data:
            a = full[s].iloc[: cut - 1]  # 月末判定は翌日の有無を見るので最終日は除く
            b = part[s].iloc[: cut - 1]
            pd.testing.assert_frame_equal(a, b, check_freq=False, obj=f"{name}/{s}")


def test_rsi_bounds():
    rng = np.random.default_rng(2)
    s = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.02, 200)))
    r = ind.rsi(s, 2).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    assert ind.rsi(pd.Series(range(1, 30), dtype=float), 2).iloc[-1] == 100


def test_cannot_afford_one_share_skips_entry():
    df = make_df([1000] * 5)
    cfg = bt.Config(initial_cash=500, max_positions=1, fractional=False)
    res = bt.run({"A": df}, {"A": sig(df, entry_days=[0])}, cfg)
    assert res.trades == [] and res.equity.iloc[-1] == 500


def test_symbol_without_data_yet_is_skipped():
    a = make_df([10] * 10)
    b = make_df([20] * 5, start=str(a.index[5].date()))  # 途中から上場
    sb = sig(b, entry_days=[0]).reindex(a.index, fill_value=False)
    sb["score"] = 0.0
    sb.iloc[0, 0] = True  # 上場前の日に買いシグナル
    cfg = bt.Config(initial_cash=1000, max_positions=1, fractional=True, costs=NO_COST)
    res = bt.run({"A": a, "B": b}, {"A": sig(a), "B": sb}, cfg)
    assert all(t.entry_date >= b.index[0] for t in res.trades)
    assert not res.equity.isna().any()
