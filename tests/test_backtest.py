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
    data = {s: make_df(closes * (1 + i * 0.1)) for i, s in enumerate(["SPY", "IEF", "BIL"])}
    bench = data["SPY"]
    cut = 450
    for name, fn in st.STRATEGIES.items():
        full = fn(data, bench)
        trunc_data = {s: d.iloc[:cut] for s, d in data.items()}
        part = fn(trunc_data, trunc_data["SPY"])
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


def test_trend_switch_holds_one_side_only():
    up = make_df(np.linspace(100, 200, 300))
    down = make_df(np.r_[np.linspace(100, 200, 250), np.linspace(200, 120, 50)])
    for spy in (up, down):
        data = {"SPY": spy, "IEF": make_df([100] * 300)}
        sigs = st.trend_switch(data, spy, risk_off="IEF")
        both = sigs["SPY"]["entry"] & sigs["IEF"]["entry"]
        assert not both.any()
    s = st.trend_switch({"SPY": down, "IEF": make_df([100] * 300)}, down, risk_off="IEF")
    assert s["SPY"]["exit"].iloc[-1] and s["IEF"]["entry"].iloc[-1]  # 下落後は債券側


def _yearly(closes, start="2020-12-28"):
    return make_df(closes, start=start)


def test_capital_gains_taxed_once_per_year():
    df = _yearly([100, 100, 110, 110, 110, 110])  # 年をまたぐ
    s = sig(df, entry_days=[0], exit_days=[2])  # 2021年に入ってから +10% で売る
    cfg = bt.Config(initial_cash=1000, max_positions=1, fractional=True, costs=NO_COST, tax=bt.TAXABLE)
    res = bt.run({"A": df}, {"A": s}, cfg)
    gain = 100.0  # 1000ドルで10株、10ドルずつ値上がり
    assert res.tax_paid == pytest.approx(gain * 0.20315)
    assert res.equity.iloc[-1] == pytest.approx(1000 + gain * (1 - 0.20315))


def test_loss_carried_forward_offsets_next_year():
    closes = [100] * 3 + [90] * 3 + [90] * 250 + [99] * 5
    df = make_df(closes, start="2020-12-24")
    exit_loss = 4  # 2020年中に損切り（-10）
    re_entry = 6
    exit_gain = len(closes) - 3
    s = sig(df, entry_days=[0, re_entry], exit_days=[exit_loss, exit_gain])
    cfg = bt.Config(initial_cash=1000, max_positions=1, fractional=True, costs=NO_COST, tax=bt.TAXABLE)
    res = bt.run({"A": df}, {"A": s}, cfg)
    assert [t.exit_date.year for t in res.trades][:1] == [2020]
    # 2020 年の損失 -100 を 2021 年の利益 +100 で相殺するので税金はゼロ
    assert res.tax_paid == pytest.approx(0, abs=1e-6)


def test_dividend_tax_only_when_held_before_ex_date():
    df = make_df([100] * 6)
    df["div_yield"] = [0, 0, 0, 0.01, 0, 0]
    cfg = bt.Config(initial_cash=1000, max_positions=1, fractional=True, costs=NO_COST, tax=bt.TAXABLE)
    held = bt.run({"A": df}, {"A": sig(df, entry_days=[0])}, cfg)
    assert held.tax_paid == pytest.approx(1000 * 0.01 * bt.TAXABLE.dividend)
    late = bt.run({"A": df}, {"A": sig(df, entry_days=[2])}, cfg)  # 権利落ち日の始値で買う
    assert late.tax_paid == 0


def test_nisa_has_no_capital_gains_tax():
    df = _yearly([100, 100, 110, 110, 110, 110])
    s = sig(df, entry_days=[0], exit_days=[2])
    cfg = bt.Config(initial_cash=1000, max_positions=1, fractional=True, costs=NO_COST, tax=bt.NISA)
    assert bt.run({"A": df}, {"A": s}, cfg).tax_paid == 0


def test_dividend_tax_reduces_shares_when_fully_invested():
    df = make_df([100] * 6)
    df["div_yield"] = [0, 0, 0, 0.02, 0, 0]
    cfg = bt.Config(initial_cash=1000, max_positions=1, fractional=True, costs=NO_COST, tax=bt.NISA)
    res = bt.run({"A": df}, {"A": sig(df, entry_days=[0])}, cfg)
    # 配当 2% に 10% の税金 → 資産は 0.2% 減り、現金はマイナスにならない
    assert res.equity.iloc[-1] == pytest.approx(1000 * (1 - 0.002))
    assert (res.equity > 0).all()


def test_position_closed_when_data_ends():
    a = make_df([10] * 10)
    b = make_df([20, 20, 20, 25])  # 4日でデータが途切れる
    cfg = bt.Config(initial_cash=1000, max_positions=1, fractional=True, costs=NO_COST)
    res = bt.run({"A": a, "B": b}, {"A": sig(a), "B": sig(b, entry_days=[0])}, cfg)
    t = res.trades[0]
    assert t.reason == "delisted" and t.exit_price == 25
    assert res.equity.iloc[-1] == pytest.approx(1000 * 25 / 20)


def test_stock_momentum_market_filter_keeps_winners():
    """相場フィルターが常に OK なら、フィルターなしと同じシグナルになる。"""
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2020-01-01", periods=400)
    data = {s: pd.DataFrame({"close": 100 * np.cumprod(1 + rng.normal(0.0004 * i, 0.01, 400))}, index=idx)
            for i, s in enumerate("ABCDEF")}
    bench = make_df(np.linspace(100, 300, 380), start="2019-12-01")  # 祝日のずれを再現するため日付を変える
    members = pd.DataFrame(True, index=idx, columns=list("ABCDEF"))
    plain = st.stock_momentum(data, bench, members, top_n=2, exit_rank=4)
    filt = st.stock_momentum(data, bench, members, top_n=2, exit_rank=4, market_filter=True)
    later = idx[idx >= bench.index[220]]  # 200日線が計算できる日以降
    later = later[later <= bench.index[-1]]
    for s in data:
        assert (plain[s].loc[later, "exit"] == filt[s].loc[later, "exit"]).all()
        assert filt[s]["exit"].dtype == bool


def test_year_end_harvest_realizes_loss_and_resets_basis():
    # 2020年に100で買い、年末に80（含み損）、2021年に100まで戻して売る
    closes = [100, 100, 90, 80, 80, 90, 100, 100]
    df = make_df(closes, start="2020-12-24")  # 12/24〜: 12/31 が年末の最終営業日
    s = sig(df, entry_days=[0], exit_days=[6])
    base = dict(initial_cash=1000, max_positions=1, fractional=True, costs=NO_COST, tax=bt.TAXABLE)
    plain = bt.run({"A": df}, {"A": s}, bt.Config(**base))
    harv = bt.run({"A": df}, {"A": s}, bt.Config(**base, harvest=True))
    # 損出しなし: 100→100 で損益ゼロ、税金ゼロ
    assert plain.tax_paid == pytest.approx(0, abs=1e-9)
    # 損出しあり: 2020年に -200 を確定（繰り越し）、2021年に 80→100 の +200 と相殺 → 税金ゼロ、手取りは同じ
    assert harv.tax_paid == pytest.approx(0, abs=1e-9)
    assert harv.equity.iloc[-1] == pytest.approx(plain.equity.iloc[-1])


def test_harvest_offsets_same_year_gain():
    # 同じ年に別の銘柄で利益が出ていれば、年末の含み損を確定させて税金が減る
    a = make_df([100, 100, 120, 120, 120, 120, 120, 120], start="2020-12-24")
    b = make_df([100, 100, 100, 80, 80, 80, 80, 80], start="2020-12-24")  # 最後の2日は2021年
    sa = sig(a, entry_days=[0], exit_days=[2])  # A は 100 で買い 120 で年内に売る
    sb = sig(b, entry_days=[0])
    base = dict(initial_cash=2000, max_positions=2, fractional=True, costs=NO_COST, tax=bt.TAXABLE)
    plain = bt.run({"A": a, "B": b}, {"A": sa, "B": sb}, bt.Config(**base))
    harv = bt.run({"A": a, "B": b}, {"A": sa, "B": sb}, bt.Config(**base, harvest=True))
    assert harv.tax_paid < plain.tax_paid


def test_withholding_moves_tax_earlier_but_same_total():
    # 2020年に1回だけ +10% で売り、その後は現金のまま年を越す
    df = make_df([100, 100, 110, 110, 110, 110, 110, 110], start="2020-12-21")
    s = sig(df, entry_days=[0], exit_days=[1])
    base = dict(initial_cash=1000, max_positions=1, fractional=True, costs=NO_COST)
    yearly = bt.run({"A": df}, {"A": s}, bt.Config(**base, tax=bt.TAXABLE))
    wh = bt.run({"A": df}, {"A": s}, bt.Config(**base, tax=bt.Tax(withhold=True)))
    assert wh.tax_paid == pytest.approx(yearly.tax_paid) == pytest.approx(100 * 0.20315)
    assert wh.equity.iloc[-1] == pytest.approx(yearly.equity.iloc[-1])
    sold = yearly.trades[0].exit_date
    assert wh.equity.loc[sold] == pytest.approx(yearly.equity.loc[sold] - 100 * 0.20315)  # 売った日に引かれる


def test_withholding_refunds_when_loss_follows():
    # 同じ年に +100 で売った後、-50 で売る → 通算 +50 の分だけ引かれている
    a = make_df([100, 100, 110, 110, 110, 110, 110, 110], start="2020-12-14")
    b = make_df([100, 100, 100, 100, 90, 90, 90, 90], start="2020-12-14")
    sa = sig(a, entry_days=[0], exit_days=[1])
    sb = sig(b, entry_days=[0], exit_days=[3])
    cfg = bt.Config(initial_cash=2000, max_positions=2, fractional=True, costs=NO_COST, tax=bt.Tax(withhold=True))
    res = bt.run({"A": a, "B": b}, {"A": sa, "B": sb}, cfg)
    realized = sum(t.exit_price * t.shares - t.entry_price * t.shares for t in res.trades)
    assert res.tax_paid == pytest.approx(max(0.0, realized) * 0.20315)
