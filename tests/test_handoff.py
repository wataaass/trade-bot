"""引き継ぎ用の判断関数が、検証で使った zoo の判断と一致することを確かめる。"""

import numpy as np
import pandas as pd
import pytest

from tradebot import data as dt
from tradebot import handoff as ho
from tradebot import zoo


def load_or_skip(symbol: str):
    """実データを取得する。ネットにつながらない環境では、失敗ではなく理由つきでスキップする。"""
    try:
        return dt.load(symbol)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{symbol} の日足を Yahoo Finance から取得できない: {type(e).__name__}: {e}")


def test_lev3_matches_zoo_on_real_spy():
    spy = load_or_skip("SPY")
    mkt = spy
    sig = zoo.c_sma_asym(200, "L3", 0.01)({"MKT": mkt}, mkt)
    days = spy.index[300::23]  # 実データの色々な日で照合
    for d in days:
        want = "UPRO" if sig["L3"]["entry"].loc[d] else "BIL"
        assert ho.target_lev3(spy["close"].loc[:d]) == want, d


def test_voltarget_matches_zoo_on_real_spy():
    spy = load_or_skip("SPY")
    data = {"MKT": spy, "L2": spy, "L3": spy, "CASH": spy}
    sig = zoo.c_voltarget_monthly_lev(200, 0.15, 0.25, 0.01)(data, spy)
    name = {"L3": "UPRO", "L2": "SSO", "MKT": "SPYM", "CASH": "BIL"}
    for d in spy.index[300::29]:
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


def test_trend_on_keeps_state_inside_band():
    idx = pd.bdate_range("2024-01-01", periods=260)
    base = np.full(260, 100.0)
    above = base.copy()
    above[-10:] = 103.0  # 線の3%上 → 持つ
    inside = above.copy()
    inside[-3:] = 100.6  # その後、線の+0.6%（遊びの中）→ 持つを続ける
    below = base.copy()
    below[-10:] = 99.0  # 線の下 → 逃げる
    back = below.copy()
    back[-3:] = 100.6  # 線の+0.6%に戻っただけ → 逃げたまま
    assert ho.trend_on(pd.Series(above, index=idx))
    assert ho.trend_on(pd.Series(inside, index=idx))
    assert not ho.trend_on(pd.Series(below, index=idx))
    assert not ho.trend_on(pd.Series(back, index=idx))


def test_every_day_matches_zoo_without_lookahead():
    """直近500営業日の毎日、その日までのデータだけで判断し、検証の判断と一致することを確かめる。"""
    spy = load_or_skip("SPY")
    data = {"MKT": spy, "L2": spy, "L3": spy, "CASH": spy}
    sig_a = zoo.c_sma_asym(200, "L3", 0.01)({"MKT": spy}, spy)
    sig_b = zoo.c_voltarget_monthly_lev(200, 0.15, 0.25, 0.01)(data, spy)
    name = {"L3": "UPRO", "L2": "SSO", "MKT": "SPYM", "CASH": "BIL"}
    close = spy["close"]
    for d in spy.index[-500:]:
        past = close.loc[:d]  # その日までのデータだけ
        want_a = "UPRO" if sig_a["L3"]["entry"].loc[d] else "BIL"
        assert ho.target_lev3(past) == want_a, d
        want_b = [name[k] for k in ("L3", "L2", "MKT", "CASH") if sig_b[k]["entry"].loc[d]]
        assert [ho.target_voltarget(past)] == want_b, d


def test_370_bars_are_enough():
    """moomoo の1回の取得（最大370本）だけで、全履歴を使った判定と同じになることを確かめる。"""
    close = load_or_skip("SPY")["close"]
    for d in close.index[close.index >= "2000-01-01"][::5]:
        past = close.loc[:d]
        assert ho.trend_on(past.iloc[-370:]) == ho.trend_on(past), d
        assert ho.target_voltarget(past.iloc[-370:]) == ho.target_voltarget(past), d


def test_closes_from_moomoo_feeds_handoff():
    spy = load_or_skip("SPY")["close"].iloc[-370:]
    # moomoo の返り値の形（新しい順に並ぶこともある）を再現する
    kl = [{"date": int(d.strftime("%Y%m%d")), "close": float(v), "open": 0} for d, v in spy.items()][::-1]
    s = ho.closes_from_moomoo(kl)
    assert s.index.is_monotonic_increasing and len(s) == 370
    assert ho.target_lev3(s) == ho.target_lev3(spy)
    assert ho.target_voltarget(s) == ho.target_voltarget(spy)
    with pytest.raises(ValueError):
        ho.closes_from_moomoo([])


def test_every_day_matches_zoo_for_b_prime():
    """B′（境目 20/35%）も、直近500営業日の毎日で検証の判断と一致する。"""
    spy = load_or_skip("SPY")
    data = {"MKT": spy, "L2": spy, "L3": spy, "CASH": spy}
    sig = zoo.c_voltarget_monthly_lev(200, 0.20, 0.35, 0.01)(data, spy)
    name = {"L3": "UPRO", "L2": "SSO", "MKT": "SPYM", "CASH": "BIL"}
    close = spy["close"]
    for d in spy.index[-500:]:
        want = [name[k] for k in ("L3", "L2", "MKT", "CASH") if sig[k]["entry"].loc[d]]
        assert [ho.target_voltarget(close.loc[:d], lo=0.20, hi=0.35)] == want, d


def test_plan_switch():
    prices = {"UPRO": 128.0, "BIL": 91.5, "SSO": 95.0}
    # 何も持っていない → 目標を買う（現金の1%を残す）
    assert ho.plan_switch({}, 2000.0, "UPRO", prices) == [{"side": "BUY", "symbol": "UPRO", "qty": 15}]
    # 目標を持っている → 何もしない
    assert ho.plan_switch({"UPRO": 15}, 80.0, "UPRO", prices) == []
    # 乗り換え → 売りが先、売った代金と現金で買う
    orders = ho.plan_switch({"UPRO": 15}, 80.0, "BIL", prices)
    assert orders[0] == {"side": "SELL", "symbol": "UPRO", "qty": 15}
    assert orders[1]["side"] == "BUY" and orders[1]["symbol"] == "BIL"
    assert orders[1]["qty"] == int((80 + 15 * 128) * 0.99 // 91.5)
    # B の乗り換え（UPRO→SSO）。0株の銘柄は売らない
    orders = ho.plan_switch({"UPRO": 10, "BIL": 0}, 5.0, "SSO", prices)
    assert [o["symbol"] for o in orders] == ["UPRO", "SSO"]


def test_switch_threshold_is_exact():
    close = load_or_skip("SPY")["close"].iloc[:-1]
    nxt = close.index[-1] + pd.Timedelta(days=1)
    kind, p = ho.switch_threshold(close)
    before = ho.trend_on(close)
    for q, flips in ((p * 1.0001, kind == "reentry"), (p * 0.9999, kind == "exit")):
        after = ho.trend_on(pd.concat([close, pd.Series([q], index=[nxt])]))
        assert (after != before) == flips, (kind, q)
    # 逃げている状態でも確かめる（下落の後）
    down = pd.Series(np.r_[np.linspace(100, 150, 250), np.linspace(150, 120, 30)],
                     index=pd.bdate_range("2024-01-01", periods=280))
    kind, p = ho.switch_threshold(down)
    assert kind == "reentry"
    nxt = down.index[-1] + pd.Timedelta(days=1)
    assert ho.trend_on(pd.concat([down, pd.Series([p * 1.0001], index=[nxt])]))
    assert not ho.trend_on(pd.concat([down, pd.Series([p * 0.9999], index=[nxt])]))


def test_every_day_matches_zoo_for_continuous_b():
    """B の連続版: 直近500営業日の毎日、前日の目標を引き継ぎながら判断し、検証の比率と一致する。"""
    spy = load_or_skip("SPY")
    close = spy["close"]
    for k in (ho.K_B, ho.K_B_PRIME):
        want = zoo.voltarget_share(close, k, ho.BAND)
        days = spy.index[-500:]
        prev = float(want.loc[:days[0]].iloc[-2])
        for d in days:
            got = ho.upro_share(close.loc[:d], k, prev)
            assert got == pytest.approx(want.loc[d], abs=1e-12), d
            prev = got


def test_upro_share_ignores_small_changes():
    idx = pd.bdate_range("2023-01-02", periods=400)
    rng = np.random.default_rng(3)
    c = pd.Series(100 * np.cumprod(1 + 0.002 + rng.normal(0, 0.015, 400)), index=idx)  # 上昇、値動き 約24%
    raw = ho.upro_share(c)
    assert 0 < raw < 1
    assert ho.upro_share(c, prev=raw + 0.05) == raw + 0.05  # 差が 0.1 未満なら前の比率のまま
    assert ho.upro_share(c, prev=raw + 0.15) == raw  # 0.1 以上なら変える
    assert ho.upro_share(c, prev=0.0) == raw  # 逃げていた状態から戻るときは必ず変える


def test_daily_plan_follows_targets_and_trades_only_on_changes():
    """直近500営業日、daily_plan の注文を約定させ続けると、持ち株の比率が検証の目標に沿い、
    注文が出るのは目標が変わった日（と初日）だけになる。"""
    spy = load_or_skip("SPY")
    close = spy["close"]
    days = spy.index[-500:]
    upro = 100 * (close / close.iloc[0]) ** 3  # 値段の目安（約定の計算用）
    bil = pd.Series(91.5, index=close.index)
    for bot, k in (("A", None), ("B", ho.K_B), ("B′", ho.K_B_PRIME)):
        if k is None:
            sig = zoo.c_sma_asym(200, "L3", 0.01)({"MKT": spy}, spy)
            want = sig["L3"]["entry"].astype(float)
        else:
            want = zoo.voltarget_share(close, k, ho.BAND)
        holdings, cash, state, n_orders = {}, 2000.0, None, 0
        for d in days:
            p = {"UPRO": float(upro.loc[d]), "BIL": float(bil.loc[d])}
            orders, state = ho.daily_plan(bot, close.loc[:d], holdings, cash, p, state)
            assert state["target"]["UPRO"] == pytest.approx(want.loc[d], abs=1e-12), (bot, d)
            if orders:
                n_orders += 1
            for o in orders:  # 同じ値段で約定したことにする
                q = o["qty"] if o["side"] == "BUY" else -o["qty"]
                holdings[o["symbol"]] = holdings.get(o["symbol"], 0) + q
                cash -= q * p[o["symbol"]]
            assert cash >= 0 and all(q >= 0 for q in holdings.values())
            value = cash + sum(q * p[s] for s, q in holdings.items())
            if orders:  # 注文した日は、比率が目標に1株分ほどの誤差で合う
                got = holdings.get("UPRO", 0) * p["UPRO"] / value
                assert abs(got - state["target"]["UPRO"]) <= p["UPRO"] / value + 0.02, (bot, d)
        changes = int((want.loc[days].diff().abs() > 1e-12).sum())
        assert n_orders <= changes + 1, bot
