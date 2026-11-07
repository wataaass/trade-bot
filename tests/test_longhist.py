"""約100年分のデータ（Fama/French・Shiller）が、知られている値の範囲に収まっているかを確かめる。"""

import pytest

from tradebot import longhist as lh


def cagr(c):
    return (c.iloc[-1] / c.iloc[0]) ** (365.25 / (c.index[-1] - c.index[0]).days) - 1


@pytest.fixture(scope="module")
def mc():
    try:
        return lh.market_and_cash()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Fama/French・Shiller のデータを取得できない: {type(e).__name__}: {e}")


def test_market_and_cash_are_plausible(mc):
    mkt, cash = mc
    assert mkt.index[0].year == 1926 and mkt.index[-1].year >= 2025
    assert 0.09 < cagr(mkt["close"]) < 0.115      # 米国株全体は約100年で年10%前後
    assert 0.025 < cagr(cash["close"]) < 0.04     # 短期国債は年3%前後
    dy = mkt["div_yield"] * 252
    assert dy.between(0.005, 0.10).mean() > 0.99  # 配当利回りは年0.5〜10%


def test_bond10_matches_known_eras(mc):
    mkt, _ = mc
    b = lh.bond10(mkt.index)["close"]
    assert 0.02 < cagr(b.loc["1950":"1980"]) < 0.04   # 金利上昇期は低い
    assert 0.09 < cagr(b.loc["1981":"2000"]) < 0.125  # 金利低下期は高い


def test_leveraged_tracks_lev_times_market(mc):
    mkt, cash = mc
    l3 = lh.leveraged(mkt, cash, 3)["close"].loc["1990":"1990-12-31"].pct_change().dropna()
    m = mkt["close"].loc["1990":"1990-12-31"].pct_change().dropna()
    assert ((l3 - 3 * m).abs() < 0.001).mean() > 0.99  # 日々ほぼ3倍（差は借入コストと信託報酬）
