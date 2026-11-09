import numpy as np
import pandas as pd

from tradebot import kronos_bot as kb


def _df(index: pd.DatetimeIndex, close: np.ndarray, volume: float = 1e6) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": volume, "div_yield": 0.0}, index=index
    )


def test_week_ends_picks_last_trading_day_of_each_week():
    idx = pd.bdate_range("2025-01-06", "2025-01-22").delete(4)  # 2025-01-10（金）を休場にする
    got = kb.week_ends(idx, "2025-01-01")
    # 最後の週（1/20〜1/22）は週の途中で終わるので含めない
    assert got == [pd.Timestamp("2025-01-09"), pd.Timestamp("2025-01-17")]


def test_select_universe_ranks_by_dollar_volume_and_needs_history():
    idx = pd.bdate_range("2025-01-01", periods=100)
    data = {
        "BIG": _df(idx, np.full(100, 100.0), 1e6),
        "MID": _df(idx, np.full(100, 50.0), 1e6),
        "NEW": _df(idx[-20:], np.full(20, 1000.0), 1e6),  # 履歴が短い
    }
    got = kb.select_universe(data, ["BIG", "MID", "NEW", "GONE"], idx[-1], n=5, lookback=60)
    assert got == ["BIG", "MID"]


def test_pick_top_with_buffer_keeps_holdings_until_they_fall_out():
    d = pd.to_datetime(["2025-01-03", "2025-01-10"])
    scores = pd.DataFrame({"A": [0.05, 0.02], "B": [0.04, 0.01], "C": [0.01, 0.03], "D": [0.0, 0.04]}, index=d)
    plain = kb.pick_top(scores, k=2)
    assert plain.loc[d[1]].to_dict() == {"A": 0.0, "B": 0.0, "C": 0.5, "D": 0.5}
    buffered = kb.pick_top(scores, k=2, buffer=1)
    # A は3位以内なので残る。B は4位に落ちたので抜け、空いた分に1位の D が入る
    assert buffered.loc[d[1]].to_dict() == {"A": 0.5, "B": 0.0, "C": 0.0, "D": 0.5}
    buffered2 = kb.pick_top(scores, k=2, buffer=2)
    assert buffered2.loc[d[1]].to_dict() == {"A": 0.5, "B": 0.5, "C": 0.0, "D": 0.0}


def test_rank_ic_and_top_minus_all():
    d = pd.to_datetime(["2025-01-03"])
    cols = list("ABCDEF")
    pred = pd.DataFrame([[6, 5, 4, 3, 2, 1]], index=d, columns=cols, dtype=float)
    real = pd.DataFrame([[0.06, 0.05, 0.04, 0.03, 0.02, 0.01]], index=d, columns=cols)
    assert kb.rank_ic(pred, real).iloc[0] == 1.0
    assert kb.rank_ic(-pred, real).iloc[0] == -1.0
    np.testing.assert_allclose(kb.top_minus_all(pred, real, 2).iloc[0], 0.055 - 0.035)


def test_forward_returns_and_momentum():
    idx = pd.bdate_range("2025-01-01", periods=10)
    data = {"A": _df(idx, np.arange(1, 11, dtype=float))}
    fr = kb.forward_returns(data, ["A"], idx[2], horizon=5)
    assert fr["A"] == 8 / 3 - 1
    assert kb.forward_returns(data, ["A"], idx[-2], horizon=5).empty  # 先の値段がまだない
    mom = kb.momentum_scores(data, ["A"], idx[-1], long=4, skip=1)
    assert mom["A"] == 9 / 6 - 1


class _FakePredictor:
    """predict_batch の入力を確かめ、最後の終値の 1.1 倍を返す。"""

    def __init__(self):
        self.calls = []

    def predict_batch(self, dfs, xs, ys, pred_len, **kw):
        self.calls.append((dfs, xs, ys, pred_len, kw))
        return [pd.DataFrame({"close": [df["close"].iloc[-1] * 1.1] * pred_len}, index=y) for df, y in zip(dfs, ys, strict=True)]


def test_predict_week_uses_only_data_up_to_date():
    idx = pd.bdate_range("2025-01-01", periods=30)
    data = {"A": _df(idx, np.arange(1, 31, dtype=float)), "B": _df(idx, np.full(30, 5.0))}
    fake = _FakePredictor()
    got = kb.predict_week(fake, data, ["A", "B"], idx[19], lookback=10, pred_len=5, sample_count=3)
    np.testing.assert_allclose(got.to_numpy(), [0.1, 0.1])
    dfs, xs, ys, pred_len, kw = fake.calls[0]
    assert len(dfs[0]) == 10 and dfs[0]["close"].iloc[-1] == 20.0  # 判定日より先を見ていない
    assert xs[0].iloc[-1] == idx[19] and ys[0].iloc[0] > idx[19] and len(ys[0]) == 5
    assert kw["sample_count"] == 3 and "amount" in dfs[0]
