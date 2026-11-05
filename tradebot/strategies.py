"""スイング向けの戦略候補。

どれも広く知られた型で、パラメータは定番の値に固定する。
過去データに合わせてパラメータを細かく調整すると、検証では勝っても本番で負けやすいため。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind


def _frame(entry: pd.Series, exit_: pd.Series, score: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"entry": entry.fillna(False), "exit": exit_.fillna(False), "score": score})


def market_ok(bench: pd.DataFrame, length: int = 200) -> pd.Series:
    """相場全体のフィルター: 指数が長期移動平均より上なら新規の買いを許す。"""
    return bench["close"] > ind.sma(bench["close"], length)


def rsi2_pullback(data: dict[str, pd.DataFrame], bench: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """上昇トレンド中の押し目買い（Connors の RSI(2)）。

    買い: 終値 > 200日線、RSI(2) < 10、相場フィルター OK
    売り: 終値 > 5日線
    優先: RSI が低いほど（深く押したものから）
    """
    mok = market_ok(bench)
    out = {}
    for s, df in data.items():
        c = df["close"]
        r = ind.rsi(c, 2)
        entry = (c > ind.sma(c, 200)) & (r < 10) & mok.reindex(c.index).fillna(False)
        exit_ = c > ind.sma(c, 5)
        out[s] = _frame(entry, exit_, -r)
    return out


def breakout(data: dict[str, pd.DataFrame], bench: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """ドンチャン・ブレイクアウト（タートル型の順張り）。

    買い: 終値が過去55日の高値を更新、相場フィルター OK
    売り: 終値が過去20日の安値を割る
    優先: 126日の値上がり率が高いほど
    """
    mok = market_ok(bench)
    out = {}
    for s, df in data.items():
        c = df["close"]
        hi = c.rolling(55).max().shift(1)
        lo = c.rolling(20).min().shift(1)
        entry = (c > hi) & mok.reindex(c.index).fillna(False)
        exit_ = c < lo
        out[s] = _frame(entry, exit_, c.pct_change(126))
    return out


def momentum_rotation(
    data: dict[str, pd.DataFrame],
    bench: pd.DataFrame,
    top_k: int = 3,
    lookback: int = 126,
) -> dict[str, pd.DataFrame]:
    """デュアル・モメンタムのローテーション（月1回入れ替え）。

    月末の終値で、126日の値上がり率の上位 top_k 銘柄を持つ。
    ただし各銘柄が200日線より上で、値上がり率がプラスのものに限る。
    条件を満たさない枠は現金で持つ。
    """
    closes = pd.DataFrame({s: df["close"] for s, df in data.items()})
    mom = closes.pct_change(lookback)
    above = closes > closes.rolling(200).mean()
    eligible = above & (mom > 0)
    ranked = mom.where(eligible).rank(axis=1, ascending=False)
    chosen = ranked <= top_k

    is_rebal = _month_end(closes.index)

    out = {}
    for s in closes:
        entry = chosen[s] & is_rebal
        exit_ = (~chosen[s]) & is_rebal
        out[s] = _frame(entry, exit_, mom[s])
    return out


def _month_end(idx: pd.DatetimeIndex) -> pd.Series:
    p = pd.Series(idx.to_period("M"), index=idx)
    out = p != p.shift(-1)
    out.iloc[-1] = False  # 最終日は月が終わったか分からない
    return out


def trend_switch(
    data: dict[str, pd.DataFrame],
    bench: pd.DataFrame,
    risk_on: str = "SPY",
    risk_off: str | None = None,
    length: int = 200,
    monthly: bool = False,
) -> dict[str, pd.DataFrame]:
    """SPY 中心＋暴落回避。

    risk_on の終値が length 日線より上なら risk_on を持つ。下なら risk_off を持つ
    （None なら現金）。monthly=True なら判定は月末だけにして、売買の往復を減らす。
    """
    c = data[risk_on]["close"]
    on = c > ind.sma(c, length)
    check = _month_end(c.index) if monthly else pd.Series(True, index=c.index)
    on_now = on & check
    off_now = (~on) & check & ind.sma(c, length).notna()
    out = {s: _frame(pd.Series(False, index=df.index), pd.Series(False, index=df.index), pd.Series(0.0, index=df.index))
           for s, df in data.items()}
    out[risk_on] = _frame(on_now, off_now, pd.Series(1.0, index=c.index))
    if risk_off:
        idx = data[risk_off].index
        out[risk_off] = _frame(off_now.reindex(idx).fillna(False), on_now.reindex(idx).fillna(False),
                               pd.Series(0.0, index=idx))
    return out


STRATEGIES = {
    "rsi2_pullback": rsi2_pullback,
    "breakout": breakout,
    "momentum_rotation": momentum_rotation,
    "trend_daily_cash": lambda d, b: trend_switch(d, b, risk_off="BIL"),
    "trend_daily_ief": lambda d, b: trend_switch(d, b, risk_off="IEF"),
    "trend_monthly_cash": lambda d, b: trend_switch(d, b, risk_off="BIL", monthly=True),
    "trend_monthly_ief": lambda d, b: trend_switch(d, b, risk_off="IEF", monthly=True),
}
