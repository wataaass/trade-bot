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

    idx = closes.index
    month_end = pd.Series(idx.to_period("M"), index=idx)
    is_rebal = month_end != month_end.shift(-1)
    is_rebal.iloc[-1] = False  # 最終日は月が終わったか分からない

    out = {}
    for s in closes:
        entry = chosen[s] & is_rebal
        exit_ = (~chosen[s]) & is_rebal
        out[s] = _frame(entry, exit_, mom[s])
    return out


STRATEGIES = {
    "rsi2_pullback": rsi2_pullback,
    "breakout": breakout,
    "momentum_rotation": momentum_rotation,
}
