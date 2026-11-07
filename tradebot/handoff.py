"""Muse に引き継ぐ bot の売買判断（参照実装）。

どれも「最後に確定した日の終値まで」の価格だけを使い、その日に持つべき銘柄を返す純粋な関数。
価格は配当・分割調整済みの終値（moomoo の履歴K線なら autype=1 前方調整、Yahoo なら Adj Close）を使う。
検証（tradebot/zoo.py）と同じ判断になることを tests/test_handoff.py で確かめている。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SMA_N = 200
VOL_N = 20


def _sma_on(close: pd.Series, n: int = SMA_N) -> bool:
    close = close.dropna()
    if len(close) < n:
        raise ValueError(f"{n}日分の終値が必要（{len(close)}日分しかない）")
    return bool(close.iloc[-1] > close.iloc[-n:].mean())


def target_lev3(spy_close: pd.Series) -> str:
    """A. SPY 3倍＋200日線: SPY の終値が200日移動平均より上なら UPRO、そうでなければ BIL。"""
    return "UPRO" if _sma_on(spy_close) else "BIL"


def target_voltarget(spy_close: pd.Series, lo: float = 0.15, hi: float = 0.25) -> str:
    """B. ボラ連動レバ＋200日線: SPY が200日線以下なら BIL。上なら、直近20日の値動き（年率）で
    15%未満は UPRO（3倍）、25%未満は SSO（2倍）、それ以上は SPYM（1倍）。"""
    if not _sma_on(spy_close):
        return "BIL"
    r = spy_close.dropna().pct_change().iloc[-VOL_N:]
    vol = float(r.std() * np.sqrt(252))
    return "UPRO" if vol < lo else "SSO" if vol < hi else "SPYM"


def target_tqqq(qqq_close: pd.Series) -> str:
    """C. QQQ 3倍＋200日線（実験枠）: QQQ の終値が200日線より上なら TQQQ、そうでなければ BIL。"""
    return "TQQQ" if _sma_on(qqq_close) else "BIL"


def momentum_6_1(closes: pd.DataFrame, members: list[str], top_n: int = 10) -> list[str]:
    """D. 個別株モメンタム6-1（実験枠）: 月末の終値で、S&P500 構成銘柄のうち
    「1か月前の終値 ÷ 6か月前の終値 − 1」が高い順に top_n 銘柄を返す。

    closes は列が銘柄、行が営業日。最後の行が月末の終値。21営業日＝1か月、126営業日＝6か月。
    """
    c = closes[[m for m in members if m in closes.columns]]
    if len(c) < 127:
        raise ValueError("127営業日分の終値が必要")
    score = c.iloc[-22] / c.iloc[-127] - 1
    score = score[c.iloc[-1].notna() & score.notna()]
    return list(score.sort_values(ascending=False, kind="stable").index[:top_n])
