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
REENTRY = 0.01  # 戻るのは200日線の1%上を超えたとき（逃げるのは200日線割れですぐ）


def _sma_on(close: pd.Series, n: int = SMA_N) -> bool:
    close = close.dropna()
    if len(close) < n:
        raise ValueError(f"{n}日分の終値が必要（{len(close)}日分しかない）")
    return bool(close.iloc[-1] > close.iloc[-n:].mean())


def trend_on(close: pd.Series, n: int = SMA_N, up: float = REENTRY) -> bool:
    """200日線の判定（非対称の遊びつき）。

    - 終値が200日移動平均を下回った日 → 「逃げる」状態になる
    - 終値が200日移動平均×(1+up) を上回った日 → 「持つ」状態になる
    - その間にいる日は、前の状態を続ける
    今の状態を、価格の履歴の中で最後にどちらかの条件を満たした日から割り出す。
    """
    c = close.dropna()
    if len(c) < n:
        raise ValueError(f"{n}日分の終値が必要（{len(c)}日分しかない）")
    sma = c.rolling(n).mean()
    on = c > sma * (1 + up)
    off = c < sma
    decided = (on | off)[sma.notna()]
    if not decided.any():
        raise ValueError("200日線の上下どちらとも決まる日が履歴にない。もっと長い履歴を渡す")
    last = decided[decided].index[-1]
    return bool(on.loc[last])


def target_lev3(spy_close: pd.Series) -> str:
    """A. SPY 3倍＋200日線: 持つ状態（trend_on）なら UPRO、逃げる状態なら BIL。"""
    return "UPRO" if trend_on(spy_close) else "BIL"


def target_voltarget(spy_close: pd.Series, lo: float = 0.15, hi: float = 0.25) -> str:
    """B. ボラ連動レバ＋200日線（改良版）:
    1. 200日線の判定（trend_on、非対称の遊びつき）が「逃げる」なら BIL
    2. そうでなければ、倍率を「前の月の最終営業日」の終値で計った値動きで決める:
       直近20営業日の日次リターンの標準偏差×√252 が 15%未満なら UPRO（3倍）、
       25%未満なら SSO（2倍）、それ以上なら SPYM（1倍）
    倍率の選び直しは月1回だけなので、乗り換え（売買と税金）が少ない。
    spy_close の index は日付であること。
    """
    c = spy_close.dropna()
    if not trend_on(c):
        return "BIL"
    last = c.index[-1]
    prev = c.loc[c.index.to_period("M") < last.to_period("M")]
    if len(prev) < VOL_N + 1:
        raise ValueError("前月末までに21営業日以上の終値が必要")
    r = prev.pct_change().iloc[-VOL_N:]
    vol = float(r.std() * np.sqrt(252))
    return "UPRO" if vol < lo else "SSO" if vol < hi else "SPYM"


def target_tqqq(qqq_close: pd.Series) -> str:
    """C. QQQ 3倍＋200日線（実験枠）: QQQ の終値が200日線より上なら TQQQ、そうでなければ BIL。
    （C は遊びなしで検証した型のまま）"""
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


def closes_from_moomoo(kline_list: list[dict]) -> pd.Series:
    """moomoo の履歴K線（quote_history_kline の data.kline_list）を、判断関数に渡せる終値の系列にする。

    各行の `date`（例: 20261106）と `close` を使う。`autype=1`（前方調整）で取得したものを渡すこと。
    日付の重複は後のものを残し、古い順に並べる。
    """
    if not kline_list:
        raise ValueError("K線が空")
    idx = pd.to_datetime([str(r["date"]) for r in kline_list], format="%Y%m%d")
    s = pd.Series([float(r["close"]) for r in kline_list], index=idx, name="close")
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def plan_switch(holdings: dict[str, int], cash: float, target: str, prices: dict[str, float],
                keep_cash: float = 0.01) -> list[dict]:
    """持つべき銘柄（target）に乗り換えるための注文を作る。

    holdings: この枠で今持っている銘柄と株数（例 {"UPRO": 15}）。cash: この枠の現金（ドル）。
    prices: 直近の値段（発注前の確認用。成行なので約定値とは少しずれる）。
    keep_cash: 約定のずれに備えて残す現金の割合（既定 1%）。
    返り値は [{"side": "SELL"|"BUY", "symbol": ..., "qty": ...}] で、売りが先。
    target を既に持っていて他に何も持っていなければ、空のリスト（何もしない）。
    """
    orders: list[dict] = []
    proceeds = 0.0
    for sym, qty in holdings.items():
        if sym != target and qty > 0:
            orders.append({"side": "SELL", "symbol": sym, "qty": int(qty)})
            proceeds += qty * prices[sym]
    if not orders and holdings.get(target, 0) > 0:
        return []
    budget = (cash + proceeds) * (1 - keep_cash)
    qty = int(budget // prices[target])
    if qty > 0:
        orders.append({"side": "BUY", "symbol": target, "qty": qty})
    return orders
