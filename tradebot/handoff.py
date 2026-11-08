"""Muse に引き継ぐ bot の売買判断（参照実装）。

どれも「最後に確定した日の終値まで」の価格だけを使い、その日に持つべき銘柄を返す純粋な関数。
価格は配当・分割調整済みの終値（moomoo の履歴K線なら autype=1 前方調整、Yahoo なら Adj Close）を使う。
検証（tradebot/zoo.py）と同じ判断になることを tests/test_handoff.py で確かめている。
毎日の手順をまとめたものは daily_plan（判定・注文・次の日に引き継ぐ状態）。
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


# B・B′ の持ち方: SSO・SPYM に乗り換えず、UPRO と BIL の比率で持つ（改良49）。
# 比率は3段階ではなく、値動きに合わせて連続的に決める（改良50）。
# 旧版（3段階）の対応: UPRO→1、SSO→2/3、SPYM→1/3、BIL→0
UPRO_SHARE = {"UPRO": 1.0, "SSO": 2 / 3, "SPYM": 1 / 3, "BIL": 0.0}
K_B = 0.15  # B: UPRO の比率 = 15% ÷ 値動き（上限 100%）
K_B_PRIME = 0.20  # B′（候補）
BAND = 0.1  # 前回の比率からの変化がこれ未満なら、売買しない


def upro_share(spy_close: pd.Series, k: float = K_B, prev: float | None = None, band: float = BAND) -> float:
    """B（連続版）の UPRO の比率（0〜1、残りは BIL）。

    1. 200日線の判定（trend_on）が「逃げる」なら 0
    2. 「持つ」なら min(1, k ÷ 値動き)。値動きは前の月の最終営業日までの直近20営業日の
       日次リターンの標準偏差 × √252（target_voltarget と同じ）。月の途中では変わらない
    3. prev（今の目標の比率）があり、どちらも 0 でなく、差が band 未満なら prev のまま（小さな売買をしない）
    """
    c = spy_close.dropna()
    if not trend_on(c):
        return 0.0
    last = c.index[-1]
    prev_month = c.loc[c.index.to_period("M") < last.to_period("M")]
    if len(prev_month) < VOL_N + 1:
        raise ValueError("前月末までに21営業日以上の終値が必要")
    vol = float(prev_month.pct_change().iloc[-VOL_N:].std() * np.sqrt(252))
    raw = min(1.0, k / vol)
    if prev is not None and prev > 0 and abs(raw - prev) < band:
        return prev
    return raw


def target_weights(spy_close: pd.Series, k: float = K_B, prev: float | None = None,
                   band: float = BAND) -> dict[str, float]:
    """B（連続版）の目標の比率 {"UPRO": w, "BIL": 1 - w}。prev は今の目標の UPRO の比率（初回は None）。

    比率が前回の目標から変わった日だけ plan_rebalance で差の分を売買する。
    """
    w = upro_share(spy_close, k, prev, band)
    return {"UPRO": w, "BIL": 1.0 - w}


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


TAX_RATE = 0.20315  # 特定口座の売買益の税率（源泉徴収あり）


def plan_rebalance(holdings: dict[str, int], cash: float, weights: dict[str, float], prices: dict[str, float],
                   keep_cash: float = 0.01, basis: dict[str, float] | None = None,
                   tax_rate: float = TAX_RATE) -> list[dict]:
    """目標の比率（target_weights）に合わせる注文を作る。差の分だけ売買し、売りが先。

    呼ぶのは、目標の比率が前回合わせたときから変わった日だけ（変わっていなければ呼ばない。
    値動きで比率がずれても直さない。検証はその前提）。
    holdings・cash・prices・keep_cash の意味は plan_switch と同じ。
    basis: 銘柄ごとの1株あたりの取得単価（移動平均）。渡すと、源泉徴収ありの口座で売りの利益から
      引かれる税金（利益 × tax_rate）を見込んで、買いの量をその分減らす（買付余力が足りずに買いが通らないのを防ぐ）。
      その年に損が出ていて実際には引かれない場合は、少し現金が余るだけ（次に目標が変わる日に使われる）。
    """
    total = cash + sum(q * prices[s] for s, q in holdings.items())
    investable = total * (1 - keep_cash)
    orders: list[dict] = []
    after = dict(holdings)
    avail = cash
    for sym, qty in holdings.items():
        if qty <= 0:
            continue
        want = weights.get(sym, 0.0) * investable
        have = qty * prices[sym]
        sell = qty if want <= 0 else min(qty, round((have - want) / prices[sym])) if have > want else 0
        if sell > 0:
            orders.append({"side": "SELL", "symbol": sym, "qty": int(sell)})
            after[sym] = qty - sell
            avail += sell * prices[sym]
            if basis and sym in basis:
                avail -= max(0.0, (prices[sym] - basis[sym]) * sell) * tax_rate
    spend = avail - keep_cash * total
    for sym, w in sorted(weights.items(), key=lambda kv: -kv[1]):
        if w <= 0:
            continue
        need = w * investable - after.get(sym, 0) * prices[sym]
        qty = int(min(need, spend) // prices[sym])
        if qty > 0:
            orders.append({"side": "BUY", "symbol": sym, "qty": qty})
            spend -= qty * prices[sym]
    return orders


def plan_deposit(amount: float, weights: dict[str, float], prices: dict[str, float]) -> list[dict]:
    """入金した額（amount）を、目標の比率で買い足す注文（売りはしない）。1株に届かない分は現金のまま。"""
    orders: list[dict] = []
    left = amount
    for sym, w in sorted(weights.items(), key=lambda kv: -kv[1]):
        if w <= 0:
            continue
        qty = int(min(amount * w, left) // prices[sym])
        if qty > 0:
            orders.append({"side": "BUY", "symbol": sym, "qty": qty})
            left -= qty * prices[sym]
    return orders


def plan_gain_harvest(realized_ytd: float, holdings: dict[str, int], basis: dict[str, float],
                      prices: dict[str, float]) -> list[dict]:
    """年末の益出し（確定申告で損失を繰り越さない人向け、改良61）。

    その年の売買損益（realized_ytd）がマイナスのとき、含み益のある銘柄を、損と相殺できる株数だけ売って
    同じ株数をすぐ買い直す注文を作る（日本にはウォッシュセール規制がないので、買い直してよい）。
    年内に受け渡しが終わる最後の約定日（証券会社が12月に案内する。米国株はたいてい12月下旬）の引け間際に出す。
    税金の年は受渡日で決まるので、年の最終営業日に売ると翌年の扱いになる。損失を確定申告で繰り越すなら、やらなくてよい（効果がない）。
    """
    orders: list[dict] = []
    loss = -realized_ytd
    for sym, qty in sorted(holdings.items(), key=lambda kv: -(prices[kv[0]] - basis.get(kv[0], prices[kv[0]]))):
        if loss <= 0:
            break
        gain = prices[sym] - basis.get(sym, prices[sym])
        # 浮く税金（含み益 × 税率）が往復の費用（手数料 0.132% ＋ 約定のずれ 0.05% を2回 ≒ 0.4%）より小さければやらない
        if qty <= 0 or gain <= 0 or gain * TAX_RATE < 0.004 * prices[sym]:
            continue
        n = min(qty, int(loss // gain))
        if n > 0:
            orders += [{"side": "SELL", "symbol": sym, "qty": n}, {"side": "BUY", "symbol": sym, "qty": n}]
            loss -= n * gain
    return orders


def switch_threshold(close: pd.Series, n: int = SMA_N, up: float = REENTRY) -> tuple[str, float]:
    """次の営業日の終値がいくらなら、200日線の判定（非対称の遊びつき）が切り替わるかを返す。

    返り値: ("exit", 価格) … 今は「持つ」。次の終値がこの価格を下回ると「逃げる」
            ("reentry", 価格) … 今は「逃げる」。次の終値がこの価格を上回ると「持つ」
    次の200日線は (直近 n−1 日の合計 + 次の終値) / n なので、そこから逆算する。
    """
    c = close.dropna()
    s = c.iloc[-(n - 1):].sum()
    if trend_on(c, n, up):
        return "exit", s / (n - 1)
    k = 1 + up
    return "reentry", k * s / (n - k)


def daily_plan(bot: str, spy_close: pd.Series, holdings: dict[str, int], cash: float,
               prices: dict[str, float], state: dict | None = None, keep_cash: float = 0.01,
               deposit: float = 0.0, risk: str = "UPRO", safe: str = "BIL",
               basis: dict[str, float] | None = None) -> tuple[list[dict], dict]:
    """1日分の手順をまとめたもの: 判定 → 目標が前回から変わったときだけ注文 → 次の日に引き継ぐ状態。

    bot: "A"、"B"、"B′"（"B'" でもよい）。spy_close: 確定した終値まで（closes_from_moomoo の返り値など）。
    holdings・cash・prices: この枠の持ち株・現金・直近の値段（plan_switch と同じ）。
    state: 前の日の返り値の state（初日は None）。中身は {"target": {"UPRO": w, "BIL": 1 - w}}。
    deposit: 前回からこの枠に入金した額（積み立て）。目標が変わらない日は、この額だけを目標の比率で買い足す。
      目標が変わる日は、入金分も含めた全体で比率を合わせる。
    risk・safe: 実際に売買する銘柄（例: safe="SGOV"、risk="SPXL"）。holdings と prices もこの名前で渡す。
      state の中身は銘柄を差し替えても {"UPRO": w, "BIL": 1 - w} のまま。
    basis: 1株あたりの取得単価（plan_rebalance と同じ）。源泉徴収ありの口座では渡す。
    返り値: (注文のリスト（売りが先。目標が変わらない日は、入金がなければ空）, 新しい state)。
    注文は翌営業日の引け間際に成行で出す。約定しなかったときは state を更新せず、翌日もう一度呼ぶ。
    """
    prev = (state or {}).get("target")
    if bot == "A":
        w = 1.0 if trend_on(spy_close) else 0.0
        target = {"UPRO": w, "BIL": 1.0 - w}
    elif bot in ("B", "B′", "B'"):
        k = K_B if bot == "B" else K_B_PRIME
        target = target_weights(spy_close, k, None if prev is None else prev["UPRO"])
    else:
        raise ValueError(f"未知の bot: {bot}")
    unchanged = prev is not None and abs(prev["UPRO"] - target["UPRO"]) < 1e-12
    held = {s: q for s, q in holdings.items() if q > 0}
    trade_w = {risk: target["UPRO"], safe: target["BIL"]}
    if unchanged and held:
        # 目標が変わらない日は、入金があればその額を目標の比率で買い足すだけ（ふだんは空）
        orders = plan_deposit(min(deposit, cash), trade_w, prices) if deposit > 0 else []
        return orders, {"target": target}
    return plan_rebalance(holdings, cash, trade_w, prices, keep_cash, basis), {"target": target}
