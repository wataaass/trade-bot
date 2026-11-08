"""目標の比率（例: UPRO 2/3 ＋ BIL 1/3）で持つ型のバックテスト。

`backtest.run` は「1銘柄に全額」を前提にしている。こちらは複数の銘柄を比率で持ち、
比率が変わった日だけ、差の分だけを売り買いする（全部を売り替えないので、確定する利益＝税金が減る）。
費用と税金の扱いは `backtest.run` と同じ: 翌日の始値で約定、取得単価は移動平均、
売買益は年ごとに通算して翌年初めに払い、損失は3年繰り越す。配当の税金は株を売って払う。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from tradebot import backtest as bt


def run_weights(
    data: dict[str, pd.DataFrame],
    target: pd.DataFrame,
    cfg: bt.Config,
    start: str | None = None,
    end: str | None = None,
    rebalance_monthly: bool = False,
) -> bt.Result:
    """target は index=日付、列=銘柄、値=その日の終値で決めた目標の比率（行の合計は1、NaN の行は判定なし）。

    目標の行が前の目標と変わった日の翌日に、差の分だけ売買する（売りを先に）。
    rebalance_monthly なら、変わらなくても月末ごとに目標の比率へ戻す（値動きでずれた分を直す）。
    """
    syms = list(target.columns)
    dates = pd.DatetimeIndex(sorted(set().union(*(data[s].index for s in syms))))
    if start:
        dates = dates[dates >= pd.Timestamp(start)]
    if end:
        dates = dates[dates <= pd.Timestamp(end)]
    o = pd.DataFrame({s: data[s]["open"] for s in syms}).reindex(dates).to_numpy()
    c = pd.DataFrame({s: data[s]["close"] for s in syms}).reindex(dates).ffill().to_numpy()
    dv = pd.DataFrame({s: data[s].get("div_yield", pd.Series(0.0, index=data[s].index)) for s in syms})
    dv = dv.reindex(dates).fillna(0.0).to_numpy()
    tw = target.reindex(dates).to_numpy(dtype=float)

    costs, tax = cfg.costs, cfg.tax
    n = len(syms)
    shares = np.zeros(n)
    basis = np.zeros(n)  # 取得価額の合計（手数料込み）
    cash = cfg.initial_cash
    realized: dict[int, float] = {}
    withheld: dict[int, float] = {}
    carry: list[list] = []
    tax_paid = 0.0
    turnover = 0.0
    equity = np.empty(len(dates))
    exposure = np.empty(len(dates))
    current: np.ndarray | None = None  # 今の目標の比率
    pending: np.ndarray | None = None

    def sell(j: int, qty: float, px: float, year: int) -> None:
        nonlocal cash, turnover
        if qty <= 0:
            return
        fill = px * (1 - costs.slippage)
        notional = fill * qty
        fee = costs.commission(notional)
        part = basis[j] * qty / shares[j]
        realized[year] = realized.get(year, 0.0) + notional - fee - part
        withhold(year)
        basis[j] -= part
        shares[j] -= qty
        cash += notional - fee
        turnover += notional

    def buy(j: int, budget: float, px: float) -> None:
        nonlocal cash, turnover
        fill = px * (1 + costs.slippage)
        budget -= costs.commission(budget)
        qty = budget / fill if cfg.fractional else math.floor(budget / fill)
        if qty <= 0:
            return
        notional = fill * qty
        fee = costs.commission(notional)
        cash -= notional + fee
        shares[j] += qty
        basis[j] += notional + fee
        turnover += notional

    def withhold(year: int) -> None:
        """源泉徴収: その年の通算の利益に対する税金と、すでに引いた額との差を、今の現金から引く（戻す）。"""
        nonlocal cash, tax_paid
        if tax is None or not tax.withhold:
            return
        due = max(0.0, realized.get(year, 0.0)) * tax.capital_gains
        delta = due - withheld.get(year, 0.0)
        cash -= delta
        tax_paid += delta
        withheld[year] = due

    def settle(year: int) -> None:
        nonlocal cash, tax_paid
        pnl = realized.pop(year, 0.0)
        if tax is None:
            return
        back = withheld.pop(year, 0.0)
        cash += back
        tax_paid -= back
        carry[:] = [x for x in carry if x[0] >= year - tax.loss_carry_years]
        if pnl < 0:
            carry.append([year, -pnl])
            return
        for x in carry:
            used = min(x[1], pnl)
            x[1] -= used
            pnl -= used
        carry[:] = [x for x in carry if x[1] > 0]
        due = pnl * tax.capital_gains
        cash -= due
        tax_paid += due

    def cover(i_px: int, year: int) -> None:
        """現金がマイナスなら、持ち株を同じ割合で売って埋める。"""
        nonlocal cash
        if cash >= 0:
            return
        value = float(np.nansum(shares * c[i_px]))
        if value <= 0:
            return
        f = min(-cash / value, 1.0)
        for j in range(n):
            if shares[j] > 0:
                proceeds = f * shares[j] * c[i_px, j]
                part = f * basis[j]
                realized[year] = realized.get(year, 0.0) + proceeds - part
                basis[j] -= part
                shares[j] *= 1 - f
        cash += f * value
        withhold(year)

    contrib: dict[pd.Timestamp, float] = {}
    for i, day in enumerate(dates):
        if i > 0 and day.year != dates[i - 1].year:
            settle(dates[i - 1].year)
            cover(i - 1, day.year)

        # 積み立て: 各月の最初の営業日に入金し、今の目標の比率で買い足す（売りはしない）
        deposit = cfg.monthly_contribution if i > 0 and day.month != dates[i - 1].month else 0.0
        if deposit > 0:
            cash += deposit
            contrib[day] = deposit

        held = shares.copy()  # 前日の終値の時点で持っていた株数（配当を受け取る分）
        if pending is not None:
            px = np.where(np.isnan(o[i]), c[i - 1], o[i])
            value = cash + float(np.nansum(shares * px))  # 発注するときの値段で総額を見積もる
            want = pending * value
            have = shares * px
            for j in range(n):  # 売りを先に
                if have[j] > want[j] and shares[j] > 0:
                    qty = shares[j] if pending[j] == 0 else (have[j] - want[j]) / px[j]
                    sell(j, min(qty, shares[j]), px[j], day.year)
            have = shares * px
            # 売りで税金が引かれた（戻った）分も含めて、総額を見積もり直す
            want = pending * (cash + float(np.nansum(have)))
            short = np.clip(want - have, 0, None)
            if short.sum() > 0:
                scale = min(1.0, max(cash, 0.0) / short.sum())
                for j in range(n):
                    if short[j] > 0:
                        buy(j, short[j] * scale, px[j])
            pending = None
        elif deposit > 0 and current is not None and cash > 0:
            px = np.where(np.isnan(o[i]), c[i - 1], o[i])
            budget = cash
            for j in range(n):
                if current[j] > 0 and not np.isnan(px[j]):
                    buy(j, budget * current[j], px[j])

        if tax is not None and i > 0:
            div = float(np.sum(dv[i] * c[i - 1] * np.minimum(held, shares))) * tax.dividend
            if div > 0:
                cash -= div
                tax_paid += div
                cover(i - 1, day.year)

        pos = float(np.nansum(shares * c[i]))
        equity[i] = cash + pos
        exposure[i] = pos / equity[i] if equity[i] > 0 else 0.0

        row = tw[i]
        if not np.isnan(row).all():
            row = np.nan_to_num(row)
            month_end = i + 1 < len(dates) and dates[i + 1].month != day.month
            if current is None or not np.allclose(row, current) or (rebalance_monthly and month_end):
                current = row
                pending = row

    last = len(dates) - 1
    for j in range(n):
        if shares[j] > 0:
            sell(j, shares[j], c[last, j], dates[last].year)
    settle(dates[last].year)
    equity[-1] = cash
    res = bt.Result(pd.Series(equity, index=dates), [], pd.Series(exposure, index=dates), tax_paid,
                    pd.Series(contrib, dtype=float))
    res.turnover = turnover  # type: ignore[attr-defined]
    return res
