"""ポートフォリオ単位の日足バックテスト。

約束事（先読みを防ぐため）:
- シグナルは t 日の終値までのデータで計算する
- 注文は t+1 日の始値で約定する（スリッページを不利な方向に乗せる）
- 損切りは t+1 日以降の安値で判定し、ギャップダウンなら始値で約定したことにする
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Costs:
    commission_rate: float = 0.00132  # moomoo 証券（日本）の米国株手数料 0.132%
    commission_max: float = 22.0  # 1注文あたりの上限（ドル）
    slippage: float = 0.0005  # 片道 0.05%

    def commission(self, notional: float) -> float:
        return min(abs(notional) * self.commission_rate, self.commission_max)


@dataclass
class Config:
    initial_cash: float = 2000.0  # 約30万円
    max_positions: int = 4
    fractional: bool = False  # 端株（小数株）を買えるか
    stop_loss: float | None = None  # 例: 0.08 なら建値から 8% 下で損切り
    max_hold_days: int | None = None
    costs: Costs = field(default_factory=Costs)


@dataclass
class Trade:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    reason: str = ""
    fees: float = 0.0  # 往復の手数料（ドル）

    @property
    def ret(self) -> float:
        """手数料とスリッページを引いた後の損益率。"""
        if not self.exit_price:
            return float("nan")
        cost = self.entry_price * self.shares
        return (self.exit_price * self.shares - cost - self.fees) / cost


@dataclass
class Result:
    equity: pd.Series
    trades: list[Trade]
    exposure: pd.Series

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame([t.__dict__ | {"ret": t.ret} for t in self.trades])


def run(
    data: dict[str, pd.DataFrame],
    signals: dict[str, pd.DataFrame],
    cfg: Config,
    start: str | None = None,
    end: str | None = None,
) -> Result:
    """signals[sym] は index=日付、列 entry(bool) / exit(bool) / score(float, 大きいほど優先)。"""
    syms = list(signals)
    dates = sorted(set().union(*(data[s].index for s in syms)))
    dates = pd.DatetimeIndex(dates)
    if start:
        dates = dates[dates >= pd.Timestamp(start)]
    if end:
        dates = dates[dates <= pd.Timestamp(end)]

    o = pd.DataFrame({s: data[s]["open"] for s in syms}).reindex(dates)
    lo = pd.DataFrame({s: data[s]["low"] for s in syms}).reindex(dates)
    c = pd.DataFrame({s: data[s]["close"] for s in syms}).reindex(dates).ffill()
    ent = pd.DataFrame({s: signals[s]["entry"] for s in syms}).reindex(dates).fillna(False).astype(bool)
    ext = pd.DataFrame({s: signals[s]["exit"] for s in syms}).reindex(dates).fillna(False).astype(bool)
    score = pd.DataFrame({s: signals[s]["score"] for s in syms}).reindex(dates)

    costs = cfg.costs
    cash = cfg.initial_cash
    open_pos: dict[str, Trade] = {}
    held_days: dict[str, int] = {}
    trades: list[Trade] = []
    equity = np.empty(len(dates))
    exposure = np.empty(len(dates))
    pending_exit: set[str] = set()
    pending_entry: list[str] = []

    def close_position(sym: str, price: float, day: pd.Timestamp, reason: str) -> None:
        nonlocal cash
        t = open_pos.pop(sym)
        held_days.pop(sym, None)
        fill = price * (1 - costs.slippage)
        notional = fill * t.shares
        fee = costs.commission(notional)
        cash += notional - fee
        t.fees += fee
        t.exit_date, t.exit_price, t.reason = day, fill, reason
        trades.append(t)

    for i, day in enumerate(dates):
        # 1) 前日の終値で決めた注文を、当日の始値で執行する
        for sym in list(pending_exit):
            if sym in open_pos and not math.isnan(o.at[day, sym]):
                close_position(sym, o.at[day, sym], day, "signal")
        pending_exit.clear()

        if pending_entry:
            mark = cash + sum(t.shares * c.at[dates[i - 1], s] for s, t in open_pos.items())
            slot = mark / cfg.max_positions
            for sym in pending_entry:
                if len(open_pos) >= cfg.max_positions or sym in open_pos:
                    continue
                px = o.at[day, sym]
                if math.isnan(px):
                    continue
                fill = px * (1 + costs.slippage)
                budget = min(slot, cash)
                budget -= costs.commission(budget)
                shares = budget / fill if cfg.fractional else math.floor(budget / fill)
                if shares <= 0:
                    continue
                notional = fill * shares
                fee = costs.commission(notional)
                cash -= notional + fee
                open_pos[sym] = Trade(sym, day, fill, shares, fees=fee)
                held_days[sym] = 0
        pending_entry = []

        # 2) 当日の値動きで損切り・保有日数を判定する
        for sym in list(open_pos):
            t = open_pos[sym]
            held_days[sym] += 1
            if cfg.stop_loss is not None:
                stop = t.entry_price * (1 - cfg.stop_loss)
                if not math.isnan(lo.at[day, sym]) and lo.at[day, sym] <= stop:
                    close_position(sym, min(o.at[day, sym], stop), day, "stop")
                    continue
            if cfg.max_hold_days is not None and held_days[sym] >= cfg.max_hold_days:
                pending_exit.add(sym)

        # 3) 当日の終値で評価し、翌日の注文を決める
        pos_value = sum(t.shares * c.at[day, s] for s, t in open_pos.items())
        equity[i] = cash + pos_value
        exposure[i] = pos_value / equity[i] if equity[i] > 0 else 0.0

        for sym in open_pos:
            if ext.at[day, sym]:
                pending_exit.add(sym)
        free = cfg.max_positions - (len(open_pos) - len(pending_exit))
        if free > 0:
            cands = [s for s in syms if ent.at[day, s] and s not in open_pos]
            cands.sort(key=lambda s: score.at[day, s] if not math.isnan(score.at[day, s]) else -np.inf, reverse=True)
            pending_entry = cands[:free]

    # 期間末に残った建玉は最終日の終値で評価して閉じる（成績計算用）
    last = dates[-1]
    for sym in list(open_pos):
        close_position(sym, c.at[last, sym], last, "end")
    equity[-1] = cash

    return Result(pd.Series(equity, index=dates), trades, pd.Series(exposure, index=dates))


def metrics(equity: pd.Series, trades: list[Trade] | None = None, exposure: pd.Series | None = None) -> dict:
    rets = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else float("nan")
    dd = equity / equity.cummax() - 1
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else float("nan")
    out = {
        "CAGR": cagr,
        "MaxDD": dd.min(),
        "Sharpe": sharpe,
        "Calmar": cagr / abs(dd.min()) if dd.min() < 0 else float("nan"),
        "Final": equity.iloc[-1],
    }
    if trades is not None:
        r = np.array([t.ret for t in trades])
        out |= {
            "Trades": len(trades),
            "WinRate": float((r > 0).mean()) if len(r) else float("nan"),
            "AvgTrade": float(r.mean()) if len(r) else float("nan"),
        }
    if exposure is not None:
        out["Exposure"] = float(exposure.mean())
    return out


def buy_and_hold(df: pd.DataFrame, cfg: Config, start: str | None = None, end: str | None = None) -> pd.Series:
    """ベンチマーク: 初日の始値で買って持ち続ける（同じ手数料・端株ルール）。"""
    d = df.loc[start:end] if (start or end) else df
    fill = d["open"].iloc[0] * (1 + cfg.costs.slippage)
    budget = cfg.initial_cash - cfg.costs.commission(cfg.initial_cash)
    shares = budget / fill if cfg.fractional else math.floor(budget / fill)
    cash = cfg.initial_cash - shares * fill - cfg.costs.commission(shares * fill)
    return cash + shares * d["close"]
