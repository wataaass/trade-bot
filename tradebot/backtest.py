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
class Tax:
    """日本の課税口座（特定口座）を想定した税金。

    - 売買益: 年ごとに損益を通算し、プラスなら 20.315%。マイナスは翌年から3年繰り越す
    - 配当: 米国で 10% 源泉徴収されたあと日本で 20.315%（外国税額控除は考えない）
    税金は翌年最初の営業日に口座の現金から払ったことにする（withhold=True なら、源泉徴収ありの特定口座と同じく売るたびに引く）。
    """

    capital_gains: float = 0.20315
    dividend: float = 1 - (1 - 0.10) * (1 - 0.20315)
    loss_carry_years: int = 3
    # 源泉徴収ありの特定口座: 売るたびに、その年の通算の利益に応じて税金を引く（損が出れば戻す）。
    # 前年までの損失の繰り越しは口座の中では使えないので、年明けの精算（確定申告）で戻す
    withhold: bool = False


TAXABLE = Tax()
NISA = Tax(capital_gains=0.0, dividend=0.10)  # NISA でも米国の源泉 10% は引かれる


@dataclass
class Config:
    initial_cash: float = 2000.0  # 約30万円
    max_positions: int = 4
    fractional: bool = False  # 端株（小数株）を買えるか
    stop_loss: float | None = None  # 例: 0.08 なら建値から 8% 下で損切り
    max_hold_days: int | None = None
    costs: Costs = field(default_factory=Costs)
    tax: Tax | None = None  # None なら税金を考えない
    harvest: bool = False  # 年末の損出し: 最終営業日の終値で含み損の建玉を売って同じ値段で買い直す


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
    tax_paid: float = 0.0

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
    ent_arr = ent.to_numpy()
    score_arr = np.nan_to_num(score.to_numpy(dtype=float), nan=-np.inf)
    sym_arr = np.array(syms, dtype=object)
    last_valid = {s: data[s]["close"].last_valid_index() for s in syms}
    dv = pd.DataFrame(
        {s: data[s]["div_yield"] if "div_yield" in data[s] else pd.Series(0.0, index=data[s].index) for s in syms}
    ).reindex(dates).fillna(0.0)
    tax = cfg.tax
    realized: dict[int, float] = {}
    withheld: dict[int, float] = {}
    carry: list[list] = []  # [年, 残っている損失]
    tax_paid = 0.0

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
        basis = t.entry_price * t.shares + t.fees
        realized[day.year] = realized.get(day.year, 0.0) + notional - fee - basis
        withhold(day.year)
        t.fees += fee
        t.exit_date, t.exit_price, t.reason = day, fill, reason
        trades.append(t)

    def cover_shortfall(day: pd.Timestamp, prices_on: pd.Timestamp) -> None:
        """税金で現金がマイナスになったら、持ち株を同じ割合で売って埋める。

        配当は調整済み価格の中で再投資されているので、配当の税金は株数を減らして払うのが正しい。
        売った分の売買益も、その年の損益に入れる。
        """
        nonlocal cash
        if cash >= 0 or not open_pos:
            return
        value = sum(t.shares * c.at[prices_on, s_] for s_, t in open_pos.items())
        if value <= 0:
            return
        f = min(-cash / value, 1.0)
        for s_, t in open_pos.items():
            proceeds = f * t.shares * c.at[prices_on, s_]
            basis = f * (t.entry_price * t.shares + t.fees)
            realized[day.year] = realized.get(day.year, 0.0) + proceeds - basis
            t.shares *= 1 - f
            t.fees *= 1 - f
        cash += f * value
        withhold(day.year)

    def withhold(year: int) -> None:
        """源泉徴収: その年の通算の利益に対する税金と、すでに引いた額との差を、今の現金から引く（戻す）。"""
        nonlocal cash, tax_paid
        if tax is None or not tax.withhold:
            return
        target = max(0.0, realized.get(year, 0.0)) * tax.capital_gains
        delta = target - withheld.get(year, 0.0)
        cash -= delta
        tax_paid += delta
        withheld[year] = target

    def settle(year: int) -> None:
        """その年の売買損益を通算して税金を払う（源泉徴収した分は、いったん戻してから精算する）。"""
        nonlocal cash, tax_paid
        pnl = realized.pop(year, 0.0)
        if tax is None:
            return
        back = withheld.pop(year, 0.0)
        cash += back
        tax_paid -= back
        carry[:] = [c_ for c_ in carry if c_[0] >= year - tax.loss_carry_years]
        if pnl < 0:
            carry.append([year, -pnl])
            return
        for c_ in carry:
            used = min(c_[1], pnl)
            c_[1] -= used
            pnl -= used
        carry[:] = [c_ for c_ in carry if c_[1] > 0]
        due = pnl * tax.capital_gains
        cash -= due
        tax_paid += due

    for i, day in enumerate(dates):
        if i > 0 and day.year != dates[i - 1].year:
            settle(dates[i - 1].year)
            cover_shortfall(day, dates[i - 1])

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

        # 2) 配当の税金、当日の値動きで損切り・保有日数を判定する
        for sym in list(open_pos):
            t = open_pos[sym]
            if last_valid[sym] is not None and day > last_valid[sym]:
                # 上場廃止・買収などでデータが途切れた: 最後の終値で手放したことにする
                close_position(sym, c.at[day, sym], day, "delisted")
                continue
            if tax is not None and dv.at[day, sym] > 0 and t.entry_date < day:
                due = dv.at[day, sym] * c.at[dates[i - 1], sym] * t.shares * tax.dividend
                cash -= due
                tax_paid += due
                cover_shortfall(day, dates[i - 1])
            held_days[sym] += 1
            if cfg.stop_loss is not None:
                stop = t.entry_price * (1 - cfg.stop_loss)
                if not math.isnan(lo.at[day, sym]) and lo.at[day, sym] <= stop:
                    close_position(sym, min(o.at[day, sym], stop), day, "stop")
                    continue
            if cfg.max_hold_days is not None and held_days[sym] >= cfg.max_hold_days:
                pending_exit.add(sym)

        # 年末の損出し（日本にはウォッシュセール規制がないので、売ってすぐ買い直してよい）
        if cfg.harvest and tax is not None and (i + 1 == len(dates) or dates[i + 1].year != day.year):
            for sym, t in open_pos.items():
                px = c.at[day, sym]
                if math.isnan(px) or px >= t.entry_price:
                    continue
                sell = px * (1 - costs.slippage) * t.shares
                fee_s = costs.commission(sell)
                buy_px = px * (1 + costs.slippage)
                fee_b = costs.commission(buy_px * t.shares)
                realized[day.year] = realized.get(day.year, 0.0) + sell - fee_s - (t.entry_price * t.shares + t.fees)
                withhold(day.year)
                cash -= (buy_px * t.shares + fee_b) - (sell - fee_s)
                t.entry_price, t.fees = buy_px, fee_b

        # 3) 当日の終値で評価し、翌日の注文を決める
        pos_value = sum(t.shares * c.at[day, s] for s, t in open_pos.items())
        equity[i] = cash + pos_value
        exposure[i] = pos_value / equity[i] if equity[i] > 0 else 0.0

        for sym in open_pos:
            if ext.at[day, sym]:
                pending_exit.add(sym)
        free = cfg.max_positions - (len(open_pos) - len(pending_exit))
        if free > 0:
            idx = np.flatnonzero(ent_arr[i])
            if len(idx):
                # スコアの高い順。同点なら銘柄の並び順（安定ソート）
                idx = idx[np.argsort(-score_arr[i, idx], kind="stable")]
                # 買えなかった（整数株に届かない等）ときに次点で埋められるよう、候補は全部渡す
                pending_entry = [s for s in sym_arr[idx] if s not in open_pos]

    # 期間末に残った建玉は最終日の終値で評価して閉じる（成績計算用）
    last = dates[-1]
    for sym in list(open_pos):
        close_position(sym, c.at[last, sym], last, "end")
    settle(last.year)
    equity[-1] = cash

    return Result(pd.Series(equity, index=dates), trades, pd.Series(exposure, index=dates), tax_paid)


def hold_signals(sym: str, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """ずっと持ち続けるシグナル（税金込みのベンチマーク用）。初日の翌日の始値で買う。"""
    t = pd.Series(True, index=df.index)
    return {sym: pd.DataFrame({"entry": t, "exit": ~t, "score": 0.0})}


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
