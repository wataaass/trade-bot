"""参照実装（handoff.daily_plan）を、実物の ETF の過去の値段で毎日動かして、バックテストと比べる。

Muse が移植した bot が、検証と同じ成績になるかの確認用。
- 判定: 毎日、その日までの SPY の終値で daily_plan を呼ぶ
- 約定: 注文は翌営業日の終値で全部約定したことにする（引け間際の成行の近似）、約定のずれ 0.05%、手数料 0.132%（上限 $22）
- 税金: 特定口座。売るたびに源泉徴収（その年の通算の利益に対して）、取得単価は移動平均、年末に精算
- 比較: 同じ条件のバックテスト（weights.run_weights、整数株・源泉徴収あり）

    python scripts/replay_handoff.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from tradebot import backtest as bt  # noqa: E402
from tradebot import data as dt  # noqa: E402
from tradebot import handoff as ho  # noqa: E402
from tradebot import weights as wt  # noqa: E402
from tradebot import zoo  # noqa: E402

COSTS = bt.Costs()
TAX = bt.Tax(withhold=True)


def replay(bot: str, spy: pd.Series, upro: pd.Series, bil: pd.Series, start: str, cash0: float = 2000.0) -> pd.Series:
    days = spy.index[spy.index >= pd.Timestamp(start)]
    hold = {"UPRO": 0, "BIL": 0}
    basis = {"UPRO": 0.0, "BIL": 0.0}  # 1株あたりの取得単価（移動平均、手数料込み）
    cash = cash0
    realized: dict[int, float] = {}
    withheld: dict[int, float] = {}
    carry: list[list] = []
    state = None
    pending: list[dict] = []
    equity = {}
    px = {"UPRO": upro, "BIL": bil}
    for i, d in enumerate(days):
        y = d.year
        if i > 0 and y != days[i - 1].year:  # 年末の精算（源泉徴収した分を戻し、通算・繰り越しして払う）
            py = days[i - 1].year
            pnl = realized.pop(py, 0.0)
            cash += withheld.pop(py, 0.0)
            carry[:] = [c for c in carry if c[0] >= py - TAX.loss_carry_years]
            if pnl < 0:
                carry.append([py, -pnl])
            else:
                for c in carry:
                    used = min(c[1], pnl)
                    c[1] -= used
                    pnl -= used
                cash -= pnl * TAX.capital_gains
        # 前日の注文を、今日の終値で約定させる（売りが先）
        for o in sorted(pending, key=lambda o: o["side"] != "SELL"):
            s, q = o["symbol"], o["qty"]
            p = float(px[s].loc[d])
            if o["side"] == "SELL":
                fill = p * (1 - COSTS.slippage)
                fee = COSTS.commission(fill * q)
                cash += fill * q - fee
                realized[y] = realized.get(y, 0.0) + (fill - basis[s]) * q - fee
                hold[s] -= q
                due = max(0.0, realized[y]) * TAX.capital_gains
                cash -= due - withheld.get(y, 0.0)
                withheld[y] = due
            else:
                fill = p * (1 + COSTS.slippage)
                fee = COSTS.commission(fill * q)
                if fill * q + fee > cash:  # 買付余力が足りなければ、買える分だけ
                    q = int((cash / (1 + COSTS.commission_rate)) // fill)
                    if q <= 0:
                        continue
                    fee = COSTS.commission(fill * q)
                basis[s] = (basis[s] * hold[s] + fill * q + fee) / (hold[s] + q)
                hold[s] += q
                cash -= fill * q + fee
        pending = []
        prices = {s: float(px[s].loc[d]) for s in px}
        equity[d] = cash + sum(hold[s] * prices[s] for s in hold)
        orders, new_state = ho.daily_plan(bot, spy.loc[:d], hold, cash, prices, state,
                                          basis={s: b for s, b in basis.items() if hold[s] > 0})
        state = new_state
        pending = orders
    # 最終日に全部売って精算した手取り（バックテストと同じ比べ方）
    d = days[-1]
    y = d.year
    for s_, q in hold.items():
        if q > 0:
            fill = float(px[s_].loc[d]) * (1 - COSTS.slippage)
            fee = COSTS.commission(fill * q)
            cash += fill * q - fee
            realized[y] = realized.get(y, 0.0) + (fill - basis[s_]) * q - fee
    cash += withheld.pop(y, 0.0)
    pnl = realized.pop(y, 0.0)
    carry[:] = [c for c in carry if c[0] >= y - TAX.loss_carry_years]
    if pnl > 0:
        for c in carry:
            used = min(c[1], pnl)
            c[1] -= used
            pnl -= used
        cash -= pnl * TAX.capital_gains
    equity[d] = cash
    return pd.Series(equity)


def main() -> None:
    spy, upro, bil = (dt.load(s)["close"] for s in ("SPY", "UPRO", "BIL"))
    idx = spy.index.intersection(upro.index).intersection(bil.index)
    spy, upro, bil = spy.loc[idx], upro.loc[idx], bil.loc[idx]
    start = "2010-06-01"
    data = {}
    for key, sym in (("MKT", "SPY"), ("L3", "UPRO"), ("CASH", "BIL")):
        d = dt.load(sym).loc[idx].copy()
        d["open"] = d["close"]
        data[key] = d
    print("| bot | 参照実装を毎日動かした結果 | バックテスト（整数株・源泉徴収） | 差 |")
    print("|---|---|---|---|")
    for bot, k in (("A", None), ("B", ho.K_B), ("B′", ho.K_B_PRIME)):
        eq = replay(bot, spy, upro, bil, start)
        if k is None:
            a = zoo.c_sma_asym(200, "L3", 0.01)(data, data["MKT"])
            w = a["L3"]["entry"].astype(float).where(a["L3"]["entry"] | a["CASH"]["entry"])
        else:
            w = zoo.voltarget_share(data["MKT"]["close"], k, ho.BAND)
        cfg = bt.Config(initial_cash=2000, fractional=False, tax=TAX)
        res = wt.run_weights({"L3": data["L3"], "CASH": data["CASH"]}, pd.DataFrame({"L3": w, "CASH": 1 - w}), cfg, start)
        r1 = bt.metrics(eq)["CAGR"]
        r2 = bt.metrics(res.equity)["CAGR"]
        print(f"| {bot} | {r1 * 100:.2f}% | {r2 * 100:.2f}% | {(r1 - r2) * 100:+.2f} |")


if __name__ == "__main__":
    main()
