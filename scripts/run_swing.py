"""S&P500 個別株のスイング型を検証し、reports/swing.md に書き出す。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from scripts.run_stocks import load_stocks  # noqa: E402
from tradebot import backtest as bt  # noqa: E402
from tradebot import data as dt  # noqa: E402
from tradebot import strategies as st  # noqa: E402
from tradebot import universe as uv  # noqa: E402

PERIODS = {
    "長期 1997-現在": ("1997-01-01", None),
    "1997-2006": ("1997-01-01", "2006-12-31"),
    "2007-2018": ("2007-01-01", "2018-12-31"),
    "2019-現在": ("2019-01-01", None),
}
VARIANTS = {
    "ブレイクアウト55/20": dict(kind="breakout55", stop=0.10, hold=None),
    "ガチ候補A: 上位100の55/20・3銘柄": dict(kind="breakout55", stop=0.10, hold=None, mom_top=100, max_pos=3),
    "ガチ候補B: 上位100の20/10・3銘柄": dict(kind="breakout20", stop=0.08, hold=None, mom_top=100, max_pos=3),
    "ガチ候補C: 上位50の55/20・2銘柄": dict(kind="breakout55", stop=0.10, hold=None, mom_top=50, max_pos=2),
    "ブレイクアウト20/10": dict(kind="breakout20", stop=0.08, hold=None),
    "押し目買い RSI(2)": dict(kind="pullback", stop=0.10, hold=10),
    "モメンタム上位50の押し目買い": dict(kind="mom_pullback", stop=0.10, hold=10),
}
MAX_POS = 5


def main() -> None:
    data = load_stocks()
    bench = dt.load("SPY")
    idx = pd.DatetimeIndex(sorted(set().union(*(d.index for d in data.values()))))
    members = uv.member_mask(idx, list(data))
    pct = lambda x: f"{x * 100:.1f}%"  # noqa: E731
    sigs = {n: st.stock_swing(data, bench, members, kind=v["kind"], mom_top=v.get("mom_top"))
            for n, v in VARIANTS.items()}
    print("signals done", flush=True)

    lines = [
        "# S&P500 個別株のスイング型の検証",
        "",
        f"- 銘柄: 株価が取れた {len(data)} 銘柄。各日に S&P500 に入っている銘柄だけ買う",
        f"- 同時に持つのは最大 {MAX_POS} 銘柄（ガチ候補は2〜3銘柄）、等金額。SPY が200日線より下の日は新規に買わない",
        "- 手数料 0.132%（上限 $22）、スリッページ 0.05%、翌日始値で約定。小数株あり、元本 $100,000",
        "- 特定口座の税金込み（最終日に全部売った手取り）。上場廃止銘柄の抜けで成績はやや良く出る",
        "",
    ]
    for label, (start, end) in PERIODS.items():
        rows = {}
        cfg = bt.Config(initial_cash=100_000, fractional=True, max_positions=1, tax=bt.TAXABLE)
        res = bt.run({"SPY": bench}, bt.hold_signals("SPY", bench), cfg, start, end)
        m = bt.metrics(res.equity)
        rows["SPY 持ち続け"] = {"年率（特定口座）": pct(m["CAGR"]), "最大DD": pct(m["MaxDD"]),
                             "Sharpe": f"{m['Sharpe']:.2f}"}
        for name, v in VARIANTS.items():
            cfg = bt.Config(initial_cash=100_000, fractional=True, max_positions=v.get("max_pos", MAX_POS),
                            stop_loss=v["stop"], max_hold_days=v["hold"], tax=bt.TAXABLE)
            res = bt.run(data, sigs[name], cfg, start, end)
            m = bt.metrics(res.equity, res.trades, res.exposure)
            rows[name] = {"年率（特定口座）": pct(m["CAGR"]), "最大DD": pct(m["MaxDD"]),
                          "Sharpe": f"{m['Sharpe']:.2f}", "取引数/年": f"{m['Trades'] / max((res.equity.index[-1] - res.equity.index[0]).days / 365.25, 1):.0f}",
                          "勝率": pct(m["WinRate"]), "平均損益/回": pct(m["AvgTrade"])}
        lines += [f"## {label}", "", pd.DataFrame(rows).T.to_markdown(), ""]
        print(label, "done", flush=True)

    out = ROOT / "reports" / "swing.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
