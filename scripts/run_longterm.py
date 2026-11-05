"""約100年分のデータで「持ち続ける」と「暴落回避型」を比べ、reports/longterm.md に書き出す。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from tradebot import backtest as bt  # noqa: E402
from tradebot import longhist as lh  # noqa: E402
from tradebot import strategies as st  # noqa: E402

ERAS = {
    "全期間 1927-2026": ("1927-06-01", None),
    "1927-1949 大恐慌・戦争": ("1927-06-01", "1949-12-31"),
    "1950-1972 戦後の成長期": ("1950-01-01", "1972-12-31"),
    "1973-1999 インフレと長期上昇": ("1973-01-01", "1999-12-31"),
    "2000-2026 ITバブル崩壊以降": ("2000-01-01", None),
}


def main() -> None:
    mkt, cash = lh.market_and_cash()
    data = {"MKT": mkt, "CASH": cash}
    strategies = {
        "持ち続け": None,
        "暴落回避・毎日判定": dict(monthly=False),
        "暴落回避・月末判定": dict(monthly=True),
    }
    pct = lambda x: f"{x * 100:.1f}%"  # noqa: E731
    lines = [
        "# 約100年分での比較（米国株全体）",
        "",
        "- 米国株全体: Fama/French の CRSP 時価総額加重（配当込み）。現金: 1か月物の米短期国債",
        f"- データ: {mkt.index[0].date()} 〜 {mkt.index[-1].date()}",
        "- 暴落回避: 米国株全体が200日線より上なら株、下なら短期国債",
        "- 約定はシグナルの翌日の終値。手数料・スリッページは今の水準（昔はもっと高かった）",
        "- 特定口座の税金: 売買益 20.315%（年ごとに通算、損失は3年繰り越し）、配当と利息は約28%",
        "- 期間ごとに元本から始め、期間の最後に全部売った後の手取りで比べる",
        "",
    ]
    rows = {}
    for era, (start, end) in ERAS.items():
        for name, kw in strategies.items():
            for tlabel, tax in (("税なし", None), ("特定口座", bt.TAXABLE)):
                cfg = bt.Config(initial_cash=100_000, fractional=True, max_positions=1, tax=tax)
                if kw is None:
                    res = bt.run({"MKT": mkt}, bt.hold_signals("MKT", mkt), cfg, start, end)
                else:
                    sig = st.trend_switch(data, mkt, risk_on="MKT", risk_off="CASH", **kw)
                    res = bt.run(data, sig, cfg, start, end)
                m = bt.metrics(res.equity, res.trades)
                r = rows.setdefault((era, name), {})
                r[f"年率（{tlabel}）"] = pct(m["CAGR"])
                if tax is None:
                    r["最大DD"] = pct(m["MaxDD"])
                    r["取引数"] = m["Trades"] if kw else 0
        print(era, "done", flush=True)

    for era in ERAS:
        table = pd.DataFrame({name: rows[(era, name)] for name in strategies}).T
        table = table[["年率（税なし）", "年率（特定口座）", "最大DD", "取引数"]]
        lines += [f"## {era}", "", table.to_markdown(), ""]

    out = ROOT / "reports" / "longterm.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
