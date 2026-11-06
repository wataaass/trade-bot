"""レバレッジ ETF（2倍・3倍）を持ち続ける場合と、200日線で現金に逃げる場合を約100年分で比べる。

reports/leverage.md に書き出す。シグナルはレバレッジをかける前の米国株全体の200日線で判断する。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from tradebot import backtest as bt  # noqa: E402
from tradebot import data as dt  # noqa: E402
from tradebot import longhist as lh  # noqa: E402
from tradebot import strategies as st  # noqa: E402

ERAS = {
    "全期間 1927-2026": ("1927-06-01", None),
    "1927-1949": ("1927-06-01", "1949-12-31"),
    "1950-1972": ("1950-01-01", "1972-12-31"),
    "1973-1999": ("1973-01-01", "1999-12-31"),
    "2000-2026": ("2000-01-01", None),
}


def check_against_real(mkt: pd.DataFrame, cash: pd.DataFrame) -> list[str]:
    """合成したレバレッジの値動きが、実在の ETF（SSO 2倍、UPRO 3倍）と合うか確かめる。"""
    lines = ["## 合成の確かめ（実在の ETF との比較、年率・配当込み）", ""]
    rows = {}
    for sym, lev in (("SSO", 2), ("UPRO", 3)):
        real = dt.load(sym)
        start = real.index[0] + pd.Timedelta(days=30)
        end = min(real.index[-1], mkt.index[-1])
        syn = lh.leveraged(mkt, cash, lev)["close"].loc[start:end]
        rr = real["close"].loc[start:end]
        yrs = (end - start).days / 365.25
        rows[f"{sym}（{lev}倍）{start.year}-"] = {
            "実在": f"{((rr.iloc[-1] / rr.iloc[0]) ** (1 / yrs) - 1) * 100:.1f}%",
            "合成": f"{((syn.iloc[-1] / syn.iloc[0]) ** (1 / yrs) - 1) * 100:.1f}%",
        }
    lines += [pd.DataFrame(rows).T.to_markdown(), ""]
    return lines


def main() -> None:
    mkt, cash = lh.market_and_cash()
    data = {"MKT": mkt, "CASH": cash, "L2": lh.leveraged(mkt, cash, 2), "L3": lh.leveraged(mkt, cash, 3)}
    pct = lambda x: f"{x * 100:.1f}%"  # noqa: E731
    variants = {
        "1倍 持ち続け": ("MKT", None),
        "2倍 持ち続け": ("L2", None),
        "3倍 持ち続け": ("L3", None),
        "2倍＋200日線（毎日）": ("L2", dict(monthly=False)),
        "3倍＋200日線（毎日）": ("L3", dict(monthly=False)),
        "2倍＋200日線（月末）": ("L2", dict(monthly=True)),
        "3倍＋200日線（月末）": ("L3", dict(monthly=True)),
    }
    lines = [
        "# レバレッジ ETF ＋ 暴落回避（約100年分）",
        "",
        "- 2倍・3倍の値動きは、米国株全体（Fama/French）から合成。借入コストは短期金利＋0.5%、信託報酬 0.9%",
        "- 200日線は、レバレッジをかける前の米国株全体で判断。下なら短期国債",
        "- 約定は翌日の終値。手数料 0.132%、特定口座の税金込み、期間の最後に全部売った手取り",
        "",
    ]
    lines += check_against_real(mkt, cash)
    for era, (start, end) in ERAS.items():
        rows = {}
        for name, (sym, kw) in variants.items():
            cfg = bt.Config(initial_cash=100_000, fractional=True, max_positions=1, tax=bt.TAXABLE)
            if kw is None:
                res = bt.run({sym: data[sym]}, bt.hold_signals(sym, data[sym]), cfg, start, end)
            else:
                u = {sym: data[sym], "CASH": cash, "MKT": mkt}
                sig = st.trend_switch(u, mkt, risk_on=sym, risk_off="CASH", signal="MKT", **kw)
                sig.pop("MKT")
                res = bt.run({sym: data[sym], "CASH": cash}, sig, cfg, start, end)
            m = bt.metrics(res.equity)
            rows[name] = {"年率（特定口座）": pct(m["CAGR"]), "最大DD": pct(m["MaxDD"]),
                          "100万円が": f"{res.equity.iloc[-1] / 100_000 * 100:,.0f}万円"}
        lines += [f"## {era}", "", pd.DataFrame(rows).T.to_markdown(), ""]
        print(era, "done", flush=True)
    out = ROOT / "reports" / "leverage.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
