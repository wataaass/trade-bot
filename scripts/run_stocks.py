"""S&P500 個別株のモメンタムを検証し、reports/stocks.md に書き出す。

先に python scripts/fetch_sp500.py で株価を取っておく。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from tradebot import backtest as bt  # noqa: E402
from tradebot import data as dt  # noqa: E402
from tradebot import strategies as st  # noqa: E402
from tradebot import universe as uv  # noqa: E402

STOCKS = ROOT / "data" / "stocks"

PERIODS = {
    "検証期間 2007-2018": ("2007-01-01", "2018-12-31"),
    "未使用期間 2019-現在": ("2019-01-01", None),
}

VARIANTS = {
    "12-1 上位5・毎月入れ替え": dict(top_n=5),
    "12-1 上位5・20位まで保持": dict(top_n=5, exit_rank=20),
    "12-1 上位5・20位まで保持・相場フィルター": dict(top_n=5, exit_rank=20, market_filter=True),
    "12-1 上位10・40位まで保持": dict(top_n=10, exit_rank=40),
    "52週高値 上位5・20位まで保持": dict(top_n=5, exit_rank=20, kind="52wh"),
}


def load_stocks() -> dict[str, pd.DataFrame]:
    out = {}
    for path in sorted(STOCKS.glob("*.csv")):
        df = pd.read_csv(path, index_col="date", parse_dates=True)
        if len(df) > 260:
            out[path.stem] = df
    return out


def main() -> None:
    data = load_stocks()
    bench = dt.load("SPY")
    idx = pd.DatetimeIndex(sorted(set().union(*(d.index for d in data.values()))))
    members = uv.member_mask(idx, list(data))
    pct = lambda x: f"{x * 100:.1f}%"  # noqa: E731

    n_all = len(uv.tickers_since("2006-01-01"))
    lines = [
        "# S&P500 個別株モメンタムの検証",
        "",
        f"- 銘柄: 2006年以降に S&P500 に入っていた {n_all} 銘柄のうち、株価が取れた {len(data)} 銘柄",
        "- 各月末に、その時点で S&P500 に入っている銘柄だけから選ぶ",
        "- 手数料 0.132%（上限 $22）、スリッページ 0.05%、翌日始値で約定。小数株あり、元本 $100,000",
        "- 特定口座: 売買益 20.315%（年ごとに通算、損失は3年繰り越し）、配当は約28%。最終日に全部売った手取り",
        "- 注意: 上場廃止になった銘柄の多くは株価が取れず抜けているので、成績はやや良く出る",
        "",
    ]
    signals = {name: st.stock_momentum(data, bench, members, **kw) for name, kw in VARIANTS.items()}

    for label, (start, end) in PERIODS.items():
        rows = {}
        for tlabel, tax in (("税なし", None), ("特定口座", bt.TAXABLE)):
            cfg = bt.Config(initial_cash=100_000, fractional=True, max_positions=1, tax=tax)
            res = bt.run({"SPY": bench}, bt.hold_signals("SPY", bench), cfg, start, end)
            m = bt.metrics(res.equity)
            rows.setdefault("SPY 持ち続け", {})[f"年率（{tlabel}）"] = pct(m["CAGR"])
            if tax is None:
                rows["SPY 持ち続け"] |= {"最大DD": pct(m["MaxDD"]), "Sharpe": f"{m['Sharpe']:.2f}", "取引数": 0}
            for name, kw in VARIANTS.items():
                n = kw["top_n"]
                cfg = bt.Config(initial_cash=100_000, fractional=True, max_positions=n, tax=tax)
                res = bt.run(data, signals[name], cfg, start, end)
                m = bt.metrics(res.equity, res.trades)
                rows.setdefault(name, {})[f"年率（{tlabel}）"] = pct(m["CAGR"])
                if tax is None:
                    rows[name] |= {"最大DD": pct(m["MaxDD"]), "Sharpe": f"{m['Sharpe']:.2f}", "取引数": m["Trades"]}
            print(label, tlabel, "done", flush=True)
        lines += [f"## {label}", "", pd.DataFrame(rows).T.to_markdown(), ""]

    out = ROOT / "reports" / "stocks.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
