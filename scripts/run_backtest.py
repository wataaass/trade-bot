"""戦略候補をまとめてバックテストし、reports/backtest.md に書き出す。

使い方:
    python scripts/run_backtest.py            # キャッシュ済みデータを使う
    python scripts/run_backtest.py --refresh  # データを取り直す
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from tradebot import backtest as bt  # noqa: E402
from tradebot import data as dt  # noqa: E402
from tradebot.strategies import STRATEGIES  # noqa: E402

# 生存者バイアスを避けるため、個別株ではなく長い歴史のある ETF で検証する
UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA",  # 米国株の指数
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB",  # セクター
    "EFA", "EEM",  # 米国外の株
    "TLT", "IEF", "GLD",  # 債券・金
]
BENCH = "SPY"

# 2007〜2018 で型を確かめ、2019〜 は「見ていない期間」として扱う
PERIODS = {
    "検証期間 2007-2018": ("2007-01-01", "2018-12-31"),
    "未使用期間 2019-現在": ("2019-01-01", None),
}

SETUPS = {
    "rsi2_pullback": dict(max_positions=4, stop_loss=0.10, max_hold_days=10),
    "breakout": dict(max_positions=4, stop_loss=0.10),
    "momentum_rotation": dict(max_positions=3),
}


def fmt(m: dict) -> dict:
    pct = lambda x: f"{x * 100:.1f}%"  # noqa: E731
    out = {
        "年率リターン": pct(m["CAGR"]),
        "最大DD": pct(m["MaxDD"]),
        "Sharpe": f"{m['Sharpe']:.2f}",
        "Calmar": f"{m['Calmar']:.2f}",
    }
    if "Trades" in m:
        out |= {"取引数": m["Trades"], "勝率": pct(m["WinRate"]), "平均損益/回": pct(m["AvgTrade"])}
    if "Exposure" in m:
        out["投資比率"] = pct(m["Exposure"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    data = dt.load_many(UNIVERSE, refresh=args.refresh)
    bench = data[BENCH]
    lines = ["# バックテスト結果", ""]
    lines += [
        f"- 対象: {', '.join(UNIVERSE)}",
        f"- データ最終日: {bench.index[-1].date()}",
        "- 手数料 0.132%（上限 $22）、スリッページ片道 0.05%、翌日始値で約定",
        "- 小数株あり（戦略そのものの力を見るため）。実際の資金 $2,000・整数株の結果は最後の表",
        "",
    ]

    for label, (start, end) in PERIODS.items():
        rows = {}
        base = bt.Config(initial_cash=100_000, fractional=True)
        bh = bt.buy_and_hold(bench, base, start, end)
        rows[f"{BENCH} 持ち続け"] = fmt(bt.metrics(bh))
        for name, fn in STRATEGIES.items():
            cfg = replace(base, **SETUPS[name])
            res = bt.run(data, fn(data, bench), cfg, start, end)
            rows[name] = fmt(bt.metrics(res.equity, res.trades, res.exposure))
        lines += [f"## {label}", "", pd.DataFrame(rows).T.to_markdown(), ""]

    # 実際の資金規模での確認（整数株、$2,000）
    rows = {}
    start, end = PERIODS["未使用期間 2019-現在"]
    small = bt.Config(initial_cash=2000, fractional=False)
    rows[f"{BENCH} 持ち続け"] = fmt(bt.metrics(bt.buy_and_hold(bench, small, start, end)))
    for name, fn in STRATEGIES.items():
        cfg = replace(small, **SETUPS[name])
        res = bt.run(data, fn(data, bench), cfg, start, end)
        rows[name] = fmt(bt.metrics(res.equity, res.trades, res.exposure))
    lines += ["## 資金 $2,000・整数株（2019-現在）", "", pd.DataFrame(rows).T.to_markdown(), ""]

    out = ROOT / "reports" / "backtest.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
