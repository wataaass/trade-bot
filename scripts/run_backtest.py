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
TREND_UNIVERSE = ["SPY", "IEF", "BIL"]  # BIL は超短期国債（現金の代わり）

# 2007〜2018 で型を確かめ、2019〜 は「見ていない期間」として扱う
PERIODS = {
    "検証期間 2007-2018": ("2007-01-01", "2018-12-31"),
    "未使用期間 2019-現在": ("2019-01-01", None),
    "全期間 2007-現在": ("2007-06-01", None),
}

SETUPS = {
    "rsi2_pullback": dict(max_positions=4, stop_loss=0.10, max_hold_days=10),
    "breakout": dict(max_positions=4, stop_loss=0.10),
    "momentum_rotation": dict(max_positions=3),
    "trend_daily_cash": dict(max_positions=1),
    "trend_daily_ief": dict(max_positions=1),
    "trend_monthly_cash": dict(max_positions=1),
    "trend_monthly_ief": dict(max_positions=1),
}


def universe_for(name: str, data: dict) -> dict:
    keys = TREND_UNIVERSE if name.startswith("trend_") else UNIVERSE
    return {k: data[k] for k in keys}



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


TAX_CANDIDATES = ["breakout", "trend_daily_cash", "trend_monthly_cash", "trend_monthly_ief"]


def tax_section(data: dict) -> list[str]:
    """税金込みの比較。最終日に全部売って税金を払ったあとの手取りで比べる。"""
    bench = data[BENCH]
    pct = lambda x: f"{x * 100:.1f}%"  # noqa: E731
    lines = [
        "## 税金込みの比較（最終日に全部売った後の手取り）",
        "",
        "- 特定口座: 売買益 20.315%（年ごとに通算、損失は3年繰り越し）、配当は米国10%＋日本20.315%",
        "- NISA: 売買益は非課税、配当は米国の10%のみ（売ると枠は翌年まで戻らない点は考えていない）",
        "- 為替コストはどの戦略も「最初にドルに替え、最後に円に戻す」1往復で同じなので省いた",
        "",
    ]
    for label in ("全期間 2007-現在", "未使用期間 2019-現在"):
        start, end = PERIODS[label]
        rows = {}
        names = [f"{BENCH} 持ち続け"] + TAX_CANDIDATES
        for name in names:
            row = {}
            for tlabel, tax in (("税なし", None), ("特定口座", bt.TAXABLE), ("NISA", bt.NISA)):
                cfg = bt.Config(initial_cash=100_000, fractional=True, max_positions=1, tax=tax)
                if name in SETUPS:
                    cfg = replace(cfg, **SETUPS[name])
                    u = universe_for(name, data)
                    res = bt.run(u, STRATEGIES[name](u, bench), cfg, start, end)
                else:
                    res = bt.run({BENCH: bench}, bt.hold_signals(BENCH, bench), cfg, start, end)
                m = bt.metrics(res.equity)
                row[f"年率（{tlabel}）"] = pct(m["CAGR"])
                if tlabel == "特定口座":
                    row["最大DD"] = pct(m["MaxDD"])
                    row["払った税金 / 元本"] = pct(res.tax_paid / cfg.initial_cash)
            rows[name] = row
        lines += [f"### {label}", "", pd.DataFrame(rows).T.to_markdown(), ""]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    data = dt.load_many(sorted(set(UNIVERSE) | set(TREND_UNIVERSE)), refresh=args.refresh)
    bench = data[BENCH]
    lines = ["# バックテスト結果", ""]
    lines += [
        f"- 対象: {', '.join(UNIVERSE)}",
        f"- データ最終日: {bench.index[-1].date()}",
        "- 手数料 0.132%（上限 $22）、スリッページ片道 0.05%、翌日始値で約定",
        "- 小数株あり（戦略そのものの力を見るため）。実際の資金 $2,000・整数株の結果は最後の表（SPY は価格 1/10 の安い S&P500 ETF で代用）\n- trend_*: SPY が200日線より上なら SPY、下なら BIL（現金代わり）か IEF（中期国債）。daily は毎日、monthly は月末だけ判定",
        "",
    ]

    for label, (start, end) in PERIODS.items():
        rows = {}
        base = bt.Config(initial_cash=100_000, fractional=True)
        bh = bt.buy_and_hold(bench, base, start, end)
        rows[f"{BENCH} 持ち続け"] = fmt(bt.metrics(bh))
        for name, fn in STRATEGIES.items():
            cfg = replace(base, **SETUPS[name])
            u = universe_for(name, data)
            res = bt.run(u, fn(u, bench), cfg, start, end)
            rows[name] = fmt(bt.metrics(res.equity, res.trades, res.exposure))
        lines += [f"## {label}", "", pd.DataFrame(rows).T.to_markdown(), ""]

    # 実際の資金規模での確認（整数株、$2,000）
    # SPY は1株が高すぎるので、同じ指数に連動する安い ETF を想定して価格を 1/10 にする
    data = dict(data)
    data[BENCH] = data[BENCH].assign(**{k: data[BENCH][k] / 10 for k in ("open", "high", "low", "close")})
    bench = data[BENCH]
    rows = {}
    start, end = PERIODS["未使用期間 2019-現在"]
    small = bt.Config(initial_cash=2000, fractional=False)
    rows[f"{BENCH} 持ち続け"] = fmt(bt.metrics(bt.buy_and_hold(bench, small, start, end)))
    for name, fn in STRATEGIES.items():
        cfg = replace(small, **SETUPS[name])
        u = universe_for(name, data)
        res = bt.run(u, fn(u, bench), cfg, start, end)
        rows[name] = fmt(bt.metrics(res.equity, res.trades, res.exposure))
    lines += ["## 資金 $2,000・整数株（2019-現在）", "", pd.DataFrame(rows).T.to_markdown(), ""]

    lines += tax_section(dt.load_many(sorted(set(UNIVERSE) | set(TREND_UNIVERSE))))

    out = ROOT / "reports" / "backtest.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
