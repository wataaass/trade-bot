"""ペーパー運用の操作。

    python scripts/live.py plan [--force] [--asof YYYY-MM-DD]
        前月末のシグナルで注文の一覧を作り、state/plans/YYYY-MM.json に書く。
        月の最初の取引日でなければ、--force がない限り何もしない。
    python scripts/live.py record --strategy momentum --side BUY --symbol NVDA --qty 2 --price 190.1 [--fee 0.5]
        約定を帳簿に書き込む。
    python scripts/live.py done
        今月の入れ替えが終わったことを帳簿に書く。
    python scripts/live.py status
        各戦略の評価額を SPYM 持ち続けと比べ、reports/live.md に書く。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from tradebot import live  # noqa: E402

PLANS = live.STATE / "plans"


def today() -> pd.Timestamp:
    return pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()


def cmd_plan(args: argparse.Namespace) -> None:
    now = today()
    ledger = live.load_ledger()
    month = now.strftime("%Y-%m")
    if ledger and ledger.get("last_rebalance") == month and not args.force:
        print(f"{month} の入れ替えは済んでいる。何もしない")
        return

    print("SPY / SPYM / IEF を取得中…")
    etf = live.load_live_data(["SPY", live.BENCH_TRADE, live.RISK_OFF])
    spy = etf["SPY"]
    if spy.index[-1] != now and not args.force:
        print(f"今日（{now.date()}）は米国市場の取引日ではない（最新のデータは {spy.index[-1].date()}）。何もしない")
        return
    last_close = spy.index[spy.index < now][-1]
    asof = pd.Timestamp(args.asof) if args.asof else live.last_month_end(spy.index, now)
    if last_close != asof and not args.force:
        print(f"今日（{now.date()}）は月の最初の取引日ではない（前日 {last_close.date()} ≠ 月末 {asof.date()}）。"
              "何もしない")
        return

    held = set()
    if ledger:
        for book in ledger["strategies"].values():
            held |= set(book["positions"])
    members = live.current_members()
    print(f"S&P500 の {len(members)} 銘柄と保有銘柄を取得中…（数分かかる）")
    stocks = live.load_live_data(sorted(set(members) | held - {live.BENCH_TRADE, live.RISK_OFF}))
    data = {**stocks, **etf}
    prices = {s: float(df["close"].loc[:last_close].iloc[-1]) for s, df in data.items()}

    if ledger is None:
        ledger = live.new_ledger(str(now.date()))
        live.save_ledger(ledger)
        print(f"帳簿を新しく作った（各戦略 ${live.INITIAL_CASH:,.0f}）")

    target = live.trend_target(spy, asof)
    ranking = live.momentum_ranking(stocks, asof)
    orders = live.plan_trend(ledger["strategies"]["trend"], target, prices)
    orders += live.plan_momentum(ledger["strategies"]["momentum"], ranking, prices)
    orders.sort(key=lambda o: o.side != "SELL")  # 売りを先に

    plan = {
        "month": month,
        "created": str(now.date()),
        "signal_date": str(asof.date()),
        "price_date": str(last_close.date()),
        "trend_target": target,
        "momentum_top10": ranking[:10],
        "orders": [o.to_dict() for o in orders],
    }
    PLANS.mkdir(parents=True, exist_ok=True)
    path = PLANS / f"{month}.json"
    path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    print(f"計画を {path.relative_to(ROOT)} に書いた")
    print(f"シグナルの日: {asof.date()}、暴落回避の行き先: {target}")
    print("モメンタム上位10:", ", ".join(ranking[:10]))
    for o in orders:
        print(f"  {o.strategy:9s} {o.side:4s} {o.symbol:6s} x{o.qty:<4d} (前日終値 {o.ref_price:.2f}) {o.reason}")


def cmd_record(args: argparse.Namespace) -> None:
    ledger = live.load_ledger()
    if ledger is None:
        sys.exit("帳簿がない。先に plan を実行する")
    live.apply_fill(ledger, args.strategy, args.side, args.symbol, args.qty, args.price, args.fee,
                    args.date or str(today().date()))
    live.save_ledger(ledger)
    book = ledger["strategies"][args.strategy]
    print(f"記録した。{args.strategy} の現金 ${book['cash']:.2f}、建玉 {book['positions']}")


def cmd_done(args: argparse.Namespace) -> None:
    ledger = live.load_ledger()
    if ledger["benchmark"]["qty"] is None:
        price = live.first_fill_price(ledger, live.BENCH_TRADE)
        if price is None:
            price = float(live.load_live_data([live.BENCH_TRADE])[live.BENCH_TRADE]["close"].iloc[-1])
        live.set_benchmark(ledger, price)
        print(f"比較用の SPYM 持ち続けを ${price:.2f} で {ledger['benchmark']['qty']} 株とした")
    ledger["last_rebalance"] = today().strftime("%Y-%m")
    live.save_ledger(ledger)
    print(f"{ledger['last_rebalance']} の入れ替えを完了にした")


def cmd_status(args: argparse.Namespace) -> None:
    ledger = live.load_ledger()
    if ledger is None:
        sys.exit("帳簿がない")
    syms = {ledger["benchmark"]["symbol"]}
    for book in ledger["strategies"].values():
        syms |= set(book["positions"])
    data = live.load_live_data(sorted(syms))
    prices = {s: float(df["close"].iloc[-1]) for s, df in data.items()}
    asof = max(df.index[-1] for df in data.values()).date()

    b = ledger["benchmark"]
    rows = {"SPYM 持ち続け（比較用）": b["cash"] + b["qty"] * prices[b["symbol"]]}
    names = {"momentum": "個別株モメンタム上位5", "trend": "暴落回避（SPYM⇄IEF）"}
    detail = []
    for key, book in ledger["strategies"].items():
        rows[names[key]] = live._value(book, prices)
        pos = ", ".join(f"{s} {p['qty']}株" for s, p in book["positions"].items()) or "なし"
        detail.append(f"- {names[key]}: 現金 ${book['cash']:.2f}、建玉 {pos}")
    table = pd.DataFrame({
        "評価額": {k: f"${v:,.2f}" for k, v in rows.items()},
        "損益": {k: f"{(v / live.INITIAL_CASH - 1) * 100:+.1f}%" for k, v in rows.items()},
    })
    lines = [
        "# ペーパー運用の成績",
        "",
        f"- 開始: {ledger['started']}、元手は各 ${live.INITIAL_CASH:,.0f}",
        f"- 評価日: {asof}（終値）。税金は引いていない",
        f"- 最後の入れ替え: {ledger['last_rebalance']}",
        "",
        table.to_markdown(),
        "",
        *detail,
        "",
    ]
    out = ROOT / "reports" / "live.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--force", action="store_true")
    p.add_argument("--asof")
    r = sub.add_parser("record")
    r.add_argument("--strategy", required=True, choices=["momentum", "trend"])
    r.add_argument("--side", required=True, choices=["BUY", "SELL"])
    r.add_argument("--symbol", required=True)
    r.add_argument("--qty", required=True, type=int)
    r.add_argument("--price", required=True, type=float)
    r.add_argument("--fee", type=float, default=0.0)
    r.add_argument("--date")
    sub.add_parser("done")
    sub.add_parser("status")
    args = ap.parse_args()
    {"plan": cmd_plan, "record": cmd_record, "done": cmd_done, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    main()
