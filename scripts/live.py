"""ペーパー運用の操作。

    python scripts/live.py plan [--force] [--force-monthly] [--asof YYYY-MM-DD]
        その日の注文の一覧を作り、state/plans/YYYY-MM-DD.json に書く。
        スイングは取引日なら毎日、月次の2戦略は月の最初の取引日だけ計画する。
    python scripts/live.py record --strategy momentum --side BUY --symbol NVDA --qty 2 --price 190.1 [--fee 0.5]
        約定を帳簿に書き込む。
    python scripts/live.py done
        その日の判断（と月初なら月次の入れ替え）が終わったことを帳簿に書く。
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

    print("SPY / SPYM / IEF を取得中…")
    etf = live.load_live_data(["SPY", live.BENCH_TRADE, live.RISK_OFF, live.LEV3, live.LEV3_OFF],
                              start=live.LIVE_START)
    spy = etf["SPY"]
    if spy.index[-1] != now and not args.force:
        print(f"今日（{now.date()}）は米国市場の取引日ではない（最新のデータは {spy.index[-1].date()}）。何もしない")
        return
    last_close = spy.index[spy.index < now][-1]
    month_end = pd.Timestamp(args.asof) if args.asof else live.last_month_end(spy.index, now)

    if ledger is not None:
        live.ensure_swing_book(ledger)
    monthly_due = (ledger is None or ledger.get("last_rebalance") != month) and (
        last_close == month_end or args.force_monthly)
    swing_due = ledger is None or ledger.get("last_swing") != str(now.date()) or args.force
    if not monthly_due and not swing_due:
        print("今日の判断は済んでいる。何もしない")
        return

    held = set()
    if ledger:
        for book in ledger["strategies"].values():
            held |= set(book["positions"])
    members = live.current_members()
    print(f"S&P500 の {len(members)} 銘柄と保有銘柄を取得中…（数分かかる）")
    stocks = live.load_live_data(sorted(set(members) | held - set(etf)),
                                 start=live.LIVE_START)
    data = {**stocks, **etf}
    prices = {s: float(df["close"].loc[:last_close].iloc[-1]) for s, df in data.items()
              if len(df.loc[:last_close])}

    if ledger is None:
        ledger = live.new_ledger(str(now.date()))
        live.ensure_swing_book(ledger)
        print(f"帳簿を新しく作った（各戦略 ${live.INITIAL_CASH:,.0f}）")
    live.save_ledger(ledger)

    orders = []
    plan = {"date": str(now.date()), "price_date": str(last_close.date()), "monthly": monthly_due,
            "swing": swing_due}
    if monthly_due:
        target = live.trend_target(spy, month_end)
        ranking = live.momentum_ranking(stocks, month_end)
        orders += live.plan_trend(ledger["strategies"]["trend"], target, prices)
        orders += live.plan_momentum(ledger["strategies"]["momentum"], ranking, prices)
        plan |= {"signal_date": str(month_end.date()), "trend_target": target, "momentum_top10": ranking[:10]}
        print(f"[月次] シグナルの日 {month_end.date()}、暴落回避の行き先 {target}、モメンタム上位10: "
              + ", ".join(ranking[:10]))
    if swing_due:
        entries, exits = live.swing_signals(stocks, spy, last_close)
        orders += live.plan_swing(ledger["strategies"]["swing"], entries, exits, prices)
        plan |= {"swing_entries": entries[:10], "swing_exits_signal": sorted(exits & set(
            ledger["strategies"]["swing"]["positions"]))}
        print(f"[スイング] {last_close.date()} の終値で 55日高値更新 {len(entries)} 銘柄: " + ", ".join(entries[:10]))
        aggr_entries, aggr_exits = live.swing_signals(stocks, spy, last_close, mom_top=live.AGGR_MOM_TOP)
        orders += live.plan_swing(ledger["strategies"]["swing_aggr"], aggr_entries, aggr_exits, prices,
                                  max_pos=live.AGGR_MAX_POS, strategy="swing_aggr")
        lev_target = live.lev3_target(spy, last_close)
        orders += live.plan_trend(ledger["strategies"]["lev3"], lev_target, prices, strategy="lev3",
                                  on_symbol=live.LEV3)
        plan |= {"swing_aggr_entries": aggr_entries[:10], "lev3_target": lev_target}
        print(f"[ガチスイング] モメンタム上位{live.AGGR_MOM_TOP}の高値更新: " + ", ".join(aggr_entries[:10]))
        print(f"[ガチ増やす] 行き先 {lev_target}")
    orders.sort(key=lambda o: o.side != "SELL")  # 売りを先に
    plan["orders"] = [o.to_dict() for o in orders]

    PLANS.mkdir(parents=True, exist_ok=True)
    path = PLANS / f"{now.date()}.json"
    path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    print(f"計画を {path.relative_to(ROOT)} に書いた（注文 {len(orders)} 件）")
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
    live.ensure_swing_book(ledger)
    now = today()
    path = PLANS / f"{now.date()}.json"
    plan = json.loads(path.read_text()) if path.exists() else {"monthly": False, "swing": True}
    if ledger["benchmark"]["qty"] is None:
        price = live.first_fill_price(ledger, live.BENCH_TRADE)
        if price is None:
            price = float(live.load_live_data([live.BENCH_TRADE])[live.BENCH_TRADE]["close"].iloc[-1])
        live.set_benchmark(ledger, price)
        print(f"比較用の SPYM 持ち続けを ${price:.2f} で {ledger['benchmark']['qty']} 株とした")
    if plan.get("monthly"):
        ledger["last_rebalance"] = now.strftime("%Y-%m")
        print(f"{ledger['last_rebalance']} の月次の入れ替えを完了にした")
    if plan.get("swing"):
        ledger["last_swing"] = str(now.date())
        ledger["swing_started"] = ledger.get("swing_started") or str(now.date())
        for name in live.DAILY_STRATEGIES:
            ledger["started_at"].setdefault(name, str(now.date()))
        print(f"{now.date()} のスイングの判断を完了にした")
    live.save_ledger(ledger)


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
    names = {"momentum": "個別株モメンタム上位5", "trend": "暴落回避（SPYM⇄IEF）",
             "swing": "スイング（55日高値ブレイクアウト）",
             "swing_aggr": "ガチスイング（上位100・3銘柄集中）",
             "lev3": "ガチ増やす（3倍 S&P500 ⇄ 短期国債）"}
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
        f"- 最後の入れ替え: 月次 {ledger['last_rebalance']}、スイング {ledger.get('last_swing')}"
        f"（毎日動く戦略の開始日: {ledger.get('started_at')}）",
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
    p.add_argument("--force", action="store_true", help="取引日でなくても計画を作る（テスト用）")
    p.add_argument("--force-monthly", action="store_true", help="月初でなくても月次の戦略を計画する")
    p.add_argument("--asof", help="月次シグナルの日付を指定する")
    r = sub.add_parser("record")
    r.add_argument("--strategy", required=True, choices=["momentum", "trend", "swing", "swing_aggr", "lev3"])
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
