"""Kronos で米国の大型株の1週間先を予測し、上位を持つ bot を検証する（reports/kronos.md）。

    pip install -e ".[kronos]"                # torch など（初回だけ）
    python scripts/fetch_sp500.py             # 株価（初回だけ）
    python scripts/run_kronos.py predict      # 予測を reports/kronos/ に貯める（途中から再開できる）
    python scripts/run_kronos.py report       # 予測から成績を計算して reports/kronos.md に書く

予測は毎週の最終営業日の終値で行い、翌営業日の始値で持ち替える。
正式な評価は Kronos の学習データが終わったあと（2024年7月以降）だけ。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tradebot import backtest as bt  # noqa: E402
from tradebot import data as dt  # noqa: E402
from tradebot import kronos_bot as kb  # noqa: E402
from tradebot import universe as uv  # noqa: E402
from tradebot import weights as wt  # noqa: E402

OUT_DIR = ROOT / "reports" / "kronos"
START = "2024-07-01"


def members_on(iv: pd.DataFrame, date: pd.Timestamp) -> list[str]:
    on = (iv["start"] <= date) & (iv["end"].isna() | (iv["end"] > date))
    return sorted(iv.loc[on, "ticker"].unique())


def cache_path(model: str, lookback: int) -> Path:
    return OUT_DIR / f"preds_{model}_lb{lookback}.csv"


def predict(args: argparse.Namespace) -> None:
    import torch

    from tradebot.kronos_model import Kronos, KronosPredictor, KronosTokenizer

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(args.model, args.lookback)
    done = set(pd.read_csv(path, parse_dates=["date"])["date"]) if path.exists() else set()

    iv = uv.intervals()
    data = kb.load_stocks()
    spy = dt.load("SPY", refresh=args.refresh_spy)
    dates = kb.week_ends(spy.index, args.start, args.end)
    todo = [d for d in dates if d not in done]
    print(f"{len(dates)} 週のうち {len(todo)} 週を予測する（{args.model}、過去 {args.lookback} 日、{args.samples} 回の平均）", flush=True)

    name, tok, ctx = kb.MODELS[args.model]
    predictor = KronosPredictor(Kronos.from_pretrained(name), KronosTokenizer.from_pretrained(tok), device="cpu", max_context=ctx)
    for i, date in enumerate(todo, 1):
        t0 = time.time()
        syms = kb.select_universe(data, members_on(iv, date), date, args.n, args.lookback)
        pred = kb.predict_week(predictor, data, syms, date, args.lookback, pred_len=5, sample_count=args.samples, batch=args.batch)
        rows = pd.DataFrame({"date": date.date(), "ticker": pred.index, "pred": pred.to_numpy().round(6)})
        rows.to_csv(path, mode="a", header=not path.exists(), index=False)
        print(f"{i}/{len(todo)} {date.date()} {len(syms)} 銘柄 {time.time() - t0:.0f}s", flush=True)


def wide(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.pivot(index="date", columns="ticker", values="pred").sort_index()


def scores_like(pred: pd.DataFrame, data: dict[str, pd.DataFrame], fn) -> pd.DataFrame:
    """Kronos と同じ日・同じ銘柄で、比べる相手の点数を作る。"""
    rows = {}
    for date, row in pred.iterrows():
        syms = list(row.dropna().index)
        rows[date] = fn(data, syms, date)
    return pd.DataFrame(rows).T.reindex(columns=pred.columns)


def mean_gap(data: dict[str, pd.DataFrame], syms: list[str], date: pd.Timestamp, lookback: int) -> pd.Series:
    """終値が過去 lookback 日の平均より下にあるほど高い点数（平均に戻ると見る単純な型）。"""
    out = {}
    for t in syms:
        c = data[t]["close"].loc[:date].iloc[-lookback:]
        out[t] = float(c.mean() / c.iloc[-1] - 1)
    return pd.Series(out)


def ic_stats(ic: pd.Series) -> dict:
    return {
        "週": len(ic),
        "順位相関の平均": ic.mean(),
        "t値": ic.mean() / ic.std() * np.sqrt(len(ic)) if ic.std() > 0 else float("nan"),
        "プラスの週": (ic > 0).mean(),
    }


def report(args: argparse.Namespace) -> None:
    path = cache_path(args.model, args.lookback)
    pred = wide(path)
    pred = pred[pred.index > kb.TRAIN_END]
    tickers = list(pred.columns)
    data = kb.load_stocks(tickers)
    pred = pred.reindex(columns=[t for t in tickers if t in data])
    real = pd.DataFrame({d: kb.forward_returns(data, list(pred.loc[d].dropna().index), d) for d in pred.index}).T

    signals = {
        "Kronos": pred,
        "平均回帰（過去90日の平均との差）": scores_like(pred, data, lambda d, s, t: mean_gap(d, s, t, args.lookback)),
        "短期の反転（過去5日で下げた順）": -scores_like(pred, data, lambda d, s, t: kb.momentum_scores(d, s, t, 5)),
        "モメンタム（過去12か月、直近1か月を除く）": scores_like(pred, data, lambda d, s, t: kb.momentum_scores(d, s, t, 252, 21)),
    }

    lines = [
        "# Kronos で大型株の1週間先を予測して上位を持つ bot",
        "",
        f"`python scripts/run_kronos.py report` で生成。予測は `{path.relative_to(ROOT)}`。",
        "",
        "## 条件",
        "",
        f"- 期間: {pred.index[0].date()} 〜 {pred.index[-1].date()}（{len(pred)} 週）。Kronos の学習データが終わった 2024年7月以降だけ",
        f"- 対象: 毎週、その時点の S&P500 の銘柄から売買代金の大きい順に {pred.notna().sum(axis=1).max()} 銘柄",
        f"- 予測: Kronos-{args.model}、過去 {args.lookback} 日の日足から5営業日先の終値を予測（10回の平均）",
        "- 売買: 毎週の最終営業日の終値で判定し、翌営業日の始値で持ち替え。同じ比率で持つ",
        "- 費用と税金: 手数料 0.132%（上限 $22）、スリッページ 0.05%、特定口座の税金（年ごとに通算）。最終日に全部売った手取り",
        f"- 元本 ${args.cash:,.0f}、端株あり（moomoo の米国株は1株未満でも買える）",
        "",
        "## 予測は当たっているか（費用を引く前）",
        "",
        "毎週、予測した上昇率と実際の5営業日の上昇率の順位相関を測る。0なら当てずっぽう。",
        "",
        "| 点数 | 週 | 順位相関の平均 | t値 | プラスの週 | 上位5の超過（週平均） | 上位10の超過（週平均） |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, sc in signals.items():
        s = ic_stats(kb.rank_ic(sc, real))
        t5 = kb.top_minus_all(sc, real, 5).mean()
        t10 = kb.top_minus_all(sc, real, 10).mean()
        lines.append(
            f"| {name} | {s['週']} | {s['順位相関の平均']:+.3f} | {s['t値']:+.2f} | {s['プラスの週']:.0%} | {t5:+.2%} | {t10:+.2%} |"
        )

    corr = {name: kb.rank_ic(sc, pred.rank(axis=1)).mean() for name, sc in signals.items() if name != "Kronos"}
    lines += ["", "Kronos の予測と、単純な型の点数の順位相関（週ごとの平均）。1に近いほど Kronos は単純な型と同じことをしている。", ""]
    lines += [f"- {name}: {v:+.2f}" for name, v in corr.items()]

    cfg = bt.Config(initial_cash=args.cash, fractional=True, tax=bt.Tax(withhold=True))
    spy = dt.load("SPY")
    all_data = {**data, "SPY": spy}
    start = pred.index[0] + pd.Timedelta(days=1)
    rows = []

    def add(name: str, target: pd.DataFrame) -> None:
        res = wt.run_weights(all_data, target, cfg, start=str(start.date()))
        m = bt.metrics(res.equity)
        years = (res.equity.index[-1] - res.equity.index[0]).days / 365.25
        rows.append(
            {
                "型": name,
                "税引き後の年率": m["CAGR"],
                "最大下落": m["MaxDD"],
                "手取り": m["Final"],
                "売買額 / 年": res.turnover / args.cash / years,  # type: ignore[attr-defined]
                "税金": res.tax_paid,
            }
        )

    for name, sc in signals.items():
        for k, buf in [(5, 0), (5, 5), (10, 0), (10, 10)]:
            label = f"{name} 上位{k}" + (f"（上位{k + buf}にいる間は持ち続ける）" if buf else "")
            add(label, kb.pick_top(sc, k, buf))
    add("対象の全銘柄を同じ比率で持つ", kb.equal_weight(pred))
    add("SPY を持ち続ける", pd.DataFrame({"SPY": 1.0}, index=pred.index[:1]))

    lines += [
        "",
        "## 手数料と税金を引いた成績",
        "",
        "| 型 | 税引き後の年率 | 最大下落 | 手取り | 売買額 / 年（元本の何倍） | 払った税金 |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['型']} | {r['税引き後の年率']:+.1%} | {r['最大下落']:.1%} | ${r['手取り']:,.0f} | {r['売買額 / 年']:.1f} | ${r['税金']:,.0f} |"
        )
    out = ROOT / "reports" / "kronos.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["predict", "report"])
    ap.add_argument("--model", default="small", choices=list(kb.MODELS))
    ap.add_argument("--lookback", type=int, default=90)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=None)
    ap.add_argument("--cash", type=float, default=2000.0)
    ap.add_argument("--refresh-spy", action="store_true")
    args = ap.parse_args()
    predict(args) if args.cmd == "predict" else report(args)


if __name__ == "__main__":
    main()
