"""戦略の動物園をまとめて評価し、reports/zoo.md と reports/zoo.csv に書き出す。

    python scripts/run_zoo.py                 # 3つの土俵すべて
    python scripts/run_zoo.py century etf     # 土俵を選ぶ

合格の条件（両方を満たす）:
  A. 全期間の税引き後の年率が、同じ土俵の持ち続けを上回る（最終日に全部売った手取りで比較）
  B. 4つの時代のうち3つ以上で、税引き前の年率が持ち続けを上回る
  C. 最大下落が、持ち続けより10ポイント以上深くない
順位: 4つの時代の「持ち続けとの差」のうち一番悪いもの（最悪の時代でどれだけ勝てたか）。
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot import backtest as bt  # noqa: E402
from tradebot import data as dt  # noqa: E402
from tradebot import longhist as lh  # noqa: E402
from tradebot import universe as uv  # noqa: E402
from tradebot import zoo  # noqa: E402

UNIVERSES = {
    "century": dict(
        label="約100年・米国株全体（1927〜2026）", bench="1倍 持ち続け",
        start="1927-06-01", end=None,
        eras={"1927-49": ("1927-06-01", "1949-12-31"), "1950-72": ("1950-01-01", "1972-12-31"),
              "1973-99": ("1973-01-01", "1999-12-31"), "2000-26": ("2000-01-01", None)},
    ),
    "etf": dict(
        label="ETF（2003〜2026）", bench="SPY 持ち続け",
        start="2003-01-01", end=None,
        eras={"2003-08": ("2003-01-01", "2008-12-31"), "2009-14": ("2009-01-01", "2014-12-31"),
              "2015-20": ("2015-01-01", "2020-12-31"), "2021-26": ("2021-01-01", None)},
    ),
    "stocks": dict(
        label="S&P500 個別株（1997〜2026）", bench="SPY 持ち続け",
        start="1997-01-01", end=None,
        eras={"1997-04": ("1997-01-01", "2004-12-31"), "2005-11": ("2005-01-01", "2011-12-31"),
              "2012-18": ("2012-01-01", "2018-12-31"), "2019-26": ("2019-01-01", None)},
    ),
}

DATA: dict = {}
BENCH: pd.DataFrame | None = None
CANDS: list = []


def cash_series(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Fama/French の短期国債を、指定の営業日に合わせた累積指数にする。"""
    _, cash = lh.market_and_cash()
    c = cash["close"].reindex(index.union(cash.index)).ffill().reindex(index)
    rf = c.pct_change().fillna(0)
    return lh._frame(c, rf)


def load_universe(name: str):
    if name == "century":
        mkt, cash = lh.market_and_cash()
        data = {"MKT": mkt, "CASH": cash, "L2": lh.leveraged(mkt, cash, 2), "L3": lh.leveraged(mkt, cash, 3)}
        return data, mkt, zoo.century_candidates()
    if name == "etf":
        syms = ["SPY", "QQQ", "IWM", "EFA", "EEM", "IEF", "TLT", "GLD", *zoo.SECTORS]
        data = {s: dt.load(s) for s in syms}
        cash = cash_series(data["SPY"].index)
        data["CASH"] = cash
        for base, lev, key in (("SPY", 2, "SPY2"), ("SPY", 3, "SPY3"), ("QQQ", 3, "QQQ3")):
            b = data[base]
            data[key] = lh.leveraged(b, cash.reindex(b.index).ffill(), lev)
        # 約100年の土俵の型をそのまま使えるよう、別名を付ける
        data["MKT"], data["L2"], data["L3"] = data["SPY"], data["SPY2"], data["SPY3"]
        cands = zoo.etf_candidates()
        cands += [zoo.Candidate(f"SPY版 {c.name}", c.family.replace("band", "spy-band"), "etf", c.make,
                                symbols=c.symbols, params=c.params, desc=c.desc.replace("株が", "SPY が"))
                  for c in zoo.band_candidates("etf")]
        return data, data["SPY"], cands
    if name == "stocks":
        from scripts.run_stocks import load_stocks
        data = load_stocks()
        bench = dt.load("SPY")
        idx = pd.DatetimeIndex(sorted(set().union(*(d.index for d in data.values()))))
        members = uv.member_mask(idx, list(data))
        cands = [zoo.Candidate("SPY 持ち続け", "hold-spy", "stocks", zoo.c_hold("SPY"), symbols=["SPY"])]
        cands += zoo.stock_candidates(members)
        data["SPY"] = bench
        return data, bench, cands
    raise ValueError(name)


def cagr(eq: pd.Series) -> float:
    eq = eq.dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if yrs > 0.5 and eq.iloc[0] > 0 else float("nan")


def evaluate(i: int) -> dict:
    c = CANDS[i]
    u = UNIVERSES[c.universe]
    syms = c.symbols or [s for s in DATA if s != "SPY" or c.universe != "stocks"]
    data = {s: DATA[s] for s in syms}
    # 個別株の土俵では SPY も渡すが、S&P500 構成銘柄の判定から外れるので売買対象にはならない
    sig = c.make(DATA, BENCH)
    sig = {s: f for s, f in sig.items() if s in data}
    out = {"name": c.name, "family": c.family, "universe": c.universe, "desc": c.desc}
    for tlabel, tax in (("pre", None), ("tax", bt.TAXABLE)):
        cfg = bt.Config(initial_cash=100_000, fractional=True, max_positions=c.max_positions,
                        stop_loss=c.stop_loss, max_hold_days=c.max_hold_days, tax=tax)
        res = bt.run(data, sig, cfg, u["start"], u["end"])
        m = bt.metrics(res.equity, res.trades)
        if tax is None:
            out["cagr_pre"] = m["CAGR"]
            out["maxdd"] = m["MaxDD"]
            out["sharpe"] = m["Sharpe"]
            yrs = (res.equity.index[-1] - res.equity.index[0]).days / 365.25
            out["trades_per_year"] = m["Trades"] / yrs
            for era, (s, e) in u["eras"].items():
                out[f"era_{era}"] = cagr(res.equity.loc[s:e])
        else:
            out["cagr_tax"] = m["CAGR"]
    return out


def init_worker(name: str) -> None:
    global DATA, BENCH, CANDS
    DATA, BENCH, CANDS = load_universe(name)


def run_universe(name: str) -> pd.DataFrame:
    init_worker(name)
    n = len(CANDS)
    print(f"[{name}] 候補 {n} 個を評価中…", flush=True)
    ctx = mp.get_context("fork")
    with ctx.Pool(4) as pool:
        rows = []
        for k, r in enumerate(pool.imap(evaluate, range(n)), 1):
            rows.append(r)
            if k % 10 == 0 or k == n:
                print(f"  {k}/{n}", flush=True)
    return pd.DataFrame(rows)


def apply_gates(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """合格の条件を計算する（保存済みの結果にも同じ条件を当て直せるよう、評価と分けている）。"""
    df = df.copy()
    u = UNIVERSES[name]
    b = df[df["name"] == u["bench"]].iloc[0]
    eras = [f"era_{e}" for e in u["eras"]]
    for e in eras:
        df[f"ex_{e}"] = df[e] - b[e]
    df["ex_tax"] = df["cagr_tax"] - b["cagr_tax"]
    df["eras_won"] = (df[[f"ex_{e}" for e in eras]] > 0).sum(axis=1)
    df["worst_ex"] = df[[f"ex_{e}" for e in eras]].min(axis=1)
    df["dd_ok"] = df["maxdd"] >= b["maxdd"] - 0.10
    df["pass"] = (df["ex_tax"] > 0) & (df["eras_won"] >= 3) & df["dd_ok"]
    return df


def fmt_pct(x: float) -> str:
    return "—" if pd.isna(x) else f"{x * 100:+.1f}" if abs(x) < 10 else f"{x * 100:.0f}"


def report(results: dict[str, pd.DataFrame]) -> None:
    pct = lambda x: "—" if pd.isna(x) else f"{x * 100:.1f}%"  # noqa: E731
    lines = [
        "# 戦略の動物園：一括評価",
        "",
        "合格の条件（両方を満たす）:",
        "",
        "- A. 全期間の**税引き後**の年率が、同じ土俵の持ち続けを上回る（特定口座、最終日に全部売った手取り）",
        "- B. 4つの時代のうち**3つ以上**で、税引き前の年率が持ち続けを上回る",
        "- C. 最大下落が、持ち続けより**10ポイント以上深くない**（ほぼ全損するような型を落とす）",
        "",
        "順位は「最悪の時代での持ち続けとの差」（大きいほど、どの時代でも崩れにくい）。",
        "手数料 0.132%（上限 $22）、スリッページ 0.05%。レバレッジ型は借入コスト（短期金利＋0.5%）と信託報酬 0.9% 込み。",
        "",
    ]
    for name, df in results.items():
        u = UNIVERSES[name]
        eras = list(u["eras"])
        b = df[df["name"] == u["bench"]].iloc[0]
        passed = df[df["pass"]].sort_values("worst_ex", ascending=False)
        lines += [
            f"## {u['label']}",
            "",
            f"候補 {len(df)} 個、合格 {len(passed)} 個（{passed['family'].nunique()} 系統）。"
            f"基準の{u['bench']}: 税引き後 {pct(b['cagr_tax'])}、最大下落 {pct(b['maxdd'])}",
            "",
        ]
        if len(passed):
            t = pd.DataFrame({
                "税引き後": passed["cagr_tax"].map(pct),
                "基準との差": passed["ex_tax"].map(lambda x: f"{x * 100:+.1f}"),
                **{e: passed[f"ex_era_{e}"].map(lambda x: f"{x * 100:+.1f}") for e in eras},
                "最大下落": passed["maxdd"].map(pct),
                "売買/年": passed["trades_per_year"].map(lambda x: f"{x:.0f}"),
            })
            t.index = passed["name"]
            lines += ["時代ごとの列は、持ち続けとの年率の差（ポイント、税引き前）。", "",
                      t.to_markdown(), ""]
        fam = (df.groupby("family").agg(候補=("name", "size"), 合格=("pass", "sum"))
               .sort_values(["合格", "候補"], ascending=False))
        fam = fam[fam["候補"] > 1]
        if len(fam):
            lines += ["系統ごとの合格数（パラメータ違いの何割が合格したか。全部合格なら偶然ではない）:", "",
                      fam.to_markdown(), ""]
    out = ROOT / "reports" / "zoo.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    pd.concat(results.values()).to_csv(ROOT / "reports" / "zoo.csv", index=False)
    print("\n".join(lines))


def main() -> None:
    args = sys.argv[1:]
    names = [] if args == ["--report"] else (args or list(UNIVERSES))  # --report: 保存済みの結果から表だけ作り直す
    results = {}
    csv = ROOT / "reports" / "zoo.csv"
    if csv.exists():  # 前回の結果を残しつつ、指定した土俵だけ更新する
        old = pd.read_csv(csv)
        for u, g in old.groupby("universe"):
            results[u] = g.reset_index(drop=True)
    for n in names:
        results[n] = run_universe(n)
    results = {k: apply_gates(v, k) for k, v in results.items()}
    report({k: results[k] for k in UNIVERSES if k in results})


if __name__ == "__main__":
    main()
