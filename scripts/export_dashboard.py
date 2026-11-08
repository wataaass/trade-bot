"""検証ダッシュボード用の CSV を書き出す。

    python scripts/export_dashboard.py OUT_DIR

書き出すもの: curves.csv（資産の推移）、summary.csv（主要な成績）、fees.csv（手数料ごと）、
outlook.csv（10年運用の幅）、rates.csv（金利ごとの上乗せ）、countries.csv（国別）。
summary.csv は reports/zoo.csv（先に scripts/run_zoo.py を実行）から読む。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts import run_zoo as rz  # noqa: E402
from tradebot import backtest as bt  # noqa: E402
from tradebot import data as dt  # noqa: E402
from tradebot import longhist as lh  # noqa: E402
from tradebot import weights as wt  # noqa: E402
from tradebot import zoo  # noqa: E402

A = lambda: zoo.c_sma_asym(200, "L3", 0.01)  # noqa: E731
SA = ["L3", "CASH"]
B_NAME, BP_NAME = "B. ボラ連動レバ", "B′. ボラ連動 20%（候補）"


def run_b(data, k, start, end, cash=2000, tax=bt.TAXABLE, **kw):
    """B・B′（UPRO と BIL の比率を値動きに合わせて連続的に、改良49・50）。"""
    w = zoo.voltarget_share(data["MKT"]["close"], k, 0.1)
    cfg = bt.Config(initial_cash=cash, fractional=True, max_positions=1, tax=tax, **kw)
    return wt.run_weights({"L3": data["L3"], "CASH": data["CASH"]}, pd.DataFrame({"L3": w, "CASH": 1 - w}),
                          cfg, start, end)


def run(data, sig, syms, start, end, cash=2000, tax=bt.TAXABLE, **kw):
    sig = {k: f for k, f in sig.items() if k in syms}
    return bt.run({k: data[k] for k in syms}, sig,
                  bt.Config(initial_cash=cash, fractional=True, max_positions=1, tax=tax, **kw), start, end)


def c_make(d, b):
    q = d["QQQ"]["close"]
    sm = q.rolling(200).mean()
    tgt = pd.Series(np.where(q > sm, "QQQ3", "CASH"), index=q.index).where(sm.notna())
    return zoo.targets_to_signals(tgt, pd.Series(True, index=q.index), ["QQQ3", "CASH"])


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    D, b, _ = rz.load_universe("etf")
    U = rz.UNIVERSES["etf"]
    strats = [("SPY 持ち続け", zoo.c_hold("SPY")(D, b), ["SPY"]),
              ("A. SPY 3倍＋200日線", A()(D, b), SA),
              (B_NAME, 0.15, None),
              ("Muse 月末型 SMA250", zoo.e_trend("SPY", "IEF", 250, monthly=True)(D, b), ["SPY", "IEF"]),
              ("C. QQQ 3倍（実験）", c_make(D, b), ["QQQ3", "CASH"]),
              (BP_NAME, 0.20, None)]
    rows = []
    for name, sig, syms in strats:
        kw = dict(cash=300_000, costs=bt.Costs(commission_max=22 * 150))
        res = run_b(D, sig, U["start"], U["end"], **kw) if syms is None else run(D, sig, syms, U["start"], U["end"], **kw)
        eq = res.equity.resample("ME").last()
        eq.iloc[-1] = res.equity.iloc[-1]
        rows += [{"month": d.strftime("%Y-%m"), "strategy": name, "value": round(v / 10000, 2)} for d, v in eq.items()]
    pd.DataFrame(rows).to_csv(out / "curves.csv", index=False)

    z = pd.read_csv(ROOT / "reports" / "zoo.csv")
    g = lambda u, n, c: z[(z.universe == u) & (z.name == n)][c].iloc[0]  # noqa: E731
    rows = []
    for label, u, n in (("SPY 持ち続け", "etf", "SPY 持ち続け"),
                        ("A. SPY 3倍＋200日線", "etf", "SPY版 3倍 200日線・戻り+1%"),
                        ("C. QQQ 3倍（実験）", "etf", "QQQ 3倍⇄CASH 200日線・毎日"),
                        ("持ち続け", "century", "1倍 持ち続け"),
                        ("A. SPY 3倍＋200日線", "century", "3倍 200日線・戻り+1%"),
                        ("SPY 持ち続け", "stocks", "SPY 持ち続け"),
                        ("D. 個別株モメンタム（実験）", "stocks", "モメンタム6-1 上位10")):
        data = {"etf": "ETF 2003〜2026", "century": "米国株全体 1927〜2026", "stocks": "S&P500 個別株 1997〜2026"}[u]
        rows.append({"strategy": label, "data": data, "cagr": round(g(u, n, "cagr_tax"), 4),
                     "maxdd": round(g(u, n, "maxdd"), 4), "trades": round(g(u, n, "trades_per_year"), 1)})
    res = run(D, strats[3][1], ["SPY", "IEF"], U["start"], U["end"])
    m = bt.metrics(res.equity, res.trades)
    yrs = (res.equity.index[-1] - res.equity.index[0]).days / 365.25
    rows.insert(3, {"strategy": "Muse 月末型 SMA250", "data": "ETF 2003〜2026", "cagr": round(m["CAGR"], 4),
                    "maxdd": round(m["MaxDD"], 4), "trades": round(m["Trades"] / yrs, 1)})
    C, bc, _ = rz.load_universe("century")
    for data, label, start, end in ((D, "ETF 2003〜2026", U["start"], U["end"]),
                                    (C, "米国株全体 1927〜2026", "1927-06-01", None)):
        for name, k in ((B_NAME, 0.15), (BP_NAME, 0.20)):
            res = run_b(data, k, start, end)
            m = bt.metrics(res.equity)
            yrs = (res.equity.index[-1] - res.equity.index[0]).days / 365.25
            w = zoo.voltarget_share(data["MKT"]["close"], k, 0.1).loc[res.equity.index]
            rows.append({"strategy": name, "data": label, "cagr": round(m["CAGR"], 4), "maxdd": round(m["MaxDD"], 4),
                         "trades": round(float((w.diff().abs() > 1e-12).sum()) / yrs, 1)})
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)

    rows = []
    for name, make, syms in (("SPY 持ち続け", zoo.c_hold("MKT"), ["MKT"]), ("A. SPY 3倍＋200日線", A(), SA),
                             (B_NAME, 0.15, None)):
        sig = make if syms is None else make(D, b)
        for fee in (0.0, 0.0005, 0.001, 0.00132, 0.0015, 0.002):
            costs = bt.Costs(commission_rate=fee)
            res = (run_b(D, sig, U["start"], U["end"], costs=costs) if syms is None
                   else run(D, sig, syms, U["start"], U["end"], costs=costs))
            rows.append({"fee": fee, "strategy": name, "cagr": round(bt.metrics(res.equity)["CAGR"], 4)})
    pd.DataFrame(rows).to_csv(out / "fees.csv", index=False)

    curves = {}
    for name, make, syms in (("持ち続け", zoo.c_hold("MKT"), ["MKT"]), ("A. SPY 3倍＋200日線", A(), SA)):
        curves[name] = run(C, make(C, bc), syms, "1927-06-01", None, cash=1.0, tax=None).equity
    for name, k in ((B_NAME, 0.15), (BP_NAME, 0.20)):
        curves[name] = run_b(C, k, "1927-06-01", None, cash=1.0, tax=None).equity
    eq = pd.DataFrame(curves)
    n = 252 * 10
    roll = ((eq.shift(-n) / eq) ** 0.1 - 1).dropna()
    hold = roll["持ち続け"]
    rows = []
    for name in curves:
        r = roll[name]
        rows.append({"strategy": name, **{f"p{q}": round(r.quantile(q / 100), 4) for q in (5, 25, 50, 75, 95)},
                     "loss_prob": round((r < 0).mean(), 4),
                     "lose_to_hold": None if name == "持ち続け" else round((r < hold).mean(), 4), "windows": len(r)})
    pd.DataFrame(rows).to_csv(out / "outlook.csv", index=False)

    yr = eq.resample("YE").last().pct_change().dropna()
    rf = C["CASH"]["close"].resample("YE").last().pct_change().reindex(yr.index)
    labels = ["1%未満", "1〜3%", "3〜5%", "5%以上"]
    bucket = pd.cut(rf, [-1, 0.01, 0.03, 0.05, 1], labels=labels)
    rows = []
    for bk in labels:
        sel = bucket == bk
        rows.append({"rate": bk, "years": int(sel.sum()), "hold": round(yr.loc[sel, "持ち続け"].mean(), 4),
                     "a_excess": round((yr.loc[sel, "A. SPY 3倍＋200日線"] - yr.loc[sel, "持ち続け"]).mean(), 4),
                     "b_excess": round((yr.loc[sel, B_NAME] - yr.loc[sel, "持ち続け"]).mean(), 4)})
    pd.DataFrame(rows).to_csv(out / "rates.csv", index=False)

    rows = []
    for sym, label in (("SPY", "米国"), ("EWG", "ドイツ"), ("EWU", "英国"), ("EWC", "カナダ"), ("EWA", "豪州"),
                       ("EWJ", "日本"), ("EWH", "香港"), ("EWL", "スイス"), ("EWQ", "フランス"), ("EWN", "オランダ"),
                       ("EWD", "スウェーデン")):
        x = dt.load(sym).loc["1996-03-18":]
        c = rz.cash_series(x.index)
        DD = {"MKT": x, "CASH": c, "L2": lh.leveraged(x, c, 2), "L3": lh.leveraged(x, c, 3)}
        r = {"country": label}
        for key, make, syms in (("hold", zoo.c_hold("MKT"), ["MKT"]), ("a", A(), SA)):
            r[key] = round(bt.metrics(run(DD, make(DD, x), syms, "1997-06-01", None).equity)["CAGR"], 4)
        r["b"] = round(bt.metrics(run_b(DD, 0.15, "1997-06-01", None).equity)["CAGR"], 4)
        rows.append(r)
    pd.DataFrame(rows).to_csv(out / "countries.csv", index=False)
    # 約定のタイミング（実物の UPRO・SSO・BIL・SPY、2010〜2026）
    real = {"MKT": dt.load("SPY"), "L2": dt.load("SSO"), "L3": dt.load("UPRO"), "CASH": dt.load("BIL")}

    def next_close(df):
        d = df.copy()
        d["open"] = df["close"]
        d["low"] = df[["low", "close"]].min(axis=1)
        return d

    def same_close(df):
        d = df.copy()
        d["open"] = df["close"].shift(1)
        d["low"] = d[["low", "open"]].min(axis=1)
        return d.dropna(subset=["open"])

    rows = []
    for name, make, syms in (("持ち続け", zoo.c_hold("MKT"), ["MKT"]), ("A", A(), SA), ("B", 0.15, None)):
        for label, tf in (("当日の終値", same_close), ("翌日の始値", lambda d: d), ("翌日の終値", next_close)):
            dd = {k: tf(v) for k, v in real.items()}
            res = (run_b(dd, make, "2010-01-01", None) if syms is None
                   else run({k: dd[k] for k in syms}, make(real, None), syms, "2010-01-01", None))
            rows.append({"strategy": name, "timing": label, "cagr": round(bt.metrics(res.equity)["CAGR"], 4)})
    pd.DataFrame(rows).to_csv(out / "timing.csv", index=False)
    print("書き出した:", ", ".join(sorted(p.name for p in out.glob("*.csv"))))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "reports" / "dashboard")
