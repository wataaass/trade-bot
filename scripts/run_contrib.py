"""毎月積み立てた場合も、A・B が持ち続けに勝つかを調べる（reports/contrib.md）。

約100年の合成データで、毎年1月に始める20年間の窓ごとに、
「元本 $2,000 を一度に入れるだけ」と「元本 $2,000 ＋毎月 $200（約3万円）を積み立てる」を比べる。
積み立ては各月の最初の営業日に入金し、その時点で持っている銘柄（A なら UPRO か BIL）を買い足す。
特定口座の税金（取得単価は移動平均）と手数料を引いた手取りの、積み立てを含めた年率（内部収益率）で比べる。
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from scripts import run_zoo as rz  # noqa: E402
from tradebot import backtest as bt  # noqa: E402
from tradebot import zoo  # noqa: E402

YEARS = 20
CASH0 = 2_000
MONTHLY = 200

VARIANTS = {
    "持ち続け": (zoo.c_hold("MKT"), ["MKT"]),
    "A": (zoo.c_sma_asym(200, "L3", 0.01), ["L3", "CASH"]),
    "B": (zoo.c_voltarget_monthly_lev(200, 0.15, 0.25, 0.01), ["MKT", "L2", "L3", "CASH"]),
    "B′": (zoo.c_voltarget_monthly_lev(200, 0.20, 0.35, 0.01), ["MKT", "L2", "L3", "CASH"]),
}

DATA: dict = {}
SIGS: dict = {}


def _init() -> None:
    data, bench, _ = rz.load_universe("century")
    DATA.update(data)
    for name, (make, syms) in VARIANTS.items():
        SIGS[name] = {s: f for s, f in make(data, bench).items() if s in syms}


def _one(job: tuple[str, int, float]) -> dict:
    name, y0, monthly = job
    sig = SIGS[name]
    cfg = bt.Config(initial_cash=CASH0, fractional=True, max_positions=1, tax=bt.TAXABLE,
                    monthly_contribution=monthly)
    res = bt.run({s: DATA[s] for s in sig}, sig, cfg, f"{y0}-01-01", f"{y0 + YEARS - 1}-12-31")
    paid = CASH0 + (res.contributions.sum() if res.contributions is not None else 0.0)
    return {"variant": name, "start": y0, "monthly": monthly, "irr": bt.money_weighted_return(res, CASH0),
            "final": float(res.equity.iloc[-1]), "paid": float(paid)}


def main() -> None:
    _init()
    years = range(1928, 2026 - YEARS + 1)
    jobs = [(n, y, m) for n in VARIANTS for y in years for m in (0.0, MONTHLY)]
    with ProcessPoolExecutor(initializer=_init) as ex:
        rows = list(ex.map(_one, jobs, chunksize=8))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "reports" / "contrib.csv", index=False)

    pct = lambda x: f"{x * 100:.1f}%"  # noqa: E731
    lines = [
        f"# 毎月積み立てた場合（約100年、{YEARS}年間の窓 {len(years)}本）",
        "",
        f"元本 ${CASH0:,} だけの場合と、元本＋毎月 ${MONTHLY}（約3万円）を積み立てた場合。"
        "特定口座の税金・手数料を引いた手取りの年率（積み立ては内部収益率）。毎年1月に始める窓。",
        "",
        "| | 積み立て | 年率 真ん中 | 悪い10% | 良い10% | 持ち続けとの差 真ん中 | 差 悪い10% | 持ち続けに勝つ窓 | 手取り 真ん中 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in (0.0, MONTHLY):
        sub = df[df.monthly == m].pivot(index="start", columns="variant", values="irr")
        fin = df[df.monthly == m].pivot(index="start", columns="variant", values="final")
        for name in VARIANTS:
            r = sub[name]
            d = r - sub["持ち続け"]
            win = "—" if name == "持ち続け" else pct((d > 0).mean())
            dm = "—" if name == "持ち続け" else f"{d.median() * 100:+.1f}"
            dl = "—" if name == "持ち続け" else f"{d.quantile(0.1) * 100:+.1f}"
            label = "なし" if m == 0 else f"月 ${MONTHLY:.0f}"
            lines.append(f"| {name} | {label} | {pct(r.median())} | {pct(r.quantile(0.1))} | {pct(r.quantile(0.9))} "
                         f"| {dm} | {dl} | {win} | ${fin[name].median():,.0f} |")
    paid = CASH0 + MONTHLY * (12 * YEARS - 1)
    lines += ["", f"積み立てありの払い込みは合計 ${paid:,}（約{paid * 150 / 10_000:.0f}万円、1ドル150円）。", ""]
    (ROOT / "reports" / "contrib.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
