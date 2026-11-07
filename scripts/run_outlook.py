"""10年運用したときの成績の幅を、約100年分の「どこから始めたか」別に集計する（reports/outlook.md）。

約100年の合成データで A・B・持ち続けを走らせ、すべての営業日を開始日とする10年間の年率（税引き前）を集める。
重なりのある窓なので独立な試行ではないが、「始めた時期の運」でどのくらい結果がぶれるかが分かる。
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
from tradebot import zoo  # noqa: E402

YEARS = 10


def main() -> None:
    data, bench, _ = rz.load_universe("century")
    variants = {
        "持ち続け（1倍）": (zoo.c_hold("MKT"), ["MKT"]),
        "A. 3倍＋200日線（戻り+1%）": (zoo.c_sma_asym(200, "L3", 0.01), ["L3", "CASH"]),
        "B. ボラ連動（改良版・戻り+1%）": (zoo.c_voltarget_monthly_lev(200, 0.15, 0.25, 0.01), ["MKT", "L2", "L3", "CASH"]),
        "2倍＋200日線（参考）": (zoo.c_sma(200, "L2"), ["L2", "CASH"]),
    }
    curves = {}
    for name, (make, syms) in variants.items():
        sig = {s: f for s, f in make(data, bench).items() if s in syms}
        res = bt.run({s: data[s] for s in syms}, sig, bt.Config(initial_cash=1.0, fractional=True, max_positions=1),
                     "1927-06-01")
        curves[name] = res.equity
    eq = pd.DataFrame(curves)
    n = 252 * YEARS
    roll = (eq.shift(-n) / eq) ** (1 / YEARS) - 1
    roll = roll.dropna()
    hold = roll["持ち続け（1倍）"]
    pct = lambda x: f"{x * 100:.1f}%"  # noqa: E731
    rows = {}
    for name in variants:
        r = roll[name]
        rows[name] = {
            "悪い5%": pct(r.quantile(0.05)),
            "悪い25%": pct(r.quantile(0.25)),
            "真ん中": pct(r.median()),
            "良い25%": pct(r.quantile(0.75)),
            "良い5%": pct(r.quantile(0.95)),
            "10年で元本割れ": pct((r < 0).mean()),
            "持ち続けに負ける": "—" if name == "持ち続け（1倍）" else pct((r < hold).mean()),
            "30万円→（真ん中）": f"{30 * (1 + r.median()) ** YEARS:.0f}万円",
            "30万円→（悪い5%）": f"{30 * (1 + r.quantile(0.05)) ** YEARS:.0f}万円",
        }
    lines = [
        f"# {YEARS}年運用したときの成績の幅（約100年分の始め方別）",
        "",
        f"- 1927〜2026年のすべての営業日を開始日にした {YEARS}年間の年率（税引き前、手数料込み）。窓の数 {len(roll):,}",
        "- 窓は重なっているので独立な試行ではない。「始めた時期の運」でどれだけぶれるかの目安",
        "- 税金は売却のたびにかかるので、A・B は実際にはここから年1.5〜3ポイント下がる（持ち続けは売るまで課税されない）",
        "",
        pd.DataFrame(rows).T.to_markdown(),
        "",
    ]
    out = ROOT / "reports" / "outlook.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
