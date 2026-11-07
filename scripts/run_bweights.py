"""B の持ち方の比較（reports/b_weights.md、改良49・50）。

- 旧: 3段階の倍率を SSO・SPYM に乗り換えて持つ
- 比率: 3段階の倍率を UPRO と BIL の比率で持つ（UPRO 1・2/3・1/3）
- 連続: UPRO の比率 = min(1, k ÷ 前月末の値動き)、残り BIL（0.1 未満の変化は無視）

約100年の合成データ、ETF 2003年〜（合成のレバレッジ）、実物の ETF 2010年〜（SPY・SSO・UPRO・BIL、引け間際の約定）。
元本 $2,000、特定口座の税引き後。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from scripts import run_zoo as rz  # noqa: E402
from tradebot import backtest as bt  # noqa: E402
from tradebot import data as dt  # noqa: E402
from tradebot import weights as wt  # noqa: E402
from tradebot import zoo  # noqa: E402

CFG = bt.Config(initial_cash=2000, fractional=True, max_positions=1, tax=bt.TAXABLE)
ERAS = {
    "約100年": [("1927-06-01", "2026-12-31"), ("1927-06-01", "1951-12-31"), ("1952-01-01", "1976-12-31"),
              ("1977-01-01", "2001-12-31"), ("2002-01-01", "2026-12-31")],
    "ETF 2003〜": [("2003-06-01", "2026-12-31"), ("2003-06-01", "2014-12-31"), ("2015-01-01", "2026-12-31")],
    "実物 2010〜": [("2010-06-01", "2026-12-31"), ("2010-06-01", "2017-12-31"), ("2018-01-01", "2026-12-31")],
}


def real_etfs() -> dict:
    out = {}
    for key, sym in {"MKT": "SPY", "L2": "SSO", "L3": "UPRO", "CASH": "BIL"}.items():
        d = dt.load(sym).copy()
        d["open"] = d["close"]  # 引け間際に約定したとみなす
        out[key] = d
    return out


def discrete_target(data: dict, lo: float, hi: float) -> pd.Series:
    syms = ["MKT", "L2", "L3", "CASH"]
    sig = zoo.c_voltarget_monthly_lev(200, lo, hi, 0.01)(data, data["MKT"])
    ent = pd.DataFrame({s: sig[s]["entry"] for s in syms})
    return ent.idxmax(axis=1).where(ent.any(axis=1))


def variants(data: dict) -> dict[str, pd.DataFrame]:
    out = {"持ち続け": pd.DataFrame({"MKT": 1.0}, index=data["MKT"].index)}
    for name, (lo, hi), k in (("B", (0.15, 0.25), 0.15), ("B′", (0.20, 0.35), 0.20)):
        tgt = discrete_target(data, lo, hi)
        out[f"{name} 旧"] = pd.DataFrame({s: (tgt == s).astype(float) for s in ["MKT", "L2", "L3", "CASH"]}
                                        ).where(tgt.notna())
        l3 = tgt.map({"L3": 1.0, "L2": 2 / 3, "MKT": 1 / 3, "CASH": 0.0})
        out[f"{name} 比率"] = pd.DataFrame({"L3": l3, "CASH": 1 - l3})
        w = zoo.voltarget_share(data["MKT"]["close"], k, 0.1)
        out[f"{name} 連続"] = pd.DataFrame({"L3": w, "CASH": 1 - w})
    a = zoo.c_sma_asym(200, "L3", 0.01)(data, data["MKT"])
    on = a["L3"]["entry"].astype(float).where(a["L3"]["entry"] | a["CASH"]["entry"])
    out["A"] = pd.DataFrame({"L3": on, "CASH": 1 - on})
    return out


def main() -> None:
    universes = {"約100年": rz.load_universe("century")[0], "ETF 2003〜": rz.load_universe("etf")[0],
                 "実物 2010〜": real_etfs()}
    lines = ["# B の持ち方の比較（改良49・50）", "",
             "税引き後の年率（元本 $2,000、特定口座）。かっこ内は最大下落。各土俵の1行目が全期間、残りは時代別。", ""]
    for uni, data in universes.items():
        v = variants(data)
        names = list(v)
        lines += [f"## {uni}", "", "| 期間 | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
        for a, b in ERAS[uni]:
            cells = []
            for n in names:
                m = bt.metrics(wt.run_weights(data, v[n], CFG, a, b).equity)
                cells.append(f"{m['CAGR'] * 100:.1f}%（{m['MaxDD'] * 100:.0f}%）")
            lines.append(f"| {a[:4]}〜{b[:4]} | " + " | ".join(cells) + " |")
        lines.append("")
    (ROOT / "reports" / "b_weights.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
