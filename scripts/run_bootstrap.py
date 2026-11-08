"""約100年の日次データを1年（252営業日）単位のかたまりで並べ替えた架空の相場で、A・B・B′ を試す（reports/bootstrap.md）。

歴史の並び順（たまたまの順番）に頼った結果かどうかを確かめる。かたまりの中のトレンドや値動きの荒さは残る。
米国株全体と短期金利の日次リターンを同じ日のかたまりで取り出し、レバレッジ ETF は longhist.leveraged と同じ式で合成する。
20年の架空の相場を N 本作り、税引き前・手数料込みの年率を比べる。
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot import backtest as bt  # noqa: E402
from tradebot import longhist as lh  # noqa: E402
from tradebot import weights as wt  # noqa: E402
from tradebot import zoo  # noqa: E402

N_PATHS = 400
YEARS = 20
BLOCK = 252
WARMUP = 252  # 200日線が使えるまでの助走（成績には入れない）

R_MKT: np.ndarray = np.empty(0)
R_CASH: np.ndarray = np.empty(0)


def _init() -> None:
    global R_MKT, R_CASH
    mkt, cash = lh.market_and_cash()
    R_MKT = mkt["close"].pct_change().fillna(0).to_numpy()
    R_CASH = cash["close"].pct_change().fillna(0).to_numpy()


def _frame(close: np.ndarray, idx: pd.DatetimeIndex) -> pd.DataFrame:
    c = pd.Series(close, index=idx)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0})


def one(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n_blocks = (YEARS * 252 + WARMUP) // BLOCK + 1
    starts = rng.integers(0, len(R_MKT) - BLOCK, n_blocks)
    rm = np.concatenate([R_MKT[s:s + BLOCK] for s in starts])
    rc = np.concatenate([R_CASH[s:s + BLOCK] for s in starts])
    idx = pd.bdate_range("2000-01-03", periods=len(rm))
    mkt = _frame(100 * np.cumprod(1 + rm), idx)
    cash = _frame(100 * np.cumprod(1 + rc), idx)
    data = {"MKT": mkt, "CASH": cash, "L3": lh.leveraged(mkt, cash, 3)}
    start = str(idx[WARMUP].date())
    cfg = bt.Config(initial_cash=2000, fractional=True)
    out = {"seed": seed}
    m = mkt["close"]
    a = zoo.c_sma_asym(200, "L3", 0.01)(data, mkt)
    on = a["L3"]["entry"].astype(float).where(a["L3"]["entry"] | a["CASH"]["entry"])
    weights = {"持ち続け": pd.DataFrame({"MKT": 1.0}, index=idx), "A": pd.DataFrame({"L3": on, "CASH": 1 - on})}
    for name, k in (("B", 0.15), ("B′", 0.20)):
        w = zoo.voltarget_share(m, k, 0.1)
        weights[name] = pd.DataFrame({"L3": w, "CASH": 1 - w})
    for name, w in weights.items():
        res = wt.run_weights(data, w, cfg, start)
        met = bt.metrics(res.equity)
        out[name] = met["CAGR"]
        out[name + "_dd"] = met["MaxDD"]
    return out


def main() -> None:
    with ProcessPoolExecutor(initializer=_init) as ex:
        rows = list(ex.map(one, range(N_PATHS), chunksize=4))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "reports" / "bootstrap.csv", index=False)
    pct = lambda x: f"{x * 100:.1f}%"  # noqa: E731
    lines = [f"# 並べ替えた架空の相場での成績（{YEARS}年 × {N_PATHS}本）", "",
             f"約100年の日次リターンを {BLOCK} 営業日のかたまりで並べ替えた架空の相場。税引き前・手数料込み、元本 $2,000。", "",
             "| | 年率 悪い10% | 真ん中 | 良い10% | 最大下落 真ん中 | 最大下落 悪い10% | 持ち続けに勝つ割合 | 持ち続けとの差 真ん中 |",
             "|---|---|---|---|---|---|---|---|"]
    for name in ("持ち続け", "A", "B", "B′"):
        r, dd = df[name], df[name + "_dd"]
        d = r - df["持ち続け"]
        win = "—" if name == "持ち続け" else pct((d > 0).mean())
        dm = "—" if name == "持ち続け" else f"{d.median() * 100:+.1f}"
        lines.append(f"| {name} | {pct(r.quantile(0.1))} | {pct(r.median())} | {pct(r.quantile(0.9))} "
                     f"| {pct(dd.median())} | {pct(dd.quantile(0.1))} | {win} | {dm} |")
    (ROOT / "reports" / "bootstrap.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
