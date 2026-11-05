"""1996年以降に S&P500 に入っていた全銘柄の日足を data/stocks/ に取得する。

    python scripts/fetch_sp500.py            # 未取得の銘柄だけ
    python scripts/fetch_sp500.py --refresh  # 全部取り直す
取れなかった銘柄は data/stocks/_missing.txt に書き出す。
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tradebot import data as dt  # noqa: E402
from tradebot import universe as uv  # noqa: E402

OUT = ROOT / "data" / "stocks"


REFRESH = False
START = "1995-01-01"


def fetch(ticker: str) -> tuple[str, str | None]:
    path = OUT / f"{ticker}.csv"
    if path.exists() and not REFRESH:
        return ticker, None
    try:
        df = dt._fetch_yahoo(uv.yahoo_symbol(ticker), START)
        if df.empty:
            return ticker, "empty"
        df.round(6).to_csv(path)
        return ticker, None
    except Exception as e:  # noqa: BLE001
        return ticker, f"{type(e).__name__}: {e}"[:120]
    finally:
        time.sleep(0.3)


def main() -> None:
    global REFRESH
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    REFRESH = ap.parse_args().refresh
    OUT.mkdir(parents=True, exist_ok=True)
    tickers = uv.tickers_since("1996-01-01")
    missing = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, (t, err) in enumerate(ex.map(fetch, tickers), 1):
            if err:
                missing.append(f"{t}\t{err}")
            if i % 100 == 0:
                print(f"{i}/{len(tickers)} 取得済み、失敗 {len(missing)}", flush=True)
    (OUT / "_missing.txt").write_text("\n".join(missing), encoding="utf-8")
    print(f"完了: {len(tickers) - len(missing)}/{len(tickers)} 銘柄")


if __name__ == "__main__":
    main()
