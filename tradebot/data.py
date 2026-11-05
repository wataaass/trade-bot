"""日足データの取得とキャッシュ。

バックテスト用の過去データは Yahoo Finance から取る（moomoo の OpenAPI は OpenD を
起動していないと使えないため）。株式分割と配当を反映した調整済み OHLC を返す。
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
_HEADERS = {"User-Agent": "Mozilla/5.0 (trade-bot backtester)"}


def _fetch_yahoo(symbol: str, start: str) -> pd.DataFrame:
    period1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    period2 = int(time.time())
    params = {
        "period1": period1,
        "period2": period2,
        "interval": "1d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }
    for attempt in range(4):
        r = requests.get(_URL.format(symbol=symbol), params=params, headers=_HEADERS, timeout=30)
        if r.status_code == 429:
            time.sleep(2 ** (attempt + 1))
            continue
        r.raise_for_status()
        break
    else:
        raise RuntimeError(f"{symbol}: Yahoo Finance のレート制限で取得できませんでした")

    result = r.json()["chart"]["result"][0]
    q = result["indicators"]["quote"][0]
    adj = result["indicators"]["adjclose"][0]["adjclose"]
    df = pd.DataFrame(
        {
            "open": q["open"],
            "high": q["high"],
            "low": q["low"],
            "close": q["close"],
            "adjclose": adj,
            "volume": q["volume"],
        },
        index=pd.to_datetime(result["timestamp"], unit="s").normalize(),
    )
    df.index.name = "date"
    # 配当: 権利落ち日に「配当 ÷ 前日の終値」を入れる（税金の計算に使う）
    df["div_yield"] = 0.0
    for ev in (result.get("events") or {}).get("dividends", {}).values():
        d = pd.to_datetime(ev["date"], unit="s").normalize()
        if d in df.index:
            pos = df.index.get_loc(d)
            if pos > 0 and df["close"].iloc[pos - 1]:
                df.iloc[pos, df.columns.get_loc("div_yield")] = ev["amount"] / df["close"].iloc[pos - 1]
    df = df.dropna(subset=["open", "close", "adjclose"])
    df = df[~df.index.duplicated(keep="last")]
    # 調整後終値との比率で OHLC をまとめて調整する
    ratio = df["adjclose"] / df["close"]
    for col in ("open", "high", "low", "close"):
        df[col] = df[col] * ratio
    return df[["open", "high", "low", "close", "volume", "div_yield"]]


def load(symbol: str, start: str = "2005-01-01", refresh: bool = False) -> pd.DataFrame:
    """銘柄の調整済み日足を返す。data/ にキャッシュする。"""
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{symbol.replace('^', '_')}.csv"
    if path.exists() and not refresh:
        df = pd.read_csv(path, index_col="date", parse_dates=True)
        if "div_yield" in df:
            return df
    df = _fetch_yahoo(symbol, start)
    df.to_csv(path)
    return df


def load_many(symbols: list[str], start: str = "2005-01-01", refresh: bool = False) -> dict[str, pd.DataFrame]:
    return {s: load(s, start, refresh) for s in symbols}
