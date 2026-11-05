"""S&P500 の各時点の構成銘柄。

元データ: https://github.com/fja05680/sp500 （MIT License, (c) Farrell J. Aultman）
を「銘柄・採用日・除外日」の区間に変換して ref/sp500_membership.csv に置いている。
最終更新（2026-08-18）以降の入れ替えは反映していない。

注意:
- 上場廃止になった銘柄は Yahoo Finance に株価がないことが多く、検証から抜け落ちる。
  倒産した銘柄が抜けるぶん、成績はやや良く出る（生存者バイアスが少し残る）
- ティッカーが別の会社に使い回された場合、古い期間の株価が別会社のものになることがある
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

MEMBERSHIP = Path(__file__).resolve().parent / "ref" / "sp500_membership.csv"


def intervals() -> pd.DataFrame:
    df = pd.read_csv(MEMBERSHIP, parse_dates=["start", "end"])
    return df


def tickers_since(date: str = "2006-01-01") -> list[str]:
    df = intervals()
    keep = df["end"].isna() | (df["end"] >= pd.Timestamp(date))
    return sorted(df.loc[keep, "ticker"].unique())


def yahoo_symbol(ticker: str) -> str:
    return ticker.replace(".", "-")


def member_mask(index: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    """index の各日に、その銘柄が S&P500 に入っていたら True。"""
    df = intervals()
    out = pd.DataFrame(False, index=index, columns=tickers)
    for row in df.itertuples():
        if row.ticker not in out:
            continue
        end = row.end if pd.notna(row.end) else index[-1] + pd.Timedelta(days=1)
        out.loc[(index >= row.start) & (index < end), row.ticker] = True
    return out
