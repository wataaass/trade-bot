"""1926年からの長期データ。

- 米国株全体: Fama/French の日次ファクター（Mkt-RF + RF）。CRSP の時価総額加重、配当込み
- 現金: 同じファイルの RF（1か月物の米短期国債）
- 配当利回り: Shiller の月次データ（D/P）。データが終わった後は SPY の実績で補う
出典:
  https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
  http://www.econ.yale.edu/~shiller/data.htm

日足の始値・高値・安値はないので、すべて終値と同じにする。シグナルの翌日の終値で約定する
ことになり、翌日の始値より1日遅いぶん保守的。
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

from . import data as dt

CACHE = Path(__file__).resolve().parent.parent / "data"
FF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
SHILLER_URL = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
_HEADERS = {"User-Agent": "Mozilla/5.0 (trade-bot backtester)"}


def _ff_daily() -> pd.DataFrame:
    path = CACHE / "ff_daily.csv"
    if not path.exists():
        r = requests.get(FF_URL, headers=_HEADERS, timeout=60)
        r.raise_for_status()
        raw = zipfile.ZipFile(io.BytesIO(r.content)).read("F-F_Research_Data_Factors_daily.csv").decode()
        lines = [ln for ln in raw.splitlines() if ln[:8].strip().isdigit() and "," in ln]
        df = pd.read_csv(io.StringIO("\n".join(lines)), header=None, names=["date", "mkt_rf", "smb", "hml", "rf"])
        df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), format="%Y%m%d")
        CACHE.mkdir(exist_ok=True)
        df.to_csv(path, index=False)
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")


def _shiller_dy() -> pd.Series:
    """月次の配当利回り（年率）。index は月初。"""
    path = CACHE / "shiller_dy.csv"
    if not path.exists():
        r = requests.get(SHILLER_URL, headers=_HEADERS, timeout=60)
        r.raise_for_status()
        x = pd.read_excel(io.BytesIO(r.content), sheet_name="Data", skiprows=7, engine="calamine")
        x = x[pd.to_numeric(x["Date"], errors="coerce").notna()]
        x = x[pd.to_numeric(x["D"], errors="coerce").notna()]
        d = x["Date"].astype(float)
        year = d.astype(int)
        month = ((d - year) * 100).round().astype(int)
        idx = pd.to_datetime(dict(year=year, month=month, day=1))
        dy = pd.Series(x["D"].astype(float).values / x["P"].astype(float).values, index=idx, name="dy")
        dy.to_csv(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)["dy"]


def _frame(close: pd.Series, div_yield: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 0, "div_yield": div_yield}
    )


def market_and_cash() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(米国株全体, 現金) の日足。価格は配当・利息込みの累積指数。"""
    ff = _ff_daily()
    mkt_ret = (ff["mkt_rf"] + ff["rf"]) / 100
    rf = ff["rf"] / 100
    mkt_close = 100 * (1 + mkt_ret).cumprod()
    cash_close = 100 * (1 + rf).cumprod()

    # 配当利回り（日割り）: Shiller の月次 D/P を、その月の営業日に均等に配る
    dy = _shiller_dy()
    month = mkt_close.index.to_period("M").to_timestamp()
    annual = pd.Series(dy.reindex(month).to_numpy(), index=mkt_close.index)
    # Shiller が終わった後は SPY の直近1年の配当利回りで補う
    spy = dt.load("SPY")
    spy_annual = spy["div_yield"].rolling(252, min_periods=200).sum().reindex(mkt_close.index, method="ffill")
    annual = annual.fillna(spy_annual).ffill()
    mkt_div = annual / 252

    return _frame(mkt_close, mkt_div), _frame(cash_close, rf)


def leveraged(mkt: pd.DataFrame, cash: pd.DataFrame, lev: float, spread: float = 0.005,
              expense: float = 0.009) -> pd.DataFrame:
    """日々 lev 倍の値動きをするレバレッジ ETF を合成する。

    日次リターン = lev × 株のリターン − (lev − 1) × (短期金利 + spread) − 信託報酬
    spread は借入の上乗せ金利、expense は信託報酬（いずれも年率）。分配金はないものとする。
    """
    r_m = mkt["close"].pct_change().fillna(0)
    rf = cash["close"].pct_change().fillna(0)
    r = lev * r_m - (lev - 1) * (rf + spread / 252) - expense / 252
    close = 100 * (1 + r).cumprod()
    close = close.clip(lower=1e-9)  # 1日で -100% 以下になったら実質ゼロ
    return _frame(close, pd.Series(0.0, index=close.index))


def _shiller_gs10() -> pd.Series:
    """米10年国債の利回り（月次、年率の小数）。index は月初。"""
    path = CACHE / "shiller_gs10.csv"
    if not path.exists():
        r = requests.get(SHILLER_URL, headers=_HEADERS, timeout=60)
        r.raise_for_status()
        x = pd.read_excel(io.BytesIO(r.content), sheet_name="Data", skiprows=7, engine="calamine")
        x = x[pd.to_numeric(x["Date"], errors="coerce").notna()]
        x = x[pd.to_numeric(x["Rate GS10"], errors="coerce").notna()]
        d = x["Date"].astype(float)
        year = d.astype(int)
        month = ((d - year) * 100).round().astype(int)
        idx = pd.to_datetime(dict(year=year, month=month, day=1))
        pd.Series(x["Rate GS10"].astype(float).values / 100, index=idx, name="gs10").to_csv(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)["gs10"]


def bond10(index: pd.DatetimeIndex) -> pd.DataFrame:
    """米10年国債の値動き（利息込み）を合成する。

    月次リターン ≒ 前月の利回り/12 ＋ 修正デュレーション × (前月の利回り − 今月の利回り)。
    修正デュレーションは、10年・半年複利の額面債の近似 (1 − (1 + y/2)^−20) / y。
    月次リターンを、その月の営業日に均等（複利）に割り振る。2002年8月以降は実在の IEF（7〜10年国債）につなぐ。
    """
    y = _shiller_gs10()
    y_prev = y.shift(1)
    dur = (1 - (1 + y_prev / 2) ** -20) / y_prev
    r_m = (y_prev / 12 + dur * (y_prev - y)).dropna()
    month = index.to_period("M").to_timestamp()
    days_in_month = pd.Series(1, index=index).groupby(month).transform("size")
    r_d = (1 + pd.Series(r_m.reindex(month).to_numpy(), index=index)) ** (1 / days_in_month) - 1
    ief = dt.load("IEF")
    ief_r = ief["close"].pct_change().reindex(index)
    splice = index >= ief.index[0] + pd.Timedelta(days=5)
    r_d = r_d.where(~splice, ief_r).fillna(0)
    close = 100 * (1 + r_d).cumprod()
    return _frame(close, pd.Series(0.0, index=index))


def shiller_cape() -> pd.Series:
    """CAPE（景気循環調整後 PER、月次）。index は月初。データが公表されるまでの遅れは考えない。"""
    path = CACHE / "shiller_cape.csv"
    if not path.exists():
        r = requests.get(SHILLER_URL, headers=_HEADERS, timeout=60)
        r.raise_for_status()
        x = pd.read_excel(io.BytesIO(r.content), sheet_name="Data", skiprows=7, engine="calamine")
        x = x[pd.to_numeric(x["Date"], errors="coerce").notna()]
        x = x[pd.to_numeric(x["CAPE"], errors="coerce").notna()]
        d = x["Date"].astype(float)
        year = d.astype(int)
        month = ((d - year) * 100).round().astype(int)
        idx = pd.to_datetime(dict(year=year, month=month, day=1))
        pd.Series(x["CAPE"].astype(float).values, index=idx, name="cape").to_csv(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)["cape"]
