"""Kronos（金融の値動きを学習した基盤モデル）で米国の大型株の1週間先を予測し、上位を持つ bot の検証。

毎週の最終営業日の終値のあとに判定し、翌営業日の始値で持ち替える。
1. その日に S&P500 に入っている銘柄から、直近60営業日の売買代金の中央値が大きい順に N 銘柄を選ぶ
2. 各銘柄の直近 lookback 日の日足を Kronos に渡し、5営業日先の終値を予測する
3. 予測した上昇率の高い順に K 銘柄を同じ比率で持つ。buffer > 0 なら、持っている銘柄は
   上位 K+buffer にいる間は持ち続ける（入れ替えを減らして手数料と税金を抑える）

Kronos の学習データは 2024年6月まで。それより前を予測させると答えを知っている状態になるので、
正式な評価は 2024年7月以降に限る。

このファイルは torch なしで動く（予測を呼ぶ `predict_week` だけが Kronos の predictor を受け取る）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

STOCKS = Path(__file__).resolve().parent.parent / "data" / "stocks"
TRAIN_END = pd.Timestamp("2024-06-30")  # Kronos の学習データの終わり（論文 arXiv:2508.02739）
MODELS = {
    "mini": ("NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-2k", 2048),
    "small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512),
    "base": ("NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base", 512),
}


def load_stocks(tickers: list[str] | None = None, min_rows: int = 260) -> dict[str, pd.DataFrame]:
    """data/stocks/ の日足を読む（scripts/fetch_sp500.py で取得したもの）。"""
    paths = sorted(STOCKS.glob("*.csv")) if tickers is None else [STOCKS / f"{t}.csv" for t in tickers]
    out = {}
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col="date", parse_dates=True)
        df = df[df["volume"] > 0]
        if len(df) >= min_rows:
            out[path.stem] = df
    return out


def week_ends(index: pd.DatetimeIndex, start: str | pd.Timestamp, end: str | pd.Timestamp | None = None) -> list[pd.Timestamp]:
    """各週の最終営業日（index の中で、次の日が別の週になる日）。最後の日は週の途中でも含めない。"""
    idx = pd.DatetimeIndex(index).sort_values()
    iso = idx.isocalendar()
    key = iso["year"].to_numpy() * 100 + iso["week"].to_numpy()
    last = np.r_[key[1:] != key[:-1], False]
    out = idx[last]
    out = out[out >= pd.Timestamp(start)]
    if end is not None:
        out = out[out <= pd.Timestamp(end)]
    return list(out)


def select_universe(
    data: dict[str, pd.DataFrame], members: list[str], date: pd.Timestamp, n: int, lookback: int, window: int = 60
) -> list[str]:
    """date に S&P500 に入っていて、lookback 日分の履歴がある銘柄から、売買代金の大きい順に n 銘柄。"""
    scores = {}
    for t in members:
        df = data.get(t)
        if df is None:
            continue
        hist = df.loc[:date]
        if len(hist) < lookback or hist.index[-1] != date:
            continue
        tail = hist.iloc[-window:]
        scores[t] = float((tail["close"] * tail["volume"]).median())
    return sorted(scores, key=lambda t: -scores[t])[:n]


def future_stamps(date: pd.Timestamp, pred_len: int) -> pd.Series:
    """予測する日付（土日を除いた次の pred_len 日）。祝日は考えない（モデルには曜日と日付の特徴としてだけ渡る）。"""
    return pd.Series(pd.bdate_range(date + pd.Timedelta(days=1), periods=pred_len))


def predict_week(
    predictor,
    data: dict[str, pd.DataFrame],
    syms: list[str],
    date: pd.Timestamp,
    lookback: int,
    pred_len: int = 5,
    sample_count: int = 5,
    batch: int = 25,
) -> pd.Series:
    """各銘柄の pred_len 営業日先の終値の予測から、date の終値に対する上昇率を返す。"""
    out = {}
    for i in range(0, len(syms), batch):
        chunk = syms[i : i + batch]
        dfs, xs, ys = [], [], []
        for t in chunk:
            hist = data[t].loc[:date].iloc[-lookback:]
            df = hist[["open", "high", "low", "close", "volume"]].astype(float).copy()
            df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
            dfs.append(df.reset_index(drop=True))
            xs.append(pd.Series(hist.index))
            ys.append(future_stamps(date, pred_len))
        preds = predictor.predict_batch(dfs, xs, ys, pred_len=pred_len, T=1.0, top_p=0.9, sample_count=sample_count, verbose=False)
        for t, df, p in zip(chunk, dfs, preds, strict=True):
            out[t] = float(p["close"].iloc[-1] / df["close"].iloc[-1] - 1)
    return pd.Series(out, name=date)


def forward_returns(data: dict[str, pd.DataFrame], syms: list[str], date: pd.Timestamp, horizon: int = 5) -> pd.Series:
    """date の終値から horizon 営業日後の終値までの実際の上昇率（予測の当たり具合を測る用）。"""
    out = {}
    for t in syms:
        c = data[t]["close"]
        if date not in c.index:
            continue
        pos = c.index.get_loc(date)
        if pos + horizon < len(c):
            out[t] = float(c.iloc[pos + horizon] / c.iloc[pos] - 1)
    return pd.Series(out, name=date)


def momentum_scores(data: dict[str, pd.DataFrame], syms: list[str], date: pd.Timestamp, long: int, skip: int = 0) -> pd.Series:
    """過去 long 営業日の上昇率（直近 skip 日は除く）。比べる相手の単純な型に使う。"""
    out = {}
    for t in syms:
        c = data[t]["close"].loc[:date]
        if len(c) > long:
            out[t] = float(c.iloc[-1 - skip] / c.iloc[-1 - long] - 1)
    return pd.Series(out, name=date)


def pick_top(scores: pd.DataFrame, k: int, buffer: int = 0) -> pd.DataFrame:
    """scores（index=判定日、列=銘柄、値=予測の上昇率、対象外は NaN）から、各判定日の目標の比率を作る。

    上位 k 銘柄を同じ比率で持つ。buffer > 0 なら、前回持っていた銘柄は上位 k+buffer にいる間は残し、
    空いた分だけ上から順に足す。
    """
    rows = {}
    held: list[str] = []
    for date, row in scores.iterrows():
        s = row.dropna().sort_values(ascending=False)
        ranked = list(s.index)
        if buffer > 0:
            keep = [t for t in held if t in ranked[: k + buffer]]
            keep = sorted(keep, key=ranked.index)[:k]
            add = [t for t in ranked if t not in keep][: k - len(keep)]
            held = keep + add
        else:
            held = ranked[:k]
        w = pd.Series(0.0, index=scores.columns)
        if held:
            w[held] = 1.0 / len(held)
        rows[date] = w
    return pd.DataFrame(rows).T


def equal_weight(scores: pd.DataFrame) -> pd.DataFrame:
    """対象の全銘柄を同じ比率で持つ（比べる相手）。"""
    has = scores.notna().astype(float)
    return has.div(has.sum(axis=1), axis=0).fillna(0.0)


def rank_ic(pred: pd.DataFrame, real: pd.DataFrame) -> pd.Series:
    """判定日ごとの、予測と実際の上昇率の順位相関（Spearman）。"""
    out = {}
    for date in pred.index:
        if date not in real.index:
            continue
        both = pd.concat([pred.loc[date], real.loc[date]], axis=1, keys=["p", "r"]).dropna()
        if len(both) >= 5:
            out[date] = float(both["p"].rank().corr(both["r"].rank()))
    return pd.Series(out, dtype=float)


def top_minus_all(pred: pd.DataFrame, real: pd.DataFrame, k: int) -> pd.Series:
    """判定日ごとの「予測の上位 k 銘柄の実際の上昇率の平均 − 全銘柄の平均」（費用を引く前）。"""
    out = {}
    for date in pred.index:
        if date not in real.index:
            continue
        both = pd.concat([pred.loc[date], real.loc[date]], axis=1, keys=["p", "r"]).dropna()
        if len(both) > k:
            top = both.sort_values("p", ascending=False).iloc[:k]
            out[date] = float(top["r"].mean() - both["r"].mean())
    return pd.Series(out, dtype=float)
