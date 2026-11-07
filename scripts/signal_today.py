"""今日の各 bot の判定を表示する（Muse の判定との突き合わせ用）。

    python scripts/signal_today.py            # 最新の終値で判定
    python scripts/signal_today.py --momentum # 個別株モメンタム（D）の上位10も計算（数分かかる）

価格は Yahoo Finance の調整済み終値。取引時間中に実行すると、その日の途中の値も入るので、
判定に使うのは「前日までに確定した終値」だけにしている。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot import handoff as ho  # noqa: E402
from tradebot import live  # noqa: E402


def confirmed(df: pd.DataFrame) -> pd.Series:
    """確定した終値だけ（ニューヨークの今日の日付の行は、まだ確定していないので除く）。"""
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    c = df["close"]
    return c[c.index < today]


def describe(name: str, close: pd.Series) -> str:
    sma = close.iloc[-200:].mean()
    gap = close.iloc[-1] / sma - 1
    return f"{name} 終値 {close.iloc[-1]:.2f}、200日線 {sma:.2f}（{gap * 100:+.1f}%）"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--momentum", action="store_true")
    args = ap.parse_args()

    data = live.load_live_data(["SPY", "QQQ"], start="2024-01-01")
    spy, qqq = confirmed(data["SPY"]), confirmed(data["QQQ"])
    asof = spy.index[-1].date()

    prev = spy.loc[spy.index.to_period("M") < spy.index[-1].to_period("M")]
    vol = float(prev.pct_change().iloc[-20:].std() * np.sqrt(252))

    print(f"判定日: {asof} の終値（翌営業日の寄り付きで発注）")
    print()
    print(f"A. SPY 3倍＋200日線        → {ho.target_lev3(spy)}")
    print(f"   {describe('SPY', spy)}")
    print(f"B. ボラ連動レバ＋200日線   → {ho.target_voltarget(spy)}")
    print(f"   前月末（{prev.index[-1].date()}）の値動き 年率 {vol * 100:.1f}%（15%未満 UPRO／25%未満 SSO／それ以上 SPYM）")
    print(f"C. QQQ 3倍＋200日線（実験）→ {ho.target_tqqq(qqq)}")
    print(f"   {describe('QQQ', qqq)}")

    if args.momentum:
        members = live.current_members()
        print()
        print(f"D. 個別株モメンタム6-1（実験）: S&P500 の {len(members)} 銘柄を取得中…")
        stocks = live.load_live_data(members, start="2025-01-01")
        closes = pd.DataFrame({s: confirmed(d) for s, d in stocks.items()})
        month_end = closes.loc[closes.index.to_period("M") < closes.index[-1].to_period("M")]
        top = ho.momentum_6_1(month_end, members)
        print(f"   前月末（{month_end.index[-1].date()}）時点の上位10: {', '.join(top)}")


if __name__ == "__main__":
    main()
