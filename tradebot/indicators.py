"""テクニカル指標。どれも t 日までのデータだけを使う。"""

from __future__ import annotations

import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def rsi(s: pd.Series, n: int) -> pd.Series:
    """Wilder の RSI。"""
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = up / down
    out = 100 - 100 / (1 + rs)
    return out.where(down != 0, 100.0)
