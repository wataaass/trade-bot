"""ペーパー運用（のちに本番）用の発注計画と帳簿。

1つの moomoo 口座の中で、戦略ごとに仮想の資金と建玉を帳簿で分けて管理する。
bot は帳簿にある自分の建玉しか売らないので、口座にある他の建玉には触れない。

流れ（毎月、最初の取引日）:
  1. plan:   前月末の終値でシグナルを計算し、注文の一覧を state/plans/YYYY-MM.json に書く
  2. 発注:   Claude が moomoo の MCP で成行注文を出す（売りを先、買いを後）
  3. record: 約定した数量と価格を帳簿 state/ledger.json に書き込む
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import data as dt
from . import indicators as ind
from . import strategies as st
from . import universe as uv

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state"
LEDGER = STATE / "ledger.json"

INITIAL_CASH = 2000.0
BENCH_TRADE = "SPYM"  # 実際に売買する安い S&P500 ETF（シグナルは SPY で計算）
RISK_OFF = "IEF"
MOMENTUM_TOP_N = 5
LIVE_START = "2024-06-01"  # 252日のモメンタムと200日線が計算できる長さ
BUY_BUFFER = 1.01  # 成行の買いで現金が足りなくならないよう、前日終値の 1% 上で数量を決める


@dataclass
class Order:
    strategy: str
    side: str  # "BUY" / "SELL"
    symbol: str
    qty: int
    ref_price: float  # 数量を決めるのに使った前日終値
    reason: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ---------------------------------------------------------------- 帳簿


def new_ledger(start: str) -> dict:
    return {
        "started": start,
        "last_rebalance": None,
        "strategies": {
            name: {"cash": INITIAL_CASH, "positions": {}} for name in ("momentum", "trend", "swing")
        },
        # 比較用: 同じ額を SPYM に入れて持ち続けた場合（実際には発注しない）。初回の入れ替えの後に決める
        "benchmark": {"symbol": BENCH_TRADE, "qty": None, "cash": INITIAL_CASH,
                      "note": "比較用の仮想ポジション。口座では発注しない"},
        "fills": [],
    }


def load_ledger() -> dict | None:
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else None


def save_ledger(ledger: dict) -> None:
    STATE.mkdir(exist_ok=True)
    LEDGER.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n")


def apply_fill(ledger: dict, strategy: str, side: str, symbol: str, qty: float, price: float, fee: float,
               date: str) -> None:
    book = ledger["strategies"][strategy]
    pos = book["positions"].get(symbol, {"qty": 0, "cost": 0.0})
    if side == "BUY":
        total = pos["qty"] * pos["cost"] + qty * price + fee
        pos["qty"] += qty
        pos["cost"] = total / pos["qty"]
        book["cash"] -= qty * price + fee
    elif side == "SELL":
        if qty > pos["qty"]:
            raise ValueError(f"{strategy}: {symbol} を {qty} 株売ろうとしたが帳簿には {pos['qty']} 株しかない")
        pos["qty"] -= qty
        book["cash"] += qty * price - fee
    else:
        raise ValueError(side)
    if pos["qty"] == 0:
        book["positions"].pop(symbol, None)
    else:
        book["positions"][symbol] = pos
    ledger["fills"].append(dict(date=date, strategy=strategy, side=side, symbol=symbol, qty=qty, price=price,
                                fee=fee))


def set_benchmark(ledger: dict, price: float, fee_rate: float = 0.00132) -> None:
    """比較用の「持ち続け」を、初回の約定と同じ価格・同じ手数料で買ったことにする。"""
    b = ledger["benchmark"]
    if b["qty"] is not None:
        return
    qty = math.floor(INITIAL_CASH / (price * (1 + fee_rate)))
    b["qty"] = qty
    b["cash"] = INITIAL_CASH - qty * price * (1 + fee_rate)
    b["price"] = price


def first_fill_price(ledger: dict, symbol: str) -> float | None:
    for f in ledger["fills"]:
        if f["symbol"] == symbol and f["side"] == "BUY":
            return f["price"]
    return None


# ---------------------------------------------------------------- シグナル


def last_month_end(index: pd.DatetimeIndex, today: pd.Timestamp) -> pd.Timestamp:
    """today より前の、最後に終わった月の最終取引日。"""
    prev = index[index < today.normalize().replace(day=1)]
    if len(prev) == 0:
        raise ValueError("前月末のデータがない")
    return prev[-1]


def trend_target(spy: pd.DataFrame, asof: pd.Timestamp) -> str:
    c = spy["close"].loc[:asof]
    on = c.iloc[-1] > ind.sma(c, 200).iloc[-1]
    return BENCH_TRADE if on else RISK_OFF


def momentum_ranking(data: dict[str, pd.DataFrame], asof: pd.Timestamp) -> list[str]:
    """asof 時点で S&P500 に入っている銘柄を、12-1 モメンタムの高い順に並べる。"""
    closes = pd.DataFrame({s: df["close"] for s, df in data.items()}).loc[:asof]
    members = uv.member_mask(closes.index[-1:], list(closes.columns)).iloc[-1]
    score = (closes.shift(21) / closes.shift(252) - 1).iloc[-1]
    score = score[members & score.notna() & closes.iloc[-1].notna()]
    return list(score.sort_values(ascending=False, kind="stable").index)


# ---------------------------------------------------------------- 発注計画


def _value(book: dict, prices: dict[str, float]) -> float:
    return book["cash"] + sum(p["qty"] * prices[s] for s, p in book["positions"].items())


def plan_trend(book: dict, target: str, prices: dict[str, float], strategy: str = "trend",
               on_symbol: str = BENCH_TRADE) -> list[Order]:
    orders = []
    cash = book["cash"]
    for sym, pos in book["positions"].items():
        if sym != target:
            orders.append(Order(strategy, "SELL", sym, int(pos["qty"]), prices[sym], f"切り替え先は {target}"))
            cash += pos["qty"] * prices[sym]
    qty = math.floor(cash / (prices[target] * BUY_BUFFER))
    if qty > 0:
        reason = "SPY が200日線より上" if target == on_symbol else "SPY が200日線より下"
        orders.append(Order(strategy, "BUY", target, qty, prices[target], reason))
    return orders


def plan_momentum(book: dict, ranking: list[str], prices: dict[str, float], top_n: int = MOMENTUM_TOP_N
                  ) -> list[Order]:
    keep = set(ranking[:top_n])
    orders = []
    cash = book["cash"]
    held = dict(book["positions"])
    for sym, pos in list(held.items()):
        if sym not in keep:
            rank = ranking.index(sym) + 1 if sym in ranking else None
            why = f"順位 {rank} 位に下がった" if rank else "S&P500 から外れたかデータがない"
            orders.append(Order("momentum", "SELL", sym, int(pos["qty"]), prices[sym], why))
            cash += pos["qty"] * prices[sym]
            held.pop(sym)
    equity = _value(book, prices)
    slot = equity / top_n
    for i, sym in enumerate(ranking, 1):
        if len(held) >= top_n or cash <= 0:
            break
        if sym in held or sym not in prices:
            continue
        budget = min(slot, cash)
        qty = math.floor(budget / (prices[sym] * BUY_BUFFER))
        if qty <= 0:
            continue  # 1株も買えない値がさ株は飛ばして次点へ
        orders.append(Order("momentum", "BUY", sym, qty, prices[sym], f"12-1 モメンタム {i} 位"))
        cash -= qty * prices[sym] * BUY_BUFFER
        held[sym] = {"qty": qty}
    return orders


def load_live_data(symbols: list[str], start: str = LIVE_START) -> dict[str, pd.DataFrame]:
    """発注計画用に最新の日足を取り直す（検証用のキャッシュ data/*.csv は触らない）。"""
    from concurrent.futures import ThreadPoolExecutor

    def one(s: str):
        try:
            return s, dt._fetch_yahoo(uv.yahoo_symbol(s), start)
        except Exception as e:  # noqa: BLE001
            print(f"  {s}: 取得失敗 {e}")
            return s, None

    with ThreadPoolExecutor(max_workers=4) as ex:
        return {s: df for s, df in ex.map(one, symbols) if df is not None and not df.empty}


def current_members() -> list[str]:
    df = uv.intervals()
    return sorted(df.loc[df["end"].isna(), "ticker"].unique())


# ---------------------------------------------------------------- スイング（毎日）

SWING_MAX_POS = 5
SWING_STOP = 0.10  # 建値から 10% 下の終値で損切り（翌日に成行で売る）


DAILY_STRATEGIES = ("swing", "swing_aggr", "lev3")


def ensure_swing_book(ledger: dict) -> None:
    """毎日動く戦略の帳簿がなければ作る（途中から追加した戦略のため）。"""
    for name in DAILY_STRATEGIES:
        ledger["strategies"].setdefault(name, {"cash": INITIAL_CASH, "positions": {}})
    ledger.setdefault("started_at", {})
    ledger.setdefault("swing_started", None)
    ledger.setdefault("last_swing", None)


def swing_signals(stocks: dict[str, pd.DataFrame], spy: pd.DataFrame, asof: pd.Timestamp,
                  mom_top: int | None = None) -> tuple[list[str], set[str]]:
    """asof の終値で、(買い候補をモメンタム順に, 売りシグナルの出た銘柄) を返す。検証と同じ関数を使う。"""
    idx = pd.DatetimeIndex(sorted(set().union(*(d.index for d in stocks.values()))))
    idx = idx[idx <= asof]
    members = uv.member_mask(idx, list(stocks))
    sig = st.stock_swing({s: d.loc[:asof] for s, d in stocks.items()}, spy.loc[:asof], members, kind="breakout55",
                         mom_top=mom_top)
    entries = [(s, f["score"].iloc[-1]) for s, f in sig.items() if len(f) and f.index[-1] == asof
               and bool(f["entry"].iloc[-1])]
    entries.sort(key=lambda x: -x[1] if pd.notna(x[1]) else float("inf"))
    exits = {s for s, f in sig.items() if len(f) and f.index[-1] == asof and bool(f["exit"].iloc[-1])}
    return [s for s, _ in entries], exits


def plan_swing(book: dict, entries: list[str], exits: set[str], prices: dict[str, float],
               max_pos: int = SWING_MAX_POS, strategy: str = "swing") -> list[Order]:
    orders = []
    cash = book["cash"]
    held = dict(book["positions"])
    for sym, pos in list(held.items()):
        why = None
        if sym not in prices:
            why = "データがない（上場廃止など）"
        elif prices[sym] <= pos["cost"] * (1 - SWING_STOP):
            why = f"損切り（建値 {pos['cost']:.2f} の {SWING_STOP:.0%} 下）"
        elif sym in exits:
            why = "20日安値を割った"
        if why:
            orders.append(Order(strategy, "SELL", sym, int(pos["qty"]), prices.get(sym, pos["cost"]), why))
            cash += pos["qty"] * prices.get(sym, pos["cost"])
            held.pop(sym)
    slot = _value(book, {**{s: p["cost"] for s, p in book["positions"].items()}, **prices}) / max_pos
    for sym in entries:
        if len(held) >= max_pos or cash <= 0:
            break
        if sym in held or sym not in prices:
            continue
        qty = math.floor(min(slot, cash) / (prices[sym] * BUY_BUFFER))
        if qty <= 0:
            continue
        orders.append(Order(strategy, "BUY", sym, qty, prices[sym], "55日高値を更新"))
        cash -= qty * prices[sym] * BUY_BUFFER
        held[sym] = {"qty": qty}
    return orders


# ---------------------------------------------------------------- ガチ枠（毎日）

LEV3 = "UPRO"  # S&P500 の3倍 ETF
LEV3_OFF = "BIL"  # 下落時の退避先（1〜3か月の米短期国債）
AGGR_MOM_TOP = 100  # ガチスイング: 12-1 モメンタム上位100銘柄だけ
AGGR_MAX_POS = 3


def lev3_target(spy: pd.DataFrame, asof: pd.Timestamp) -> str:
    """SPY の終値が200日線より上なら 3倍 ETF、下なら短期国債。"""
    c = spy["close"].loc[:asof]
    return LEV3 if c.iloc[-1] > ind.sma(c, 200).iloc[-1] else LEV3_OFF
