"""戦略の動物園（zoo）: たくさんの候補を同じ物差しで評価する。

各候補は「土俵（universe）」「シグナルを作る関数」「設定」の組。
評価は bt.run を2回（税なし・特定口座）走らせ、
- 全期間の税引き後の年率（最終日に全部売った手取り）
- 時代ごとの税引き前の年率（同じ実行の評価額を時代で切り出す）
を、同じ土俵の「持ち続け」と比べる。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import backtest as bt
from . import indicators as ind
from . import strategies as st


@dataclass
class Candidate:
    name: str
    family: str  # 同じアイデアのパラメータ違いは同じ family
    universe: str  # "century" / "etf" / "stocks"
    make: Callable[[dict, pd.DataFrame], dict]  # (data, bench) -> signals
    max_positions: int = 1
    stop_loss: float | None = None
    max_hold_days: int | None = None
    desc: str = ""
    symbols: list[str] | None = None  # 使う銘柄（None なら土俵の全銘柄）
    params: dict = field(default_factory=dict)


# ---------------------------------------------------------------- 共通の部品


def month_end(idx: pd.DatetimeIndex) -> pd.Series:
    return st._month_end(idx)


def targets_to_signals(target: pd.Series, check: pd.Series, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """日ごとの「持つべき銘柄」(None は何も持たない) から、売買シグナルを作る。

    check が True の日だけ判定する（毎日なら全部 True、月次なら月末だけ True）。
    """
    out = {}
    for s in symbols:
        want = (target == s)
        entry = want & check
        exit_ = (~want) & check
        out[s] = pd.DataFrame({"entry": entry, "exit": exit_, "score": 0.0}, index=target.index)
    return out


def realized_vol(close: pd.Series, n: int = 20) -> pd.Series:
    return close.pct_change().rolling(n).std() * np.sqrt(252)


# ---------------------------------------------------------------- 土俵 century（約100年、米国株全体）
# 銘柄: MKT（1倍）、L2（2倍）、L3（3倍）、CASH（短期国債）


def c_sma(n: int, lev: str = "MKT", monthly: bool = False):
    def make(data, bench):
        m = data["MKT"]["close"]
        on = m > ind.sma(m, n)
        tgt = pd.Series(np.where(on, lev, "CASH"), index=m.index).where(ind.sma(m, n).notna())
        check = month_end(m.index) if monthly else pd.Series(True, index=m.index)
        return targets_to_signals(tgt, check, [lev, "CASH"])
    return make


def c_cross(fast: int, slow: int, lev: str = "MKT"):
    def make(data, bench):
        m = data["MKT"]["close"]
        f, s = ind.sma(m, fast), ind.sma(m, slow)
        tgt = pd.Series(np.where(f > s, lev, "CASH"), index=m.index).where(s.notna())
        return targets_to_signals(tgt, pd.Series(True, index=m.index), [lev, "CASH"])
    return make


def c_tsmom(months: int = 12, lev: str = "MKT"):
    """時系列モメンタム: 過去 months か月の株のリターンが短期国債を上回っていれば株（月末判定）。"""
    def make(data, bench):
        m, c = data["MKT"]["close"], data["CASH"]["close"]
        n = 21 * months
        on = (m / m.shift(n)) > (c / c.shift(n))
        tgt = pd.Series(np.where(on, lev, "CASH"), index=m.index).where(m.shift(n).notna())
        return targets_to_signals(tgt, month_end(m.index), [lev, "CASH"])
    return make


def c_voltarget(sma_n: int | None = 200, lo: float = 0.12, hi: float = 0.20, monthly: bool = False):
    """値動きの大きさでレバレッジを変える: 穏やかなら3倍、普通なら2倍、荒れたら1倍。

    sma_n を指定すると、株が sma_n 日線より下のときは短期国債に逃げる。
    """
    def make(data, bench):
        m = data["MKT"]["close"]
        v = realized_vol(m, 20)
        lev = np.where(v < lo, "L3", np.where(v < hi, "L2", "MKT"))
        tgt = pd.Series(lev, index=m.index).where(v.notna())
        if sma_n:
            sm = ind.sma(m, sma_n)
            tgt = tgt.where(m > sm, "CASH").where(sm.notna())
        check = month_end(m.index) if monthly else pd.Series(True, index=m.index)
        return targets_to_signals(tgt, check, ["MKT", "L2", "L3", "CASH"])
    return make


def c_season(lev: str = "MKT"):
    """Sell in May: 11〜4月は株、5〜10月は短期国債。"""
    def make(data, bench):
        m = data["MKT"]["close"]
        on = pd.Series(m.index.month.isin([11, 12, 1, 2, 3, 4]), index=m.index)
        tgt = pd.Series(np.where(on, lev, "CASH"), index=m.index)
        return targets_to_signals(tgt, month_end(m.index), [lev, "CASH"])
    return make


def c_hold(sym: str):
    def make(data, bench):
        m = data[sym]["close"]
        return targets_to_signals(pd.Series(sym, index=m.index), pd.Series(True, index=m.index), [sym])
    return make


def century_candidates() -> list[Candidate]:
    cs: list[Candidate] = []
    for lev, lname in (("MKT", "1倍"), ("L2", "2倍"), ("L3", "3倍")):
        cs.append(Candidate(f"{lname} 持ち続け", f"hold-{lev}", "century", c_hold(lev), symbols=[lev]))
        for n in (50, 100, 150, 200, 250):
            cs.append(Candidate(f"{lname} {n}日線・毎日", f"sma-daily-{lev}", "century", c_sma(n, lev),
                                symbols=[lev, "CASH"], params={"n": n},
                                desc=f"米国株が{n}日線より上なら{lname}、下なら短期国債。毎日判定"))
            cs.append(Candidate(f"{lname} {n}日線・月末", f"sma-monthly-{lev}", "century", c_sma(n, lev, True),
                                symbols=[lev, "CASH"], params={"n": n},
                                desc=f"米国株が{n}日線より上なら{lname}、下なら短期国債。月末だけ判定"))
        cs.append(Candidate(f"{lname} 50/200クロス", f"cross-{lev}", "century", c_cross(50, 200, lev),
                            symbols=[lev, "CASH"], desc=f"50日線が200日線より上なら{lname}、下なら短期国債"))
        for mo in (6, 12):
            cs.append(Candidate(f"{lname} 時系列モメンタム{mo}か月", f"tsmom-{lev}", "century", c_tsmom(mo, lev),
                                symbols=[lev, "CASH"], params={"months": mo},
                                desc=f"過去{mo}か月の株のリターンが短期国債より上なら{lname}。月末判定"))
        cs.append(Candidate(f"{lname} Sell in May", f"season-{lev}", "century", c_season(lev),
                            symbols=[lev, "CASH"], desc=f"11〜4月は{lname}、5〜10月は短期国債"))
    for sma_n in (None, 200):
        for lo, hi in ((0.10, 0.18), (0.12, 0.20), (0.15, 0.25)):
            tag = "＋200日線" if sma_n else ""
            cs.append(Candidate(f"ボラ連動レバ{tag}（{int(lo*100)}/{int(hi*100)}%）", f"voltarget{'-sma' if sma_n else ''}",
                                "century", c_voltarget(sma_n, lo, hi), symbols=["MKT", "L2", "L3", "CASH"],
                                params={"lo": lo, "hi": hi},
                                desc=f"直近20日の値動き（年率）が{int(lo*100)}%未満なら3倍、{int(hi*100)}%未満なら2倍、それ以上は1倍"
                                     + ("。米国株が200日線より下なら短期国債" if sma_n else "")))
    cs += band_candidates("century")
    return cs


# ---------------------------------------------------------------- 土俵 etf（2003年〜）
# 銘柄: SPY, QQQ, IWM, EFA, EEM, IEF, TLT, GLD, セクター9本, CASH（短期国債）、
#       SSO/UPRO 相当の合成（SPY2/SPY3）、QQQ の合成3倍（QQQ3）

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"]


def e_gem(lookback: int = 252, bond: str = "IEF"):
    """デュアル・モメンタム（Antonacci の GEM）: 月末に、米国株の12か月リターンが短期国債を
    上回れば米国株と先進国株の強いほう、下回れば債券。"""
    def make(data, bench):
        spy, efa, cash = data["SPY"]["close"], data["EFA"]["close"], data["CASH"]["close"]
        idx = spy.index
        r = lambda s: (s / s.shift(lookback)).reindex(idx)  # noqa: E731
        rs, re, rc = r(spy), r(efa), r(cash)
        best = pd.Series(np.where(rs >= re, "SPY", "EFA"), index=idx)
        tgt = best.where(rs > rc, bond).where(rs.notna() & re.notna() & rc.notna())
        return targets_to_signals(tgt, month_end(idx), ["SPY", "EFA", bond])
    return make


def e_trend(asset: str, risk_off: str, n: int = 200, signal: str | None = None, monthly: bool = True):
    def make(data, bench):
        sig = data[signal or asset]["close"]
        idx = data[asset].index
        sig = sig.reindex(idx)
        sm = ind.sma(sig, n)
        tgt = pd.Series(np.where(sig > sm, asset, risk_off), index=idx).where(sm.notna())
        check = month_end(idx) if monthly else pd.Series(True, index=idx)
        return targets_to_signals(tgt, check, [asset, risk_off])
    return make


def e_rotation(universe: list[str], top_k: int, lookback: int, filt: bool):
    def make(data, bench):
        sub = {s: data[s] for s in universe}
        sig = st.momentum_rotation(sub, bench, top_k=top_k, lookback=lookback)
        if filt:
            mok = st.market_ok(bench).reindex(next(iter(sig.values())).index).fillna(False).astype(bool)
            for s in sig:
                sig[s]["entry"] &= mok
                sig[s]["exit"] |= (~mok) & month_end(sig[s].index)
        return sig
    return make


def etf_candidates() -> list[Candidate]:
    cs: list[Candidate] = []
    cs.append(Candidate("SPY 持ち続け", "hold-spy", "etf", c_hold("SPY"), symbols=["SPY"]))
    cs.append(Candidate("QQQ 持ち続け", "hold-qqq", "etf", c_hold("QQQ"), symbols=["QQQ"]))
    cs.append(Candidate("SPY 3倍 持ち続け", "hold-spy3", "etf", c_hold("SPY3"), symbols=["SPY3"]))
    cs.append(Candidate("QQQ 3倍 持ち続け", "hold-qqq3", "etf", c_hold("QQQ3"), symbols=["QQQ3"]))
    for lb in (126, 252):
        for bond in ("IEF", "TLT", "CASH"):
            cs.append(Candidate(f"デュアルモメンタム（{lb // 21}か月・退避{bond}）", "gem", "etf", e_gem(lb, bond),
                                symbols=["SPY", "EFA", bond, "CASH"], params={"lookback": lb, "bond": bond},
                                desc=f"月末に、SPY の{lb // 21}か月リターンが短期国債より上なら SPY と EFA の強いほう、下なら {bond}"))
    for asset, name in (("SPY", "SPY"), ("QQQ", "QQQ"), ("SPY2", "SPY 2倍"), ("SPY3", "SPY 3倍"), ("QQQ3", "QQQ 3倍")):
        base = "QQQ" if asset.startswith("QQQ") else "SPY"
        for off in ("IEF", "CASH", "GLD"):
            for monthly in (True, False):
                when = "月末" if monthly else "毎日"
                cs.append(Candidate(f"{name}⇄{off} 200日線・{when}", f"trend-{asset}-{'m' if monthly else 'd'}", "etf",
                                    e_trend(asset, off, 200, signal=base, monthly=monthly),
                                    symbols=[asset, off, base], params={"off": off},
                                    desc=f"{base} が200日線より上なら {name}、下なら {off}。{when}判定"))
    for k in (1, 2, 3):
        for lb in (63, 126, 252):
            for filt in (False, True):
                cs.append(Candidate(f"セクター上位{k}（{lb // 21}か月）{'＋相場フィルター' if filt else ''}",
                                    f"sector{'-f' if filt else ''}", "etf", e_rotation(SECTORS, k, lb, filt),
                                    max_positions=k, symbols=SECTORS, params={"k": k, "lookback": lb},
                                    desc=f"月末に、9セクターETFの{lb // 21}か月リターン上位{k}本を持つ"
                                         + ("。SPY が200日線より下なら現金" if filt else "")))
    # 約100年の土俵と同じ型を、実在の SPY で確かめる（MKT=SPY、L2/L3=SPY の2倍・3倍の合成）
    for c in century_candidates():
        if c.family.startswith(("hold", "sma-daily-MKT", "sma-monthly-MKT")):
            continue
        if c.family.startswith(("sma-", "cross")) and c.params.get("n", 200) not in (150, 200, 250) \
                and not c.family.startswith("cross"):
            continue
        cs.append(Candidate(f"SPY版 {c.name}", f"spy-{c.family}", "etf", c.make, symbols=c.symbols,
                            params=c.params, desc=c.desc.replace("米国株", "SPY")))
    assets = ["SPY", "QQQ", "IWM", "EFA", "EEM", "IEF", "TLT", "GLD"]
    for k in (1, 2, 3):
        for lb in (126, 252):
            cs.append(Candidate(f"資産クラス上位{k}（{lb // 21}か月）", "asset-rot", "etf", e_rotation(assets, k, lb, False),
                                max_positions=k, symbols=assets, params={"k": k, "lookback": lb},
                                desc=f"月末に、株・債券・金など8本の{lb // 21}か月リターン上位{k}本（200日線より上のものだけ）"))
    return cs


# ---------------------------------------------------------------- 土俵 stocks（1997年〜、S&P500 個別株）


def s_lowvol(members: pd.DataFrame, top_n: int, window: int = 252):
    """値動きの小さい株（過去1年の日次リターンの標準偏差が小さい順）を月末に top_n 持つ。"""
    def make(data, bench):
        closes = pd.DataFrame({s: d["close"] for s, d in data.items()})
        mem = members.reindex(index=closes.index, columns=closes.columns).fillna(False).astype(bool)
        vol = closes.pct_change().rolling(window, min_periods=200).std()
        score = (-vol).where(mem & closes.notna())
        rank = score.rank(axis=1, ascending=False)
        rebal = month_end(closes.index).to_numpy()[:, None]
        entry = (rank <= top_n) & rebal
        exit_ = (~(rank <= top_n)) & rebal
        return {s: st._frame(entry[s], exit_[s], score[s]) for s in closes}
    return make


def s_mom(members: pd.DataFrame, top_n: int, kind: str = "12-1", exit_rank=None, market_filter=False,
          lookback: int | None = None):
    def make(data, bench):
        if lookback:
            closes = pd.DataFrame({s: d["close"] for s, d in data.items()})
            mem = members.reindex(index=closes.index, columns=closes.columns).fillna(False).astype(bool)
            score = (closes.shift(21) / closes.shift(lookback) - 1).where(mem & closes.notna())
            rank = score.rank(axis=1, ascending=False)
            rebal = month_end(closes.index).to_numpy()[:, None]
            entry = (rank <= top_n) & rebal
            exit_ = (~(rank <= (exit_rank or top_n))) & rebal
            return {s: st._frame(entry[s], exit_[s], score[s]) for s in closes}
        return st.stock_momentum(data, bench, members, top_n=top_n, exit_rank=exit_rank,
                                 market_filter=market_filter, kind=kind)
    return make


def s_swing(members: pd.DataFrame, kind: str, mom_top=None):
    def make(data, bench):
        return st.stock_swing(data, bench, members, kind=kind, mom_top=mom_top)
    return make


def stock_candidates(members: pd.DataFrame) -> list[Candidate]:
    cs: list[Candidate] = []
    for n in (10, 20, 30):
        cs.append(Candidate(f"低ボラ株 上位{n}", "lowvol", "stocks", s_lowvol(members, n), max_positions=n,
                            params={"n": n}, desc=f"月末に、過去1年の値動きが小さい S&P500 株を{n}銘柄持つ"))
    for n in (5, 10, 20):
        cs.append(Candidate(f"モメンタム12-1 上位{n}", "mom12", "stocks", s_mom(members, n), max_positions=n,
                            params={"n": n}, desc=f"月末に、直近1か月を除く12か月の値上がり率上位{n}銘柄"))
        cs.append(Candidate(f"モメンタム6-1 上位{n}", "mom6", "stocks", s_mom(members, n, lookback=126),
                            max_positions=n, params={"n": n}, desc=f"月末に、直近1か月を除く6か月の値上がり率上位{n}銘柄"))
        cs.append(Candidate(f"モメンタム12-1 上位{n}＋相場フィルター", "mom12-f", "stocks",
                            s_mom(members, n, market_filter=True), max_positions=n, params={"n": n},
                            desc=f"12-1 上位{n}。SPY が200日線より下の月末は全部売って現金"))
    cs.append(Candidate("55日ブレイクアウト（5銘柄）", "breakout", "stocks", s_swing(members, "breakout55"),
                        max_positions=5, stop_loss=0.10, desc="55日高値更新で買い、20日安値割れで売り"))
    cs.append(Candidate("20日ブレイクアウト（5銘柄）", "breakout", "stocks", s_swing(members, "breakout20"),
                        max_positions=5, stop_loss=0.08, desc="20日高値更新で買い、10日安値割れで売り"))
    return cs


# ---------------------------------------------------------------- 改良版: 200日線に「遊び」を設ける


def c_sma_band(n: int, lev: str = "L3", band: float = 0.01, off: str = "CASH"):
    """移動平均の上下に band の幅の「遊び」を設ける。

    株の終値が 移動平均×(1+band) を上回ったら lev を持ち、移動平均×(1−band) を下回ったら off に逃げる。
    その間にいる間は、前の状態を続ける。線の近くでの行ったり来たり（ダマシ）と、そのたびの税金を減らす。
    """
    def make(data, bench):
        m = data["MKT"]["close"]
        sm = ind.sma(m, n)
        state = pd.Series(np.where(m > sm * (1 + band), 1.0, np.where(m < sm * (1 - band), 0.0, np.nan)),
                          index=m.index).ffill()
        tgt = pd.Series(np.where(state == 1, lev, off), index=m.index).where(state.notna())
        return targets_to_signals(tgt, pd.Series(True, index=m.index), [lev, off])
    return make


def band_candidates(universe: str) -> list[Candidate]:
    cs = []
    for lev, lname in (("L2", "2倍"), ("L3", "3倍")):
        for n in (150, 200, 250):
            for band in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05):
                cs.append(Candidate(f"{lname} {n}日線 遊び{band * 100:g}%", f"band-{lev}-{n}", universe,
                                    c_sma_band(n, lev, band), symbols=[lev, "CASH"],
                                    params={"n": n, "band": band},
                                    desc=f"株が{n}日線の{band * 100:g}%上を超えたら{lname}、{band * 100:g}%下を割ったら短期国債"))
    return cs


def c_voltarget_monthly_lev(sma_n: int = 200, lo: float = 0.15, hi: float = 0.25):
    """B の改良版: 200日線で逃げる・戻る判定は毎日、倍率（3倍・2倍・1倍）の選び直しは月1回だけ。

    乗り換え（＝売買と税金）を減らすため。
    """
    def make(data, bench):
        m = data["MKT"]["close"]
        idx = m.index
        v = realized_vol(m, 20)
        lev_daily = pd.Series(np.where(v < lo, "L3", np.where(v < hi, "L2", "MKT")), index=idx).where(v.notna())
        me = month_end(idx)
        # 前月の最終営業日の終値で決めた倍率を、翌月の1日目から月末まで使う
        # （＝各日の倍率は「前の月の最終営業日の値動き」で決まる。月末当日はまだ前月分を使う）
        lev = lev_daily.where(me).ffill().shift(1)
        sm = ind.sma(m, sma_n)
        tgt = lev.where(m > sm, "CASH").where(sm.notna() & lev.notna())
        return targets_to_signals(tgt, pd.Series(True, index=idx), ["MKT", "L2", "L3", "CASH"])
    return make


def c_combo(lev: str = "L3", mode: str = "and", n: int = 200, months: int = 12):
    """200日線と時系列モメンタムの組み合わせ（毎日判定）。

    mode="and": 両方が上向きのときだけ lev（守り重視）
    mode="or" : どちらかが上向きなら lev（ダマシで逃げにくい）
    """
    def make(data, bench):
        m, c = data["MKT"]["close"], data["CASH"]["close"]
        sm = ind.sma(m, n)
        k = 21 * months
        a = m > sm
        b = (m / m.shift(k)) > (c / c.shift(k)).reindex(m.index)
        on = (a & b) if mode == "and" else (a | b)
        tgt = pd.Series(np.where(on, lev, "CASH"), index=m.index).where(sm.notna() & m.shift(k).notna())
        return targets_to_signals(tgt, pd.Series(True, index=m.index), [lev, "CASH"])
    return make


def c_sma_trail(n: int = 200, lev: str = "L3", trail: float = 0.25):
    """200日線＋トレーリングストップ（毎日判定）。

    株が n 日線より上なら lev を持つ。ただし、持ち始めてからの lev の最高値から trail 以上下がったら
    短期国債に逃げ、株が一度 n 日線を割ってから再び上抜けるまで戻らない。
    """
    def make(data, bench):
        m = data["MKT"]["close"]
        lv = data[lev]["close"].reindex(m.index)
        sm = ind.sma(m, n)
        above = (m > sm).to_numpy()
        valid = sm.notna().to_numpy()
        lvv = lv.to_numpy()
        out = np.empty(len(m), dtype=object)
        holding, stopped, peak = False, False, 0.0
        for i in range(len(m)):
            if not valid[i]:
                out[i] = None
                continue
            if not above[i]:
                holding, stopped = False, False
            elif not holding and not stopped:
                holding, peak = True, lvv[i]
            if holding:
                peak = max(peak, lvv[i])
                if lvv[i] <= peak * (1 - trail):
                    holding, stopped = False, True
            out[i] = lev if holding else "CASH"
        tgt = pd.Series(out, index=m.index)
        return targets_to_signals(tgt, pd.Series(True, index=m.index), [lev, "CASH"])
    return make


def c_cape_lev(cape: pd.Series, lo_pct: float = 0.5, hi_pct: float = 0.8, window_years: int = 30, sma_n: int = 200):
    """割高さでレバレッジを変える＋200日線。

    前月の CAPE が、その時点までの過去 window_years 年の中で下位 lo_pct なら3倍、hi_pct までなら2倍、
    それより割高なら1倍。株が200日線より下なら短期国債。倍率は月1回だけ選び直す。
    """
    def make(data, bench):
        m = data["MKT"]["close"]
        idx = m.index
        n = window_years * 12
        pct = cape.rolling(n, min_periods=n // 2).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True)
        pct = pct.shift(1)  # 前月の値を使う（公表の遅れの代わり）
        month = idx.to_period("M").to_timestamp()
        p = pd.Series(pct.reindex(month).to_numpy(), index=idx).ffill()
        lev = pd.Series(np.where(p < lo_pct, "L3", np.where(p < hi_pct, "L2", "MKT")), index=idx).where(p.notna())
        sm = ind.sma(m, sma_n)
        tgt = lev.where(m > sm, "CASH").where(sm.notna() & lev.notna())
        return targets_to_signals(tgt, pd.Series(True, index=idx), ["MKT", "L2", "L3", "CASH"])
    return make
