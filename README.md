# trade-bot

moomoo 証券で動かすスイングトレード bot。検証 → ペーパー運用 → 少額の本番、の順に進める。

## いまの段階: 厳選した bot を Muse に引き継ぎ

- 仕様書: `docs/HANDOFF_MUSE.md`（約300候補から採用2・実験枠2）
- 売買判断の参照実装: `tradebot/handoff.py`
- 全候補の評価: `reports/zoo.md`（`python scripts/run_zoo.py` で再現、元本 $2,000）
- 改良の記録（試して採用したこと・やめたこと）: `reports/improvements.md`
- 10年運用したときの幅: `reports/outlook.md`（`python scripts/run_outlook.py`）
- 今日の判定: `python scripts/signal_today.py`
- Claude 側のペーパー bot（`scripts/live.py`）は 2026-11-07 に停止

## 検証の仕組み

```bash
pip install -e ".[dev]"
python scripts/run_backtest.py   # 結果は reports/backtest.md
python -m pytest
```

- 過去データ: Yahoo Finance の調整済み日足（`data/` にキャッシュ、コミットしない）
- 約定: シグナルの翌日の始値。手数料 0.132%（上限 $22）とスリッページ 0.05% を引く
- 期間: 2007〜2018 で型を確かめ、2019〜 は結果を見るだけの期間にする

## 構成

| ファイル | 役割 |
|---|---|
| `tradebot/data.py` | 日足の取得とキャッシュ |
| `tradebot/indicators.py` | 移動平均、RSI |
| `tradebot/strategies.py` | 戦略の候補 |
| `tradebot/backtest.py` | ポートフォリオ単位のバックテストと成績指標 |
| `scripts/run_backtest.py` | 候補の一括比較 |

## 秘密情報

口座番号や API の設定は `.env` に置き、コミットしない。
