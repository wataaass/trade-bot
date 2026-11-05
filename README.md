# trade-bot

moomoo 証券で動かすスイングトレード bot。検証 → ペーパー運用 → 少額の本番、の順に進める。

## いまの段階: 検証

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
