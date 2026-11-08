# trade-bot

moomoo 証券で動かすトレード bot の検証と、厳選した bot の仕様。検証 → ペーパー運用 → 少額の本番、の順に進める。

## いまの段階: 厳選した bot を Muse に引き継ぎ

約300の候補を、手数料と日本の特定口座の税金を引いたうえで評価し、採用2つ・実験枠2つに絞った。ペーパー運用は Muse が行う。

| bot | 中身 | 税引き後の年率（2003年〜、元本 $2,000） |
|---|---|---|
| A（採用） | SPY が200日線より上なら UPRO（S&P500 の3倍）、下なら BIL（短期国債） | 14.8%（持ち続け 9.7%） |
| B（採用） | A と同じ判定で逃げ、持つときは UPRO の比率を「15% ÷ 値動き」にして残りを BIL | 13.8%（最大下落は A より浅い） |
| B′（候補） | B の 15% を 20% にした版 | 15.2% |

毎日の判定と注文は `tradebot/handoff.py` の `daily_plan` 1つで出せる。

| 読むもの | 中身 |
|---|---|
| `docs/HANDOFF_MUSE.md` | 仕様書（結論、選び方、bot ごとのルール、税金、本番の注意） |
| `docs/MUSE_BRIEF.md` | Muse への依頼文 |
| `docs/MANUAL.md` | 手で回すときの手順 |
| `reports/improvements.md` | 改良の記録（試して採用したこと・やめたこと、61項目、冒頭に目次） |
| `reports/zoo.md` | 全候補の評価表 |
| `reports/outlook.md` | 10年運用したときの成績の幅 |
| [検証ダッシュボード](https://claude.ai/artifact/8ZxiVQjhXMtTeKDkj8EWtN) | 主な結果のグラフ（開くには共有が必要） |

## よく使うコマンド

```bash
pip install -e ".[dev]"
python scripts/signal_today.py          # 今日の判定（A・B・C、--momentum で D も）
python scripts/run_zoo.py               # 全候補の評価（元本 $2,000、reports/zoo.md）
python scripts/run_outlook.py           # 10年運用の幅（reports/outlook.md）
python scripts/run_contrib.py           # 毎月積み立てた場合（reports/contrib.md）
python scripts/run_bweights.py          # B の持ち方の比較（reports/b_weights.md）
python scripts/replay_handoff.py         # 参照実装を過去の値段で毎日動かしてバックテストと比べる
python scripts/export_dashboard.py DIR  # ダッシュボード用の CSV
python -m pytest                        # テスト
ruff check tradebot scripts tests       # 静的解析
```

push のたびに、GitHub Actions でテストと静的解析が走る（`.github/workflows/test.yml`）。

## 検証の前提

- 価格: Yahoo Finance の調整済み日足（`data/` にキャッシュ、コミットしない）。約100年の米国株全体は Fama/French と Shiller のデータ
- 費用: 手数料 0.132%（上限 $22）、スリッページ 0.05%。レバレッジ ETF は借入コスト（短期金利＋0.5%）と信託報酬 0.9% を引いて合成
- 税金: 特定口座。売買益 20.315%（年ごとに通算、損失は3年繰り越し）、配当 約28%。最終日に全部売った手取りで比べる
- 約定: シグナルの翌日。実物の ETF は始値、合成のレバレッジ ETF は終値（推奨する発注は「翌営業日の引け間際」）

## 構成

| ファイル | 役割 |
|---|---|
| `tradebot/handoff.py` | 引き継ぐ bot の売買判断（Muse が移植する参照実装） |
| `tradebot/zoo.py` | 評価する候補の一覧と、そのシグナル |
| `tradebot/backtest.py` | ポートフォリオ単位のバックテスト（手数料・税金・損失の繰り越し・損出し・積み立て） |
| `tradebot/weights.py` | 複数の銘柄を比率で持つ型のバックテスト（B の持ち方） |
| `tradebot/longhist.py` | 約100年分のデータ（米国株全体、短期国債、10年国債、CAPE）とレバレッジの合成 |
| `tradebot/strategies.py` | 基本の戦略（移動平均、モメンタム、ブレイクアウトなど） |
| `tradebot/data.py`、`tradebot/universe.py` | 日足の取得、S&P500 の各時点の構成銘柄 |
| `tradebot/live.py`、`scripts/live.py` | 旧ペーパー bot（2026-11-07 に停止） |

## 秘密情報

口座番号や API の設定は `.env` に置き、コミットしない。
