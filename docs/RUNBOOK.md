# 毎日の売買手順（Claude の定期実行用）

ペーパー運用の bot を動かす手順。平日のニューヨーク時間 12:13 に起動される。

- スイング（55日高値ブレイクアウト）: 取引日は毎日判断する
- 月次の2戦略（個別株モメンタム上位5、暴落回避）: 月の最初の取引日だけ入れ替える
- 祝日や、その日の判断が済んでいる日は、`plan` が「何もしない」と言って終わる

## 守ること

- 使うのは moomoo の**シミュレーション口座 `1459674`（米国株、market=100）だけ**。`trading_order_*`（本番口座）は絶対に使わない
- 口座には bot 以外の建玉（SPY、QQQ など）もある。**帳簿 `state/ledger.json` にない銘柄・数量は売らない**
- 計画にない注文は出さない。おかしいと思ったら発注せずに止めて、理由を報告する

## 手順

1. リポジトリを用意する
   - `/home/user/trade-bot` がなければ、`add_repo`（owner `wataaass`, repo `trade-bot`, access `push`）で追加して clone する
   - ブランチ `claude/effort-estimation-z1y12z` を checkout して pull する
   - `pip install -e ".[dev]"`
2. 計画を作る: `python scripts/live.py plan`
   - 「何もしない」と出たら、そこで終わり（報告も不要）
   - 計画は `state/plans/YYYY-MM-DD.json` に書かれる。注文が0件でも手順5の `done` は実行する
3. 発注する（売りを全部先に、買いを後に）。`mcp__moomoo__sim_trade_input_order` を使う
   - `acc_id="1459674"`, `market=100`, `order_type=3`（成行）, `order_trade_time_type=1`（通常取引時間のみ）
   - `order_side`: 買い `1`、売り `2`
   - `text`: `trade-bot <strategy> <YYYY-MM>`
   - 売りの前に、帳簿の数量以下であることを確かめる
4. 約定を確かめる: `mcp__moomoo__sim_trade_history_order_list`（`acc_id="1459674"`, `market=100`）
   - `status=4` が約定済み。`avg_fill_price` と `cum_qty` を使う。`status=2` は注文中なので、少し待ってから見直す
   - 約定したものだけ記録する。手数料は本番の想定で `min(数量×価格×0.00132, 22)` を入れる
   - `python scripts/live.py record --strategy <momentum|trend|swing> --side <BUY|SELL> --symbol <銘柄> --qty <数量> --price <約定価格> --fee <手数料>`
   - 約定しなかった注文があれば、記録せずに報告する
5. 締める
   - `python scripts/live.py done`
   - `python scripts/live.py status`（`reports/live.md` が更新される）
   - `python -m pytest -q`
   - `state/` と `reports/live.md` をコミットして push する
6. 報告する（日本語、短く）: 売買した銘柄、各戦略と SPYM 持ち続けの損益、気になった点。注文が0件の日は報告しない
