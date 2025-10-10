# SPY/取引日ユーティリティ

`common/utils_spy.py` には SPY データ取得および取引日（NYSE）関連の補助関数が実装されています。ドキュメント外だった機能も含め、要点を以下に整理します。

## 主要関数

- get_spy_data_cached_v2(mode="backtest"|"today")
  - SPY.csv をキャッシュから読み込みます。
  - 検索順序: backtest=base→full_backup, today=rolling→base→full_backup。
- get_spy_with_indicators(spy_df=None)
  - SPY に SMA100/200 を付与し、必要に応じて `data_cache` へ反映します。
- get_latest_nyse_trading_day(today)
  - 引数日付以下で最新の営業日を返します。
- get_next_nyse_trading_day(current)
  - 引数日付の翌営業日を返します。
- get_signal_target_trading_day(now=None)
  - 当日シグナル抽出で基準とする日付（通常は直近営業日、終値後は翌営業日）を返します。
- calculate_trading_days_lag(cache_date, target_date)
  - 営業日ベースでのラグ日数を返します。キャッシュが古い場合の許容判定で使用。
- resolve_signal_entry_date(base_date)
  - シグナル日から翌営業日（エントリー予定日）を返します。

## 実装メモ

- pandas-market-calendars を利用して NYSE カレンダーを参照します。
- UI 実行時は Streamlit 経由で簡易メッセージを表示します（CLI では無害化）。
- 列名の大小文字ゆらぎや MultiIndex 列に対して防御的に正規化しています。

## 使用例

- System3 の latest_only 処理で、`calculate_trading_days_lag` と `resolve_signal_entry_date` を利用。
- `scripts/run_all_systems_today.py` でターゲット日付と最新キャッシュ日付の差分評価に活用。
