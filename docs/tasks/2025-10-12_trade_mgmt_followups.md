# Trade Management Follow-ups (2025-10-12)

このドキュメントは、トレード管理統合の後続改善タスクをまとめたものです。安全に段階的に進めるため、各タスクは独立して実施可能に分割しています。

## 1) finalize_allocation 既定の見直し

- 目的: finalize_allocation の `include_trade_management` 既定値を `False` に戻し、引数未指定の既存呼び出しが壊れないようにする。
- 現状: 既定が True。market_data_dict/signal_date を渡さない呼び出しで ValueError になり得る。
- 対応:
  - [ ] `core/final_allocation.py` のシグネチャを `include_trade_management: bool = False` に変更。
  - [ ] 呼び出し側のうち、Trade Mgmt が必要な箇所（`scripts/run_all_systems_today.py` など）は明示的に True を渡す（現状そのように指定済み）。
  - [ ] 影響範囲のテストを実行（Quick Test / Full Quality）。
- 受け入れ条件: 既存のツール・テストがエラーなく実行でき、日次パイプラインでトレード管理列が付与される。

## 2) 環境変数アクセスの統一化

- 目的: 直接 `os.environ.get()` を使っている箇所を `config.environment.get_env_config()` 経由へ段階的に移行し、一貫性と安全性を高める。
- 現状: `run_all_systems_today.py` や一部 common/core に直接参照が多数残存。
- 対応（段階的）:
  - [ ] フェーズ A: 日次実行パスの重要箇所のみ（ログ制御・並列制御）を `get_env_config()` へ置換。
  - [ ] フェーズ B: 残りの today-pipeline 関連（today_data_loader、strategy_runner、phase02_basic_data）。
  - [ ] フェーズ C: UI 側（apps/app_today_signals.py）と補助ツール。
  - [ ] 必要に応じて `EnvironmentConfig` に不足プロパティを追加し、docs/technical/environment_variables.md を更新。
- 受け入れ条件: 置換後も挙動が変わらず、テストと Full Quality が PASS。

## 3) UI へのトレード管理列の露出

- 目的: 当日シグナルのテーブルに、エントリー/ストップ/利確/トレーリング/期間などの主要列を表示する。
- 対応案:
  - [ ] `apps/app_today_signals.py` のテーブル生成部で、以下の列を優先的に追加表示（存在する場合のみ）:
    - `entry_type`, `entry_price_final`, `stop_price`, `profit_target_price`, `use_trailing_stop`, `trailing_stop_pct`, `max_holding_days`, `entry_atr`, `risk_per_share`, `total_risk`
  - [ ] 値が NaN の列は自動的に非表示にするか、ツールチップ説明のみ付与。
  - [ ] 列見出しの日本語ラベルとツールチップ文を追加（例: 仕掛け価格・損切・利確目標・トレーリング%・最長保有日数・ATR・1 株あたりリスク・合計リスク）。
- 受け入れ条件: UI 起動時に追加列が表示され（データがある場合）、レイアウト崩れがない。`--compact` モードでもログノイズの増加がない。

## 共通の実施手順

1. 変更後は `Black Format` → `Lint & Format` → `Quick Test Run` → `Full Quality Check` を順に実行。
2. 日次パイプラインのミニ検証: `python scripts/run_all_systems_today.py --test-mode mini --skip-external --benchmark` を実行し、CSV/ログで列追加を確認。
3. 影響範囲が UI のみの場合は Streamlit 起動テストも実施。

## 備考

- 外部ネットワーク呼び出しは追加しないこと。
- 既存の CLI フラグや DEFAULT_ALLOCATIONS を壊さないこと。
- キャッシュ IO は CacheManager 経由を維持。
