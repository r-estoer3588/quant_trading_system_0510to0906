# Copilot Instructions (Condensed)

目的: このリポジトリで AI エージェントが安全かつ即戦力で編集・追加を行うための最小必須知識。過度な抽象説明より「何を / どこで / どう守るか」。

## 1. コア構造 / Entry Points
- UI: `apps/app_integrated.py`（統合タブ） / 当日シグナル UI 補助: `apps/app_today_signals.py`。
- 日次パイプライン: `scripts/run_all_systems_today.py`（8 フェーズ: symbols → load → shared indicators → filters(2-phase) → setup → signals → allocation → save/notify）。
- 戦略分離: ロジック `core/system{1..7}.py` / ラッパ `strategies/system*_strategy.py` / 統合 BT `common/integrated_backtest.py`。

## 2. データキャッシュ階層 (絶対ルール)
- 階層: `rolling`(直近300日, 今日用) → `base`(指標付与長期) → `full_backup`(原本)。
- 取得順: today = rolling→base→full_backup / backtest = base→full_backup。
- 直接 CSV 読み禁止: すべて `common/cache_manager.py::CacheManager` 経由 (Feather 優先, CSV フォールバック)。
- 指標キャッシュ: `data_cache/indicators_systemX_cache/`。

## 3. システム特性 / 不変条件
- ロング: 1,3,4,5 / ショート: 2,6,7。System7 = SPY 固定 (変更禁止)。
- Two-Phase: Filter列判定 → Setup列判定 → ランキング → 配分。
- 主なランキングキー例: S1=ROC200, S2=ADX7, S3=3日下落, S4=RSI4 低, S5=ADX7, S6=6日上昇。
- 配分: スロット/金額制 + `data/symbol_system_map.json`。`DEFAULT_ALLOCATIONS` を壊さない。

## 4. 設定 & 環境
- 優先順位: JSON > YAML > .env (`config/settings.py::get_settings`)。新規出力は `get_settings(create_dirs=True)` が返すパス配下のみ。
- 主要環境例: `COMPACT_TODAY_LOGS`, `ENABLE_PROGRESS_EVENTS`, `ROLLING_ISSUES_VERBOSE_HEAD`。

## 5. 開発ワークフロー (必須コマンド)
```powershell
pip install -r requirements.txt            # 初回
streamlit run apps/app_integrated.py       # UI
python scripts/run_all_systems_today.py --parallel --save-csv
python scripts/run_all_systems_today.py --test-mode mini --skip-external --benchmark  # 2秒高速検証
pytest -q                                   # 決定性テスト
pre-commit run --files <changed_files>
```

## 6. 守るべき禁止事項 / ガードレール
1. Public API / 既存 CLI フラグ / System7 SPY / DEFAULT_ALLOCATIONS を破壊変更しない。
2. 外部ネットワーク呼び出しをテスト経路に追加しない（`--test-mode` + `--skip-external` 互換保持）。
3. キャッシュ直接 I/O 禁止（必ず CacheManager）。
4. 新規巨大依存追加は避け、パフォーマンス影響はベンチマーク (`--benchmark`) で確認。

## 7. 実装パターン
- Two-Phase: `today_filters.py` → Setup ラベル生成 → `today_signals.py` が抽出。
- ログ最適化: `COMPACT_TODAY_LOGS=1` で詳細を DEBUG へ。進捗は `ENABLE_PROGRESS_EVENTS=1` + `logs/progress_today.jsonl`。
- DataFrame 操作は重複列を増やさない (冗長列除去済み方針)。

## 8. コードスタイル / 品質
- snake_case / PascalCase / 型ヒント推奨。`ruff` + `black`。決定性: `common/testing.py` 利用。
- 変更後: fast pipeline mini モード + `pytest -q` を想定。

## 9. 追加時のチェックリスト (PR 前)
- [ ] Cache 経由のみか
- [ ] System7/SPY/CLI フラグへ影響なし
- [ ] mini テスト 2秒パス / pytest パス
- [ ] 新規出力パスは settings 管理下
- [ ] ログ量増加なし / 必要なら COMPACT 対応コメント

不明点や曖昧な規約は PR 説明に背景を記述し合意形成してください。
