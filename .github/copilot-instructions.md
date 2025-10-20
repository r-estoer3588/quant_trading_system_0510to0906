# Copilot Instructions (Condensed)

目的: このリポジトリで AI エージェントが安全かつ即戦力で編集・追加を行うための最小必須知識。過度な抽象説明より「何を / どこで / どう守るか」。

## 0. 必須参照ドキュメント (Always Reference)

**最重要**: 質問・回答・編集前に必ず `docs/README.md` を参照すること。プロジェクト全体の構造・システム仕様・技術詳細・運用ガイドが統合された包括的なナビゲーションハブ。相互参照リンクで関連文書への効率的アクセスが可能。

- **統合目次**: [docs/README.md](docs/README.md) - 4 分野（クイックスタート・システム概要・技術文書・運用ガイド）
- **システム仕様**: [docs/systems/](docs/systems/) - System1-7 詳細仕様
- **技術詳細**: [docs/technical/](docs/technical/) - アーキテクチャ・指標・MCP 統合
- **処理フロー**: [docs/today_signal_scan/](docs/today_signal_scan/) - 8 フェーズ詳細
- **運用手順**: [docs/operations/](docs/operations/) - 自動実行・通知・監視

## 1. コア構造 / Entry Points

```markdown
# Copilot instructions — repository quick reference

目的: AI エージェントがこのリポジトリで速やかに安全に編集できるよう、"必ず守ること" と "すぐ役立つ局所ルール" を短くまとめる。

## 必読（最初に開く）

- `docs/README.md` — システム設計・運用・コマンドの統合ハブ。編集前は必ず参照。

## 主要エントリとデータフロー（一目で）

- UI: `apps/app_integrated.py` / `apps/app_today_signals.py` (Streamlit)
- 日次パイプライン: `scripts/run_all_systems_today.py`（symbols → load → indicators → filters → setup → signals → allocation → save/notify）
- 戦略分離: 低レベルロジックは `core/systemX.py`、Strategy は `strategies/systemX_strategy.py` に置く。

## 絶対守るルール（短い）

- キャッシュ I/O は常に `common/cache_manager.py::CacheManager` 経由。直接 `pd.read_csv(data_cache/...)` 禁止。
- System7 は SPY 固定。`core/system7.py` を改変しない。
- 設定は `config/settings.py::get_settings()` 経由で作成・参照。環境変数は `config/environment.py::get_env_config()` を使う（直接 `os.environ.get()` 禁止）。

## デバッグ & 診断（すぐ使える）

- 診断スナップショット: `results_csv_test/diagnostics_test/diagnostics_snapshot_*.json` を参照。
- デバッグツール: `tools/debug_finalize_allocation.py`（finalize の再現）、`scripts/investigate_entry_issue.py`（調査自動化）。
- UI キャプチャ: `tools/capture_ui_screenshot.py` / `tools/run_and_snapshot.ps1` と Playwright の `e2e/` テストを参照。

## 開発コマンド（頻出）

- 初回依存: `pip install -r requirements.txt`
- UI: `streamlit run apps/app_integrated.py`
- 当日一式実行（本番と同様）: `python scripts/run_all_systems_today.py --parallel --save-csv`
- 速い検証: `python scripts/run_all_systems_today.py --test-mode mini --skip-external --benchmark`
- テスト: `pytest -q`（`--maxfail=1` などを併用可）

## パターンと注意点（具体例）

- Two‑Phase パターン: `common/today_filters.py` → `common/system_setup_predicates.py` → `core/systemX.py` → `strategies/*`。
- Allocation contract: `core/final_allocation.py::finalize_allocation(per_system, strategies=None, symbol_system_map=None)`。
  - 期待: callers は可能な限り `strategies` と `symbol_system_map` を渡す。未指定時はファイルからフォールバックするが、CI では `ALLOCATION_REQUIRE_STRATEGIES=1` を推奨。
- キャッシュインデックス: Feather の DatetimeIndex 必須（`docs/technical/cache_index_requirements.md`）。

## PR 前チェック（必須）

- キャッシュ経由か、System7/SPY を壊していないか、`get_settings()` を経由した出力先かを確認。
- `--test-mode mini` で短時間で動作確認 → `pytest -q` を通す。
- 新規環境変数追加は `config/environment.py` と `docs/technical/environment_variables.md` に追記。

---

質問や曖昧な要件があれば、どのファイルに対する変更かと候補となるコマンド（例: `python scripts/run_all_systems_today.py --test-mode mini`）を教えてください。簡潔に追加修正します。
```
