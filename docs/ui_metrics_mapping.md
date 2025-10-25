## UI 指標とコード参照（Tgt / FILpass / STUpass / TRDlist / Entry / Exit）

このドキュメントは、Streamlit UI（`apps/app_today_signals.py`）のシステム別メトリクス行に表示される各項目がどの処理段階で計算され、どのコードが担当しているかをまとめた短い参照です。

- Tgt

  - 意味: ユニバースの目標件数（表示上のターゲット）
  - 算出元: `GLOBAL_STAGE_METRICS` スナップショットの `target`、もしくは `filter_pass` のフォールバック
  - 参照コード: `apps/app_today_signals.py::StageTracker._apply_snapshot` / `StageTracker.set_universe_target`

- FILpass (最新)

  - 意味: Phase2 相当の事前フィルター通過銘柄数（最新行ベース）
  - 算出元: prepare 層での prefilter 集計（最新行）
  - 参照コード: `common/today_signals.py::_compute_filter_pass`（`compute_today_signals` 実行中）
  - UI 表示箇所: `apps/app_today_signals.py::StageTracker._render_metrics`

- STUpass (最新)

  - 意味: セットアップ条件が最新行で成立している銘柄数（最新行ベース）
  - 算出元: prepare 層の最新行集計（例: system3 の `drop3d`閾値など）
  - 参照コード: `common/today_signals.py::_compute_setup_pass`
  - 備考: 履歴（過去日に `setup==True` だった）とは別に扱われます。

- TRDlist (候補)

  - 意味: 各システムで生成された候補リストの件数（`strategy.generate_candidates` の出力）
  - 算出元: 各システムの `generate_candidates` 実行結果（選択された候補日ベース）
  - 参照コード: 各システム実装（例: `core/system3.py::generate_candidates_system3`）、集約は `compute_today_signals` → `per_system` に格納
  - UI 表示箇所: `apps/app_today_signals.py::StageTracker.finalize_counts` が `per_system` を参照して `cand` を設定

- Entry (配分後)

  - 意味: 最終的に配分（allocation）されるエントリー件数（`AllocationSummary.final_counts` を優先）
  - 算出元: `core/final_allocation` が返す `AllocationSummary.final_counts`。なければ `final_df['system']` のカウントを使用
  - 参照コード: `core/final_allocation.py`（AllocationSummary 作成）、`apps/app_today_signals.py::_interpret_compute_today_result` と `StageTracker.finalize_counts`

- Exit
  - 意味: 本日予定の手仕舞い（Exit）件数（保有ポジションの解析結果）
  - 算出元: `analyze_exit_candidates` による保持ポジション解析
  - 参照コード: `apps/app_today_signals.py::analyze_exit_candidates` / `_evaluate_position_for_exit`

注意: UI の表示は複数の段階（prepare / generate / allocation）を並列に表示しています。特に `STUpass` は「最新行ベース」で算出されるため、過去に `setup` が成立していた履歴（prepare 層の任意日）とは異なる値になります。履歴情報を同時表示したい場合は、追加の列やトグル（履歴表示モード）を検討してください。
