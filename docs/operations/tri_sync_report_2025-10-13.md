---
title: フル実行 tri-sync 結果レポート（2025-10-13）
date: 2025-10-13
---

## 概要

- 対象日: 2025-10-13（基準日: 2025-10-10）
- 対象銘柄: フィルタ後 6126–6132（実行により微差）
- 実行モード: latest_only（デフォルト）
- 実行形態:
  - 非並列フル実行（--parallel なし）
  - 並列フル実行（--parallel）

## 結果サマリ

非並列（開始 16:48:12 → 終了 17:03:05, 総実行 891.61s）

- 最終候補: 合計 38（system1=10, system2=10, system4=10, system5=8）
- 出力:
  - CSV: data_cache/signals/signals_final_2025-10-13.csv（38 行）
  - 検証: results_csv/validation/validation_report_2025-10-13.json（errors=0, warnings=1: 重複シンボル情報）
  - ベンチ: results_csv/benchmark_None_20251013_170305.json
  - ログ: logs/today_signals_20251013_1648.log / logs/progress_today.jsonl

並列（開始 18:11:49 → 終了 18:26:39, 総実行 888.93s）

- 最終候補: 合計 38（system1=10, system2=10, system4=10, system5=8）
- 出力:
  - CSV: data_cache/signals/signals_final_2025-10-13.csv（38 行）
  - 検証: results_csv/validation/validation_report_2025-10-13.json（errors=0, warnings=1）
  - ベンチ: results_csv/benchmark_None_20251013_182639.json
  - ログ: logs/today_signals_20251013_1811.log / logs/progress_today.jsonl

補足（phase4 性能）

- 非並列: phase4_signal_generation=824.08s
- 並列: phase4_signal_generation=820.79s

## tri-sync（3 点同期）検証

- パイプライン完了: OK（pipeline_complete=true）
- システム別候補（イベント/スクショ集計）:
  - system1=10, system2=10, system3=0, system4=10, system5=10（抽出候補）, system6=0, system7=0
- CSV 最終採用との差分:
  - system5: 抽出候補 10 → 最終採用 8（配分段階フィルタで 2 件落選）
- 整合性チェック:
  - 配分合計 == JSONL 合計: OK
  - diagnostics ranked_top_n_count とイベント候補数: OK
  - フル実行 diagnostics_mode: 空（OK）
  - マッチ率: 6/7 システムで完全一致（system5 の差は既知・合理的）

tri-sync 生成物

- サマリ JSON: screenshots/progress_tracking/sync_summary.json
- 詳細 JSON: screenshots/progress_tracking/sync_analysis.json
- レポート: screenshots/progress_tracking/ANALYSIS_REPORT.md

## 既知の注意点

- system5 は最終配分での制約（資金・スロット・重複排除等）により、抽出候補と最終採用に小差が出る場合がある。
- system1 の latest_only 区間は重く、ログ心拍が粗く見えることがある（無音に見えても処理は継続）。

## 次の改善案（任意）

- system1 候補生成の進捗ログをもう一段階細かく（対象件数 N ごとに心拍）
- perf snapshot の定期採取（phase4 の内訳把握・回帰検知）
- 並列度の調整オプション（CPU/IO に合わせたワーカー制御）
