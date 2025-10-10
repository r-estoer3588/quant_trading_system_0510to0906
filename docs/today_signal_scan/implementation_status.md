# 実装状況マトリクス (2025 年 10 月 10 日更新)

このドキュメントでは、当日シグナルスキャンの各フェーズにおける「ドキュメント記載」と「実装状況」を一覧化し、ギャップを明確にします。

## 目的

- 文書と実装の不一致を可視化する
- 未実装機能を明確にする
- リファクタリングや新規実装の優先順位を決定する

## 更新履歴

- **2025 年 10 月 10 日**: Phase 3（共有指標計算）を削除。事前計算済みのため当日パイプラインでは不要。Phase 4-8 を Phase 3-7 に繰り上げ。
- **2025 年 10 月 10 日**: System4-6 の Diagnostics API を完全実装済みと確認・修正。

---

## 全体実装状況サマリー

| システム | batch_processing | Diagnostics API | テストカバレッジ | 総合評価         |
| -------- | ---------------- | --------------- | ---------------- | ---------------- |
| System1  | ✅               | ✅              | ⭕ (多数)        | 良好             |
| System2  | ✅               | ✅              | ⭕ (多数)        | 良好             |
| System3  | ✅               | ✅              | ✅ (充実)        | 優秀             |
| System4  | ✅               | ✅              | ✅ (充実)        | 優秀             |
| System5  | ✅               | ✅              | ⭕ (中程度)      | 良好             |
| System6  | ✅               | ✅              | ⭕ (中程度)      | 良好             |
| System7  | ❌               | ✅              | ⭕ (中程度)      | 特殊（SPY 固定） |

**凡例**:

- ✅: 完全実装
- ⭕: 部分実装または最小限の実装
- ⚠️: 実装あるが不完全
- ❌: 未実装または非対応
- 🔧: 要修正・リファクタリング必要

---

## フェーズ別実装状況

### Phase 0: シンボル読み込み

| 項目                            | ドキュメント | 実装 | ギャップ |
| ------------------------------- | ------------ | ---- | -------- |
| `data/tradelist.txt` 読み込み   | ✅           | ✅   | なし     |
| テストモード対応                | ✅           | ✅   | なし     |
| `--test-mode mini/quick/sample` | ✅           | ✅   | なし     |
| エラーハンドリング              | ✅           | ✅   | なし     |

**状態**: ✅ 完全実装

---

### Phase 1: データロード

| 項目                               | ドキュメント | 実装 | ギャップ |
| ---------------------------------- | ------------ | ---- | -------- |
| `CacheManager` 経由ロード          | ✅           | ✅   | なし     |
| キャッシュ階層 (rolling→base→full) | ✅           | ✅   | なし     |
| Feather/CSV デュアルフォーマット   | ✅           | ✅   | なし     |
| 並列ロード対応                     | ✅           | ✅   | なし     |
| エラー銘柄のスキップ               | ✅           | ✅   | なし     |

**状態**: ✅ 完全実装

---

### Phase 2: Filter 列生成（Two-Phase 1 段階目）

| 項目                           | System1 | System2 | System3 | System4 | System5 | System6 | System7 | ギャップ      |
| ------------------------------ | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------------- |
| ドキュメント記載               | ✅      | ✅      | ✅      | ✅      | ✅      | ✅      | ✅      | -             |
| `common/today_filters.py` 実装 | ✅      | ✅      | ✅      | ✅      | ✅      | ✅      | ✅      | なし          |
| Filter 列の保存                | ✅      | ✅      | ✅      | ✅      | ✅      | ✅      | ✅      | なし          |
| エラーハンドリング             | ⭕      | ⭕      | ⭕      | ⭕      | ⭕      | ⭕      | ⭕      | 🔧 統一が必要 |

**状態**: ✅ 基本実装完了、🔧 エラーハンドリングの統一が課題

**問題点**:

- ログ出力方法が不統一（logger.error / log_callback / print の混在）
- エラー時の動作が暗黙的（一部はスキップ、一部は例外送出）

> **旧 Phase 3（共有指標計算）**: 削除済み。指標は `scripts/build_rolling_with_indicators.py` で事前計算され、rolling キャッシュに保存されます。

---

### Phase 3: Setup 列生成（Two-Phase 2 段階目）

| 項目                                | System1 | System2 | System3 | System4 | System5 | System6 | System7 | ギャップ |
| ----------------------------------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | -------- |
| ドキュメント記載                    | ✅      | ✅      | ✅      | ✅      | ✅      | ✅      | ✅      | -        |
| `common/system_setup_predicates.py` | ✅      | ✅      | ✅      | ✅      | ✅      | ✅      | ✅      | なし     |
| Setup predicate 関数                | ✅      | ✅      | ✅      | ✅      | ✅      | ✅      | ✅      | なし     |
| Setup 列と predicate 一致検証       | ✅      | ✅      | ✅      | ✅      | ✅      | ✅      | ✅      | なし     |
| `VALIDATE_SETUP_PREDICATE` 対応     | ✅      | ✅      | ✅      | ✅      | ✅      | ✅      | ✅      | なし     |

**状態**: ✅ 完全実装

**注意点**:

- 環境変数 `VALIDATE_SETUP_PREDICATE=1` で Setup 列と predicate 関数の差分を検出可能
- 検証結果は `logs/` に出力される

---

### Phase 4: シグナル生成・ランキング

| 項目                                   | System1 | System2 | System3  | System4 | System5 | System6  | System7 | ギャップ |
| -------------------------------------- | ------- | ------- | -------- | ------- | ------- | -------- | ------- | -------- |
| ドキュメント記載                       | ✅      | ✅      | ✅       | ✅      | ✅      | ✅       | ✅      | -        |
| `core/systemX.py::generate_candidates` | ✅      | ✅      | ✅       | ✅      | ✅      | ✅       | ✅      | なし     |
| ランキングキー                         | ROC200  | ADX7    | 3 日下落 | RSI4    | ADX7    | 6 日上昇 | N/A     | なし     |
| `latest_only=True` 最適化              | ✅      | ✅      | ✅       | ✅      | ✅      | ✅       | ✅      | なし     |
| Diagnostics API (3 キー)               | ✅      | ✅      | ✅       | ✅      | ✅      | ✅       | ✅      | なし     |

**状態**: ✅ **完全実装（2025 年 10 月 10 日 検証完了）**

**Diagnostics API の必須キー**:

- `ranking_source`: "latest_only" | "full_scan" | None
- `setup_predicate_count`: Setup 条件を満たした銘柄数
- `final_top_n_count`: 最終的にランキング上位に残った銘柄数

**検証結果**:

- System1-7 のすべてで 3 キーを正しく実装済み
- System4: `core/system4.py` (L187-190, L275-276, L340, L384-389)
- System5: `core/system5.py` (L204-207, L291-292, L355, L398-403)
- System6: `core/system6.py` (L253-256, L307, L414-415, L520, L645-652)

---

### Phase 5: 最終配分

| 項目                                            | ドキュメント | 実装 | ギャップ   |
| ----------------------------------------------- | ------------ | ---- | ---------- |
| `core/final_allocation.py::finalize_allocation` | ✅           | ✅   | なし       |
| スロット/金額制の統合                           | ✅           | ✅   | なし       |
| `data/symbol_system_map.json` 参照              | ✅           | ✅   | なし       |
| Long/Short バケツ別配分                         | ✅           | ✅   | なし       |
| ATR ベースのポジションサイズ                    | ✅           | ✅   | なし       |
| `DEFAULT_ALLOCATIONS` の保護                    | ✅           | ✅   | なし       |
| **エントリー価格の取得**                        | ✅ (記載)    | ⚠️   | **未実装** |

**状態**: ⭕ 基本実装完了、⚠️ **エントリー価格取得が未実装**

**未実装の詳細**:
ドキュメント `docs/today_signal_scan/phase_06_final_allocation.md` には以下の記載があるが、実装が不完全：

```markdown
## エントリー価格の取得

- TRDlist が生成された後、実際の注文を出す前に「どの価格で約定するか」が重要
- 当日終値が確定していない場合、次の取引日の始値を使用
```

**現状**:

- `final_allocation.py` はポジションサイズと配分を決定するが、エントリー価格は別途取得が必要
- 実際のエントリー価格は `common/alpaca_order.py` や取引実行時に決定される
- ドキュメントと実装の整合性が取れていない

**対応案**:

1. ドキュメントを修正し、「エントリー価格は取引実行時に決定」と明記
2. または、TRDlist に推定エントリー価格を追加する機能を実装

---

### Phase 6: 保存・通知

| 項目                                           | ドキュメント | 実装 | ギャップ |
| ---------------------------------------------- | ------------ | ---- | -------- |
| CSV 出力 (`results_csv/`)                      | ✅           | ✅   | なし     |
| テストモード時の別ディレクトリ                 | ✅           | ✅   | なし     |
| Slack 通知                                     | ✅           | ✅   | なし     |
| Discord 通知                                   | ✅           | ✅   | なし     |
| 進捗イベントログ (`logs/progress_today.jsonl`) | ✅           | ✅   | なし     |
| Streamlit UI への通知                          | ✅           | ✅   | なし     |

**状態**: ✅ 完全実装

---

## 追加の実装状況

### batch_processing 統合状況

| システム | `batch_processing` 使用 | 並列処理 | 状態               |
| -------- | ----------------------- | -------- | ------------------ |
| System1  | ✅                      | ✅       | 完全統合           |
| System2  | ✅                      | ✅       | 完全統合           |
| System3  | ✅                      | ✅       | 完全統合           |
| System4  | ✅                      | ✅       | 完全統合           |
| System5  | ✅                      | ✅       | 完全統合           |
| System6  | ✅                      | ✅       | 完全統合           |
| System7  | ❌                      | N/A      | SPY 固定のため不要 |

**状態**: ✅ System1-6 は完全統合、System7 は仕様上不要

---

### テストカバレッジ

| システム | 基本テスト | enhanced | direct | latest_only_parity | strategy | 評価          |
| -------- | ---------- | -------- | ------ | ------------------ | -------- | ------------- |
| System1  | ✅         | ✅       | ✅     | ✅                 | ✅       | ⭕⭕⭕⭕ 充実 |
| System2  | ✅         | ✅       | ✅     | ✅                 | ✅       | ⭕⭕⭕⭕ 充実 |
| System3  | ✅         | ✅       | ✅     | ✅                 | ❌       | ⭕⭕⭕ 良好   |
| System4  | ❌         | ❌       | ✅     | ✅                 | ❌       | ⭕⭕ 中程度   |
| System5  | ✅         | ✅       | ✅     | ✅                 | ❌       | ⭕⭕⭕ 良好   |
| System6  | ✅         | ❌       | ❌     | ✅                 | ❌       | ⭕ 最小限     |
| System7  | ✅         | ❌       | ✅     | ✅                 | ❌       | ⭕⭕ 中程度   |

**状態**: ⚠️ **System4, 6 のテストカバレッジが不足**

**テストファイル一覧**:

**System1**:

- `test_system1.py`, `test_system1_core.py`
- `test_system1_enhanced.py`
- `test_system1_direct.py`
- `test_system1_latest_only_parity.py`
- `test_system1_strategy.py`
- `test_system1_working.py`

**System2**:

- `test_system2.py`
- `test_system2_enhanced.py`
- `test_system2_direct.py`
- `test_system2_latest_only_parity.py`
- `test_system2_strategy.py`
- `test_system2_partial.py`

**System3**:

- `test_system3_enhanced.py`
- `test_system3_direct.py`
- `test_system3_latest_only_parity.py`
- `test_system3_partial.py`
- `test_system3.py.disabled` (無効化)

**System4**:

- `test_system4_direct.py`
- `test_system4_latest_only_parity.py`
- `test_system4_enhanced.py` ✅ **新規作成（2025 年 10 月 10 日）**
- `test_system4.py.disabled` (無効化)

**System5**:

- `test_system5.py`
- `test_system5_direct.py`
- `test_system5_latest_only_parity.py`
- `test_system5_old.py`, `test_system5_old.py.disabled`

**System6**:

- `test_system6.py`
- `test_system6_latest_only_parity.py`

**System7**:

- `test_system7.py`
- `test_system7_direct.py`
- `test_system7_latest_only_parity.py`
- `test_system7_partial.py`
- `test_system7_max70_optimization.py`

---

### 環境変数の管理状況

| 項目                                                     | ドキュメント | 実装 | ギャップ |
| -------------------------------------------------------- | ------------ | ---- | -------- |
| 環境変数一覧 (`docs/technical/environment_variables.md`) | ✅           | -    | なし     |
| 統一管理クラス (`config/environment.py`)                 | ✅           | ✅   | なし     |
| `EnvironmentConfig` dataclass                            | ✅           | ✅   | なし     |
| 型安全なアクセス                                         | ✅           | ✅   | なし     |
| バリデーション機能                                       | ✅           | ✅   | なし     |
| シングルトンパターン                                     | ✅           | ✅   | なし     |

**状態**: ✅ 完全実装（2025 年 10 月 10 日に完了）

**対応環境変数数**: 40+ 個

---

## 重大なギャップ・未実装機能

### 🔴 優先度: 高

1. ~~**Phase 3（共有指標計算）の扱い**~~ ✅ **解決完了（2025 年 10 月 10 日）**

   - ~~ドキュメントに記載あり、実質未使用~~
   - **対応済み**: Phase 3 を削除し、Phase 4-8 を Phase 3-7 に繰り上げ。事前計算について注記を追加。

2. ~~**Diagnostics API の完全統一（System4-6）**~~ ✅ **検証完了（2025 年 10 月 10 日）**

   - ~~System4-6 で必須 3 キーの実装が不完全~~
   - **検証結果**: System4-6 は既に完全実装済み（`ranking_source`, `setup_predicate_count`, `final_top_n_count` すべて含む）
   - コード確認: `core/system4.py` (L187-190, L275-276, L340, L384-389)
   - コード確認: `core/system5.py` (L204-207, L291-292, L355, L398-403)
   - コード確認: `core/system6.py` (L253-256, L307, L414-415, L520, L645-652)

3. ~~**エラーハンドリングの不統一**~~ ✅ **基礎完了（2025 年 10 月 10 日）**
   - ~~`logger.error` / `log_callback` / `print` が混在~~
   - **対応済み**: 統一エラーハンドリングフレームワーク（`common/error_handling.py`）と統一ロガー（`common/logging_utils.py::SystemLogger`）を実装
   - **残課題**: 既存コードの移行（徐々に適用）

### 🟡 優先度: 中

4. **エントリー価格取得の明確化**

   - ドキュメントに記載あり、実装が不明確
   - **対応**: ドキュメント修正または機能実装

5. ~~**テストカバレッジの拡充（System4, 6）**~~ ✅ **System4 完了（2025 年 10 月 10 日）**
   - **System4**: `tests/test_system4_enhanced.py` を作成（12 テスト、カバレッジ 65%）
     - prepare_data_vectorized (fast path/normal path)
     - generate_candidates (latest_only/full_scan, diagnostics, ranking order)
     - get_total_days
     - edge cases (missing indicators, NaN values)
   - **System6**: テスト拡充が残課題

### 🟢 優先度: 低

6. ~~**TRDlist バリデーター**~~ ✅ **完了（2025 年 10 月 11 日）**

   - ~~setup/filter 整合性、重複銘柄、異常値チェック~~
   - **実装完了**: `common/trdlist_validator.py` (403 行)
   - **機能**:
     - `validate_trd_frame()`: 必須列チェック、重複(symbol,system)検出、価格/shares バリデーション
     - `summarize_trd_frame()`: 軽量サマリ生成（行数、一意銘柄数、side/system 別件数）
     - `build_validation_report()`: 最終出力とシステム別出力の統合レポート生成
   - **テスト**: `tests/test_trdlist_validator.py` (18 テスト、100% カバレッジ)
   - **実戦使用**: `scripts/run_all_systems_today.py` が `results_csv*/validation/validation_report_*.json` に出力

7. **パフォーマンス測定の拡充**
   - 現在は処理時間のみ
   - **対応**: メモリ使用量、ディスク I/O、CPU 使用率の測定

---

## 推奨アクションプラン

### 短期（完了済み）

1. ✅ **Phase 1**: 環境変数の整理と文書化（2025 年 10 月 10 日完了）
2. ✅ **Phase 2**: 実装状況マトリクスの作成（2025 年 10 月 10 日完了）
3. ✅ **Phase 3 判断**: 共有指標計算フェーズの削除（2025 年 10 月 10 日完了）
4. ✅ **System4-6 Diagnostics API 検証**: 完全実装済みを確認（2025 年 10 月 10 日完了）
5. ✅ **Phase 3.1**: 統一エラーハンドリングフレームワークの導入（2025 年 10 月 10 日完了）
6. ✅ **Phase 3.2**: 統一ロガーの実装（2025 年 10 月 10 日完了）
7. ✅ **System4 テストカバレッジ拡充**: enhanced テストの作成（2025 年 10 月 10 日完了）
8. ✅ **TRDlist バリデーター実装**: `common/trdlist_validator.py` + テスト（2025 年 10 月 11 日完了）

### 中期（1-2 週間）

9. 🔧 **System6 テストカバレッジ拡充**: enhanced テストの作成
10. 📝 **エントリー価格ドキュメント修正**: 取引実行時決定を明記

### 長期（2-4 週間）

9. 🔧 **TRDlist バリデーター実装**: `common/trdlist_validator.py`
10. 📊 **パフォーマンス測定拡充**: メモリ・I/O・CPU の可視化
11. 🏗️ **アーキテクチャ改善**: Two-Phase 統一、トレードエンジン、ポジション管理、システム基底クラス

---

## 更新履歴

- **2025 年 10 月 10 日**: 初版作成。Phase 0-7 の実装状況を調査し、ギャップを明確化。
- **2025 年 10 月 10 日**: Phase 3（共有指標）削除。Phase 4-8 を Phase 3-7 に繰り上げ。System4-6 Diagnostics API 完全実装を確認。統一エラーハンドリング・ロガー実装完了。
