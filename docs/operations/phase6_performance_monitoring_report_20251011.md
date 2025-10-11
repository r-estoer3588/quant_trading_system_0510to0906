# Phase 6: パフォーマンス測定拡充 - 実装報告

**実施日時**: 2025 年 10 月 11 日 14:00 - 14:30  
**実施内容**: メモリ・CPU・ディスク I/O 詳細測定機能の追加

---

## ✅ 実装内容

### 1. PerformanceMonitor クラスの実装

**新規ファイル**: `common/performance_monitor.py` (約 360 行)

#### 主な機能

1. **リソーススナップショット**:

   - メモリ使用量（RSS, VMS, 使用率%）
   - CPU 使用率（プロセス単位）
   - ディスク I/O（読み込み/書き込みバイト数）
   - スレッド数

2. **フェーズごとの測定**:

   - コンテキストマネージャー `with monitor.measure("phase_name")` で簡単測定
   - 開始時と終了時のリソース差分を自動計算
   - duration_sec, memory_delta_mb, cpu_avg_percent, io_read_delta_mb, io_write_delta_mb

3. **レポート生成**:

   - JSON 形式で詳細レポート出力
   - システム情報（CPU 数、総メモリ等）を記録
   - フェーズ別詳細とサマリー集計

4. **psutil 依存**:
   - psutil インストール時のみ有効化（未インストール時は自動的に無効化）
   - enabled=False で明示的に無効化も可能

#### 使用例

```python
from common.performance_monitor import PerformanceMonitor

# 初期化
monitor = PerformanceMonitor(enabled=True)

# フェーズ測定
with monitor.measure("phase1_symbols"):
    # シンボル取得処理
    pass

with monitor.measure("phase2_load"):
    # データロード処理
    pass

# レポート取得
report = monitor.get_report()
print(report["summary"]["total_duration_sec"])
print(report["summary"]["peak_memory_mb"])

# JSONに保存
monitor.save_report("logs/perf/detailed_metrics.json")

# コンソールにサマリー出力
monitor.print_summary()
```

---

### 2. テストの実装

**新規ファイル**: `tests/test_performance_monitor.py` (約 180 行)

#### テスト内容

- ✅ モニター初期化テスト
- ✅ 無効化モードテスト
- ✅ 単一フェーズ測定テスト
- ✅ 複数フェーズ測定テスト
- ✅ レポート生成テスト
- ✅ レポート保存テスト
- ✅ メモリトラッキングテスト（10MB 割り当てで検証）
- ✅ グローバルモニターテスト
- ✅ psutil 未インストール時の動作テスト
- ✅ ResourceSnapshot テスト

#### テスト結果

```
============== 11 passed in 4.22s ===============
Coverage: 73% (common/performance_monitor.py)
```

---

## 📊 測定可能なメトリクス

### 既存（perf_snapshot.py）

- 総実行時間
- システム別実行時間
- シンボル数
- 候補数
- キャッシュ I/O 回数（read_feather, read_csv, write_feather, write_csv）

### 新規（performance_monitor.py）

| カテゴリ         | メトリクス         | 単位   | 説明                                  |
| ---------------- | ------------------ | ------ | ------------------------------------- |
| **メモリ**       | memory_rss_mb      | MB     | 物理メモリ使用量（Resident Set Size） |
|                  | memory_vms_mb      | MB     | 仮想メモリ使用量                      |
|                  | memory_percent     | %      | メモリ使用率                          |
|                  | memory_delta_mb    | MB     | フェーズ前後のメモリ増減              |
|                  | memory_peak_mb     | MB     | ピークメモリ使用量                    |
| **CPU**          | cpu_percent        | %      | CPU 使用率（プロセス単位）            |
|                  | cpu_avg_percent    | %      | フェーズ平均 CPU 使用率               |
| **ディスク I/O** | io_read_bytes      | バイト | 累積読み込みバイト数                  |
|                  | io_write_bytes     | バイト | 累積書き込みバイト数                  |
|                  | io_read_delta_mb   | MB     | フェーズ中の読み込み量                |
|                  | io_write_delta_mb  | MB     | フェーズ中の書き込み量                |
| **プロセス**     | num_threads        | 個     | スレッド数                            |
| **システム**     | cpu_count_logical  | 個     | 論理 CPU 数                           |
|                  | cpu_count_physical | 個     | 物理 CPU 数                           |
|                  | total_memory_gb    | GB     | システム総メモリ                      |

---

## 🔄 既存システムとの統合計画

### Phase 6.1: CLI フラグ拡張（次のタスク）

`run_all_systems_today.py` に `--detailed-perf` フラグを追加：

```python
parser.add_argument(
    "--detailed-perf",
    action="store_true",
    help="詳細パフォーマンス測定（メモリ・CPU・ディスクI/O）を有効化",
)
```

### Phase 6.2: パイプライン統合（次のタスク）

主要フェーズで測定を実施：

- `phase0_setup`: 初期化処理
- `phase1_symbols`: シンボル取得
- `phase2_load`: データロード
- `phase3_filters`: フィルタ処理（Two-Phase）
- `phase4_signals`: シグナル生成
- `phase5_allocation`: 配分計算
- `phase6_save`: CSV 保存・通知

### Phase 6.3: レポート出力（次のタスク）

- 出力先: `logs/perf/detailed_metrics_YYYYMMDD_HHMMSS.json`
- コンソールにサマリー表示（`--detailed-perf` 時）
- 既存の `--benchmark` フラグと併用可能

---

## 🎯 達成状況

| タスク                        | 状態        | 備考                                 |
| ----------------------------- | ----------- | ------------------------------------ |
| PerformanceMonitor クラス実装 | ✅ 完了     | 360 行、psutil 依存                  |
| テスト実装                    | ✅ 完了     | 11 テスト、73%カバレッジ             |
| psutil 未インストール対応     | ✅ 完了     | 自動無効化＋テスト                   |
| レポート JSON 出力            | ✅ 完了     | システム情報＋フェーズ詳細＋サマリー |
| コンソールサマリー出力        | ✅ 完了     | print_summary() メソッド             |
| run_all_systems_today.py 統合 | ⏳ 次タスク | --detailed-perf フラグ追加           |
| パイプライン各フェーズ測定    | ⏳ 次タスク | 主要 6 フェーズで測定                |
| ドキュメント作成              | ✅ 完了     | 本レポート                           |

---

## 📝 使用上の注意

### 1. psutil のインストール

```bash
pip install psutil
```

既に `requirements.txt` に含まれているため、通常のセットアップで自動インストールされます。

### 2. パフォーマンスオーバーヘッド

- **CPU 測定**: 各スナップショットで 0.1 秒間隔の CPU 測定実施（軽微なオーバーヘッド）
- **メモリ測定**: ほぼオーバーヘッドなし
- **ディスク I/O**: プラットフォーム依存（Windows/Linux/macOS で挙動が異なる）

### 3. プラットフォーム互換性

- **Windows**: 完全サポート（ディスク I/O 含む）
- **Linux**: 完全サポート
- **macOS**: 基本サポート（一部 I/O カウンター未対応の可能性）

### 4. 既存の perf_snapshot との併用

両方を同時に有効化可能：

- `--perf-snapshot`: 既存の軽量測定（実行時間、キャッシュ I/O 回数）
- `--detailed-perf`: 新規の詳細測定（メモリ、CPU、ディスク I/O）

---

## 🚀 次のステップ

### 即座に実施可能

1. **run_all_systems_today.py への統合**:

   - `--detailed-perf` フラグ追加
   - 各フェーズで `monitor.measure()` 実行
   - レポート自動保存

2. **ベンチマークスクリプト作成**:

   - Mini/Quick/Sample モードでパフォーマンス比較
   - メモリ使用量の推移グラフ生成（Matplotlib 等）

3. **CI/CD 統合**:
   - GitHub Actions でパフォーマンス回帰テスト
   - PR ごとにメモリ使用量の変化を自動レポート

---

## 📊 Phase 6 完了条件

- [x] PerformanceMonitor クラス実装
- [x] テスト実装（カバレッジ 70%以上）
- [x] psutil 未インストール対応
- [x] レポート JSON 出力機能
- [x] コンソールサマリー出力
- [ ] run_all_systems_today.py 統合
- [ ] パイプライン各フェーズ測定
- [ ] ドキュメント整備

**現在の進捗**: 5/8 (62.5%)

---

## 結論

✅ **Phase 6 の基礎実装が完了しました**

- PerformanceMonitor クラス: メモリ・CPU・ディスク I/O の詳細測定機能を提供
- 包括的なテスト: 11 テスト全て Pass、73%カバレッジ
- psutil 依存の適切な処理: 未インストール時も安全に動作

次のタスクは `run_all_systems_today.py` への統合で、実際のパイプラインで測定を開始できます。

---

**実施者**: GitHub Copilot AI Agent  
**完了日時**: 2025-10-11 14:30  
**所要時間**: 約 30 分
