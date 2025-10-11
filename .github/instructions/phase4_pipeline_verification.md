# Phase4: Pipeline Verification & Log Validation

## 🎯 目的

Mini パイプラインの動作を継続的に検証し、ログの簡潔性と診断情報の正確性を担保。TRD（Trading Day）リスト長の検証と、不整合のトリアージを実施。

## 📋 前提条件（Phase2–3 完了済み）

- ✅ Mini パイプライン実行成功（Exit Code 0、SPY 取り込み成功）
- ✅ Diagnostics 統一キー導入済み
- ✅ テストモード時の鮮度緩和ロジック実装済み

## 🔧 実装タスク

### Task 4.1: TRD リスト長の検証（優先度: 中）

**目的**: 各システムの candidate 抽出で、日付リストの長さが想定範囲内か検証

**対象ファイル**:

- `scripts/run_all_systems_today.py`（mini モード実行時の検証ロジック追加）
- 新規: `tools/verify_trd_length.py`（スタンドアロン検証ツール）

**実装内容**:

```python
# tools/verify_trd_length.py
def verify_trd_length(by_date: dict, system_id: str, expected_max: int = 5) -> dict:
    """Trading day リストの長さを検証。

    Args:
        by_date: {date: [候補リスト]} の辞書
        system_id: システム ID（ログ用）
        expected_max: 想定される最大日数（mini=1, quick=5, full=30 など）

    Returns:
        検証結果の辞書（valid, actual_length, exceeded, message）
    """
    actual_len = len(by_date)
    exceeded = actual_len > expected_max

    result = {
        "system_id": system_id,
        "valid": not exceeded,
        "expected_max": expected_max,
        "actual_length": actual_len,
        "exceeded": exceeded,
        "message": (
            f"OK: {system_id} TRD length={actual_len} (max={expected_max})"
            if not exceeded
            else f"⚠️ {system_id} TRD length={actual_len} exceeds max={expected_max}"
        ),
    }
    return result
```

**統合**:

- `compute_today_signals()` 内で各システムの `by_date` を検証
- `--test-mode mini` 時は `expected_max=1`、`quick` は `5`、`sample` は `10`
- 検証失敗時はログに警告を出力（エラーで止めない）

**検証**:

```bash
python scripts/run_all_systems_today.py --test-mode mini --skip-external
# ログに "OK: system1 TRD length=1 (max=1)" などが出力されることを確認
```

---

### Task 4.2: Compact Log Validation（優先度: 低）

**目的**: `COMPACT_TODAY_LOGS=1` 時のログが過剰にならないか検証

**対象**:

- 既存の compact log モード（`scripts/run_all_systems_today.py`）
- `common/today_filters.py`（フィルタ段階のログ）
- `common/today_signals.py`（シグナル抽出のログ）

**実装内容**:

1. **ログ行数カウンタ**を導入
   - Mini モード実行時のログ行数を計測
   - Compact モード ON/OFF での差分を比較
2. **基準値設定**:

   - Mini モード（1 銘柄）: compact OFF で最大 500 行、compact ON で最大 200 行
   - 超過時は警告（CI で検出可能にする）

3. **検証スクリプト**:

```python
# tools/validate_log_compactness.py
def count_log_lines(log_file: Path) -> int:
    """ログファイルの行数をカウント（空行除く）。"""
    with open(log_file, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

def validate_compact_log(log_path: Path, mode: str, max_lines: int) -> dict:
    """Compact log の行数を検証。

    Args:
        log_path: ログファイルパス
        mode: "compact_on" | "compact_off"
        max_lines: 最大許容行数

    Returns:
        検証結果の辞書
    """
    actual = count_log_lines(log_path)
    valid = actual <= max_lines

    return {
        "mode": mode,
        "max_lines": max_lines,
        "actual_lines": actual,
        "valid": valid,
        "message": (
            f"OK: {mode} log lines={actual} (max={max_lines})"
            if valid
            else f"⚠️ {mode} log lines={actual} exceeds max={max_lines}"
        ),
    }
```

**検証手順**:

```bash
# Compact OFF で実行
export COMPACT_TODAY_LOGS=0
python scripts/run_all_systems_today.py --test-mode mini --skip-external > logs/mini_verbose.log 2>&1

# Compact ON で実行
export COMPACT_TODAY_LOGS=1
python scripts/run_all_systems_today.py --test-mode mini --skip-external > logs/mini_compact.log 2>&1

# 行数比較
python tools/validate_log_compactness.py --verbose logs/mini_verbose.log --compact logs/mini_compact.log
```

---

### Task 4.3: Discrepancy Triage（優先度: 低）

**目的**: setup_predicate_count と ranked_top_n_count の差分を分類し、原因をトリアージ

**対象**:

- `common/system_diagnostics.py`（新規ユーティリティ）

**実装内容**:

```python
# common/system_diagnostics.py
def triage_candidate_discrepancy(diag: dict) -> dict:
    """Setup 通過数と最終候補数の差分を分類。

    分類:
    - "exact_match": setup_count == final_count（理想）
    - "ranking_filtered": setup_count > final_count（ランキングで絞り込み）
    - "zero_setup": setup_count == 0（フィルタで全滅）
    - "unexpected": その他（要調査）

    Returns:
        {category, setup_count, final_count, diff, message}
    """
    setup_count = int(diag.get("setup_predicate_count", 0))
    final_count = int(diag.get("ranked_top_n_count", 0))
    diff = setup_count - final_count

    if setup_count == final_count:
        category = "exact_match"
        message = f"Setup {setup_count} == Final {final_count}"
    elif setup_count > final_count >= 0:
        category = "ranking_filtered"
        message = f"Setup {setup_count} → Final {final_count} (filtered {diff})"
    elif setup_count == 0:
        category = "zero_setup"
        message = "No candidates passed setup"
    else:
        category = "unexpected"
        message = f"⚠️ Setup {setup_count} vs Final {final_count} (unexpected)"

    return {
        "category": category,
        "setup_count": setup_count,
        "final_count": final_count,
        "diff": diff,
        "message": message,
    }
```

**統合**:

- Mini パイプライン実行後に各システムの diagnostics をトリアージ
- 結果を `results_csv_test/discrepancy_triage.json` に保存

**検証**:

```bash
python scripts/run_all_systems_today.py --test-mode mini --skip-external
# 実行後に discrepancy_triage.json が生成され、各システムの分類が記録されることを確認
```

---

## 📊 完了条件

- [ ] TRD リスト長検証ツール実装（`tools/verify_trd_length.py`）
- [ ] Mini パイプラインに TRD 検証を統合
- [ ] Compact log 行数検証ツール実装（`tools/validate_log_compactness.py`）
- [ ] Discrepancy triage ユーティリティ実装（`common/system_diagnostics.py`）
- [ ] Mini 実行後に `results_csv_test/discrepancy_triage.json` が生成される

## 🔗 関連ドキュメント

- `docs/operations/daily_execution.md`（運用ガイド、Phase7 で作成予定）
- `scripts/run_all_systems_today.py`（メインパイプライン）
- `common/today_filters.py`（フィルタロジック）

## 🚀 開始コマンド

```bash
# Mini パイプライン実行（TRD 検証含む）
python scripts/run_all_systems_today.py --test-mode mini --skip-external --benchmark

# Compact log 比較
export COMPACT_TODAY_LOGS=0
python scripts/run_all_systems_today.py --test-mode mini --skip-external 2>&1 | tee logs/mini_verbose.log

export COMPACT_TODAY_LOGS=1
python scripts/run_all_systems_today.py --test-mode mini --skip-external 2>&1 | tee logs/mini_compact.log

# 行数比較
wc -l logs/mini_*.log
```

## 📝 注意事項

- TRD 検証は警告のみ（エラーで停止しない）
- Compact log は DEBUG レベルの詳細を抑制するが、重要な警告は残す
- Discrepancy triage は定常的に実行し、unexpected カテゴリが増えたら調査
