# Phase7-8: Documentation & Quality Gates

## 🎯 目的

Diagnostics API のドキュメント整備、README 更新、CHANGELOG 記録、品質ゲート（mypy, Codacy CI）の適用、最終受け入れテストを実施。

## 📋 前提条件（Phase2–6 完了済み）

- ✅ Diagnostics 統一キー導入・テスト完了
- ✅ Snapshot export & diff 比較ツール実装済み
- ✅ Mini パイプライン検証済み

## 🔧 実装タスク

### Phase7: Documentation

#### Task 7.1: Diagnostics API ドキュメント（優先度: 高）

**目的**: 統一 diagnostics キーの仕様と使用例を文書化

**対象**:

- 新規: `docs/technical/diagnostics.md`

**実装内容**:

````markdown
# Diagnostics API リファレンス

## 概要

各システムの候補生成関数は、以下の統一キーを含む `diagnostics` 辞書を返します。

## 統一キー一覧

### 全システム共通

| キー                    | 型  | 説明                         | 例            |
| ----------------------- | --- | ---------------------------- | ------------- |
| `ranking_source`        | str | "latest_only" / "full_scan"  | "latest_only" |
| `setup_predicate_count` | int | Setup 条件を満たした行数     | 5             |
| `final_top_n_count`     | int | 最終候補件数（ランキング後） | 3             |

### System1 専用

| キー                        | 型   | 説明                             |
| --------------------------- | ---- | -------------------------------- |
| `predicate_only_pass_count` | int  | Setup predicate のみ通過した件数 |
| `mismatch_flag`             | bool | Setup 列との不一致があれば True  |
| `count_a`                   | int  | フィルタ a 通過件数（レガシー）  |
| `count_b`                   | int  | フィルタ b 通過件数（レガシー）  |
| `count_c` ~ `count_f`       | int  | 各フィルタ段階の通過件数         |

## 使用例

### System1 candidates 生成

```python
from core.system1 import generate_system1_candidates

candidates, diagnostics = generate_system1_candidates(
    df, current_date, latest_only=True
)

print(diagnostics)
# {
#   "ranking_source": "latest_only",
#   "setup_predicate_count": 5,
#   "final_top_n_count": 3,
#   "predicate_only_pass_count": 5,
#   "mismatch_flag": False,
#   ...
# }
```
````

### Diagnostics フォールバック

```python
from common.system_diagnostics import get_diagnostics_with_fallback

safe_diag = get_diagnostics_with_fallback(raw_diagnostics, "system1")
# 欠損値は -1 でフォールバック
```

## トラブルシューティング

### `setup_predicate_count` が `-1` になる

**原因**: Diagnostics が欠損している  
**対処**: `get_diagnostics_with_fallback()` でラップして安全にアクセス

### `mismatch_flag` が `True` になる

**原因**: Setup 列と shared predicate の結果が不一致  
**対処**: `VALIDATE_SETUP_PREDICATE=1` で詳細ログを確認し、predicate ロジックを修正

## 関連ファイル

- `common/system_setup_predicates.py`: Setup predicate 実装
- `core/system1.py` ~ `core/system7.py`: Diagnostics 生成箇所
- `tools/export_diagnostics_snapshot.py`: Snapshot export ツール

````

---

#### Task 7.2: README 更新（優先度: 中）
**目的**: Diagnostics 機能の追加を README に反映

**対象**:
- `README.md`（既存）

**追加内容**:
```markdown
## 新機能: Diagnostics API

各システムの候補生成時に、詳細な診断情報を取得できるようになりました。

### 主な診断キー
- `setup_predicate_count`: Setup 条件通過件数
- `final_top_n_count`: 最終候補件数
- `ranking_source`: "latest_only" or "full_scan"

詳細は [docs/technical/diagnostics.md](docs/technical/diagnostics.md) を参照。

### Snapshot Export
Mini パイプライン実行後に診断情報を JSON エクスポート可能:
```bash
python scripts/run_all_systems_today.py --test-mode mini --skip-external
cat results_csv_test/diagnostics_snapshot_*.json | jq '.systems'
````

### 差分比較ツール

2 つのスナップショットを比較:

```bash
python tools/compare_diagnostics_snapshots.py \
  --baseline baseline.json \
  --current current.json \
  --output diff.json \
  --summary
```

````

---

#### Task 7.3: CHANGELOG 記録（優先度: 中）
**目的**: Phase0-7 の変更を CHANGELOG に記録

**対象**:
- `CHANGELOG.md`（既存）

**追加内容**:
```markdown
## [Unreleased]

### Added
- **Diagnostics API**: 統一キー（`setup_predicate_count`, `final_top_n_count`, `ranking_source`）を全システムに導入
- **Setup Predicates**: `common/system_setup_predicates.py` に共通 predicate 関数を実装
- **Snapshot Export**: `tools/export_diagnostics_snapshot.py` で診断情報を JSON 出力
- **Diff Comparison**: `tools/compare_diagnostics_snapshots.py` でスナップショット差分比較
- **TRD Validation**: `tools/verify_trd_length.py` で Trading Day リスト長を検証
- **Zero TRD Escalation**: 全システム候補ゼロ時に通知を送信

### Changed
- **Test Mode Freshness**: Mini/Quick/Sample モード時のデータ鮮度許容を 365 日に緩和（`scripts/run_all_systems_today.py`）
- **System6 Filter**: HV50 条件を two-phase フィルタに統合
- **Diagnostics Enrichment**: Systems 1-7 で統一キーを出力（System6 は別タスクで統合予定）

### Fixed
- **SPY Rolling Cache**: テストモードで SPY が読み込まれない問題を修正（freshness_tolerance 緩和）

### Tests
- **Parametric Diagnostics Tests**: `tests/diagnostics/test_diagnostics_param_all_systems.py` で Systems 1-7 を網羅
- **Minimal Diagnostics Tests**: 個別システムの diagnostics 形式を検証

### Documentation
- **Diagnostics API**: `docs/technical/diagnostics.md` に仕様と使用例を追加
- **README**: Diagnostics 機能の紹介セクション追加
````

---

### Phase8: Cleanup & Quality Gates

#### Task 8.1: mypy 静的型チェック（優先度: 高）

**目的**: 型ヒント違反を検出し、品質を担保

**対象**:

- 変更ファイル全体（`common/`, `core/`, `tools/`, `scripts/`）

**実装手順**:

1. **mypy 実行**:

```bash
mypy --config-file mypy.ini common/ core/ scripts/ tools/
```

2. **エラー分類**:

   - Critical: `None` の不適切な扱い、型不一致
   - Warning: 型ヒント不足、Any の多用

3. **修正方針**:
   - Critical エラーは即修正
   - Warning は可能な範囲で型ヒント追加

**検証**:

```bash
mypy --config-file mypy.ini common/system_setup_predicates.py core/system1.py
# No errors expected
```

---

#### Task 8.2: Codacy CLI 検証（優先度: 中）

**目的**: Codacy ルールに違反していないか検証

**対象**:

- 変更ファイル全体

**実装手順**:

1. **Codacy CLI 実行**:

```bash
codacy-analysis-cli analyze --directory . --tool ruff --format json > codacy_report/results.sarif
```

2. **結果確認**:

   - Security issues（脆弱性）: ゼロを維持
   - Code smells: 可能な範囲で修正
   - Complexity: 複雑度 10 以下を目指す

3. **修正適用**:

```bash
ruff check --fix common/ core/ scripts/ tools/
black common/ core/ scripts/ tools/
```

**検証**:

```bash
# Codacy レポート確認
cat codacy_report/results.sarif | jq '.runs[0].results | length'
# 0 または低い値が理想
```

---

#### Task 8.3: 最終受け入れテスト（優先度: 高）

**目的**: Phase0-7 の全機能が正常に動作することを確認

**テスト項目**:

1. **Mini パイプライン End-to-End**:

```bash
python scripts/run_all_systems_today.py --test-mode mini --skip-external --benchmark
# 期待: Exit Code 0, SPY loaded, System1 候補 1 件以上
```

2. **Diagnostics Snapshot Export**:

```bash
# Snapshot 生成
python scripts/run_all_systems_today.py --test-mode mini --skip-external

# JSON 確認
cat results_csv_test/diagnostics_snapshot_*.json | jq '.systems[] | {system_id, diagnostics}'
# 期待: 全システムで統一キーが存在
```

3. **Diff Comparison**:

```bash
# 2 回実行してベースラインと current を作成
python scripts/run_all_systems_today.py --test-mode mini --skip-external
cp results_csv_test/diagnostics_snapshot_*.json baseline.json

# 再実行
python scripts/run_all_systems_today.py --test-mode mini --skip-external
cp results_csv_test/diagnostics_snapshot_*.json current.json

# 比較
python tools/compare_diagnostics_snapshots.py \
  --baseline baseline.json \
  --current current.json \
  --output diff.json \
  --summary

# 結果確認
cat diff.json | jq '.diffs[] | select(.category != "no_change")'
```

4. **pytest All Tests**:

```bash
pytest -q --tb=short
# 期待: All tests pass, warnings 3 以下
```

5. **TRD Validation**:

```bash
python scripts/run_all_systems_today.py --test-mode mini --skip-external 2>&1 | grep "TRD length"
# 期待: "OK: system1 TRD length=1 (max=1)" などが出力
```

---

#### Task 8.4: Cleanup & Commit（優先度: 低）

**目的**: 不要なデバッグファイルを削除し、変更をコミット

**実装手順**:

1. **不要ファイル削除**:

```bash
# デバッグ用スクリプト
rm -f debug_*.py check_*.py test_*.py temp_*.py tmp_*.txt

# Snapshot の古いバージョン
rm -f results_csv_test/diagnostics_snapshot_old_*.json
```

2. **Git 確認**:

```bash
git status
git diff
```

3. **Commit**:

```bash
git add .
git commit -m "Phase0-7: Diagnostics API & Setup Predicates Unification

- Added unified diagnostics keys (setup_predicate_count, final_top_n_count, ranking_source)
- Implemented shared setup predicates in common/system_setup_predicates.py
- Integrated predicates into Systems 1-7 (System6 separate task)
- Added snapshot export and diff comparison tools
- Relaxed freshness tolerance in test modes for SPY loading
- Updated documentation (diagnostics.md, README.md, CHANGELOG.md)
- Passed mini regression pipeline and pytest
"
```

---

## 📊 完了条件

- [ ] Diagnostics API ドキュメント作成（`docs/technical/diagnostics.md`）
- [ ] README 更新（Diagnostics セクション追加）
- [ ] CHANGELOG 記録（Phase0-7 変更内容）
- [ ] mypy 静的型チェック実行・修正
- [ ] Codacy CLI 検証・修正
- [ ] 最終受け入れテスト全項目 Pass
- [ ] 不要ファイル削除
- [ ] Git commit 完了

## 🔗 関連ドキュメント

- `docs/technical/diagnostics.md`（新規作成）
- `docs/README.md`（統合ナビゲーション）
- `CHANGELOG.md`（変更履歴）

## 🚀 開始コマンド

```bash
# Documentation 作成
code docs/technical/diagnostics.md

# README 更新
code README.md

# CHANGELOG 更新
code CHANGELOG.md

# mypy 実行
mypy --config-file mypy.ini common/ core/ scripts/ tools/

# Codacy CLI
codacy-analysis-cli analyze --directory . --tool ruff

# 最終受け入れテスト
pytest -q --tb=short
python scripts/run_all_systems_today.py --test-mode mini --skip-external --benchmark

# Cleanup
rm -f debug_*.py check_*.py test_*.py temp_*.py tmp_*.txt
git status
```

## 📝 注意事項

- **mypy エラーは Critical のみ必須修正**（Warning は許容）
- **Codacy の Security issues はゼロを維持**
- **最終受け入れテストで Exit Code 0 が必須**
- **Commit メッセージは簡潔に（Phase0-7 の要約）**
- **不要ファイルは削除前に念のためバックアップ推奨**

---

## 🎉 Phase0-8 完了後の状態

### 達成内容

- ✅ Diagnostics 統一キー導入（全システム）
- ✅ Setup predicates 共通化
- ✅ Snapshot export & diff 比較ツール実装
- ✅ Mini パイプライン End-to-End 検証
- ✅ ドキュメント整備（diagnostics.md, README, CHANGELOG）
- ✅ 品質ゲート通過（pytest, mypy, Codacy）

### 次のステップ（オプション）

- System6 への shared predicate 統合（Phase3 で実施予定）
- CI/CD への Codacy 統合（自動品質チェック）
- Production モードでの通知テスト（ゼロ TRD エスカレーション）
