# CI/CD ワークフロー統合プラン

## 現状分析

現在 3 つのワークフローが存在し、一部機能が重複:

1. **ci.yml** - 基本的なリント＋テスト＋ Codacy
2. **ci_quality.yml** - 品質ゲート（mypy, pytest, 警告集計, Codacy）
3. **quality-check.yml** - 自動修正＋コミット
4. **coverage-report.yml** (NEW) - カバレッジ計測＋レポート

## 統合案

### Option A: 段階的統合（推奨）

**メリット**: 既存ワークフローを壊さず、段階的に移行可能

**構成**:

```
.github/workflows/
├── ci-unified.yml          # 統合CI（リント＋テスト＋カバレッジ）
├── quality-gate.yml        # 品質ゲート（mypy＋警告集計）
├── auto-fix.yml            # 自動修正（push時）
└── codacy.yml              # Codacy分析（分離）
```

**実装**:

1. `ci-unified.yml` を新規作成（ci.yml + coverage-report.yml 統合）
2. `ci_quality.yml` を `quality-gate.yml` にリネーム・整理
3. `quality-check.yml` を `auto-fix.yml` にリネーム
4. Codacy 処理を分離（重複削減）
5. 旧ファイルは `.github/workflows/archive/` に移動

### Option B: 完全統合

**メリット**: ワークフロー数最小化、管理コスト削減

**構成**:

```
.github/workflows/
├── main-ci.yml             # すべてのCI処理を1つに統合
└── auto-fix.yml            # 自動修正のみ分離
```

**デメリット**:

- 1 つのワークフローが長くなる
- 部分的な実行が難しい
- トラブルシューティングが複雑化

### Option C: 役割別分離（現状維持+整理）

**メリット**: 役割が明確、柔軟性高い

**構成**:

```
.github/workflows/
├── lint.yml                # リント（ruff, black）
├── test.yml                # テスト（pytest）
├── coverage.yml            # カバレッジ計測
├── type-check.yml          # 型チェック（mypy）
├── security.yml            # セキュリティ（Codacy, bandit）
└── auto-fix.yml            # 自動修正
```

**デメリット**: ワークフロー数が多い、並列実行でコスト増

## 推奨: Option A 実装手順

### Step 1: ci-unified.yml 作成

```yaml
name: CI Unified

on:
  push:
    branches: ["**"]
  pull_request:
    branches: ["**"]

jobs:
  lint-and-format:
    # ruff + black

  test-and-coverage:
    needs: [lint-and-format]
    # pytest + coverage
    # System7 threshold check

  upload-artifacts:
    needs: [test-and-coverage]
    # HTML/XML/JSON artifacts
```

### Step 2: 既存ワークフロー整理

- `ci.yml` → archive
- `ci_quality.yml` → 品質指標部分のみ抽出
- `coverage-report.yml` → ci-unified.yml に統合

### Step 3: README.md 更新

- ワークフローバッジ更新
- 統合後の構成を説明

## 次のステップ

1. **Option A を実装する** （推奨）
2. **Option C を実装する** （細かく分離したい場合）
3. **現状維持** （coverage-report.yml を追加のみ）

どれを選びますか？
