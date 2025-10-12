# AI-Powered Code Review Automation

## 🎯 目的

コミット前に AI がコードをレビューし、潜在的な問題を指摘。

## 📋 実装プラン

### Phase 1: GitHub Actions で AI レビュー

```yaml
# .github/workflows/ai-code-review.yml
name: AI Code Review

on:
  pull_request:
    types: [opened, synchronize]
    paths:
      - "**.py"

jobs:
  review:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0 # 差分取得のため

      - name: Get changed files
        id: changed-files
        uses: tj-actions/changed-files@v41
        with:
          files: |
            **.py

      - name: AI Review with GitHub Copilot
        uses: github/copilot-code-review-action@v1
        with:
          files: ${{ steps.changed-files.outputs.all_changed_files }}
          focus_areas: |
            - キャッシュマネージャーの直接I/O禁止ルール
            - System7のSPY固定ルール
            - Two-Phase Filter/Setup パターン
            - 環境変数の型安全アクセス

      - name: Review Python specific issues
        run: |
          # カスタムルールチェック
          python tools/check_project_rules.py ${{ steps.changed-files.outputs.all_changed_files }}
```

### Phase 2: プロジェクト固有ルールチェッカー

```python
# tools/check_project_rules.py
"""
プロジェクト固有のルールをチェック
"""
import sys
import re
from pathlib import Path

RULES = {
    "direct_csv_read": {
        "pattern": r"pd\.read_csv\(['\"]data_cache/",
        "message": "❌ CacheManager を使用せずに直接 CSV を読み込んでいます",
        "severity": "error"
    },
    "system7_spy_violation": {
        "pattern": r"system7.*symbols.*!=.*SPY",
        "message": "⚠️  System7 は SPY 固定です",
        "severity": "warning",
        "files": ["core/system7.py"]
    },
    "env_direct_access": {
        "pattern": r"os\.environ\.get\(",
        "message": "⚠️  環境変数は get_env_config() 経由でアクセスしてください",
        "severity": "warning",
        "exclude_files": ["config/environment.py"]
    },
    "missing_diagnostics": {
        "pattern": r"def generate_system\d+_candidates.*\):",
        "message": "ℹ️  Diagnostics API を実装していますか？",
        "severity": "info",
        "files": ["core/system*.py"]
    }
}

def check_file(file_path: Path):
    """ファイルのルール違反をチェック"""
    content = file_path.read_text(encoding='utf-8')
    violations = []

    for rule_name, rule in RULES.items():
        # ファイル制限チェック
        if 'files' in rule:
            if not any(file_path.match(pattern) for pattern in rule['files']):
                continue

        # 除外ファイルチェック
        if 'exclude_files' in rule:
            if any(file_path.match(pattern) for pattern in rule['exclude_files']):
                continue

        # パターンマッチ
        matches = re.finditer(rule['pattern'], content, re.MULTILINE)
        for match in matches:
            line_num = content[:match.start()].count('\n') + 1
            violations.append({
                'file': str(file_path),
                'line': line_num,
                'rule': rule_name,
                'message': rule['message'],
                'severity': rule['severity']
            })

    return violations

def main(file_paths):
    all_violations = []

    for file_path in file_paths:
        path = Path(file_path)
        if path.suffix == '.py':
            violations = check_file(path)
            all_violations.extend(violations)

    # レポート出力
    if all_violations:
        print("📋 Code Review Findings:\n")

        for v in all_violations:
            icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}[v['severity']]
            print(f"{icon} {v['file']}:{v['line']}")
            print(f"   {v['message']}\n")

        # エラーがあれば終了コード 1
        errors = [v for v in all_violations if v['severity'] == 'error']
        if errors:
            print(f"\n❌ {len(errors)} error(s) found. Please fix before committing.")
            return 1
        else:
            print(f"\n⚠️  {len(all_violations)} warning(s) found. Review recommended.")
            return 0
    else:
        print("✅ No rule violations found.")
        return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

### Phase 3: ローカル pre-commit フック

```yaml
# .pre-commit-config.yaml に追加
- repo: local
  hooks:
    - id: project-rules
      name: Project-Specific Rules Check
      entry: python tools/check_project_rules.py
      language: system
      types: [python]
      pass_filenames: true
```

## 🔧 使い方

### 自動実行（pre-commit）

```powershell
git add core/system1.py
git commit -m "Update System1 logic"

# → 自動でルールチェック実行
```

### 手動実行

```powershell
# 特定ファイル
python tools/check_project_rules.py core/system1.py

# 全Pythonファイル
python tools/check_project_rules.py (Get-ChildItem -Recurse -Filter *.py).FullName
```

## 📈 メリット

- ✅ プロジェクトルールの自動強制
- ✅ レビュー工数削減
- ✅ 新規メンバーのオンボーディング支援
- ✅ 一貫性のあるコードベース
