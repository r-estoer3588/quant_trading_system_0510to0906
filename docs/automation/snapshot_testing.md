# Automated Test Data Generation & Snapshot Testing

## 🎯 目的

コード変更時に、テストデータを自動生成してスナップショット比較を行い、意図しない変更を検出。

## 📋 実装プラン

### Phase 1: pre-commit フックでスナップショット自動生成

```yaml
# .pre-commit-config.yaml に追加
repos:
  - repo: local
    hooks:
      - id: snapshot-test
        name: Snapshot Test (Core Changes)
        entry: python tools/auto_snapshot.py
        language: system
        pass_filenames: false
        files: ^(core|common|strategies)/.*\.py$
        stages: [pre-push]
```

### Phase 2: スナップショット自動生成ツール

```python
# tools/auto_snapshot.py
"""
コア変更時に自動でスナップショットを生成し、差分を検出
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def main():
    print("🔍 Generating test snapshots...")

    # 1. ミニテスト実行
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_all_systems_today.py",
            "--test-mode", "mini",
            "--skip-external",
            "--save-csv"
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("❌ Test run failed:")
        print(result.stderr)
        return 1

    # 2. スナップショット保存
    snapshot_dir = Path(f"snapshots/auto_{datetime.now():%Y%m%d_%H%M%S}")
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # 結果CSVをコピー
    import shutil
    for csv in Path("results_csv_test").glob("*.csv"):
        shutil.copy(csv, snapshot_dir / csv.name)

    # 3. 前回スナップショットと比較
    previous = sorted(Path("snapshots").glob("auto_*"))
    if len(previous) >= 2:
        prev_dir = previous[-2]

        result = subprocess.run(
            [
                sys.executable,
                "tools/compare_snapshots.py",
                "--baseline", str(prev_dir),
                "--current", str(snapshot_dir),
                "--threshold", "0.01"  # 1%未満の差分は許容
            ],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("⚠️  Snapshot differences detected:")
            print(result.stdout)
            print("\n📊 Review the diff report:")
            print(f"   {snapshot_dir}/diff_report.html")

            # 差分を許容するか確認
            response = input("\nContinue with commit? (y/N): ")
            if response.lower() != 'y':
                return 1

    print(f"✅ Snapshot saved: {snapshot_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

### Phase 3: スナップショット比較ツール

```python
# tools/compare_snapshots.py
"""
2つのスナップショットディレクトリを比較
"""
import argparse
import pandas as pd
from pathlib import Path
import json

def compare_csv(baseline_path: Path, current_path: Path, threshold: float = 0.01):
    """CSV ファイルの差分を検出"""
    baseline_df = pd.read_csv(baseline_path)
    current_df = pd.read_csv(current_path)

    # 行数の差分
    row_diff = abs(len(current_df) - len(baseline_df)) / max(len(baseline_df), 1)

    # カラムの差分
    col_diff = set(current_df.columns) ^ set(baseline_df.columns)

    # 数値カラムの平均値差分
    numeric_cols = baseline_df.select_dtypes(include=['number']).columns
    value_diffs = {}

    for col in numeric_cols:
        if col in current_df.columns:
            baseline_mean = baseline_df[col].mean()
            current_mean = current_df[col].mean()

            if baseline_mean != 0:
                diff_pct = abs(current_mean - baseline_mean) / abs(baseline_mean)
                if diff_pct > threshold:
                    value_diffs[col] = {
                        'baseline': baseline_mean,
                        'current': current_mean,
                        'diff_pct': diff_pct
                    }

    return {
        'row_diff_pct': row_diff,
        'column_diff': list(col_diff),
        'value_diffs': value_diffs,
        'has_significant_diff': row_diff > threshold or bool(col_diff) or bool(value_diffs)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.01)
    args = parser.parse_args()

    results = {}

    # 全CSVファイルを比較
    for current_csv in args.current.glob("*.csv"):
        baseline_csv = args.baseline / current_csv.name

        if baseline_csv.exists():
            diff = compare_csv(baseline_csv, current_csv, args.threshold)
            results[current_csv.name] = diff

    # レポート生成
    report_path = args.current / "diff_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)

    # サマリー表示
    significant_diffs = [k for k, v in results.items() if v['has_significant_diff']]

    if significant_diffs:
        print("⚠️  Files with significant differences:")
        for file in significant_diffs:
            print(f"   - {file}")
            for col, diff in results[file]['value_diffs'].items():
                print(f"     • {col}: {diff['diff_pct']:.2%} change")
        return 1
    else:
        print("✅ No significant differences detected")
        return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
```

## 🔧 使い方

### 自動実行（pre-push フック）

```powershell
# core/ や common/ を変更後
git push origin branch0906

# → 自動でスナップショット生成 & 比較
```

### 手動実行

```powershell
# スナップショット生成
python tools/auto_snapshot.py

# 比較
python tools/compare_snapshots.py `
    --baseline snapshots/auto_20251012_120000 `
    --current snapshots/auto_20251012_130000 `
    --threshold 0.01
```

## 📈 メリット

- ✅ コード変更の影響を自動検出
- ✅ 意図しない動作変更を防止
- ✅ リファクタリング時の安心感
- ✅ レビュー時の客観的指標
