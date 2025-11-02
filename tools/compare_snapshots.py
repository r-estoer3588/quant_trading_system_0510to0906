"""
スナップショット比較ツール

2つのスナップショットディレクトリを比較し、差分をレポート。

使い方:
    python tools/compare_snapshots.py \\
        --baseline snapshots/auto_20251012_120000 \\
        --current snapshots/auto_20251012_130000 \\
        --threshold 0.01
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Set


def compare_csv(
    baseline_path: Path, current_path: Path, threshold: float = 0.01
) -> Dict:
    """CSV ファイルの差分を検出"""
    try:
        import pandas as pd
    except ImportError:
        return {"error": "pandas not installed", "has_significant_diff": False}

    try:
        baseline_df = pd.read_csv(baseline_path)
        current_df = pd.read_csv(current_path)
    except Exception as e:
        return {"error": str(e), "has_significant_diff": False}

    # 行数の差分
    row_diff_pct = abs(len(current_df) - len(baseline_df)) / max(len(baseline_df), 1)

    # カラムの差分
    col_diff = set(current_df.columns) ^ set(baseline_df.columns)

    # 数値カラムの平均値差分
    numeric_cols = baseline_df.select_dtypes(include=["number"]).columns
    value_diffs = {}

    for col in numeric_cols:
        if col in current_df.columns:
            baseline_mean = baseline_df[col].mean()
            current_mean = current_df[col].mean()

            if abs(baseline_mean) > 0.001:  # ゼロ除算回避
                diff_pct = abs(current_mean - baseline_mean) / abs(baseline_mean)
                if diff_pct > threshold:
                    value_diffs[col] = {
                        "baseline": float(baseline_mean),
                        "current": float(current_mean),
                        "diff_pct": float(diff_pct),
                    }

    return {
        "row_count": {
            "baseline": len(baseline_df),
            "current": len(current_df),
            "diff_pct": float(row_diff_pct),
        },
        "column_diff": list(col_diff),
        "value_diffs": value_diffs,
        "has_significant_diff": (
            row_diff_pct > threshold or bool(col_diff) or bool(value_diffs)
        ),
    }


def find_matching_files(baseline_dir: Path, current_dir: Path) -> Set[str]:
    """両方のディレクトリに存在するファイルを検出"""
    baseline_files = {f.name for f in baseline_dir.glob("*.csv")}
    current_files = {f.name for f in current_dir.glob("*.csv")}

    return baseline_files & current_files


def main():
    parser = argparse.ArgumentParser(description="スナップショット比較")
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="ベースラインスナップショットディレクトリ",
    )
    parser.add_argument(
        "--current", type=Path, required=True, help="現在のスナップショットディレクトリ"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="許容する差分率（デフォルト: 0.01 = 1%%）",
    )
    args = parser.parse_args()

    if not args.baseline.exists():
        print(f"❌ Baseline directory not found: {args.baseline}")
        return 1

    if not args.current.exists():
        print(f"❌ Current directory not found: {args.current}")
        return 1

    # 共通ファイルを検出
    common_files = find_matching_files(args.baseline, args.current)

    if not common_files:
        print("⚠️  No common CSV files found")
        return 0

    results = {}

    # 全CSVファイルを比較
    for filename in sorted(common_files):
        baseline_csv = args.baseline / filename
        current_csv = args.current / filename

        diff = compare_csv(baseline_csv, current_csv, args.threshold)
        results[filename] = diff

    # レポート生成
    report_path = args.current / "diff_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # サマリー表示
    significant_diffs = [
        k for k, v in results.items() if v.get("has_significant_diff", False)
    ]

    if significant_diffs:
        print(f"⚠️  Files with significant differences (>{args.threshold:.0%}):\n")

        for filename in significant_diffs:
            print(f"  📄 {filename}")

            diff = results[filename]

            # 行数差分
            row_info = diff.get("row_count", {})
            if row_info.get("diff_pct", 0) > args.threshold:
                print(
                    f"     • Row count: {row_info['baseline']} → {row_info['current']} ({row_info['diff_pct']:+.1%})"
                )

            # カラム差分
            col_diff = diff.get("column_diff", [])
            if col_diff:
                print(f"     • Column diff: {col_diff}")

            # 値差分
            value_diffs = diff.get("value_diffs", {})
            for col, vdiff in list(value_diffs.items())[:3]:  # 最大3件表示
                print(
                    f"     • {col}: {vdiff['diff_pct']:+.1%} change ({vdiff['baseline']:.2f} → {vdiff['current']:.2f})"
                )

            if len(value_diffs) > 3:
                print(f"     • ... and {len(value_diffs) - 3} more value diffs")

            print()

        print(f"📊 Detailed report: {report_path}")
        return 1
    else:
        print(
            f"✅ No significant differences detected (threshold: {args.threshold:.0%})"
        )
        print(f"   Compared {len(common_files)} file(s)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
