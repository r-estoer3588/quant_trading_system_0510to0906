"""Compare two diagnostics snapshots and generate diff report.

2 つの diagnostics スナップショット JSON を比較し、差分を分類します。
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def compare_snapshots(baseline_path: Path, current_path: Path) -> dict[str, Any]:
    """2 つの diagnostics スナップショットを比較。

    Args:
        baseline_path: ベースライン JSON
        current_path: 現在の JSON

    Returns:
        差分の辞書（システムごとの増減、カテゴリ分類を含む）
    """
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    with open(current_path, "r", encoding="utf-8") as f:
        current = json.load(f)

    # システム ID でマッピング
    baseline_systems = {s["system_id"]: s for s in baseline.get("systems", [])}
    current_systems = {s["system_id"]: s for s in current.get("systems", [])}

    diffs = []
    for sys_id in sorted(set(baseline_systems.keys()) | set(current_systems.keys())):
        b_diag = baseline_systems.get(sys_id, {}).get("diagnostics", {})
        c_diag = current_systems.get(sys_id, {}).get("diagnostics", {})

        diff = {
            "system_id": sys_id,
            "setup_predicate_count": {
                "baseline": b_diag.get("setup_predicate_count", -1),
                "current": c_diag.get("setup_predicate_count", -1),
                "diff": c_diag.get("setup_predicate_count", -1)
                - b_diag.get("setup_predicate_count", -1),
            },
            "final_top_n_count": {
                "baseline": b_diag.get("final_top_n_count", -1),
                "current": c_diag.get("final_top_n_count", -1),
                "diff": c_diag.get("final_top_n_count", -1) - b_diag.get("final_top_n_count", -1),
            },
            "category": _classify_diff(b_diag, c_diag),
        }
        diffs.append(diff)

    return {
        "baseline_date": baseline.get("export_date"),
        "current_date": current.get("export_date"),
        "diffs": diffs,
    }


def _classify_diff(baseline: dict, current: dict) -> str:
    """差分をカテゴリ分類（no_change, increase, decrease, new, removed）。"""
    b_final = baseline.get("final_top_n_count", -1)
    c_final = current.get("final_top_n_count", -1)

    if b_final == -1 and c_final >= 0:
        return "new"
    elif b_final >= 0 and c_final == -1:
        return "removed"
    elif b_final == c_final:
        return "no_change"
    elif c_final > b_final:
        return "increase"
    else:
        return "decrease"


def summarize_diff(diff_result: dict) -> dict[str, int]:
    """差分カテゴリごとの集計を返す。

    Returns:
        {category: count} の辞書
    """
    categories = [d["category"] for d in diff_result["diffs"]]
    return dict(Counter(categories))


def _pick_latest_two(dir_path: Path) -> tuple[Path, Path]:
    files = sorted(
        (p for p in dir_path.glob("*.json") if p.is_file()),
        key=lambda p: (p.stat().st_mtime, p.name),
        reverse=True,
    )
    if len(files) < 2:
        raise FileNotFoundError(
            "スナップショット JSON が2件以上必要です（--dir には最新2件を想定）"
        )
    return files[1], files[0]  # baseline, current


def main():
    """CLI エントリーポイント。"""
    parser = argparse.ArgumentParser(description="Compare diagnostics snapshots")
    parser.add_argument(
        "--baseline",
        type=Path,
        required=False,
        help="Baseline snapshot JSON path",
    )
    parser.add_argument(
        "--current",
        type=Path,
        required=False,
        help="Current snapshot JSON path",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        required=False,
        help="Directory to auto-pick latest two snapshot JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output diff JSON path",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary of diff categories",
    )
    args = parser.parse_args()

    # 入力決定: --dir があればそこから最新2件、なければ --baseline/--current 必須
    if args.dir:
        base_path, curr_path = _pick_latest_two(args.dir)
    else:
        if not args.baseline or not args.current:
            raise SystemExit("--baseline と --current を指定するか、--dir を使用してください")
        base_path, curr_path = args.baseline, args.current

    # スナップショット比較
    diff_result = compare_snapshots(base_path, curr_path)

    # 出力
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(diff_result, f, indent=2, ensure_ascii=False)
        logger.info("Diff result saved to %s", args.output)

    # サマリ表示
    if args.summary:
        summary = summarize_diff(diff_result)
        print("\n=== Diff Category Summary ===")
        for category, count in sorted(summary.items()):
            print(f"{category}: {count}")
        print()

    # 変更があったシステムのみ表示
    changed_systems = [d for d in diff_result["diffs"] if d["category"] != "no_change"]
    if changed_systems:
        print("\n=== Changed Systems ===")
        for d in changed_systems:
            print(f"\nSystem: {d['system_id']} ({d['category']})")
            print(f"  setup_predicate_count: {d['setup_predicate_count']}")
            print(f"  final_top_n_count: {d['final_top_n_count']}")
    else:
        print("\n=== No Changes Detected ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
