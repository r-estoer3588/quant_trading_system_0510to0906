"""
JSONL進捗ログから進捗後退(regression)を検出するツール

使い方:
    python tools/detect_progress_regression.py logs/progress_today.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Any


def detect_regressions(jsonl_path: Path) -> list[dict[str, Any]]:
    """進捗後退を検出"""
    regressions = []
    last_progress: dict[str, float] = {}  # system -> last_progress

    with open(jsonl_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                event = json.loads(line)

                # progress_updateイベントのみ処理
                if event.get("event_type") != "progress_update":
                    continue

                system = event.get("system", "unknown")
                phase = event.get("phase", "unknown")
                progress = event.get("data", {}).get("progress")

                if progress is None:
                    continue

                key = f"{system}_{phase}"

                # 進捗後退を検出
                if key in last_progress:
                    if progress < last_progress[key]:
                        regressions.append(
                            {
                                "line": line_num,
                                "timestamp": event.get("timestamp"),
                                "system": system,
                                "phase": phase,
                                "prev_progress": last_progress[key],
                                "current_progress": progress,
                                "diff": progress - last_progress[key],
                            }
                        )

                last_progress[key] = progress

            except json.JSONDecodeError:
                continue

    return regressions


def main() -> None:
    parser = argparse.ArgumentParser(description="JSONL進捗ログから後退検出")
    parser.add_argument("jsonl_path", type=Path, help="progress_today.jsonlのパス")

    args = parser.parse_args()

    if not args.jsonl_path.exists():
        print(f"❌ ファイルが見つかりません: {args.jsonl_path}")
        return

    print(f"🔍 進捗後退検出中: {args.jsonl_path}\n")

    regressions = detect_regressions(args.jsonl_path)

    if not regressions:
        print("✅ 進捗後退は検出されませんでした!")
    else:
        print(f"⚠️  {len(regressions)}件の進捗後退を検出:\n")
        for reg in regressions:
            print(
                f"Line {reg['line']:4d} | {reg['timestamp']} | "
                f"{reg['system']:8s} {reg['phase']:10s} | "
                f"{reg['prev_progress']:5.1f}% → {reg['current_progress']:5.1f}% "
                f"({reg['diff']:+.1f}%)"
            )


if __name__ == "__main__":
    main()
