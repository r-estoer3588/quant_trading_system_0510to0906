"""
スクショ解析結果とJSONL進捗ログの同期検証ツール

Usage example:
    python tools/verify_ui_jsonl_sync.py \
        --screenshots screenshots/progress_tracking/analysis_results.json \
        --jsonl logs/progress_today.jsonl \
        --output screenshots/progress_tracking/sync_verification.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_jsonl_timestamp(ts_str: str) -> datetime:
    """Parse JSONL timestamp like '2025/10/13 6:52:07'."""
    parts = ts_str.split()
    date_parts = parts[0].split("/")
    time_parts = parts[1].split(":")

    year = int(date_parts[0])
    month = int(date_parts[1])
    day = int(date_parts[2])
    hour = int(time_parts[0])
    minute = int(time_parts[1])
    second = int(time_parts[2])

    return datetime(year, month, day, hour, minute, second)


def parse_screenshot_timestamp(ts_str: str) -> datetime:
    """Parse screenshot timestamp like '2025-10-13 06:52:09.856'."""
    return datetime.strptime(ts_str[:19], "%Y-%m-%d %H:%M:%S")


def load_jsonl_events(jsonl_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            event = json.loads(line.strip())
            events.append(event)
    return events


def calculate_expected_progress(
    system_name: str, total_systems: int = 7
) -> tuple[float, float]:
    system_order = {
        "system1": 0,
        "system2": 1,
        "system3": 2,
        "system4": 3,
        "system5": 4,
        "system6": 5,
        "system7": 6,
    }
    order = system_order.get(system_name.lower(), 0)
    start = (order / total_systems) * 100
    end = ((order + 1) / total_systems) * 100
    return round(start, 1), round(end, 1)


def find_nearest_jsonl_event(
    screenshot_ts: datetime, events: list[dict[str, Any]], time_window: int = 10
) -> dict[str, Any] | None:
    closest_event = None
    min_diff = float("inf")
    for event in events:
        try:
            event_ts = parse_jsonl_timestamp(event["timestamp"])
            diff = abs((event_ts - screenshot_ts).total_seconds())
            if diff <= time_window and diff < min_diff:
                min_diff = diff
                closest_event = event
        except Exception:
            continue
    return closest_event


def verify_single_screenshot(
    screenshot_data: dict[str, Any], jsonl_events: list[dict[str, Any]]
) -> dict[str, Any]:
    issues: list[str] = []
    result: dict[str, Any] = {
        "file": screenshot_data.get("file"),
        "timestamp": screenshot_data.get("timestamp"),
        "issues": issues,
        "status": "ok",
    }

    try:
        screenshot_ts = parse_screenshot_timestamp(screenshot_data["timestamp"])
    except Exception:
        issues.append("タイムスタンプパース失敗")
        result["status"] = "error"
        return result

    nearest_event = find_nearest_jsonl_event(screenshot_ts, jsonl_events)
    if not nearest_event:
        issues.append("対応するJSONLイベントが見つかりません")
        result["status"] = "warning"
        return result

    result["nearest_jsonl_event"] = {
        "timestamp": nearest_event.get("timestamp"),
        "event_type": nearest_event.get("event_type"),
        "system": nearest_event.get("data", {}).get("system"),
    }

    ui_progress = screenshot_data.get("progress_percentage")
    jsonl_system = nearest_event.get("data", {}).get("system")
    if ui_progress is not None and jsonl_system:
        expected_start, expected_end = calculate_expected_progress(jsonl_system)
        if ui_progress < expected_start - 5 or ui_progress > expected_end + 5:
            issues.append(
                "進捗バー不一致: UI="
                + str(ui_progress)
                + "%, 期待範囲="
                + str(expected_start)
                + "-"
                + str(expected_end)
                + "% (for "
                + str(jsonl_system)
                + ")"
            )
            result["status"] = "issue"

    ui_system = screenshot_data.get("system_name")
    if ui_system and jsonl_system:
        if ui_system.lower() != jsonl_system.lower():
            issues.append(
                "システム名不一致: UI="
                + str(ui_system)
                + ", JSONL="
                + str(jsonl_system)
            )
            result["status"] = "issue"

    ui_candidates = screenshot_data.get("candidates")
    jsonl_candidates = nearest_event.get("data", {}).get("candidates")
    if ui_candidates and jsonl_candidates is not None:
        if int(ui_candidates) != int(jsonl_candidates):
            issues.append(
                "候補数不一致: UI="
                + str(ui_candidates)
                + ", JSONL="
                + str(jsonl_candidates)
            )
            result["status"] = "issue"

    return result


def detect_progress_regression(
    screenshot_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    regressions: list[dict[str, Any]] = []
    for i in range(1, len(screenshot_results)):
        prev = screenshot_results[i - 1]
        curr = screenshot_results[i]
        prev_progress = prev.get("progress_percentage")
        curr_progress = curr.get("progress_percentage")
        if prev_progress is not None and curr_progress is not None:
            if curr_progress < prev_progress - 2:
                regressions.append(
                    {
                        "prev_file": prev.get("file"),
                        "curr_file": curr.get("file"),
                        "prev_timestamp": prev.get("timestamp"),
                        "curr_timestamp": curr.get("timestamp"),
                        "prev_progress": prev_progress,
                        "curr_progress": curr_progress,
                        "regression_amount": round(prev_progress - curr_progress, 1),
                    }
                )
    return regressions


def main() -> None:
    parser = argparse.ArgumentParser(description="UI/JSONL同期検証")
    parser.add_argument(
        "--screenshots",
        type=str,
        default="screenshots/progress_tracking/analysis_results.json",
        help="スクショ解析結果JSON",
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default="logs/progress_today.jsonl",
        help="JSONL進捗ログ",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="screenshots/progress_tracking/sync_verification.json",
        help="検証結果出力先",
    )
    args = parser.parse_args()

    screenshots_file = Path(args.screenshots)
    jsonl_file = Path(args.jsonl)
    output_file = Path(args.output)

    print("📂 スクショ解析結果: " + str(screenshots_file))
    print("📂 JSONL進捗ログ: " + str(jsonl_file))

    with open(screenshots_file, encoding="utf-8") as f:
        screenshot_results = json.load(f)

    jsonl_events = load_jsonl_events(jsonl_file)

    print("🔍 スクショ: " + str(len(screenshot_results)) + " 枚")
    print("🔍 JSONLイベント: " + str(len(jsonl_events)) + " 件")

    verification_results: list[dict[str, Any]] = []
    for screenshot_data in screenshot_results:
        if "error" in screenshot_data:
            continue
        result = verify_single_screenshot(screenshot_data, jsonl_events)
        verification_results.append(result)
        if result["status"] == "issue":
            file_str = str(result.get("file"))
            issues_joined = ", ".join(result.get("issues", []))
            msg = "⚠️  " + file_str + ": " + issues_joined
            print(msg)

    regressions = detect_progress_regression(screenshot_results)

    issues_found_val = sum(1 for r in verification_results if r["status"] == "issue")
    progress_regressions_val = len(regressions)

    final_report = {
        "summary": {
            "total_screenshots": len(screenshot_results),
            "verified_screenshots": len(verification_results),
            "issues_found": issues_found_val,
            "progress_regressions": progress_regressions_val,
        },
        "verification_results": verification_results,
        "progress_regressions": regressions,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        from common.io_utils import write_json

        write_json(output_file, final_report, ensure_ascii=False, indent=2)
    except Exception:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)

    print("\n✅ 検証完了")
    print("📊 結果: " + str(output_file))
    print("🔴 問題検出: " + str(issues_found_val) + " 件")
    print("⚠️  進捗後退: " + str(progress_regressions_val) + " 件")

    if regressions:
        print("\n🔴 進捗後退の詳細:")
        for reg in regressions:
            left = "  " + str(reg.get("prev_progress")) + "% → "
            right = str(reg.get("curr_progress")) + "% (後退: "
            mid = str(reg.get("regression_amount")) + "% )"
            print(left + right + mid)
            left_file = "    " + str(reg.get("prev_file"))
            right_file = " → " + str(reg.get("curr_file"))
            print(left_file + right_file)


if __name__ == "__main__":
    main()
