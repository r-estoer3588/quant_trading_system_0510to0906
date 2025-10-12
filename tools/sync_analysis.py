"""
JSONLイベントとスクリーンショットの同期分析ツール

スクショのタイムスタンプとJSONLイベントを照合し、
UI表示が正しく同期しているかを検証する。

Usage:
    python tools/sync_analysis.py
"""

from datetime import datetime
import json
from pathlib import Path
import re


def load_jsonl_events(jsonl_path: Path) -> list[dict]:
    """JSONLファイルからイベントをロード"""
    events = []
    if not jsonl_path.exists():
        return events

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    return events


def parse_screenshot_timestamp(filename: str) -> datetime | None:
    """
    スクショファイル名からdatetimeオブジェクトを生成
    例: progress_20251013_065209_856.png -> 2025-10-13 06:52:09
    """
    match = re.match(r"progress_(\d{8})_(\d{6})_\d+\.png", filename)
    if match:
        date_str = match.group(1)
        time_str = match.group(2)
        dt_str = (
            f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} "
            f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
        )
        return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    return None


def find_nearest_event(screenshot_time: datetime, events: list[dict]) -> dict | None:
    """スクショの時刻に最も近いJSONLイベントを検索"""
    min_delta = None
    nearest = None

    for event in events:
        try:
            # JSONLのタイムスタンプ形式: "2025/10/13 6:52:07" または ISO形式
            ts_str = event.get("timestamp", "")
            if "/" in ts_str:
                # "2025/10/13 6:52:07" 形式
                event_time = datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S")
            else:
                # ISO形式
                event_time = datetime.fromisoformat(ts_str)

            delta = abs((screenshot_time - event_time).total_seconds())
            if min_delta is None or delta < min_delta:
                min_delta = delta
                nearest = event
        except Exception:
            continue

    return nearest


def analyze_sync():
    """メイン分析処理"""
    # JSONLイベントロード
    jsonl_path = Path("logs/progress_today.jsonl")
    events = load_jsonl_events(jsonl_path)

    print(f"📋 JSONLイベント: {len(events)} 件")

    # system_start/complete イベントのみ抽出
    system_events = [
        e for e in events if e.get("event_type") in ["system_start", "system_complete"]
    ]
    print(f"   うちsystem関連: {len(system_events)} 件\n")

    # スクショ一覧取得
    screenshot_dir = Path("screenshots/progress_tracking")
    screenshots = sorted(screenshot_dir.glob("progress_*.png"))

    print(f"📸 スクリーンショット: {len(screenshots)} 枚\n")

    # 同期分析
    print("=" * 80)
    print("同期分析結果")
    print("=" * 80)

    sync_results = []

    for screenshot in screenshots:
        ss_time = parse_screenshot_timestamp(screenshot.name)
        if not ss_time:
            continue

        # 最も近いイベントを検索
        nearest = find_nearest_event(ss_time, system_events)

        if nearest:
            event_type = nearest.get("event_type")
            data = nearest.get("data", {})
            system = data.get("system", "unknown")
            candidates = data.get("candidates")

            result = {
                "screenshot": screenshot.name,
                "screenshot_time": ss_time.strftime("%H:%M:%S"),
                "event_type": event_type,
                "system": system,
                "candidates": candidates,
                "event_timestamp": nearest.get("timestamp"),
            }
            sync_results.append(result)

    # 結果出力
    print(f"\n{'時刻':<10} {'スクショ':<35} {'イベント':<18} {'System':<8} {'候補':<6}")
    print("-" * 80)

    for r in sync_results[:20]:  # 最初の20件を表示
        cand_str = (
            str(r.get("candidates", "-")) if r.get("candidates") is not None else "-"
        )
        print(
            f"{r['screenshot_time']:<10} {r['screenshot']:<35} "
            f"{r['event_type']:<18} {r['system']:<8} {cand_str:<6}"
        )

    if len(sync_results) > 20:
        print(f"\n   ... 他 {len(sync_results) - 20} 件\n")

    # JSON出力
    output_path = Path("screenshots/progress_tracking/sync_analysis.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sync_results, f, indent=2, ensure_ascii=False)

    print(f"\n📊 詳細結果: {output_path}")

    # 各システムのスクショ枚数をカウント
    print("\n【システム別スクショ枚数】")
    system_counts = {}
    for r in sync_results:
        system = r.get("system", "unknown")
        system_counts[system] = system_counts.get(system, 0) + 1

    for system in sorted(system_counts.keys()):
        print(f"   {system}: {system_counts[system]} 枚")


if __name__ == "__main__":
    analyze_sync()
