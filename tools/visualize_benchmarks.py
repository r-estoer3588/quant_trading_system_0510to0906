"""
ベンチマーク履歴の可視化ツール

使い方:
    python tools/visualize_benchmarks.py
    python tools/visualize_benchmarks.py --days 30  # 直近30日間
"""

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description="ベンチマーク履歴の可視化")
    parser.add_argument(
        "--days", type=int, default=14, help="表示する日数（デフォルト: 14日）"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/performance_trend.png"),
        help="出力ファイル名",
    )
    args = parser.parse_args()

    history_file = Path("benchmarks/history.jsonl")

    if not history_file.exists():
        print("❌ No benchmark history found")
        print(f"   Expected: {history_file}")
        return 1

    # 履歴読み込み
    history = []
    with open(history_file, encoding="utf-8") as f:
        for line in f:
            try:
                history.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not history:
        print("❌ No valid benchmark data found")
        return 1

    # 日付フィルタ
    cutoff = datetime.now() - timedelta(days=args.days)
    filtered = [h for h in history if datetime.fromisoformat(h["timestamp"]) >= cutoff]

    if not filtered:
        print(f"⚠️  No data in the last {args.days} days")
        filtered = history[-20:]  # 最新20件にフォールバック

    # matplotlib のインポート（遅延読み込み）
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("⚠️  matplotlib/pandas not installed. Showing text summary instead.\n")

        # テキストサマリー表示
        print(f"📊 Benchmark Summary (last {len(filtered)} runs):\n")

        for h in filtered[-10:]:  # 最新10件
            timestamp = datetime.fromisoformat(h["timestamp"])
            total_time = h.get("total_time", 0)
            print(
                f"  {timestamp:%Y-%m-%d %H:%M} | {total_time:.2f}s | {h.get('git_commit', 'N/A')[:8]}"
            )

        return 0

    # DataFrame 変換
    df = pd.DataFrame(filtered)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # フェーズごとのタイムライン
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Phase times (stacked area)
    phase_times = pd.DataFrame(df["phase_times"].tolist(), index=df["timestamp"])

    if not phase_times.empty:
        # Phase 別の積み上げエリアチャート
        phase_times.plot(kind="area", stacked=True, ax=ax1, alpha=0.7)
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Time (seconds)")
        ax1.set_title("Performance Breakdown by Phase")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

    # Total time trend
    ax2.plot(df["timestamp"], df["total_time"], marker="o", linewidth=2, markersize=4)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Total Time (seconds)")
    ax2.set_title("Total Execution Time Trend")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

    # 平均線を追加
    mean_time = df["total_time"].mean()
    ax2.axhline(
        y=mean_time,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Mean: {mean_time:.2f}s",
    )
    ax2.legend()

    plt.tight_layout()

    # 保存
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"✅ Chart saved: {args.output}")

    # 統計サマリー
    print(f"\n📊 Statistics (last {len(filtered)} runs):")
    print(f"   Mean:   {df['total_time'].mean():.2f}s")
    print(f"   Median: {df['total_time'].median():.2f}s")
    print(f"   Min:    {df['total_time'].min():.2f}s")
    print(f"   Max:    {df['total_time'].max():.2f}s")
    print(f"   Std:    {df['total_time'].std():.2f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
