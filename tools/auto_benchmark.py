"""
自動ベンチマーク実行ツール

コミット前にパフォーマンスベンチマークを実行し、回帰を検出。

使い方:
    python tools/auto_benchmark.py
    python tools/auto_benchmark.py --threshold 0.15  # 15%の劣化まで許容
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

BENCHMARK_HISTORY = Path("benchmarks/history.jsonl")
BENCHMARK_DIR = Path("benchmarks")


def run_benchmark() -> Optional[Dict]:
    """ベンチマーク実行"""
    print("🔍 Running performance benchmark...")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_all_systems_today.py",
            "--test-mode",
            "mini",
            "--skip-external",
            "--benchmark",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    if result.returncode != 0:
        print("❌ Benchmark failed:")
        print(result.stderr)
        return None

    # ベンチマーク結果を読み込み
    benchmark_files = list(Path("results_csv_test").glob("benchmark_*.json"))
    if not benchmark_files:
        print("⚠️  No benchmark file found")
        return None

    latest_benchmark = max(benchmark_files, key=lambda p: p.stat().st_mtime)
    with open(latest_benchmark, encoding="utf-8") as f:
        data = json.load(f)

    return data


def compare_with_baseline(
    current: Dict, baseline: Dict, threshold: float = 0.10
) -> List[Dict]:
    """ベースラインと比較（デフォルト10%以上の劣化を検出）"""
    regressions = []

    # フェーズごとの時間を比較
    current_phases = current.get("phase_times", {})
    baseline_phases = baseline.get("phase_times", {})

    for phase, current_time in current_phases.items():
        baseline_time = baseline_phases.get(phase, 0)

        if baseline_time > 0:
            regression_pct = (current_time - baseline_time) / baseline_time

            if regression_pct > threshold:
                regressions.append(
                    {
                        "phase": phase,
                        "baseline": baseline_time,
                        "current": current_time,
                        "regression_pct": regression_pct,
                    }
                )

    # 全体時間の比較
    current_total = current.get("total_time", 0)
    baseline_total = baseline.get("total_time", 0)

    if baseline_total > 0:
        total_regression = (current_total - baseline_total) / baseline_total

        if total_regression > threshold:
            regressions.append(
                {
                    "phase": "TOTAL",
                    "baseline": baseline_total,
                    "current": current_total,
                    "regression_pct": total_regression,
                }
            )

    return regressions


def get_baseline() -> Optional[Dict]:
    """過去7日間の中央値をベースラインとして使用"""
    if not BENCHMARK_HISTORY.exists():
        return None

    # 履歴読み込み
    history = []
    with open(BENCHMARK_HISTORY, encoding="utf-8") as f:
        for line in f:
            try:
                history.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if len(history) < 3:
        return None

    # 直近7日間
    recent = history[-7:]

    # フェーズごとの中央値を計算
    baseline = {"phase_times": {}, "total_time": 0}

    # 全フェーズ名を収集
    all_phases = set()
    for h in recent:
        all_phases.update(h.get("phase_times", {}).keys())

    # 中央値計算
    for phase in all_phases:
        times = [h["phase_times"].get(phase, 0) for h in recent]
        times = [t for t in times if t > 0]  # 0を除外

        if times:
            times.sort()
            mid = len(times) // 2
            if len(times) % 2 == 0:
                baseline["phase_times"][phase] = (times[mid - 1] + times[mid]) / 2
            else:
                baseline["phase_times"][phase] = times[mid]

    # 全体時間の中央値
    total_times = [h.get("total_time", 0) for h in recent if h.get("total_time", 0) > 0]
    if total_times:
        total_times.sort()
        mid = len(total_times) // 2
        if len(total_times) % 2 == 0:
            baseline["total_time"] = (total_times[mid - 1] + total_times[mid]) / 2
        else:
            baseline["total_time"] = total_times[mid]

    return baseline


def save_benchmark(data: Dict):
    """ベンチマーク結果を履歴に保存"""
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    # Git commit 取得（失敗しても続行）
    try:
        git_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        git_commit = "unknown"

    record = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_commit,
        "phase_times": data.get("phase_times", {}),
        "total_time": data.get("total_time", 0),
    }

    with open(BENCHMARK_HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    """メイン処理"""
    import argparse

    parser = argparse.ArgumentParser(description="自動ベンチマーク実行")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="許容する劣化率（デフォルト: 0.10 = 10%%）",
    )
    parser.add_argument(
        "--skip-interactive", action="store_true", help="対話的確認をスキップ"
    )
    args = parser.parse_args()

    # ベンチマーク実行
    current = run_benchmark()
    if not current:
        return 1

    print(f"✅ Benchmark completed: {current.get('total_time', 0):.2f}s\n")

    # ベースライン取得
    baseline = get_baseline()

    if baseline:
        # 回帰検出
        regressions = compare_with_baseline(current, baseline, threshold=args.threshold)

        if regressions:
            print(
                f"⚠️  Performance regressions detected (>{args.threshold:.0%} slower):\n"
            )
            for reg in regressions:
                print(f"  • {reg['phase']}: {reg['regression_pct']:+.1%} slower")
                print(
                    f"    Baseline: {reg['baseline']:.2f}s → Current: {reg['current']:.2f}s"
                )

            print(
                f"\n❌ Performance degraded by >{args.threshold:.0%}. Review changes."
            )

            if not args.skip_interactive:
                response = input("\nContinue anyway? (y/N): ")
                if response.lower() != "y":
                    return 1
            else:
                return 1
        else:
            print(
                f"✅ No performance regressions detected (threshold: {args.threshold:.0%})"
            )
    else:
        print("ℹ️  No baseline available (need 3+ historical runs)")
        print("   This run will be used as baseline for future comparisons.")

    # 結果を履歴に保存
    save_benchmark(current)
    print(f"\n📊 Benchmark saved to: {BENCHMARK_HISTORY}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
