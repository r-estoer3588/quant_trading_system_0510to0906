#!/usr/bin/env python3
"""Daily cache update pipeline wrapper script.

過渡期対応として、cache_daily_data.py と build_rolling_with_indicators.py を
順次実行するためのスクリプトです。

Usage:
    python scripts/update_cache_all.py                    # シリアル実行
    python scripts/update_cache_all.py --parallel         # rolling構築を並列実行
    python scripts/update_cache_all.py --workers 4        # 4並列で実行
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# プロジェクトルートをPYTHONPATHに追加
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def format_duration(seconds: float) -> str:
    """秒数を見やすい形式で表示（分または秒）。"""
    return f"{seconds/60:.1f}分" if seconds >= 60 else f"{seconds:.1f}秒"


def run_subprocess(cmd: list[str], description: str) -> float:
    """サブプロセスを実行して所要時間を返す。"""
    print(f"🚀 {description} 開始")

    start_time = time.time()
    try:
        subprocess.run(cmd, check=True, cwd=ROOT_DIR, capture_output=False)
        duration = time.time() - start_time
        print(f"✅ {description} 完了 (所要時間: {format_duration(duration)})")
        return duration
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(
            f"❌ {description} 失敗 (Exit Code: {e.returncode}, 所要時間: {format_duration(duration)})"
        )
        raise


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Daily cache update pipeline - "
            "cache_daily_data.py + build_rolling_with_indicators.py"
        )
    )
    parser.add_argument(
        "--parallel", action="store_true", help="build_rolling_with_indicatorsで並列処理を有効化"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="build_rolling_with_indicatorsのワーカー数指定 (0=デフォルト)",
    )
    parser.add_argument(
        "--skip-cache-daily",
        action="store_true",
        help="cache_daily_data.pyをスキップしてrolling構築のみ実行",
    )

    args = parser.parse_args()

    pipeline_start = time.time()
    start_dt = datetime.fromtimestamp(pipeline_start).strftime("%Y-%m-%d %H:%M:%S")
    print("🚀 Daily Cache Update Pipeline 開始")
    print(f"🕐 開始日時: {start_dt}")
    print(f" 作業ディレクトリ: {ROOT_DIR}")

    total_duration = 0.0
    duration1 = 0.0

    try:
        # Step 1: cache_daily_data.py
        if not args.skip_cache_daily:
            print("\n📥 Step 1: Daily data caching (cache_daily_data.py)")
            print("   ↳ EODHD API → full_backup/ + base/")

            cache_cmd = [sys.executable, "scripts/cache_daily_data.py"]
            duration1 = run_subprocess(cache_cmd, "cache_daily_data.py")
            total_duration += duration1
        else:
            print("\n⏭️ Step 1: cache_daily_data.py をスキップします")

        # Step 2: build_rolling_with_indicators.py
        print("\n🔁 Step 2: Rolling cache rebuild (build_rolling_with_indicators.py)")
        print("   ↳ full_backup/ → rolling/ (指標付き330日データ)")

        rolling_cmd = [sys.executable, "scripts/build_rolling_with_indicators.py"]

        # 並列処理設定をシンプルに
        if args.workers > 0:
            rolling_cmd.extend(["--workers", str(args.workers)])
            print(f"   🔧 並列処理: {args.workers} ワーカー")
        elif args.parallel:
            print("   🔧 並列処理: デフォルトワーカー数")
        else:
            rolling_cmd.extend(["--workers", "1"])
            print("   🔧 シリアル実行")

        duration2 = run_subprocess(rolling_cmd, "build_rolling_with_indicators.py")
        total_duration += duration2

        # 完了サマリー
        pipeline_end = time.time()
        end_dt = datetime.fromtimestamp(pipeline_end).strftime("%Y-%m-%d %H:%M:%S")
        # total_duration は個別ステップ合計、実測の壁時計時間との差分はオーバーヘッド
        wall_elapsed = pipeline_end - pipeline_start
        print(
            f"\n🎉 Daily Cache Update Pipeline 完了! (総所要時間: {format_duration(total_duration)})"
        )
        print(f"   🕐 開始日時: {start_dt}")
        print(f"   🕐 終了日時: {end_dt}")
        print(f"   ⏱️ 経過(壁時計): {format_duration(wall_elapsed)}  / ステップ合計: {format_duration(total_duration)}")
        if not args.skip_cache_daily:
            print(f"   📋 cache_daily_data: {format_duration(duration1)}")
        print(f"   📋 build_rolling: {format_duration(duration2)}")
        print("\n💡 次に実行できること:")
        print("   • python scripts/run_all_systems_today.py --parallel --save-csv")
        print("   • streamlit run app_integrated.py")

        return 0

    except subprocess.CalledProcessError:
        return 1
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
