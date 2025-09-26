#!/usr/bin/env python3
"""
スクリプト実行時間を測定するためのベンチマーク用ユーティリティ
cache_daily_data.py、build_rolling_with_indicators.py、update_from_bulk_last_day.py
の実行時間を測定する。
"""

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent
SCRIPTS_DIR = ROOT_DIR / "scripts"


def measure_execution_time(
    command: list[str], description: str, timeout_seconds: int = 1800
) -> dict[str, Any]:
    """コマンドの実行時間を測定する"""
    logger.info(f"開始: {description}")
    logger.info(f"コマンド: {' '.join(command)}")

    start_time = time.time()

    try:
        # プロセス実行
        result = subprocess.run(
            command, cwd=ROOT_DIR, capture_output=True, text=True, timeout=timeout_seconds
        )

        end_time = time.time()
        duration = end_time - start_time

        success = result.returncode == 0

        logger.info(f"完了: {description} - {duration:.2f}秒 - {'成功' if success else '失敗'}")

        return {
            "description": description,
            "command": " ".join(command),
            "duration_seconds": round(duration, 2),
            "success": success,
            "return_code": result.returncode,
            "stdout_lines": len(result.stdout.splitlines()) if result.stdout else 0,
            "stderr_lines": len(result.stderr.splitlines()) if result.stderr else 0,
            "stdout_preview": result.stdout[:200] if result.stdout else "",
            "stderr_preview": result.stderr[:200] if result.stderr else "",
        }

    except subprocess.TimeoutExpired:
        end_time = time.time()
        duration = end_time - start_time
        logger.warning(f"タイムアウト: {description} - {duration:.2f}秒でタイムアウト")

        return {
            "description": description,
            "command": " ".join(command),
            "duration_seconds": round(duration, 2),
            "success": False,
            "return_code": -1,
            "error": "timeout",
            "timeout_seconds": timeout_seconds,
        }

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        logger.error(f"エラー: {description} - {duration:.2f}秒 - {e}")

        return {
            "description": description,
            "command": " ".join(command),
            "duration_seconds": round(duration, 2),
            "success": False,
            "return_code": -2,
            "error": str(e),
        }


def main():
    """メイン実行関数"""
    results = []

    # 測定対象のスクリプト（軽量版のテスト用）
    benchmarks = [
        {
            "command": ["python", "scripts/cache_daily_data.py", "--help"],
            "description": "cache_daily_data.py ヘルプ表示",
            "timeout": 30,
        },
        {
            "command": ["python", "scripts/build_rolling_with_indicators.py", "--help"],
            "description": "build_rolling_with_indicators.py ヘルプ表示",
            "timeout": 30,
        },
        {
            "command": ["python", "scripts/update_from_bulk_last_day.py", "--help"],
            "description": "update_from_bulk_last_day.py ヘルプ表示",
            "timeout": 30,
        },
        # 実際のデータ処理（少数銘柄で測定）
        {
            "command": [
                "python",
                "scripts/build_rolling_with_indicators.py",
                "--max-symbols",
                "10",
                "--workers",
                "2",
            ],
            "description": "build_rolling_with_indicators.py (10銘柄限定)",
            "timeout": 300,
        },
    ]

    # 各ベンチマークを実行
    for bench in benchmarks:
        result = measure_execution_time(
            bench["command"], bench["description"], bench.get("timeout", 1800)
        )
        results.append(result)

        # 間隔を空ける
        time.sleep(1)

    # 結果を整理して出力
    print("\n" + "=" * 80)
    print("実行時間測定結果サマリー")
    print("=" * 80)

    for result in results:
        status = "✓" if result["success"] else "✗"
        print(f"{status} {result['description']:50} {result['duration_seconds']:8.2f}秒")
        if not result["success"]:
            if "error" in result:
                print(f"   エラー: {result.get('error', '不明')}")
            print(f"   戻り値: {result.get('return_code', 'N/A')}")
        print()

    # JSONファイルに結果を保存
    output_file = ROOT_DIR / "benchmark_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"timestamp": time.time(), "results": results}, f, indent=2, ensure_ascii=False)

    logger.info(f"結果をJSONファイルに保存: {output_file}")


if __name__ == "__main__":
    main()
