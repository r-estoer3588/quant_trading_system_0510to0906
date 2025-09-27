#!/usr/bin/env python3
"""
並列I/O最適化ベンチマークテスト

複数のI/O手法を比較して最適な読み込みパターンを検証
"""

import logging
import sys
from pathlib import Path

# プロジェクトルートを認識
sys.path.append(str(Path(__file__).resolve().parent.parent))

from common.io_optimization_benchmark import IOOptimizationBenchmark


def test_io_optimization():
    """I/O最適化テスト実行"""

    # ロギング設定
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    print("=== 並列I/O最適化ベンチマークテスト ===")

    # ベンチマーク設定
    benchmark = IOOptimizationBenchmark(
        max_workers=3, chunk_size=5, memory_monitoring=True  # 軽量テスト用  # 小バッチサイズ
    )

    # テスト実行パラメータ
    profile = "rolling"  # 豊富なファイルが期待される
    sample_size = 10  # 軽量テスト

    print(f"対象プロファイル: {profile}")
    print(f"サンプルサイズ: {sample_size}")
    print(f"並列ワーカー: {benchmark.max_workers}")
    print(f"バッチサイズ: {benchmark.chunk_size}")

    try:
        print("\nベンチマーク実行開始...")
        results = benchmark.run_comprehensive_benchmark(profile=profile, sample_size=sample_size)

        if not results:
            print("⚠️ ベンチマーク結果が空です - ファイルが見つからない可能性があります")
            return

        print(f"\n{len(results)}種類の手法でベンチマーク完了")

        # レポート生成
        report = benchmark.generate_benchmark_report(results)
        csv_path, json_path = benchmark.export_benchmark_results(results, report)

        # 結果表示
        print("\n=== 性能比較結果 ===")
        baseline_time = None
        baseline_throughput = None

        for result in results:
            method = result.method_name
            wall_time = result.wall_clock_seconds
            throughput = result.throughput_mb_per_sec
            memory = result.memory_peak_mb
            success = result.success_rate * 100

            # 改善率計算（初期化）
            time_improvement = ""
            throughput_improvement = ""

            # ベースライン設定（Sequential）
            if "Sequential" in method:
                baseline_time = wall_time
                baseline_throughput = throughput
                print(f"{method} (ベースライン):")
            else:
                # 改善率計算
                if baseline_time and baseline_time > 0:
                    time_change = (baseline_time / wall_time - 1) * 100
                    time_improvement = f" ({time_change:+.1f}%)"
                if baseline_throughput and baseline_throughput > 0:
                    throughput_change = (throughput / baseline_throughput - 1) * 100
                    throughput_improvement = f" ({throughput_change:+.1f}%)"

                print(f"{method}:")

            print(f"  実行時間: {wall_time:.3f}秒{time_improvement}")
            print(f"  スループット: {throughput:.2f} MB/s{throughput_improvement}")
            print(f"  メモリピーク: {memory:.1f}MB")
            print(f"  成功率: {success:.1f}%")
            if result.error_messages:
                print(f"  エラー: {len(result.error_messages)}件")
            print()

        # パフォーマンスサマリー
        perf = report["performance_summary"]
        print("=== 最優秀結果 ===")
        print(
            f"最高スループット: {perf['best_throughput']['method']} "
            f"({perf['best_throughput']['value']:.2f} MB/s)"
        )
        print(
            f"最速実行: {perf['fastest_wall_time']['method']} "
            f"({perf['fastest_wall_time']['value']:.3f}秒)"
        )
        print(
            f"最小メモリ: {perf['lowest_memory']['method']} "
            f"({perf['lowest_memory']['value']:.1f}MB)"
        )

        # 推奨事項
        recommendations = report.get("recommendations", [])
        if recommendations:
            print("\n=== 推奨事項 ===")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        else:
            print("\n✅ 特定の手法が明確に優位ではありません")

        print("\n=== 詳細出力ファイル ===")
        print(f"CSV詳細: {csv_path}")
        print(f"JSONレポート: {json_path}")

        # pyarrow可用性チェック
        try:
            import pyarrow

            print(f"\n✅ pyarrow利用可能: バージョン {pyarrow.__version__}")
        except ImportError:
            print("\n⚠️ pyarrow未インストール - pip install pyarrow で高速化可能")

        print("\n✅ I/O最適化ベンチマークテスト完了")

    except Exception as e:
        logger.error(f"ベンチマーク実行中にエラー: {e}")
        raise


if __name__ == "__main__":
    try:
        test_io_optimization()
    except KeyboardInterrupt:
        print("\n⚠️ ユーザー中断")
    except Exception as e:
        print(f"❌ エラー: {e}")
        sys.exit(1)
