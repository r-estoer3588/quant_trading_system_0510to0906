"""
並列I/O最適化プロトタイプ - キャッシュ読み込み性能改善

従来のシーケンシャル読み込みと以下アプローチの性能比較：
1. ThreadPoolExecutor並列読み込み
2. pyarrow.read_csv高速パーサー
3. バッチread→concat結合パターン
4. Feather形式対応（高速バイナリ）

目標：大量ファイル処理でのwall-clock時間短縮検証
"""

import gc
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from config.settings import get_settings

logger = logging.getLogger(__name__)

# Optional imports（環境により使用可否判定）
try:
    # import pyarrow.csv as pa_csv  # Imported within function scope to avoid F821 error
    PYARROW_AVAILABLE = False  # Set to False to disable PyArrow functionality
except ImportError:
    PYARROW_AVAILABLE = False
    logger.info("pyarrow not available - will use pandas only")


@dataclass
class IOBenchmarkResult:
    """I/O性能ベンチマーク結果"""

    method_name: str
    file_count: int
    total_size_mb: float
    wall_clock_seconds: float
    throughput_mb_per_sec: float
    memory_peak_mb: float
    success_rate: float
    error_messages: list[str]

    # データ品質指標
    total_rows: int
    total_columns: int
    average_columns_per_file: float

    # システム負荷指標
    cpu_time_seconds: float | None = None
    gc_collections: int | None = None

    def to_dict(self) -> dict:
        return {
            "method": self.method_name,
            "files": self.file_count,
            "size_mb": self.total_size_mb,
            "wall_time": self.wall_clock_seconds,
            "throughput": self.throughput_mb_per_sec,
            "memory_mb": self.memory_peak_mb,
            "success_rate": self.success_rate,
            "rows": self.total_rows,
            "columns": self.total_columns,
            "avg_cols": self.average_columns_per_file,
            "errors": len(self.error_messages),
        }


class IOOptimizationBenchmark:
    """I/O最適化ベンチマーク"""

    def __init__(
        self, max_workers: int = 4, chunk_size: int = 50, memory_monitoring: bool = True
    ):
        """
        Args:
            max_workers: ThreadPool最大ワーカー数
            chunk_size: バッチサイズ
            memory_monitoring: メモリ監視有効/無効
        """
        self.settings = get_settings(create_dirs=True)
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.memory_monitoring = memory_monitoring

        # メモリ監視用
        if memory_monitoring:
            try:
                import psutil

                self.process = psutil.Process()
                self.psutil_available = True
            except ImportError:
                logger.warning("psutil not available - memory monitoring disabled")
                self.psutil_available = False
        else:
            self.psutil_available = False

    def _get_memory_usage_mb(self) -> float:
        """現在のメモリ使用量（MB）"""
        if self.psutil_available:
            return self.process.memory_info().rss / (1024 * 1024)
        return 0.0

    def _get_file_sample(
        self, cache_dir: Path, sample_size: int | None = None
    ) -> list[tuple[Path, float]]:
        """ファイルサンプル取得（パス, サイズMB）"""
        if not cache_dir.exists():
            return []

        csv_files = list(cache_dir.glob("*.csv"))
        feather_files = list(cache_dir.glob("*.feather"))

        # CSVを優先、Featherが利用可能な場合は適宜選択
        files = csv_files if csv_files else feather_files

        if not files:
            logger.warning(f"No files found in {cache_dir}")
            return []

        # ファイルサイズ情報付与
        file_info = []
        for file_path in files:
            try:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                file_info.append((file_path, size_mb))
            except Exception:
                continue

        # サンプリング
        if sample_size and len(file_info) > sample_size:
            # サイズ順ソート後に均等サンプリング
            file_info.sort(key=lambda x: x[1])
            step = len(file_info) // sample_size
            file_info = file_info[::step][:sample_size]

        logger.info(f"Selected {len(file_info)} files from {cache_dir}")
        return file_info

    def _method_sequential_pandas(
        self, files: list[tuple[Path, float]]
    ) -> IOBenchmarkResult:
        """1. シーケンシャル pandas.read_csv"""
        method_name = "Sequential_pandas"
        start_time = time.time()
        initial_memory = self._get_memory_usage_mb()
        peak_memory = initial_memory

        dataframes = []
        total_size_mb = sum(size for _, size in files)
        errors = []
        total_rows = 0
        total_columns = 0

        try:
            for file_path, _ in files:
                try:
                    df = pd.read_csv(file_path)
                    dataframes.append(df)
                    total_rows += len(df)
                    total_columns += len(df.columns)

                    # メモリ監視
                    if self.memory_monitoring:
                        current_memory = self._get_memory_usage_mb()
                        peak_memory = max(peak_memory, current_memory)

                except Exception as e:
                    errors.append(f"{file_path.name}: {str(e)}")
                    continue

            # 結合（実際の処理想定）
            if dataframes:
                combined_df = pd.concat(dataframes, ignore_index=True)
                final_memory = self._get_memory_usage_mb()
                peak_memory = max(peak_memory, final_memory)
                del combined_df

        except Exception as e:
            errors.append(f"Concatenation failed: {str(e)}")

        wall_time = time.time() - start_time
        success_rate = (len(files) - len(errors)) / len(files) if files else 0.0
        throughput = total_size_mb / wall_time if wall_time > 0 else 0.0
        avg_columns = total_columns / len(files) if files else 0.0

        # ガベージコレクション
        del dataframes
        gc.collect()

        return IOBenchmarkResult(
            method_name=method_name,
            file_count=len(files),
            total_size_mb=total_size_mb,
            wall_clock_seconds=wall_time,
            throughput_mb_per_sec=throughput,
            memory_peak_mb=peak_memory - initial_memory,
            success_rate=success_rate,
            error_messages=errors,
            total_rows=total_rows,
            total_columns=total_columns,
            average_columns_per_file=avg_columns,
        )

    def _method_threaded_pandas(
        self, files: list[tuple[Path, float]]
    ) -> IOBenchmarkResult:
        """2. ThreadPool並列 pandas.read_csv"""
        method_name = f"Threaded_pandas_{self.max_workers}workers"
        start_time = time.time()
        initial_memory = self._get_memory_usage_mb()
        peak_memory = initial_memory

        dataframes = []
        total_size_mb = sum(size for _, size in files)
        errors = []
        total_rows = 0
        total_columns = 0

        def load_single_file(file_info: tuple[Path, float]) -> pd.DataFrame | None:
            """単一ファイル読み込み"""
            file_path, _ = file_info
            try:
                return pd.read_csv(file_path)
            except Exception as e:
                errors.append(f"{file_path.name}: {str(e)}")
                return None

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(load_single_file, file_info) for file_info in files
                ]

                for future in as_completed(futures):
                    df = future.result()
                    if df is not None:
                        dataframes.append(df)
                        total_rows += len(df)
                        total_columns += len(df.columns)

                    # メモリ監視
                    if self.memory_monitoring:
                        current_memory = self._get_memory_usage_mb()
                        peak_memory = max(peak_memory, current_memory)

            # 結合
            if dataframes:
                combined_df = pd.concat(dataframes, ignore_index=True)
                final_memory = self._get_memory_usage_mb()
                peak_memory = max(peak_memory, final_memory)
                del combined_df

        except Exception as e:
            errors.append(f"Threading failed: {str(e)}")

        wall_time = time.time() - start_time
        success_rate = (len(files) - len(errors)) / len(files) if files else 0.0
        throughput = total_size_mb / wall_time if wall_time > 0 else 0.0
        avg_columns = total_columns / len(files) if files else 0.0

        # クリーンアップ
        del dataframes
        gc.collect()

        return IOBenchmarkResult(
            method_name=method_name,
            file_count=len(files),
            total_size_mb=total_size_mb,
            wall_clock_seconds=wall_time,
            throughput_mb_per_sec=throughput,
            memory_peak_mb=peak_memory - initial_memory,
            success_rate=success_rate,
            error_messages=errors,
            total_rows=total_rows,
            total_columns=total_columns,
            average_columns_per_file=avg_columns,
        )

    def _method_pyarrow_csv(self, files: list[tuple[Path, float]]) -> IOBenchmarkResult:
        """3. pyarrow.csv高速パーサー"""
        method_name = "PyArrow_CSV"

        if not PYARROW_AVAILABLE:
            return IOBenchmarkResult(
                method_name=method_name,
                file_count=len(files),
                total_size_mb=sum(size for _, size in files),
                wall_clock_seconds=0.0,
                throughput_mb_per_sec=0.0,
                memory_peak_mb=0.0,
                success_rate=0.0,
                error_messages=["pyarrow not available"],
                total_rows=0,
                total_columns=0,
                average_columns_per_file=0.0,
            )

        # Import within function scope
        try:
            import pyarrow.csv as pa_csv
        except ImportError:
            return IOBenchmarkResult(
                method_name=method_name,
                file_count=len(files),
                total_size_mb=sum(size for _, size in files),
                wall_clock_seconds=0.0,
                throughput_mb_per_sec=0.0,
                memory_peak_mb=0.0,
                success_rate=0.0,
                error_messages=["pyarrow import failed"],
                total_rows=0,
                total_columns=0,
                average_columns_per_file=0.0,
            )

        start_time = time.time()
        initial_memory = self._get_memory_usage_mb()
        peak_memory = initial_memory

        dataframes = []
        total_size_mb = sum(size for _, size in files)
        errors = []
        total_rows = 0
        total_columns = 0

        try:
            for file_path, _ in files:
                try:
                    # pyarrow.csv で読み込み、pandasに変換
                    table = pa_csv.read_csv(file_path)
                    df = table.to_pandas()
                    dataframes.append(df)
                    total_rows += len(df)
                    total_columns += len(df.columns)

                    # メモリ監視
                    if self.memory_monitoring:
                        current_memory = self._get_memory_usage_mb()
                        peak_memory = max(peak_memory, current_memory)

                except Exception as e:
                    errors.append(f"{file_path.name}: {str(e)}")
                    continue

            # 結合
            if dataframes:
                combined_df = pd.concat(dataframes, ignore_index=True)
                final_memory = self._get_memory_usage_mb()
                peak_memory = max(peak_memory, final_memory)
                del combined_df

        except Exception as e:
            errors.append(f"PyArrow processing failed: {str(e)}")

        wall_time = time.time() - start_time
        success_rate = (len(files) - len(errors)) / len(files) if files else 0.0
        throughput = total_size_mb / wall_time if wall_time > 0 else 0.0
        avg_columns = total_columns / len(files) if files else 0.0

        # クリーンアップ
        del dataframes
        gc.collect()

        return IOBenchmarkResult(
            method_name=method_name,
            file_count=len(files),
            total_size_mb=total_size_mb,
            wall_clock_seconds=wall_time,
            throughput_mb_per_sec=throughput,
            memory_peak_mb=peak_memory - initial_memory,
            success_rate=success_rate,
            error_messages=errors,
            total_rows=total_rows,
            total_columns=total_columns,
            average_columns_per_file=avg_columns,
        )

    def _method_batched_concat(
        self, files: list[tuple[Path, float]]
    ) -> IOBenchmarkResult:
        """4. バッチread→concat最適化"""
        method_name = f"Batched_concat_{self.chunk_size}batch"
        start_time = time.time()
        initial_memory = self._get_memory_usage_mb()
        peak_memory = initial_memory

        all_dataframes = []
        total_size_mb = sum(size for _, size in files)
        errors = []
        total_rows = 0
        total_columns = 0

        try:
            # チャンク単位でバッチ処理
            for i in range(0, len(files), self.chunk_size):
                chunk_files = files[i : i + self.chunk_size]
                chunk_dfs = []

                # チャンク内読み込み
                for file_path, _ in chunk_files:
                    try:
                        df = pd.read_csv(file_path)
                        chunk_dfs.append(df)
                        total_rows += len(df)
                        total_columns += len(df.columns)
                    except Exception as e:
                        errors.append(f"{file_path.name}: {str(e)}")
                        continue

                # チャンク内結合
                if chunk_dfs:
                    chunk_combined = pd.concat(chunk_dfs, ignore_index=True)
                    all_dataframes.append(chunk_combined)
                    del chunk_dfs  # 早期メモリ解放

                    # メモリ監視
                    if self.memory_monitoring:
                        current_memory = self._get_memory_usage_mb()
                        peak_memory = max(peak_memory, current_memory)

            # 最終結合
            if all_dataframes:
                final_combined = pd.concat(all_dataframes, ignore_index=True)
                final_memory = self._get_memory_usage_mb()
                peak_memory = max(peak_memory, final_memory)
                del final_combined

        except Exception as e:
            errors.append(f"Batched processing failed: {str(e)}")

        wall_time = time.time() - start_time
        success_rate = (len(files) - len(errors)) / len(files) if files else 0.0
        throughput = total_size_mb / wall_time if wall_time > 0 else 0.0
        avg_columns = total_columns / len(files) if files else 0.0

        # クリーンアップ
        del all_dataframes
        gc.collect()

        return IOBenchmarkResult(
            method_name=method_name,
            file_count=len(files),
            total_size_mb=total_size_mb,
            wall_clock_seconds=wall_time,
            throughput_mb_per_sec=throughput,
            memory_peak_mb=peak_memory - initial_memory,
            success_rate=success_rate,
            error_messages=errors,
            total_rows=total_rows,
            total_columns=total_columns,
            average_columns_per_file=avg_columns,
        )

    def run_comprehensive_benchmark(
        self, profile: str = "rolling", sample_size: int = 20
    ) -> list[IOBenchmarkResult]:
        """包括的I/Oベンチマーク実行"""
        logger.info(
            f"Starting I/O benchmark - profile: {profile}, sample: {sample_size}"
        )

        # ファイルサンプル準備
        cache_dir = self.settings.DATA_CACHE_DIR / profile
        files = self._get_file_sample(cache_dir, sample_size)

        if not files:
            logger.error(f"No files available for benchmark in {cache_dir}")
            return []

        total_mb = sum(size for _, size in files)
        logger.info(f"Benchmark target: {len(files)} files, {total_mb:.2f}MB total")

        # 各手法でベンチマーク実行
        methods = [
            self._method_sequential_pandas,
            self._method_threaded_pandas,
            self._method_pyarrow_csv,
            self._method_batched_concat,
        ]

        results = []
        for method in methods:
            logger.info(f"Running {method.__name__}...")
            try:
                result = method(files)
                results.append(result)
                logger.info(
                    f"{result.method_name}: {result.wall_clock_seconds:.3f}s, {result.throughput_mb_per_sec:.2f}MB/s"
                )
            except Exception as e:
                logger.error(f"{method.__name__} failed: {e}")
                continue

        return results

    def generate_benchmark_report(self, results: list[IOBenchmarkResult]) -> dict:
        """ベンチマークレポート生成"""
        if not results:
            return {"error": "No benchmark results available"}

        # 最高性能の特定
        best_throughput = max(results, key=lambda r: r.throughput_mb_per_sec)
        fastest_wall_time = min(results, key=lambda r: r.wall_clock_seconds)
        lowest_memory = min(results, key=lambda r: r.memory_peak_mb)

        report = {
            "benchmark_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_methods": len(results),
                "target_file_count": results[0].file_count if results else 0,
                "total_size_mb": results[0].total_size_mb if results else 0,
            },
            "performance_summary": {
                "best_throughput": {
                    "method": best_throughput.method_name,
                    "value": best_throughput.throughput_mb_per_sec,
                    "unit": "MB/s",
                },
                "fastest_wall_time": {
                    "method": fastest_wall_time.method_name,
                    "value": fastest_wall_time.wall_clock_seconds,
                    "unit": "seconds",
                },
                "lowest_memory": {
                    "method": lowest_memory.method_name,
                    "value": lowest_memory.memory_peak_mb,
                    "unit": "MB",
                },
            },
            "detailed_results": [result.to_dict() for result in results],
            "recommendations": [],
        }

        # 推奨事項生成
        recommendations = []

        # スループット比較
        baseline_throughput = next(
            (r.throughput_mb_per_sec for r in results if "Sequential" in r.method_name),
            0,
        )
        if baseline_throughput > 0:
            for result in results:
                if result.method_name != "Sequential_pandas":
                    improvement = (
                        result.throughput_mb_per_sec / baseline_throughput - 1
                    ) * 100
                    if improvement > 10:
                        recommendations.append(
                            f"{result.method_name}: {improvement:.1f}%性能向上 - 採用推奨"
                        )

        # メモリ効率性
        if lowest_memory.memory_peak_mb < best_throughput.memory_peak_mb * 0.8:
            recommendations.append(
                f"{lowest_memory.method_name}: メモリ効率優秀 ({lowest_memory.memory_peak_mb:.1f}MB) - 大規模処理時推奨"
            )

        report["recommendations"] = recommendations
        return report

    def export_benchmark_results(
        self,
        results: list[IOBenchmarkResult],
        report: dict,
        output_dir: Path | None = None,
    ) -> tuple[Path, Path]:
        """ベンチマーク結果出力"""
        if output_dir is None:
            output_dir = self.settings.LOGS_DIR / "io_benchmark"
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # CSV出力（詳細結果）
        csv_path = output_dir / f"io_benchmark_results_{timestamp}.csv"
        if results:
            df = pd.DataFrame([result.to_dict() for result in results])
            # Use centralized helper to ensure UTF-8 and consistent behavior
            try:
                from common.io_utils import df_to_csv

                df_to_csv(df, csv_path, index=False)
            except Exception:
                # Fallback to pandas if helper is unavailable for some reason
                df.to_csv(csv_path, index=False)

        # JSON出力（レポート）
        json_path = output_dir / f"io_benchmark_report_{timestamp}.json"
        # Prefer centralized JSON writer for UTF-8 sanitization
        try:
            from common.io_utils import write_json

            write_json(json_path, report, ensure_ascii=False, indent=2)
        except Exception:
            import json

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

        return csv_path, json_path


def main():
    """CLI実行"""
    import argparse

    parser = argparse.ArgumentParser(description="I/O Optimization Benchmark")
    parser.add_argument(
        "--profile",
        choices=["base", "rolling", "full_backup"],
        default="rolling",
        help="ベンチマーク対象プロファイル",
    )
    parser.add_argument("--sample", type=int, default=15, help="サンプルファイル数")
    parser.add_argument("--workers", type=int, default=4, help="並列ワーカー数")
    parser.add_argument("--chunk-size", type=int, default=10, help="バッチサイズ")

    args = parser.parse_args()

    # ロギング設定
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("I/O最適化ベンチマーク開始")
    print(f"プロファイル: {args.profile}")
    print(f"サンプルサイズ: {args.sample}")
    print(f"並列ワーカー: {args.workers}")
    print(f"バッチサイズ: {args.chunk_size}")

    # ベンチマーク実行
    benchmark = IOOptimizationBenchmark(
        max_workers=args.workers, chunk_size=args.chunk_size, memory_monitoring=True
    )

    results = benchmark.run_comprehensive_benchmark(
        profile=args.profile, sample_size=args.sample
    )

    if not results:
        print("❌ ベンチマーク実行失敗")
        return

    # レポート生成・出力
    report = benchmark.generate_benchmark_report(results)
    csv_path, json_path = benchmark.export_benchmark_results(results, report)

    # 結果表示
    print("\n=== I/O性能ベンチマーク結果 ===")
    perf = report["performance_summary"]
    print(
        f"最高スループット: {perf['best_throughput']['method']} ({perf['best_throughput']['value']:.2f} MB/s)"
    )
    print(
        f"最速実行時間: {perf['fastest_wall_time']['method']} ({perf['fastest_wall_time']['value']:.3f}秒)"
    )
    print(
        f"最小メモリ: {perf['lowest_memory']['method']} ({perf['lowest_memory']['value']:.1f}MB)"
    )

    print("\n=== 各手法詳細 ===")
    for result in results:
        print(f"{result.method_name}:")
        print(f"  実行時間: {result.wall_clock_seconds:.3f}秒")
        print(f"  スループット: {result.throughput_mb_per_sec:.2f} MB/s")
        print(f"  メモリピーク: {result.memory_peak_mb:.1f}MB")
        print(f"  成功率: {result.success_rate * 100:.1f}%")
        print(f"  エラー: {len(result.error_messages)}件")

    # 推奨事項
    recommendations = report.get("recommendations", [])
    if recommendations:
        print("\n=== 推奨事項 ===")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")

    print("\n結果出力:")
    print(f"  詳細CSV: {csv_path}")
    print(f"  レポートJSON: {json_path}")


if __name__ == "__main__":
    main()
