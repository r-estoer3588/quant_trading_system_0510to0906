"""パフォーマンス最適化ユーティリティ。

バックテスト並列化、キャッシュ最適化、メモリプロファイリングの機能を提供。

特に backtest_utils.py と integrated_backtest.py との統合を簡素化。
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import logging
import os
import time
from typing import Any, Callable, TypeVar

import pandas as pd

logger = logging.getLogger(__name__)

# 型定義
T = TypeVar("T")
BacktestTaskFn = Callable[[str, Any], tuple[str, Any]]


def get_optimal_worker_count(
    *,
    use_process_pool: bool = True,
    max_workers: int | None = None,
    fallback_to_threads: bool = False,
) -> int:
    """最適なワーカー数を決定。

    Args:
        use_process_pool: ProcessPool を使用するか（True）/ ThreadPool（False）
        max_workers: 最大ワーカー数（override）
        fallback_to_threads: ProcessPool エラー時に ThreadPool にフォールバック

    Returns:
        最適なワーカー数
    """
    if max_workers is not None:
        return max(1, int(max_workers))

    cpu_count = os.cpu_count() or 4
    # I/O 待ちが多いので、保守的に cpu_count 70% 使用
    optimal = max(1, int(cpu_count * 0.7))
    # 上限: 8（過剰な thread/process 生成を回避）
    return min(optimal, 8)


class ParallelBacktestRunner:
    """バックテスト並列実行エンジン。

    ProcessPool または ThreadPool を使用して複数シンボルの
    バックテストを並列実行。

    使用例:
        runner = ParallelBacktestRunner(max_workers=4)
        results = runner.run_batch(symbols, backtest_fn)
        for symbol, result in results:
            print(f"{symbol}: {result}")
    """

    def __init__(
        self,
        *,
        max_workers: int | None = None,
        use_process_pool: bool = True,
        fallback_to_threads: bool = True,
        timeout: int | None = 300,
        verbose: bool = False,
    ):
        """Initialize parallel backtest runner.

        Args:
            max_workers: Max worker count (auto-detect if None)
            use_process_pool: Use ProcessPoolExecutor (True) or ThreadPoolExecutor
            fallback_to_threads: Fallback to threads if ProcessPool fails
            timeout: Task timeout in seconds (None for no timeout)
            verbose: Enable verbose logging
        """
        self.max_workers = get_optimal_worker_count(
            use_process_pool=use_process_pool, max_workers=max_workers
        )
        self.use_process_pool = use_process_pool
        self.fallback_to_threads = fallback_to_threads
        self.timeout = timeout
        self.verbose = verbose

    def run_batch(
        self,
        items: list[str | tuple[str, Any]],
        task_fn: BacktestTaskFn,
        *,
        progress_callback: Callable[[str, int, int], None] | None = None,
        error_callback: Callable[[str, Exception], None] | None = None,
    ) -> list[tuple[str, Any]]:
        """実行バッチ処理。

        Args:
            items: 処理対象アイテム（文字列 or (key, value) tuples）
            task_fn: タスク関数。(key, value) を受け取り (key, result) を返す
            progress_callback: 進捗通知関数 (key, completed, total)
            error_callback: エラー通知関数 (key, exception)

        Returns:
            [(key, result), ...] のリスト
        """
        # item の正規化
        items_normalized: list[tuple[str, Any]] = []
        for item in items:
            if isinstance(item, str):
                items_normalized.append((item, None))
            elif isinstance(item, tuple) and len(item) == 2:
                items_normalized.append(item)
            else:
                items_normalized.append((str(item), None))

        total = len(items_normalized)
        completed = 0
        results: list[tuple[str, Any]] = []

        # ProcessPool トライ (エラーはスキップして ThreadPool に fallback)
        if self.use_process_pool:
            executor_cls = ProcessPoolExecutor
        else:
            executor_cls = ThreadPoolExecutor
        try:
            with executor_cls(max_workers=self.max_workers) as executor:
                # タスク投入
                futures = {}
                for key, value in items_normalized:
                    future = executor.submit(task_fn, key, value)
                    futures[future] = key

                # 完了順に処理
                for future in as_completed(futures, timeout=self.timeout):
                    key = futures[future]
                    try:
                        key_out, result = future.result()
                        results.append((key_out, result))
                        completed += 1
                        if progress_callback:
                            progress_callback(key, completed, total)
                    except Exception as e:
                        if error_callback:
                            error_callback(key, e)
                        results.append((key, None))
                        completed += 1
                        if progress_callback:
                            progress_callback(key, completed, total)
                        if self.verbose:
                            logger.warning(f"Error in {key}: {e}")

        except Exception as pool_error:
            if self.fallback_to_threads and self.use_process_pool:
                if self.verbose:
                    msg = (
                        f"ProcessPool failed, falling back to ThreadPool: {pool_error}"
                    )
                    logger.info(msg)
                return self._run_batch_threads(
                    items_normalized, task_fn, progress_callback, error_callback
                )
            raise

        return results

    def _run_batch_threads(
        self,
        items_normalized: list[tuple[str, Any]],
        task_fn: BacktestTaskFn,
        progress_callback: Callable[[str, int, int], None] | None = None,
        error_callback: Callable[[str, Exception], None] | None = None,
    ) -> list[tuple[str, Any]]:
        """ThreadPool による実行（ProcessPool フォールバック）。"""
        total = len(items_normalized)
        completed = 0
        results: list[tuple[str, Any]] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for key, value in items_normalized:
                future = executor.submit(task_fn, key, value)
                futures[future] = key

            for future in as_completed(futures, timeout=self.timeout):
                key = futures[future]
                try:
                    key_out, result = future.result()
                    results.append((key_out, result))
                    completed += 1
                    if progress_callback:
                        progress_callback(key, completed, total)
                except Exception as e:
                    if error_callback:
                        error_callback(key, e)
                    results.append((key, None))
                    completed += 1
                    if progress_callback:
                        progress_callback(key, completed, total)

        return results


class PerformanceTimer:
    """パフォーマンス計測ユーティリティ。

    context manager で自動的に実行時間を計測・ログ出力。

    使用例:
        with PerformanceTimer("backtest", verbose=True):
            result = run_backtest(data)
    """

    def __init__(
        self,
        label: str,
        verbose: bool = False,
        threshold_ms: float | None = None,
    ):
        """Initialize performance timer.

        Args:
            label: Operation label for logging
            verbose: Print timing even if below threshold
            threshold_ms: Only log if duration > threshold_ms (None = always log)
        """
        self.label = label
        self.verbose = verbose
        self.threshold_ms = threshold_ms or 0.0
        self.start_time: float | None = None
        self.elapsed_ms: float | None = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.start_time is not None:
            end_time = time.perf_counter()
            self.elapsed_ms = (end_time - self.start_time) * 1000
            if self.elapsed_ms >= self.threshold_ms or self.verbose:
                logger.info(f"[PERF] {self.label}: {self.elapsed_ms:.2f}ms")

    def get_elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed_ms if self.elapsed_ms is not None else 0.0


class DataFrameCache:
    """シンプルな DataFrame キャッシュ（read-only view 提供）。

    複数スレッド/プロセスで同じ DataFrame に安全にアクセスするための
    基本的なキャッシュ層。
    """

    def __init__(self, max_size_mb: int = 500):
        """Initialize DataFrame cache.

        Args:
            max_size_mb: Maximum cache size in MB
        """
        self._cache: dict[str, pd.DataFrame] = {}
        self.max_size_mb = max_size_mb

    def put(self, key: str, df: pd.DataFrame) -> None:
        """キャッシュに DataFrame を格納（コピー）。"""
        self._cache[key] = df.copy()

    def get(self, key: str) -> pd.DataFrame | None:
        """キャッシュから DataFrame を取得（view 参照）。"""
        return self._cache.get(key)

    def get_or_compute(
        self, key: str, compute_fn: Callable[[], pd.DataFrame]
    ) -> pd.DataFrame:
        """キャッシュがあれば返す。なければ compute して格納後に返す。"""
        if key not in self._cache:
            self._cache[key] = compute_fn()
        return self._cache[key]

    def clear(self) -> None:
        """キャッシュをクリア。"""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


__all__ = [
    "get_optimal_worker_count",
    "ParallelBacktestRunner",
    "PerformanceTimer",
    "DataFrameCache",
]
