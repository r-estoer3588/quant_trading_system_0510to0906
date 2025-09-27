"""バッチ処理とプロセスプールの共通ロジック。

全System*.pyで共有されるバッチ処理パターンを統一：
- プロセスプール処理
- プログレス/ログコールバック管理
- エラー集計とレポート
- バッチサイズ自動調整
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, TypeVar
from collections.abc import Callable


from common.utils import BatchSizeMonitor, resolve_batch_size

T = TypeVar("T")


def process_symbols_batch(
    symbols: list[str],
    process_func: Callable[[str], tuple[str, T | None]],
    *,
    batch_size: int | None = None,
    use_process_pool: bool = False,
    max_workers: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
    skip_callback: Callable[[str, str], None] | None = None,
    system_name: str = "unknown",
) -> tuple[dict[str, T], list[str]]:
    """シンボルリストのバッチ処理を実行する。

    Args:
        symbols: 処理対象シンボルリスト
        process_func: 各シンボルを処理する関数 (symbol) -> (symbol, result|None)
        batch_size: バッチサイズ（Noneの場合は自動調整）
        use_process_pool: プロセスプールを使用するかどうか
        max_workers: 最大ワーカー数
        progress_callback: プログレス報告用コールバック
        log_callback: ログ出力用コールバック
        skip_callback: エラー時のスキップ用コールバック
        system_name: システム名（ログ用）

    Returns:
        (成功結果辞書, エラーシンボルリスト)
    """
    if not symbols:
        return {}, []

    # バッチサイズ調整
    effective_batch_size = resolve_batch_size(batch_size, len(symbols))

    if log_callback:
        log_callback(
            f"{system_name}: Processing {len(symbols)} symbols "
            f"(batch_size={effective_batch_size}, process_pool={use_process_pool})"
        )

    # バッチサイズ監視
    monitor = BatchSizeMonitor(effective_batch_size)

    results = {}
    error_symbols = []

    if use_process_pool and len(symbols) > 1:
        # プロセスプール処理
        results, error_symbols = _process_with_pool(
            symbols,
            process_func,
            max_workers,
            progress_callback,
            skip_callback,
            monitor,
            system_name,
        )
    else:
        # シーケンシャル処理
        results, error_symbols = _process_sequential(
            symbols,
            process_func,
            progress_callback,
            skip_callback,
            monitor,
            system_name,
        )

    # 最終レポート
    if log_callback:
        log_callback(
            f"{system_name}: Completed {len(results)} symbols, " f"errors: {len(error_symbols)}"
        )
        if error_symbols and len(error_symbols) <= 10:
            log_callback(f"{system_name}: Error symbols: {error_symbols}")

    return results, error_symbols


def _process_with_pool(
    symbols: list[str],
    process_func: Callable[[str], tuple[str, T | None]],
    max_workers: int | None,
    progress_callback: Callable[[str], None] | None,
    skip_callback: Callable[[str, str], None] | None,
    monitor: BatchSizeMonitor,
    system_name: str,
) -> tuple[dict[str, T], list[str]]:
    """プロセスプールでの並列処理を実行する。"""
    results = {}
    error_symbols = []

    # CPUコア数に基づくワーカー数設定
    if max_workers is None:
        max_workers = min(len(symbols), os.cpu_count() or 4)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 全タスクを投入
        future_to_symbol = {executor.submit(process_func, symbol): symbol for symbol in symbols}

        # 完了順に結果を収集
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                returned_symbol, result = future.result()

                if result is not None:
                    results[returned_symbol] = result
                    if progress_callback:
                        progress_callback(f"Processed {returned_symbol}")
                else:
                    error_symbols.append(returned_symbol)
                    if skip_callback:
                        skip_callback(returned_symbol, f"{system_name}_process_error")

            except Exception as e:
                error_symbols.append(symbol)
                if skip_callback:
                    skip_callback(symbol, f"{system_name}_exception_{type(e).__name__}: {e}")

            # バッチサイズ監視更新
            monitor.update()

    return results, error_symbols


def _process_sequential(
    symbols: list[str],
    process_func: Callable[[str], tuple[str, T | None]],
    progress_callback: Callable[[str], None] | None,
    skip_callback: Callable[[str, str], None] | None,
    monitor: BatchSizeMonitor,
    system_name: str,
) -> tuple[dict[str, T], list[str]]:
    """シーケンシャル処理を実行する。"""
    results = {}
    error_symbols = []

    for symbol in symbols:
        try:
            returned_symbol, result = process_func(symbol)

            if result is not None:
                results[returned_symbol] = result
                if progress_callback:
                    progress_callback(f"Processed {returned_symbol}")
            else:
                error_symbols.append(returned_symbol)
                if skip_callback:
                    skip_callback(returned_symbol, f"{system_name}_process_error")

        except Exception as e:
            error_symbols.append(symbol)
            if skip_callback:
                skip_callback(symbol, f"{system_name}_exception_{type(e).__name__}: {e}")

        # バッチサイズ監視更新
        monitor.update()

    return results, error_symbols


def create_progress_reporter(
    total_count: int,
    system_name: str,
    log_callback: Callable[[str], None] | None = None,
) -> Callable[[str], None]:
    """プログレス報告関数を作成する。

    Args:
        total_count: 総処理数
        system_name: システム名
        log_callback: ログ出力用コールバック

    Returns:
        プログレス報告用関数
    """
    processed_count = 0

    def report_progress(message: str) -> None:
        nonlocal processed_count
        processed_count += 1

        if log_callback and processed_count % max(1, total_count // 10) == 0:
            progress_pct = (processed_count / total_count) * 100
            log_callback(
                f"{system_name}: Progress {processed_count}/{total_count} "
                f"({progress_pct:.1f}%) - {message}"
            )

    return report_progress


def aggregate_errors(error_symbols: list[str], system_name: str) -> dict[str, Any]:
    """エラーシンボルリストを集計してサマリーを作成する。

    Args:
        error_symbols: エラーシンボルリスト
        system_name: システム名

    Returns:
        エラーサマリー辞書
    """
    return {
        "system": system_name,
        "total_errors": len(error_symbols),
        "error_symbols": error_symbols[:50],  # 最初の50個のみ保持
        "truncated": len(error_symbols) > 50,
    }


__all__ = [
    "process_symbols_batch",
    "create_progress_reporter",
    "aggregate_errors",
]
