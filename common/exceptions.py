from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools
import logging
import threading
from typing import Any


class TradingError(Exception):
    """プロジェクト共通の上位例外。"""


class DataValidationError(TradingError):
    pass


class TaskTimeoutError(TradingError):
    pass


# エラーコード体系
class ErrorCode:
    """構造化エラーコード体系。"""

    # データ関連エラー
    DATA_LOAD_FAILED = "DATA001"
    DATA_VALIDATION_FAILED = "DATA002"
    DATA_MISSING_COLUMNS = "DATA003"
    DATA_INSUFFICIENT_ROWS = "DATA004"
    DATA_CACHE_ERROR = "DATA005"

    # システム関連エラー
    SYSTEM_CONFIG_ERROR = "SYS001"
    SYSTEM_MEMORY_ERROR = "SYS002"
    SYSTEM_TIMEOUT_ERROR = "SYS003"
    SYSTEM_API_ERROR = "SYS004"

    # 計算関連エラー
    CALC_INDICATOR_ERROR = "CALC001"
    CALC_SIGNAL_ERROR = "CALC002"
    CALC_ALLOCATION_ERROR = "CALC003"

    # UI関連エラー
    UI_PROGRESS_ERROR = "UI001"
    UI_DISPLAY_ERROR = "UI002"


class CodedError(TradingError):
    """エラーコード付きの例外。"""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(f"[{code}] {message}")


def log_with_code(
    logger: logging.Logger,
    level: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> None:
    """エラーコード付きでログを出力。"""
    formatted_msg = f"[{code}] {message}"
    if details:
        formatted_msg += f" | Details: {details}"
    logger.log(level, formatted_msg)


def handle_exceptions(
    *,
    logger: logging.Logger | None = None,
    reraise: bool = False,
    default: Any = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """関数を安全実行するデコレータ。
    - 例外はログに記録し、`reraise=False` の場合は `default` を返す。
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                log = logger or logging.getLogger(func.__module__)
                log.exception("Unhandled exception in %s: %s", func.__name__, e)
                if reraise:
                    raise
                return default

        return wrapper

    return decorator


def run_with_timeout(fn: Callable[..., Any], timeout: float, *args, **kwargs) -> Any:
    """別スレッドで関数を実行し、`timeout` 秒で打ち切り。"""
    result_container: list[Any] = []
    err_container: list[BaseException] = []

    def target():
        try:
            result_container.append(fn(*args, **kwargs))
        except BaseException as e:  # noqa: BLE001
            err_container.append(e)

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise TaskTimeoutError(f"timeout after {timeout} seconds: {fn!r}")
    if err_container:
        raise err_container[0]
    return result_container[0] if result_container else None


def map_with_timeout(
    fn: Callable[[Any], Any],
    iterable: Iterable[Any],
    *,
    max_workers: int = 8,
    per_item_timeout: float | None = None,
    return_exceptions: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[Any], list[tuple[Any, BaseException]]]:
    """並列mapで例外とタイムアウトを吸収して返すユーティリティ。
    戻り値: (results_list, errors_list[(input, exc), ...])
    """
    items = list(iterable)
    total = len(items)
    results: list[Any] = [None] * total
    errors: list[tuple[Any, BaseException]] = []

    def _call(idx_item: tuple[int, Any]):
        i, item = idx_item
        if per_item_timeout is None:
            return i, fn(item)
        return i, run_with_timeout(fn, per_item_timeout, item)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_call, (i, it)): i for i, it in enumerate(items)}
        for done, fut in enumerate(as_completed(futures), 1):
            try:
                i, val = fut.result()
                results[i] = val
            except BaseException as e:  # noqa: BLE001
                idx = futures[fut]
                if return_exceptions:
                    errors.append((items[idx], e))
                else:
                    raise
            finally:
                if progress:
                    try:
                        progress(done, total)
                    except Exception:
                        pass

    return results, errors
