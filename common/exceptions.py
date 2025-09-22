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
