"""Error notification helper for scheduler tasks.

Provides utilities to send error notifications via Slack
when scheduler tasks fail.

Usage:
    from common.error_notifier import notify_error, run_with_error_handling

    notify_error("task_name", "Error message", traceback_str)

    # Or wrap a function
    run_with_error_handling("task_name", task_function)
"""

from __future__ import annotations

from datetime import datetime
import logging
import traceback
from typing import Any, Callable

logger = logging.getLogger(__name__)


def notify_error(
    task_name: str,
    error_message: str,
    traceback_str: str | None = None,
) -> None:
    """Send error notification via Slack.

    Args:
        task_name: Name of the task that failed
        error_message: Error message
        traceback_str: Optional traceback string
    """
    try:
        from common.notifier import create_notifier

        notifier = create_notifier(platform="slack", fallback=True)

        # Truncate traceback if too long
        if traceback_str and len(traceback_str) > 500:
            traceback_str = traceback_str[:500] + "\n... (truncated)"

        message = f"""🚨 **エラー発生**: {task_name}

❌ **エラー内容**
{error_message}

📍 **発生時刻**
{datetime.now().strftime('%Y-%m-%d %H:%M:%S JST')}"""

        if traceback_str:
            message += f"""

📝 **トレースバック**
```
{traceback_str}
```"""

        notifier.send(f"🚨 タスクエラー: {task_name}", message, channel=None)
        logger.info(f"Error notification sent for task: {task_name}")

    except Exception as e:
        logger.error(f"Failed to send error notification: {e}")


def run_with_error_handling(
    task_name: str,
    task_func: Callable[[], Any],
    notify_on_error: bool = True,
) -> tuple[bool, Any]:
    """Run a task function with error handling and notification.

    Args:
        task_name: Name of the task for logging/notification
        task_func: Function to execute
        notify_on_error: Whether to send Slack notification on error

    Returns:
        Tuple of (success, result_or_error)
    """
    try:
        result = task_func()
        return True, result
    except Exception as e:
        error_msg = str(e)
        tb_str = traceback.format_exc()

        logger.exception(f"Task {task_name} failed: {error_msg}")

        if notify_on_error:
            notify_error(task_name, error_msg, tb_str)

        return False, e


def create_error_handler(task_name: str) -> Callable:
    """Create a decorator for error handling.

    Usage:
        @create_error_handler("my_task")
        def my_task():
            ...
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e)
                tb_str = traceback.format_exc()
                logger.exception(f"Task {task_name} failed: {error_msg}")
                notify_error(task_name, error_msg, tb_str)
                raise

        return wrapper

    return decorator


if __name__ == "__main__":
    # Test error notification
    logging.basicConfig(level=logging.INFO)

    def failing_task():
        raise ValueError("Test error for notification")

    success, result = run_with_error_handling("test_task", failing_task)
    print(f"Success: {success}")
