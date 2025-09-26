"""
Rate-limited logging utilities for verbose log optimization.
"""

from __future__ import annotations

import logging
import time
from threading import Lock
from typing import Any


class RateLimitedLogger:
    """
    ログメッセージにレート制限を適用するクラス。

    同種のメッセージが短時間に重複出力されることを防ぐ。
    """

    def __init__(self, logger: logging.Logger, default_interval: float = 5.0):
        """
        Args:
            logger: 実際のロガー
            default_interval: デフォルトの制限間隔（秒）
        """
        self.logger = logger
        self.default_interval = default_interval
        self._last_logged: dict[str, float] = {}
        self._lock = Lock()

    def _should_log(self, message_key: str, interval: float) -> bool:
        """メッセージがログ出力可能かチェック。"""
        current_time = time.time()

        with self._lock:
            last_time = self._last_logged.get(message_key, 0.0)
            if current_time - last_time >= interval:
                self._last_logged[message_key] = current_time
                return True
            return False

    def debug_rate_limited(
        self,
        message: str,
        *args: Any,
        interval: float | None = None,
        message_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        レート制限付きのDEBUGログ出力。

        Args:
            message: ログメッセージ
            *args: メッセージのフォーマット引数
            interval: 制限間隔（デフォルトは default_interval）
            message_key: メッセージキー（デフォルトは message の最初の50文字）
            **kwargs: logger.debug への追加引数
        """
        interval = interval or self.default_interval
        message_key = message_key or message[:50]

        if self._should_log(message_key, interval):
            self.logger.debug(message, *args, **kwargs)

    def info_rate_limited(
        self,
        message: str,
        *args: Any,
        interval: float | None = None,
        message_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """レート制限付きのINFOログ出力。"""
        interval = interval or self.default_interval
        message_key = message_key or message[:50]

        if self._should_log(message_key, interval):
            self.logger.info(message, *args, **kwargs)

    def warning_rate_limited(
        self,
        message: str,
        *args: Any,
        interval: float | None = None,
        message_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """レート制限付きのWARNINGログ出力。"""
        interval = interval or self.default_interval
        message_key = message_key or message[:50]

        if self._should_log(message_key, interval):
            self.logger.warning(message, *args, **kwargs)

    def clear_history(self) -> None:
        """ログ履歴をクリアする。"""
        with self._lock:
            self._last_logged.clear()


def create_rate_limited_logger(
    logger_name: str, default_interval: float = 5.0
) -> RateLimitedLogger:
    """
    レート制限付きロガーを作成する便利関数。

    Args:
        logger_name: ロガー名
        default_interval: デフォルトの制限間隔

    Returns:
        RateLimitedLogger インスタンス
    """
    logger = logging.getLogger(logger_name)
    return RateLimitedLogger(logger, default_interval)


# 標準的なレート制限ロガーインスタンス
_rate_limited_logger = None


def get_global_rate_limited_logger() -> RateLimitedLogger:
    """グローバルなレート制限ロガーを取得。"""
    global _rate_limited_logger
    if _rate_limited_logger is None:
        _rate_limited_logger = create_rate_limited_logger("rate_limited", 3.0)
    return _rate_limited_logger
