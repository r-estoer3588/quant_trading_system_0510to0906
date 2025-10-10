"""統一エラーハンドリングフレームワーク

このモジュールは、logger.error/log_callback/print の混在を解消し、
全システムで統一されたエラーハンドリングを提供します。

Usage:
    from common.error_handling import SystemErrorHandler, QuantTradingError

    # エラーハンドラーの作成
    handler = SystemErrorHandler(
        system_name="System3",
        logger=logger,
        log_callback=log_callback  # Optional
    )

    # エラーログ出力（logger + log_callback の両方に出力）
    handler.error("データロードに失敗しました", symbol="AAPL")

    # 例外発生
    raise DataError("キャッシュファイルが見つかりません", symbol="AAPL")
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

# 既存の共通例外を継承
from common.exceptions import TradingError

# ===== カスタム例外階層 =====


class QuantTradingError(TradingError):
    """量的トレーディングシステム全体の基底例外

    Attributes:
        message: エラーメッセージ
        context: エラー発生時のコンテキスト情報（銘柄名・システム番号等）
    """

    def __init__(self, message: str, **context: Any):
        self.message = message
        self.context = context
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """メッセージとコンテキストをフォーマット"""
        if not self.context:
            return self.message

        context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
        return f"{self.message} ({context_str})"


class DataError(QuantTradingError):
    """データロード・キャッシュ関連のエラー

    使用例:
        - キャッシュファイルが見つからない
        - データ形式が不正
        - 必須カラムが欠落
        - データ行数が不足
    """


class CalculationError(QuantTradingError):
    """指標計算・シグナル生成関連のエラー

    使用例:
        - 指標計算中の数値エラー
        - ランキング処理の失敗
        - Setup条件の評価エラー
    """


class SystemError(QuantTradingError):
    """システム固有のエラー

    使用例:
        - システム設定の不備
        - 環境変数の不正
        - API呼び出しの失敗
    """


class AllocationError(QuantTradingError):
    """配分計算関連のエラー

    使用例:
        - ポジションサイズ計算の失敗
        - 配分合計の超過
        - ATR計算の失敗
    """


class ValidationError(QuantTradingError):
    """データバリデーション関連のエラー

    使用例:
        - Setup/Filter列の不整合
        - 重複銘柄の検出
        - 異常値（負のシェア数等）
    """


# ===== 統一エラーハンドラー =====


class SystemErrorHandler:
    """システム全体で統一されたエラーハンドリング

    logger と log_callback の両方に出力し、エラーハンドリングを統一します。

    Args:
        system_name: システム名（例: "System3", "CacheManager"）
        logger: Python標準のloggerインスタンス
        log_callback: オプションのコールバック関数（Streamlit UI等への通知用）
        compact_mode: コンパクトモード（詳細をDEBUGレベルに落とす）

    Example:
        >>> handler = SystemErrorHandler("System3", logger, log_callback)
        >>> handler.info("処理開始", symbol_count=100)
        >>> handler.warning("キャッシュミス", symbol="AAPL")
        >>> handler.error("データロード失敗", symbol="TSLA", reason="file not found")
    """

    def __init__(
        self,
        system_name: str,
        logger: logging.Logger | None = None,
        log_callback: Callable[[str], None] | None = None,
        compact_mode: bool = False,
    ):
        self.system_name = system_name
        self.logger = logger or logging.getLogger(__name__)
        self.log_callback = log_callback
        self.compact_mode = compact_mode

    def _format_message(self, message: str, **context: Any) -> str:
        """メッセージとコンテキストをフォーマット"""
        if not context:
            return f"{self.system_name}: {message}"

        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        return f"{self.system_name}: {message} ({context_str})"

    def _log(self, level: int, message: str, **context: Any) -> None:
        """logger と log_callback の両方にログ出力"""
        formatted_msg = self._format_message(message, **context)

        # logger に出力
        self.logger.log(level, formatted_msg)

        # log_callback にも出力（存在する場合）
        if self.log_callback:
            # ログレベルに応じたプレフィックス
            level_prefix = {
                logging.DEBUG: "[DEBUG]",
                logging.INFO: "[INFO]",
                logging.WARNING: "[WARN]",
                logging.ERROR: "[ERROR]",
                logging.CRITICAL: "[CRITICAL]",
            }.get(level, "")

            if level_prefix:
                callback_msg = f"{level_prefix} {formatted_msg}"
            else:
                callback_msg = formatted_msg

            try:
                self.log_callback(callback_msg)
            except Exception as e:
                # log_callback の失敗は無視（無限ループ防止）
                self.logger.debug(f"log_callback failed: {e}", exc_info=True)

    def debug(self, message: str, **context: Any) -> None:
        """DEBUGレベルのログ出力"""
        self._log(logging.DEBUG, message, **context)

    def info(self, message: str, **context: Any) -> None:
        """INFOレベルのログ出力"""
        if self.compact_mode:
            self._log(logging.DEBUG, message, **context)
        else:
            self._log(logging.INFO, message, **context)

    def warning(self, message: str, **context: Any) -> None:
        """WARNINGレベルのログ出力"""
        self._log(logging.WARNING, message, **context)

    def error(self, message: str, **context: Any) -> None:
        """ERRORレベルのログ出力"""
        self._log(logging.ERROR, message, **context)

    def critical(self, message: str, **context: Any) -> None:
        """CRITICALレベルのログ出力"""
        self._log(logging.CRITICAL, message, **context)

    def exception(self, message: str, exc_info: Exception | None = None, **context: Any) -> None:
        """例外情報付きのERRORレベルログ出力

        Args:
            message: エラーメッセージ
            exc_info: 例外インスタンス（Noneの場合は現在の例外を使用）
            **context: コンテキスト情報
        """
        formatted_msg = self._format_message(message, **context)

        # logger に例外情報付きで出力
        self.logger.error(formatted_msg, exc_info=exc_info or True)

        # log_callback には例外の型と簡潔なメッセージのみ
        if self.log_callback:
            exc_type = type(exc_info).__name__ if exc_info else "Exception"
            callback_msg = f"[ERROR] {formatted_msg} | Exception: {exc_type}"
            try:
                self.log_callback(callback_msg)
            except Exception as e:
                self.logger.debug(f"log_callback failed: {e}", exc_info=True)


# ===== ヘルパー関数 =====


def create_handler_from_env(
    system_name: str,
    logger: logging.Logger | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> SystemErrorHandler:
    """環境変数を考慮してSystemErrorHandlerを作成

    COMPACT_TODAY_LOGS 環境変数が有効な場合、compact_mode=True

    Args:
        system_name: システム名
        logger: loggerインスタンス
        log_callback: オプションのコールバック

    Returns:
        SystemErrorHandler インスタンス
    """
    from config.environment import get_env_config

    env = get_env_config()
    return SystemErrorHandler(
        system_name=system_name,
        logger=logger,
        log_callback=log_callback,
        compact_mode=env.compact_logs,
    )


def handle_data_error(
    handler: SystemErrorHandler,
    operation: str,
    symbol: str | None = None,
    exc: Exception | None = None,
) -> None:
    """データエラーの統一ハンドリング

    Args:
        handler: SystemErrorHandler インスタンス
        operation: 操作名（例: "データロード", "キャッシュ読み込み"）
        symbol: 銘柄コード（オプション）
        exc: 例外インスタンス（オプション）
    """
    context = {}
    if symbol:
        context["symbol"] = symbol

    if exc:
        handler.exception(
            f"{operation}に失敗しました",
            exc_info=exc,
            **context,
        )
    else:
        handler.error(f"{operation}に失敗しました", **context)


def handle_calculation_error(
    handler: SystemErrorHandler,
    operation: str,
    symbol: str | None = None,
    indicator: str | None = None,
    exc: Exception | None = None,
) -> None:
    """計算エラーの統一ハンドリング

    Args:
        handler: SystemErrorHandler インスタンス
        operation: 操作名（例: "指標計算", "シグナル生成"）
        symbol: 銘柄コード（オプション）
        indicator: 指標名（オプション）
        exc: 例外インスタンス（オプション）
    """
    context = {}
    if symbol:
        context["symbol"] = symbol
    if indicator:
        context["indicator"] = indicator

    if exc:
        handler.exception(
            f"{operation}に失敗しました",
            exc_info=exc,
            **context,
        )
    else:
        handler.error(f"{operation}に失敗しました", **context)


# ===== 既存コードとの互換性 =====


def log_error_safe(
    message: str,
    logger: logging.Logger | None = None,
    log_callback: Callable[[str], None] | None = None,
    **context: Any,
) -> None:
    """既存コードとの互換性のための簡易ログ関数

    Args:
        message: エラーメッセージ
        logger: loggerインスタンス
        log_callback: オプションのコールバック
        **context: コンテキスト情報
    """
    if logger:
        context_str = f" ({', '.join(f'{k}={v}' for k, v in context.items())})" if context else ""
        logger.error(f"{message}{context_str}")

    if log_callback:
        try:
            log_callback(f"[ERROR] {message}")
        except Exception:
            pass  # log_callback の失敗は無視
