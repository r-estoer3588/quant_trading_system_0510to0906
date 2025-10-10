from collections.abc import Callable
import logging
import logging.handlers
from pathlib import Path
import time
from typing import Any

from config.settings import Settings

# ===== SystemLogger: error_handling.py との統合 =====


class SystemLogger:
    """統一ロガークラス（error_handling.py との統合用）

    このクラスは common/error_handling.py::SystemErrorHandler と連携し、
    logger + log_callback の両方に統一されたログ出力を提供します。

    使用例:
        >>> from common.logging_utils import SystemLogger
        >>> sys_logger = SystemLogger.create("System3", log_callback=callback)
        >>> sys_logger.info("処理開始", symbol_count=100)
        >>> sys_logger.error("エラー発生", symbol="AAPL")

    Args:
        system_name: システム名（例: "System3", "CacheManager"）
        logger: Python標準のloggerインスタンス
        log_callback: オプションのコールバック関数
        compact_mode: コンパクトモード（環境変数で自動設定）
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

    @classmethod
    def create(
        cls,
        system_name: str,
        logger: logging.Logger | None = None,
        log_callback: Callable[[str], None] | None = None,
    ) -> "SystemLogger":
        """環境変数を考慮してSystemLoggerを作成

        COMPACT_TODAY_LOGS 環境変数が有効な場合、compact_mode=True

        Args:
            system_name: システム名
            logger: loggerインスタンス
            log_callback: オプションのコールバック

        Returns:
            SystemLogger インスタンス
        """
        from config.environment import get_env_config

        env = get_env_config()
        return cls(
            system_name=system_name,
            logger=logger,
            log_callback=log_callback,
            compact_mode=env.compact_logs,
        )

    def _format_message(self, message: str, **context: Any) -> str:
        """メッセージとコンテキストをフォーマット"""
        if not context:
            return f"{self.system_name}: {message}"
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        return f"{self.system_name}: {message} ({context_str})"

    def _log(self, level: int, message: str, **context: Any) -> None:
        """logger と log_callback の両方にログ出力"""
        formatted_msg = self._format_message(message, **context)
        self.logger.log(level, formatted_msg)

        if self.log_callback:
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
            except Exception:
                pass  # log_callback の失敗は無視

    def debug(self, message: str, **context: Any) -> None:
        """DEBUGレベルのログ出力"""
        self._log(logging.DEBUG, message, **context)

    def info(self, message: str, **context: Any) -> None:
        """INFOレベルのログ出力（compact_mode ではDEBUGに降格）"""
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
        """例外情報付きのERRORレベルログ出力"""
        formatted_msg = self._format_message(message, **context)
        self.logger.error(formatted_msg, exc_info=exc_info or True)

        if self.log_callback:
            exc_type = type(exc_info).__name__ if exc_info else "Exception"
            callback_msg = f"[ERROR] {formatted_msg} | Exception: {exc_type}"
            try:
                self.log_callback(callback_msg)
            except Exception:
                pass


# ===== 既存の関数（互換性維持） =====


def setup_logging(settings: Settings) -> logging.Logger:
    """ロギング設定を標準 logging で初期化して root ロガーを返す。
    - 日次ローテーション: rotation == "daily"
    - それ以外: サイズローテーション（MB 指定の例: "10 MB" は 10*1024*1024）
    """
    level = getattr(logging, settings.logging.level.upper(), logging.INFO)
    log_dir = Path(settings.LOGS_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / settings.logging.filename

    logger = logging.getLogger()
    logger.setLevel(level)

    # 既存ハンドラをクリア
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    rotation = settings.logging.rotation.lower()
    handler: logging.handlers.TimedRotatingFileHandler | logging.handlers.RotatingFileHandler
    if rotation == "daily":
        handler = logging.handlers.TimedRotatingFileHandler(
            filename=str(log_path), when="midnight", backupCount=7, encoding="utf-8"
        )
    else:
        # 例: "10 MB" -> 10485760
        size_bytes: int | None = None
        try:
            num = float(rotation.split()[0])
            unit = rotation.split()[1].lower() if len(rotation.split()) > 1 else "b"
            mult = 1
            if unit.startswith("k"):
                mult = 1024
            elif unit.startswith("m"):
                mult = 1024 * 1024
            elif unit.startswith("g"):
                mult = 1024 * 1024 * 1024
            size_bytes = int(num * mult)
        except Exception:
            size_bytes = 10 * 1024 * 1024
        handler = logging.handlers.RotatingFileHandler(
            filename=str(log_path), maxBytes=size_bytes, backupCount=5, encoding="utf-8"
        )

    handler.setFormatter(fmt)
    logger.addHandler(handler)

    # コンソールにも出す
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.debug("Logging initialized")
    return logger


def log_with_progress(
    i: int,
    total: int,
    start_time: float,
    *,
    prefix: str = "処理",
    batch: int = 50,
    log_func: Callable[[str], None] | None = None,
    progress_func: Callable[[float], None] | None = None,
    extra_msg: str | None = None,
    unit: str = "件",
    silent: bool = False,
) -> None:
    """ストリームリット/CLIの両方で使える共通進捗ログ。
    - log_func: 文字列を受け取る関数（例: `st.text`, `logger.info`）
    - progress_func: 0..1 の進捗率を受け取る関数（例: `st.progress`）
    - silent: True の場合、log_func へのメッセージ出力を抑制（進捗バーのみ更新）
    """
    if i % batch != 0 and i != total:
        return
    elapsed = time.time() - start_time
    remain = (elapsed / max(i, 1)) * (total - i) if total > 0 else 0
    msg = (
        f"{prefix}: {i}/{total} {unit} 完了 | "
        f"経過: {int(elapsed // 60)}分{int(elapsed % 60)}秒 / "
        f"残り: 約{int(remain // 60)}分{int(remain % 60)}秒"
    )
    if extra_msg:
        msg += f"\n{extra_msg}"
    if log_func and not silent:
        try:
            log_func(msg)
        except Exception:
            pass
    if progress_func:
        try:
            progress_func(i / total if total else 0.0)
        except Exception:
            pass
