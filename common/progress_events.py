"""
Progress events logging in JSONL format for real-time monitoring.

JSONLファイルに進捗イベントを記録し、UIでリアルタイム表示に使用する。
出力先: logs/progress_today.jsonl
"""

from __future__ import annotations

from datetime import datetime
import json
import logging
import os
from pathlib import Path
import threading
from typing import Any

from config.settings import get_settings

logger = logging.getLogger(__name__)


class ProgressEventEmitter:
    """
    進捗イベントをJSONL形式でログファイルに出力するクラス。

    使用方法:
        emitter = ProgressEventEmitter()
        emitter.emit("system1_start", {"symbol_count": 100})
        emitter.emit("system1_progress", {"processed": 50, "total": 100})
        emitter.emit("system1_complete", {"processed": 100})
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.enabled = os.getenv("ENABLE_PROGRESS_EVENTS", "1") == "1"
        self._log_file: Path | None = None
        self._file_handle: object | None = None
        self.logger = logging.getLogger(__name__)
        self._initialized = True

        if self.enabled:
            self._setup_log_file()

    def _setup_log_file(self) -> None:
        """ログファイルのパスを設定し、初期化する。"""
        try:
            settings = get_settings(create_dirs=True)
            logs_dir = Path(settings.LOGS_DIR)
            logs_dir.mkdir(parents=True, exist_ok=True)

            self._log_file = logs_dir / "progress_today.jsonl"

            # Reset file on initialization
            self.reset()

        except Exception as e:
            self.logger.error(f"Failed to setup progress event log file: {e}")
            self.enabled = False

    def emit(
        self, event_type: str, data: dict[str, Any] | None = None, level: str = "info"
    ) -> None:
        """
        進捗イベントをJSONLファイルに出力する。

        Args:
            event_type: イベントタイプ（例: "system1_start", "system1_progress"）
            data: イベントデータ（辞書形式）
            level: ログレベル（info, warning, error）
        """
        if not self.enabled or not self._log_file:
            return

        try:
            event_record = {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "level": level,
                "data": data or {},
            }

            # JSONLファイルに追記
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_record, ensure_ascii=False) + "\n")
                f.flush()  # リアルタイム表示のため即座にフラッシュ

        except Exception as e:
            self.logger.error(f"Failed to emit progress event: {e}")

    def reset(self) -> None:
        """ログファイルを初期化（既存内容をクリア）する。"""
        if not self.enabled or not self._log_file:
            return

        try:
            # ファイルをクリア
            with open(self._log_file, "w", encoding="utf-8") as f:
                initial_event = {
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "session_start",
                    "level": "info",
                    "data": {"message": "Progress logging session started"},
                }
                f.write(json.dumps(initial_event, ensure_ascii=False) + "\n")
                f.flush()

        except Exception as e:
            self.logger.error(f"Failed to reset progress event log: {e}")

    def emit_system_start(
        self,
        system_name: str,
        symbol_count: int = 0,
        additional_data: dict[str, Any] | None = None,
    ) -> None:
        """システム処理開始イベントを出力する便利メソッド。"""
        data = {"system": system_name, "symbol_count": symbol_count}
        if additional_data:
            data.update(additional_data)
        self.emit(f"{system_name}_start", data)

    def emit_system_progress(
        self,
        system_name: str,
        processed: int,
        total: int,
        additional_data: dict[str, Any] | None = None,
    ) -> None:
        """システム処理進捗イベントを出力する便利メソッド。"""
        data = {
            "system": system_name,
            "processed": processed,
            "total": total,
            "percentage": round(processed / total * 100, 1) if total > 0 else 0,
        }
        if additional_data:
            data.update(additional_data)
        self.emit(f"{system_name}_progress", data)

    def emit_system_complete(
        self,
        system_name: str,
        final_count: int = 0,
        additional_data: dict[str, Any] | None = None,
    ) -> None:
        """システム処理完了イベントを出力する便利メソッド。"""
        data = {"system": system_name, "final_count": final_count}
        if additional_data:
            data.update(additional_data)
        self.emit(f"{system_name}_complete", data)

    def emit_phase(
        self,
        phase_name: str,
        status: str = "start",
        additional_data: dict[str, Any] | None = None,
    ) -> None:
        """処理フェーズイベントを出力する便利メソッド。"""
        data = {"phase": phase_name, "status": status}
        if additional_data:
            data.update(additional_data)
        self.emit(f"phase_{phase_name}_{status}", data)


# グローバルインスタンス
_progress_emitter = ProgressEventEmitter()


def emit_progress(
    event_type: str, data: dict[str, Any] | None = None, level: str = "info"
) -> None:
    """
    グローバル進捗エミッターに進捗イベントを送信する。

    Args:
        event_type: イベントタイプ
        data: イベントデータ
        level: ログレベル
    """
    _progress_emitter.emit(event_type, data, level)


def reset_progress_log() -> None:
    """進捗ログをリセットする。"""
    _progress_emitter.reset()


def emit_system_start(system_name: str, symbol_count: int = 0, **kwargs) -> None:
    """システム開始イベントのショートカット。"""
    _progress_emitter.emit_system_start(system_name, symbol_count, kwargs)


def emit_system_progress(
    system_name: str, processed: int, total: int, **kwargs
) -> None:
    """システム進捗イベントのショートカット。"""
    _progress_emitter.emit_system_progress(system_name, processed, total, kwargs)


def emit_system_complete(system_name: str, final_count: int = 0, **kwargs) -> None:
    """システム完了イベントのショートカット。"""
    _progress_emitter.emit_system_complete(system_name, final_count, kwargs)


def emit_phase(phase_name: str, status: str = "start", **kwargs) -> None:
    """フェーズイベントのショートカット。"""
    _progress_emitter.emit_phase(phase_name, status, kwargs)
