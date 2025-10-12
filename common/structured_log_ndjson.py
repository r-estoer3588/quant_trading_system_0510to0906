"""NDJSON 構造化ログ書き出しユーティリティ。

設計方針:
- 遅延 open: 初回 write でファイルを開く
- 1 行 = 1 JSON オブジェクト (ensure_ascii=False)
- エラーは一度だけ警告し以後黙る (ログ出力経路を汚染しない)
- スレッドセーフ簡易対応: ロック (threading.Lock) で排他
- ローテーション等は最小実装 (今後拡張)。

環境変数:
STRUCTURED_LOG_NDJSON=1 で有効化 (呼び出し側制御)。
オプション:
STRUCTURED_LOG_NDJSON_DIR: 出力ディレクトリ (未指定時 settings.logs_dir/structured)
STRUCTURED_LOG_NDJSON_PREFIX: ファイル名プレフィックス (デフォルト run)
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Optional, TextIO

try:
    # 遅延 import を避け settings が無い極小テストでも動くようにする
    from common.config_loader import get_settings  # type: ignore
except Exception:  # pragma: no cover - フォールバック
    get_settings = None  # フォールバック: settings 取得不能時


@dataclass(slots=True)
class _WriterState:
    path: Path
    opened: bool = False
    error_once: bool = False


class NDJSONWriter:
    """NDJSON ファイルへ構造化ログを書き込む簡易 Writer."""

    def __init__(self, base_dir: Optional[Path] = None, prefix: str = "run") -> None:
        self._lock = threading.Lock()
        self._state: Optional[_WriterState] = None
        self._prefix = prefix
        self._base_dir = base_dir or self._resolve_default_dir()
        self._file: TextIO | None = None
        # バッファ設定
        self._buffer: list[str] = []
        self._buffer_lines = self._env_int("STRUCTURED_LOG_BUFFER_LINES", default=0)
        self._buffer_flush_ms = self._env_int(
            "STRUCTURED_LOG_BUFFER_FLUSH_MS", default=0
        )
        self._last_flush_time = time.time()
        # ローテーション設定
        self._max_mb = self._env_int("STRUCTURED_LOG_MAX_MB", default=0)
        self._max_lines = self._env_int("STRUCTURED_LOG_MAX_LINES", default=0)
        self._current_lines = 0
        self._part_index = 0

    def _resolve_default_dir(self) -> Path:
        # settings 経由で logs/structured 配下を作成
        if get_settings:
            try:
                settings = get_settings(create_dirs=True)
                root = Path(settings.logs_dir) / "structured"
                root.mkdir(parents=True, exist_ok=True)
                return root
            except Exception:
                pass
        # フォールバック: ./logs/structured
        root = Path("logs") / "structured"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _ensure_open(self) -> None:
        if self._state and self._state.opened:
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        suffix = f"_part{self._part_index}" if self._part_index else ""
        fname = f"{self._prefix}_{ts}{suffix}.ndjson"
        path = self._base_dir / fname
        self._state = _WriterState(path=path, opened=False, error_once=False)
        try:
            self._file = open(path, "a", encoding="utf-8")  # テキスト append
            self._state.opened = True
            self._current_lines = 0
        except Exception:
            # open 失敗時は以後黙る
            self._state.error_once = True

    def _env_int(self, name: str, default: int = 0) -> int:
        try:
            v = os.environ.get(name)
            if not v:
                return default
            return max(0, int(v))
        except Exception:
            return default

    def _should_time_flush(self) -> bool:
        if self._buffer_flush_ms <= 0:
            return False
        return (time.time() - self._last_flush_time) * 1000 >= self._buffer_flush_ms

    def _rotate_if_needed(self) -> None:
        if not self._file or not self._state:
            return
        try:
            size_ok = True
            if self._max_mb > 0:
                self._file.flush()
                sz = self._state.path.stat().st_size
                if sz >= self._max_mb * 1024 * 1024:
                    size_ok = False
            lines_ok = (
                True
                if self._max_lines <= 0
                else (self._current_lines < self._max_lines)
            )
            if size_ok and lines_ok:
                return
        except Exception:
            return
        # rotate
        try:
            self._file.close()
        except Exception:
            pass
        self._part_index += 1
        self._state = None
        self._ensure_open()

    def _flush_locked(self, force: bool = False) -> None:
        if not self._buffer:
            return
        if not force:
            if self._buffer_lines <= 0 and not self._should_time_flush():
                return
            if (
                self._buffer_lines > 0
                and len(self._buffer) < self._buffer_lines
                and not self._should_time_flush()
            ):
                return
        self._ensure_open()
        if not self._file or not self._state or not self._state.opened:
            self._buffer.clear()
            return
        try:
            self._file.write("\n".join(self._buffer) + "\n")
            self._current_lines += len(self._buffer)
        except Exception:
            pass
        finally:
            self._buffer.clear()
            self._last_flush_time = time.time()
            self._rotate_if_needed()

    def write(self, obj: dict[str, Any]) -> None:
        if not isinstance(obj, dict):  # 型防御
            return
        with self._lock:
            if self._state is None or not self._state.opened:
                self._ensure_open()
            if not self._state or not self._state.opened:
                return
            try:
                line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
                if self._buffer_lines > 0 or self._buffer_flush_ms > 0:
                    self._buffer.append(line)
                    self._flush_locked(force=False)
                else:
                    # 既存挙動: 即時書き出し
                    self._ensure_open()
                    if self._file and self._state and self._state.opened:
                        self._file.write(line + "\n")
                        self._current_lines += 1
                        self._file.flush()
                        self._rotate_if_needed()
            except Exception:
                if self._state and not self._state.error_once:
                    try:
                        print("[NDJSONWriter] write error; suppress further warnings")
                    except Exception:  # pragma: no cover
                        pass
                    self._state.error_once = True

    def close(self) -> None:
        with self._lock:
            # バッファを強制 flush
            try:
                self._flush_locked(force=True)
            except Exception:
                pass
            try:
                if self._file:
                    self._file.close()
            finally:
                self._file = None
                self._state = None


_global_ndjson_writer: Optional[NDJSONWriter] = None


def get_global_ndjson_writer() -> Optional[NDJSONWriter]:
    return _global_ndjson_writer


def maybe_init_global_writer(
    env_flag: str = "STRUCTURED_LOG_NDJSON",
) -> Optional[NDJSONWriter]:
    global _global_ndjson_writer
    if _global_ndjson_writer is not None:
        return _global_ndjson_writer
    flag = (os.environ.get(env_flag) or "").lower() in {"1", "true", "yes"}
    if not flag:
        return None
    prefix = os.environ.get("STRUCTURED_LOG_NDJSON_PREFIX") or "run"
    base_dir = None
    dir_env = os.environ.get("STRUCTURED_LOG_NDJSON_DIR")
    if dir_env:
        try:
            base_dir = Path(dir_env)
            base_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            base_dir = None
    _global_ndjson_writer = NDJSONWriter(base_dir=base_dir, prefix=prefix)
    return _global_ndjson_writer


def close_global_writer() -> None:
    global _global_ndjson_writer
    if _global_ndjson_writer:
        try:
            _global_ndjson_writer.close()
        finally:
            _global_ndjson_writer = None
