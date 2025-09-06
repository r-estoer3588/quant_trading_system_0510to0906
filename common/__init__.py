"""Common utilities and components package.

このパッケージをインポートしたタイミングで標準出力／標準エラーを UTF-8
に再設定し、Windows などで発生する CP932 由来のエンコードエラーを回避する。
"""

from __future__ import annotations

import sys


def _ensure_utf8() -> None:
    """Reconfigure stdout/stderr to UTF-8 if possible."""

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            # `reconfigure` が利用できない環境ではそのまま継続
            pass


_ensure_utf8()
