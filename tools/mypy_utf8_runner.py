"""UTF-8 強制で mypy を実行する補助スクリプト。

Windows 日本語ロケール (cp932) 環境で設定ファイル/依存に UTF-8 特殊バイト (例: 0x85) が含まれ
UnicodeDecodeError を誘発するケースの暫定回避策。

使い方:
  python tools/mypy_utf8_runner.py core/system2.py core/system3.py

挙動:
  - 環境変数 PYTHONUTF8=1, PYTHONIOENCODING=utf-8 を一時的に付与
  - mypy モジュールをサブプロセス実行
  - 戻りコード/標準出力/エラーを転送

注意:
  - グローバル (mypy.ini) の既存設定を尊重
  - 特定ファイルのみチェックしたい場合、引数に列挙
  - 追加で簡易ミニ設定を使いたい場合は --config-file をそのまま渡せます
"""

from __future__ import annotations

import os
import subprocess
import sys


def main(argv: list[str]) -> int:
    if not argv:
        print("Usage: python tools/mypy_utf8_runner.py <files or options>")
        return 1

    env = os.environ.copy()
    # UTF-8 強制 (PEP 540 / PYTHONUTF8) + IO エンコーディング明示
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    # Windows のコードページ出力を抑制 (純粋 UTF-8 文字化け防止 - 任意)
    if os.name == "nt":
        # chcp 65001 はサブプロセスチェーンで副作用小さいため省略
        pass

    cmd = [sys.executable, "-m", "mypy", *argv]
    print("[mypy-utf8] executing:", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    except FileNotFoundError:
        print("mypy not installed. Run: pip install mypy", file=sys.stderr)
        return 2

    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        # stderr も UTF-8 としてそのまま出力
        print(proc.stderr, end="", file=sys.stderr)

    if proc.returncode == 0:
        print("[mypy-utf8] Success (exit=0)")
    else:
        print(f"[mypy-utf8] Finished with exit code {proc.returncode}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
