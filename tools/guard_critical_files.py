from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


CRITICAL_PATHS = [
    Path("common/ui_components.py"),
    Path("common/ui_bridge.py"),
    Path("common/ui_tabs.py"),
    Path("app_integrated.py"),
]


def _staged_files() -> set[Path]:
    try:
        out = subprocess.check_output(["git", "diff", "--cached", "--name-only"], text=True)
    except Exception:
        return set()
    return {Path(p.strip()) for p in out.splitlines() if p.strip()}


def main() -> int:
    if os.getenv("ALLOW_CRITICAL_CHANGES") == "1":
        return 0
    staged = _staged_files()
    touched = [str(p) for p in CRITICAL_PATHS if p in staged]
    if not touched:
        return 0

    msg = (
        "保護対象ファイルへの変更が検出されました:\n  - "
        + "\n  - ".join(touched)
        + "\n\n意図した変更であれば、環境変数を付けて実行してください:\n"
        + "  ALLOW_CRITICAL_CHANGES=1 git commit -m \"...\"\n\n"
        + "一時的にガードを解除する場合のみ上記を使用し、通常は PR とレビューで変更してください。\n"
    )
    sys.stderr.write(msg)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

