"""
簡易 i18n 自動適用ツール
- 指定ディレクトリ内の .py ファイルを走査
- st.button("..."), st.checkbox('...') 等の最初のリテラル引数を tr("...") へ置換
- ファイル先頭に `from common.i18n import tr` を追加（未定義の場合）
- dry-run で差分を出力。--apply 指定で実ファイルを更新（バックアップを .bak に作成）

使い方（PowerShell / CMD）:
  python tools/i18n_apply_patch.py --path common        # dry-run
  python tools/i18n_apply_patch.py --path common --apply
"""

from __future__ import annotations

import re
import argparse
from pathlib import Path
from typing import List

# 対象となる st.* ウィジェット関数名（最初の文字列引数を翻訳する想定）
WIDGETS = [
    "button",
    "checkbox",
    "title",
    "header",
    "subheader",
    "number_input",
    "text_input",
    "selectbox",
    "radio",
    "metric",
    "info",
    "warning",
    "error",
    "success",
    "write",
    "markdown",
]

# 正規表現: st.<widget>( <STRING_LITERAL>
# - グループ1: st.<widget>(
# - グループ2: 文字列リテラル（"..." / '...' / triple-quote を含む）
STR_PAT = r'([ \t]*st\.({widgets})\()\s*([urbfURBF]*("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"[^"\\\n]*(?:\\.[^"\\\n]*)*"|\'[^\'\\\n]*(?:\\.[^\'\\\n]*)*\'))'.format(
    widgets="|".join(WIDGETS)
)

RE = re.compile(STR_PAT, flags=re.MULTILINE)

IMPORT_LINE = "from common.i18n import tr\n"


def find_py_files(base: Path) -> List[Path]:
    return [
        p
        for p in base.rglob("*.py")
        if ".venv" not in str(p) and "site-packages" not in str(p)
    ]


def ensure_import(lines: List[str]) -> List[str]:
    joined = "".join(lines[:40])  # 先頭付近だけ確認
    if "from common.i18n import tr" in joined or "import common.i18n" in joined:
        return lines
    # 挿入位置: 他の imports の直後、モジュール docstring の後
    insert_at = 0
    for i, ln in enumerate(lines[:40]):
        if ln.strip().startswith("import ") or ln.strip().startswith("from "):
            insert_at = i + 1
    lines.insert(insert_at, IMPORT_LINE)
    return lines


def transform_content(text: str) -> tuple[str, int]:
    """
    対象パターンを tr(...) に置換。戻り値: (new_text, replacements_count)
    """

    def repl(m: re.Match) -> str:
        prefix = m.group(1)  # includes leading whitespace + st.widget(
        widget = m.group(2)
        string_lit = m.group(4)
        # すでに tr(...) に包まれている場合は無視
        before = m.group(0)
        if re.search(r"\btr\s*\(\s*" + re.escape(string_lit), before):
            return before
        return f"{prefix}tr({string_lit})"

    new_text, count = RE.subn(repl, text)
    return new_text, count


def process_file(path: Path, apply: bool = False) -> int:
    text = path.read_text(encoding="utf-8")
    new_text, count = transform_content(text)
    if count == 0:
        return 0
    if apply:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            path.rename(backup)
            backup.write_text(text, encoding="utf-8")
            # restore original name for writing new content
            path.write_text(new_text, encoding="utf-8")
        else:
            # overwrite original safely
            path.write_text(new_text, encoding="utf-8")
        # ensure import
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        lines = ensure_import(lines)
        path.write_text("".join(lines), encoding="utf-8")
    else:
        # dry-run: just print snippet
        print(f"[DRY ] {path} -> {count} replacements")
        # show first few diffs
        for m in RE.finditer(text):
            start = max(m.start() - 40, 0)
            end = min(m.end() + 40, len(text))
            print("..." + text[start:end].replace("\n", "\\n") + "...")
            break
    return count


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--path", "-p", required=True, help="target directory (relative to repo root)"
    )
    p.add_argument(
        "--apply", action="store_true", help="apply changes (otherwise dry-run)"
    )
    args = p.parse_args()

    base = Path(args.path)
    if not base.exists() or not base.is_dir():
        print("path not found:", base)
        return

    files = find_py_files(base)
    total = 0
    for f in files:
        try:
            cnt = process_file(f, apply=args.apply)
            total += cnt
        except Exception as e:
            print(f"error processing {f}: {e}")
    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"{mode}: total replacements = {total}")


if __name__ == "__main__":
    main()
