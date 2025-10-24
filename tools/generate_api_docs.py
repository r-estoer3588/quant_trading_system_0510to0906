"""
API ドキュメント自動生成ツール

Python モジュールの docstring から markdown ドキュメントを自動生成します。

使い方:
    python tools/generate_api_docs.py --module core.system1 --output docs/api/system1.md
    python tools/generate_api_docs.py --all --output-dir docs/api/
"""

import argparse
import importlib
import inspect
from pathlib import Path
import sys
from typing import Any


def extract_docstring(obj: Any) -> str:
    """オブジェクトから docstring を抽出"""
    doc = inspect.getdoc(obj)
    return doc if doc else ""


def format_signature(name: str, obj: Any) -> str:
    """関数/メソッドのシグネチャを整形"""
    try:
        sig = inspect.signature(obj)
        return f"{name}{sig}"
    except (ValueError, TypeError):
        return name


def generate_function_doc(name: str, func: Any) -> str:
    """関数のドキュメントを生成"""
    doc = extract_docstring(func)
    sig = format_signature(name, func)

    return f"""### `{sig}`

{doc}

"""


def generate_class_doc(name: str, cls: Any) -> str:
    """クラスのドキュメントを生成"""
    doc = extract_docstring(cls)

    markdown = f"""## クラス: `{name}`

{doc}

### メソッド

"""

    # メソッドを抽出
    methods = []
    for method_name, method in inspect.getmembers(cls, inspect.isfunction):
        if not method_name.startswith("_") or method_name == "__init__":
            methods.append((method_name, method))

    for method_name, method in methods:
        method_doc = extract_docstring(method)
        method_sig = format_signature(method_name, method)

        markdown += f"""#### `{method_sig}`

{method_doc}

"""

    return markdown


def generate_module_doc(module_name: str, output_path: Path) -> bool:
    """モジュールのドキュメントを生成"""
    try:
        # モジュールをインポート
        module = importlib.import_module(module_name)

        # ドキュメント生成開始
        markdown = f"""# {module_name}

{extract_docstring(module)}

---

"""

        # クラスを抽出
        classes = []
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module_name:
                classes.append((name, obj))

        if classes:
            markdown += "## クラス一覧\n\n"
            for name, cls in classes:
                markdown += generate_class_doc(name, cls)

        # 関数を抽出
        functions = []
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if obj.__module__ == module_name and not name.startswith("_"):
                functions.append((name, obj))

        if functions:
            markdown += "## 関数一覧\n\n"
            for name, func in functions:
                markdown += generate_function_doc(name, func)

        # ファイルに保存
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")

        print(f"✅ {module_name} のドキュメントを生成しました: {output_path}")
        return True

    except Exception as e:
        print(f"❌ {module_name} のドキュメント生成に失敗: {e}")
        return False


def generate_all_docs(output_dir: Path):
    """全モジュールのドキュメントを生成"""
    modules = [
        # Core systems
        ("core.system1", "system1.md"),
        ("core.system2", "system2.md"),
        ("core.system3", "system3.md"),
        ("core.system4", "system4.md"),
        ("core.system5", "system5.md"),
        ("core.system6", "system6.md"),
        ("core.system7", "system7.md"),
        ("core.final_allocation", "final_allocation.md"),
        # Common utilities
        ("common.cache_manager", "cache_manager.md"),
        ("common.indicator_access", "indicator_access.md"),
        ("common.system_diagnostics", "system_diagnostics.md"),
        ("common.today_filters", "today_filters.md"),
        ("common.today_signals", "today_signals.md"),
        ("common.integrated_backtest", "integrated_backtest.md"),
        # Strategies
        ("strategies.system1_strategy", "strategies/system1_strategy.md"),
        ("strategies.system2_strategy", "strategies/system2_strategy.md"),
        ("strategies.system3_strategy", "strategies/system3_strategy.md"),
        ("strategies.system4_strategy", "strategies/system4_strategy.md"),
        ("strategies.system5_strategy", "strategies/system5_strategy.md"),
        ("strategies.system6_strategy", "strategies/system6_strategy.md"),
        ("strategies.system7_strategy", "strategies/system7_strategy.md"),
    ]

    success_count = 0
    total_count = len(modules)

    for module_name, output_file in modules:
        output_path = output_dir / output_file
        if generate_module_doc(module_name, output_path):
            success_count += 1

    print(
        f"\n📊 結果: {success_count}/{total_count} モジュールのドキュメントを生成しました"
    )


def main():
    parser = argparse.ArgumentParser(description="API ドキュメント自動生成")
    parser.add_argument(
        "--module", type=str, help="生成対象のモジュール名（例: core.system1）"
    )
    parser.add_argument("--output", type=Path, help="出力ファイルパス")
    parser.add_argument(
        "--all", action="store_true", help="全モジュールのドキュメントを生成"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/api"),
        help="出力ディレクトリ（--all 使用時）",
    )
    args = parser.parse_args()

    if args.all:
        print("📚 全モジュールのドキュメントを生成中...")
        generate_all_docs(args.output_dir)
    elif args.module and args.output:
        generate_module_doc(args.module, args.output)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
