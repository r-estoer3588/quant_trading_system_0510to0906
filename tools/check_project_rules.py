"""
プロジェクト固有のルールをチェック

使い方:
    python tools/check_project_rules.py core/system1.py
    python tools/check_project_rules.py common/*.py
"""

import re
import sys
from pathlib import Path
from typing import Dict, List

# プロジェクトルール定義
RULES = {
    "direct_csv_read": {
        "pattern": r"pd\.read_csv\(['\"]data_cache/",
        "message": "❌ CacheManager を使用せずに直接 CSV を読み込んでいます",
        "severity": "error",
        "suggestion": "common.cache_manager.CacheManager を使用してください",
    },
    "direct_feather_read": {
        "pattern": r"pd\.read_feather\(['\"]data_cache/",
        "message": "❌ CacheManager を使用せずに直接 Feather を読み込んでいます",
        "severity": "error",
        "suggestion": "common.cache_manager.CacheManager を使用してください",
    },
    "system7_spy_violation": {
        "pattern": r"# System7.*symbols.*!=.*['\"]SPY['\"]",
        "message": "⚠️  System7 は SPY 固定です",
        "severity": "warning",
        "files": ["core/system7.py"],
    },
    "env_direct_access": {
        "pattern": r"os\.environ\.get\(['\"](?!PYTHONPATH|PATH|HOME)",
        "message": "⚠️  環境変数は get_env_config() 経由でアクセスしてください",
        "severity": "warning",
        "exclude_files": ["config/environment.py", "config/settings.py"],
        "suggestion": "from config.environment import get_env_config",
    },
    "missing_type_hints": {
        "pattern": r"def \w+\([^)]*\):",
        "message": "ℹ️  型ヒントの追加を検討してください",
        "severity": "info",
        "exclude_files": ["tests/*", "tools/*"],
    },
    "hardcoded_paths": {
        "pattern": r"['\"]C:\\\\|['\"]c:\\\\|['\"]D:\\\\|['\"]d:\\\\",
        "message": "❌ ハードコードされたパスがあります",
        "severity": "error",
        "suggestion": "Path(__file__).parent または settings から取得してください",
    },
    "default_allocations_change": {
        "pattern": r"DEFAULT_(LONG|SHORT)_ALLOCATIONS\s*=",
        "message": "⚠️  DEFAULT_ALLOCATIONS の変更は慎重に行ってください",
        "severity": "warning",
        "files": ["core/final_allocation.py"],
    },
}


def check_file(file_path: Path) -> List[Dict]:
    """ファイルのルール違反をチェック"""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        return [
            {
                "file": str(file_path),
                "line": 0,
                "rule": "file_read_error",
                "message": f"❌ ファイル読み込みエラー: {e}",
                "severity": "error",
            }
        ]

    violations = []

    for rule_name, rule in RULES.items():
        # ファイル制限チェック
        if "files" in rule:
            if not any(file_path.match(pattern) for pattern in rule["files"]):
                continue

        # 除外ファイルチェック
        if "exclude_files" in rule:
            if any(file_path.match(pattern) for pattern in rule["exclude_files"]):
                continue

        # パターンマッチ
        matches = re.finditer(rule["pattern"], content, re.MULTILINE)
        for match in matches:
            line_num = content[: match.start()].count("\n") + 1

            violation = {
                "file": str(file_path),
                "line": line_num,
                "rule": rule_name,
                "message": rule["message"],
                "severity": rule["severity"],
            }

            if "suggestion" in rule:
                violation["suggestion"] = rule["suggestion"]

            violations.append(violation)

    return violations


def main(file_paths: List[str]) -> int:
    """メイン処理"""
    all_violations = []

    for file_path_str in file_paths:
        path = Path(file_path_str)

        if path.is_dir():
            # ディレクトリの場合は再帰的に .py ファイルを探す
            for py_file in path.rglob("*.py"):
                violations = check_file(py_file)
                all_violations.extend(violations)
        elif path.suffix == ".py":
            violations = check_file(path)
            all_violations.extend(violations)

    # レポート出力
    if all_violations:
        print("📋 Code Review Findings:\n")

        # 重要度別にソート
        severity_order = {"error": 0, "warning": 1, "info": 2}
        all_violations.sort(
            key=lambda v: (severity_order[v["severity"]], v["file"], v["line"])
        )

        for v in all_violations:
            icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}[v["severity"]]
            print(f"{icon} {v['file']}:{v['line']}")
            print(f"   {v['message']}")

            if "suggestion" in v:
                print(f"   💡 {v['suggestion']}")

            print()

        # サマリー
        errors = [v for v in all_violations if v["severity"] == "error"]
        warnings = [v for v in all_violations if v["severity"] == "warning"]
        infos = [v for v in all_violations if v["severity"] == "info"]

        print("📊 Summary:")
        print(f"   Errors:   {len(errors)}")
        print(f"   Warnings: {len(warnings)}")
        print(f"   Info:     {len(infos)}")

        # エラーがあれば終了コード 1
        if errors:
            print(f"\n❌ {len(errors)} error(s) found. Please fix before committing.")
            return 1
        elif warnings:
            print(f"\n⚠️  {len(warnings)} warning(s) found. Review recommended.")
            return 0
        else:
            print(f"\nℹ️  {len(infos)} informational item(s) found.")
            return 0
    else:
        print("✅ No rule violations found.")
        return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/check_project_rules.py <file_or_dir>...")
        print("\nExamples:")
        print("  python tools/check_project_rules.py core/system1.py")
        print("  python tools/check_project_rules.py common/")
        print("  python tools/check_project_rules.py core/ common/")
        sys.exit(1)

    sys.exit(main(sys.argv[1:]))
