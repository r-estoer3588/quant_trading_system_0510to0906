"""Compact log の行数検証ユーティリティ。

COMPACT_TODAY_LOGS=1 時のログが過剰にならないか検証します。
Mini モード時の基準値超過時は警告を出力。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


def count_log_lines(log_file: Path) -> int:
    """ログファイルの行数をカウント（空行除く）。

    Args:
        log_file: ログファイルのパス

    Returns:
        空行を除いた行数

    Example:
        >>> from pathlib import Path
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        ...     f.write("line1\\n")
        ...     f.write("\\n")
        ...     f.write("line2\\n")
        ...     temp_path = Path(f.name)
        >>> count_log_lines(temp_path)
        2
        >>> temp_path.unlink()
    """
    if not log_file.exists():
        return 0

    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def validate_compact_log(
    log_path: Path,
    mode: str,
    max_lines: int,
) -> dict[str, Any]:
    """Compact log の行数を検証。

    Args:
        log_path: ログファイルパス
        mode: "compact_on" | "compact_off"
        max_lines: 最大許容行数

    Returns:
        検証結果の辞書:
        {
            "mode": str,
            "max_lines": int,
            "actual_lines": int,
            "valid": bool,
            "message": str
        }

    Example:
        >>> from pathlib import Path
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        ...     for i in range(100):
        ...         f.write(f"log line {i}\\n")
        ...     temp_path = Path(f.name)
        >>> result = validate_compact_log(temp_path, "compact_on", 200)
        >>> result["valid"]
        True
        >>> temp_path.unlink()
    """
    actual = count_log_lines(log_path)
    valid = actual <= max_lines

    return {
        "mode": mode,
        "max_lines": max_lines,
        "actual_lines": actual,
        "valid": valid,
        "message": (
            f"OK: {mode} log lines={actual} (max={max_lines})"
            if valid
            else f"⚠️ {mode} log lines={actual} exceeds max={max_lines}"
        ),
    }


def compare_compact_logs(
    verbose_path: Path,
    compact_path: Path,
    max_verbose: int = 500,
    max_compact: int = 200,
) -> dict[str, Any]:
    """Compact モード ON/OFF のログ行数を比較検証。

    Args:
        verbose_path: Compact OFF 時のログファイル
        compact_path: Compact ON 時のログファイル
        max_verbose: Verbose モードの最大許容行数
        max_compact: Compact モードの最大許容行数

    Returns:
        比較結果の辞書:
        {
            "verbose": dict,  # validate_compact_log の結果
            "compact": dict,  # validate_compact_log の結果
            "reduction": int,  # 削減された行数
            "reduction_pct": float,  # 削減率（%）
            "all_valid": bool,  # 両方が基準内か
            "summary": str  # サマリーメッセージ
        }

    Example:
        >>> from pathlib import Path
        >>> import tempfile
        >>> # Verbose log (500 lines)
        >>> with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        ...     for i in range(500):
        ...         f.write(f"verbose log {i}\\n")
        ...     verbose_path = Path(f.name)
        >>> # Compact log (200 lines)
        >>> with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        ...     for i in range(200):
        ...         f.write(f"compact log {i}\\n")
        ...     compact_path = Path(f.name)
        >>> result = compare_compact_logs(verbose_path, compact_path)
        >>> result["all_valid"]
        True
        >>> result["reduction"]
        300
        >>> verbose_path.unlink()
        >>> compact_path.unlink()
    """
    # 環境変数で閾値上書き（CI で柔軟に制御）
    try:
        env_max_verbose = os.environ.get("MAX_VERBOSE_LINES")
        if env_max_verbose:
            max_verbose = int(env_max_verbose)
    except Exception:
        pass
    try:
        env_max_compact = os.environ.get("MAX_COMPACT_LINES")
        if env_max_compact:
            max_compact = int(env_max_compact)
    except Exception:
        pass

    verbose_result = validate_compact_log(verbose_path, "compact_off", max_verbose)
    compact_result = validate_compact_log(compact_path, "compact_on", max_compact)

    reduction = verbose_result["actual_lines"] - compact_result["actual_lines"]
    reduction_pct = (
        (reduction / verbose_result["actual_lines"] * 100)
        if verbose_result["actual_lines"] > 0
        else 0.0
    )

    all_valid = verbose_result["valid"] and compact_result["valid"]

    summary_lines = [
        "=== Log Compactness Validation ===",
        f"Verbose (COMPACT_TODAY_LOGS=0): {verbose_result['actual_lines']} lines (max={max_verbose})",
        f"Compact (COMPACT_TODAY_LOGS=1): {compact_result['actual_lines']} lines (max={max_compact})",
        f"Reduction: {reduction} lines ({reduction_pct:.1f}%)",
        "",
    ]

    if all_valid:
        summary_lines.append("✓ Both modes are within limits")
    else:
        summary_lines.append("✗ One or more modes exceeded limits:")
        if not verbose_result["valid"]:
            summary_lines.append(f"  - {verbose_result['message']}")
        if not compact_result["valid"]:
            summary_lines.append(f"  - {compact_result['message']}")

    return {
        "verbose": verbose_result,
        "compact": compact_result,
        "reduction": reduction,
        "reduction_pct": reduction_pct,
        "all_valid": all_valid,
        "summary": "\n".join(summary_lines),
    }


def main() -> None:
    """CLI エントリーポイント。"""
    parser = argparse.ArgumentParser(
        description="Validate log compactness between COMPACT_TODAY_LOGS=0 and =1"
    )
    parser.add_argument(
        "--verbose", type=Path, required=True, help="Path to verbose log file"
    )
    parser.add_argument(
        "--compact", type=Path, required=True, help="Path to compact log file"
    )
    parser.add_argument(
        "--max-verbose",
        type=int,
        default=500,
        help="Max allowed lines for verbose mode (default: 500)",
    )
    parser.add_argument(
        "--max-compact",
        type=int,
        default=200,
        help="Max allowed lines for compact mode (default: 200)",
    )

    args = parser.parse_args()

    result = compare_compact_logs(
        verbose_path=args.verbose,
        compact_path=args.compact,
        max_verbose=args.max_verbose,
        max_compact=args.max_compact,
    )

    print(result["summary"])

    # Exit code: 0=success, 1=validation failed
    exit_code = 0 if result["all_valid"] else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
