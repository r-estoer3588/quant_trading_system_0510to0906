"""TRD (Trading Day) リスト長の検証ユーティリティ。

各システムの候補抽出で、日付リストの長さが想定範囲内かを検証します。
Mini モード時は 1 日、Quick モード時は 5 日など、モードに応じた上限を設定。
"""

from __future__ import annotations

from typing import Any


def verify_trd_length(
    by_date: dict[Any, list],
    system_id: str,
    expected_max: int = 5,
) -> dict[str, Any]:
    """Trading day リストの長さを検証。

    Args:
        by_date: {日付: [候補リスト]} の辞書
        system_id: システム ID（ログ用）
        expected_max: 想定される最大日数
            - mini モード: 1
            - quick モード: 5
            - sample モード: 10
            - full モード: 30 など

    Returns:
        検証結果の辞書:
        {
            "system_id": str,
            "valid": bool,
            "expected_max": int,
            "actual_length": int,
            "exceeded": bool,
            "message": str
        }

    Example:
        >>> by_date = {"2024-01-15": ["AAPL", "MSFT"]}
        >>> result = verify_trd_length(by_date, "system1", expected_max=1)
        >>> result["valid"]
        True
        >>> result["message"]
        'OK: system1 TRD length=1 (max=1)'
    """
    actual_len = len(by_date)
    exceeded = actual_len > expected_max

    result = {
        "system_id": system_id,
        "valid": not exceeded,
        "expected_max": expected_max,
        "actual_length": actual_len,
        "exceeded": exceeded,
        "message": (
            f"OK: {system_id} TRD length={actual_len} (max={expected_max})"
            if not exceeded
            else f"⚠️ {system_id} TRD length={actual_len} exceeds max={expected_max}"
        ),
    }
    return result


def get_expected_max_for_mode(test_mode: str | None) -> int:
    """テストモードに応じた TRD 最大日数を取得。

    Args:
        test_mode: "mini", "quick", "sample", None (full)

    Returns:
        想定される最大日数

    Example:
        >>> get_expected_max_for_mode("mini")
        1
        >>> get_expected_max_for_mode("quick")
        5
        >>> get_expected_max_for_mode(None)
        30
    """
    mode_map = {
        "mini": 1,
        "quick": 5,
        "sample": 10,
        None: 30,  # full モード
    }
    return mode_map.get(test_mode, 30)


def verify_all_systems_trd_length(
    systems_by_date: dict[str, dict[Any, list]],
    test_mode: str | None = None,
) -> dict[str, dict[str, Any]]:
    """複数システムの TRD 長を一括検証。

    Args:
        systems_by_date: {system_id: {日付: [候補リスト]}} の辞書
        test_mode: テストモード ("mini", "quick", "sample", None)

    Returns:
        {system_id: 検証結果} の辞書

    Example:
        >>> systems_by_date = {
        ...     "system1": {"2024-01-15": ["AAPL"]},
        ...     "system2": {"2024-01-15": ["TSLA"], "2024-01-16": ["NVDA"]},
        ... }
        >>> results = verify_all_systems_trd_length(systems_by_date, test_mode="mini")
        >>> results["system1"]["valid"]
        True
        >>> results["system2"]["valid"]
        False
    """
    expected_max = get_expected_max_for_mode(test_mode)
    results = {}

    for system_id, by_date in systems_by_date.items():
        results[system_id] = verify_trd_length(by_date, system_id, expected_max)

    return results


def format_trd_verification_summary(
    results: dict[str, dict[str, Any]],
) -> str:
    """TRD 検証結果をサマリー形式でフォーマット。

    Args:
        results: verify_all_systems_trd_length() の戻り値

    Returns:
        整形されたサマリー文字列

    Example:
        >>> results = {
        ...     "system1": {"valid": True, "message": "OK: system1 TRD length=1 (max=1)"},
        ...     "system2": {"valid": False, "message": "⚠️ system2 TRD length=2 (max=1)"},
        ... }
        >>> summary = format_trd_verification_summary(results)
        >>> "OK: system1" in summary
        True
        >>> "⚠️ system2" in summary
        True
    """
    lines = ["=== TRD Length Verification Summary ==="]

    valid_count = sum(1 for r in results.values() if r["valid"])
    total_count = len(results)

    for _, result in sorted(results.items()):
        status_icon = "✓" if result["valid"] else "✗"
        lines.append(f"{status_icon} {result['message']}")

    lines.append("")
    lines.append(f"Total: {valid_count}/{total_count} systems passed")

    return "\n".join(lines)


if __name__ == "__main__":
    # 簡易テスト用のサンプル実行
    sample_data = {
        "system1": {"2024-01-15": ["AAPL", "MSFT"]},
        "system2": {"2024-01-15": ["TSLA"], "2024-01-16": ["NVDA"]},
        "system3": {"2024-01-15": ["GOOGL"]},
    }

    print("=== Mini モード検証 ===")
    results_mini = verify_all_systems_trd_length(sample_data, test_mode="mini")
    print(format_trd_verification_summary(results_mini))

    print("\n=== Quick モード検証 ===")
    results_quick = verify_all_systems_trd_length(sample_data, test_mode="quick")
    print(format_trd_verification_summary(results_quick))
