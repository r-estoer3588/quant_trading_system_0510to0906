"""
Test suite for common/ui_components.py functionality
Tests utility functions, data processing, and result summarization
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pandas as pd

from common.ui_components import (
    _mtime_or_zero,
    clean_date_column,
    default_log_callback,
    extract_zero_reason_from_logs,
)


def test_clean_date_column_valid_dates():
    """有効な日付を含むDataFrameのテスト"""
    df = pd.DataFrame(
        {
            "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "value": [1, 2, 3],
        }
    )

    result = clean_date_column(df, "Date")

    assert len(result) == 3
    assert pd.api.types.is_datetime64_any_dtype(result["Date"])


def test_clean_date_column_mixed_dates():
    """有効・無効な日付が混在する場合のテスト"""
    df = pd.DataFrame(
        {
            "Date": ["2023-01-01", "invalid_date", "2023-01-03"],
            "value": [1, 2, 3],
        }
    )

    result = clean_date_column(df, "Date")

    # Invalid date should be dropped
    assert len(result) == 2  # One invalid date dropped
    assert pd.api.types.is_datetime64_any_dtype(result["Date"])


def test_clean_date_column_no_date_column():
    """Date列が存在しない場合のテスト"""
    df = pd.DataFrame({"value": [1, 2, 3]})

    result = clean_date_column(df, "Date")

    # Should return original DataFrame unchanged
    assert len(result) == 3
    assert "Date" not in result.columns


def test_clean_date_column_empty_dataframe():
    """空のDataFrameでのテスト"""
    df = pd.DataFrame()

    result = clean_date_column(df, "Date")

    assert len(result) == 0
    assert isinstance(result, pd.DataFrame)


def test_default_log_callback_basic():
    """default_log_callbackの基本テスト"""
    start_time = time.time() - 60  # 1 minute ago

    result = default_log_callback(50, 100, start_time)

    assert isinstance(result, str)
    assert "50/100" in result
    assert "件" in result
    assert "経過" in result
    assert "残り目安" in result


def test_default_log_callback_zero_processed():
    """処理件数が0の場合のテスト"""
    start_time = time.time()

    result = default_log_callback(0, 100, start_time)

    assert isinstance(result, str)
    assert "0/100" in result


def test_mtime_or_zero_existing_file():
    """存在するファイルのmtimeテスト"""
    with patch("os.path.getmtime") as mock_getmtime:
        mock_getmtime.return_value = 1640995200.0  # 2022-01-01

        result = _mtime_or_zero("/fake/path")

        assert result == 1640995200.0
        mock_getmtime.assert_called_once_with("/fake/path")


def test_mtime_or_zero_nonexistent_file():
    """存在しないファイルのmtimeテスト"""
    with patch("os.path.getmtime") as mock_getmtime:
        mock_getmtime.side_effect = FileNotFoundError()

        result = _mtime_or_zero("/nonexistent/path")

        assert result == 0.0


def test_extract_zero_reason_from_logs_found():
    """ログから候補0件理由を抽出するテスト - 見つかる場合"""
    logs = [
        "通常ログ1",
        "候補0件理由: RSI条件を満たすシンボルがない",
        "通常ログ2",
    ]

    result = extract_zero_reason_from_logs(logs)

    assert result == "RSI条件を満たすシンボルがない"


def test_extract_zero_reason_from_logs_setup_failure():
    """ログからセットアップ不成立理由を抽出するテスト"""
    logs = [
        "通常ログ1",
        "セットアップ不成立: データ不足",
        "通常ログ2",
    ]

    result = extract_zero_reason_from_logs(logs)

    assert result == "データ不足"


def test_extract_zero_reason_from_logs_empty():
    """空のログでのテスト"""
    result = extract_zero_reason_from_logs([])
    assert result is None


def test_extract_zero_reason_from_logs_none():
    """Noneログでのテスト"""
    result = extract_zero_reason_from_logs(None)
    assert result is None


def test_extract_zero_reason_from_logs_not_found():
    """理由が見つからない場合のテスト"""
    logs = ["通常ログ1", "通常ログ2", "通常ログ3"]

    result = extract_zero_reason_from_logs(logs)

    assert result is None
