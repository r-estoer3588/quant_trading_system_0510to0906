#!/usr/bin/env python3
"""RuntimeError機能の動作確認テスト

指標不足時にRuntimeErrorで即座に停止することを確認する
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd

from common.indicators_validation import (
    IndicatorValidationError,
    validate_precomputed_indicators,
)


def test_runtime_error_with_missing_indicators():
    """指標不足時のRuntimeError動作確認"""
    print("🧪 RuntimeError機能テスト開始")

    # 意図的に指標が不足したデータを作成
    incomplete_data = {
        "AAPL": pd.DataFrame(
            {
                "Close": [100, 101, 102],
                "Volume": [1000000, 1100000, 1200000],
                # ATR10, SMA200, ROC200などの重要な指標が欠損
            }
        ),
        "SPY": pd.DataFrame(
            {
                "Close": [400, 401, 402],
                "Volume": [5000000, 5100000, 5200000],
                # 同様に重要な指標が欠損
            }
        ),
    }

    try:
        # 指標検証を実行（strict_mode=Trueで不足時エラー）
        validation_passed, missing_report = validate_precomputed_indicators(
            incomplete_data,
            systems=[1, 2, 3, 4, 5, 6, 7],  # 全System
            strict_mode=True,  # 不足時は即座停止
            log_callback=lambda x: print(f"  📋 {x}"),
        )

        print("❌ テスト失敗: RuntimeErrorが発生しませんでした")
        return False

    except IndicatorValidationError as e:
        print("✅ テスト成功: IndicatorValidationErrorが正しく発生")
        print(f"  📋 エラーメッセージ: {str(e)[:100]}...")
        return True

    except Exception as e:
        print(f"❌ テスト失敗: 予期しないエラー - {type(e).__name__}: {e}")
        return False


def test_runtime_error_with_complete_indicators():
    """指標が完全な場合の正常動作確認"""
    print("\n🧪 正常データでのテスト開始")

    # 完全な指標を含むデータを作成（簡易版）
    complete_data = {
        "SPY": pd.DataFrame(
            {
                "Close": [400, 401, 402],
                "Volume": [5000000, 5100000, 5200000],
                # 必要最小限の指標を追加
                "ATR10": [5.0, 5.1, 5.2],
                "ATR20": [6.0, 6.1, 6.2],
                "ATR40": [7.0, 7.1, 7.2],
                "ATR50": [7.5, 7.6, 7.7],
                "SMA25": [395, 396, 397],
                "SMA50": [390, 391, 392],
                "SMA100": [385, 386, 387],
                "SMA150": [380, 381, 382],
                "SMA200": [375, 376, 377],
                "RSI3": [50, 51, 52],
                "RSI4": [48, 49, 50],
                "ADX7": [30, 31, 32],
                "ROC200": [0.05, 0.06, 0.07],
                "DollarVolume20": [2e9, 2.1e9, 2.2e9],
                "DollarVolume50": [1.8e9, 1.9e9, 2.0e9],
                "AvgVolume50": [4.5e6, 4.6e6, 4.7e6],
                "ATR_Ratio": [0.08, 0.09, 0.10],
                "Return_Pct": [0.002, 0.003, 0.004],
                "Return_3D": [0.005, 0.006, 0.007],
                "Return_6D": [0.008, 0.009, 0.010],
                "UpTwoDays": [0, 1, 1],
                "Drop3D": [0, 0, 1],
                "HV50": [0.2, 0.21, 0.22],
                "Min_50": [350, 351, 352],
                "Max_70": [450, 451, 452],
            }
        )
    }

    try:
        # 指標検証を実行
        validation_passed, missing_report = validate_precomputed_indicators(
            complete_data,
            systems=[1, 2, 3, 4, 5, 6, 7],
            strict_mode=True,
            log_callback=lambda x: print(f"  📋 {x}"),
        )

        if validation_passed:
            print("✅ テスト成功: 完全な指標データで正常通過")
            return True
        else:
            print("❌ テスト失敗: 完全データでも検証に失敗")
            return False

    except Exception as e:
        print(f"❌ テスト失敗: 予期しないエラー - {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    print("🎯 RuntimeError機能動作確認テスト")
    print("=" * 50)

    # テスト1: 指標不足時のRuntimeError
    test1_success = test_runtime_error_with_missing_indicators()

    # テスト2: 完全指標での正常動作
    test2_success = test_runtime_error_with_complete_indicators()

    print("\n" + "=" * 50)
    print("📊 テスト結果サマリー")
    print(f"  指標不足時RuntimeError: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"  完全指標時正常動作: {'✅ PASS' if test2_success else '❌ FAIL'}")

    if test1_success and test2_success:
        print("\n🎉 すべてのテストが成功しました！")
        print("✅ RuntimeError方針が正常に実装・動作しています")
        exit(0)
    else:
        print("\n💥 一部のテストが失敗しました")
        exit(1)
