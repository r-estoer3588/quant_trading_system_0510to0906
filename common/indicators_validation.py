# indicators_validation.py
# 当日シグナル実行時の指標事前計算チェック機能

from __future__ import annotations

import pandas as pd

from common.cache_manager import get_indicator_column_flexible


class IndicatorValidationError(Exception):
    """指標不足による実行停止エラー"""

    pass


# System別必須指標定義
SYSTEM_REQUIRED_INDICATORS = {
    1: {"ATR10", "SMA200", "ROC200", "DollarVolume20"},
    2: {"ATR10", "ADX7", "DollarVolume20"},
    3: {"ATR10", "Drop3D", "DollarVolume20"},
    4: {"ATR10", "RSI4", "DollarVolume20", "UpTwoDays"},
    5: {"ATR10", "ADX7", "DollarVolume20"},
    6: {"ATR10", "Return_6D", "DollarVolume20"},
    7: {"ATR10", "SMA25", "SMA50"},  # SPY固定
}

# 共通必須指標（全Systemで必要）
COMMON_REQUIRED_INDICATORS = {
    "ATR10",
    "ATR20",
    "ATR40",
    "ATR50",
    "SMA25",
    "SMA50",
    "SMA100",
    "SMA150",
    "SMA200",
    "RSI3",
    "RSI4",
    "ADX7",
    "ROC200",
    "DollarVolume20",
    "DollarVolume50",
    "AvgVolume50",
    "ATR_Ratio",
    "Return_Pct",
    "Return_3D",
    "Return_6D",
    "UpTwoDays",
    "Drop3D",
    "HV50",
    "Min_50",
    "Max_70",
}


def validate_precomputed_indicators(
    data_dict: dict[str, pd.DataFrame],
    systems: list[int] | None = None,
    strict_mode: bool = True,
    log_callback=None,
) -> tuple[bool, dict[str, list[str]]]:
    """
    指標事前計算状況を検証し、不足があればエラーレポートを返す

    Args:
        data_dict: 銘柄別データ辞書
        systems: チェック対象システム番号リスト（Noneなら全System）
        strict_mode: True=不足時エラー、False=警告のみ
        log_callback: ログ出力関数

    Returns:
        (validation_passed, missing_indicators_report)
    """
    if not data_dict:
        return True, {}

    if systems is None:
        systems = list(SYSTEM_REQUIRED_INDICATORS.keys())

    if log_callback is None:

        def log_callback(x: str) -> None:
            pass

    # 全システムで必要な指標を収集
    all_required = set(COMMON_REQUIRED_INDICATORS)
    for system_num in systems:
        if system_num in SYSTEM_REQUIRED_INDICATORS:
            all_required.update(SYSTEM_REQUIRED_INDICATORS[system_num])

    missing_report = {}
    validation_errors = []

    # サンプル銘柄でのチェック（最初の10銘柄程度）
    sample_symbols = list(data_dict.keys())[: min(10, len(data_dict))]

    for symbol in sample_symbols:
        df = data_dict[symbol]
        if df is None or df.empty:
            continue

        missing_for_symbol = []

        for indicator in all_required:
            # 大文字・小文字柔軟チェック
            found_col = get_indicator_column_flexible(df, indicator)
            if found_col is None:
                missing_for_symbol.append(indicator)

        if missing_for_symbol:
            missing_report[symbol] = missing_for_symbol
            if len(missing_for_symbol) > 5:  # 多数不足の場合は簡潔にする
                validation_errors.append(
                    f"{symbol}: {len(missing_for_symbol)}個の指標が不足 "
                    f"(例: {', '.join(missing_for_symbol[:3])}...)"
                )
            else:
                validation_errors.append(f"{symbol}: {', '.join(missing_for_symbol)}")

    # 検証結果の判定
    validation_passed = len(missing_report) == 0

    if not validation_passed:
        error_summary = f"指標事前計算チェックで不足を検出: {len(missing_report)}/{len(sample_symbols)}銘柄で問題あり"
        log_callback(f"❌ {error_summary}")

        if len(validation_errors) <= 5:
            for error in validation_errors:
                log_callback(f"   • {error}")
        else:
            for error in validation_errors[:3]:
                log_callback(f"   • {error}")
            log_callback(f"   ... 他{len(validation_errors)-3}件の問題")

        if strict_mode:
            detailed_msg = "\\n".join(
                [
                    "🚨 指標事前計算が不足しています",
                    f"対象システム: {systems}",
                    f"不足銘柄: {len(missing_report)}/{len(sample_symbols)}",
                    "解決方法: scripts/build_rolling_with_indicators.py を実行してください",
                ]
            )
            raise IndicatorValidationError(detailed_msg)
    else:
        log_callback("✅ 指標事前計算チェック: 全て正常")

    return validation_passed, missing_report


def quick_indicator_check(data_dict: dict[str, pd.DataFrame], log_callback=None) -> bool:
    """
    高速な指標存在チェック（サンプル銘柄のみ）

    Returns:
        True=十分な指標が存在, False=指標不足
    """
    if not data_dict:
        return True

    if log_callback is None:

        def log_callback(x: str) -> None:
            pass

    # 最初の3銘柄をサンプリング
    sample_symbols = list(data_dict.keys())[:3]

    # 最低限必要な指標
    key_indicators = ["ATR10", "SMA50", "RSI4", "DollarVolume20"]

    for symbol in sample_symbols:
        df = data_dict[symbol]
        if df is None or df.empty:
            continue

        found_count = 0
        for indicator in key_indicators:
            if get_indicator_column_flexible(df, indicator) is not None:
                found_count += 1

        # 4つ中3つ以上見つかれば良しとする
        if found_count < 3:
            log_callback(f"⚠️  高速チェック: {symbol}で指標不足 ({found_count}/4)")
            return False

    log_callback("✅ 高速指標チェック: OK")
    return True
