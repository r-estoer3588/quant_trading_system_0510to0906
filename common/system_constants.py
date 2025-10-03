"""システム共通の定数定義。

各System*.pyで使用される定数を一元管理：
- 必須カラム名
- 最小行数要件
- 閾値パラメータ
- フィルタリング条件
"""

from __future__ import annotations

# === 基本データ要件 ===
REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")

# システム別最小行数要件
MIN_ROWS_SYSTEM1 = 200  # ROC200 + SMA200 必要
MIN_ROWS_SYSTEM2 = 150  # RSI3 + ADX7 ベース
MIN_ROWS_SYSTEM3 = 150  # 3日下落 + 各種指標
MIN_ROWS_SYSTEM4 = 200  # SMA200 + HV50 必要
MIN_ROWS_SYSTEM5 = 150  # ADX7 + ATR ベース
MIN_ROWS_SYSTEM6 = 100  # 比較的短期指標
MIN_ROWS_SYSTEM7 = 150  # SPY専用、基本指標

# === System1 (Long ROC200) 定数 ===
SYSTEM1_ROC_PERIOD = 200
SYSTEM1_SMA_PERIOD = 200
SYSTEM1_MIN_PRICE = 5.0
SYSTEM1_MIN_DOLLAR_VOLUME = 25_000_000  # 25M

# === System2 (Short RSI spike) 定数 ===
SYSTEM2_RSI_PERIOD = 3
SYSTEM2_RSI_THRESHOLD = 90  # RSI3 > 90
SYSTEM2_ADX_PERIOD = 7
SYSTEM2_MIN_PRICE = 5.0
SYSTEM2_MIN_DOLLAR_VOLUME = 25_000_000  # 25M
SYSTEM2_ATR_RATIO_THRESHOLD = 0.03  # 3%

# === System3 (Long mean-reversion) 定数 ===
SYSTEM3_MIN_PRICE = 5.0
SYSTEM3_MIN_DOLLAR_VOLUME = 25_000_000  # 25M
SYSTEM3_ATR_RATIO_THRESHOLD = 0.05  # 5%
SYSTEM3_DROP_3D_THRESHOLD = 0.125  # 12.5% 3日下落

# === System4 (Long trend low-vol pullback) 定数 ===
SYSTEM4_MIN_PRICE = 5.0
SYSTEM4_MIN_DOLLAR_VOLUME = 25_000_000  # 25M
SYSTEM4_RSI_THRESHOLD = 50  # RSI4 < 50
SYSTEM4_HV_PERIOD = 50
SYSTEM4_SMA_PERIOD = 200

# === System5 (Long mean-reversion with high ADX) 定数 ===
SYSTEM5_MIN_PRICE = 5.0
SYSTEM5_MIN_DOLLAR_VOLUME = 25_000_000  # 25M
SYSTEM5_ATR_PCT_THRESHOLD = 0.025  # 2.5%
SYSTEM5_ADX_THRESHOLD = 55  # ADX7 > 55
SYSTEM5_ADX_PERIOD = 7

# === System6 (Short mean-reversion momentum burst) 定数 ===
SYSTEM6_MIN_PRICE = 5.0
SYSTEM6_MIN_DOLLAR_VOLUME = 25_000_000  # 25M
SYSTEM6_RETURN_6D_THRESHOLD = 0.20  # 20% 6日リターン
SYSTEM6_ATR_PERIOD = 10
SYSTEM6_DOLLAR_VOLUME_PERIOD = 50

# === System7 (SPY short catastrophe hedge) 定数 ===
SYSTEM7_SYMBOL = "SPY"  # 固定シンボル
SYSTEM7_MIN_ROWS = 150

# === 共通フィルター閾値 ===
DEFAULT_MIN_PRICE = 5.0
DEFAULT_MIN_DOLLAR_VOLUME = 25_000_000  # 25M USD
DEFAULT_ATR_RATIO_THRESHOLD = 0.03  # 3%

# === 指標計算パラメータ ===
ATR_PERIOD_DEFAULT = 10
RSI_PERIOD_DEFAULT = 4
ADX_PERIOD_DEFAULT = 7
DOLLAR_VOLUME_PERIOD_DEFAULT = 20

# === システム別必須指標リスト ===
SYSTEM1_REQUIRED_INDICATORS = ["roc200", "sma200", "dollarvolume20"]

SYSTEM2_REQUIRED_INDICATORS = [
    "rsi3",
    "adx7",
    "atr10",
    "dollarvolume20",
    "atr_ratio",
    "twodayup",
]

SYSTEM3_REQUIRED_INDICATORS = [
    "atr10",
    "dollarvolume20",
    "atr_ratio",
    "drop3d",
]

SYSTEM4_REQUIRED_INDICATORS = [
    "rsi4",
    "sma200",
    "atr40",  # Required for ATR ratio calculation
    "hv50",
    "dollarvolume50",
]

SYSTEM5_REQUIRED_INDICATORS = [
    "adx7",
    "atr10",
    "dollarvolume20",
    "atr_pct",
]

SYSTEM6_REQUIRED_INDICATORS = [
    "atr10",
    "dollarvolume50",
    "return_6d",
    "uptwodays",
]

SYSTEM7_REQUIRED_INDICATORS = [
    "atr50",  # ATR50 (lowercase, as used in system7)
    "min_50",  # Min_50 - 50日の最低価格
    "max_70",  # Max_70 - 70日の最高価格
]

# === システム別設定マッピング ===
SYSTEM_CONFIGS = {
    "system1": {
        "min_rows": MIN_ROWS_SYSTEM1,
        "required_indicators": SYSTEM1_REQUIRED_INDICATORS,
        "min_price": SYSTEM1_MIN_PRICE,
        "min_dollar_volume": SYSTEM1_MIN_DOLLAR_VOLUME,
    },
    "system2": {
        "min_rows": MIN_ROWS_SYSTEM2,
        "required_indicators": SYSTEM2_REQUIRED_INDICATORS,
        "min_price": SYSTEM2_MIN_PRICE,
        "min_dollar_volume": SYSTEM2_MIN_DOLLAR_VOLUME,
        "atr_ratio_threshold": SYSTEM2_ATR_RATIO_THRESHOLD,
        "rsi_threshold": SYSTEM2_RSI_THRESHOLD,
    },
    "system3": {
        "min_rows": MIN_ROWS_SYSTEM3,
        "required_indicators": SYSTEM3_REQUIRED_INDICATORS,
        "min_price": SYSTEM3_MIN_PRICE,
        "min_dollar_volume": SYSTEM3_MIN_DOLLAR_VOLUME,
        "atr_ratio_threshold": SYSTEM3_ATR_RATIO_THRESHOLD,
        "drop_3d_threshold": SYSTEM3_DROP_3D_THRESHOLD,
    },
    "system4": {
        "min_rows": MIN_ROWS_SYSTEM4,
        "required_indicators": SYSTEM4_REQUIRED_INDICATORS,
        "min_price": SYSTEM4_MIN_PRICE,
        "min_dollar_volume": SYSTEM4_MIN_DOLLAR_VOLUME,
        "rsi_threshold": SYSTEM4_RSI_THRESHOLD,
    },
    "system5": {
        "min_rows": MIN_ROWS_SYSTEM5,
        "required_indicators": SYSTEM5_REQUIRED_INDICATORS,
        "min_price": SYSTEM5_MIN_PRICE,
        "min_dollar_volume": SYSTEM5_MIN_DOLLAR_VOLUME,
        "atr_pct_threshold": SYSTEM5_ATR_PCT_THRESHOLD,
        "adx_threshold": SYSTEM5_ADX_THRESHOLD,
    },
    "system6": {
        "min_rows": MIN_ROWS_SYSTEM6,
        "required_indicators": SYSTEM6_REQUIRED_INDICATORS,
        "min_price": SYSTEM6_MIN_PRICE,
        "min_dollar_volume": SYSTEM6_MIN_DOLLAR_VOLUME,
        "return_6d_threshold": SYSTEM6_RETURN_6D_THRESHOLD,
    },
    "system7": {
        "min_rows": MIN_ROWS_SYSTEM7,
        "required_indicators": SYSTEM7_REQUIRED_INDICATORS,
        "symbol": SYSTEM7_SYMBOL,
    },
}


def get_system_config(system_name: str) -> dict:
    """指定されたシステムの設定を取得する。

    Args:
        system_name: システム名（例: "system1", "system2"）

    Returns:
        システム設定辞書

    Raises:
        KeyError: 未知のシステム名の場合
    """
    return SYSTEM_CONFIGS[system_name.lower()]


__all__ = [
    # 基本定数
    "REQUIRED_COLUMNS",
    "DEFAULT_MIN_PRICE",
    "DEFAULT_MIN_DOLLAR_VOLUME",
    "DEFAULT_ATR_RATIO_THRESHOLD",
    # システム別最小行数
    "MIN_ROWS_SYSTEM1",
    "MIN_ROWS_SYSTEM2",
    "MIN_ROWS_SYSTEM3",
    "MIN_ROWS_SYSTEM4",
    "MIN_ROWS_SYSTEM5",
    "MIN_ROWS_SYSTEM6",
    "MIN_ROWS_SYSTEM7",
    # システム別必須指標
    "SYSTEM1_REQUIRED_INDICATORS",
    "SYSTEM2_REQUIRED_INDICATORS",
    "SYSTEM3_REQUIRED_INDICATORS",
    "SYSTEM4_REQUIRED_INDICATORS",
    "SYSTEM5_REQUIRED_INDICATORS",
    "SYSTEM6_REQUIRED_INDICATORS",
    "SYSTEM7_REQUIRED_INDICATORS",
    # 設定管理
    "SYSTEM_CONFIGS",
    "get_system_config",
]
