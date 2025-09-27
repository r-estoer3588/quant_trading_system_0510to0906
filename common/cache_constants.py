"""キャッシュ関連の設定定数"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class CacheConstants:
    """キャッシュ管理の定数定義"""

    # ファイル形式
    SUPPORTED_FILE_EXTENSIONS: ClassVar[tuple[str, ...]] = (
        ".csv",
        ".parquet",
        ".feather",
    )
    DEFAULT_FILE_EXTENSION: ClassVar[str] = ".csv"

    # CSV フォーマット
    DEFAULT_DECIMAL_POINT: ClassVar[str] = "."
    DEFAULT_FIELD_SEPARATOR: ClassVar[str] = ","

    # 丸め処理の桁数設定
    PRICE_DECIMAL_PLACES: ClassVar[int] = 2
    PERCENTAGE_DECIMAL_PLACES: ClassVar[int] = 4
    VOLUME_DECIMAL_PLACES: ClassVar[int] = 0

    # 健全性チェック
    NAN_RATE_WARNING_THRESHOLD: ClassVar[float] = 0.99  # 99%以上NaNで警告
    RECENT_DATA_WINDOW_SIZE: ClassVar[int] = 120  # 健全性チェックの対象期間

    # Rolling キャッシュ
    DEFAULT_PRUNE_THRESHOLD_DAYS: ClassVar[int] = 5  # 剪定実行の閾値
    DEFAULT_ROLLING_ANCHOR_TICKER: ClassVar[str] = "SPY"

    # 列名グループ定義
    PRICE_ATR_COLUMNS: ClassVar[frozenset[str]] = frozenset(
        {
            "open",
            "close",
            "high",
            "low",
            "atr10",
            "atr14",
            "atr20",
            "atr40",
            "atr50",
            "adjusted_close",
            "adjclose",
            "adj_close",
        }
    )

    VOLUME_COLUMNS: ClassVar[frozenset[str]] = frozenset(
        {"volume", "dollarvolume20", "dollarvolume50", "avgvolume50"}
    )

    OSCILLATOR_COLUMNS: ClassVar[frozenset[str]] = frozenset({"rsi3", "rsi4", "rsi14", "adx7"})

    PERCENTAGE_COLUMNS: ClassVar[frozenset[str]] = frozenset(
        {
            "roc200",
            "return_3d",
            "return_6d",
            "atr_ratio",
            "atr_pct",
            "hv50",
            "return_pct",
        }
    )

    # 必須OHLCV列
    REQUIRED_OHLC_COLUMNS: ClassVar[frozenset[str]] = frozenset({"open", "high", "low", "close"})

    REQUIRED_OHLCV_COLUMNS: ClassVar[frozenset[str]] = frozenset(
        {"open", "high", "low", "close", "volume"}
    )

    # 指標計算用の期間設定
    SMA_PERIODS: ClassVar[tuple[int, ...]] = (25, 50, 100, 150, 200)
    EMA_PERIODS: ClassVar[tuple[int, ...]] = (20, 50)
    ATR_PERIODS: ClassVar[tuple[int, ...]] = (10, 14, 40, 50)
    RSI_PERIODS: ClassVar[tuple[int, ...]] = (3, 14)

    # その他の指標パラメータ
    ROC_PERIOD: ClassVar[int] = 200
    HV_PERIOD: ClassVar[int] = 50
    HV_ANNUALIZATION_FACTOR: ClassVar[float] = 252.0  # 年間営業日数
    DOLLAR_VOLUME_PERIODS: ClassVar[tuple[int, ...]] = (20, 50)


@dataclass(frozen=True)
class IndicatorMinObservations:
    """各指標が有効値を持つために必要な最小観測日数"""

    # SMA系
    SMA25: ClassVar[int] = 20
    SMA50: ClassVar[int] = 50
    SMA100: ClassVar[int] = 100
    SMA150: ClassVar[int] = 150
    SMA200: ClassVar[int] = 200

    # EMA系
    EMA20: ClassVar[int] = 1
    EMA50: ClassVar[int] = 1

    # ATR系
    ATR10: ClassVar[int] = 11
    ATR14: ClassVar[int] = 15
    ATR20: ClassVar[int] = 21
    ATR40: ClassVar[int] = 41
    ATR50: ClassVar[int] = 51

    # オシレータ系
    ADX7: ClassVar[int] = 14
    RSI3: ClassVar[int] = 3
    RSI4: ClassVar[int] = 4
    RSI14: ClassVar[int] = 14

    # その他
    ROC200: ClassVar[int] = 201
    HV50: ClassVar[int] = 51

    # ボリューム系
    DOLLARVOLUME20: ClassVar[int] = 20
    DOLLARVOLUME50: ClassVar[int] = 50
    AVGVOLUME50: ClassVar[int] = 50

    # リターン系
    RETURN_3D: ClassVar[int] = 4
    RETURN_6D: ClassVar[int] = 7
    RETURN_PCT: ClassVar[int] = 2
    DROP3D: ClassVar[int] = 4

    # ATR比率系
    ATR_RATIO: ClassVar[int] = 11
    ATR_PCT: ClassVar[int] = 11

    @classmethod
    def as_dict(cls) -> dict[str, int]:
        """辞書形式で全ての定数を取得"""
        return {
            "sma25": cls.SMA25,
            "sma50": cls.SMA50,
            "sma100": cls.SMA100,
            "sma150": cls.SMA150,
            "sma200": cls.SMA200,
            "ema20": cls.EMA20,
            "ema50": cls.EMA50,
            "atr10": cls.ATR10,
            "atr14": cls.ATR14,
            "atr20": cls.ATR20,
            "atr40": cls.ATR40,
            "atr50": cls.ATR50,
            "adx7": cls.ADX7,
            "rsi3": cls.RSI3,
            "rsi4": cls.RSI4,
            "rsi14": cls.RSI14,
            "roc200": cls.ROC200,
            "hv50": cls.HV50,
            "dollarvolume20": cls.DOLLARVOLUME20,
            "dollarvolume50": cls.DOLLARVOLUME50,
            "avgvolume50": cls.AVGVOLUME50,
            "return_3d": cls.RETURN_3D,
            "return_6d": cls.RETURN_6D,
            "return_pct": cls.RETURN_PCT,
            "drop3d": cls.DROP3D,
            "atr_ratio": cls.ATR_RATIO,
            "atr_pct": cls.ATR_PCT,
        }


@dataclass(frozen=True)
class CacheFilePaths:
    """キャッシュファイルパス関連の定数"""

    BASE_SUBDIR: ClassVar[str] = "base"
    LEGACY_CACHE_DIR: ClassVar[str] = "data_cache"
    TEMP_FILE_SUFFIX: ClassVar[str] = ".tmp"
    META_FILE_PREFIX: ClassVar[str] = "_"
