"""リファクタリング後のCacheManagerクラス"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from common.cache_file_io import CacheFileIO
from common.cache_health_checker import CacheHealthChecker
from common.dataframe_utils import (
    prepare_dataframe_for_cache,
    round_dataframe,
    standardize_ohlcv_columns,
    validate_required_columns,
)
from common.utils import safe_filename
from config.settings import Settings, get_settings
from indicators_common import add_indicators

logger = logging.getLogger(__name__)

# 定数定義
BASE_SUBDIR = "base"
MAIN_INDICATOR_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "sma25",
    "sma50",
    "sma100",
    "sma150",
    "sma200",
    "ema20",
    "ema50",
    "atr10",
    "atr14",
    "atr20",
    "atr40",
    "atr50",
    "adx7",
    "rsi3",
    "rsi4",
    "rsi14",
    "roc200",
    "hv50",
    "dollarvolume20",
    "dollarvolume50",
    "avgvolume50",
    "return_3d",
    "return_6d",
    "return6d",
    "return_pct",
    "drop3d",
    "atr_ratio",
    "atr_pct",
)


class CacheManager:
    """リファクタリング後のキャッシュ管理クラス（責任分割済み）"""

    _DEFAULT_INSTANCE: ClassVar[CacheManager | None] = None

    def __init__(self, settings: Settings):
        self.settings = settings
        self.full_dir = Path(settings.cache.full_dir)
        self.rolling_dir = Path(settings.cache.rolling_dir)
        self.rolling_cfg = settings.cache.rolling
        self.rolling_meta_path = self.rolling_dir / self.rolling_cfg.meta_file

        # 各ディレクトリを作成
        self.full_dir.mkdir(parents=True, exist_ok=True)
        self.rolling_dir.mkdir(parents=True, exist_ok=True)

        # 依存コンポーネント
        self.file_io = CacheFileIO(settings)
        self.health_checker = CacheHealthChecker("[CacheManager]")

        # ローカル設定
        self._ui_prefix = "[CacheManager]"

    @classmethod
    def get_default_instance(cls) -> CacheManager:
        """デフォルトインスタンスを取得（シングルトンパターン）"""
        if cls._DEFAULT_INSTANCE is None:
            settings = get_settings(create_dirs=False)
            cls._DEFAULT_INSTANCE = cls(settings)
        return cls._DEFAULT_INSTANCE

    @classmethod
    def reset_default_instance(cls) -> None:
        """デフォルトインスタンスをリセット（主にテスト用）"""
        cls._DEFAULT_INSTANCE = None

    def read(self, ticker: str, profile: str) -> pd.DataFrame | None:
        """キャッシュからDataFrameを読み取り"""
        base_dir = self.full_dir if profile == "full" else self.rolling_dir
        path = self.file_io.detect_file_path(base_dir, ticker)

        df = self.file_io.read_dataframe(path)
        if df is None:
            return None

        # データの正規化
        df = prepare_dataframe_for_cache(df)

        # 健全性チェック
        self.health_checker.check_dataframe_health(df, ticker, profile)

        return df

    def write_atomic(self, df: pd.DataFrame, ticker: str, profile: str) -> None:
        """DataFrameをアトミックにキャッシュへ書き込み"""
        base_dir = self.full_dir if profile == "full" else self.rolling_dir
        path = self.file_io.detect_file_path(base_dir, ticker)

        # 丸め処理
        if profile == "rolling":
            round_decimals = getattr(self.rolling_cfg, "round_decimals", None)
        else:
            round_decimals = getattr(self.settings.cache, "round_decimals", None)

        df_to_write = round_dataframe(df, round_decimals)

        # アトミック書き込み
        self.file_io.write_dataframe_atomic(df_to_write, path)

    def upsert_both(self, ticker: str, new_rows: pd.DataFrame) -> None:
        """新しい行を'full'と'rolling'両方のキャッシュに追加"""
        for profile in ("full", "rolling"):
            self._upsert_one(ticker, new_rows, profile)

    def _upsert_one(self, ticker: str, new_rows: pd.DataFrame, profile: str) -> None:
        """単一プロファイルへの行追加処理"""
        # 新しい行の前処理
        if new_rows is not None and not new_rows.empty and "date" in new_rows.columns:
            new_rows = new_rows.copy()
            new_rows["date"] = pd.to_datetime(new_rows["date"], errors="coerce")
            new_rows = new_rows.dropna(subset=["date"])

        # 既存データの読み取り
        current_data = self.read(ticker, profile)

        # データのマージ
        if current_data is None or current_data.empty:
            merged = new_rows.copy() if new_rows is not None else pd.DataFrame()
        else:
            merged = (
                pd.concat([current_data, new_rows], ignore_index=True)
                if new_rows is not None
                else current_data
            )

        if not merged.empty:
            # ソート・重複除去
            merged = (
                merged.sort_values("date")
                .drop_duplicates("date")
                .reset_index(drop=True)
            )

            # Rolling窓の適用
            if profile == "rolling":
                merged = self._enforce_rolling_window(merged)

            # 指標の再計算
            merged = self._recompute_indicators(merged)

        # 書き込み
        if not merged.empty:
            self.write_atomic(merged, ticker, profile)

    @property
    def _rolling_target_length(self) -> int:
        """Rolling窓の目標長"""
        return int(self.rolling_cfg.base_lookback_days + self.rolling_cfg.buffer_days)

    def _enforce_rolling_window(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling窓のサイズ制限を適用"""
        if "date" not in df.columns or df.empty:
            return df

        target_length = self._rolling_target_length
        if len(df) > target_length:
            return df.tail(target_length).reset_index(drop=True)
        return df

    def _recompute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """基本OHLCデータが更新された際に派生指標を再計算"""
        if df is None or df.empty or "date" not in df.columns:
            return df

        # 必須列のチェック
        required_cols = {"open", "high", "low", "close"}
        is_valid, missing_cols = validate_required_columns(df, required_cols)
        if not is_valid:
            logger.warning(f"必須列不足のため指標計算をスキップ: {missing_cols}")
            return df

        # データの準備
        base_data = df.copy()
        base_data["date"] = pd.to_datetime(base_data["date"], errors="coerce")
        base_data = (
            base_data.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        )

        if base_data.empty:
            return df

        # OHLCV列の数値化
        for col in ("open", "high", "low", "close", "volume"):
            if col in base_data.columns:
                base_data[col] = pd.to_numeric(base_data[col], errors="coerce")

        # indicators_commonとの互換性のための列名変換
        case_mapping = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }

        base_renamed = base_data.rename(
            columns={k: v for k, v in case_mapping.items() if k in base_data.columns}
        )
        base_renamed["Date"] = base_renamed["date"]

        try:
            # 指標計算
            enriched = add_indicators(base_renamed)
            enriched = enriched.drop(columns=["Date"], errors="ignore")
            enriched.columns = [str(c).lower() for c in enriched.columns]
            enriched["date"] = pd.to_datetime(
                enriched.get("date", base_data["date"]), errors="coerce"
            )

            # 元のDataFrameと指標計算結果をマージ
            result = df.copy()
            ohlcv_columns = {"open", "high", "low", "close", "volume"}

            for col, series in enriched.items():
                if col == "date":
                    result["date"] = series
                    continue
                if col in ohlcv_columns:
                    # 元のOHLCV列は保持
                    continue
                # 指標列は更新
                result[col] = series

            # 重複列の除去
            return result.loc[:, ~result.columns.duplicated(keep="first")]

        except Exception as e:
            logger.error(f"指標再計算失敗: {e}")
            return df

    def prune_rolling_if_needed(self, anchor_ticker: str = "SPY") -> dict[str, int]:
        """Rolling キャッシュの定期的な剪定"""
        try:
            # メタデータの読み取り
            if self.rolling_meta_path.exists():
                meta_content = self.rolling_meta_path.read_text(encoding="utf-8")
                last_meta = json.loads(meta_content)
            else:
                last_meta = {"anchor_rows_at_prune": 0}
        except (FileNotFoundError, json.JSONDecodeError):
            last_meta = {"anchor_rows_at_prune": 0}

        # アンカー銘柄のデータサイズチェック
        anchor_df = self.read(anchor_ticker, "rolling")
        if anchor_df is None or anchor_df.empty:
            logger.info(f"{self._ui_prefix} rolling未整備のためprune処理をスキップ")
            return {"pruned_files": 0, "dropped_rows_total": 0}

        current_rows = len(anchor_df)
        previous_rows = int(last_meta.get("anchor_rows_at_prune", 0))
        progress = current_rows - previous_rows

        prune_threshold = int(self.rolling_cfg.prune_chunk_days)
        if progress < prune_threshold:
            logger.info(
                f"{self._ui_prefix} 進捗{progress}営業日 (< {prune_threshold}) "
                "のためprune不要"
            )
            return {"pruned_files": 0, "dropped_rows_total": 0}

        # 剪定処理の開始
        logger.info(
            f"{self._ui_prefix} ⏳ prune開始: anchor={anchor_ticker}, "
            f"進捗={progress}営業日"
        )

        pruned_files = 0
        dropped_rows_total = 0

        # 各ファイルの剪定
        for file_path in self.rolling_dir.glob("*.*"):
            if file_path.name.startswith("_"):  # メタファイルはスキップ
                continue

            df = self.read(file_path.stem, "rolling")
            if df is None or df.empty:
                continue

            droppable_rows = len(df) - self._rolling_target_length
            rows_to_drop = min(prune_threshold, droppable_rows)

            if rows_to_drop > 0:
                pruned_df = df.iloc[rows_to_drop:].reset_index(drop=True)
                self.write_atomic(pruned_df, file_path.stem, "rolling")
                pruned_files += 1
                dropped_rows_total += rows_to_drop

        # メタデータの更新
        new_meta = {"anchor_rows_at_prune": current_rows}
        self.rolling_meta_path.write_text(
            json.dumps(new_meta, indent=2), encoding="utf-8"
        )

        logger.info(
            f"{self._ui_prefix} ✅ prune完了: files={pruned_files}, "
            f"dropped_rows={dropped_rows_total}"
        )

        return {"pruned_files": pruned_files, "dropped_rows_total": dropped_rows_total}


def _base_directory() -> Path:
    """ベースキャッシュディレクトリを取得"""
    settings = get_settings(create_dirs=True)
    base_dir = Path(settings.DATA_CACHE_DIR) / BASE_SUBDIR
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def compute_base_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCVのDataFrameに共通ベース指標を付加（従来の互換関数）"""
    if df is None or df.empty:
        return df

    normalized = df.copy()

    # 列名正規化
    rename_map = {c: c.lower() for c in normalized.columns}
    normalized = normalized.rename(columns=rename_map)

    # Date列の設定
    if "date" in normalized.columns:
        normalized = normalized.rename(columns={"date": "Date"})
    if "Date" in normalized.columns:
        normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
        normalized = (
            normalized.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
        )

    # OHLCV列の標準化
    normalized = standardize_ohlcv_columns(normalized.reset_index())
    if "Date" in normalized.columns:
        normalized = normalized.set_index("Date")

    # 必須列チェック
    required_columns = {"High", "Low", "Close"}
    if not required_columns.issubset(normalized.columns):
        missing = required_columns - set(normalized.columns)
        logger.warning(f"必須列欠落のためインジ計算をスキップ: missing={missing}")
        return normalized.reset_index()

    # 価格系列の数値化
    close = pd.to_numeric(normalized["Close"], errors="coerce")
    high = pd.to_numeric(normalized["High"], errors="coerce")
    low = pd.to_numeric(normalized["Low"], errors="coerce")
    volume = None
    if "Volume" in normalized.columns:
        volume = pd.to_numeric(normalized["Volume"], errors="coerce")

    # SMA計算
    for period in [25, 50, 100, 150, 200]:
        normalized[f"sma{period}"] = close.rolling(period).mean()

    # EMA計算
    for period in [20, 50]:
        normalized[f"ema{period}"] = close.ewm(span=period, adjust=False).mean()

    # ATR計算
    true_range = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)

    for period in [10, 14, 40, 50]:
        normalized[f"atr{period}"] = true_range.rolling(period).mean()

    # RSI計算（Wilderのスムージング）
    def calculate_rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(alpha=1 / period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    for period in [3, 14]:
        normalized[f"rsi{period}"] = calculate_rsi(close, period)

    # ROC & HV計算
    normalized["roc200"] = close.pct_change(200) * 100.0
    log_returns = (close / close.shift(1)).apply(np.log)
    standard_deviation = log_returns.rolling(50).std()
    normalized["hv50"] = standard_deviation * np.sqrt(252) * 100.0

    # ボリューム系指標
    if volume is not None:
        normalized["dollarvolume20"] = (close * volume).rolling(20).mean()
        normalized["dollarvolume50"] = (close * volume).rolling(50).mean()

    return normalized.reset_index()


def base_cache_path(symbol: str) -> Path:
    """ベースキャッシュファイルのパスを取得"""
    return _base_directory() / f"{safe_filename(symbol)}.csv"


def save_base_cache(
    symbol: str, df: pd.DataFrame, settings: Settings | None = None
) -> Path:
    """ベースキャッシュDataFrameをCSVファイルに保存"""
    file_path = base_cache_path(symbol)
    df_to_save = df.reset_index() if df.index.name is not None else df
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if settings is None:
        settings = get_settings(create_dirs=True)

    # 丸め処理
    round_decimals = getattr(getattr(settings, "cache", None), "round_decimals", None)
    df_rounded = round_dataframe(df_to_save, round_decimals)

    # CSV書き込み
    df_rounded.to_csv(file_path, index=False)

    return file_path


def load_base_cache(
    symbol: str,
    *,
    rebuild_if_missing: bool = True,
    cache_manager: CacheManager | None = None,
    min_last_date: pd.Timestamp | None = None,
    allowed_recent_dates: Iterable[object] | None = None,
    prefer_precomputed_indicators: bool = True,
) -> pd.DataFrame | None:
    """ベースキャッシュを読み込み（鮮度検証・再構築オプション付き）"""
    cache_mgr = cache_manager or CacheManager.get_default_instance()
    file_path = base_cache_path(symbol)
    df = None

    # 既存キャッシュの読み取り
    if file_path.exists():
        try:
            df = pd.read_csv(file_path, parse_dates=["Date"])
            df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
        except Exception:
            df = None

    # 鮮度チェック
    if df is not None:
        last_date = df.index[-1] if not df.empty else None
        is_stale = False

        if allowed_recent_dates and last_date:
            allowed_timestamps = set()
            for date_obj in allowed_recent_dates:
                try:
                    if date_obj is not None:
                        # より安全な型変換
                        ts = pd.Timestamp(str(date_obj))
                        allowed_timestamps.add(ts.normalize())
                except (ValueError, TypeError):
                    continue
            if last_date.normalize() not in allowed_timestamps:
                is_stale = True

        if not is_stale and min_last_date and last_date:
            if last_date.normalize() < pd.Timestamp(min_last_date).normalize():
                is_stale = True

        if not is_stale:
            return df.reset_index()

        if not rebuild_if_missing:
            return df.reset_index()

        logger.info(f"ベースキャッシュ({symbol})が古いため再構築します")
        df = None

    # キャッシュ再構築
    if df is None and rebuild_if_missing:
        raw_data = (
            cache_mgr.read(symbol, "full")
            or cache_mgr.read(symbol, "rolling")
            or _read_legacy_cache(symbol)
        )

        if raw_data is not None and not raw_data.empty:
            # 既存の指標列を優先する場合
            try:
                lowercase_columns = {c.lower() for c in raw_data.columns}
            except Exception:
                lowercase_columns = set()

            has_indicators = any(
                col in lowercase_columns for col in MAIN_INDICATOR_COLUMNS
            )

            if prefer_precomputed_indicators and has_indicators:
                output = raw_data.copy()
                # Date列の正規化
                if "date" in output.columns and "Date" not in output.columns:
                    output = output.rename(columns={"date": "Date"})
                if "Date" not in output.columns and isinstance(
                    output.index, pd.DatetimeIndex
                ):
                    try:
                        output = output.reset_index()
                        if output.columns[0].lower() == "index":
                            output = output.rename(columns={output.columns[0]: "Date"})
                    except Exception:
                        pass

                try:
                    save_base_cache(symbol, output, cache_mgr.settings)
                except Exception:
                    # フォールバック: 指標を計算し直す
                    output = compute_base_indicators(raw_data)
                    save_base_cache(symbol, output, cache_mgr.settings)
                return output

            # 通常の指標計算
            output = compute_base_indicators(raw_data)
            save_base_cache(symbol, output, cache_mgr.settings)
            return output

    return df.reset_index() if df is not None else None


def _read_legacy_cache(symbol: str) -> pd.DataFrame | None:
    """レガシーキャッシュ位置からの読み取り（後方互換性）"""
    legacy_path = Path("data_cache") / f"{safe_filename(symbol)}.csv"
    if not legacy_path.exists():
        return None
    try:
        return pd.read_csv(legacy_path)
    except Exception:
        return None
