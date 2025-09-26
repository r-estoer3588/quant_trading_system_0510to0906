"""繝ｪ繝輔ぃ繧ｯ繧ｿ繝ｪ繝ｳ繧ｰ蠕後・CacheManager繧ｯ繝ｩ繧ｹ"""

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

# 螳壽焚螳夂ｾｩ
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
    "return_6d",
    "return_pct",
    "drop3d",
    "atr_ratio",
    "atr_pct",
)


class CacheManager:
    """繝ｪ繝輔ぃ繧ｯ繧ｿ繝ｪ繝ｳ繧ｰ蠕後・繧ｭ繝｣繝・す繝･邂｡逅・け繝ｩ繧ｹ・郁ｲｬ莉ｻ蛻・牡貂医∩・・""

    _DEFAULT_INSTANCE: ClassVar[CacheManager | None] = None

    def __init__(self, settings: Settings):
        self.settings = settings
        self.full_dir = Path(settings.cache.full_dir)
        self.rolling_dir = Path(settings.cache.rolling_dir)
        self.rolling_cfg = settings.cache.rolling
        self.rolling_meta_path = self.rolling_dir / self.rolling_cfg.meta_file

        # 蜷・ョ繧｣繝ｬ繧ｯ繝医Μ繧剃ｽ懈・
        self.full_dir.mkdir(parents=True, exist_ok=True)
        self.rolling_dir.mkdir(parents=True, exist_ok=True)

        # 萓晏ｭ倥さ繝ｳ繝昴・繝阪Φ繝・
        self.file_io = CacheFileIO(settings)
        self.health_checker = CacheHealthChecker("[CacheManager]")

        # 繝ｭ繝ｼ繧ｫ繝ｫ險ｭ螳・
        self._ui_prefix = "[CacheManager]"

    @classmethod
    def get_default_instance(cls) -> CacheManager:
        """繝・ヵ繧ｩ繝ｫ繝医う繝ｳ繧ｹ繧ｿ繝ｳ繧ｹ繧貞叙蠕暦ｼ医す繝ｳ繧ｰ繝ｫ繝医Φ繝代ち繝ｼ繝ｳ・・""
        if cls._DEFAULT_INSTANCE is None:
            settings = get_settings(create_dirs=False)
            cls._DEFAULT_INSTANCE = cls(settings)
        return cls._DEFAULT_INSTANCE

    @classmethod
    def reset_default_instance(cls) -> None:
        """繝・ヵ繧ｩ繝ｫ繝医う繝ｳ繧ｹ繧ｿ繝ｳ繧ｹ繧偵Μ繧ｻ繝・ヨ・井ｸｻ縺ｫ繝・せ繝育畑・・""
        cls._DEFAULT_INSTANCE = None

    def read(self, ticker: str, profile: str) -> pd.DataFrame | None:
        """繧ｭ繝｣繝・す繝･縺九ｉDataFrame繧定ｪｭ縺ｿ蜿悶ｊ"""
        base_dir = self.full_dir if profile == "full" else self.rolling_dir
        path = self.file_io.detect_file_path(base_dir, ticker)

        df = self.file_io.read_dataframe(path)
        if df is None:
            return None

        # 繝・・繧ｿ縺ｮ豁｣隕丞喧
        df = prepare_dataframe_for_cache(df)

        # 蛛･蜈ｨ諤ｧ繝√ぉ繝・け
        self.health_checker.check_dataframe_health(df, ticker, profile)

        return df

    def write_atomic(self, df: pd.DataFrame, ticker: str, profile: str) -> None:
        """DataFrame繧偵い繝医Α繝・け縺ｫ繧ｭ繝｣繝・す繝･縺ｸ譖ｸ縺崎ｾｼ縺ｿ"""
        base_dir = self.full_dir if profile == "full" else self.rolling_dir
        path = self.file_io.detect_file_path(base_dir, ticker)

        # 荳ｸ繧∝・逅・
        if profile == "rolling":
            round_decimals = getattr(self.rolling_cfg, "round_decimals", None)
        else:
            round_decimals = getattr(self.settings.cache, "round_decimals", None)

        df_to_write = round_dataframe(df, round_decimals)

        # 繧｢繝医Α繝・け譖ｸ縺崎ｾｼ縺ｿ
        self.file_io.write_dataframe_atomic(df_to_write, path)

    def upsert_both(self, ticker: str, new_rows: pd.DataFrame) -> None:
        """譁ｰ縺励＞陦後ｒ'full'縺ｨ'rolling'荳｡譁ｹ縺ｮ繧ｭ繝｣繝・す繝･縺ｫ霑ｽ蜉"""
        for profile in ("full", "rolling"):
            self._upsert_one(ticker, new_rows, profile)

    def _upsert_one(self, ticker: str, new_rows: pd.DataFrame, profile: str) -> None:
        """蜊倅ｸ繝励Ο繝輔ぃ繧､繝ｫ縺ｸ縺ｮ陦瑚ｿｽ蜉蜃ｦ逅・""
        # 譁ｰ縺励＞陦後・蜑榊・逅・
        if new_rows is not None and not new_rows.empty and "date" in new_rows.columns:
            new_rows = new_rows.copy()
            new_rows["date"] = pd.to_datetime(new_rows["date"], errors="coerce")
            new_rows = new_rows.dropna(subset=["date"])

        # 譌｢蟄倥ョ繝ｼ繧ｿ縺ｮ隱ｭ縺ｿ蜿悶ｊ
        current_data = self.read(ticker, profile)

        # 繝・・繧ｿ縺ｮ繝槭・繧ｸ
        if current_data is None or current_data.empty:
            merged = new_rows.copy() if new_rows is not None else pd.DataFrame()
        else:
            merged = (
                pd.concat([current_data, new_rows], ignore_index=True)
                if new_rows is not None
                else current_data
            )

        if not merged.empty:
            # 繧ｽ繝ｼ繝医・驥崎､・勁蜴ｻ
            merged = merged.sort_values("date").drop_duplicates("date").reset_index(drop=True)

            # Rolling遯薙・驕ｩ逕ｨ
            if profile == "rolling":
                merged = self._enforce_rolling_window(merged)

            # 謖・ｨ吶・蜀崎ｨ育ｮ・
            merged = self._recompute_indicators(merged)

        # 譖ｸ縺崎ｾｼ縺ｿ
        if not merged.empty:
            self.write_atomic(merged, ticker, profile)

    @property
    def _rolling_target_length(self) -> int:
        """Rolling遯薙・逶ｮ讓咎聞"""
        return int(self.rolling_cfg.base_lookback_days + self.rolling_cfg.buffer_days)

    def _enforce_rolling_window(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling遯薙・繧ｵ繧､繧ｺ蛻ｶ髯舌ｒ驕ｩ逕ｨ"""
        if "date" not in df.columns or df.empty:
            return df

        target_length = self._rolling_target_length
        if len(df) > target_length:
            return df.tail(target_length).reset_index(drop=True)
        return df

    def _recompute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """蝓ｺ譛ｬOHLC繝・・繧ｿ縺梧峩譁ｰ縺輔ｌ縺滄圀縺ｫ豢ｾ逕滓欠讓吶ｒ蜀崎ｨ育ｮ・""
        if df is None or df.empty or "date" not in df.columns:
            return df

        # 蠢・亥・縺ｮ繝√ぉ繝・け
        required_cols = {"open", "high", "low", "close"}
        is_valid, missing_cols = validate_required_columns(df, required_cols)
        if not is_valid:
            logger.warning(f"蠢・亥・荳崎ｶｳ縺ｮ縺溘ａ謖・ｨ呵ｨ育ｮ励ｒ繧ｹ繧ｭ繝・・: {missing_cols}")
            return df

        # 繝・・繧ｿ縺ｮ貅門ｙ
        base_data = df.copy()
        base_data["date"] = pd.to_datetime(base_data["date"], errors="coerce")
        base_data = base_data.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        if base_data.empty:
            return df

        # OHLCV蛻励・謨ｰ蛟､蛹・
        for col in ("open", "high", "low", "close", "volume"):
            if col in base_data.columns:
                base_data[col] = pd.to_numeric(base_data[col], errors="coerce")

        # indicators_common縺ｨ縺ｮ莠呈鋤諤ｧ縺ｮ縺溘ａ縺ｮ蛻怜錐螟画鋤
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
            # 謖・ｨ呵ｨ育ｮ・
            enriched = add_indicators(base_renamed)
            enriched = enriched.drop(columns=["Date"], errors="ignore")
            enriched.columns = [str(c).lower() for c in enriched.columns]
            enriched["date"] = pd.to_datetime(
                enriched.get("date", base_data["date"]), errors="coerce"
            )

            # 蜈・・DataFrame縺ｨ謖・ｨ呵ｨ育ｮ礼ｵ先棡繧偵・繝ｼ繧ｸ
            result = df.copy()
            ohlcv_columns = {"open", "high", "low", "close", "volume"}

            for col, series in enriched.items():
                if col == "date":
                    result["date"] = series
                    continue
                if col in ohlcv_columns:
                    # 蜈・・OHLCV蛻励・菫晄戟
                    continue
                # 謖・ｨ吝・縺ｯ譖ｴ譁ｰ
                result[col] = series

            # 驥崎､・・縺ｮ髯､蜴ｻ
            return result.loc[:, ~result.columns.duplicated(keep="first")]

        except Exception as e:
            logger.error(f"謖・ｨ吝・險育ｮ怜､ｱ謨・ {e}")
            return df

    def prune_rolling_if_needed(self, anchor_ticker: str = "SPY") -> dict[str, int]:
        """Rolling 繧ｭ繝｣繝・す繝･縺ｮ螳壽悄逧・↑蜑ｪ螳・""
        try:
            # 繝｡繧ｿ繝・・繧ｿ縺ｮ隱ｭ縺ｿ蜿悶ｊ
            if self.rolling_meta_path.exists():
                meta_content = self.rolling_meta_path.read_text(encoding="utf-8")
                last_meta = json.loads(meta_content)
            else:
                last_meta = {"anchor_rows_at_prune": 0}
        except (FileNotFoundError, json.JSONDecodeError):
            last_meta = {"anchor_rows_at_prune": 0}

        # 繧｢繝ｳ繧ｫ繝ｼ驫俶氛縺ｮ繝・・繧ｿ繧ｵ繧､繧ｺ繝√ぉ繝・け
        anchor_df = self.read(anchor_ticker, "rolling")
        if anchor_df is None or anchor_df.empty:
            logger.info(f"{self._ui_prefix} rolling譛ｪ謨ｴ蛯吶・縺溘ａprune蜃ｦ逅・ｒ繧ｹ繧ｭ繝・・")
            return {"pruned_files": 0, "dropped_rows_total": 0}

        current_rows = len(anchor_df)
        previous_rows = int(last_meta.get("anchor_rows_at_prune", 0))
        progress = current_rows - previous_rows

        prune_threshold = int(self.rolling_cfg.prune_chunk_days)
        if progress < prune_threshold:
            logger.info(
                f"{self._ui_prefix} 騾ｲ謐養progress}蝟ｶ讌ｭ譌･ (< {prune_threshold}) " "縺ｮ縺溘ａprune荳崎ｦ・
            )
            return {"pruned_files": 0, "dropped_rows_total": 0}

        # 蜑ｪ螳壼・逅・・髢句ｧ・
        logger.info(
            f"{self._ui_prefix} 竢ｳ prune髢句ｧ・ anchor={anchor_ticker}, " f"騾ｲ謐・{progress}蝟ｶ讌ｭ譌･"
        )

        pruned_files = 0
        dropped_rows_total = 0

        # 蜷・ヵ繧｡繧､繝ｫ縺ｮ蜑ｪ螳・
        for file_path in self.rolling_dir.glob("*.*"):
            if file_path.name.startswith("_"):  # 繝｡繧ｿ繝輔ぃ繧､繝ｫ縺ｯ繧ｹ繧ｭ繝・・
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

        # 繝｡繧ｿ繝・・繧ｿ縺ｮ譖ｴ譁ｰ
        new_meta = {"anchor_rows_at_prune": current_rows}
        self.rolling_meta_path.write_text(json.dumps(new_meta, indent=2), encoding="utf-8")

        logger.info(
            f"{self._ui_prefix} 笨・prune螳御ｺ・ files={pruned_files}, "
            f"dropped_rows={dropped_rows_total}"
        )

        return {"pruned_files": pruned_files, "dropped_rows_total": dropped_rows_total}


def _base_directory() -> Path:
    """繝吶・繧ｹ繧ｭ繝｣繝・す繝･繝・ぅ繝ｬ繧ｯ繝医Μ繧貞叙蠕・""
    settings = get_settings(create_dirs=True)
    base_dir = Path(settings.DATA_CACHE_DIR) / BASE_SUBDIR
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def compute_base_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV縺ｮDataFrame縺ｫ蜈ｱ騾壹・繝ｼ繧ｹ謖・ｨ吶ｒ莉伜刈・亥ｾ捺擂縺ｮ莠呈鋤髢｢謨ｰ・・""
    if df is None or df.empty:
        return df

    normalized = df.copy()

    # 蛻怜錐豁｣隕丞喧
    rename_map = {c: c.lower() for c in normalized.columns}
    normalized = normalized.rename(columns=rename_map)

    # Date蛻励・險ｭ螳・
    if "date" in normalized.columns:
        normalized = normalized.rename(columns={"date": "Date"})
    if "Date" in normalized.columns:
        normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
        normalized = normalized.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

    # OHLCV蛻励・讓呎ｺ門喧
    normalized = standardize_ohlcv_columns(normalized.reset_index())
    if "Date" in normalized.columns:
        normalized = normalized.set_index("Date")

    # 蠢・亥・繝√ぉ繝・け
    required_columns = {"High", "Low", "Close"}
    if not required_columns.issubset(normalized.columns):
        missing = required_columns - set(normalized.columns)
        logger.warning(f"蠢・亥・谺關ｽ縺ｮ縺溘ａ繧､繝ｳ繧ｸ險育ｮ励ｒ繧ｹ繧ｭ繝・・: missing={missing}")
        return normalized.reset_index()

    # 萓｡譬ｼ邉ｻ蛻励・謨ｰ蛟､蛹・
    close = pd.to_numeric(normalized["Close"], errors="coerce")
    high = pd.to_numeric(normalized["High"], errors="coerce")
    low = pd.to_numeric(normalized["Low"], errors="coerce")
    volume = None
    if "Volume" in normalized.columns:
        volume = pd.to_numeric(normalized["Volume"], errors="coerce")

    # SMA險育ｮ・
    for period in [25, 50, 100, 150, 200]:
        normalized[f"sma{period}"] = close.rolling(period).mean()

    # EMA險育ｮ・
    for period in [20, 50]:
        normalized[f"ema{period}"] = close.ewm(span=period, adjust=False).mean()

    # ATR險育ｮ・
    true_range = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)

    for period in [10, 14, 40, 50]:
        normalized[f"atr{period}"] = true_range.rolling(period).mean()

    # RSI險育ｮ暦ｼ・ilder縺ｮ繧ｹ繝繝ｼ繧ｸ繝ｳ繧ｰ・・
    def calculate_rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(alpha=1 / period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    for period in [3, 14]:
        normalized[f"rsi{period}"] = calculate_rsi(close, period)

    # ROC & HV險育ｮ・
    normalized["roc200"] = close.pct_change(200) * 100.0
    log_returns = (close / close.shift(1)).apply(np.log)
    standard_deviation = log_returns.rolling(50).std()
    normalized["hv50"] = standard_deviation * np.sqrt(252) * 100.0

    # 繝懊Μ繝･繝ｼ繝邉ｻ謖・ｨ・
    if volume is not None:
        normalized["dollarvolume20"] = (close * volume).rolling(20).mean()
        normalized["dollarvolume50"] = (close * volume).rolling(50).mean()

    return normalized.reset_index()


def base_cache_path(symbol: str) -> Path:
    """繝吶・繧ｹ繧ｭ繝｣繝・す繝･繝輔ぃ繧､繝ｫ縺ｮ繝代せ繧貞叙蠕・""
    return _base_directory() / f"{safe_filename(symbol)}.csv"


def save_base_cache(symbol: str, df: pd.DataFrame, settings: Settings | None = None) -> Path:
    """繝吶・繧ｹ繧ｭ繝｣繝・す繝･DataFrame繧辰SV繝輔ぃ繧､繝ｫ縺ｫ菫晏ｭ・""
    file_path = base_cache_path(symbol)
    df_to_save = df.reset_index() if df.index.name is not None else df
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if settings is None:
        settings = get_settings(create_dirs=True)

    # 荳ｸ繧∝・逅・
    round_decimals = getattr(getattr(settings, "cache", None), "round_decimals", None)
    df_rounded = round_dataframe(df_to_save, round_decimals)

    # CSV譖ｸ縺崎ｾｼ縺ｿ
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
    """繝吶・繧ｹ繧ｭ繝｣繝・す繝･繧定ｪｭ縺ｿ霎ｼ縺ｿ・磯ｮｮ蠎ｦ讀懆ｨｼ繝ｻ蜀肴ｧ狗ｯ峨が繝励す繝ｧ繝ｳ莉倥″・・""
    cache_mgr = cache_manager or CacheManager.get_default_instance()
    file_path = base_cache_path(symbol)
    df = None

    # 譌｢蟄倥く繝｣繝・す繝･縺ｮ隱ｭ縺ｿ蜿悶ｊ
    if file_path.exists():
        try:
            df = pd.read_csv(file_path, parse_dates=["Date"])
            df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
        except Exception:
            df = None

    # 魄ｮ蠎ｦ繝√ぉ繝・け
    if df is not None:
        last_date = df.index[-1] if not df.empty else None
        is_stale = False

        if allowed_recent_dates and last_date:
            allowed_timestamps = set()
            for date_obj in allowed_recent_dates:
                try:
                    if date_obj is not None:
                        # 繧医ｊ螳牙・縺ｪ蝙句､画鋤
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

        logger.info(f"繝吶・繧ｹ繧ｭ繝｣繝・す繝･({symbol})縺悟商縺・◆繧∝・讒狗ｯ峨＠縺ｾ縺・)
        df = None

    # 繧ｭ繝｣繝・す繝･蜀肴ｧ狗ｯ・
    if df is None and rebuild_if_missing:
        raw_data = (
            cache_mgr.read(symbol, "full")
            or cache_mgr.read(symbol, "rolling")
            or _read_legacy_cache(symbol)
        )

        if raw_data is not None and not raw_data.empty:
            # 譌｢蟄倥・謖・ｨ吝・繧貞━蜈医☆繧句ｴ蜷・
            try:
                lowercase_columns = {c.lower() for c in raw_data.columns}
            except Exception:
                lowercase_columns = set()

            has_indicators = any(col in lowercase_columns for col in MAIN_INDICATOR_COLUMNS)

            if prefer_precomputed_indicators and has_indicators:
                output = raw_data.copy()
                # Date蛻励・豁｣隕丞喧
                if "date" in output.columns and "Date" not in output.columns:
                    output = output.rename(columns={"date": "Date"})
                if "Date" not in output.columns and isinstance(output.index, pd.DatetimeIndex):
                    try:
                        output = output.reset_index()
                        if output.columns[0].lower() == "index":
                            output = output.rename(columns={output.columns[0]: "Date"})
                    except Exception:
                        pass

                try:
                    save_base_cache(symbol, output, cache_mgr.settings)
                except Exception:
                    # 繝輔か繝ｼ繝ｫ繝舌ャ繧ｯ: 謖・ｨ吶ｒ險育ｮ励＠逶ｴ縺・
                    output = compute_base_indicators(raw_data)
                    save_base_cache(symbol, output, cache_mgr.settings)
                return output

            # 騾壼ｸｸ縺ｮ謖・ｨ呵ｨ育ｮ・
            output = compute_base_indicators(raw_data)
            save_base_cache(symbol, output, cache_mgr.settings)
            return output

    return df.reset_index() if df is not None else None


def _read_legacy_cache(symbol: str) -> pd.DataFrame | None:
    """繝ｬ繧ｬ繧ｷ繝ｼ繧ｭ繝｣繝・す繝･菴咲ｽｮ縺九ｉ縺ｮ隱ｭ縺ｿ蜿悶ｊ・亥ｾ梧婿莠呈鋤諤ｧ・・""
    legacy_path = Path("data_cache") / f"{safe_filename(symbol)}.csv"
    if not legacy_path.exists():
        return None
    try:
        return pd.read_csv(legacy_path)
    except Exception:
        return None
