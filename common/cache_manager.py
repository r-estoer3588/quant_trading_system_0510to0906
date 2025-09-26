from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import threading
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
import pandas as pd

from common.utils import describe_dtype, safe_filename
from config.settings import Settings, get_settings
from indicators_common import add_indicators

logger = logging.getLogger(__name__)

BASE_SUBDIR = "base"


class _RollingIssueAggregator:
    """
    rolling cache未整備ログを集約し、冗長出力を制御するクラス。

    環境変数:
    - COMPACT_TODAY_LOGS=1: 集約機能有効化
    - ROLLING_ISSUES_VERBOSE_HEAD=N: 先頭N件のみ詳細WARNING、以降はDEBUG
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.compact_mode = os.getenv("COMPACT_TODAY_LOGS", "0") == "1"
        self.verbose_head = int(os.getenv("ROLLING_ISSUES_VERBOSE_HEAD", "20"))
        self.issues = defaultdict(list)  # category -> [symbols]
        self.warning_count = 0
        self.logger = logging.getLogger(__name__)
        self._initialized = True

        if self.compact_mode:
            # プロセス終了時にサマリーを出力
            atexit.register(self._output_summary)

    def report_issue(self, category: str, symbol: str, message: str = "") -> None:
        """
        rolling cache の未整備問題を報告する。

        Args:
            category: 問題カテゴリ（例: "missing_rolling", "insufficient_data"）
            symbol: 対象シンボル
            message: 追加メッセージ（省略可）
        """
        if not self.compact_mode:
            # 従来通りの個別WARNING
            full_msg = f"[{category}] {symbol}"
            if message:
                full_msg += f": {message}"
            self.logger.warning(full_msg)
            return

        # 集約モード
        self.issues[category].append(symbol)
        self.warning_count += 1

        # 先頭N件のみ詳細WARNING
        if len(self.issues[category]) <= self.verbose_head:
            full_msg = f"[{category}] {symbol}"
            if message:
                full_msg += f": {message}"
            self.logger.warning(full_msg)
        else:
            # N件を超えたらDEBUGレベル
            full_msg = f"[{category}] {symbol}"
            if message:
                full_msg += f": {message}"
            self.logger.debug(full_msg)

    def _output_summary(self) -> None:
        """プロセス終了時にカテゴリ別サマリを出力する。"""
        if not self.issues:
            return

        self.logger.info("=== Rolling Cache Issues Summary ===")
        total_issues = sum(len(symbols) for symbols in self.issues.values())
        self.logger.info(f"Total issues reported: {total_issues}")

        for category, symbols in self.issues.items():
            unique_symbols = list(set(symbols))  # 重複除去
            count = len(unique_symbols)

            if count <= 10:
                symbol_list = ", ".join(unique_symbols)
                self.logger.info(f"[{category}]: {count} symbols - {symbol_list}")
            else:
                sample = ", ".join(unique_symbols[:5])
                self.logger.info(
                    f"[{category}]: {count} symbols - {sample} ... (+{count-5} more)"
                )


# グローバルインスタンス
_rolling_issue_aggregator = _RollingIssueAggregator()


def report_rolling_issue(category: str, symbol: str, message: str = "") -> None:
    """
    rolling cache の未整備問題をグローバルアグリゲーターに報告する。

    Args:
        category: 問題カテゴリ（例: "missing_rolling", "insufficient_data"）
        symbol: 対象シンボル
        message: 追加メッセージ（省略可）
    """
    _rolling_issue_aggregator.report_issue(category, symbol, message)


def round_dataframe(df: pd.DataFrame, decimals: int | None) -> pd.DataFrame:
    """Return a DataFrame rounded to the requested number of decimals.

    pandas.DataFrame.round は数値列のみを対象とし、日付や文字列列には影響しない。
    ただし ``decimals`` が不正値の場合や丸め処理が例外を送出した場合は、
    元の DataFrame をそのまま返す。
    """
    if df is None or decimals is None:
        return df

    try:
        decimals_int = int(decimals)
    except (ValueError, TypeError):
        return df

    # Define category-specific rounding by column name (lowercase)
    price_atr_cols = {
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
    volume_cols = {"volume", "dollarvolume20", "dollarvolume50", "avgvolume50"}
    oscillator_cols = {"rsi3", "rsi4", "rsi14", "adx7"}
    pct_cols = {
        "roc200",
        "return_3d",
        "return_6d",
        "atr_ratio",
        "atr_pct",
        "hv50",
        "return_pct",
    }

    out = df.copy()
    lc_map = {c.lower(): c for c in out.columns}

    def _safe_round(series: pd.Series, ndigits: int) -> pd.Series:
        try:
            return pd.to_numeric(series, errors="coerce").round(ndigits)
        except Exception:
            return series

    # Group and apply rounding
    groups = {
        2: price_atr_cols | oscillator_cols,
        4: pct_cols,
    }
    rounded_cols = set()

    for ndigits, names in groups.items():
        cols_to_round = [lc_map[lname] for lname in names if lname in lc_map]
        for col in cols_to_round:
            out[col] = _safe_round(out[col], ndigits)
        rounded_cols.update(cols_to_round)

    # Volume-like columns
    vol_cols_to_round = [lc_map[lname] for lname in volume_cols if lname in lc_map]
    for col in vol_cols_to_round:
        try:
            s = pd.to_numeric(out[col], errors="coerce").round(0)
            out[col] = s.astype("Int64")
        except Exception:
            out[col] = _safe_round(out[col], 0)
    rounded_cols.update(vol_cols_to_round)

    # Remaining numeric columns
    for col in out.columns:
        if col not in rounded_cols and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = _safe_round(out[col], decimals_int)

    return out


def make_csv_formatters(
    frame: pd.DataFrame, dec_point: str = ".", thous_sep: str | None = None
) -> dict:
    """Create a pandas.to_csv formatters dict honoring decimal point and thousands sep.

    Returns: dict mapping column name -> callable
    """
    lc_map = {c.lower(): c for c in frame.columns}
    fmt: dict = {}

    def _add_thousands_sep(int_str: str, sep: str) -> str:
        neg = int_str.startswith("-")
        prefix = "-" if neg else ""
        num_str = int_str[1:] if neg else int_str
        parts = []
        while num_str:
            parts.append(num_str[-3:])
            num_str = num_str[:-3]
        return prefix + sep.join(reversed(parts))

    def _num_formatter(nd: int):
        def _f(x):
            if pd.isna(x):
                return ""
            try:
                s = f"{float(x):.{nd}f}"
                if thous_sep:
                    int_part, _, frac_part = s.partition(".")
                    int_part = _add_thousands_sep(int_part, thous_sep)
                    s = f"{int_part}.{frac_part}" if frac_part else int_part
                return s.replace(".", dec_point) if dec_point != "." else s
            except (ValueError, TypeError):
                return str(x)

        return _f

    def _int_formatter():
        def _f(x):
            if pd.isna(x):
                return ""
            try:
                s = f"{int(round(float(x))):d}"
                return _add_thousands_sep(s, thous_sep) if thous_sep else s
            except (ValueError, TypeError):
                return str(x)

        return _f

    # Define formatting rules
    rules = {
        _num_formatter(2): [
            "open",
            "close",
            "high",
            "low",
            "atr10",
            "atr14",
            "atr20",
            "atr40",
            "atr50",
            "rsi3",
            "rsi4",
            "rsi14",
            "adx7",
        ],
        _num_formatter(4): [
            "roc200",
            "return_3d",
            "return_6d",
            "atr_ratio",
            "atr_pct",
            "hv50",
        ],
        _int_formatter(): ["volume", "dollarvolume20", "dollarvolume50", "avgvolume50"],
    }

    for formatter, names in rules.items():
        for name in names:
            if name in lc_map:
                fmt[lc_map[name]] = formatter
    return fmt


def _write_dataframe_to_csv(df: pd.DataFrame, path: Path, settings: Settings) -> None:
    """Utility to write a DataFrame to CSV with formatting."""
    # Be defensive: tests may supply a SimpleNamespace for settings lacking
    # nested attributes (e.g. settings.cache.csv). Use getattr fallbacks and
    # sensible defaults to avoid raising AttributeError here.
    try:
        cache_cfg = getattr(settings, "cache", None)
        csv_cfg = getattr(cache_cfg, "csv", None)
        dec_point = getattr(csv_cfg, "decimal_point", ".")
        thous_sep = getattr(csv_cfg, "thousands_sep", None)
        sep = getattr(csv_cfg, "field_sep", ",")

        fmt_map = make_csv_formatters(df, dec_point=dec_point, thous_sep=thous_sep)

        if fmt_map:
            df_out = df.copy()
            for col, func in fmt_map.items():
                if col in df_out.columns:
                    try:
                        df_out[col] = df_out[col].apply(func)
                    except Exception:
                        df_out[col] = df_out[col].astype(str)
            df_out.to_csv(path, index=False, sep=sep)
        else:
            df.to_csv(path, index=False, decimal=dec_point, sep=sep)
    except Exception as e:
        logger.error(f"Failed to write formatted CSV {path.name}: {e}")
        try:
            df.to_csv(path, index=False)
        except Exception as e2:
            logger.error(f"Failed to write CSV fallback {path.name}: {e2}")


# 健全性チェックで参照する主要指標列（読み込み後は小文字化される）
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
    "return_pct",
    "drop3d",
    "atr_ratio",
    "atr_pct",
)

# 各指標列が有効値を持つために最低限必要とする観測日数の目安
_INDICATOR_MIN_OBSERVATIONS: dict[str, int] = {
    "sma25": 20,
    "sma50": 50,
    "sma100": 100,
    "sma150": 150,
    "sma200": 200,
    "ema20": 1,
    "ema50": 1,
    "atr10": 11,
    "atr14": 15,
    "atr20": 21,
    "atr40": 41,
    "atr50": 51,
    "adx7": 14,
    "rsi3": 3,
    "rsi4": 4,
    "rsi14": 14,
    "roc200": 201,
    "hv50": 51,
    "dollarvolume20": 20,
    "dollarvolume50": 50,
    "avgvolume50": 50,
    "return_3d": 4,
    "return_6d": 7,
    "return_pct": 2,
    "drop3d": 4,
    "atr_ratio": 11,
    "atr_pct": 11,
}


class CacheManager:
    """
    二層キャッシュ管理（full / rolling）。
    - 既存のフォーマット(csv/parquet)は自動検出・踏襲
    - system5/6スタイルのコメント・進捗ログ粒度を踏襲
    """

    _GLOBAL_WARNED: ClassVar[set[tuple[str, str, str]]] = set()

    def __init__(self, settings: Settings):
        self.settings = settings
        self.full_dir = Path(settings.cache.full_dir)
        self.rolling_dir = Path(settings.cache.rolling_dir)
        self.rolling_cfg = settings.cache.rolling
        self.file_format = getattr(settings.cache, "file_format", "auto")
        self.rolling_meta_path = self.rolling_dir / self.rolling_cfg.meta_file
        self.full_dir.mkdir(parents=True, exist_ok=True)
        self.rolling_dir.mkdir(parents=True, exist_ok=True)
        self._ui_prefix = "[CacheManager]"
        self._warned = self._GLOBAL_WARNED

    def _warn_once(
        self, ticker: str, profile: str, category: str, message: str
    ) -> None:
        key = (ticker, profile, category)
        if key in self._warned:
            return
        self._warned.add(key)
        logger.warning(message)

    def _recompute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recalculate derived indicator columns when base OHLC data is updated."""
        if df is None or df.empty or "date" not in df.columns:
            return df

        required = {"open", "high", "low", "close"}
        if not required.issubset(set(df.columns)):
            return df

        base = df.copy()
        base["date"] = pd.to_datetime(base["date"], errors="coerce")
        base = base.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        if base.empty:
            return df

        for col in ("open", "high", "low", "close", "volume"):
            if col in base.columns:
                base[col] = pd.to_numeric(base[col], errors="coerce")

        case_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
        base_renamed = base.rename(
            columns={k: v for k, v in case_map.items() if k in base.columns}
        )
        base_renamed["Date"] = base_renamed["date"]

        try:
            enriched = add_indicators(base_renamed)
            enriched = enriched.drop(columns=["Date"], errors="ignore")
            # 指標列を標準化（大文字統一）、その他は小文字化
            enriched = standardize_indicator_columns(enriched)
            # 基本列（date, open, high等）のみ小文字に変換
            basic_cols = {"open", "high", "low", "close", "volume", "date"}
            enriched.columns = [
                c.lower() if c.lower() in basic_cols else c for c in enriched.columns
            ]
            enriched["date"] = pd.to_datetime(
                enriched.get("date", base["date"]), errors="coerce"
            )

            # Overwrite indicator columns with freshly computed values while
            # preserving original OHLCV and date columns. This ensures appended
            # rows receive correct indicator values and existing indicators are
            # consistent with the latest OHLC history.
            combined = df.copy()
            ohlcv = {"open", "high", "low", "close", "volume"}
            for col, series in enriched.items():
                if col == "date":
                    # ensure date column exists and normalized
                    combined["date"] = series
                    continue
                if col in ohlcv:
                    # keep original OHLCV from df
                    continue
                # replace or create indicator columns from enriched
                combined[col] = series

            # drop any duplicated columns just in case
            return combined.loc[:, ~combined.columns.duplicated(keep="first")]
        except Exception as e:
            logger.error(f"Failed to recompute indicators: {e}")
            return df

    def _detect_path(self, base_dir: Path, ticker: str) -> Path:
        """Detects the cache file path, defaulting based on settings."""
        for ext in [".csv", ".parquet", ".feather"]:
            if (p := base_dir / f"{ticker}{ext}").exists():
                return p

        fmt = (self.file_format or "auto").lower()
        if fmt == "parquet":
            return base_dir / f"{ticker}.parquet"
        if fmt == "feather":
            return base_dir / f"{ticker}.feather"
        return base_dir / f"{ticker}.csv"

    def _read_with_fallback(
        self, path: Path, ticker: str, profile: str
    ) -> pd.DataFrame | None:
        """Reads a file with specific logic for different formats and fallbacks."""
        if not path.exists():
            return None

        try:
            if path.suffix == ".feather":
                return pd.read_feather(path)
            if path.suffix == ".parquet":
                return pd.read_parquet(path)
            if path.suffix == ".csv":
                try:
                    return pd.read_csv(path, parse_dates=["date"])
                except ValueError as e:
                    if "Missing column provided to 'parse_dates': 'date'" in str(e):
                        df = pd.read_csv(path)
                        if "Date" in df.columns:
                            df = df.rename(columns={"Date": "date"})
                            df["date"] = pd.to_datetime(df["date"], errors="coerce")
                        return df
                    raise
            return None
        except Exception as e:
            msg = f"{self._ui_prefix} 読み込み失敗: {path.name} ({e})"
            self._warn_once(ticker, profile, f"read_error:{path.name}", msg)
            # Try CSV as a last resort if another format failed
            if path.suffix != ".csv":
                csv_path = path.with_suffix(".csv")
                if csv_path.exists():
                    return self._read_with_fallback(csv_path, ticker, profile)
            return None

    def read(self, ticker: str, profile: str) -> pd.DataFrame | None:
        """Reads data from the cache, handling different formats and normalization."""
        base = self.full_dir if profile == "full" else self.rolling_dir
        path = self._detect_path(base, ticker)

        df = self._read_with_fallback(path, ticker, profile)
        if df is None:
            # rolling cacheが見つからない場合は集約ログに報告
            if profile == "rolling":
                report_rolling_issue(
                    "missing_rolling", ticker, "rolling cache not found"
                )
            return None

        # Normalize columns
        df.columns = [c.lower() for c in df.columns]
        if df.columns.has_duplicates:
            df = df.loc[:, ~df.columns.duplicated(keep="first")]

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = (
                df.dropna(subset=["date"])
                .sort_values("date")
                .drop_duplicates("date")
                .reset_index(drop=True)
            )

        # 指標列を大文字に標準化（新機能）
        df = standardize_indicator_columns(df)

        self._perform_health_check(df, ticker, profile)
        return df

    def write_atomic(self, df: pd.DataFrame, ticker: str, profile: str) -> None:
        """Atomically writes a DataFrame to the cache."""
        base = self.full_dir if profile == "full" else self.rolling_dir
        base.mkdir(parents=True, exist_ok=True)
        path = self._detect_path(base, ticker)
        tmp_path = path.with_suffix(path.suffix + ".tmp")

        try:
            # settings may be a SimpleNamespace in tests; use getattr fallbacks
            if profile == "rolling":
                round_dec = getattr(
                    getattr(self, "rolling_cfg", None), "round_decimals", None
                )
            else:
                # Prefer nested settings.cache.round_decimals when available
                round_dec = None
                try:
                    round_dec = getattr(self.settings.cache, "round_decimals", None)
                except Exception:
                    try:
                        round_dec = getattr(self.settings, "round_decimals", None)
                    except Exception:
                        round_dec = None
            df_to_write = round_dataframe(df, round_dec)

            if path.suffix == ".parquet":
                df_to_write.to_parquet(tmp_path, index=False)
            elif path.suffix == ".feather":
                df_to_write.reset_index(drop=True).to_feather(tmp_path)
            else:  # .csv
                _write_dataframe_to_csv(df_to_write, tmp_path, self.settings)

            shutil.move(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError as e:
                    logger.error(f"Failed to remove temporary file {tmp_path}: {e}")

    def _check_nan_rates(self, df: pd.DataFrame, ticker: str, profile: str):
        """Checks for high NaN rates in key indicator columns."""
        try:
            if df is None or df.empty:
                return

            # Build list of indicator columns present in df
            cols = [c for c in df.columns if c.lower() in _INDICATOR_MIN_OBSERVATIONS]
            if not cols:
                return

            # For each column compute a NaN ratio over a recent window but
            # exclude the initial warm-up rows where the indicator cannot exist.
            recent_window = min(len(df), 120)
            warnings: list[tuple[str, float]] = []
            for col in cols:
                try:
                    series = pd.to_numeric(df[col], errors="coerce").reset_index(
                        drop=True
                    )
                except Exception:
                    continue

                lookback = int(_INDICATOR_MIN_OBSERVATIONS.get(col.lower(), 0))

                # If series is shorter than lookback, skip — indicator not applicable
                # This handles newly listed stocks where NaN is expected
                if lookback and len(series) <= lookback:
                    continue

                # If the entire series is NaN, warn (indicator missing entirely)
                # But only after confirming we have sufficient data length
                if series.isna().all():
                    warnings.append((col, 1.0))
                    continue

                # Consider only the recent portion where values should be populated
                eval_series = series.tail(recent_window)
                try:
                    nan_ratio = float(eval_series.isna().mean())
                except Exception:
                    nan_ratio = 1.0

                if nan_ratio >= 0.99:
                    warnings.append((col, nan_ratio))

            if warnings:
                # Log a single warning summarizing affected columns
                parts = ", ".join(f"{c}:{r:.2%}" for c, r in warnings)
                msg = f"{self._ui_prefix} ⚠️ {ticker} {profile} cache: NaN率高 ({parts})"
                self._warn_once(ticker, profile, f"nan_rate:{parts}", msg)
        except Exception as e:
            logger.error(f"{self._ui_prefix} NaN率チェック失敗: {e}")

    def _check_column_dtypes(self, df: pd.DataFrame, ticker: str, profile: str):
        """Checks for incorrect dtypes in OHLCV columns."""
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                dtype_repr = describe_dtype(df[col])
                msg = f"{self._ui_prefix} ⚠️ {ticker} {profile} cache: {col}型不一致 ({dtype_repr})"
                self._warn_once(ticker, profile, f"dtype:{col}:{dtype_repr}", msg)

    def _check_non_positive_prices(self, df: pd.DataFrame, ticker: str, profile: str):
        """Checks if price columns contain only non-positive values."""
        for col in ["close", "high", "low"]:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                if not vals.empty and (vals <= 0).all():
                    msg = (
                        f"{self._ui_prefix} ⚠️ {ticker} {profile} cache: {col}全て非正値"
                    )
                    self._warn_once(ticker, profile, f"non_positive:{col}", msg)

    def _perform_health_check(
        self, df: pd.DataFrame, ticker: str, profile: str
    ) -> None:
        """Performs a series of health checks on the DataFrame."""
        if df is None or df.empty:
            return
        try:
            # Perform NaN rate checks first
            self._check_nan_rates(df, ticker, profile)
            self._check_column_dtypes(df, ticker, profile)
            self._check_non_positive_prices(df, ticker, profile)
        except Exception as e:
            msg = f"{self._ui_prefix} ⚠️ {ticker} {profile} cache: 健全性チェック失敗 ({e})"
            self._warn_once(
                ticker, profile, f"healthcheck_error:{type(e).__name__}", msg
            )

    def upsert_both(self, ticker: str, new_rows: pd.DataFrame) -> None:
        """Upserts new rows into both 'full' and 'rolling' caches."""
        for profile in ("full", "rolling"):
            self._upsert_one(ticker, new_rows, profile)

    def _upsert_one(self, ticker: str, new_rows: pd.DataFrame, profile: str) -> None:
        if new_rows is not None and not new_rows.empty and "date" in new_rows.columns:
            new_rows = new_rows.copy()
            new_rows["date"] = pd.to_datetime(new_rows["date"], errors="coerce")
            new_rows = new_rows.dropna(subset=["date"])

        cur = self.read(ticker, profile)
        if cur is None or cur.empty:
            merged = new_rows.copy() if new_rows is not None else pd.DataFrame()
        else:
            merged = (
                pd.concat([cur, new_rows], ignore_index=True)
                if new_rows is not None
                else cur
            )

        if not merged.empty:
            merged = (
                merged.sort_values("date")
                .drop_duplicates("date")
                .reset_index(drop=True)
            )
            if profile == "rolling":
                merged = self._enforce_rolling_window(merged)
            merged = self._recompute_indicators(merged)

        if not merged.empty:
            self.write_atomic(merged, ticker, profile)

    @property
    def _rolling_target_len(self) -> int:
        return int(self.rolling_cfg.base_lookback_days + self.rolling_cfg.buffer_days)

    def _enforce_rolling_window(self, df: pd.DataFrame) -> pd.DataFrame:
        if "date" not in df.columns or df.empty:
            return df
        target_len = self._rolling_target_len
        return (
            df.tail(target_len).reset_index(drop=True) if len(df) > target_len else df
        )

    def prune_rolling_if_needed(self, anchor_ticker: str = "SPY") -> dict:
        """Prunes the rolling cache if enough new data has been added."""
        try:
            meta_text = self.rolling_meta_path.read_text(encoding="utf-8")
            last_meta = json.loads(meta_text)
        except (FileNotFoundError, json.JSONDecodeError):
            last_meta = {"anchor_rows_at_prune": 0}

        anchor_df = self.read(anchor_ticker, "rolling")
        if anchor_df is None or anchor_df.empty:
            logger.info(f"{self._ui_prefix} rolling未整備のためpruneスキップ")
            return {"pruned_files": 0, "dropped_rows_total": 0}

        cur_rows = len(anchor_df)
        prev_rows = int(last_meta.get("anchor_rows_at_prune", 0))
        progressed = cur_rows - prev_rows

        prune_chunk = int(self.rolling_cfg.prune_chunk_days)
        if progressed < prune_chunk:
            msg = f"{self._ui_prefix} 進捗{progressed}営業日 (<{prune_chunk}) のためprune不要"
            logger.info(msg)
            return {"pruned_files": 0, "dropped_rows_total": 0}

        msg = f"{self._ui_prefix} ⏳ prune開始: anchor={anchor_ticker}, 進捗={progressed}営業日"
        logger.info(msg)
        pruned_files, dropped_total = 0, 0

        for path in self.rolling_dir.glob("*.*"):
            if path.name.startswith("_"):
                continue

            df = self.read(path.stem, "rolling")
            if df is None or df.empty:
                continue

            can_drop = len(df) - self._rolling_target_len
            drop_n = min(prune_chunk, can_drop)
            if drop_n > 0:
                new_df = df.iloc[drop_n:].reset_index(drop=True)
                self.write_atomic(new_df, path.stem, "rolling")
                pruned_files += 1
                dropped_total += drop_n

        self.rolling_meta_path.write_text(
            json.dumps({"anchor_rows_at_prune": cur_rows}, indent=2), encoding="utf-8"
        )
        msg = f"{self._ui_prefix} ✅ prune完了: files={pruned_files}, dropped_rows={dropped_total}"
        logger.info(msg)
        return {"pruned_files": pruned_files, "dropped_rows_total": dropped_total}

    def analyze_rolling_gaps(self, system_symbols: list[str] | None = None) -> dict:
        """
        rolling cache の整備状況を分析し、未整備シンボルの詳細ログを集約する。

        Args:
            system_symbols: 分析対象シンボルのリスト。Noneの場合はbase cacheの全シンボル
                を対象とする。

        Returns:
            分析結果辞書:
            - total_symbols: 分析対象シンボル数
            - available_in_rolling: rolling cacheに存在するシンボル数
            - missing_from_rolling: rolling cacheに存在しないシンボル数
            - missing_symbols: 未整備シンボルのリスト
            - coverage_percentage: カバレッジ率（%）
        """
        logger.info(f"{self._ui_prefix} 🔍 rolling cache整備状況の分析を開始")

        # 分析対象シンボルの決定
        if system_symbols is None:
            # base cacheから全シンボルを取得
            base_files = list(self.full_dir.parent.glob(f"{BASE_SUBDIR}/*.*"))
            system_symbols = [p.stem for p in base_files if not p.name.startswith("_")]
            logger.info(
                f"{self._ui_prefix} base cacheから{len(system_symbols)}シンボルを検出"
            )
        else:
            logger.info(
                f"{self._ui_prefix} 指定された{len(system_symbols)}シンボルを分析対象とします"
            )

        available_symbols = []
        missing_symbols = []

        # 各シンボルのrolling cache存在確認
        for symbol in system_symbols:
            rolling_data = self.read(symbol, "rolling")
            if rolling_data is not None and not rolling_data.empty:
                available_symbols.append(symbol)
            else:
                missing_symbols.append(symbol)
                # 集約ログに未整備を報告
                report_rolling_issue("missing_from_analysis", symbol)

        total_symbols = len(system_symbols)
        available_count = len(available_symbols)
        missing_count = len(missing_symbols)
        coverage_percentage = (
            (available_count / total_symbols * 100) if total_symbols > 0 else 0
        )

        # 結果ログ
        logger.info(f"{self._ui_prefix} 📊 分析完了:")
        logger.info(f"{self._ui_prefix}   - 分析対象: {total_symbols}シンボル")
        logger.info(
            f"{self._ui_prefix}   - rolling cache整備済み: {available_count}シンボル"
        )
        logger.info(
            f"{self._ui_prefix}   - rolling cache未整備: {missing_count}シンボル"
        )
        logger.info(f"{self._ui_prefix}   - カバレッジ: {coverage_percentage:.1f}%")

        if missing_symbols:
            # 集約ログによる未整備シンボル報告（既存のwarningを置き換え）
            for symbol in missing_symbols[:10]:  # 先頭10件を詳細報告
                report_rolling_issue("missing_analysis_detailed", symbol)

            # 従来の形式のログも条件付きで維持（集約無効時のフォールバック）
            if not _rolling_issue_aggregator.compact_mode:
                logger.warning(
                    f"{self._ui_prefix} 🚨 未整備シンボル: {missing_symbols[:10]}"
                )
                if len(missing_symbols) > 10:
                    logger.warning(
                        f"{self._ui_prefix}   ... 他{len(missing_symbols) - 10}シンボル"
                    )

        return {
            "total_symbols": total_symbols,
            "available_in_rolling": available_count,
            "missing_from_rolling": missing_count,
            "missing_symbols": missing_symbols,
            "coverage_percentage": coverage_percentage,
        }

    def get_rolling_health_summary(self) -> dict:
        """
        rolling cache の健全性サマリーを取得する。

        Returns:
            健全性サマリー辞書:
            - meta_exists: メタファイルの存在
            - meta_content: メタファイルの内容
            - rolling_files_count: rolling cacheファイル数
            - target_length: 目標データ長
            - anchor_symbol_status: アンカーシンボル（SPY）の状態
        """
        logger.info(f"{self._ui_prefix} 🩺 rolling cache健全性チェックを開始")

        # メタファイル確認
        meta_exists = self.rolling_meta_path.exists()
        meta_content = {}
        if meta_exists:
            try:
                with open(self.rolling_meta_path, encoding="utf-8") as f:
                    meta_content = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"{self._ui_prefix} メタファイル読み込み失敗: {e}")

        # rolling cacheファイル数
        rolling_files = [
            p for p in self.rolling_dir.glob("*.*") if not p.name.startswith("_")
        ]
        rolling_files_count = len(rolling_files)

        # 目標データ長
        target_length = self._rolling_target_len

        # SPYアンカーの状態確認
        spy_data = self.read("SPY", "rolling")
        anchor_status = {
            "exists": spy_data is not None and not spy_data.empty,
            "rows": len(spy_data) if spy_data is not None else 0,
            "meets_target": False,
        }
        if spy_data is not None:
            anchor_status["meets_target"] = len(spy_data) >= target_length

        result = {
            "meta_exists": meta_exists,
            "meta_content": meta_content,
            "rolling_files_count": rolling_files_count,
            "target_length": target_length,
            "anchor_symbol_status": anchor_status,
        }

        logger.info(f"{self._ui_prefix} ✅ 健全性チェック完了:")
        logger.info(
            f"{self._ui_prefix}   - メタファイル: {'存在' if meta_exists else '不在'}"
        )
        logger.info(f"{self._ui_prefix}   - rolling files: {rolling_files_count}個")
        spy_status = "正常" if anchor_status["meets_target"] else "要確認"
        logger.info(
            f"{self._ui_prefix}   - SPY状態: {spy_status} ({anchor_status['rows']}行)"
        )

        return result


def _base_dir() -> Path:
    settings = get_settings(create_dirs=True)
    base = Path(settings.DATA_CACHE_DIR) / BASE_SUBDIR
    base.mkdir(parents=True, exist_ok=True)
    return base


def compute_base_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCVのDataFrameに共通ベース指標を付加して返す。"""
    if df is None or df.empty:
        return df

    x = df.copy()

    # Normalize column names
    rename_map = {c: c.lower() for c in x.columns}
    x = x.rename(columns=rename_map)

    # Ensure 'Date' column and set as index
    if "date" in x.columns:
        x = x.rename(columns={"date": "Date"})
    if "Date" in x.columns:
        x["Date"] = pd.to_datetime(x["Date"], errors="coerce")
        x = x.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

    # Standardize OHLCV column names
    ohlcv_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "adjusted_close": "Close",
        "adj_close": "Close",
        "adjclose": "Close",
        "close": "Close",
        "volume": "Volume",
        "vol": "Volume",
    }
    final_rename = {c: ohlcv_map[c] for c in x.columns if c in ohlcv_map}
    x = x.rename(columns=final_rename)

    required = {"High", "Low", "Close"}
    if not required.issubset(x.columns):
        missing_cols = required - set(x.columns)
        logger.warning(
            f"{__name__}: 必須列欠落のためインジ計算をスキップ: missing={missing_cols}"
        )
        return x.reset_index()

    close = pd.to_numeric(x["Close"], errors="coerce")
    high = pd.to_numeric(x["High"], errors="coerce")
    low = pd.to_numeric(x["Low"], errors="coerce")
    vol = None
    if "Volume" in x.columns:
        vol = pd.to_numeric(x["Volume"], errors="coerce")

    # SMA/EMA - 大文字統一
    for n in [25, 50, 100, 150, 200]:
        x[f"SMA{n}"] = close.rolling(n).mean()
    for n in [20, 50]:
        x[f"EMA{n}"] = close.ewm(span=n, adjust=False).mean()

    # ATR - 大文字統一
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    for n in [10, 14, 20, 40, 50]:
        x[f"ATR{n}"] = tr.rolling(n).mean()

    # RSI (Wilder) - 大文字統一
    def _rsi(s: pd.Series, n: int) -> pd.Series:
        delta = s.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(alpha=1 / n, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    for n in [3, 4, 14]:
        x[f"RSI{n}"] = _rsi(close, n)

    # ROC & HV - 大文字統一
    x["ROC200"] = close.pct_change(200) * 100.0
    log_ret = (close / close.shift(1)).apply(np.log)
    std_dev = log_ret.rolling(50).std()
    x["HV50"] = std_dev * np.sqrt(252) * 100.0

    # DollarVolume - 大文字統一
    if vol is not None:
        x["DollarVolume20"] = (close * vol).rolling(20).mean()
        x["DollarVolume50"] = (close * vol).rolling(50).mean()

    return x.reset_index()


def get_indicator_column_flexible(df: pd.DataFrame, indicator: str) -> pd.Series | None:
    """指標列を大文字・小文字両対応で取得。大文字優先、小文字をフォールバック。

    Args:
        df: 対象DataFrame
        indicator: 指標名（例: "ATR10"）

    Returns:
        該当する列のSeries、存在しない場合はNone
    """
    # 大文字優先
    if indicator in df.columns:
        return df[indicator]

    # 小文字フォールバック
    lower_indicator = indicator.lower()
    if lower_indicator in df.columns:
        return df[lower_indicator]

    return None


def standardize_indicator_columns(df: pd.DataFrame) -> pd.DataFrame:
    """指標列名を大文字に標準化。既存の小文字列は削除。

    Args:
        df: 対象DataFrame

    Returns:
        標準化されたDataFrame
    """
    result = df.copy()

    # 標準化する指標のマッピング（小文字 -> 大文字）
    indicator_mapping = {
        "atr10": "ATR10",
        "atr14": "ATR14",
        "atr20": "ATR20",
        "atr40": "ATR40",
        "atr50": "ATR50",
        "sma25": "SMA25",
        "sma50": "SMA50",
        "sma100": "SMA100",
        "sma150": "SMA150",
        "sma200": "SMA200",
        "ema20": "EMA20",
        "ema50": "EMA50",
        "rsi3": "RSI3",
        "rsi4": "RSI4",
        "rsi14": "RSI14",
        "roc200": "ROC200",
        "hv50": "HV50",
        "dollarvolume20": "DollarVolume20",
        "dollarvolume50": "DollarVolume50",
        "avgvolume50": "AvgVolume50",
        "adx7": "ADX7",
    }

    # 小文字 -> 大文字への変換と重複削除
    for old_name, new_name in indicator_mapping.items():
        if old_name in result.columns and new_name not in result.columns:
            # 小文字列を大文字にリネーム
            result = result.rename(columns={old_name: new_name})
        elif old_name in result.columns and new_name in result.columns:
            # 両方存在する場合は小文字列を削除
            result = result.drop(columns=[old_name])

    return result


def base_cache_path(symbol: str) -> Path:
    return _base_dir() / f"{safe_filename(symbol)}.csv"


def save_base_cache(
    symbol: str, df: pd.DataFrame, settings: Settings | None = None
) -> Path:
    """Saves the base cache DataFrame to a CSV file."""
    path = base_cache_path(symbol)
    df_reset = df.reset_index() if df.index.name is not None else df
    path.parent.mkdir(parents=True, exist_ok=True)

    # Backwards-compatible: if settings is not provided, obtain global settings
    if settings is None:
        settings = get_settings(create_dirs=True)

    round_dec = getattr(getattr(settings, "cache", None), "round_decimals", None)
    df_to_write = round_dataframe(df_reset, round_dec)

    _write_dataframe_to_csv(df_to_write, path, settings)
    return path


_DEFAULT_CACHE_MANAGER: CacheManager | None = None


def _get_default_cache_manager() -> CacheManager:
    """Returns a module-level singleton CacheManager instance."""
    global _DEFAULT_CACHE_MANAGER
    if _DEFAULT_CACHE_MANAGER is None:
        settings = get_settings(create_dirs=False)
        _DEFAULT_CACHE_MANAGER = CacheManager(settings)
    return _DEFAULT_CACHE_MANAGER


def _read_legacy_cache(symbol: str) -> pd.DataFrame | None:
    """Reads from the old cache location for backward compatibility."""
    legacy_path = Path("data_cache") / f"{safe_filename(symbol)}.csv"
    if not legacy_path.exists():
        return None
    try:
        return pd.read_csv(legacy_path)
    except Exception:
        return None


def load_base_cache(
    symbol: str,
    *,
    rebuild_if_missing: bool = True,
    cache_manager: CacheManager | None = None,
    min_last_date: pd.Timestamp | None = None,
    allowed_recent_dates: Iterable[object] | None = None,
    prefer_precomputed_indicators: bool = True,
) -> pd.DataFrame | None:
    """Loads base cache, with optional freshness validation and rebuilding."""
    cm = cache_manager or _get_default_cache_manager()
    path = base_cache_path(symbol)
    df: pd.DataFrame | None = None

    if path.exists():
        try:
            df = pd.read_csv(path, parse_dates=["Date"])
            df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
        except Exception:
            df = None

    if df is not None:
        last_date = df.index[-1] if not df.empty else None
        is_stale = False
        if allowed_recent_dates and last_date:
            allowed_set = {
                pd.Timestamp(cast(Any, d)).normalize()
                for d in allowed_recent_dates
                if d is not None and not pd.isna(d)
            }
            if last_date.normalize() not in allowed_set:
                is_stale = True
        if not is_stale and min_last_date and last_date:
            if last_date.normalize() < pd.Timestamp(min_last_date).normalize():
                is_stale = True

        if not is_stale:
            return df.reset_index()

        if not rebuild_if_missing:
            return df.reset_index()

        msg = f"{__name__}: base cache for {symbol} is stale, rebuilding."
        logger.info(msg)
        df = None  # Force rebuild

    if df is None and rebuild_if_missing:
        raw = (
            cm.read(symbol, "full")
            or cm.read(symbol, "rolling")
            or _read_legacy_cache(symbol)
        )
        if raw is not None and not raw.empty:
            # If caller prefers to reuse precomputed indicator columns and
            # the raw frame appears to contain indicator columns, avoid
            # calling expensive `compute_base_indicators` and save the raw
            # data as the base cache directly.
            try:
                lc_cols = {c.lower() for c in raw.columns}
            except Exception:
                lc_cols = set()

            has_any_indicator = any((col in lc_cols) for col in MAIN_INDICATOR_COLUMNS)
            if prefer_precomputed_indicators and has_any_indicator:
                out = raw.copy()
                # Normalize date column name to 'Date' for base cache consistency
                if "date" in out.columns and "Date" not in out.columns:
                    out = out.rename(columns={"date": "Date"})
                # If index is a DatetimeIndex and there's no Date column, expose it
                if "Date" not in out.columns and isinstance(
                    out.index, pd.DatetimeIndex
                ):
                    try:
                        out = out.reset_index()
                        # ensure the date column is named 'Date'
                        if out.columns[0].lower() == "index":
                            out = out.rename(columns={out.columns[0]: "Date"})
                    except Exception:
                        pass
                try:
                    save_base_cache(symbol, out, cm.settings)
                except Exception:
                    # fallback to computing indicators if saving fails
                    out = compute_base_indicators(raw)
                    save_base_cache(symbol, out, cm.settings)
                return out

            # otherwise compute indicators normally
            out = compute_base_indicators(raw)
            save_base_cache(symbol, out, cm.settings)
            return out

    return df.reset_index() if df is not None else None
