from __future__ import annotations

from collections.abc import Iterable
import json
import logging
import os
from pathlib import Path
import shutil
from typing import ClassVar, Any, cast

from indicators_common import add_indicators
import numpy as np
import pandas as pd

from common.utils import describe_dtype, safe_filename
from config.settings import get_settings, Settings

logger = logging.getLogger(__name__)

BASE_SUBDIR = "base"


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
        "6d_return",
        "return6d",
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
            "6d_return",
            "return6d",
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
    "6d_return",
    "return6d",
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
    "6d_return": 7,
    "return6d": 7,
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
            enriched.columns = [str(c).lower() for c in enriched.columns]
            enriched["date"] = pd.to_datetime(
                enriched.get("date", base["date"]), errors="coerce"
            )

            # Merge indicators back, but do NOT overwrite existing indicator columns.
            # Keep the original DataFrame's columns when present and only add
            # missing indicator columns produced by `add_indicators`.
            combined = df.copy()
            for col, series in enriched.items():
                # never replace the date column
                if col == "date":
                    continue
                # only add new columns that are not already present
                if col not in combined.columns:
                    combined[col] = series

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
                # If the entire series is NaN, warn (indicator missing entirely)
                if series.isna().all():
                    warnings.append((col, 1.0))
                    continue

                # If series is shorter than lookback, skip — indicator not applicable
                if lookback and len(series) <= lookback:
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

    # SMA/EMA
    for n in [25, 50, 100, 150, 200]:
        x[f"sma{n}"] = close.rolling(n).mean()
    for n in [20, 50]:
        x[f"ema{n}"] = close.ewm(span=n, adjust=False).mean()

    # ATR
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    for n in [10, 14, 40, 50]:
        x[f"atr{n}"] = tr.rolling(n).mean()

    # RSI (Wilder)
    def _rsi(s: pd.Series, n: int) -> pd.Series:
        delta = s.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(alpha=1 / n, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    for n in [3, 14]:
        x[f"rsi{n}"] = _rsi(close, n)

    # ROC & HV
    x["roc200"] = close.pct_change(200) * 100.0
    log_ret = (close / close.shift(1)).apply(np.log)
    std_dev = log_ret.rolling(50).std()
    x["hv50"] = std_dev * np.sqrt(252) * 100.0

    if vol is not None:
        x["dollarvolume20"] = (close * vol).rolling(20).mean()
        x["dollarvolume50"] = (close * vol).rolling(50).mean()

    return x.reset_index()


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
                if pd.notna(d)
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
