from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

from common.cache_format import safe_filename
from common.cache_io import CacheFileManager
from common.cache_validation import perform_cache_health_check
from common.cache_warnings import report_rolling_issue
from config.settings import Settings, get_settings
from indicators_common import add_indicators

logger = logging.getLogger(__name__)

BASE_SUBDIR = "base"


class CacheManager:
    """
    二層キャッシュ管理（full / rolling）。
    リファクタリング済み：入出力・検証・警告は専用モジュールに委譲。
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.full_dir = Path(settings.cache.full_dir)
        self.rolling_dir = Path(settings.cache.rolling_dir)
        self.rolling_cfg = settings.cache.rolling
        self.rolling_meta_path = self.rolling_dir / self.rolling_cfg.meta_file

        # ディレクトリ作成
        self.full_dir.mkdir(parents=True, exist_ok=True)
        self.rolling_dir.mkdir(parents=True, exist_ok=True)

        # 入出力管理
        self.file_manager = CacheFileManager(settings)

    def _read_base_and_tail(self, ticker: str, tail_rows: int = 330) -> pd.DataFrame | None:
        """baseキャッシュを読み込み、rolling相当の行数でtail処理を行う"""
        try:
            # baseディレクトリから読み込み
            base_dir = self.full_dir.parent / "base"
            path = self.file_manager.detect_path(base_dir, ticker)

            if not path.exists():
                return None

            df = self.file_manager.read_with_fallback(path, ticker, "base")
            if df is None or df.empty:
                return None

            # tail処理でrolling相当のサイズに
            return df.tail(tail_rows)

        except Exception as e:
            logger.warning(f"Failed to read base and tail for {ticker}: {e}")
            return None

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
        base_renamed = base.rename(columns={k: v for k, v in case_map.items() if k in base.columns})
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
            enriched["date"] = pd.to_datetime(enriched.get("date", base["date"]), errors="coerce")

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

    def read(self, ticker: str, profile: str) -> pd.DataFrame | None:
        """指定プロファイルからデータを読み込む。"""
        if profile == "rolling":
            # rolling優先、フォールバック処理
            path = self.file_manager.detect_path(self.rolling_dir, ticker)
            df = self.file_manager.read_with_fallback(path, ticker, profile)

            if df is None or df.empty:
                # baseからrolling相当を生成
                report_rolling_issue("missing_rolling", ticker, "fallback to base+tail")
                df = self._read_base_and_tail(ticker)

                if df is not None and not df.empty:
                    # rolling形式で保存
                    try:
                        self.file_manager.write_atomic(df, path, ticker, profile)
                        logger.debug(f"Generated rolling cache for {ticker}")
                    except Exception as e:
                        logger.warning(f"Failed to save generated rolling for {ticker}: {e}")

                return df

            # データサイズチェック
            if len(df) < self.rolling_cfg.base_lookback_days:
                report_rolling_issue(
                    "insufficient_data",
                    ticker,
                    f"rows={len(df)}, expected>={self.rolling_cfg.base_lookback_days}",
                )

            return df

        elif profile == "full":
            path = self.file_manager.detect_path(self.full_dir, ticker)
            return self.file_manager.read_with_fallback(path, ticker, profile)

        else:
            raise ValueError(f"Unsupported profile: {profile}")

    def write_atomic(self, df: pd.DataFrame, ticker: str, profile: str) -> None:
        """指定プロファイルにデータをアトミック書き込み。"""
        if profile == "rolling":
            dir_path = self.rolling_dir
        elif profile == "full":
            dir_path = self.full_dir
        else:
            raise ValueError(f"Unsupported profile: {profile}")

        path = self.file_manager.detect_path(dir_path, ticker)

        # 健全性チェック
        perform_cache_health_check(df, ticker, profile)

        # メモリ最適化
        optimized_df = self.file_manager.optimize_dataframe_memory(df)

        # 書き込み
        self.file_manager.write_atomic(optimized_df, path, ticker, profile)

    def upsert_both(self, ticker: str, new_rows: pd.DataFrame) -> None:
        """full と rolling 両方に upsert（更新・挿入）処理を実行する。"""
        self._upsert_one(ticker, new_rows, "full")
        self._upsert_one(ticker, new_rows, "rolling")

    def _upsert_one(self, ticker: str, new_rows: pd.DataFrame, profile: str) -> None:
        """単一プロファイルに対する upsert 処理。"""
        if new_rows is None or new_rows.empty:
            return

        # 既存データ読み込み
        existing = self.read(ticker, profile)

        if existing is None or existing.empty:
            # 新規作成
            to_save = new_rows.copy()
        else:
            # マージ処理
            combined = pd.concat([existing, new_rows], ignore_index=True)
            if "date" in combined.columns:
                combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
                combined = combined.dropna(subset=["date"])
                combined = combined.drop_duplicates(subset=["date"], keep="last")
                combined = combined.sort_values("date").reset_index(drop=True)
            to_save = combined

        # rolling制限適用
        if profile == "rolling":
            to_save = self._enforce_rolling_window(to_save)

        # 指標再計算
        to_save = self._recompute_indicators(to_save)

        # 保存
        self.write_atomic(to_save, ticker, profile)

    @property
    def _ui_prefix(self) -> str:
        return "[CacheManager]"

    def _enforce_rolling_window(self, df: pd.DataFrame) -> pd.DataFrame:
        """rolling ウィンドウサイズ制限を適用する。"""
        if df is None or df.empty:
            return df
        max_rows = self.rolling_cfg.base_lookback_days + self.rolling_cfg.buffer_days
        return df.tail(max_rows)

    def prune_rolling_if_needed(self, anchor_ticker: str = "SPY") -> dict:
        """Rolling cache の容量管理とプルーニングを実行。"""
        try:
            # アンカー銘柄の最新日付を取得
            anchor_df = self.read(anchor_ticker, "rolling")
            if anchor_df is None or anchor_df.empty or "date" not in anchor_df.columns:
                return {
                    "status": "error",
                    "message": f"アンカー銘柄 {anchor_ticker} のデータが取得できません",
                }

            anchor_df["date"] = pd.to_datetime(anchor_df["date"], errors="coerce")
            anchor_latest = anchor_df["date"].max()
            if pd.isna(anchor_latest):
                return {
                    "status": "error",
                    "message": f"アンカー銘柄 {anchor_ticker} の日付が不正です",
                }

            # Rolling ディレクトリのファイル一覧
            rolling_files = list(self.rolling_dir.glob("*.csv")) + list(
                self.rolling_dir.glob("*.feather")
            )
            if not rolling_files:
                return {
                    "status": "success",
                    "message": "プルーニング対象ファイルなし",
                    "pruned": 0,
                }

            pruned_count = 0
            staleness_threshold = self.rolling_cfg.max_staleness_days

            for file_path in rolling_files:
                ticker_name = file_path.stem
                try:
                    df = self.file_manager.read_with_fallback(file_path, ticker_name, "rolling")
                    if df is None or df.empty or "date" not in df.columns:
                        continue

                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    file_latest = df["date"].max()

                    if pd.isna(file_latest):
                        continue

                    days_stale = (anchor_latest - file_latest).days
                    if days_stale > staleness_threshold:
                        file_path.unlink()
                        pruned_count += 1
                        logger.info(
                            f"Pruned stale rolling cache: {ticker_name} ({days_stale} days stale)"
                        )

                except Exception as e:
                    logger.warning(f"プルーニング処理中エラー {ticker_name}: {e}")
                    continue

            return {
                "status": "success",
                "message": f"プルーニング完了: {pruned_count} ファイル削除",
                "pruned": pruned_count,
                "anchor_date": anchor_latest.strftime("%Y-%m-%d"),
            }

        except Exception as e:
            return {"status": "error", "message": f"プルーニング処理中にエラー: {e}"}

    def analyze_rolling_gaps(self, system_symbols: list[str] | None = None) -> dict:
        """Rolling cache のギャップ分析を実行。"""
        try:
            if system_symbols is None:
                # デフォルトシンボル取得
                try:
                    from common.symbols_manifest import load_symbol_manifest

                    manifest = load_symbol_manifest()
                    system_symbols = list(manifest.get("symbols", {}).keys())
                except Exception:
                    system_symbols = []

            if not system_symbols:
                return {
                    "status": "error",
                    "message": "分析対象シンボルが見つかりません",
                }

            missing_files = []
            insufficient_data = []
            stale_data = []
            healthy_count = 0

            # SPY を基準日付として使用
            spy_df = self.read("SPY", "rolling")
            if spy_df is not None and not spy_df.empty and "date" in spy_df.columns:
                spy_df["date"] = pd.to_datetime(spy_df["date"], errors="coerce")
                reference_date = spy_df["date"].max()
            else:
                reference_date = pd.Timestamp.now().normalize()

            min_required_rows = self.rolling_cfg.base_lookback_days
            max_stale_days = self.rolling_cfg.max_staleness_days

            for symbol in system_symbols:
                try:
                    df = self.read(symbol, "rolling")
                    if df is None or df.empty:
                        missing_files.append(symbol)
                        continue

                    if "date" not in df.columns:
                        insufficient_data.append(f"{symbol}(no_date_col)")
                        continue

                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    valid_dates = df["date"].dropna()

                    if len(valid_dates) == 0:
                        insufficient_data.append(f"{symbol}(no_valid_dates)")
                        continue

                    if len(df) < min_required_rows:
                        insufficient_data.append(f"{symbol}(rows={len(df)})")
                        continue

                    latest_date = valid_dates.max()
                    if pd.notna(latest_date) and pd.notna(reference_date):
                        days_behind = (reference_date - latest_date).days
                        if days_behind > max_stale_days:
                            stale_data.append(f"{symbol}({days_behind}d)")
                            continue

                    healthy_count += 1

                except Exception as e:
                    logger.warning(f"ギャップ分析エラー {symbol}: {e}")
                    missing_files.append(f"{symbol}(error)")

            return {
                "status": "success",
                "total_symbols": len(system_symbols),
                "healthy": healthy_count,
                "missing_files": len(missing_files),
                "insufficient_data": len(insufficient_data),
                "stale_data": len(stale_data),
                "missing_list": missing_files[:10],  # 最初の10件のみ
                "insufficient_list": insufficient_data[:10],
                "stale_list": stale_data[:10],
                "reference_date": (
                    reference_date.strftime("%Y-%m-%d") if pd.notna(reference_date) else "N/A"
                ),
            }

        except Exception as e:
            return {"status": "error", "message": f"ギャップ分析中にエラー: {e}"}

    def get_rolling_health_summary(self) -> dict:
        """Rolling cache の健康状態サマリーを取得。"""
        try:
            rolling_files = list(self.rolling_dir.glob("*.csv")) + list(
                self.rolling_dir.glob("*.feather")
            )

            if not rolling_files:
                return {
                    "status": "success",
                    "total_files": 0,
                    "message": "Rolling cache ファイルが存在しません",
                }

            total_files = len(rolling_files)
            readable_files = 0
            total_rows = 0
            date_range_info = {}

            for file_path in rolling_files[:20]:  # サンプリング
                try:
                    ticker = file_path.stem
                    df = self.file_manager.read_with_fallback(file_path, ticker, "rolling")
                    if df is not None and not df.empty:
                        readable_files += 1
                        total_rows += len(df)

                        if "date" in df.columns:
                            df["date"] = pd.to_datetime(df["date"], errors="coerce")
                            valid_dates = df["date"].dropna()
                            if len(valid_dates) > 0:
                                date_range_info[ticker] = {
                                    "start": valid_dates.min().strftime("%Y-%m-%d"),
                                    "end": valid_dates.max().strftime("%Y-%m-%d"),
                                    "rows": len(df),
                                }
                except Exception:
                    continue

            return {
                "status": "success",
                "total_files": total_files,
                "readable_files": readable_files,
                "sample_total_rows": total_rows,
                "avg_rows_per_file": (total_rows / readable_files if readable_files > 0 else 0),
                "sample_date_ranges": date_range_info,
            }

        except Exception as e:
            return {"status": "error", "message": f"健康状態チェック中にエラー: {e}"}

    def read_batch_parallel(
        self,
        symbols: list[str],
        profile: str = "rolling",
        max_workers: int | None = None,
        fallback_profile: str | None = "full",
        progress_callback=None,
    ) -> dict[str, pd.DataFrame]:
        """複数銘柄のデータを並列で読み込む。"""
        if not symbols:
            return {}

        if max_workers is None:
            max_workers = min(8, len(symbols))

        results = {}
        completed = 0

        def read_single(symbol: str) -> tuple[str, pd.DataFrame | None]:
            df = self.read(symbol, profile)
            if df is None and fallback_profile:
                df = self.read(symbol, fallback_profile)
            return symbol, df

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {executor.submit(read_single, sym): sym for sym in symbols}

            for future in as_completed(future_to_symbol):
                symbol, df = future.result()
                if df is not None:
                    results[symbol] = df

                completed += 1
                if progress_callback and completed % 50 == 0:
                    progress_callback(completed, len(symbols))

        return results

    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrameのメモリ使用量を最適化する（委譲）。"""
        return self.file_manager.optimize_dataframe_memory(df)

    def remove_unnecessary_columns(
        self, df: pd.DataFrame, keep_columns: list[str] | None = None
    ) -> pd.DataFrame:
        """不要な列を除去する（委譲）。"""
        return self.file_manager.remove_unnecessary_columns(df, keep_columns)


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
        logger.warning(f"{__name__}: 必須列欠落のためインジ計算をスキップ: missing={missing_cols}")
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
    """大文字小文字を区別せずに指標列を取得する。"""
    if df is None or df.empty:
        return None

    # 完全一致を最初に試行
    if indicator in df.columns:
        return df[indicator]

    # 小文字変換で検索
    lower_indicator = indicator.lower()
    for col in df.columns:
        if col.lower() == lower_indicator:
            return df[col]

    return None


def standardize_indicator_columns(df: pd.DataFrame) -> pd.DataFrame:
    """指標列名を標準形式に統一する。"""
    if df is None or df.empty:
        return df

    result = df.copy()

    # 標準化マップ（小文字 -> 標準形式）
    standard_map = {
        "sma25": "SMA25",
        "sma50": "SMA50",
        "sma100": "SMA100",
        "sma150": "SMA150",
        "sma200": "SMA200",
        "ema20": "EMA20",
        "ema50": "EMA50",
        "atr10": "ATR10",
        "atr14": "ATR14",
        "atr20": "ATR20",
        "atr40": "ATR40",
        "atr50": "ATR50",
        "rsi3": "RSI3",
        "rsi4": "RSI4",
        "rsi14": "RSI14",
        "adx7": "ADX7",
        "roc200": "ROC200",
        "hv50": "HV50",
        "dollarvolume20": "DollarVolume20",
        "dollarvolume50": "DollarVolume50",
        "avgvolume50": "AvgVolume50",
    }

    rename_dict = {}
    for col in result.columns:
        col_lower = col.lower()
        if col_lower in standard_map:
            rename_dict[col] = standard_map[col_lower]

    if rename_dict:
        result = result.rename(columns=rename_dict)

    return result


def base_cache_path(symbol: str) -> Path:
    return _base_dir() / f"{safe_filename(symbol)}.csv"


def save_base_cache(symbol: str, df: pd.DataFrame, settings: Settings | None = None) -> Path:
    """Base キャッシュを保存し、パスを返す。"""
    path = base_cache_path(symbol)
    if settings is None:
        settings = get_settings(create_dirs=True)

    # 新しいフォーマット処理を使用
    from common.cache_format import write_dataframe_to_csv

    df_reset = df.reset_index() if hasattr(df, "index") and df.index.name is not None else df
    df_reset = df_reset.rename(columns={c: str(c).lower() for c in df_reset.columns})
    write_dataframe_to_csv(df_reset, path, settings)
    return path


_DEFAULT_CACHE_MANAGER: CacheManager | None = None


def _get_default_cache_manager() -> CacheManager:
    global _DEFAULT_CACHE_MANAGER
    if _DEFAULT_CACHE_MANAGER is None:
        settings = get_settings(create_dirs=True)
        _DEFAULT_CACHE_MANAGER = CacheManager(settings)
    return _DEFAULT_CACHE_MANAGER


def _read_legacy_cache(symbol: str) -> pd.DataFrame | None:
    """Legacy cache から読み込む（互換性のため）。"""
    try:
        legacy_path = Path("data_cache") / f"{safe_filename(symbol)}.csv"
        if legacy_path.exists():
            return pd.read_csv(legacy_path)
    except Exception:
        pass
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
    """Base キャッシュを読み込む（下位互換性のため保持）。"""
    if cache_manager is None:
        cache_manager = _get_default_cache_manager()

    try:
        # まず base から読み込み
        base_dir = cache_manager.full_dir.parent / "base"
        if base_dir.exists():
            df = cache_manager.read(symbol, "full")  # full として読み込み
            if df is not None and not df.empty:
                return df

        # フォールバック: legacy
        return _read_legacy_cache(symbol)

    except Exception as e:
        logger.warning(f"load_base_cache failed for {symbol}: {e}")
        return None
