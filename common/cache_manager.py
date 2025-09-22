from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from collections.abc import Iterable
from typing import ClassVar

import numpy as np
import pandas as pd

from common.utils import describe_dtype, safe_filename
from indicators_common import add_indicators
from config.settings import get_settings

logger = logging.getLogger(__name__)

BASE_SUBDIR = "base"

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


class CacheManager:
    """
    二層キャッシュ管理（full / rolling）。
    - 既存のフォーマット(csv/parquet)は自動検出・踏襲
    - system5/6スタイルのコメント・進捗ログ粒度を踏襲
    """

    _GLOBAL_WARNED: ClassVar[set[tuple[str, str, str]]] = set()

    def __init__(self, settings):
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

        if df is None or df.empty:
            return df
        try:
            work = df.copy()
        except Exception:
            work = pd.DataFrame(df)
        work.columns = [str(c).lower() for c in work.columns]
        if "date" not in work.columns:
            return df
        required = {"open", "high", "low", "close"}
        if not required.issubset(set(work.columns)):
            return df
        base = work.copy()
        base["date"] = pd.to_datetime(base["date"], errors="coerce")
        base = base.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        if base.empty:
            return df
        for col in ("open", "high", "low", "close", "volume"):
            if col in base.columns:
                base[col] = pd.to_numeric(base[col], errors="coerce")
        base["Date"] = base["date"].dt.normalize()
        case_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
        for src, dst in case_map.items():
            if src in base.columns:
                base[dst] = base[src]
        try:
            enriched = add_indicators(base)
        except Exception:
            return df
        enriched = enriched.drop(columns=["Date"], errors="ignore")
        enriched.columns = [str(c).lower() for c in enriched.columns]
        enriched["date"] = pd.to_datetime(
            enriched.get("date", base["date"]), errors="coerce"
        )
        combined = work.copy()
        for col, series in enriched.items():
            combined[col] = series
        combined = combined.loc[:, ~pd.Index(combined.columns).duplicated(keep="first")]
        original_cols = [str(c).lower() for c in df.columns]
        new_cols = [col for col in combined.columns if col not in original_cols]
        ordered_cols = original_cols + new_cols
        try:
            combined = combined.reindex(columns=ordered_cols)
        except Exception:
            combined = combined.reindex(columns=sorted(set(combined.columns)))
        return combined

    # ---------- path/format detection ----------
    def _detect_path(self, base_dir: Path, ticker: str) -> Path:
        csv_path = base_dir / f"{ticker}.csv"
        pq_path = base_dir / f"{ticker}.parquet"
        feather_path = base_dir / f"{ticker}.feather"
        for path in (csv_path, pq_path, feather_path):
            if path.exists():
                return path
        fmt = (self.file_format or "auto").lower()
        if fmt == "parquet":
            return pq_path
        if fmt == "feather":
            return feather_path
        return csv_path

    # ---------- IO ----------
    def read(self, ticker: str, profile: str) -> pd.DataFrame | None:
        base = self.full_dir if profile == "full" else self.rolling_dir
        path = self._detect_path(base, ticker)
        if not path.exists():
            return None
        try:
            if path.suffix == ".feather":
                try:
                    df = pd.read_feather(path)
                except Exception as e:
                    self._warn_once(
                        ticker,
                        profile,
                        "read_feather_fail",
                        (
                            f"{self._ui_prefix} feather読込失敗: {path.name} ({e}) "
                            "→ csvへフォールバック試行"
                        ),
                    )
                    csv_path = path.with_suffix(".csv")
                    if csv_path.exists():
                        try:
                            df = pd.read_csv(csv_path, parse_dates=["date"])
                        except ValueError as e2:
                            if (
                                "Missing column provided to 'parse_dates': 'date'"
                                in str(e2)
                            ):
                                df = pd.read_csv(csv_path)
                                if "Date" in df.columns:
                                    df = df.rename(columns={"Date": "date"})
                                    df["date"] = pd.to_datetime(df["date"])
                                else:
                                    raise
                            else:
                                raise
                        except Exception as e2:
                            self._warn_once(
                                ticker,
                                profile,
                                "read_csv_fail",
                                f"{self._ui_prefix} csv読込も失敗: {csv_path.name} ({e2})",
                            )
                            return None
                    else:
                        return None
            elif path.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                try:
                    df = pd.read_csv(path, parse_dates=["date"])
                except ValueError as e:
                    if "Missing column provided to 'parse_dates': 'date'" in str(e):
                        df = pd.read_csv(path)
                        if "Date" in df.columns:
                            df = df.rename(columns={"Date": "date"})
                            df["date"] = pd.to_datetime(df["date"])
                        else:
                            raise
                    else:
                        raise
        except Exception as e:  # pragma: no cover - log and continue
            category = f"read_error:{path.name}:{type(e).__name__}:{str(e)}"
            self._warn_once(
                ticker,
                profile,
                category,
                f"{self._ui_prefix} 読み込み失敗: {path.name} ({e})",
            )
            return None
        # 正規化: 列名を小文字化
        df.columns = [c.lower() for c in df.columns]
        # 列名の重複を除去（例: CSVに 'date' と 'Date' が混在していた場合）
        try:
            cols = pd.Index(df.columns)
            if len(cols) != len(cols.unique()):
                df = df.loc[:, ~cols.duplicated(keep="first")]
        except Exception:
            pass
        if "date" in df.columns:
            try:
                # 型が混在（str/datetime）しても確実に datetime 化
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            except Exception:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = (
                df.dropna(subset=["date"])  # 不正日付を除外
                .sort_values("date")
                .drop_duplicates("date")
                .reset_index(drop=True)
            )
        # --- 健全性チェック: NaN・型不一致・異常値 ---
        try:
            nan_rate = 0.0
            if df.size > 0:
                try:
                    base_days = int(self.rolling_cfg.base_lookback_days)
                except Exception:
                    base_days = 300
                try:
                    buffer_days = int(self.rolling_cfg.buffer_days)
                except Exception:
                    buffer_days = 30
                window = max(1, base_days + buffer_days)
                if profile == "rolling":
                    recent_df = df.tail(window)
                else:
                    recent_df = df.tail(max(window, 252))
                target_cols = [
                    col for col in MAIN_INDICATOR_COLUMNS if col in recent_df.columns
                ]
                if not target_cols:
                    target_cols = list(recent_df.columns)
                col_rates: list[float] = []
                for col in target_cols:
                    series_like = recent_df[col]
                    if not isinstance(series_like, pd.Series):
                        series_like = pd.Series(series_like)
                    if series_like.dropna().empty:
                        col_rates.append(1.0)
                        continue
                    first_valid = series_like.first_valid_index()
                    if first_valid is None:
                        col_rates.append(1.0)
                        continue
                    try:
                        trimmed = series_like.loc[first_valid:]
                    except Exception:
                        try:
                            loc = series_like.index.get_loc(first_valid)
                        except Exception:
                            loc = 0
                        trimmed = series_like.iloc[loc:]
                    if trimmed.empty:
                        col_rates.append(1.0)
                        continue
                    col_rates.append(float(trimmed.isna().mean()))
                if col_rates:
                    if any(rate >= 1.0 for rate in col_rates):
                        nan_rate = 1.0
                    else:
                        nan_rate = float(np.mean(col_rates))
            if nan_rate > 0.20:
                category = f"nan_rate:{round(float(nan_rate), 4)}"
                self._warn_once(
                    ticker,
                    profile,
                    category,
                    f"{self._ui_prefix} ⚠️ {ticker} {profile} cache: 主要指標NaN率高 "
                    f"({nan_rate:.2%})",
                )
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    series_like = df[col]
                    if not pd.api.types.is_numeric_dtype(series_like):
                        dtype_repr = describe_dtype(series_like)
                        category = f"dtype:{col}:{dtype_repr}"
                        self._warn_once(
                            ticker,
                            profile,
                            category,
                            f"{self._ui_prefix} ⚠️ {ticker} {profile} cache: {col}型不一致 "
                            f"({dtype_repr})",
                        )
            for col in ["close", "high", "low"]:
                if col in df.columns:
                    vals = pd.to_numeric(df[col], errors="coerce")
                    if (vals <= 0).all():
                        category = f"non_positive:{col}"
                        self._warn_once(
                            ticker,
                            profile,
                            category,
                            (
                                f"{self._ui_prefix} ⚠️ {ticker} {profile} cache: "
                                f"{col}全て非正値"
                            ),
                        )
        except Exception as e:
            category = f"healthcheck_error:{type(e).__name__}:{str(e)}"
            self._warn_once(
                ticker,
                profile,
                category,
                f"{self._ui_prefix} ⚠️ {ticker} {profile} cache: 健全性チェック失敗 "
                f"({e})",
            )
        return df

    def write_atomic(self, df: pd.DataFrame, ticker: str, profile: str) -> None:
        base = self.full_dir if profile == "full" else self.rolling_dir
        base.mkdir(parents=True, exist_ok=True)
        path = self._detect_path(base, ticker)
        tmp = path.with_suffix(path.suffix + ".tmp")
        # 丸め桁数の判定: profile が 'rolling' の場合は rolling 設定を優先し、なければ全体設定を使う
        try:
            round_dec = None
            cfg_round = getattr(self.settings.cache, "round_decimals", None)
            if profile == "rolling":
                roll_round = getattr(
                    self.settings.cache.rolling, "round_decimals", None
                )
                round_dec = roll_round if roll_round is not None else cfg_round
            else:
                round_dec = cfg_round
        except Exception:
            round_dec = None
        df_to_write = df
        if round_dec is not None:
            try:
                df_to_write = df.copy()
                # pandas.DataFrame.round は数値列のみを丸める
                df_to_write = df_to_write.round(int(round_dec))
            except Exception:
                df_to_write = df
        try:
            if path.suffix == ".parquet":
                df_to_write.to_parquet(tmp, index=False)
            elif path.suffix == ".feather":
                df_to_write.reset_index(drop=True).to_feather(tmp)
            else:
                df_to_write.to_csv(tmp, index=False)
            shutil.move(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass

    # ---------- upsert ----------
    def upsert_both(self, ticker: str, new_rows: pd.DataFrame) -> None:
        """EODHD更新を①full②rollingに同時反映"""
        for profile in ("full", "rolling"):
            self._upsert_one(ticker, new_rows, profile)

    def _upsert_one(self, ticker: str, new_rows: pd.DataFrame, profile: str) -> None:
        # 入力行の日付を厳密に正規化（str/Timestamp 混在対策）
        if new_rows is not None and not new_rows.empty:
            try:
                if "date" in new_rows.columns:
                    new_rows = new_rows.copy()
                    new_rows["date"] = pd.to_datetime(new_rows["date"], errors="coerce")
                    new_rows = new_rows.dropna(subset=["date"]).reset_index(drop=True)
            except Exception:
                pass
        cur = self.read(ticker, profile)
        if cur is None or cur.empty:
            merged = new_rows.copy()
        else:
            merged = pd.concat([cur, new_rows], ignore_index=True)
            merged = (
                merged.sort_values("date")
                .drop_duplicates("date")
                .reset_index(drop=True)  # noqa: E501
            )

        if profile == "rolling":
            merged = self._enforce_rolling_window(merged)

        merged = self._recompute_indicators(merged)
        self.write_atomic(merged, ticker, profile)

    # ---------- rolling window & prune ----------
    @property
    def _rolling_target_len(self) -> int:
        return int(self.rolling_cfg.base_lookback_days + self.rolling_cfg.buffer_days)

    @property
    def _prune_chunk(self) -> int:
        return int(self.rolling_cfg.prune_chunk_days)

    def _enforce_rolling_window(self, df: pd.DataFrame) -> pd.DataFrame:
        if "date" not in df.columns or df.empty:
            return df
        target = self._rolling_target_len
        if len(df) <= target:
            return df
        return df.iloc[-target:].reset_index(drop=True)

    def prune_rolling_if_needed(self, anchor_ticker: str = "SPY") -> dict:
        last_meta = {"anchor_rows_at_prune": 0}
        if self.rolling_meta_path.exists():
            try:
                last_meta = json.loads(
                    self.rolling_meta_path.read_text(encoding="utf-8")
                )  # noqa: E501
            except Exception:
                pass

        anchor_df = self.read(anchor_ticker, "rolling")
        if anchor_df is None or anchor_df.empty:
            logger.info(f"{self._ui_prefix} rolling未整備のためpruneスキップ")
            return {"pruned_files": 0, "dropped_rows_total": 0}

        cur_rows = len(anchor_df)
        prev_rows = int(last_meta.get("anchor_rows_at_prune", 0))
        progressed = max(0, cur_rows - prev_rows)

        if progressed < self._prune_chunk:
            logger.info(
                f"{self._ui_prefix} 進捗{progressed}営業日 (<{self._prune_chunk}) のためprune不要"
            )
            return {"pruned_files": 0, "dropped_rows_total": 0}

        logger.info(
            f"{self._ui_prefix} ⏳ prune開始: anchor={anchor_ticker}, 進捗={progressed}営業日"
        )

        pruned_files = 0
        dropped_total = 0
        for path in self.rolling_dir.glob("*.*"):
            if path.name.startswith("_"):
                continue
            ticker = path.stem
            df = self.read(ticker, "rolling")
            if df is None or df.empty:
                continue

            keep_min = self._rolling_target_len
            can_drop = max(0, len(df) - keep_min)
            drop_n = min(self._prune_chunk, can_drop)
            if drop_n <= 0:
                continue

            new_df = df.iloc[drop_n:].reset_index(drop=True)
            self.write_atomic(new_df, ticker, "rolling")

            pruned_files += 1
            dropped_total += drop_n

        self.rolling_meta_path.write_text(
            json.dumps(
                {"anchor_rows_at_prune": cur_rows},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info(
            f"{self._ui_prefix} ✅ prune完了: files={pruned_files},"
            f" dropped_rows={dropped_total}"
        )
        return {"pruned_files": pruned_files, "dropped_rows_total": dropped_total}


def _base_dir() -> Path:
    settings = get_settings(create_dirs=True)
    base = Path(settings.DATA_CACHE_DIR) / BASE_SUBDIR
    base.mkdir(parents=True, exist_ok=True)
    return base


def compute_base_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCVのDataFrameに共通ベース指標を付加して返す。
    必要列: Open, High, Low, Close, Volume
    出力列:
      SMA25/100/150/200, EMA20/50, ATR10/14/40/50, RSI3/14, ROC200, HV50
    """
    if df is None or df.empty:
        return df
    x = df.copy()

    # 'Date' / 'date' が両方存在するなど、日付列重複を事前に整理
    try:
        date_like = [c for c in x.columns if str(c).lower() == "date"]
        if len(date_like) >= 2:
            keep = "Date" if "Date" in date_like else date_like[0]
            drop_cols = [c for c in date_like if c != keep]
            x = x.drop(columns=drop_cols, errors="ignore")
    except Exception:
        pass

    # 列名の正規化（大小・同義語を統一）
    lower_map = {c.lower(): c for c in x.columns}
    rename_map: dict[str, str] = {}
    if "date" in lower_map and "Date" not in x.columns:
        rename_map[lower_map["date"]] = "Date"
    # Close は adjusted を優先
    for key in ("adjusted_close", "adj_close", "adjclose", "close"):
        if key in lower_map:
            rename_map.setdefault(lower_map[key], "Close")
            break
    # その他の標準OHLCV
    mapping = {
        "Open": ("open",),
        "High": ("high",),
        "Low": ("low",),
        "Volume": ("volume", "vol"),
    }
    for canon, candidates in mapping.items():
        for key in candidates:
            if key in lower_map:
                rename_map.setdefault(lower_map[key], canon)
                break
    if rename_map:
        x = x.rename(columns=rename_map)
    # 日付インデックス化（可能なら）
    if "Date" in x.columns:
        x["Date"] = pd.to_datetime(x["Date"], errors="coerce")
        x = x.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

    # 必須列チェック
    required = ["High", "Low", "Close"]
    missing = [c for c in required if c not in x.columns]
    if missing:
        logger.warning(
            f"{__name__}: 必須列欠落のためインジ計算をスキップ: missing={missing}"
        )
        return x

    close = pd.to_numeric(x["Close"], errors="coerce")
    high = pd.to_numeric(x["High"], errors="coerce")
    low = pd.to_numeric(x["Low"], errors="coerce")
    vol = x.get("Volume")
    if vol is not None:
        vol = pd.to_numeric(vol, errors="coerce")

    # SMA/EMA
    x["SMA25"] = close.rolling(25).mean()
    x["SMA50"] = close.rolling(50).mean()
    x["SMA100"] = close.rolling(100).mean()
    x["SMA150"] = close.rolling(150).mean()
    x["SMA200"] = close.rolling(200).mean()
    x["EMA20"] = close.ewm(span=20, adjust=False).mean()
    x["EMA50"] = close.ewm(span=50, adjust=False).mean()

    # True Range / ATR
    tr = pd.concat(
        [
            (high - low),
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    x["ATR10"] = tr.rolling(10).mean()
    x["ATR14"] = tr.rolling(14).mean()
    x["ATR40"] = tr.rolling(40).mean()
    x["ATR50"] = tr.rolling(50).mean()

    # RSI 3/14 (Wilder)
    def _rsi(series: pd.Series, window: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(alpha=1 / window, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    x["RSI3"] = _rsi(close, 3)
    x["RSI14"] = _rsi(close, 14)

    # ROC200 (%)
    x["ROC200"] = close.pct_change(200) * 100.0

    # HV50 (% 年率)
    # np.log は型チェッカー上で ndarray を返すと解釈されるため、Series.apply を使って Series を維持
    ret = (close / close.shift(1)).apply(np.log)
    x["HV50"] = ret.rolling(50).std() * np.sqrt(252) * 100

    # 補助: 流動性系
    if vol is not None:
        x["DollarVolume20"] = (close * vol).rolling(20).mean()
        x["DollarVolume50"] = (close * vol).rolling(50).mean()

    return x


def base_cache_path(symbol: str) -> Path:
    return _base_dir() / f"{safe_filename(symbol)}.csv"


def save_base_cache(symbol: str, df: pd.DataFrame) -> Path:
    path = base_cache_path(symbol)
    df_reset = df.reset_index() if df.index.name is not None else df
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        settings = get_settings(create_dirs=False)
        round_dec = getattr(settings.cache, "round_decimals", None)
    except Exception:
        round_dec = None
    df_to_write = df_reset
    if round_dec is not None:
        try:
            df_to_write = df_reset.round(int(round_dec))
        except Exception:
            df_to_write = df_reset
    df_to_write.to_csv(path, index=False)
    return path


_DEFAULT_CACHE_MANAGER: CacheManager | None = None


def _get_default_cache_manager() -> CacheManager:
    """モジュール共通で使い回す CacheManager を返す。"""

    global _DEFAULT_CACHE_MANAGER
    if _DEFAULT_CACHE_MANAGER is None:
        settings = get_settings(create_dirs=False)
        _DEFAULT_CACHE_MANAGER = CacheManager(settings)
    return _DEFAULT_CACHE_MANAGER


def _read_legacy_cache(symbol: str) -> pd.DataFrame | None:
    """`data_cache/`直下の旧形式CSVを直接読み込む（互換目的）。"""

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
) -> pd.DataFrame | None:  # noqa: E501
    """Load base cache with optional freshness validation.

    When ``allowed_recent_dates`` or ``min_last_date`` is provided, the on-disk cache
    is considered stale if it does not contain a recent trading day. Stale caches are
    rebuilt from the latest full/rolling dataset when ``rebuild_if_missing`` is True.
    """

    def _detect_last_date(frame: pd.DataFrame | None) -> pd.Timestamp | None:
        if frame is None or getattr(frame, "empty", True):
            return None
        try:
            if isinstance(frame.index, pd.DatetimeIndex) and len(frame.index):
                return pd.Timestamp(frame.index[-1]).normalize()
        except Exception:
            pass
        for col in ("Date", "date"):
            if col in frame.columns:
                try:
                    series = pd.to_datetime(frame[col], errors="coerce").dropna()
                    if not series.empty:
                        return pd.Timestamp(series.iloc[-1]).normalize()
                except Exception:
                    continue
        return None

    allowed_set: set[pd.Timestamp] = set()
    if allowed_recent_dates:
        for candidate in allowed_recent_dates:
            try:
                ts = pd.Timestamp(candidate)
            except Exception:
                continue
            if pd.isna(ts):
                continue
            allowed_set.add(ts.normalize())

    if min_last_date is not None:
        try:
            min_norm: pd.Timestamp | None = pd.Timestamp(min_last_date).normalize()
        except Exception:
            min_norm = None
    else:
        min_norm = None

    path = base_cache_path(symbol)
    df: pd.DataFrame | None = None
    if path.exists():
        try:
            df = pd.read_csv(path, parse_dates=["Date"])
        except ValueError as exc:
            if "Missing column provided to 'parse_dates': 'Date'" in str(exc):
                try:
                    df = pd.read_csv(path)
                except Exception:
                    df = None
                else:
                    if df is not None:
                        if "Date" in df.columns:
                            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                        elif "date" in df.columns:
                            df["Date"] = pd.to_datetime(df["date"], errors="coerce")
                            df = df.drop(columns=["date"], errors="ignore")
                        else:
                            df = None
            else:
                df = None
        except Exception:
            df = None
        if df is not None:
            try:
                df = df.dropna(subset=["Date"])
                df = df.sort_values("Date").set_index("Date")
            except Exception:
                df = None

    if df is not None:
        last_date = _detect_last_date(df)
        stale = False
        if allowed_set and (last_date is None or last_date not in allowed_set):
            stale = True
        if not stale and min_norm is not None:
            if last_date is None or last_date < min_norm:
                stale = True
        if stale:
            if rebuild_if_missing:
                try:
                    logger.info(
                        "%s base cache stale -> rebuild: %s (last=%s)",
                        __name__,
                        symbol,
                        last_date.date() if last_date is not None else "None",
                    )
                except Exception:
                    pass
                df = None
            else:
                return df
        else:
            return df

    if not rebuild_if_missing:
        return df

    cm = cache_manager or _get_default_cache_manager()
    raw = None
    try:
        raw = cm.read(symbol, "full")
        if raw is None or getattr(raw, "empty", False):
            raw = cm.read(symbol, "rolling")
    except Exception:
        raw = None

    if raw is not None and not raw.empty:
        if "Date" not in raw.columns:
            if "date" in raw.columns:
                raw = raw.rename(columns={"date": "Date"})
            else:
                raw = raw.copy()
                raw["Date"] = pd.NaT

    if (raw is None or raw.empty) and rebuild_if_missing:
        raw = _read_legacy_cache(symbol)

    if raw is None or raw.empty:
        return None
    out = compute_base_indicators(raw)
    save_base_cache(symbol, out)
    return out
