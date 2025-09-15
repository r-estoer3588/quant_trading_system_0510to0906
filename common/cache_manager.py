from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from common.utils import get_cached_data, safe_filename
from config.settings import get_settings

logger = logging.getLogger(__name__)

BASE_SUBDIR = "base"


class CacheManager:
    """
    二層キャッシュ管理（full / rolling）。
    - 既存のフォーマット(csv/parquet)は自動検出・踏襲
    - system5/6スタイルのコメント・進捗ログ粒度を踏襲
    """

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

    # ---------- path/format detection ----------
    def _detect_path(self, base_dir: Path, ticker: str) -> Path:
        csv_path = base_dir / f"{ticker}.csv"
        pq_path = base_dir / f"{ticker}.parquet"
        if csv_path.exists():
            return csv_path
        if pq_path.exists():
            return pq_path
        return csv_path if self.file_format in ("auto", "csv") else pq_path

    # ---------- IO ----------
    def read(self, ticker: str, profile: str) -> pd.DataFrame | None:
        base = self.full_dir if profile == "full" else self.rolling_dir
        path = self._detect_path(base, ticker)
        if not path.exists():
            return None
        try:
            if path.suffix == ".parquet":
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
            logger.warning(f"{self._ui_prefix} 読み込み失敗: {path.name} ({e})")
            return None
        df.columns = [c.lower() for c in df.columns]
        if "date" in df.columns:
            df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        return df

    def write_atomic(self, df: pd.DataFrame, ticker: str, profile: str) -> None:
        base = self.full_dir if profile == "full" else self.rolling_dir
        base.mkdir(parents=True, exist_ok=True)
        path = self._detect_path(base, ticker)
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            if path.suffix == ".parquet":
                df.to_parquet(tmp, index=False)
            else:
                df.to_csv(tmp, index=False)
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
    ret = np.log(close / close.shift(1))
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
    df_reset.to_csv(path, index=False)
    return path


def load_base_cache(
    symbol: str, *, rebuild_if_missing: bool = True
) -> pd.DataFrame | None:  # noqa: E501
    """data_cache/base/{symbol}.csv を優先的に読み込む。
    無ければ CacheManager の full/rolling から再構築して保存（rebuild_if_missing=True）。
    いずれも無ければ None。
    """
    path = base_cache_path(symbol)
    if path.exists():
        try:
            df = pd.read_csv(path, parse_dates=["Date"]) if path.exists() else None
            if df is not None:
                df = df.sort_values("Date").set_index("Date")
            return df
        except Exception:
            pass

    if not rebuild_if_missing:
        return None

    # CacheManager から再構築（full -> rolling）
    try:
        from common.cache_manager import CacheManager  # 遅延importの循環回避
        from config.settings import get_settings

        cm = CacheManager(get_settings(create_dirs=False))
        raw = cm.read(symbol, "full") or cm.read(symbol, "rolling")
        if raw is not None and not raw.empty:
            # 列名の正規化（必要に応じて）
            if "Date" not in raw.columns:
                if "date" in raw.columns:
                    raw = raw.rename(columns={"date": "Date"})
                else:
                    raw = raw.copy()
                    raw["Date"] = pd.NaT
    except Exception:
        raw = get_cached_data(symbol)
    if raw is None or raw.empty:
        return None
    out = compute_base_indicators(raw)
    save_base_cache(symbol, out)
    return out
