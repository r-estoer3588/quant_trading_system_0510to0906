import argparse
from collections.abc import Callable, Iterable
import concurrent.futures
from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path
import sys
import threading
import time

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import requests

# from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.cache_manager import (  # noqa: E402
    CacheManager,
    compute_base_indicators,
    save_base_cache,
)
from config.settings import get_settings  # noqa: E402

try:
    from common.cache_format import round_dataframe  # noqa: E402
except ImportError:  # pragma: no cover - tests may stub cache_manager

    def round_dataframe(df: pd.DataFrame, decimals: int | None) -> pd.DataFrame:
        if decimals is None:
            return df
        try:
            decimals_int = int(decimals)
        except Exception:
            return df
        try:
            return df.copy().round(decimals_int)
        except Exception:
            try:
                return df.round(decimals_int)
            except Exception:
                return df


from common.indicators_common import add_indicators  # noqa: E402
from common.symbol_universe import build_symbol_universe_from_settings  # noqa: E402
from common.utils import safe_filename  # noqa: E402

load_dotenv()
API_KEY = os.getenv("EODHD_API_KEY")
PROGRESS_STEP_DEFAULT = 0


@dataclass(slots=True)
class BulkUpdateStats:
    """集計情報（テストや他モジュールからの再利用用）。"""

    fetched_rows: int = 0
    filtered_rows: int = 0
    processed_symbols: int = 0
    updated_symbols: int = 0
    filter_stats: dict[str, int | bool] = field(default_factory=dict)
    universe_error: bool = False
    universe_error_message: str | None = None
    estimated_symbols: int = 0
    progress_step_used: int = 0

    @property
    def has_payload(self) -> bool:
        """取得データが存在したか。"""

        return self.fetched_rows > 0

    @property
    def has_updates(self) -> bool:
        """キャッシュ更新が行われたか。"""

        return self.updated_symbols > 0


class CacheUpdateInterrupted(
    Exception,
):  # noqa: N801,N818 - keep historical name for callers
    """進捗情報を保持する中断例外。

    元々は KeyboardInterrupt を継承していましたが、
    呼び出し側で bare Exception を捕捉するコードが存在するため
    ここでは一般的な Exception を継承して握ることで、
    呼び出し側が適切に中断を検知して後処理できるようにします。
    """

    def __init__(self, processed: int, updated: int) -> None:
        super().__init__("cache update interrupted")
        self.processed = processed
        self.updated = updated


def _normalize_symbol(symbol: object) -> str:
    # None や float('nan') などを明示的に判定
    if symbol is None or (isinstance(symbol, float) and pd.isna(symbol)):
        return ""
    sym_norm = str(symbol).upper().strip()
    if not sym_norm:
        return ""
    if sym_norm.endswith(".US"):
        sym_norm = sym_norm.rsplit(".", 1)[0]
    return sym_norm


def _resolve_code_series(df: pd.DataFrame) -> pd.Series | None:
    for col in df.columns:
        if str(col).lower() == "code":
            return df[col]
    return None


def _estimate_symbol_counts(df: pd.DataFrame) -> tuple[int, int]:
    series = _resolve_code_series(df)
    if series is None:
        return 0, 0
    codes = series.dropna().astype(str).str.strip()
    codes = codes[codes != ""]
    if codes.empty:
        return 0, 0
    normalized = codes.map(_normalize_symbol)
    normalized = normalized[normalized != ""]
    original_count = int(codes.nunique())
    normalized_count = int(normalized.nunique())
    return original_count, normalized_count


_PRICE_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "adjclose",
)


def _drop_rows_without_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows that lack numeric price information altogether."""

    if df is None or df.empty:
        return df
    price_cols = [col for col in _PRICE_COLUMNS if col in df.columns]
    if not price_cols:
        return df
    numeric = df[price_cols].apply(pd.to_numeric, errors="coerce")
    mask = numeric.notna().any(axis=1)
    if mask.all():
        return df
    return pd.DataFrame(df.loc[mask].reset_index(drop=True))


def _resolve_progress_step(progress_target: int, requested_step: int) -> int:
    """進捗表示のステップを決定する。"""

    if requested_step and requested_step > 0:
        try:
            return max(int(requested_step), 1)
        except Exception:
            return 1
    if progress_target <= 0:
        return 1
    approx_step = max(progress_target // 200, 1)
    return min(approx_step, 20)


def _round_numeric_columns(df: pd.DataFrame, decimals: int | None) -> pd.DataFrame:
    """数値列を小数点以下 ``decimals`` 桁に丸めた DataFrame を返す。"""

    if decimals is None:
        return df
    try:
        dec = int(decimals)
    except (TypeError, ValueError):
        return df
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return df
    rounded = df.copy()
    try:
        rounded[numeric.columns] = numeric.round(dec)
    except Exception:
        return df
    return rounded


def _filter_bulk_data_by_universe(
    df: pd.DataFrame, symbols: Iterable[str]
) -> tuple[pd.DataFrame, dict[str, int | bool]]:
    """Filter the bulk-last-day payload by the provided symbol universe."""

    normalized_set = {str(sym).upper().strip() for sym in symbols if str(sym).strip()}
    stats: dict[str, int | bool] = {
        "allowed": len(normalized_set),
        "matched_rows": 0,
        "removed_rows": 0,
        "matched_symbols": 0,
        "missing_symbols": 0,
        "code_column_found": False,
    }
    if not normalized_set or df is None or df.empty:
        return df, stats

    series = _resolve_code_series(df)
    if series is None:
        return df, stats

    stats["code_column_found"] = True
    normalized_codes = series.map(_normalize_symbol)
    mask = normalized_codes.isin(normalized_set)
    matched_rows = int(mask.sum())
    stats["matched_rows"] = matched_rows
    stats["removed_rows"] = int(len(mask) - matched_rows)
    matched_symbols = {sym for sym in normalized_codes[mask] if sym}
    stats["matched_symbols"] = len(matched_symbols)
    stats["missing_symbols"] = len(normalized_set - matched_symbols)

    if matched_rows == len(df):
        return df, stats

    filtered = df.loc[mask].copy()
    return filtered, stats


def _resolve_base_dir(cm: CacheManager) -> Path:
    try:
        data_cache = Path(cm.settings.DATA_CACHE_DIR)
    except Exception:
        try:
            data_cache = Path(cm.full_dir).parent
        except Exception:
            data_cache = Path("data_cache")
    return data_cache / "base"


def _get_available_memory_mb() -> int | None:
    """Return available physical memory in MB, or None if not determinable."""
    try:
        import psutil

        mem = psutil.virtual_memory()
        return int(mem.available // (1024 * 1024))
    except Exception:
        return None


def _get_configured_rate_limit(cm: CacheManager) -> int | None:
    """Return configured API rate limit (requests per minute) from env or settings."""
    try:
        env = os.getenv("EODHD_RATE_LIMIT_PER_MIN") or os.getenv("API_RATE_LIMIT_PER_MIN")
        if env:
            try:
                return int(env)
            except Exception:
                pass
    except Exception:
        pass
    try:
        return int(getattr(cm.settings.cache, "api_rate_limit_per_min", 0)) or None
    except Exception:
        return None


def _extract_price_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjclose",
                "volume",
            ]
        )
    working = df.copy()
    lower_map = {str(col).lower(): col for col in working.columns}
    col_aliases = {
        "date": "date",
        "index": "date",
        "timestamp": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adjusted_close": "adjclose",
        "adj_close": "adjclose",
        "adjclose": "adjclose",
        "close_adj": "adjclose",
        "volume": "volume",
        "vol": "volume",
    }
    data: dict[str, pd.Series] = {}
    for alias, target in col_aliases.items():
        src = lower_map.get(alias)
        if src is not None:
            val = working[src]
            try:
                if isinstance(val, pd.DataFrame):
                    # 重複列名で DataFrame が返る場合は最後の列を採用
                    try:
                        last_col = val.columns[-1]
                        series_or_df = val[last_col]
                        # 万一 MultiIndex 等で DataFrame のまま返った場合に備える
                        if isinstance(series_or_df, pd.DataFrame):
                            try:
                                last_c = series_or_df.columns[-1]
                                series_or_df = series_or_df.get(last_c)
                            except Exception:
                                series_or_df = pd.Series(
                                    series_or_df.to_numpy().ravel(),
                                    index=getattr(series_or_df, "index", None),
                                )
                        val = series_or_df
                    except Exception:
                        # 最終手段: 値配列を 1 次元化
                        val = pd.Series(
                            getattr(val, "to_numpy", lambda: np.array(val))().ravel(),
                            index=getattr(val, "index", None),
                        )
                else:
                    # 2次元の配列ライクは1次元に圧縮
                    arr = getattr(val, "values", None)
                    if arr is not None and getattr(arr, "ndim", 1) > 1:
                        val = pd.Series(arr[:, -1], index=getattr(val, "index", None))
            except Exception:
                pass
            # 値を Series に正規化してから格納（None や DataFrame を排除）
            try:
                if not isinstance(val, pd.Series):
                    arr = getattr(val, "values", None)
                    if arr is None:
                        arr = np.array(val)
                    val = pd.Series(arr, index=getattr(val, "index", None))
            except Exception:
                # 型が合わなければスキップ
                continue
            data[target] = val
    frame = pd.DataFrame(data)
    if "adjclose" not in frame.columns and "close" in frame.columns:
        frame["adjclose"] = frame["close"]
    return frame


def _prepare_indicator_source(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    renamed = raw.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adjclose": "AdjClose",
            "volume": "Volume",
        }
    )
    if "Date" not in renamed.columns:
        return pd.DataFrame()
    renamed["Date"] = pd.to_datetime(renamed["Date"], errors="coerce")
    renamed = renamed.dropna(subset=["Date"]).sort_values("Date")
    if "Close" not in renamed.columns and "AdjClose" in renamed.columns:
        renamed["Close"] = renamed["AdjClose"]
    if "AdjClose" not in renamed.columns and "Close" in renamed.columns:
        renamed["AdjClose"] = renamed["Close"]
    if "Close" not in renamed.columns:
        return pd.DataFrame()
    return renamed.set_index("Date")


def _build_rolling_frame(full_df: pd.DataFrame, cm: CacheManager) -> pd.DataFrame:
    if full_df is None or full_df.empty:
        return full_df
    try:
        base_days = int(getattr(cm.rolling_cfg, "base_lookback_days", 0))
    except Exception:
        base_days = 0
    try:
        buffer_days = int(getattr(cm.rolling_cfg, "buffer_days", 0))
    except Exception:
        buffer_days = 0
    keep = max(base_days + buffer_days, 1)
    if len(full_df) <= keep:
        return full_df
    return full_df.iloc[-keep:].reset_index(drop=True)


def _concat_excluding_all_na(
    a: pd.DataFrame | None, b: pd.DataFrame | None, *, ignore_index: bool = True
) -> pd.DataFrame:
    """Concatenate two DataFrames while skipping columns empty in both frames.

    This preserves the previous behaviour pandas used to have where empty/all-NA
    columns were ignored for dtype determination. Future pandas versions will
    change that behaviour, so we explicitly drop such columns before concat.
    """
    if a is None or (hasattr(a, "empty") and a.empty):
        a = pd.DataFrame()
    if b is None or (hasattr(b, "empty") and b.empty):
        b = pd.DataFrame()
    if a.empty and b.empty:
        return pd.DataFrame()
    # Find columns present in either frame
    cols = set(a.columns) | set(b.columns)
    keep: list[str] = []
    for col in cols:
        a_col_all_na = True
        b_col_all_na = True
        if col in a.columns:
            try:
                a_col_all_na = a[col].dropna().empty
            except Exception:
                a_col_all_na = False
        if col in b.columns:
            try:
                b_col_all_na = b[col].dropna().empty
            except Exception:
                b_col_all_na = False
        # keep the column if at least one side has non-all-NA values
        if not (a_col_all_na and b_col_all_na):
            keep.append(col)
    # Subset frames to kept columns (if column absent, pandas will fill NA)
    a_sub: pd.DataFrame
    if a.empty:
        a_sub = pd.DataFrame(columns=keep)
    else:
        a_cols = [c for c in keep if c in a.columns]
        a_sub = a.loc[:, a_cols]
    b_sub: pd.DataFrame
    if b.empty:
        b_sub = pd.DataFrame(columns=keep)
    else:
        b_cols = [c for c in keep if c in b.columns]
        b_sub = b.loc[:, b_cols]
    return pd.concat([a_sub, b_sub], ignore_index=ignore_index)


def _merge_existing_full(
    new_full: pd.DataFrame, existing_full: pd.DataFrame | None
) -> pd.DataFrame:
    """既存データに新規データを追加・更新してマージする。

    既存データをベースとし、新規データで上書き/追加する。
    これにより既存の指標列も保持される。
    """
    if existing_full is None or existing_full.empty:
        return new_full

    # 両方をインデックス化（Date / date の両対応）
    previous = existing_full.copy()
    new_data = new_full.copy()

    def _ensure_date_index(df: pd.DataFrame) -> pd.DataFrame:
        if "Date" in df.columns:
            # 列を信頼し、型を揃えてから index に設定
            try:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            except Exception:
                pass
            out = df.set_index("Date", drop=False)
            return out
        if "date" in df.columns:
            try:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            except Exception:
                pass
            out = df.set_index("date", drop=False)
            # 列名を上位仕様の Date に統一
            if "Date" not in out.columns and "date" in out.columns:
                out = out.rename(columns={"date": "Date"})
            return out
        # フォールバック: 変更せず返す（後段でそのまま結合）
        return df

    previous = _ensure_date_index(previous)
    new_data = _ensure_date_index(new_data)

    # concatで結合（新規データを後に配置して重複時は新規優先）
    combined = pd.concat([previous, new_data])

    # 重複する日付は新規データを優先（最後の値を残す）
    combined = combined[~combined.index.duplicated(keep="last")]

    # 日付順にソート（DateTimeIndex 以外でも安定ソート）
    try:
        combined = combined.sort_index()
    except Exception:
        # index が日時でない場合は Date 列でソートを試みる
        if "Date" in combined.columns:
            try:
                combined = combined.sort_values("Date")
            except Exception:
                pass

    # 列順を整える（新規データの列を前に、既存の追加列を後ろに）
    # 大文字小文字違いの重複（例: Close と close）が混在すると後段処理で
    # 列重複によるエラーの原因になるため、base側の列名(小文字)と衝突する
    # 既存列は除外する。
    base_columns = list(new_full.columns)
    base_lower = {str(c).lower() for c in base_columns}
    prev_tail = [
        c for c in previous.columns if c not in base_columns and str(c).lower() not in base_lower
    ]
    ordered = base_columns + prev_tail
    final_columns = [c for c in ordered if c in combined.columns]

    # インデックスをリセットして返す（Date 列は保持）
    result = combined.reset_index(drop=True)
    # 念のため最終的にも列重複を排除（同名があっても最終出現を優先）
    dedup = result.loc[:, ~result.columns.duplicated(keep="last")]
    # ordered に存在し、かつ dedup に残っている列を順序維持で返す
    final = [c for c in final_columns if c in dedup.columns]
    return dedup[final]


def fetch_bulk_last_day() -> pd.DataFrame | None:
    url = "https://eodhistoricaldata.com/api/eod-bulk-last-day/US" f"?api_token={API_KEY}&fmt=json"
    masked_url = url
    try:
        if API_KEY:
            masked_url = url.replace(str(API_KEY), "***")
    except Exception:
        masked_url = url
    print(f"[DEBUG] Bulk API URL: {masked_url}", flush=True)
    try:
        response = requests.get(url, timeout=30)
        print(f"[DEBUG] Bulk API status: {response.status_code}", flush=True)
    except requests.RequestException as exc:
        print("Error fetching bulk data:", exc, flush=True)
        return None
    if response.status_code != 200:
        print("Error fetching bulk data:", response.status_code, flush=True)
        return None
    try:
        payload = response.json()
    except ValueError as exc:
        print("Error parsing bulk data:", exc, flush=True)
        return None
    row_count = len(payload) if isinstance(payload, list) else 0
    print(f"[DEBUG] Bulk API rows: {row_count}", flush=True)
    if row_count:
        try:
            sample_codes = [item.get("code", "") for item in payload[:5]]
            print(f"[DEBUG] Bulk API sample codes: {sample_codes}", flush=True)
        except Exception:
            pass
    return pd.DataFrame(payload)


def append_to_cache(
    df: pd.DataFrame,
    cm: CacheManager,
    *,
    progress_callback: Callable[[int, int, int], None] | None = None,
    progress_step: int = PROGRESS_STEP_DEFAULT,
    tail_rows: int | None = None,
) -> tuple[int, int]:
    """
    取得した1日分のデータを CacheManager の full/rolling にインクリメンタル反映する。
    戻り値: (対象銘柄数, upsert 成功数)
    progress_callback: 処理済み銘柄数/総数/更新数を報告するコールバック
    """
    if df is None or df.empty:
        return 0, 0
    # 必要列を小文字に統一し、'date' を datetime 化
    cols_map = {
        "code": "code",
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adjusted_close": "adjusted_close",
        "volume": "volume",
    }
    # 列名大文字・混在対策
    df = df.copy()
    # Use rename mapping to keep type information consistent
    df = df.rename(columns={c: str(c).lower() for c in df.columns})
    df = df.rename(columns={k: v for k, v in cols_map.items() if k in df.columns})
    if "date" not in df.columns:
        return 0, 0
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])  # 不正日時を除外
    # シンボル列（code）が無い場合は更新対象なし
    if "code" not in df.columns:
        return 0, 0
    df = _drop_rows_without_price_data(df)
    if df.empty:
        return 0, 0
    grouped = df.groupby("code")
    original_count, normalized_count = _estimate_symbol_counts(df)
    if original_count == 0 and normalized_count == 0 and grouped.ngroups == 0:
        return 0, 0
    progress_target = max(original_count, normalized_count, grouped.ngroups)
    step = _resolve_progress_step(progress_target, progress_step)
    # レポート間隔を短縮: 最大で10件ごとに報告するよう上限を設ける
    try:
        step = min(int(step), 10)
    except Exception:
        step = max(1, step)
    total = 0
    updated = 0
    last_report = 0
    last_report_time = time.monotonic()
    if progress_callback and progress_target:
        try:
            progress_callback(0, progress_target, 0)
        except Exception:
            pass

    # Worker function to process one symbol group.
    # Returns (processed_flag, updated_flag)
    def _process_symbol(sym, g):
        try:
            sym_norm = _normalize_symbol(sym)
            if not sym_norm:
                return 0, 0, sym_norm
            keep_cols = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjusted_close",
                "adjclose",
                "volume",
            ]
            cols_exist = [c for c in keep_cols if c in g.columns]
            if not cols_exist:
                return 0, 0, sym_norm
            rows = g[cols_exist].copy()
            # Use rename mapping to keep type information consistent
            rows = rows.rename(columns={c: str(c).lower() for c in rows.columns})
            # グループ内で重複列名がある場合はここで除去
            try:
                cols_idx = pd.Index(rows.columns)
                rows = rows.loc[:, ~cols_idx.duplicated(keep="last")]
            except Exception:
                pass
            try:
                existing_full = cm.read(sym_norm, "full")
            except Exception:
                existing_full = None
            existing_raw = _extract_price_frame(existing_full)
            new_raw = _extract_price_frame(rows)

            # 既存の最終日付以上の新規データが無ければ処理をスキップ（高速化）
            try:
                skip_if_no_new = (os.getenv("BULK_SKIP_IF_NO_NEW_DATE") or "1").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
            except Exception:
                skip_if_no_new = True
            if skip_if_no_new:
                try:
                    last_old = None
                    if (
                        existing_raw is not None
                        and not existing_raw.empty
                        and "date" in existing_raw.columns
                    ):
                        last_old = pd.to_datetime(existing_raw["date"], errors="coerce").max()
                    last_new = None
                    if new_raw is not None and not new_raw.empty and "date" in new_raw.columns:
                        last_new = pd.to_datetime(new_raw["date"], errors="coerce").max()
                    if pd.notna(last_old) and pd.notna(last_new) and last_new <= last_old:
                        return 1, 0, sym_norm
                except Exception:
                    pass
            combined = _concat_excluding_all_na(existing_raw, new_raw, ignore_index=True)
            if combined.empty:
                return 0, 0, sym_norm
            # 列重複を除去（特に 'date' の重複で Series ではなく DataFrame になるのを防ぐ）
            try:
                combined = combined.loc[:, ~combined.columns.duplicated(keep="last")]
            except Exception:
                pass
            # 'Date' / 'date' 候補から1本のSeriesに正規化し、残りは削除
            date_like_cols = [
                c for c in combined.columns if str(c).lower() in ("date", "timestamp", "index")
            ]
            chosen = None
            for c in date_like_cols:
                try:
                    s = combined[c]
                    # DataFrame（二重カラム）になっている場合は最後の列を採用
                    if isinstance(s, pd.DataFrame):
                        try:
                            arr2 = s.to_numpy()
                            if getattr(arr2, "ndim", 1) > 1:
                                arr2 = arr2[:, -1]
                            s = pd.Series(arr2, index=s.index)
                        except Exception:
                            continue
                    chosen = s
                except Exception:
                    continue
            if chosen is not None:
                combined = combined.drop(
                    columns=[col for col in date_like_cols if col in combined.columns]
                )
                # 1次元配列に強制変換してから datetime 変換（2D代入エラー回避）
                try:
                    arr = getattr(chosen, "values", None)
                    if arr is None:
                        arr = np.array(chosen)
                    if getattr(arr, "ndim", 1) > 1:
                        try:
                            arr = arr[:, -1]
                        except Exception:
                            arr = np.ravel(arr)
                    dates = pd.to_datetime(arr, errors="coerce")
                except Exception:
                    dates = pd.to_datetime(chosen, errors="coerce")
                combined["date"] = dates
            combined = combined.dropna(subset=["date"]).sort_values("date")
            combined = combined.drop_duplicates("date", keep="last")
            combined = combined.reset_index(drop=True)
            if combined.empty:
                return 0, 0, sym_norm
            # 価格列が全行でNaN（有効な価格情報なし）の場合はスキップ
            price_cols = [
                c for c in ["open", "high", "low", "close", "volume"] if c in combined.columns
            ]
            if price_cols:
                try:
                    numeric = combined[price_cols].apply(pd.to_numeric, errors="coerce")
                    if numeric.notna().any(axis=1).sum() == 0:
                        return 0, 0, sym_norm
                except Exception:
                    pass
            # 指標再計算はテールのみ（必要十分な履歴長を tail_rows で確保）
            if tail_rows is None or tail_rows <= 0:
                try:
                    tail_rows_eff = int(os.getenv("BULK_TAIL_ROWS") or 240)
                except Exception:
                    tail_rows_eff = 240
            else:
                tail_rows_eff = int(tail_rows)

            # テール抽出（最後の tail_rows_eff 本）
            combined_tail = combined
            try:
                if len(combined) > tail_rows_eff:
                    combined_tail = combined.iloc[-tail_rows_eff:].reset_index(drop=True)
            except Exception:
                pass

            indicator_source = _prepare_indicator_source(combined_tail)
            if indicator_source.empty:
                return 0, 0, sym_norm

            enriched = None
            # 指標計算: テストの決定性を保つため deterministic のときのみロック。
            if stop_event.is_set():
                return 0, 0, sym_norm
            try:
                if use_lock:
                    with add_indicators_lock:
                        if stop_event.is_set():
                            return 0, 0, sym_norm
                        enriched = add_indicators(indicator_source)
                else:
                    enriched = add_indicators(indicator_source)
            except KeyboardInterrupt:
                stop_event.set()
                raise
            except Exception as exc:
                print(f"{sym_norm}: indicator calc error - {exc}")
                enriched = indicator_source.copy()
            if enriched is None:
                enriched = indicator_source.copy()

            # cache_daily_data.py に合わせて列は大文字系で保持
            full_tail_ready = enriched.reset_index().sort_values("Date").reset_index(drop=True)
            # Close/AdjClose を相互補完
            if "Close" not in full_tail_ready.columns and "AdjClose" in full_tail_ready.columns:
                full_tail_ready["Close"] = full_tail_ready["AdjClose"]
            if "AdjClose" not in full_tail_ready.columns and "Close" in full_tail_ready.columns:
                full_tail_ready["AdjClose"] = full_tail_ready["Close"]
            # 既存 full とテールをマージ（重複日は新しい方＝テールを優先）
            full_ready = _merge_existing_full(full_tail_ready, existing_full)
            prev_full_sorted = None
            if existing_full is not None and not existing_full.empty:
                prev_full_sorted = existing_full.copy().sort_values("date").reset_index(drop=True)
            try:
                cm.write_atomic(full_ready, sym_norm, "full")
            except Exception as exc:
                print(f"{sym_norm}: write full error - {exc}")
                return 1, 0, sym_norm

            # rolling frame: full_ready の末尾を切り出すだけで再計算コストを削減
            rolling_ready = _build_rolling_frame(full_ready, cm)

            try:
                cm.write_atomic(rolling_ready, sym_norm, "rolling")
            except Exception as exc:
                print(f"{sym_norm}: write rolling error - {exc}")

            skip_base = (os.getenv("BULK_SKIP_BASE_INDICATORS") or "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if skip_base:
                base_df = None
            else:
                try:
                    base_df = compute_base_indicators(full_ready)
                except Exception as exc:
                    print(f"{sym_norm}: base indicator error - {exc}")
                    base_df = None

            # Ensure we always produce a base CSV (tests expect it). If
            # indicator computation failed or returned empty, fall back to
            # writing the full_ready frame so a base file exists.
            try:
                base_path = base_dir / f"{safe_filename(sym_norm)}.csv"
                base_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    settings = getattr(cm, "settings", None)
                    cache_obj = getattr(settings, "cache", None)
                    dec = getattr(cache_obj, "round_decimals", None)
                except Exception:
                    dec = None

                if base_df is not None and not base_df.empty:
                    base_reset = round_dataframe(base_df.reset_index(), dec)
                else:
                    base_reset = round_dataframe(full_ready.reset_index(), dec)

                # cache_daily_data.py の base CSV 形式に合わせる
                if "index" in base_reset.columns:
                    base_reset = base_reset.drop(columns=["index"])
                if "date" in base_reset.columns and "Date" not in base_reset.columns:
                    base_reset = base_reset.rename(columns={"date": "Date"})
                # AdjClose があれば Close をそれで埋める
                if "AdjClose" in base_reset.columns:
                    base_reset["Close"] = base_reset["AdjClose"]

                # Save base cache using CacheManager's save_base_cache
                # function (now supports feather)
                base_path = save_base_cache(sym_norm, base_reset, cm.settings)
            except Exception as exc:
                print(f"{sym_norm}: write base error - {exc}")

            was_updated = 0
            if prev_full_sorted is None or not full_ready.equals(prev_full_sorted):
                was_updated = 1
            return 1, was_updated, sym_norm
        except Exception as exc:
            print(f"{sym}: unexpected error - {exc}")
            return 0, 0, str(sym)

    try:
        base_dir = _resolve_base_dir(cm)
        # determine worker count
        try:
            cfg_workers = getattr(cm.settings.cache, "bulk_update_workers", None)
        except Exception:
            cfg_workers = None
        try:
            env_workers_str = os.getenv("BULK_UPDATE_WORKERS")
            env_workers = int(env_workers_str) if env_workers_str else None
        except Exception:
            env_workers = None
        try:
            # For this task (mix of IO and CPU), prefer more threads than cores
            # to hide IO waits. Use 2x CPU heuristic, bounded, but also reduce
            # workers when memory or API rate limits are low.
            cpu = os.cpu_count() or 1
            default_workers = max(2, min(32, int(cpu * 2)))
        except Exception:
            default_workers = 4

        # Consider available memory: assume ~100MB per worker as heuristic
        mem_mb = _get_available_memory_mb()
        if mem_mb is not None and mem_mb > 0:
            try:
                mem_based = max(1, int(mem_mb // 100))
                # don't exceed a reasonable cap
                mem_based = min(mem_based, 64)
                default_workers = min(default_workers, mem_based)
            except Exception:
                pass

        # Consider configured API rate limit (requests per minute)
        # If specified, convert to approximate concurrent workers to avoid
        # exceeding rate
        try:
            rate = _get_configured_rate_limit(cm)
            if rate and rate > 0:
                # assume each worker may perform ~1 request per 2 seconds on average
                approx_per_min = int(max(1, rate // 2))
                rate_based = max(1, min(64, approx_per_min))
                default_workers = min(default_workers, rate_based)
        except Exception:
            pass

        max_workers = env_workers or cfg_workers or default_workers
        # Deterministic mode: single worker under tests or when requested
        deterministic = False
        try:
            if os.environ.get("PYTEST_CURRENT_TEST"):
                deterministic = True
        except Exception:
            pass
        try:
            det_env = (os.getenv("BULK_UPDATE_DETERMINISTIC") or "").strip().lower()
            if det_env in {"1", "true", "yes", "on"}:
                deterministic = True
        except Exception:
            pass
        try:
            if deterministic:
                max_workers = 1
            else:
                max_workers = max(1, int(max_workers or default_workers))
        except Exception:
            max_workers = max(1, default_workers)

        # Log explicitly the chosen worker count so users can verify
        try:
            print(f"ℹ️ worker count selected: {max_workers}")
        except Exception:
            pass

        # Submit tasks to thread pool. We use a shared stop_event and a
        # lock to serialize calls to add_indicators so that a KeyboardInterrupt
        # raised by one worker prevents other workers from attempting the
        # same call (keeps tests deterministic).
        stop_event = threading.Event()
        add_indicators_lock = threading.Lock()
        # 非決定モードではロックを外して指標計算の同時並列を許可
        use_lock = deterministic
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exe:
            # Submit up to max_workers tasks and then feed remaining as tasks complete
            pending: set[concurrent.futures.Future] = set()
            # Iterator over grouped to avoid materializing entire groups in memory
            grouped_iter = iter(grouped)

            def _submit_next() -> bool:
                try:
                    sym_next, g_next = next(grouped_iter)
                except StopIteration:
                    return False
                fut_next = exe.submit(_process_symbol, sym_next, g_next)
                pending.add(fut_next)
                return True

            # Prime initial batch
            for _ in range(max_workers):
                if not _submit_next():
                    break

            while pending:
                done, pending = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for fut in done:
                    try:
                        proc_flag, upd_flag, symname = fut.result()
                    except BaseException as exc:
                        if isinstance(exc, KeyboardInterrupt):
                            total += 1
                            stop_event.set()
                            # drain without starting new tasks
                            pending.clear()
                            raise
                        print(f"symbol task failed: {exc}")
                        proc_flag, upd_flag = 0, 0
                    total += int(proc_flag)
                    updated += int(upd_flag)
                    if progress_callback:
                        effective_target = max(progress_target, total)
                        now = time.monotonic()
                        time_elapsed = now - last_report_time
                        should_report = (
                            total == effective_target
                            or total - last_report >= step
                            or total == 1
                            or time_elapsed >= 2.0
                        )
                        if should_report:
                            try:
                                progress_callback(total, effective_target, updated)
                            except Exception:
                                pass
                            last_report = total
                            last_report_time = now
                    # backfill: submit the next task for each completed one
                    _submit_next()
    except KeyboardInterrupt:  # pragma: no cover - 手動中断
        if progress_callback and total and total != last_report:
            effective_target = max(progress_target, total)
            try:
                progress_callback(total, effective_target, updated)
            except Exception:
                pass
        raise CacheUpdateInterrupted(total, updated) from None
    # rolling のメンテナンス
    try:
        cm.prune_rolling_if_needed(anchor_ticker="SPY")
    except Exception:
        pass
    if progress_callback and total and total != last_report:
        effective_target = max(progress_target, total)
        try:
            progress_callback(total, effective_target, updated)
        except Exception:
            pass
    return total, updated


def run_bulk_update(
    cm: CacheManager,
    *,
    data: pd.DataFrame | None = None,
    universe: Iterable[str] | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
    progress_step: int = PROGRESS_STEP_DEFAULT,
    fetch_universe: bool = True,
    tail_rows: int | None = None,
) -> BulkUpdateStats:
    """Fetch (or reuse) bulk last-day data and upsert into caches."""

    stats = BulkUpdateStats()

    working = data if data is not None else fetch_bulk_last_day()
    if working is None or working.empty:
        print("[DEBUG] Bulk payload is empty (fetch failure or no rows)", flush=True)
        return stats

    stats.fetched_rows = len(working)
    print(f"[DEBUG] Raw bulk rows: {stats.fetched_rows}", flush=True)

    target_universe: list[str] | None = None
    if universe is not None:
        target_universe = []
        for sym in universe:
            if not isinstance(sym, str):
                continue
            trimmed = sym.strip()
            if trimmed:
                target_universe.append(trimmed)
        print(
            f"[DEBUG] CLI-provided universe count: {len(target_universe)}",
            flush=True,
        )
    elif fetch_universe:
        try:
            settings = getattr(cm, "settings", None)
            if settings is not None:
                fetched = build_symbol_universe_from_settings(settings)
                cleaned: list[str] = []
                for sym in fetched:
                    if not isinstance(sym, str):
                        continue
                    trimmed = sym.strip()
                    if trimmed:
                        cleaned.append(trimmed)
                target_universe = cleaned
                print(
                    f"[DEBUG] Universe from settings: {len(target_universe)}",
                    flush=True,
                )
        except Exception as exc:
            stats.universe_error = True
            stats.universe_error_message = str(exc)
            print(f"[DEBUG] Universe fetch error: {exc}", flush=True)
            target_universe = None

    filtered = working
    filter_stats: dict[str, int | bool]
    if target_universe:
        filtered, filter_stats = _filter_bulk_data_by_universe(working, target_universe)
        print(
            "[DEBUG] Filtered rows by universe: "
            f"{filter_stats.get('matched_rows', 0)} / {len(working)}",
            flush=True,
        )
        missing = filter_stats.get("missing_symbols", 0)
        if missing:
            print(f"[DEBUG] Missing symbols in universe: {missing}", flush=True)
    else:
        has_code = _resolve_code_series(working) is not None
        filter_stats = {
            "allowed": 0,
            "matched_rows": len(filtered),
            "removed_rows": 0,
            "matched_symbols": 0,
            "missing_symbols": 0,
            "code_column_found": has_code,
        }

    stats.filter_stats = filter_stats
    stats.filtered_rows = len(filtered)
    print(f"[DEBUG] Rows after drop-empty-universe: {stats.filtered_rows}", flush=True)

    prepared = _drop_rows_without_price_data(filtered)
    print(
        "[DEBUG] Rows after price-column check: "
        f"{len(prepared)} (removed {len(filtered) - len(prepared)})",
        flush=True,
    )
    if prepared.empty:
        print("[DEBUG] No rows with price data; aborting bulk update", flush=True)
        return stats

    original_count, normalized_count = _estimate_symbol_counts(prepared)
    stats.estimated_symbols = max(original_count, normalized_count)
    resolved_step = _resolve_progress_step(stats.estimated_symbols, progress_step)

    total, updated = append_to_cache(
        prepared,
        cm,
        progress_callback=progress_callback,
        progress_step=resolved_step,
        tail_rows=tail_rows,
    )

    stats.processed_symbols = total
    stats.updated_symbols = updated
    stats.progress_step_used = resolved_step
    return stats


def main():
    parser = argparse.ArgumentParser(description="Bulk last-day updater")
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="処理する最大銘柄数（デバッグ用に小さく回す）",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="カンマ区切りで対象シンボルを指定（例: AAPL,MSFT,GOOGL）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="ワーカ数を明示指定（環境変数 BULK_UPDATE_WORKERS より優先）",
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="base 指標計算をスキップして保存（高速化）",
    )
    parser.add_argument(
        "--no-skip-date",
        action="store_true",
        help="新規日付が無い銘柄も処理（デフォルトは新規日付なしをスキップ）",
    )
    parser.add_argument(
        "--tail-rows",
        type=int,
        default=None,
        help=("指標の再計算に用いるテール行数（未指定時は環境変数 BULK_TAIL_ROWS または 240）"),
    )
    args = parser.parse_args()
    if not API_KEY:
        print("EODHD_API_KEY が未設定です (.env を確認)", flush=True)
        return
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("🚀 EODHD Bulk Last Day 更新を開始します...", flush=True)
    print(f"⏰ 開始時刻: {start_time}", flush=True)
    print("📡 API リクエストを送信中...", flush=True)
    settings = get_settings(create_dirs=True)
    cm = CacheManager(settings)

    # 大量の NaN 率ログなどのヘルスチェック出力を抑止して速度重視にする（実行中のみ適用）
    _prev_cache_health_silent = os.environ.get("CACHE_HEALTH_SILENT")
    os.environ["CACHE_HEALTH_SILENT"] = "1"

    # 実行オプションに応じて環境変数をセット（このプロセス内のみ有効）
    if args.workers is not None and args.workers > 0:
        os.environ["BULK_UPDATE_WORKERS"] = str(args.workers)
    if args.skip_base:
        os.environ["BULK_SKIP_BASE_INDICATORS"] = "1"
    if args.no_skip_date:
        os.environ["BULK_SKIP_IF_NO_NEW_DATE"] = "0"

    progress_state = {"processed": 0, "total": 0, "updated": 0}

    def _report_progress(
        processed: int,
        total_symbols: int,
        updated_count: int,
    ) -> None:
        progress_state["processed"] = processed
        progress_state["total"] = total_symbols
        progress_state["updated"] = updated_count
        print(
            f"  ⏳ 処理中: {processed}/{total_symbols} 銘柄 (更新 {updated_count})",
            flush=True,
        )

    # 20 秒ごとのハートビート（進捗が停滞しても稼働中と分かるようにする）
    _stop_hb = threading.Event()

    def _heartbeat() -> None:
        while not _stop_hb.wait(20.0):
            now_hb = datetime.now().strftime("%H:%M:%S")
            processed = progress_state.get("processed", 0)
            total = progress_state.get("total", 0)
            updated = progress_state.get("updated", 0)
            if total <= 0:
                total = max(processed, 0)
            print(
                (f"  💓 Heartbeat {now_hb}: {processed}/{total} processed " f"(updated {updated})"),
                flush=True,
            )

    _hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    _hb_thread.start()

    try:
        # ユニバース制限の構築
        universe_limit = None
        if args.symbols:
            universe_limit = [s.strip() for s in args.symbols.split(",") if s.strip()]

        if universe_limit is None and args.max_symbols:
            # 設定からユニバースを取得して小さく制限
            try:
                fetched = build_symbol_universe_from_settings(settings)
                if isinstance(fetched, (list, tuple)):
                    universe_limit = list(fetched)[: args.max_symbols]
            except Exception:
                universe_limit = None

        result = run_bulk_update(
            cm,
            universe=universe_limit,
            progress_callback=_report_progress,
            progress_step=PROGRESS_STEP_DEFAULT,
            fetch_universe=(universe_limit is None),
            tail_rows=args.tail_rows,
        )
    except CacheUpdateInterrupted as exc:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_symbols = max(progress_state.get("total", 0), exc.processed)
        print("🛑 ユーザーにより更新処理が中断されました", flush=True)
        print(
            f"   ↳ {now} 時点 | 処理済み: {exc.processed}/{total_symbols}"
            f" 銘柄 / 更新済み: {exc.updated} 銘柄",
            flush=True,
        )
        # ハートビート停止と環境変数の復元
        _stop_hb.set()
        try:
            _hb_thread.join(timeout=2.0)
        except Exception:
            pass
        if _prev_cache_health_silent is None:
            try:
                del os.environ["CACHE_HEALTH_SILENT"]
            except Exception:
                pass
        else:
            os.environ["CACHE_HEALTH_SILENT"] = _prev_cache_health_silent
        return
    except KeyboardInterrupt:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_symbols = max(
            progress_state.get("total", 0),
            progress_state.get("processed", 0),
        )
        processed_count = progress_state.get("processed", 0)
        updated_count = progress_state.get("updated", 0)
        print("🛑 ユーザーにより更新処理が中断されました", flush=True)
        print(
            f"   ↳ {now} 時点 | 処理済み: {processed_count}/{total_symbols}"
            f" 銘柄 / 更新済み: {updated_count} 銘柄",
            flush=True,
        )
        # ハートビート停止と環境変数の復元
        _stop_hb.set()
        try:
            _hb_thread.join(timeout=2.0)
        except Exception:
            pass
        if _prev_cache_health_silent is None:
            try:
                del os.environ["CACHE_HEALTH_SILENT"]
            except Exception:
                pass
        else:
            os.environ["CACHE_HEALTH_SILENT"] = _prev_cache_health_silent
        return

    if not result.has_payload:
        print("No data to update.", flush=True)
        # ハートビート停止と環境変数の復元
        _stop_hb.set()
        try:
            _hb_thread.join(timeout=2.0)
        except Exception:
            pass
        if _prev_cache_health_silent is None:
            try:
                del os.environ["CACHE_HEALTH_SILENT"]
            except Exception:
                pass
        else:
            os.environ["CACHE_HEALTH_SILENT"] = _prev_cache_health_silent
        return

    print(f"📦 取得件数(未フィルタ): {result.fetched_rows} 行", flush=True)

    progress_state["total"] = max(
        progress_state.get("total", 0),
        result.estimated_symbols,
    )
    total_symbols = max(
        progress_state.get("total", 0),
        result.processed_symbols,
        result.estimated_symbols,
    )
    if result.progress_step_used and total_symbols:
        print(
            "🧮 進捗ログ間隔: "
            f"{result.progress_step_used} 銘柄ごと (対象 {total_symbols} 銘柄想定)",
            flush=True,
        )

    filter_stats = result.filter_stats
    allowed_count = int(filter_stats.get("allowed", 0))
    if allowed_count:
        print(
            f"🎯 フィルタ済みユニバース参照数: {allowed_count} 銘柄",
            flush=True,
        )
        if not filter_stats.get("code_column_found"):
            print(
                "  ↳ 取得データに code 列が無いためフィルタリングを適用できませんでした",
                flush=True,
            )
        else:
            matched_symbols = int(filter_stats.get("matched_symbols", 0))
            matched_rows = int(filter_stats.get("matched_rows", 0))
            removed_rows = int(filter_stats.get("removed_rows", 0))
            missing_symbols = int(filter_stats.get("missing_symbols", 0))
            print(
                f"  ↳ Bulk データ一致: {matched_symbols} 銘柄 / {matched_rows} 行",
                flush=True,
            )
            if removed_rows:
                print(
                    f"  ↳ フィルタで除外: {removed_rows} 行",
                    flush=True,
                )
            if missing_symbols:
                print(
                    f"  ↳ Bulk データに存在しない銘柄: {missing_symbols} 件",
                    flush=True,
                )
    elif result.universe_error:
        message = result.universe_error_message or "理由不明"
        print(
            "⚠️ 銘柄ユニバース取得に失敗したためフィルタリングをスキップします:",
            message,
            flush=True,
        )
    else:
        print(
            "⚠️ 銘柄ユニバース取得結果が空だったためフィルタリングをスキップします",
            flush=True,
        )

    if result.filtered_rows == 0:
        print("⚠️ フィルタ後の対象データが 0 件のため処理を終了します。", flush=True)
        return

    if result.estimated_symbols:
        print(
            f"📊 取得件数: {result.filtered_rows} 行 / 銘柄数(推定): {result.estimated_symbols}",
            flush=True,
        )
    else:
        print(f"📊 取得件数: {result.filtered_rows} 行", flush=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        "✅ "
        f"{now} | 対象: {result.processed_symbols} 銘柄 / 更新: "
        f"{result.updated_symbols} 銘柄（full/rolling へ反映）",
        flush=True,
    )

    # 正常終了時のハートビート停止と環境変数の復元
    _stop_hb.set()
    try:
        _hb_thread.join(timeout=2.0)
    except Exception:
        pass
    if _prev_cache_health_silent is None:
        try:
            del os.environ["CACHE_HEALTH_SILENT"]
        except Exception:
            pass
    else:
        os.environ["CACHE_HEALTH_SILENT"] = _prev_cache_health_silent


if __name__ == "__main__":
    main()
