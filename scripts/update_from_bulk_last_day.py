from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
import pandas as pd
import requests

# from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.cache_manager import CacheManager, compute_base_indicators  # noqa: E402
from config.settings import get_settings  # noqa: E402

try:
    from common.cache_manager import round_dataframe  # type: ignore # noqa: E402
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


from indicators_common import add_indicators  # noqa: E402

from common.symbol_universe import build_symbol_universe_from_settings  # noqa: E402
from common.utils import safe_filename  # noqa: E402

load_dotenv()
API_KEY = os.getenv("EODHD_API_KEY")
PROGRESS_STEP_DEFAULT = 100


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

    @property
    def has_payload(self) -> bool:
        """取得データが存在したか。"""

        return self.fetched_rows > 0

    @property
    def has_updates(self) -> bool:
        """キャッシュ更新が行われたか。"""

        return self.updated_symbols > 0


class CacheUpdateInterrupted(Exception):
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


def _extract_price_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adjclose", "volume"])
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
            data[target] = working[src]
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
    a: pd.DataFrame | None, b: pd.DataFrame | None, **kwargs
) -> pd.DataFrame:
    """Concatenate two DataFrames while excluding columns that are empty or all-NA in both.

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
    a_sub = (
        a.loc[:, [c for c in keep if c in a.columns]] if not a.empty else pd.DataFrame(columns=keep)
    )
    b_sub = (
        b.loc[:, [c for c in keep if c in b.columns]] if not b.empty else pd.DataFrame(columns=keep)
    )
    return pd.concat([a_sub, b_sub], ignore_index=kwargs.get("ignore_index", True))


def _merge_existing_full(
    new_full: pd.DataFrame, existing_full: pd.DataFrame | None
) -> pd.DataFrame:
    if existing_full is None or existing_full.empty:
        return new_full
    base_columns = list(new_full.columns)
    merged = new_full.set_index("date")
    previous = existing_full.copy().set_index("date")
    for col in previous.columns:
        if col == "date":
            continue
        if col not in merged.columns:
            merged[col] = previous[col]
        else:
            merged[col] = merged[col].combine_first(previous[col])
    ordered = base_columns + [c for c in previous.columns if c not in base_columns]
    merged = merged.reset_index()
    return merged.loc[:, ordered]


def fetch_bulk_last_day() -> pd.DataFrame | None:
    url = f"https://eodhistoricaldata.com/api/eod-bulk-last-day/US?api_token={API_KEY}&fmt=json"
    try:
        response = requests.get(url, timeout=30)
    except requests.RequestException as exc:
        print("Error fetching bulk data:", exc)
        return None
    if response.status_code != 200:
        print("Error fetching bulk data:", response.status_code)
        return None
    try:
        payload = response.json()
    except ValueError as exc:
        print("Error parsing bulk data:", exc)
        return None
    return pd.DataFrame(payload)


def append_to_cache(
    df: pd.DataFrame,
    cm: CacheManager,
    *,
    progress_callback: Callable[[int, int, int], None] | None = None,
    progress_step: int = PROGRESS_STEP_DEFAULT,
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
    df.columns = [str(c).lower() for c in df.columns]
    df = df.rename(columns={k: v for k, v in cols_map.items() if k in df.columns})
    if "date" not in df.columns:
        return 0, 0
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])  # 不正日時を除外
    # シンボル列（code）が無い場合は更新対象なし
    if "code" not in df.columns:
        return 0, 0
    grouped = df.groupby("code")
    original_count, normalized_count = _estimate_symbol_counts(df)
    if original_count == 0 and normalized_count == 0 and grouped.ngroups == 0:
        return 0, 0
    progress_target = max(original_count, normalized_count, grouped.ngroups)
    step = max(int(progress_step or 0), 1)
    total = 0
    updated = 0
    last_report = 0
    if progress_callback and progress_target:
        try:
            progress_callback(0, progress_target, 0)
        except Exception:
            pass
    try:
        base_dir = _resolve_base_dir(cm)
        # round_decimals not needed here; rounding is handled at write time
        for sym, g in grouped:
            sym_norm = _normalize_symbol(sym)
            if not sym_norm:
                continue
            total += 1
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
                continue
            rows = g[cols_exist].copy()
            rows.columns = [str(c).lower() for c in rows.columns]
            existing_full: pd.DataFrame | None
            try:
                existing_full = cm.read(sym_norm, "full")
            except Exception:
                existing_full = None
            existing_raw = _extract_price_frame(existing_full)
            new_raw = _extract_price_frame(rows)
            combined = _concat_excluding_all_na(existing_raw, new_raw, ignore_index=True)
            if combined.empty:
                continue
            combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
            combined = combined.dropna(subset=["date"]).sort_values("date")
            combined = combined.drop_duplicates("date", keep="last")
            combined = combined.reset_index(drop=True)
            if combined.empty:
                continue
            indicator_source = _prepare_indicator_source(combined)
            if indicator_source.empty:
                continue
            try:
                enriched = add_indicators(indicator_source)
            except Exception as exc:
                print(f"{sym_norm}: indicator calc error - {exc}")
                enriched = indicator_source.copy()
            full_ready = (
                enriched.reset_index()
                .rename(columns=str.lower)
                .sort_values("date")
                .reset_index(drop=True)
            )
            if "adjusted_close" not in full_ready.columns and "adjclose" in full_ready.columns:
                full_ready["adjusted_close"] = full_ready["adjclose"]
            elif "adjusted_close" in full_ready.columns and "adjclose" in full_ready.columns:
                mask_adj = full_ready["adjusted_close"].isna()
                if mask_adj.any():
                    full_ready.loc[mask_adj, "adjusted_close"] = full_ready.loc[
                        mask_adj, "adjclose"
                    ]
            full_ready = _merge_existing_full(full_ready, existing_full)
            prev_full_sorted: pd.DataFrame | None = None
            if existing_full is not None and not existing_full.empty:
                prev_full_sorted = existing_full.copy().sort_values("date").reset_index(drop=True)
            try:
                cm.write_atomic(full_ready, sym_norm, "full")
            except Exception as exc:
                print(f"{sym_norm}: write full error - {exc}")
                continue

            # rolling 用フレームを作成し、主要指標を付与してから書き込む
            rolling_raw = _build_rolling_frame(full_ready, cm)
            try:
                rolling_ind = compute_base_indicators(rolling_raw)
            except Exception as exc:
                print(f"{sym_norm}: rolling base indicator error - {exc}")
                rolling_ind = None
            if rolling_ind is None or getattr(rolling_ind, "empty", False):
                rolling_ready = rolling_raw
            else:
                try:
                    rolling_ready = (
                        rolling_ind.reset_index()
                        .rename(columns=str.lower)
                        .sort_values("date")
                        .reset_index(drop=True)
                    )
                except Exception:
                    rolling_ready = rolling_raw

            try:
                cm.write_atomic(rolling_ready, sym_norm, "rolling")
            except Exception as exc:
                print(f"{sym_norm}: write rolling error - {exc}")

            try:
                base_df = compute_base_indicators(full_ready)
            except Exception as exc:
                print(f"{sym_norm}: base indicator error - {exc}")
                base_df = None
            if base_df is not None and not base_df.empty:
                base_path = base_dir / f"{safe_filename(sym_norm)}.csv"
                base_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    try:
                        dec = getattr(cm.settings.cache, "round_decimals", None)
                    except Exception:
                        dec = None
                    base_reset = round_dataframe(base_df.reset_index(), dec)
                    base_reset.to_csv(base_path, index=False)
                except Exception as exc:
                    print(f"{sym_norm}: write base error - {exc}")
            if prev_full_sorted is None or not full_ready.equals(prev_full_sorted):
                updated += 1
            if progress_callback:
                effective_target = max(progress_target, total)
                should_report = (
                    total == effective_target or total - last_report >= step or total == 1
                )
                if should_report:
                    try:
                        progress_callback(total, effective_target, updated)
                    except Exception:
                        pass
                    last_report = total
    except KeyboardInterrupt:  # pragma: no cover - 手動中断
        # 捕捉して進捗情報を含む専用例外で呼び出し側へ伝える
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
) -> BulkUpdateStats:
    """Fetch (or reuse) bulk last-day data and upsert into caches."""

    stats = BulkUpdateStats()

    working = data if data is not None else fetch_bulk_last_day()
    if working is None or working.empty:
        return stats

    stats.fetched_rows = len(working)

    target_universe: list[str] | None = None
    if universe is not None:
        target_universe = [
            str(sym).strip() for sym in universe if isinstance(sym, str) and str(sym).strip()
        ]
    elif fetch_universe:
        try:
            settings = getattr(cm, "settings", None)
            if settings is not None:
                fetched = build_symbol_universe_from_settings(settings)
                target_universe = [
                    str(sym).strip() for sym in fetched if isinstance(sym, str) and str(sym).strip()
                ]
        except Exception as exc:
            stats.universe_error = True
            stats.universe_error_message = str(exc)
            target_universe = None

    filtered = working
    filter_stats: dict[str, int | bool]
    if target_universe:
        filtered, filter_stats = _filter_bulk_data_by_universe(working, target_universe)
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

    if filtered.empty:
        return stats

    original_count, normalized_count = _estimate_symbol_counts(filtered)
    stats.estimated_symbols = max(original_count, normalized_count)

    total, updated = append_to_cache(
        filtered,
        cm,
        progress_callback=progress_callback,
        progress_step=progress_step,
    )

    stats.processed_symbols = total
    stats.updated_symbols = updated
    return stats


def main():
    if not API_KEY:
        print("EODHD_API_KEY が未設定です (.env を確認)", flush=True)
        return
    print("🚀 EODHD Bulk Last Day 更新を開始します...", flush=True)
    print("📡 API リクエストを送信中...", flush=True)
    settings = get_settings(create_dirs=True)
    cm = CacheManager(settings)

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

    try:
        result = run_bulk_update(
            cm,
            progress_callback=_report_progress,
            progress_step=PROGRESS_STEP_DEFAULT,
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
        return

    if not result.has_payload:
        print("No data to update.", flush=True)
        return

    progress_state["total"] = max(
        progress_state.get("total", 0),
        result.estimated_symbols,
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


if __name__ == "__main__":
    main()
