import os
import sys
from datetime import datetime
from pathlib import Path
from collections.abc import Callable, Iterable

# from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings  # noqa: E402
from common.cache_manager import CacheManager  # noqa: E402
from common.symbol_universe import build_symbol_universe_from_settings  # noqa: E402

load_dotenv()
API_KEY = os.getenv("EODHD_API_KEY")
PROGRESS_STEP_DEFAULT = 100


class CacheUpdateInterrupted(KeyboardInterrupt):
    """KeyboardInterrupt に伴う進捗情報を保持する例外"""

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


def _filter_bulk_data_by_universe(
    df: pd.DataFrame, symbols: Iterable[str]
) -> tuple[pd.DataFrame, dict[str, int | bool]]:
    """Filter the bulk-last-day payload by the provided symbol universe."""

    normalized_set = {
        str(sym).upper().strip() for sym in symbols if str(sym).strip()
    }
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


def fetch_bulk_last_day() -> pd.DataFrame | None:
    url = "https://eodhistoricaldata.com/api/eod-bulk-last-day/US" f"?api_token={API_KEY}&fmt=json"
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
    interrupt_exc: BaseException | None = None
    try:
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
                "volume",
            ]
            cols_exist = [c for c in keep_cols if c in g.columns]
            if not cols_exist:
                continue
            rows = g[cols_exist].copy()
            try:
                cm.upsert_both(sym_norm, rows)
            except Exception as e:
                print(f"{sym}: upsert error - {e}")
            else:
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
    except KeyboardInterrupt as exc:  # pragma: no cover - 手動中断
        interrupt_exc = exc
    if interrupt_exc is not None:
        if progress_callback and total and total != last_report:
            effective_target = max(progress_target, total)
            try:
                progress_callback(total, effective_target, updated)
            except Exception:
                pass
        raise CacheUpdateInterrupted(total, updated) from interrupt_exc
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


def main():
    if not API_KEY:
        print("EODHD_API_KEY が未設定です (.env を確認)", flush=True)
        return
    print("🚀 EODHD Bulk Last Day 更新を開始します...", flush=True)
    print("📡 API リクエストを送信中...", flush=True)
    data = fetch_bulk_last_day()
    if data is None or data.empty:
        print("No data to update.", flush=True)
        return

    settings = get_settings(create_dirs=True)

    universe_error = False
    try:
        universe = build_symbol_universe_from_settings(settings)
    except Exception as exc:
        print(
            "⚠️ 銘柄ユニバース取得に失敗したためフィルタリングをスキップします:"
            f" {exc}",
            flush=True,
        )
        universe = []
        universe_error = True

    data, filter_stats = _filter_bulk_data_by_universe(data, universe)
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
                "  ↳ Bulk データ一致:"
                f" {matched_symbols} 銘柄 / {matched_rows} 行",
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
    elif not universe_error:
        print(
            "⚠️ 銘柄ユニバース取得結果が空だったためフィルタリングをスキップします",
            flush=True,
        )

    if data.empty:
        print("⚠️ フィルタ後の対象データが 0 件のため処理を終了します。", flush=True)
        return

    original_count, normalized_count = _estimate_symbol_counts(data)
    estimated_symbols = max(original_count, normalized_count)
    if estimated_symbols:
        print(
            f"📊 取得件数: {len(data)} 行 / 銘柄数(推定): {estimated_symbols}",
            flush=True,
        )
    else:
        print(f"📊 取得件数: {len(data)} 行", flush=True)

    cm = CacheManager(settings)

    progress_state = {
        "processed": 0,
        "total": estimated_symbols,
        "updated": 0,
    }

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
        total, updated = append_to_cache(
            data,
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
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"✅ {now} | 対象: {total} 銘柄 / 更新: {updated} 銘柄（full/rolling へ反映）",
        flush=True,
    )


if __name__ == "__main__":
    main()
