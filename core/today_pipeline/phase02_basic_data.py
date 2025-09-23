"""Phase 2: load foundational price data for today's signal pipeline."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import os
from threading import Lock
from typing import Any

import pandas as pd

from common.cache_manager import CacheManager, load_base_cache
from config.settings import get_settings

try:  # pragma: no cover - optional dependency
    from common.utils_spy import get_latest_nyse_trading_day
except Exception:  # pragma: no cover

    def get_latest_nyse_trading_day(today: pd.Timestamp | None = None) -> pd.Timestamp:  # type: ignore[override]
        if today is None:
            return pd.Timestamp.now().normalize()
        try:
            return pd.Timestamp(today).normalize()
        except Exception:
            return pd.Timestamp.now().normalize()


# --- constants --------------------------------------------------------------------


@dataclass(frozen=True)
class RequiredColumns:
    """Container for rolling cache validation requirements."""

    required: tuple[str, ...] = (
        "date",
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
        "atr14",
        "atr40",
        "roc200",
    )
    important: tuple[str, ...] = (
        "ema20",
        "ema50",
        "atr10",
        "atr20",
        "atr50",
        "adx7",
        "rsi3",
        "rsi14",
        "hv50",
        "return6d",
        "drop3d",
    )
    nan_threshold: float = 0.20
    recent_window: int = 120
    recent_strict_window: int = 30
    recent_strict_threshold: float = 0.0


REQUIRED_COLUMNS = RequiredColumns()


def _has_recent_valid_window(
    numeric: pd.Series,
    *,
    window: int = REQUIRED_COLUMNS.recent_window,
    nan_threshold: float = REQUIRED_COLUMNS.nan_threshold,
    strict_window: int = REQUIRED_COLUMNS.recent_strict_window,
    strict_threshold: float = REQUIRED_COLUMNS.recent_strict_threshold,
) -> bool:
    """Return True if the trailing rows contain enough non-NaN values."""

    if numeric.empty:
        return False

    recent_len = int(min(len(numeric), window))
    if recent_len <= 0:
        return False
    recent = numeric.iloc[-recent_len:]
    try:
        recent_ratio = float(recent.isna().mean())
    except Exception:
        recent_ratio = 1.0
    if recent_ratio <= nan_threshold:
        return True

    strict_len = int(min(len(numeric), strict_window))
    if strict_len <= 0:
        return False
    strict_recent = recent.iloc[-strict_len:]
    try:
        strict_ratio = float(strict_recent.isna().mean())
    except Exception:
        strict_ratio = 1.0
    return strict_ratio <= strict_threshold


@dataclass(slots=True)
class MissingDetail:
    """Describe remediation result for a symbol whose rolling cache needed work."""

    symbol: str
    status: str
    missing_required: str = ""
    missing_optional: str = ""
    nan_columns: str = ""
    rows_before: int = 0
    rows_after: int = 0
    action: str = ""
    resolved: bool = False
    note: str = ""


@dataclass(slots=True)
class BasicDataLoadResult:
    """Outcome for the basic data loading phase."""

    data: dict[str, pd.DataFrame]
    missing_details: list[MissingDetail]
    stats: dict[str, int]
    base_cache: dict[str, pd.DataFrame] | None = None


# --- cache helpers ----------------------------------------------------------------


@dataclass(slots=True)
class BaseCachePool:
    """Shared base cache dictionary used to reduce repeated IO."""

    cache_manager: CacheManager
    shared: dict[str, pd.DataFrame] | None = None
    hits: int = 0
    loads: int = 0
    failures: int = 0
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:  # pragma: no cover - simple guard
        if self.shared is None:
            self.shared = {}

    def get(
        self,
        symbol: str,
        *,
        rebuild_if_missing: bool = True,
        min_last_date: pd.Timestamp | None = None,
        allowed_recent_dates: Iterable[pd.Timestamp] | None = None,
    ) -> tuple[pd.DataFrame | None, bool]:
        allowed_set = {pd.Timestamp(d).normalize() for d in allowed_recent_dates or ()}
        min_norm: pd.Timestamp | None = None
        if min_last_date is not None:
            try:
                min_norm = pd.Timestamp(min_last_date).normalize()
            except Exception:  # pragma: no cover - defensive
                min_norm = None

        if self.shared is not None:
            with self._lock:
                cached = self.shared.get(symbol)
            if cached is not None:
                last = _extract_last_cache_date(cached)
                stale = False
                if allowed_set and (last is None or last not in allowed_set):
                    stale = True
                if not stale and min_norm is not None and (last is None or last < min_norm):
                    stale = True
                if not stale:
                    self.hits += 1
                    return cached, True
                with self._lock:
                    self.shared.pop(symbol, None)

        frame = load_base_cache(
            symbol,
            rebuild_if_missing=rebuild_if_missing,
            cache_manager=self.cache_manager,
            min_last_date=min_norm,
            allowed_recent_dates=allowed_set or None,
        )
        if frame is None or getattr(frame, "empty", True):
            self.failures += 1
        if frame is not None and self.shared is not None:
            with self._lock:
                self.shared[symbol] = frame
        self.loads += 1
        return frame, False

    def sync_to(self, target: dict[str, pd.DataFrame] | None) -> None:
        if target is None or self.shared is None or target is self.shared:
            return
        with self._lock:
            target.update(self.shared)

    def snapshot_stats(self) -> dict[str, int]:
        with self._lock:
            size = len(self.shared or {})
            return {
                "hits": self.hits,
                "loads": self.loads,
                "failures": self.failures,
                "size": size,
            }


# --- utilities --------------------------------------------------------------------


def _extract_last_cache_date(df: pd.DataFrame | None) -> pd.Timestamp | None:
    if df is None or getattr(df, "empty", True):
        return None
    for col in ("date", "Date"):
        if col in df.columns:
            try:
                values = pd.to_datetime(df[col].to_numpy(), errors="coerce")
                values = values.dropna()
                if not values.empty:
                    try:
                        last_val = values[-1]
                    except Exception:
                        # fallback to list indexing
                        last_val = list(values)[-1]
                    return pd.Timestamp(last_val).normalize()
            except Exception:
                continue
    try:
        idx = pd.to_datetime(df.index.to_numpy(), errors="coerce")
        mask = ~pd.isna(idx)
        if mask.any():
            return pd.Timestamp(idx[mask][-1]).normalize()
    except Exception:
        pass
    return None


def _recent_trading_days(today: pd.Timestamp | None, max_back: int) -> list[pd.Timestamp]:
    if today is None:
        return []
    out: list[pd.Timestamp] = []
    seen: set[pd.Timestamp] = set()
    current = pd.Timestamp(today).normalize()
    steps = max(0, int(max_back))
    for _ in range(steps + 1):
        if current in seen:
            break
        out.append(current)
        seen.add(current)
        prev_candidate = get_latest_nyse_trading_day(current - pd.Timedelta(days=1))
        prev_candidate = pd.Timestamp(prev_candidate).normalize()
        if prev_candidate == current:
            break
        current = prev_candidate
    return out


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "adj_close": "AdjClose",
        "adjusted_close": "AdjClose",
    }
    try:
        return df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    except Exception:  # pragma: no cover
        return df


def _normalize_loaded(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or getattr(df, "empty", True):
        return None
    try:
        if "Date" not in df.columns:
            work = df.copy()
            if "date" in work.columns:
                work["Date"] = pd.to_datetime(work["date"].to_numpy(), errors="coerce")
            else:
                work["Date"] = pd.to_datetime(df.index.to_numpy(), errors="coerce")
            df = work
        df["Date"] = pd.to_datetime(df["Date"].to_numpy(), errors="coerce").normalize()
    except Exception:
        pass
    normalized = _normalize_ohlcv(df)
    try:
        fill_cols = [
            c for c in ("Open", "High", "Low", "Close", "Volume") if c in normalized.columns
        ]
        if fill_cols:
            normalized = normalized.copy()
            try:
                filled = normalized[fill_cols].apply(pd.to_numeric, errors="coerce")
            except Exception:
                filled = normalized[fill_cols]
            normalized.loc[:, fill_cols] = filled.ffill().bfill()
    except Exception:
        pass
    try:
        if "Date" in normalized.columns:
            normalized = normalized.dropna(subset=["Date"])
    except Exception:
        pass
    return normalized


def build_rolling_from_base(
    symbol: str,
    base_df: pd.DataFrame,
    target_len: int,
    cache_manager: CacheManager | None = None,
) -> pd.DataFrame | None:
    if base_df is None or getattr(base_df, "empty", True):
        return None
    try:
        work = base_df.copy()
    except Exception:
        work = base_df
    if work.index.name is not None:
        work = work.reset_index()
    if "Date" in work.columns:
        work["date"] = pd.to_datetime(work["Date"].to_numpy(), errors="coerce")
    elif "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"].to_numpy(), errors="coerce")
    else:
        return None
    work = work.dropna(subset=["date"]).sort_values("date")
    col_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "AdjClose": "adjusted_close",
        "Adj Close": "adjusted_close",
        "Volume": "volume",
    }
    for src, dst in list(col_map.items()):
        if src in work.columns:
            work = work.rename(columns={src: dst})
    sliced = work.tail(int(target_len)).reset_index(drop=True)
    if sliced.empty:
        return None
    if cache_manager is not None:
        try:
            cache_manager.write_atomic(sliced, symbol, "rolling")
        except Exception:  # pragma: no cover
            pass
    return sliced


def analyze_rolling_frame(df: pd.DataFrame | None) -> tuple[bool, dict[str, Any]]:
    if df is None or getattr(df, "empty", True):
        return False, {"status": "rolling_missing"}
    try:
        columns = list(df.columns)
    except Exception:
        columns = []
    col_map = {str(col).lower(): col for col in columns}
    missing_required = [c for c in REQUIRED_COLUMNS.required if c not in col_map]
    missing_optional = [c for c in REQUIRED_COLUMNS.important if c not in col_map]
    nan_required: list[tuple[str, float]] = []
    nan_optional: list[tuple[str, float]] = []
    for name in set(REQUIRED_COLUMNS.required).union(REQUIRED_COLUMNS.important):
        actual = col_map.get(name)
        if actual is None:
            continue
        try:
            numeric = pd.to_numeric(df[actual], errors="coerce")
        except Exception:
            continue
        try:
            ratio = float(numeric.isna().mean())
        except Exception:
            continue
        if ratio > REQUIRED_COLUMNS.nan_threshold:
            if name in REQUIRED_COLUMNS.required:
                nan_required.append((name, ratio))
            else:
                nan_optional.append((name, ratio))
    issues: dict[str, Any] = {}
    fatal = False
    if missing_required:
        issues["missing_required"] = missing_required
        fatal = True
    if nan_required or nan_optional:
        issues["nan_columns"] = [*nan_required, *nan_optional]
    if nan_required:
        fatal = True
    if missing_optional:
        issues["missing_optional"] = missing_optional
    if fatal:
        issues.setdefault("status", "missing_required" if missing_required else "nan_columns")
        return False, issues
    if missing_optional:
        issues.setdefault("status", "missing_optional")
        return True, issues
    if nan_optional:
        issues.setdefault("status", "nan_optional")
        return True, issues
    return True, {}


def _format_nan_columns(values: Iterable[tuple[str, float]]) -> str:
    return ", ".join(f"{name}:{ratio:.1%}" for name, ratio in values)


def _build_missing_detail(symbol: str, issues: dict[str, Any], rows_before: int) -> MissingDetail:
    missing_required = issues.get("missing_required") or []
    missing_optional = issues.get("missing_optional") or []
    nan_columns = issues.get("nan_columns") or []
    return MissingDetail(
        symbol=symbol,
        status=str(issues.get("status", "missing")),
        missing_required=", ".join(str(x) for x in missing_required),
        missing_optional=", ".join(str(x) for x in missing_optional),
        nan_columns=_format_nan_columns(list(nan_columns)),
        rows_before=int(rows_before),
    )


def _issues_to_note(issues: dict[str, Any]) -> str:
    if not issues:
        return ""
    parts: list[str] = []
    missing_required = issues.get("missing_required") or []
    if missing_required:
        parts.append("required=" + ", ".join(str(x) for x in missing_required))
    missing_optional = issues.get("missing_optional") or []
    if missing_optional:
        parts.append("optional=" + ", ".join(str(x) for x in missing_optional))
    nan_columns = issues.get("nan_columns") or []
    if nan_columns:
        parts.append("nan=" + _format_nan_columns(list(nan_columns)))
    return "; ".join(parts)


def _merge_note(base: str, addition: str) -> str:
    parts = [part for part in [base, addition] if part]
    return " / ".join(parts)


def load_basic_data_phase(
    symbols: Iterable[str],
    *,
    cache_manager: CacheManager | None = None,
    settings: Any | None = None,
    symbol_data: dict[str, pd.DataFrame] | None = None,
    base_cache: dict[str, pd.DataFrame] | None = None,
    today: pd.Timestamp | None = None,
    freshness_tolerance: int | None = None,
    parallel: bool | None = None,
    max_workers: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    log: Callable[[str], None] | None = None,
) -> BasicDataLoadResult:
    """Load rolling cache data for the prepared universe and ensure coverage."""

    symbols = list(dict.fromkeys(symbols))
    total_syms = len(symbols)

    if settings is None:
        settings = get_settings(create_dirs=False)
    cache_manager = cache_manager or CacheManager(settings)

    if freshness_tolerance is None:
        try:
            freshness_tolerance = int(settings.cache.rolling.max_staleness_days)
        except Exception:
            freshness_tolerance = 2
    freshness_tolerance = max(0, int(freshness_tolerance))

    try:
        target_len = int(
            settings.cache.rolling.base_lookback_days + settings.cache.rolling.buffer_days
        )
    except Exception:
        target_len = 0

    stats_lock = Lock()
    stats: dict[str, int] = {}

    def _record_stat(key: str) -> None:
        with stats_lock:
            stats[key] = stats.get(key, 0) + 1

    recent_allowed: set[pd.Timestamp] = set()
    if today is not None and freshness_tolerance >= 0:
        try:
            recent_allowed = {
                pd.Timestamp(d).normalize()
                for d in _recent_trading_days(pd.Timestamp(today), freshness_tolerance)
            }
        except Exception:
            recent_allowed = set()

    # compute min of recent_allowed if present (value not used directly but keep logic clear)
    if recent_allowed:
        try:
            _ = min(recent_allowed)
        except Exception:
            pass

    gap_probe_days = max(freshness_tolerance + 5, 10)

    def _estimate_gap_days(
        today_dt: pd.Timestamp | None, last_dt: pd.Timestamp | None
    ) -> int | None:
        if today_dt is None or last_dt is None:
            return None
        try:
            recent = _recent_trading_days(pd.Timestamp(today_dt), gap_probe_days)
        except Exception:
            recent = []
        for offset, dt in enumerate(recent):
            if dt == last_dt:
                return offset
        try:
            return max(0, int((pd.Timestamp(today_dt) - pd.Timestamp(last_dt)).days))
        except Exception:
            return None

    def _pick_symbol_data(sym: str) -> pd.DataFrame | None:
        if not symbol_data or sym not in symbol_data:
            return None
        df = symbol_data.get(sym)
        if df is None or getattr(df, "empty", True):
            return None
        try:
            x = df.copy()
        except Exception:
            x = df
        if x.index.name is not None:
            x = x.reset_index()
        if "date" in x.columns:
            x["date"] = pd.to_datetime(x["date"].to_numpy(), errors="coerce")
        elif "Date" in x.columns:
            x["date"] = pd.to_datetime(x["Date"].to_numpy(), errors="coerce")
        else:
            return None
        col_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adjusted_close",
            "AdjClose": "adjusted_close",
            "Volume": "volume",
        }
        for src, dst in list(col_map.items()):
            if src in x.columns:
                x = x.rename(columns={src: dst})
        if "date" not in x.columns or "close" not in x.columns:
            return None
        return x.dropna(subset=["date"]).sort_values("date")

    missing_details: list[MissingDetail] = []

    def _load_one(sym: str) -> tuple[str, pd.DataFrame | None, MissingDetail | None]:
        source: str | None = None
        df = _pick_symbol_data(sym)
        rebuild_reason: str | None = None
        last_seen_date: pd.Timestamp | None = None
        gap_days: int | None = None
        issues: dict[str, Any] | None = None
        detail: MissingDetail | None = None
        rows_before = 0 if df is None else int(len(df))

        if df is None or getattr(df, "empty", True):
            try:
                df = cache_manager.read(sym, "rolling")
            except Exception:
                df = None
        else:
            source = "prefetched"
        if df is None or getattr(df, "empty", True):
            source = None

        needs_rebuild = False
        if df is None or getattr(df, "empty", True):
            needs_rebuild = True
        if df is not None and not getattr(df, "empty", True) and source is None:
            source = "rolling"
        if df is not None and not getattr(df, "empty", True):
            last_seen_date = _extract_last_cache_date(df)
            if last_seen_date is None:
                rebuild_reason = rebuild_reason or "missing_date"
                needs_rebuild = True
            else:
                last_seen_date = pd.Timestamp(last_seen_date).normalize()
                if today is not None and recent_allowed and last_seen_date not in recent_allowed:
                    rebuild_reason = "stale"
                    gap_days = _estimate_gap_days(pd.Timestamp(today), last_seen_date)
                    needs_rebuild = True

        normalized: pd.DataFrame | None = None
        if not needs_rebuild:
            normalized = _normalize_loaded(df)
            if normalized is None or getattr(normalized, "empty", True):
                needs_rebuild = True
                issues = {"status": rebuild_reason or "normalize_failed"}
            else:
                ok, detected = analyze_rolling_frame(normalized)
                if not ok:
                    needs_rebuild = True
                    issues = detected
        else:
            issues = {"status": rebuild_reason or "missing"}

        if needs_rebuild:
            if issues is None:
                issues = {"status": rebuild_reason or "missing"}
            detail = _build_missing_detail(sym, issues, rows_before)
            note_reason = rebuild_reason or issues.get("status", "missing")
            reason_map = {
                "stale": "鮮度不足",
                "missing_date": "日付欠損",
                "length": "行数不足",
            }
            reason_label = reason_map.get(note_reason, "未整備")
            if detail is not None:
                detail.action = "manual_rebuild_required"
                detail.resolved = False
                detail.note = _merge_note(detail.note, "manual_rebuild_required")
            skip_parts: list[str] = []
            if rebuild_reason == "stale":
                gap_label = f"約{gap_days}営業日" if gap_days is not None else "不明"
                last_label = str(last_seen_date.date()) if last_seen_date is not None else "不明"
                skip_parts.append(f"最終日={last_label}")
                skip_parts.append(f"ギャップ={gap_label}")
            elif rebuild_reason == "length":
                skip_parts.append(f"len={rows_before}/{target_len}")
            elif rebuild_reason == "missing_date":
                skip_parts.append("date列欠損")
            elif df is None or getattr(df, "empty", True):
                skip_parts.append("rolling未生成")
            if log is not None:
                msg = f"⛔ rolling未整備: {sym} ({reason_label})"
                if skip_parts:
                    msg += " | " + ", ".join(skip_parts)
                msg += " → 手動で rolling キャッシュを更新してください"
                try:
                    log(msg)
                except Exception:  # pragma: no cover - defensive
                    pass
            _record_stat("manual_rebuild_required")
            _record_stat("failed")
            return sym, None, detail

        ok_after, issues_after = analyze_rolling_frame(normalized)
        if issues_after and detail is not None:
            detail.missing_required = ", ".join(
                str(x) for x in issues_after.get("missing_required", [])
            )
            detail.missing_optional = ", ".join(
                str(x) for x in issues_after.get("missing_optional", [])
            )
            detail.nan_columns = _format_nan_columns(issues_after.get("nan_columns", []))
            detail.note = _merge_note(detail.note, _issues_to_note(issues_after))
        if ok_after:
            if detail is not None:
                detail.rows_after = int(len(normalized))
                detail.resolved = True
            _record_stat(source or "rolling")
            return sym, normalized, detail
        if detail is not None and normalized is not None:
            detail.rows_after = int(len(normalized))
        _record_stat("failed")
        return sym, None, detail

    def _run_parallel() -> dict[str, pd.DataFrame]:
        data: dict[str, pd.DataFrame] = {}
        if total_syms == 0:
            return data
        nonlocal max_workers
        if max_workers is None:
            try:
                cfg_workers = getattr(settings.cache.rolling, "load_max_workers", None)
                if cfg_workers:
                    max_workers = int(cfg_workers)
            except Exception:
                max_workers = None
        if max_workers is None:
            cpu_count = max(1, (os.cpu_count() or 4))
            max_workers = max(4, cpu_count * 2)
        max_workers_local = max(1, min(int(max_workers), total_syms))
        with ThreadPoolExecutor(max_workers=max_workers_local) as executor:
            futures = {executor.submit(_load_one, sym): sym for sym in symbols}
            processed = 0
            for fut in as_completed(futures):
                sym, df, detail = fut.result()
                if df is not None and not getattr(df, "empty", True):
                    data[sym] = df
                if detail is not None:
                    missing_details.append(detail)
                processed += 1
                if progress_callback:
                    try:
                        progress_callback(processed, total_syms)
                    except Exception:  # pragma: no cover
                        pass
        return data

    def _run_sequential() -> dict[str, pd.DataFrame]:
        data: dict[str, pd.DataFrame] = {}
        processed = 0
        for sym in symbols:
            sym, df, detail = _load_one(sym)
            if df is not None and not getattr(df, "empty", True):
                data[sym] = df
            if detail is not None:
                missing_details.append(detail)
            processed += 1
            if progress_callback:
                try:
                    progress_callback(processed, total_syms)
                except Exception:  # pragma: no cover
                    pass
        return data

    if parallel is None:
        env_parallel = (os.environ.get("BASIC_DATA_PARALLEL", "") or "").strip().lower()
        if env_parallel in ("1", "true", "yes"):
            parallel = total_syms > 1
        elif env_parallel in ("0", "false", "no"):
            parallel = False
        else:
            try:
                threshold = int(os.environ.get("BASIC_DATA_PARALLEL_THRESHOLD", "200"))
            except Exception:
                threshold = 200
            parallel = total_syms >= max(0, threshold)

    data = _run_parallel() if parallel else _run_sequential()

    try:
        summary_map = {
            "prefetched": "事前供給",
            "rolling": "rolling再利用",
            "manual_rebuild_required": "手動対応",
            "failed": "失敗",
        }
        summary_parts = [
            f"{label}={stats.get(key, 0)}" for key, label in summary_map.items() if stats.get(key)
        ]
        if summary_parts:
            if log is not None:
                try:
                    log("📊 基礎データロード内訳: " + " / ".join(summary_parts))
                except Exception:
                    pass
    except Exception:
        pass

    return BasicDataLoadResult(
        data=data,
        missing_details=missing_details,
        stats=stats,
        base_cache=base_cache,
    )
