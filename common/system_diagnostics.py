"""Shared helpers for per-system diagnostics snapshots.

This module centralises the logic for computing cross-system diagnostics values
such as filter/setup pass counts and final candidate counts.  Strategies can use
``build_system_diagnostics`` immediately after candidate generation to capture
consistent metrics for UI / logging.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SystemDiagnostics:
    """A structured container for system-wide diagnostic data."""

    data: dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def merge(self, other: dict[str, Any]) -> None:
        self.data.update(other)

    def to_dict(self) -> dict[str, Any]:
        return self.data.copy()


RowPredicate = Callable[[pd.Series], bool]

# --- Diagnostics keys (shared across systems) ---
DIAG_RANKING_SOURCE = "ranking_source"
DIAG_SETUP_PRED_COUNT = "setup_predicate_count"
# Ranked result count after TopN ranking (single source of truth)
DIAG_RANKED_TOPN_COUNT = "ranked_top_n_count"
DIAG_PRED_ONLY_PASS = "predicate_only_pass_count"
DIAG_MISMATCH_FLAG = "mismatch_flag"

# System1 specific fallback counters (maintain for compatibility)
DIAG_S1_COUNT_A = "count_a"
DIAG_S1_COUNT_B = "count_b"
DIAG_S1_COUNT_C = "count_c"
DIAG_S1_COUNT_D = "count_d"
DIAG_S1_COUNT_E = "count_e"
DIAG_S1_COUNT_F = "count_f"

# --- Triage categories ---
TRIAGE_EXACT_MATCH = "exact_match"
TRIAGE_RANKING_FILTERED = "ranking_filtered"
TRIAGE_ZERO_SETUP = "zero_setup"
TRIAGE_UNEXPECTED = "unexpected"


def _coerce_bool(value: Any) -> bool:
    """Coerce arbitrary values to boolean in a NaN/None-safe way.

    Keep behavior identical to the previous implementation while reducing the
    number of return statements to satisfy static analysis guidance for
    readability and maintainability.
    """
    if isinstance(value, bool):
        return value

    result = False
    try:
        if value is None:
            result = False
        elif isinstance(value, (int, float)):
            v = float(value)
            result = (not math.isnan(v)) and (not math.isinf(v)) and (v != 0.0)
        else:
            text = str(value).strip().lower()
            if not text:
                result = False
            elif text in {"1", "true", "yes", "on"}:
                result = True
            elif text in {"0", "false", "no", "off"}:
                result = False
            else:
                result = bool(value)
    except Exception:
        result = False

    return result


def numeric_greater_than(column: str, threshold: float) -> RowPredicate:
    """Return predicate that checks ``column > threshold`` (NaN safe)."""

    def _predicate(row: pd.Series) -> bool:
        try:
            value = row.get(column)
            if value is None:
                return False
            val = float(value)
            if math.isnan(val):
                return False
            return val > threshold
        except Exception:
            return False

    return _predicate


def numeric_greater_equal(column: str, threshold: float) -> RowPredicate:
    """Return predicate that checks ``column >= threshold`` (NaN safe)."""

    def _predicate(row: pd.Series) -> bool:
        try:
            value = row.get(column)
            if value is None:
                return False
            val = float(value)
            if math.isnan(val):
                return False
            return val >= threshold
        except Exception:
            return False

    return _predicate


def numeric_is_finite(column: str) -> RowPredicate:
    """Return predicate that is true when the column holds a finite number."""

    def _predicate(row: pd.Series) -> bool:
        try:
            value = row.get(column)
            if value is None:
                return False
            val = float(value)
            return not math.isnan(val) and not math.isinf(val)
        except Exception:
            return False

    return _predicate


@dataclass(slots=True)
class SystemDiagnosticSpec:
    """Specification for per-system diagnostics collection."""

    filter_key: str | None = "filter"
    setup_key: str | None = "setup"
    filter_predicate: RowPredicate | None = None
    setup_predicate: RowPredicate | None = None
    rank_predicate: RowPredicate | None = None
    extra_predicates: dict[str, RowPredicate] = field(default_factory=dict)
    rank_metric_name: str | None = None
    mode: str = "latest"


def _evaluate(
    row: pd.Series, *, key: str | None, predicate: RowPredicate | None
) -> bool:
    if predicate is not None:
        try:
            return bool(predicate(row))
        except Exception:
            return False
    if key is None:
        return False
    try:
        return _coerce_bool(row.get(key))
    except Exception:
        return False


def build_system_diagnostics(
    system_name: str,
    prepared_dict: Mapping[str, pd.DataFrame] | None,
    candidates_by_date: Mapping[Any, Mapping[str, Any]] | None,
    *,
    top_n: int | None,
    spec: SystemDiagnosticSpec | None = None,
    base_payload: Mapping[str, Any] | None = None,
    latest_only: bool = True,
) -> dict[str, Any]:
    """Compose a diagnostics payload for ``system_name``.

    Args:
        system_name: Target system identifier (e.g. ``"system1"``).
        prepared_dict: Dictionary of prepared per-symbol DataFrames.
        candidates_by_date: Resulting candidate map produced by the system.
        top_n: Ranking cap used for generation.
        spec: Optional system-specific predicates.
        base_payload: Existing diagnostics that should be merged.
        latest_only: Whether this run represents the latest-only fast path.

    Returns:
        Dictionary with normalized diagnostics counters.
    """

    spec = spec or SystemDiagnosticSpec()
    payload: dict[str, Any] = dict(base_payload or {})

    payload["system"] = system_name
    payload["mode"] = spec.mode if spec.mode else ("latest" if latest_only else "full")

    if top_n is not None:
        try:
            payload["top_n"] = int(top_n)
        except Exception:
            pass

    total_symbols = len(prepared_dict) if prepared_dict else 0
    payload["total_symbols"] = total_symbols
    payload["symbols_total"] = total_symbols

    symbols_with_data = 0
    filter_pass = 0
    setup_pass = 0
    rank_pass = 0
    extra_counts: dict[str, int] = {}
    latest_rows: list[pd.Series] = []

    if prepared_dict:
        for df in prepared_dict.values():
            if isinstance(df, pd.DataFrame) and not df.empty:
                symbols_with_data += 1
                row = df.iloc[-1]
                latest_rows.append(row)
                if _evaluate(row, key=spec.filter_key, predicate=spec.filter_predicate):
                    filter_pass += 1
                if _evaluate(row, key=spec.setup_key, predicate=spec.setup_predicate):
                    setup_pass += 1
                if spec.rank_predicate is not None and spec.rank_predicate(row):
                    rank_pass += 1
            else:
                continue

    payload["symbols_with_data"] = symbols_with_data
    payload["filter_pass"] = filter_pass
    payload["filter_pass_count"] = filter_pass
    payload["setup_flag_true"] = setup_pass
    payload["setup_true"] = setup_pass
    payload["setup_true_count"] = setup_pass
    payload["rank_metric_valid"] = rank_pass
    payload["rank_metric_valid_count"] = rank_pass

    final_count = 0
    if candidates_by_date:
        for symbol_map in candidates_by_date.values():
            if isinstance(symbol_map, Mapping):
                final_count += sum(1 for _ in symbol_map.keys())
    payload["final_pass"] = final_count
    # Single source of truth: write only ranked_top_n_count
    payload["ranked_top_n_count"] = final_count

    if spec.rank_metric_name:
        payload["rank_metric_name"] = spec.rank_metric_name

    if spec.extra_predicates and latest_rows:
        for label, predicate in spec.extra_predicates.items():
            try:
                extra_counts[label] = sum(1 for row in latest_rows if predicate(row))
            except Exception:
                extra_counts[label] = 0
        if extra_counts:
            existing_extra = payload.get("extra_counts")
            if isinstance(existing_extra, Mapping):
                merged = dict(existing_extra)
                merged.update(extra_counts)
                payload["extra_counts"] = merged
            else:
                payload["extra_counts"] = extra_counts

    return payload


def get_diagnostics_with_fallback(diag: dict | None, system_id: str) -> dict:
    """Diagnostics が欠損している場合にデフォルト値を返す。

    Args:
        diag: 元の diagnostics 辞書（None 可）
        system_id: システム ID（ログ用）

    Returns:
        統一キーを含む辞書（欠損時はデフォルト値）
    """
    if diag is None or not isinstance(diag, dict):
        # Use lazy formatting to avoid building strings when the level is disabled
        try:
            logger.warning(
                "%s: diagnostics is None or invalid, using fallback", system_id
            )
        except Exception:
            pass
        diag = {}

    # Read the unified key only
    ranked_val = int(diag.get(DIAG_RANKED_TOPN_COUNT, -1))

    return {
        DIAG_RANKING_SOURCE: diag.get(DIAG_RANKING_SOURCE, "unknown"),
        DIAG_SETUP_PRED_COUNT: int(diag.get(DIAG_SETUP_PRED_COUNT, -1)),
        DIAG_RANKED_TOPN_COUNT: ranked_val,
        # Keep legacy field in normalized view for transitional compatibility
        "final_top_n_count": ranked_val,
        DIAG_PRED_ONLY_PASS: int(diag.get(DIAG_PRED_ONLY_PASS, -1)),
        DIAG_MISMATCH_FLAG: bool(diag.get(DIAG_MISMATCH_FLAG, False)),
        # System1 専用キー（他システムでは -1 でフォールバック）
        DIAG_S1_COUNT_A: int(diag.get(DIAG_S1_COUNT_A, -1)),
        DIAG_S1_COUNT_B: int(diag.get(DIAG_S1_COUNT_B, -1)),
        DIAG_S1_COUNT_C: int(diag.get(DIAG_S1_COUNT_C, -1)),
        DIAG_S1_COUNT_D: int(diag.get(DIAG_S1_COUNT_D, -1)),
        DIAG_S1_COUNT_E: int(diag.get(DIAG_S1_COUNT_E, -1)),
        DIAG_S1_COUNT_F: int(diag.get(DIAG_S1_COUNT_F, -1)),
    }


def triage_candidate_discrepancy(diag: dict[str, Any]) -> dict[str, Any]:
    """Setup 通過数と最終候補数の差分を分類。

    分類カテゴリ:
    - "exact_match": setup_count == final_count（理想的な状態）
    - "ranking_filtered": setup_count > final_count（ランキングで絞り込み）
    - "zero_setup": setup_count == 0（フィルタで全滅）
    - "unexpected": その他（要調査）

    Args:
        diag: システム診断情報の辞書
            - setup_predicate_count: Setup 条件通過数
            - ranked_top_n_count: 最終候補数（ランキング後）

    Returns:
        トリアージ結果の辞書:
        {
            "category": str,
            "setup_count": int,
            "final_count": int,
            "diff": int,
            "message": str
        }

    Example:
        >>> # 完全一致のケース
    >>> diag = {"setup_predicate_count": 5, "ranked_top_n_count": 5}
        >>> result = triage_candidate_discrepancy(diag)
        >>> result["category"]
        'exact_match'
    """
    setup_count = int(diag.get("setup_predicate_count", 0))
    # Single key: ranked_top_n_count
    try:
        final_count = int(diag.get("ranked_top_n_count", 0))
    except Exception:
        final_count = 0
    diff = setup_count - final_count

    if setup_count == final_count:
        category = TRIAGE_EXACT_MATCH
        message = f"Setup {setup_count} == Final {final_count}"
    elif setup_count > final_count >= 0:
        category = TRIAGE_RANKING_FILTERED
        message = f"Setup {setup_count} → Final {final_count} (filtered {diff})"
    elif setup_count == 0:
        category = TRIAGE_ZERO_SETUP
        message = "No candidates passed setup"
    else:
        category = TRIAGE_UNEXPECTED
        message = f"⚠️ Setup {setup_count} vs Final {final_count} (unexpected)"

    return {
        "category": category,
        "setup_count": setup_count,
        "final_count": final_count,
        "diff": diff,
        "message": message,
    }


def triage_all_systems(
    system_diagnostics: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """複数システムの診断情報を一括トリアージ。

    Args:
        system_diagnostics: {system_id: 診断情報} の辞書

    Returns:
        {system_id: トリアージ結果} の辞書
    """
    results = {}

    for system_id, diag in system_diagnostics.items():
        triage_result = triage_candidate_discrepancy(diag)
        # システム ID を結果に追加
        triage_result["system_id"] = system_id
        results[system_id] = triage_result

    return results


def format_triage_summary(
    triage_results: dict[str, dict[str, Any]],
) -> str:
    """トリアージ結果をサマリー形式でフォーマット。

    Args:
        triage_results: triage_all_systems() の戻り値

    Returns:
        整形されたサマリー文字列
    """
    lines = ["=== Candidate Discrepancy Triage Summary ==="]

    # カテゴリ別に集計
    category_counts: dict[str, int] = {}
    for result in triage_results.values():
        category = result.get("category", "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1

    # システム別の結果を出力
    for system_id, result in sorted(triage_results.items()):
        category = result.get("category", "unknown")
        message = result.get("message", "")

        # カテゴリごとにアイコンを変更
        if category == TRIAGE_EXACT_MATCH:
            icon = "✓"
        elif category == TRIAGE_RANKING_FILTERED:
            icon = "→"
        elif category == TRIAGE_ZERO_SETUP:
            icon = "∅"
        else:  # unexpected
            icon = "⚠️"

        lines.append(f"{icon} [{system_id}] {message}")

    # カテゴリ別サマリー
    lines.append("")
    lines.append("Category Summary:")
    for category, count in sorted(category_counts.items()):
        lines.append(f"  {category}: {count} system(s)")

    return "\n".join(lines)


def get_unexpected_systems(
    triage_results: dict[str, dict[str, Any]],
) -> list[str]:
    """Unexpected カテゴリのシステム ID リストを取得。

    Args:
        triage_results: triage_all_systems() の戻り値

    Returns:
        Unexpected カテゴリに分類されたシステム ID のリスト
    """
    unexpected = []

    for system_id, result in triage_results.items():
        if result.get("category") == TRIAGE_UNEXPECTED:
            unexpected.append(system_id)

    return sorted(unexpected)
