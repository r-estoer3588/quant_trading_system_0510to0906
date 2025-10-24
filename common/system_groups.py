"""Utilities for grouping systems into long/short buckets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

SYSTEM_SIDE_GROUPS: dict[str, tuple[str, ...]] = {
    "long": ("system1", "system3", "system5"),
    "short": ("system2", "system4", "system6", "system7"),
}

# 明示的に表示順を制御する（long → short）。
GROUP_ORDER: tuple[str, ...] = tuple(SYSTEM_SIDE_GROUPS.keys())

GROUP_DISPLAY_NAMES: dict[str, str] = {
    "long": "Long (System1,3,5)",
    "short": "Short (System2,4,6,7)",
}

# システムラベルの正規化対応表
SYSTEM_LABELS: dict[str, str] = {
    "system1": "System1",
    "system2": "System2",
    "system3": "System3",
    "system4": "System4",
    "system5": "System5",
    "system6": "System6",
    "system7": "System7",
}


def _normalize_system_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return name.strip().lower()


def _format_label(name: str) -> str:
    if name in GROUP_DISPLAY_NAMES:
        return GROUP_DISPLAY_NAMES[name]
    if name.startswith("system") and name[6:].isdigit():
        return f"System{name[6:]}"
    if name == "others":
        return "その他"
    return name


def _normalize_counts(counts: Mapping[str, Any]) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for key, value in counts.items():
        norm_key = _normalize_system_name(key)
        if not norm_key:
            continue
        try:
            normalized[norm_key] = normalized.get(norm_key, 0) + int(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _normalize_values(values: Mapping[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in values.items():
        norm_key = _normalize_system_name(key)
        if not norm_key:
            continue
        try:
            normalized[norm_key] = normalized.get(norm_key, 0.0) + float(value)
        except (TypeError, ValueError):
            continue
    return normalized


def summarize_group_totals(
    counts: Mapping[str, Any],
    values: Mapping[str, Any] | None = None,
) -> list[tuple[str, int, float | None]]:
    normalized_counts = _normalize_counts(counts)
    normalized_values = _normalize_values(values) if values is not None else {}
    summary: list[tuple[str, int, float | None]] = []
    used: set[str] = set()

    for group_key in GROUP_ORDER:
        members: Iterable[str] = SYSTEM_SIDE_GROUPS.get(group_key, ())
        total_count = 0
        total_value = 0.0
        for member in members:
            member_norm = _normalize_system_name(member)
            total_count += int(normalized_counts.get(member_norm, 0))
            total_value += float(normalized_values.get(member_norm, 0.0))
            used.add(member_norm)
        summary.append(
            (
                group_key,
                total_count,
                total_value if values is not None else None,
            )
        )

    for key in sorted(normalized_counts.keys()):
        if key in used:
            continue
        total_value = (
            float(normalized_values.get(key, 0.0)) if values is not None else None
        )
        summary.append((key, int(normalized_counts[key]), total_value))

    return summary


def format_group_counts(counts: Mapping[str, Any]) -> list[str]:
    summary = summarize_group_totals(counts)
    return [f"{_format_label(key)}={count}" for key, count, _ in summary]


def format_group_counts_and_values(
    counts: Mapping[str, Any],
    values: Mapping[str, Any],
) -> list[str]:
    summary = summarize_group_totals(counts, values)
    lines: list[str] = []
    for key, count, total_value in summary:
        if total_value is None:
            lines.append(f"{_format_label(key)}: {count}件")
        else:
            lines.append(f"{_format_label(key)}: {count}件 / ${total_value:,.0f}")
    return lines


def format_cache_coverage_report(
    total_symbols: int,
    available_count: int,
    missing_count: int,
    coverage_percentage: float,
    missing_symbols: list[str],
) -> dict[str, Any]:
    """
    rolling cache分析結果を見やすい形式でフォーマットする。

    Args:
        total_symbols: 分析対象シンボル総数
        available_count: rolling cache整備済みシンボル数
        missing_count: rolling cache未整備シンボル数
        coverage_percentage: カバレッジ率
        missing_symbols: 未整備シンボルのリスト

    Returns:
        フォーマット済み分析結果辞書
    """
    # カバレッジ状況の判定
    if coverage_percentage >= 90:
        status = "✅ 良好"
        priority = "低"
    elif coverage_percentage >= 70:
        status = "⚠️ 要改善"
        priority = "中"
    else:
        status = "🚨 緊急"
        priority = "高"

    # 未整備シンボルのサマリー作成（最大10件表示）
    missing_summary = []
    if missing_symbols:
        shown_symbols = missing_symbols[:10]
        missing_summary = shown_symbols
        if len(missing_symbols) > 10:
            missing_summary.append(f"... 他{len(missing_symbols) - 10}シンボル")

    return {
        "status": status,
        "priority": priority,
        "summary": {
            "total": total_symbols,
            "available": available_count,
            "missing": missing_count,
            "coverage": f"{coverage_percentage:.1f}%",
        },
        "missing_symbols_preview": missing_summary,
        "recommendations": _generate_cache_recommendations(
            coverage_percentage, missing_count
        ),
    }


def _generate_cache_recommendations(coverage: float, missing_count: int) -> list[str]:
    """カバレッジ率に基づいて推奨アクションを生成する。"""
    recommendations = []

    if coverage < 50:
        recommendations.append("🔥 緊急: 基盤となるrolling cacheの構築が必要です")
        recommendations.append(
            "📋 アクション: scripts/run_all_systems_today.py実行でrolling cache自動生成"
        )

    elif coverage < 70:
        recommendations.append("⚡ 重要: rolling cache整備率を向上させる必要があります")
        recommendations.append(
            "🔧 確認: cache_daily_data.pyによる日次データ更新の実行状況"
        )

    elif coverage < 90:
        recommendations.append("📈 改善: 残り未整備シンボルの対応を推奨します")

    else:
        recommendations.append("🎉 excellent: rolling cache整備状況は良好です")

    if missing_count > 0:
        recommendations.append(
            f"📊 詳細: 未整備{missing_count}シンボルの個別確認を推奨"
        )

    return recommendations


def analyze_system_symbols_coverage(
    system_symbols_map: dict[str, list[str]], cache_analysis_results: dict
) -> dict[str, Any]:
    """
    システム別のrolling cache整備状況を分析する。

    Args:
        system_symbols_map: システム名をキーとするシンボルリストのマップ
        cache_analysis_results: CacheManager.analyze_rolling_gaps()の結果

    Returns:
        システム別カバレッジ分析結果
    """
    missing_symbols = set(cache_analysis_results.get("missing_symbols", []))
    system_coverage = {}

    for system_name, symbols in system_symbols_map.items():
        if not symbols:
            continue

        system_missing = [s for s in symbols if s in missing_symbols]
        total = len(symbols)
        missing_count = len(system_missing)
        available = total - missing_count
        coverage = (available / total * 100) if total > 0 else 0

        system_coverage[system_name] = {
            "total_symbols": total,
            "available": available,
            "missing": missing_count,
            "coverage_percentage": coverage,
            "missing_symbols": system_missing,
            "status": "✅" if coverage >= 90 else "⚠️" if coverage >= 70 else "🚨",
        }

    # グループ別サマリー
    group_summary = {}
    for group_name, system_list in SYSTEM_SIDE_GROUPS.items():
        group_total = 0
        group_available = 0
        group_missing = []

        for system in system_list:
            if system in system_coverage:
                stats = system_coverage[system]
                group_total += stats["total_symbols"]
                group_available += stats["available"]
                group_missing.extend(stats["missing_symbols"])

        group_coverage = (group_available / group_total * 100) if group_total > 0 else 0
        group_summary[group_name] = {
            "total_symbols": group_total,
            "available": group_available,
            "missing": len(group_missing),
            "coverage_percentage": group_coverage,
            "status": (
                "✅" if group_coverage >= 90 else "⚠️" if group_coverage >= 70 else "🚨"
            ),
        }

    return {
        "by_system": system_coverage,
        "by_group": group_summary,
        "overall": cache_analysis_results,
    }
