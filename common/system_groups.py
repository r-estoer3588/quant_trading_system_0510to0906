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
