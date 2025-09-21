from __future__ import annotations

"""Stage metrics tracking shared between the CLI runner and Streamlit UI.

This module centralizes the bookkeeping for per-system stage progress metrics
and stage completion events.  The runner (``run_all_systems_today``) records
each progress update which is then consumed by the Streamlit UI via
``StageTracker``.  Keeping the state in a dedicated module allows both sides to
stay in sync even when the process pool is used or when the UI attaches after
processing has started.
"""

from dataclasses import dataclass
from threading import Lock
from collections import deque
from typing import Deque, Dict, Iterable


DEFAULT_SYSTEM_ORDER = tuple(f"system{i}" for i in range(1, 8))


def _normalize_count(value: object | None) -> int | None:
    """Safely coerce the incoming counter value to ``int`` when possible."""

    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return None


@dataclass(slots=True)
class StageEvent:
    """A single stage progress notification for a system."""

    system: str
    progress: int
    filter_count: int | None = None
    setup_count: int | None = None
    candidate_count: int | None = None
    entry_count: int | None = None

    def as_tuple(
        self,
    ) -> tuple[str, int, int | None, int | None, int | None, int | None]:
        """Return a tuple representation matching the legacy queue payload."""

        return (
            self.system,
            self.progress,
            self.filter_count,
            self.setup_count,
            self.candidate_count,
            self.entry_count,
        )


@dataclass(slots=True)
class StageSnapshot:
    """Aggregated metrics for a system across the end-to-end pipeline."""

    progress: int = 0
    target: int | None = None
    filter_pass: int | None = None
    setup_pass: int | None = None
    candidate_count: int | None = None
    entry_count: int | None = None
    exit_count: int | None = None

    def copy(self) -> "StageSnapshot":
        """Return a shallow copy to avoid leaking internal references."""

        return StageSnapshot(
            progress=self.progress,
            target=self.target,
            filter_pass=self.filter_pass,
            setup_pass=self.setup_pass,
            candidate_count=self.candidate_count,
            entry_count=self.entry_count,
            exit_count=self.exit_count,
        )


class StageMetricsStore:
    """Thread-safe store for stage progress snapshots and queued events."""

    _DISPLAY_KEYS = ("target", "filter", "setup", "cand", "entry", "exit")

    def __init__(self, system_order: Iterable[str] | None = None) -> None:
        self._snapshots: Dict[str, StageSnapshot] = {}
        self._events: Deque[StageEvent] = deque()
        self._lock = Lock()
        self._universe_target: int | None = None
        self.stage_counts: Dict[str, dict[str, int | None]] = {}
        self._display_order: list[str] = []

        if system_order is not None:
            for name in system_order:
                key = self._normalize_system_name(name)
                if key not in self._display_order:
                    self._display_order.append(key)
                if key not in self.stage_counts:
                    self.stage_counts[key] = self._new_display_bucket()

    # ------------------------------------------------------------------
    # lifecycle helpers
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear snapshots, queued events and universe target state."""

        with self._lock:
            self._snapshots.clear()
            self._events.clear()
            self._universe_target = None

    def drain_events(self) -> list[StageEvent]:
        """Return and clear all queued stage events."""

        with self._lock:
            if not self._events:
                return []
            events = list(self._events)
            self._events.clear()
        return events

    # ------------------------------------------------------------------
    # recorders
    # ------------------------------------------------------------------
    def record_stage(
        self,
        system: str,
        progress: int,
        filter_count: object | None = None,
        setup_count: object | None = None,
        candidate_count: object | None = None,
        entry_count: object | None = None,
        *,
        emit_event: bool = True,
    ) -> StageSnapshot:
        """Update a system snapshot and optionally queue a stage event."""

        system_key = self._normalize_system_name(system)

        try:
            progress_int = int(progress)
        except Exception:
            progress_int = 0
        progress_int = max(0, min(100, progress_int))

        filter_int = _normalize_count(filter_count)
        setup_int = _normalize_count(setup_count)
        candidate_int = _normalize_count(candidate_count)
        entry_int = _normalize_count(entry_count)

        with self._lock:
            snapshot = self._snapshots.setdefault(system_key, StageSnapshot())
            snapshot.progress = max(snapshot.progress, progress_int)

            if filter_int is not None:
                if progress_int == 0:
                    snapshot.target = filter_int
                    if self._universe_target is None:
                        self._universe_target = filter_int
                else:
                    snapshot.filter_pass = filter_int
            if setup_int is not None:
                snapshot.setup_pass = setup_int
            if candidate_int is not None:
                snapshot.candidate_count = candidate_int
            if entry_int is not None:
                snapshot.entry_count = entry_int

            event = StageEvent(
                system_key,
                snapshot.progress,
                filter_int,
                setup_int,
                candidate_int,
                entry_int,
            )
            if emit_event:
                self._events.append(event)

            return snapshot.copy()

    def record_exit(
        self, system: str, exit_count: object | None, *, emit_event: bool = False
    ) -> StageSnapshot:
        """Update the exit count for a system and return the snapshot."""

        system_key = self._normalize_system_name(system)
        exit_int = _normalize_count(exit_count)

        with self._lock:
            snapshot = self._snapshots.setdefault(system_key, StageSnapshot())
            snapshot.exit_count = exit_int
            # Exit は UI 表示のみであり現在はイベントキューに乗せない
            return snapshot.copy()

    # ------------------------------------------------------------------
    # accessors
    # ------------------------------------------------------------------
    def get_snapshot(self, system: str) -> StageSnapshot | None:
        """Return a copy of the snapshot for ``system`` if present."""

        system_key = self._normalize_system_name(system)
        with self._lock:
            snapshot = self._snapshots.get(system_key)
            return snapshot.copy() if snapshot is not None else None

    def all_snapshots(self) -> dict[str, StageSnapshot]:
        """Return a shallow copy of all stored snapshots."""

        with self._lock:
            return {name: snap.copy() for name, snap in self._snapshots.items()}

    def set_universe_target(self, target: object | None) -> None:
        """Persist the shared universe target (Tgt) display value."""

        if target is None:
            with self._lock:
                self._universe_target = None
            return
        normalized = _normalize_count(target)
        with self._lock:
            self._universe_target = normalized

    def get_universe_target(self) -> int | None:
        """Return the globally shared universe target if available."""

        with self._lock:
            return self._universe_target

    # ------------------------------------------------------------------
    # display helpers for Streamlit UI
    # ------------------------------------------------------------------
    def ensure_display_metrics(self, system: object) -> dict[str, int | None]:
        """Return a mutable metrics bucket for ``system`` creating it on demand."""

        system_key = self._normalize_system_name(system)
        with self._lock:
            bucket = self._ensure_display_bucket_locked(system_key)
        return bucket

    def get_display_metrics(self, system: object) -> dict[str, int | None]:
        """Return a copy of the display metrics for ``system``."""

        system_key = self._normalize_system_name(system)
        with self._lock:
            bucket = self._ensure_display_bucket_locked(system_key)
            return dict(bucket)

    def systems(self) -> list[str]:
        """Return systems known to the display store preserving configured order."""

        with self._lock:
            ordered = list(self._display_order)
            extras = [name for name in self.stage_counts if name not in ordered]
        extras.sort()
        return ordered + extras

    @staticmethod
    def clamp_trdlist(value: object | None) -> int | None:
        """Normalize the TRDlist count for display (non-negative, bounded)."""

        if value is None:
            return None
        try:
            count = int(value)
        except Exception:
            try:
                count = int(float(value))
            except Exception:
                return None
        if count < 0:
            return 0
        return min(count, 9999)

    @staticmethod
    def _normalize_system_name(system: object) -> str:
        try:
            system_key = str(system or "").strip().lower()
        except Exception:
            system_key = ""
        return system_key or "unknown"

    def _new_display_bucket(self) -> dict[str, int | None]:
        return {key: None for key in self._DISPLAY_KEYS}

    def _ensure_display_bucket_locked(
        self, system_key: str
    ) -> dict[str, int | None]:
        bucket = self.stage_counts.get(system_key)
        if bucket is None:
            bucket = self._new_display_bucket()
            self.stage_counts[system_key] = bucket
        if system_key not in self._display_order:
            self._display_order.append(system_key)
        return bucket


GLOBAL_STAGE_METRICS = StageMetricsStore()

__all__ = [
    "DEFAULT_SYSTEM_ORDER",
    "GLOBAL_STAGE_METRICS",
    "StageEvent",
    "StageMetricsStore",
    "StageSnapshot",
]
