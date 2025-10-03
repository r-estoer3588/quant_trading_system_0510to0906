from __future__ import annotations

"""Lightweight performance snapshot utility.

Collects:
- total_time_sec
- per_system: {system_name: {elapsed_sec, symbol_count}}
- cache_io: {read_feather, read_csv, write_feather, write_csv}
- latest_only (bool)
- timestamp (ISO8601 UTC)

Usage:
    from common.perf_snapshot import PerfSnapshot
    ps = PerfSnapshot(enabled=args.perf_snapshot)
    with ps.run(latest_only=latest_only_flag):
        # run pipeline; within pipeline call ps.mark_system_start(name), ps.mark_system_end(name, symbols)
    # automatically writes file if enabled

CacheManager hooks should call increment_cache_io(kind) where kind in
('read_feather','read_csv','write_feather','write_csv'). We avoid introducing
imports inside CacheManager to prevent cycles; instead we do a local import and
fail silently if module not present.
"""

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from config.settings import get_settings

# Schema versioning: increment when JSON structure or semantics changes
SCHEMA_VERSION = 2  # v1: initial (no candidate_count), v2: candidate_count + version

_CACHE_IO_COUNTER = {
    "read_feather": 0,
    "read_csv": 0,
    "write_feather": 0,
    "write_csv": 0,
}


def increment_cache_io(kind: str) -> None:
    """Increment a cache IO counter.

    Called from CacheManager after each successful physical read/write.
    Unknown kinds are ignored to keep it robust.
    """
    if kind in _CACHE_IO_COUNTER:
        _CACHE_IO_COUNTER[kind] += 1


@dataclass
class _SystemEntry:
    start_ts: float
    end_ts: float | None = None
    symbol_count: int | None = None
    candidate_count: int | None = None

    def elapsed(self) -> float | None:
        if self.end_ts is None:
            return None
        return self.end_ts - self.start_ts


@dataclass
class PerfSnapshot:
    enabled: bool = False
    _start_ts: float | None = None
    _end_ts: float | None = None
    latest_only: bool | None = None
    per_system: Dict[str, _SystemEntry] = field(default_factory=dict)
    output_path: Path | None = None

    def mark_system_start(self, name: str) -> None:
        if not self.enabled:
            return
        if name not in self.per_system:
            self.per_system[name] = _SystemEntry(start_ts=time.perf_counter())

    def mark_system_end(
        self,
        name: str,
        symbol_count: int | None = None,
        candidate_count: int | None = None,
    ) -> None:
        if not self.enabled:
            return
        entry = self.per_system.get(name)
        if not entry:
            # auto-create if start missed
            entry = _SystemEntry(start_ts=time.perf_counter())
            self.per_system[name] = entry
        if entry.end_ts is None:
            entry.end_ts = time.perf_counter()
        if symbol_count is not None:
            entry.symbol_count = symbol_count
        if candidate_count is not None:
            entry.candidate_count = candidate_count

    @contextmanager
    def run(self, latest_only: bool | None = None):  # type: ignore[override]
        if not self.enabled:
            yield self
            return
        # --- reset state for a fresh run ---
        self.per_system.clear()
        for k in list(_CACHE_IO_COUNTER.keys()):
            _CACHE_IO_COUNTER[k] = 0
        self.latest_only = latest_only
        self._start_ts = time.perf_counter()
        try:
            yield self
        finally:
            self._end_ts = time.perf_counter()
            try:
                self._finalize_and_write()
            except Exception:
                pass

    # -- internal helpers --------------------------------------------------
    def _snapshot_dict(self) -> dict:
        total_time = None
        if self._start_ts is not None and self._end_ts is not None:
            total_time = self._end_ts - self._start_ts
        systems_payload = {}
        for k, v in self.per_system.items():
            systems_payload[k] = {
                "elapsed_sec": round(v.elapsed() or 0.0, 6),
                "symbol_count": v.symbol_count,
                "candidate_count": v.candidate_count,
            }
        payload = {
            "schema_version": SCHEMA_VERSION,
            "timestamp": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
            "latest_only": self.latest_only,
            "total_time_sec": round(total_time or 0.0, 6),
            "per_system": systems_payload,
            "cache_io": dict(_CACHE_IO_COUNTER),
        }
        return payload

    def _finalize_and_write(self) -> None:
        settings = get_settings(create_dirs=True)
        log_dir = Path(settings.LOGS_DIR) / "perf_snapshots"
        log_dir.mkdir(parents=True, exist_ok=True)
        flag = "latest" if self.latest_only else "full"
        # Date granularity to seconds to avoid overwrite when multiple runs per day
        stamp = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
        path = log_dir / f"perf_{stamp}_{flag}.json"
        payload = self._snapshot_dict()
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self.output_path = path


# Convenience accessor for external modules -------------------------------
_global_perf: PerfSnapshot | None = None


def get_global_perf() -> PerfSnapshot | None:
    return _global_perf


def enable_global_perf(enabled: bool) -> PerfSnapshot:
    global _global_perf  # noqa: PLW0603 (intentional simple module-level singleton)
    _global_perf = PerfSnapshot(enabled=enabled)
    return _global_perf
