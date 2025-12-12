"""Stage Reporter - Progress reporting for UI integration.

Extracted from run_all_systems_today.py for better modularity.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from common.stage_metrics import GLOBAL_STAGE_METRICS

# Module-level state for stage callbacks
_PER_SYSTEM_STAGE: Callable[..., None] | None = None
_PER_SYSTEM_EXIT: Callable[[str, int], None] | None = None
_SET_STAGE_UNIVERSE_TARGET: Callable[[int | None], None] | None = None

_STAGE_EVENT_PUMP_THREAD: threading.Thread | None = None
_STAGE_EVENT_PUMP_STOP: threading.Event | None = None
_STAGE_EVENT_PUMP_INTERVAL = 0.25

# Adaptive interval settings
_STAGE_EVENT_PUMP_ADAPTIVE = True
_STAGE_EVENT_PUMP_MIN_INTERVAL = 0.1
_STAGE_EVENT_PUMP_MAX_INTERVAL = 1.0
_STAGE_EVENT_PUMP_IDLE_THRESHOLD = 5


def _stage(
    system: str,
    progress: int,
    filter_count: int | None = None,
    setup_count: int | None = None,
    candidate_count: int | None = None,
    entry_count: int | None = None,
) -> None:
    """Internal stage update function."""
    cb = _PER_SYSTEM_STAGE
    if callable(cb):
        try:
            cb(
                system,
                progress,
                filter_count,
                setup_count,
                candidate_count,
                entry_count,
            )
        except Exception:
            pass


class StageReporter:
    """Callable wrapper that forwards stage progress with system name."""

    __slots__ = ("system", "_queue")

    def __init__(self, system: str, queue: Any | None = None) -> None:
        self.system = str(system or "").strip().lower() or "unknown"
        self._queue = queue

    def __call__(
        self,
        progress: int,
        filter_count: int | None = None,
        setup_count: int | None = None,
        candidate_count: int | None = None,
        entry_count: int | None = None,
    ) -> None:
        if self._queue is not None:
            try:
                self._queue.put(
                    (
                        self.system,
                        progress,
                        filter_count,
                        setup_count,
                        candidate_count,
                        entry_count,
                    ),
                    block=False,
                )
            except Exception:
                pass
            return
        _stage(
            self.system,
            progress,
            filter_count,
            setup_count,
            candidate_count,
            entry_count,
        )


def register_stage_callback(callback: Callable[..., None] | None) -> None:
    """Register per-system stage callback and ensure event pump is running."""
    global _PER_SYSTEM_STAGE
    _PER_SYSTEM_STAGE = callback
    if callable(callback):
        _ensure_stage_event_pump()
    else:
        _stop_stage_event_pump()


def register_stage_exit_callback(callback: Callable[[str, int], None] | None) -> None:
    """Register per-system exit callback (UI integration helper)."""
    global _PER_SYSTEM_EXIT
    _PER_SYSTEM_EXIT = callback


def register_universe_target_callback(
    callback: Callable[[int | None], None] | None,
) -> None:
    """Register callback to update shared universe target in the UI."""
    global _SET_STAGE_UNIVERSE_TARGET
    _SET_STAGE_UNIVERSE_TARGET = callback


def _drain_stage_event_queue() -> None:
    """Drain pending stage events from the global metrics queue."""
    try:
        events = GLOBAL_STAGE_METRICS.drain_events()
        for event in events:
            cb = _PER_SYSTEM_STAGE
            if callable(cb):
                try:
                    cb(
                        event.system,
                        event.progress,
                        event.filter_count,
                        event.setup_count,
                        event.candidate_count,
                        event.entry_count,
                    )
                except Exception:
                    pass
    except Exception:
        pass


def _ensure_stage_event_pump(interval: float | None = None) -> None:
    """Start background thread that periodically drains stage events."""
    global _STAGE_EVENT_PUMP_THREAD, _STAGE_EVENT_PUMP_STOP

    cb = _PER_SYSTEM_STAGE
    if not cb or not callable(cb):
        return

    if isinstance(_STAGE_EVENT_PUMP_THREAD, threading.Thread):
        if _STAGE_EVENT_PUMP_THREAD.is_alive():
            return

    stop_event = threading.Event()
    _STAGE_EVENT_PUMP_STOP = stop_event

    base_interval = float(
        interval if interval is not None else _STAGE_EVENT_PUMP_INTERVAL
    )

    def _pump() -> None:
        current_interval = base_interval
        idle_count = 0

        while not stop_event.is_set():
            events_processed = False
            try:
                _drain_stage_event_queue()
                try:
                    metrics_events = len(GLOBAL_STAGE_METRICS.drain_events())
                    if metrics_events > 0:
                        events_processed = True
                except Exception:
                    pass

                if _STAGE_EVENT_PUMP_ADAPTIVE:
                    if events_processed:
                        current_interval = max(
                            _STAGE_EVENT_PUMP_MIN_INTERVAL, current_interval * 0.8
                        )
                        idle_count = 0
                    else:
                        idle_count += 1
                        if idle_count >= _STAGE_EVENT_PUMP_IDLE_THRESHOLD:
                            current_interval = min(
                                _STAGE_EVENT_PUMP_MAX_INTERVAL, current_interval * 1.2
                            )

            except Exception:
                pass

            stop_event.wait(current_interval)

    pump_thread = threading.Thread(target=_pump, name="stage-event-pump", daemon=True)
    _STAGE_EVENT_PUMP_THREAD = pump_thread
    pump_thread.start()


def _stop_stage_event_pump(timeout: float = 1.0) -> None:
    """Stop background event pump thread if running."""
    global _STAGE_EVENT_PUMP_STOP, _STAGE_EVENT_PUMP_THREAD

    if isinstance(_STAGE_EVENT_PUMP_STOP, threading.Event):
        _STAGE_EVENT_PUMP_STOP.set()

    if isinstance(_STAGE_EVENT_PUMP_THREAD, threading.Thread):
        if _STAGE_EVENT_PUMP_THREAD.is_alive():
            if threading.current_thread() is not _STAGE_EVENT_PUMP_THREAD:
                _STAGE_EVENT_PUMP_THREAD.join(timeout)

    _STAGE_EVENT_PUMP_STOP = None
    _STAGE_EVENT_PUMP_THREAD = None
