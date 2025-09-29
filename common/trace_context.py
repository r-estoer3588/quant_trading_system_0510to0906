"""Phase-based trace ID system using ContextVar.

This module provides:
1. ContextVar-based trace ID management for 8-phase processing
2. Phase context tracking and hierarchy
3. Integration with structured logging
4. Thread-safe trace ID propagation
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ProcessingPhase(Enum):
    """8-phase processing phases."""

    SYMBOLS = "symbols"
    LOAD = "load"
    SHARED_INDICATORS = "shared_indicators"
    FILTERS = "filters"
    SETUP = "setup"
    SIGNALS = "signals"
    ALLOCATION = "allocation"
    SAVE_NOTIFY = "save_notify"

    @property
    def display_name(self) -> str:
        """Human-readable phase name."""
        names = {
            self.SYMBOLS: "Symbol Universe",
            self.LOAD: "Data Loading",
            self.SHARED_INDICATORS: "Shared Indicators",
            self.FILTERS: "Filter Conditions",
            self.SETUP: "Setup Conditions",
            self.SIGNALS: "Signal Generation",
            self.ALLOCATION: "Position Allocation",
            self.SAVE_NOTIFY: "Save & Notify",
        }
        return names.get(self, self.value.title())


@dataclass
class TraceContext:
    """Trace context information."""

    trace_id: str
    parent_trace_id: Optional[str] = None
    phase: Optional[ProcessingPhase] = None
    system: Optional[str] = None
    symbol: Optional[str] = None
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def create_child(
        self,
        phase: Optional[ProcessingPhase] = None,
        system: Optional[str] = None,
        symbol: Optional[str] = None,
        **metadata,
    ) -> TraceContext:
        """Create child trace context."""
        return TraceContext(
            trace_id=generate_trace_id(),
            parent_trace_id=self.trace_id,
            phase=phase or self.phase,
            system=system or self.system,
            symbol=symbol or self.symbol,
            metadata={**self.metadata, **metadata},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "trace_id": self.trace_id,
            "parent_trace_id": self.parent_trace_id,
            "phase": self.phase.value if self.phase else None,
            "system": self.system,
            "symbol": self.symbol,
            "start_time": self.start_time,
            "metadata": self.metadata,
        }


# ContextVar for trace context
_trace_context: ContextVar[Optional[TraceContext]] = ContextVar(
    "trace_context", default=None
)

# ContextVar for phase hierarchy
_phase_stack: ContextVar[List[ProcessingPhase]] = ContextVar("phase_stack", default=[])


def generate_trace_id() -> str:
    """Generate a unique trace ID."""
    return str(uuid.uuid4())[:8]  # Short ID for readability


def get_current_trace_context() -> Optional[TraceContext]:
    """Get current trace context."""
    return _trace_context.get()


def get_current_trace_id() -> Optional[str]:
    """Get current trace ID."""
    context = get_current_trace_context()
    return context.trace_id if context else None


def get_current_phase() -> Optional[ProcessingPhase]:
    """Get current processing phase."""
    context = get_current_trace_context()
    return context.phase if context else None


def get_phase_stack() -> List[ProcessingPhase]:
    """Get current phase stack."""
    return _phase_stack.get().copy()


@contextmanager
def trace_context(
    phase: Optional[ProcessingPhase] = None,
    system: Optional[str] = None,
    symbol: Optional[str] = None,
    trace_id: Optional[str] = None,
    **metadata,
):
    """Context manager for trace context."""
    current_context = get_current_trace_context()

    if current_context and not trace_id:
        # Create child context
        new_context = current_context.create_child(
            phase=phase, system=system, symbol=symbol, **metadata
        )
    else:
        # Create new root context
        new_context = TraceContext(
            trace_id=trace_id or generate_trace_id(),
            phase=phase,
            system=system,
            symbol=symbol,
            metadata=metadata,
        )

    # Update phase stack
    current_stack = get_phase_stack()
    if phase and (not current_stack or current_stack[-1] != phase):
        new_stack = current_stack + [phase]
    else:
        new_stack = current_stack

    # Set context vars
    token_context = _trace_context.set(new_context)
    token_stack = _phase_stack.set(new_stack)

    try:
        yield new_context
    finally:
        _trace_context.reset(token_context)
        _phase_stack.reset(token_stack)


@contextmanager
def phase_context(
    phase: ProcessingPhase,
    system: Optional[str] = None,
    symbol: Optional[str] = None,
    **metadata,
):
    """Context manager specifically for phase tracking."""
    with trace_context(phase=phase, system=system, symbol=symbol, **metadata) as ctx:
        yield ctx


def run_with_trace(func, *args, **kwargs):
    """Run function with current trace context copied."""
    ctx = copy_context()
    return ctx.run(func, *args, **kwargs)


# Integration with structured logging
def get_trace_info_for_logging() -> Dict[str, Any]:
    """Get trace information for logging."""
    context = get_current_trace_context()
    if not context:
        return {}

    return {
        "trace_id": context.trace_id,
        "parent_trace_id": context.parent_trace_id,
        "phase": context.phase.value if context.phase else None,
        "system": context.system,
        "symbol": context.symbol,
        "phase_stack": [p.value for p in get_phase_stack()],
    }


# Phase progress tracking
@dataclass
class PhaseProgress:
    """Track progress within a phase."""

    phase: ProcessingPhase
    total_items: int
    completed_items: int = 0
    failed_items: int = 0
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    current_item: Optional[str] = None

    @property
    def progress_pct(self) -> float:
        """Calculate progress percentage."""
        if self.total_items == 0:
            return 100.0
        return (self.completed_items / self.total_items) * 100.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total_processed = self.completed_items + self.failed_items
        if total_processed == 0:
            return 100.0
        return (self.completed_items / total_processed) * 100.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "phase": self.phase.value,
            "total_items": self.total_items,
            "completed_items": self.completed_items,
            "failed_items": self.failed_items,
            "progress_pct": self.progress_pct,
            "success_rate": self.success_rate,
            "start_time": self.start_time,
            "current_item": self.current_item,
        }


# Global phase progress tracking
_phase_progress: ContextVar[Optional[PhaseProgress]] = ContextVar(
    "phase_progress", default=None
)


def start_phase_progress(phase: ProcessingPhase, total_items: int) -> PhaseProgress:
    """Start tracking progress for a phase."""
    progress = PhaseProgress(phase=phase, total_items=total_items)
    _phase_progress.set(progress)
    return progress


def update_phase_progress(
    completed: Optional[int] = None,
    failed: Optional[int] = None,
    current_item: Optional[str] = None,
) -> Optional[PhaseProgress]:
    """Update current phase progress."""
    progress = _phase_progress.get()
    if not progress:
        return None

    if completed is not None:
        progress.completed_items = completed
    if failed is not None:
        progress.failed_items = failed
    if current_item is not None:
        progress.current_item = current_item

    return progress


def get_phase_progress() -> Optional[PhaseProgress]:
    """Get current phase progress."""
    return _phase_progress.get()


# Utility functions for 8-phase integration
def create_run_trace_id() -> str:
    """Create a trace ID for an entire run."""
    return f"run_{generate_trace_id()}"


@contextmanager
def trading_run_context(run_id: Optional[str] = None, **metadata):
    """Context for an entire trading run."""
    with trace_context(trace_id=run_id or create_run_trace_id(), **metadata) as ctx:
        yield ctx


def log_phase_transition(
    from_phase: Optional[ProcessingPhase], to_phase: ProcessingPhase
):
    """Log phase transition with trace context."""
    from common.structured_logging import get_trading_logger

    logger = get_trading_logger()
    trace_info = get_trace_info_for_logging()

    logger.get_logger("phase_transition").info(
        f"Phase transition: {from_phase.value if from_phase else 'None'} -> {to_phase.value}",
        extra={
            "event_type": "phase_transition",
            "from_phase": from_phase.value if from_phase else None,
            "to_phase": to_phase.value,
            **trace_info,
        },
    )


# Phase timing utilities
@dataclass
class PhaseTimer:
    """Track timing for phases."""

    phase: ProcessingPhase
    start_time: float
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """Get duration in seconds."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "phase": self.phase.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "metadata": self.metadata,
        }


_phase_timers: ContextVar[Dict[ProcessingPhase, PhaseTimer]] = ContextVar(
    "phase_timers", default={}
)


@contextmanager
def timed_phase(phase: ProcessingPhase, **metadata):
    """Context manager for timing phases."""
    import time

    start_time = time.perf_counter()
    timer = PhaseTimer(phase=phase, start_time=start_time, metadata=metadata)

    # Store timer
    timers = _phase_timers.get().copy()
    timers[phase] = timer
    token = _phase_timers.set(timers)

    try:
        with phase_context(phase, **metadata):
            yield timer
    finally:
        timer.end_time = time.perf_counter()
        _phase_timers.reset(token)

        # Log timing
        from common.structured_logging import get_trading_logger

        logger = get_trading_logger()
        logger.log_performance_event(
            f"phase_{phase.value}",
            timer.duration,
            {**metadata, **get_trace_info_for_logging()},
        )


def get_phase_timers() -> Dict[ProcessingPhase, PhaseTimer]:
    """Get all phase timers."""
    return _phase_timers.get().copy()
