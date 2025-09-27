"""Enhanced structured logging system for the quant trading system.

This module provides comprehensive logging infrastructure with structured
JSON logs, performance metrics, and monitoring capabilities.
"""

import json
import logging
import os
import sys
import threading
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import psutil


@dataclass
class LogEvent:
    """Structured log event."""

    timestamp: str
    level: str
    logger: str
    message: str
    module: str = ""
    function: str = ""
    line: int = 0
    thread_id: int = 0
    process_id: int = 0
    system: str = "quant_trading"
    version: str = "1.0"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class MetricEvent:
    """Performance metric event."""

    timestamp: str
    metric_name: str
    value: float
    unit: str = "count"
    tags: Dict[str, str] = field(default_factory=dict)
    system: str = "quant_trading"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Extract frame information
        frame_info = traceback.extract_stack()[-10:]  # Last 10 frames
        caller_frame = None

        for frame in reversed(frame_info):
            if (
                "logging" not in frame.filename
                and "structured_logging" not in frame.filename
            ):
                caller_frame = frame
                break

        event = LogEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=record.levelname,
            logger=record.name,
            message=record.getMessage(),
            module=caller_frame.filename.split("/")[-1] if caller_frame else "",
            function=caller_frame.name if caller_frame else "",
            line=caller_frame.lineno if caller_frame else 0,
            thread_id=threading.get_ident(),
            process_id=os.getpid(),
            extra=getattr(record, "extra", {}),
        )

        return json.dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":"))


class MetricsCollector:
    """Thread-safe metrics collection."""

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path("logs/metrics")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._metrics: Dict[str, list] = defaultdict(list)
        self._timers: Dict[str, float] = {}
        self._counters: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

        # Setup metrics logger
        self.logger = logging.getLogger("metrics")
        handler = logging.FileHandler(self.log_dir / "metrics.jsonl")
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def record_metric(
        self, name: str, value: float, unit: str = "count", **tags
    ) -> None:
        """Record a metric value."""
        with self._lock:
            event = MetricEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                metric_name=name,
                value=value,
                unit=unit,
                tags=tags,
            )

            self._metrics[name].append(event)
            self.logger.info(json.dumps(event.to_dict()))

    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter metric."""
        with self._lock:
            self._counters[name] += value
            self.record_metric(name, self._counters[name], "count")

    @contextmanager
    def timer(self, name: str, **tags):
        """Context manager for timing operations."""
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_delta = end_memory - start_memory

            self.record_metric(f"{name}_duration", elapsed, "seconds", **tags)
            self.record_metric(f"{name}_memory_delta", memory_delta, "MB", **tags)

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "metrics_count": len(self._metrics),
                "total_events": sum(len(events) for events in self._metrics.values()),
            }


class TradingSystemLogger:
    """Enhanced logging system for the trading application."""

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = MetricsCollector(self.log_dir)
        self._setup_loggers()

    def _setup_loggers(self) -> None:
        """Setup structured loggers."""
        # Main application logger
        self.app_logger = logging.getLogger("quant_trading")
        self.app_logger.setLevel(logging.INFO)

        # JSON file handler
        json_handler = logging.FileHandler(self.log_dir / "application.jsonl")
        json_handler.setFormatter(StructuredFormatter())
        self.app_logger.addHandler(json_handler)

        # Console handler for development
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self.app_logger.addHandler(console_handler)

        # Performance logger
        self.perf_logger = logging.getLogger("performance")
        perf_handler = logging.FileHandler(self.log_dir / "performance.jsonl")
        perf_handler.setFormatter(StructuredFormatter())
        self.perf_logger.addHandler(perf_handler)
        self.perf_logger.setLevel(logging.INFO)

        # Error logger
        self.error_logger = logging.getLogger("errors")
        error_handler = logging.FileHandler(self.log_dir / "errors.jsonl")
        error_handler.setFormatter(StructuredFormatter())
        self.error_logger.addHandler(error_handler)
        self.error_logger.setLevel(logging.WARNING)

    def get_logger(self, name: str, **context) -> logging.Logger:
        """Get a logger with context."""
        logger = logging.getLogger(f"quant_trading.{name}")

        # Add context to all log records
        if context:

            class ContextFilter(logging.Filter):
                def filter(self, record):
                    record.extra = context
                    return True

            logger.addFilter(ContextFilter())

        return logger

    def log_allocation_event(
        self, event_type: str, system: str, details: Dict[str, Any]
    ) -> None:
        """Log allocation-related events."""
        logger = self.get_logger("allocation")
        logger.info(
            f"Allocation event: {event_type}",
            extra={"event_type": event_type, "system": system, "details": details},
        )

        # Record metrics
        self.metrics.increment_counter(f"allocation_events_{event_type}")
        self.metrics.increment_counter(f"allocation_events_{system}")

    def log_performance_event(
        self, operation: str, duration: float, details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log performance events."""
        self.perf_logger.info(
            f"Performance: {operation}",
            extra={
                "operation": operation,
                "duration": duration,
                "details": details or {},
            },
        )

        self.metrics.record_metric(
            f"operation_{operation}_duration", duration, "seconds"
        )

    def log_error(
        self, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log errors with context."""
        self.error_logger.error(
            f"Error: {str(error)}",
            extra={
                "error_type": error.__class__.__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                "context": context or {},
            },
            exc_info=True,
        )

        self.metrics.increment_counter(f"errors_{error.__class__.__name__}")


# Global instance
_logger_instance: Optional[TradingSystemLogger] = None


def get_trading_logger() -> TradingSystemLogger:
    """Get or create the global trading system logger."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = TradingSystemLogger()
    return _logger_instance


# Convenience functions
def log_allocation_event(event_type: str, system: str, details: Dict[str, Any]) -> None:
    """Log allocation event using global logger."""
    get_trading_logger().log_allocation_event(event_type, system, details)


def log_performance_event(
    operation: str, duration: float, details: Optional[Dict[str, Any]] = None
) -> None:
    """Log performance event using global logger."""
    get_trading_logger().log_performance_event(operation, duration, details)


def log_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """Log error using global logger."""
    get_trading_logger().log_error(error, context)


# Context manager for performance monitoring
@contextmanager
def monitor_performance(operation: str, **context):
    """Monitor performance of an operation."""
    logger = get_trading_logger()
    with logger.metrics.timer(operation, **context):
        start_time = time.perf_counter()
        try:
            yield
        except Exception as e:
            log_error(e, {"operation": operation, **context})
            raise
        finally:
            duration = time.perf_counter() - start_time
            log_performance_event(operation, duration, context)


# Integration with final_allocation module
def enhanced_finalize_allocation(*args, **kwargs):
    """Wrapper for finalize_allocation with logging."""
    from core.final_allocation import finalize_allocation

    with monitor_performance(
        "finalize_allocation", system_count=len(kwargs.get("per_system", {}))
    ):
        try:
            result_df, summary = finalize_allocation(*args, **kwargs)

            # Log allocation details
            log_allocation_event(
                "allocation_completed",
                "all_systems",
                {
                    "mode": summary.mode,
                    "total_positions": len(result_df),
                    "long_systems": len(summary.long_allocations),
                    "short_systems": len(summary.short_allocations),
                },
            )

            return result_df, summary

        except Exception as e:
            log_error(e, {"function": "finalize_allocation", "args_count": len(args)})
            raise


if __name__ == "__main__":
    # Example usage
    logger = get_trading_logger()

    # Example allocation event
    log_allocation_event("position_created", "system1", {"symbol": "AAPL", "qty": 100})

    # Example performance monitoring
    with monitor_performance("test_operation"):
        time.sleep(0.1)  # Simulate work

    # Example error logging
    try:
        raise ValueError("Test error")
    except ValueError as e:
        log_error(e, {"test": True})

    print("Logging examples completed. Check logs/ directory.")
    print("Metrics summary:", logger.metrics.get_summary())
