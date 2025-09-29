"""Enhanced structured logging system for the quant trading system.

This module provides comprehensive logging infrastructure with structured
JSON logs, performance metrics, monitoring capabilities, trace ID integration,
and error code system support. Integrates with trading_errors.ErrorCode.
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

try:
    import psutil
except ImportError:
    psutil = None

# Import the superior error code system
try:
    from .trading_errors import ErrorCode, ErrorContext, TradingError

    TRADING_ERRORS_AVAILABLE = True
except ImportError:
    TRADING_ERRORS_AVAILABLE = False

    # Fallback error codes for backwards compatibility
    class ErrorCode:
        """Fallback error codes."""

        DAT004E = "DAT004E"  # SPY data corruption (maps to SPY001E)
        SYS001E = "SYS001E"  # System execution failed
        DAT001E = "DAT001E"  # Cache file not found


# Import trace context if available
try:
    from .trace_context import get_current_trace_context

    TRACE_CONTEXT_AVAILABLE = True
except ImportError:
    TRACE_CONTEXT_AVAILABLE = False

    def get_current_trace_context():
        return None


# Legacy error code mapping for backwards compatibility
class ErrorCodes:
    """Legacy error codes mapped to new ErrorCode enum."""

    @classmethod
    def _get_error_code(cls, legacy_code: str) -> str:
        """Map legacy codes to new ErrorCode system."""
        if TRADING_ERRORS_AVAILABLE:
            # Map legacy codes to new AAA123E format
            mapping = {
                "SPY001E": ErrorCode.DAT004E.value,  # SPY data corruption
                "SYS001E": ErrorCode.SYS001E.value,  # System execution failed
                "DATA001E": ErrorCode.DAT001E.value,  # Cache file not found
                "CACHE001E": ErrorCode.DAT001E.value,  # Cache file not found
                "FIL001E": ErrorCode.SIG002E.value,  # Filter condition error
                "STU001E": ErrorCode.SIG003E.value,  # Setup condition error
                "TRD001E": ErrorCode.SIG001E.value,  # Signal generation failed
                "ENTRY001E": ErrorCode.ALC001E.value,  # Allocation calculation failed
                "EXIT001E": ErrorCode.ALC001E.value,  # Allocation calculation failed
            }
            return mapping.get(legacy_code, legacy_code)
        return legacy_code

    @classmethod
    def get_error_description(cls, error_code: str) -> str:
        """Get Japanese description for error code."""
        # Legacy code descriptions
        legacy_descriptions = {
            "SPY001E": "SPYデータが利用不可、全システム停止",
            "SYS001E": "システム実行エラー",
            "DATA001E": "重要データが見つからない",
            "CACHE001E": "キャッシュアクセスエラー",
            "FIL001E": "フィルター段階での重大エラー",
            "STU001E": "セットアップ段階での重大エラー",
            "TRD001E": "取引候補段階での重大エラー",
            "ENTRY001E": "エントリー段階での重大エラー",
            "EXIT001E": "エグジット段階での重大エラー",
        }

        # Try to get description from new ErrorCode system if available
        if TRADING_ERRORS_AVAILABLE:
            mapped_code = cls._get_error_code(error_code)
            try:
                for ec in ErrorCode:
                    if ec.value == mapped_code:
                        # Return Japanese description from ErrorCode enum docstring or name
                        if hasattr(ec, "_japanese_description"):
                            return ec._japanese_description
                        # Fallback to basic mapping
                        error_descriptions = {
                            "DAT001E": "キャッシュファイルが見つからない",
                            "DAT002E": "データ形式が無効",
                            "DAT003E": "データが古すぎる",
                            "DAT004E": "SPYデータの破損",
                            "DAT005E": "データの整合性チェック失敗",
                            "SIG001E": "シグナル生成に失敗",
                            "SIG002E": "フィルター条件エラー",
                            "SIG003E": "セットアップ条件エラー",
                            "ALC001E": "配分計算に失敗",
                            "SYS001E": "システム実行エラー",
                            "NET001E": "ネットワークタイムアウト",
                        }
                        return error_descriptions.get(
                            mapped_code, f"エラー: {mapped_code}"
                        )
            except Exception:
                pass

        # Fallback to legacy descriptions
        return legacy_descriptions.get(error_code, f"エラー: {error_code}")

    @classmethod
    def get_formatted_error(cls, error_code: str) -> str:
        """Get formatted error code with Japanese description."""
        description = cls.get_error_description(error_code)
        if TRADING_ERRORS_AVAILABLE:
            mapped_code = cls._get_error_code(error_code)
            # 新しいコードを使用し、旧コード表示は省略
            return f"[{mapped_code}] {description}"
        return f"[{error_code}] {description}"

    # SPY dependency errors (class attributes for backwards compatibility)
    SPY001E = "SPY001E"
    SYS001E = "SYS001E"
    DATA001E = "DATA001E"
    CACHE001E = "CACHE001E"
    FIL001E = "FIL001E"
    STU001E = "STU001E"
    TRD001E = "TRD001E"
    ENTRY001E = "ENTRY001E"
    EXIT001E = "EXIT001E"

    @classmethod
    def get_mapped_code(cls, legacy_code: str) -> str:
        """Get the mapped error code for use in logging."""
        return cls._get_error_code(legacy_code)


@dataclass
class LogEvent:
    """Structured log event with trace context."""

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
    trace_id: Optional[str] = None
    parent_trace_id: Optional[str] = None
    phase: Optional[str] = None
    trading_system: Optional[str] = None
    symbol: Optional[str] = None
    error_code: Optional[str] = None  # Standardized error code
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
    """JSON formatter for structured logging with trace context."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON with trace information."""
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

        # Get trace context information
        trace_info = {}
        try:
            from .trace_context import get_trace_info_for_logging

            trace_info = get_trace_info_for_logging()
        except ImportError:
            pass  # Trace context not available

        event = LogEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=record.levelname,
            logger=record.name,
            message=record.getMessage(),
            module=caller_frame.filename.split("/")[-1] if caller_frame else "",
            function=caller_frame.name if caller_frame else "",
            line=caller_frame.lineno if caller_frame and caller_frame.lineno else 0,
            thread_id=threading.get_ident(),
            process_id=os.getpid(),
            trace_id=trace_info.get("trace_id"),
            parent_trace_id=trace_info.get("parent_trace_id"),
            phase=trace_info.get("phase"),
            trading_system=trace_info.get("system"),
            symbol=trace_info.get("symbol"),
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
        """Context manager for timing operations with memory tracking."""
        start_time = time.perf_counter()
        start_memory = None
        if psutil:
            try:
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            except Exception:
                start_memory = None

        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time

            memory_delta = None
            if psutil and start_memory is not None:
                try:
                    end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    memory_delta = end_memory - start_memory
                except Exception:
                    memory_delta = None

            self.record_metric(f"{name}_duration", elapsed, "seconds", **tags)
            if memory_delta is not None:
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
    """Enhanced logging system for the trading application with error handling and UI integration."""

    def __init__(self, log_dir: Optional[Path] = None, ring_buffer_size: int = 1000):
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = MetricsCollector(self.log_dir)

        # Ring buffer for UI Logger
        self.ring_buffer_size = ring_buffer_size
        self.ring_buffer = []
        self.ring_buffer_lock = threading.Lock()

        # Error tracking
        self.error_count = 0
        self.errors_by_code = defaultdict(int)

        self._setup_loggers()
        self._setup_additional_loggers()  # 追加されました

    def _add_to_ring_buffer(self, log_record: Dict[str, Any]) -> None:
        """Add log record to ring buffer for UI consumption."""
        with self.ring_buffer_lock:
            self.ring_buffer.append(log_record)
            if len(self.ring_buffer) > self.ring_buffer_size:
                self.ring_buffer.pop(0)

    def get_ring_buffer(self, last_n: Optional[int] = None) -> list[Dict[str, Any]]:
        """Get ring buffer contents for UI."""
        with self.ring_buffer_lock:
            if last_n is None:
                return self.ring_buffer.copy()
            return self.ring_buffer[-last_n:] if last_n > 0 else []

    def clear_ring_buffer(self) -> None:
        """Clear ring buffer."""
        with self.ring_buffer_lock:
            self.ring_buffer.clear()

    def _setup_loggers(self) -> None:
        """Setup structured loggers."""
        # Main application logger
        self.app_logger = logging.getLogger("quant_trading")
        self.app_logger.setLevel(logging.INFO)

        # Custom handler that feeds ring buffer
        class RingBufferHandler(logging.Handler):
            def __init__(self, trading_logger):
                super().__init__()
                self.trading_logger = trading_logger

            def emit(self, record):
                try:
                    log_data = {
                        "timestamp": datetime.now().isoformat(),
                        "level": record.levelname,
                        "logger": record.name,
                        "message": record.getMessage(),
                        "module": getattr(record, "module", ""),
                        "function": getattr(record, "funcName", ""),
                        "line": getattr(record, "lineno", 0),
                    }

                    # Add trace context if available
                    try:
                        from .trace_context import get_trace_info_for_logging

                        trace_info = get_trace_info_for_logging()
                        log_data.update(trace_info)
                    except ImportError:
                        pass

                    self.trading_logger._add_to_ring_buffer(log_data)
                except Exception:
                    pass  # Don't let logging errors break the system

        # JSON file handler
        json_handler = logging.FileHandler(self.log_dir / "application.jsonl")
        json_handler.setFormatter(StructuredFormatter())
        self.app_logger.addHandler(json_handler)

        # Ring buffer handler
        ring_handler = RingBufferHandler(self)
        self.app_logger.addHandler(ring_handler)

    def _get_trace_id(self, trace_context: Dict = None) -> Optional[str]:
        """Get current trace ID from context or trace system."""
        if trace_context and "trace_id" in trace_context:
            return trace_context["trace_id"]

        if TRACE_CONTEXT_AVAILABLE:
            try:
                trace_info = get_current_trace_context()
                return trace_info.trace_id if trace_info else None
            except Exception:
                return None
        return None

    def log_with_error_code(
        self, level: str, message: str, error_code: str, **extra
    ) -> None:
        """Log message with standardized error code and Japanese description."""
        # Map legacy error codes to new system if available
        if TRADING_ERRORS_AVAILABLE:
            error_code_value = ErrorCodes.get_mapped_code(error_code)

            # Create ErrorContext if this is an error level
            if level.upper() == "ERROR":
                try:
                    # Try to find the matching ErrorCode enum
                    for ec in ErrorCode:
                        if ec.value == error_code_value:
                            error_context = ErrorContext(
                                error_code=ec,
                                system_id=extra.get("trading_system"),
                                trace_id=self._get_trace_id(),
                                timestamp=datetime.now(),
                                details=extra,
                            )
                            extra["error_context"] = error_context
                            break
                except Exception:
                    pass  # If mapping fails, continue with original code
        else:
            error_code_value = error_code

        # Get formatted error code with Japanese description
        formatted_error = ErrorCodes.get_formatted_error(error_code)

        logger_level = getattr(logging, level.upper(), logging.INFO)
        extra_data = {
            "error_code": error_code_value,
            "formatted_error": formatted_error,
            **extra,
        }
        self.app_logger.log(
            logger_level, f"{formatted_error} {message}", extra=extra_data
        )

        # Track error statistics
        if level.upper() == "ERROR":
            self.error_count += 1
            self.errors_by_code[error_code_value] += 1

    def log_spy_error(self, message: str, **extra) -> None:
        """Log SPY-related error with standardized code and Japanese description."""
        self.log_with_error_code("ERROR", message, ErrorCodes.SPY001E, **extra)

    def log_system_error(
        self, message: str, system: Optional[str] = None, **extra
    ) -> None:
        """Log system execution error with standardized code and Japanese description."""
        if system:
            extra["trading_system"] = system
        self.log_with_error_code("ERROR", message, ErrorCodes.SYS001E, **extra)

    def log_filter_error(
        self, message: str, system: Optional[str] = None, **extra
    ) -> None:
        """Log filter phase error with standardized code and Japanese description."""
        if system:
            extra["trading_system"] = system
        extra["phase"] = "filter"
        self.log_with_error_code("ERROR", message, ErrorCodes.FIL001E, **extra)

    def log_setup_error(
        self, message: str, system: Optional[str] = None, **extra
    ) -> None:
        """Log setup phase error with standardized code and Japanese description."""
        if system:
            extra["trading_system"] = system
        extra["phase"] = "setup"
        self.log_with_error_code("ERROR", message, ErrorCodes.STU001E, **extra)

    def log_trade_candidate_error(
        self, message: str, system: Optional[str] = None, **extra
    ) -> None:
        """Log trade candidate phase error with standardized code and Japanese description."""
        if system:
            extra["trading_system"] = system
        extra["phase"] = "trade_candidate"
        self.log_with_error_code("ERROR", message, ErrorCodes.TRD001E, **extra)

    def log_entry_error(
        self, message: str, system: Optional[str] = None, **extra
    ) -> None:
        """Log entry phase error with standardized code and Japanese description."""
        if system:
            extra["trading_system"] = system
        extra["phase"] = "entry"
        self.log_with_error_code("ERROR", message, ErrorCodes.ENTRY001E, **extra)

    def log_exit_error(
        self, message: str, system: Optional[str] = None, **extra
    ) -> None:
        """Log exit phase error with standardized code and Japanese description."""
        if system:
            extra["trading_system"] = system
        extra["phase"] = "exit"
        self.log_with_error_code("ERROR", message, ErrorCodes.EXIT001E, **extra)

    def _setup_additional_loggers(self) -> None:
        """Setup additional loggers for console, performance, and errors."""
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
        """Log errors with context and error code classification."""
        error_context = context or {}

        # Classify error if it's not already a TradingError
        if hasattr(error, "error_code") and hasattr(error, "error_code"):
            # Already a TradingError
            trading_error = error
            error_code = getattr(error, "error_code").value
        else:
            # Classify the exception
            try:
                from .trading_errors import ErrorContext, classify_exception

                error_ctx = ErrorContext(
                    timestamp=datetime.now().isoformat(),
                    phase=error_context.get("phase", "unknown"),
                    system=error_context.get("system"),
                    symbol=error_context.get("symbol"),
                )
                trading_error = classify_exception(error, error_ctx)
                error_code = trading_error.error_code.value
            except ImportError:
                trading_error = error
                error_code = "SYS002E"  # Fallback error code

        # Track error statistics
        self.error_count += 1
        self.errors_by_code[error_code] += 1

        # Prepare log data
        log_data = {
            "error_code": error_code,
            "error_type": error.__class__.__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": error_context,
        }

        # Add error to ring buffer for UI
        ui_error_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "ERROR",
            "logger": "errors",
            "message": f"[{error_code}] {str(error)}",
            "error_code": error_code,
            "retryable": getattr(trading_error, "retryable", False),
        }
        self._add_to_ring_buffer(ui_error_data)

        self.error_logger.error(
            f"Error: {str(error)}",
            extra=log_data,
            exc_info=True,
        )

        self.metrics.increment_counter(f"errors_{error.__class__.__name__}")
        self.metrics.increment_counter(f"error_codes_{error_code}")

    def log_trading_error(
        self, trading_error, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log TradingError with full context."""
        self.log_error(trading_error, context)

        # Additional logging for TradingError
        if hasattr(trading_error, "to_dict"):
            self.error_logger.info(
                "Structured error details",
                extra={"structured_error": trading_error.to_dict()},
            )

    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary for monitoring."""
        return {
            "total_errors": self.error_count,
            "errors_by_code": dict(self.errors_by_code),
            "recent_errors": self.get_ring_buffer(last_n=10),
        }


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
