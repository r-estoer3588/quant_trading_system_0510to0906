"""Trading system error definitions and structured error handling.

This module provides:
1. AAA123E format error code enumeration
2. TradingError exception classes
3. Error classification and retry policies
4. Structured error output for UI integration
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import time
from typing import Any, Dict, Optional


class ErrorCode(Enum):
    """Trading system error codes in AAA123E format.

    Format: [Category][Number][Severity]
    - Category: DAT=Data, SIG=Signal, ALC=Allocation, SYS=System, NET=Network
    - Number: 001-999
    - Severity: E=Error, W=Warning, I=Info
    """

    # Data errors (DAT)
    DAT001E = "DAT001E"  # Cache file not found
    DAT002E = "DAT002E"  # Invalid data format
    DAT003E = "DAT003E"  # Missing required columns
    DAT004E = "DAT004E"  # SPY data corruption
    DAT005E = "DAT005E"  # Indicator calculation failure
    DAT006W = "DAT006W"  # Rolling data incomplete
    DAT007E = "DAT007E"  # Symbol universe empty
    DAT008E = "DAT008E"  # Date range invalid

    # Signal errors (SIG)
    SIG001E = "SIG001E"  # Signal generation failed
    SIG002E = "SIG002E"  # Filter condition error
    SIG003E = "SIG003E"  # Setup condition error
    SIG004W = "SIG004W"  # No signals generated
    SIG005E = "SIG005E"  # Signal merge conflict
    SIG006E = "SIG006E"  # System strategy error

    # Allocation errors (ALC)
    ALC001E = "ALC001E"  # Allocation calculation failed
    ALC002E = "ALC002E"  # Position size invalid
    ALC003E = "ALC003E"  # Symbol mapping error
    ALC004W = "ALC004W"  # Partial allocation
    ALC005E = "ALC005E"  # Cash insufficient

    # System errors (SYS)
    SYS001E = "SYS001E"  # Configuration invalid
    SYS002E = "SYS002E"  # Pipeline phase failure
    SYS003E = "SYS003E"  # Parallel execution error
    SYS004E = "SYS004E"  # Resource allocation failed
    SYS005I = "SYS005I"  # System initialization

    # Network errors (NET)
    NET001E = "NET001E"  # API connection failed
    NET002E = "NET002E"  # Data download timeout
    NET003W = "NET003W"  # Rate limit exceeded
    NET004E = "NET004E"  # Authentication failed


@dataclass
class ErrorContext:
    """Context information for error reporting."""

    timestamp: str
    phase: str
    system: Optional[str] = None
    symbol: Optional[str] = None
    trace_id: Optional[str] = None
    additional: Dict[str, Any] = None

    def __post_init__(self):
        if self.additional is None:
            self.additional = {}


class TradingError(Exception):
    """Base trading system exception with structured error information."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or ErrorContext(
            timestamp=datetime.now().isoformat(), phase="unknown"
        )
        self.cause = cause
        self.retryable = retryable

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for structured output."""
        return {
            "error_code": self.error_code.value,
            "message": self.message,
            "retryable": self.retryable,
            "context": {
                "timestamp": self.context.timestamp,
                "phase": self.context.phase,
                "system": self.context.system,
                "symbol": self.context.symbol,
                "trace_id": self.context.trace_id,
                "additional": self.context.additional,
            },
            "cause": str(self.cause) if self.cause else None,
        }


class DataError(TradingError):
    """Data-related errors."""

    pass


class SignalError(TradingError):
    """Signal generation errors."""

    pass


class AllocationError(TradingError):
    """Allocation calculation errors."""

    pass


class SystemError(TradingError):
    """System-level errors."""

    pass


class NetworkError(TradingError):
    """Network-related errors."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retryable = True  # Network errors are generally retryable


class RetryPolicy:
    """Exponential backoff retry policy."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt (0-based)."""
        delay = self.base_delay * (self.backoff_factor**attempt)
        return min(delay, self.max_delay)

    def should_retry(self, attempt: int, error: Exception) -> bool:
        """Determine if retry should be attempted."""
        if attempt >= self.max_attempts:
            return False

        if isinstance(error, TradingError):
            return error.retryable

        # Default: don't retry unknown exceptions
        return False


def retry_with_backoff(
    func,
    args=None,
    kwargs=None,
    policy: Optional[RetryPolicy] = None,
    error_context: Optional[ErrorContext] = None,
):
    """Execute function with exponential backoff retry."""
    args = args or ()
    kwargs = kwargs or {}
    policy = policy or RetryPolicy()

    last_exception = None

    for attempt in range(policy.max_attempts):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e

            if not policy.should_retry(attempt, e):
                break

            if attempt < policy.max_attempts - 1:
                delay = policy.calculate_delay(attempt)
                time.sleep(delay)

    # If we get here, all retries failed
    if isinstance(last_exception, TradingError):
        raise last_exception
    else:
        # Wrap unknown exceptions
        raise SystemError(
            f"Operation failed after {policy.max_attempts} attempts: {str(last_exception)}",
            ErrorCode.SYS003E,
            context=error_context,
            cause=last_exception,
        )


# Error classification helpers
def classify_exception(e: Exception, context: Optional[ErrorContext] = None) -> TradingError:
    """Classify and wrap exceptions as TradingError."""
    if isinstance(e, TradingError):
        return e

    # Common exception patterns
    error_msg = str(e)

    if "FileNotFoundError" in str(type(e)) or "file not found" in error_msg.lower():
        return DataError(
            f"Cache file not found: {error_msg}",
            ErrorCode.DAT001E,
            context=context,
            cause=e,
        )

    if "KeyError" in str(type(e)) and any(col in error_msg for col in ["column", "key"]):
        return DataError(
            f"Missing required data column: {error_msg}",
            ErrorCode.DAT003E,
            context=context,
            cause=e,
        )

    if "ConnectionError" in str(type(e)) or "timeout" in error_msg.lower():
        return NetworkError(
            f"Network operation failed: {error_msg}",
            ErrorCode.NET001E,
            context=context,
            cause=e,
        )

    # Default: system error
    return SystemError(
        f"Unexpected error: {error_msg}", ErrorCode.SYS002E, context=context, cause=e
    )


# UI integration helpers
def format_error_for_ui(error: TradingError) -> Dict[str, Any]:
    """Format error for UI display."""
    return {
        "type": "error",
        "code": error.error_code.value,
        "message": error.message,
        "timestamp": error.context.timestamp,
        "phase": error.context.phase,
        "system": error.context.system,
        "retryable": error.retryable,
        "details": error.context.additional,
    }


def create_error_summary(errors: list[TradingError]) -> Dict[str, Any]:
    """Create summary of multiple errors for UI."""
    if not errors:
        return {"total": 0, "by_category": {}, "retryable_count": 0}

    by_category = {}
    retryable_count = 0

    for error in errors:
        category = error.error_code.value[:3]  # First 3 chars (DAT, SIG, etc.)
        by_category[category] = by_category.get(category, 0) + 1

        if error.retryable:
            retryable_count += 1

    return {
        "total": len(errors),
        "by_category": by_category,
        "retryable_count": retryable_count,
        "latest_error": format_error_for_ui(errors[-1]) if errors else None,
    }
