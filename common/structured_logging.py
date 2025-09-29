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
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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

    # Fallback classes for backwards compatibility
    @dataclass
    class FallbackErrorCode:
        """Fallback error codes."""

        code: str = "SYS001E"
        _japanese_description: str = "システム実行エラー"

        DAT004E = "DAT004E"  # SPY data corruption (maps to SPY001E)
        SYS001E = "SYS001E"  # System execution failed
        DAT001E = "DAT001E"  # Cache file not found

    @dataclass
    class FallbackErrorContext:
        """Fallback error context."""

        timestamp: str
        system: str
        operation: str = ""
        details: Dict[str, Any] = None

    class FallbackTradingError(Exception):
        """Fallback trading error."""

        def __init__(self, message: str, error_code: Optional[str] = None):
            super().__init__(message)
            self.error_code = error_code

    # Assign to standard names for compatibility
    ErrorCode = FallbackErrorCode
    ErrorContext = FallbackErrorContext
    TradingError = FallbackTradingError


# Import trace context if available
try:
    from .trace_context import get_current_trace_context

    TRACE_CONTEXT_AVAILABLE = True
except ImportError:
    TRACE_CONTEXT_AVAILABLE = False

    def get_current_trace_context() -> Optional[Dict[str, Any]]:
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
                            return str(ec._japanese_description)
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


@dataclass
class AnomalyAlert:
    """異常検知アラート情報。"""

    timestamp: str
    alert_type: str  # "cpu_spike", "memory_spike", "performance_degradation"
    severity: str  # "warning", "critical"
    message: str
    current_value: float
    threshold: float
    baseline: float = 0.0
    system_context: Dict[str, Any] = field(default_factory=dict)


class SystemAnomalyDetector:
    """リアルタイム異常検知システム。

    CPU/メモリ使用量の急激な変化を検知し、
    しきい値超過時にアラート出力とログ記録を行う。
    """

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path("logs/anomaly")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # アラート設定
        self.cpu_threshold = 80.0  # CPU使用率80%でアラート
        self.memory_threshold = 80.0  # メモリ使用率80%でアラート
        self.cpu_spike_threshold = 30.0  # 30%以上の急激な増加
        self.memory_spike_threshold = 25.0  # 25%以上の急激な増加

        # 履歴データ（最新10回分）
        self.cpu_history = deque(maxlen=10)
        self.memory_history = deque(maxlen=10)

        # アラート記録
        self.alerts: List[AnomalyAlert] = []
        self._alert_lock = threading.Lock()

        # ログ設定
        self.logger = logging.getLogger("anomaly_detector")
        handler = logging.FileHandler(self.log_dir / "anomalies.jsonl")
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def check_system_health(self, system_name: str = "unknown") -> List[AnomalyAlert]:
        """システム健康状態をチェックし、異常があればアラートを生成。"""
        alerts = []

        if not psutil:
            return alerts

        try:
            # 現在のリソース状況取得
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # 履歴に追加
            self.cpu_history.append(cpu_percent)
            self.memory_history.append(memory_percent)

            current_time = datetime.now(timezone.utc).isoformat()

            # CPU異常検知
            cpu_alert = self._check_cpu_anomaly(cpu_percent, current_time, system_name)
            if cpu_alert:
                alerts.append(cpu_alert)

            # メモリ異常検知
            memory_alert = self._check_memory_anomaly(
                memory_percent, current_time, system_name
            )
            if memory_alert:
                alerts.append(memory_alert)

            # アラートをログに記録
            for alert in alerts:
                self._log_alert(alert)

        except Exception as e:
            self.logger.error(f"System health check failed: {e}")

        return alerts

    def _check_cpu_anomaly(
        self, current_cpu: float, timestamp: str, system_name: str
    ) -> Optional[AnomalyAlert]:
        """CPU異常をチェック。"""

        # しきい値チェック
        if current_cpu > self.cpu_threshold:
            severity = "critical" if current_cpu > 90 else "warning"
            return AnomalyAlert(
                timestamp=timestamp,
                alert_type="cpu_threshold",
                severity=severity,
                message=f"CPU使用率が高すぎます: {current_cpu:.1f}%",
                current_value=current_cpu,
                threshold=self.cpu_threshold,
                system_context={"system": system_name},
            )

        # 急激な変化チェック
        if len(self.cpu_history) >= 2:
            previous_cpu = self.cpu_history[-2]
            cpu_delta = current_cpu - previous_cpu

            if cpu_delta > self.cpu_spike_threshold:
                return AnomalyAlert(
                    timestamp=timestamp,
                    alert_type="cpu_spike",
                    severity="warning",
                    message=f"CPU使用率が急激に増加: {previous_cpu:.1f}% → {current_cpu:.1f}%",
                    current_value=current_cpu,
                    threshold=self.cpu_spike_threshold,
                    baseline=previous_cpu,
                    system_context={"system": system_name, "delta": cpu_delta},
                )

        return None

    def _check_memory_anomaly(
        self, current_memory: float, timestamp: str, system_name: str
    ) -> Optional[AnomalyAlert]:
        """メモリ異常をチェック。"""

        # しきい値チェック
        if current_memory > self.memory_threshold:
            severity = "critical" if current_memory > 90 else "warning"
            return AnomalyAlert(
                timestamp=timestamp,
                alert_type="memory_threshold",
                severity=severity,
                message=f"メモリ使用率が高すぎます: {current_memory:.1f}%",
                current_value=current_memory,
                threshold=self.memory_threshold,
                system_context={"system": system_name},
            )

        # 急激な変化チェック
        if len(self.memory_history) >= 2:
            previous_memory = self.memory_history[-2]
            memory_delta = current_memory - previous_memory

            if memory_delta > self.memory_spike_threshold:
                return AnomalyAlert(
                    timestamp=timestamp,
                    alert_type="memory_spike",
                    severity="warning",
                    message=f"メモリ使用率が急激に増加: {previous_memory:.1f}% → {current_memory:.1f}%",
                    current_value=current_memory,
                    threshold=self.memory_spike_threshold,
                    baseline=previous_memory,
                    system_context={"system": system_name, "delta": memory_delta},
                )

        return None

    def _log_alert(self, alert: AnomalyAlert) -> None:
        """アラートをログに記録。"""
        with self._alert_lock:
            self.alerts.append(alert)

            # ログレベル決定
            if alert.severity == "critical":
                log_level = logging.ERROR
            else:
                log_level = logging.WARNING

            # 構造化ログ出力
            log_data = {
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "current_value": alert.current_value,
                "threshold": alert.threshold,
                "baseline": alert.baseline,
                "system_context": alert.system_context,
            }

            self.logger.log(
                log_level, f"ANOMALY_DETECTED: {alert.message}", extra=log_data
            )

    def get_recent_alerts(self, hours: int = 1) -> List[AnomalyAlert]:
        """最近のアラートを取得。"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        return [
            alert
            for alert in self.alerts
            if datetime.fromisoformat(alert.timestamp.replace("Z", "+00:00"))
            > cutoff_time
        ]


class PerformanceDegradationDetector:
    """パフォーマンス劣化検知システム。

    ベースライン学習と偏差計算により、処理速度低下を検知し
    通常より大幅に遅い処理を自動検出してアラート出力。
    """

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path("logs/performance")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # ベースライン設定
        self.baseline_window = 20  # 過去20回の実行でベースライン計算
        self.degradation_threshold = 2.0  # 2倍以上遅い場合にアラート
        self.min_samples = 5  # 最低5回の実行データが必要

        # 実行履歴（操作別）
        self.execution_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.baseline_window)
        )

        # アラート記録
        self.degradation_alerts: List[AnomalyAlert] = []
        self._alerts_lock = threading.Lock()

        # ログ設定
        self.logger = logging.getLogger("performance_detector")
        handler = logging.FileHandler(self.log_dir / "performance_degradation.jsonl")
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def record_execution(
        self, operation: str, duration: float, system_name: str = "unknown"
    ) -> Optional[AnomalyAlert]:
        """実行時間を記録し、劣化があれば検知。"""

        # 履歴に追加
        self.execution_history[operation].append(
            {
                "duration": duration,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "system": system_name,
            }
        )

        # 劣化チェック
        return self._check_performance_degradation(operation, duration, system_name)

    def _check_performance_degradation(
        self, operation: str, current_duration: float, system_name: str
    ) -> Optional[AnomalyAlert]:
        """パフォーマンス劣化をチェック。"""

        history = self.execution_history[operation]

        # 十分なサンプル数があるかチェック
        if len(history) < self.min_samples:
            return None

        # ベースライン計算（現在実行を除く過去データ）
        past_durations = [h["duration"] for h in list(history)[:-1]]
        if not past_durations:
            return None

        baseline_avg = sum(past_durations) / len(past_durations)
        baseline_std = self._calculate_std(past_durations, baseline_avg)

        # 劣化判定
        if current_duration > baseline_avg * self.degradation_threshold:
            severity = (
                "critical" if current_duration > baseline_avg * 3.0 else "warning"
            )

            alert = AnomalyAlert(
                timestamp=datetime.now(timezone.utc).isoformat(),
                alert_type="performance_degradation",
                severity=severity,
                message=f"操作 '{operation}' が異常に遅くなっています: {current_duration:.2f}s (通常: {baseline_avg:.2f}s)",
                current_value=current_duration,
                threshold=baseline_avg * self.degradation_threshold,
                baseline=baseline_avg,
                system_context={
                    "system": system_name,
                    "operation": operation,
                    "slowdown_factor": current_duration / baseline_avg,
                    "baseline_std": baseline_std,
                    "sample_count": len(past_durations),
                },
            )

            self._log_degradation_alert(alert)
            return alert

        return None

    def _calculate_std(self, values: List[float], mean: float) -> float:
        """標準偏差を計算。"""
        if len(values) <= 1:
            return 0.0

        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance**0.5

    def _log_degradation_alert(self, alert: AnomalyAlert) -> None:
        """劣化アラートをログに記録。"""
        with self._alerts_lock:
            self.degradation_alerts.append(alert)

            # ログレベル決定
            log_level = (
                logging.ERROR if alert.severity == "critical" else logging.WARNING
            )

            # 構造化ログ出力
            log_data = {
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "operation": alert.system_context.get("operation"),
                "current_duration": alert.current_value,
                "baseline_duration": alert.baseline,
                "slowdown_factor": alert.system_context.get("slowdown_factor"),
                "threshold": alert.threshold,
                "system": alert.system_context.get("system"),
                "sample_count": alert.system_context.get("sample_count"),
            }

            self.logger.log(
                log_level, f"PERFORMANCE_DEGRADATION: {alert.message}", extra=log_data
            )

    def get_operation_baseline(self, operation: str) -> Optional[Dict[str, float]]:
        """操作のベースライン統計を取得。"""
        history = self.execution_history[operation]

        if len(history) < self.min_samples:
            return None

        durations = [h["duration"] for h in history]
        avg = sum(durations) / len(durations)
        std = self._calculate_std(durations, avg)

        return {
            "average": avg,
            "std_dev": std,
            "sample_count": len(durations),
            "min": min(durations),
            "max": max(durations),
            "threshold": avg * self.degradation_threshold,
        }

    def get_recent_degradations(self, hours: int = 1) -> List[AnomalyAlert]:
        """最近の劣化アラートを取得。"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        return [
            alert
            for alert in self.degradation_alerts
            if datetime.fromisoformat(alert.timestamp.replace("Z", "+00:00"))
            > cutoff_time
        ]


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

    @contextmanager
    def system_benchmark(self, system_name: str, operation: str = "processing", **tags):
        """システム特化のベンチマーク用コンテキストマネージャー。

        メモリ使用量、処理時間、データ量を自動計測し、
        システム分析レポートに反映可能な形式で記録。
        """
        start_time = time.perf_counter()
        start_memory = None
        cpu_percent_start = None

        if psutil:
            try:
                process = psutil.Process()
                start_memory = process.memory_info().rss / 1024 / 1024  # MB
                cpu_percent_start = process.cpu_percent()
            except Exception:
                start_memory = None
                cpu_percent_start = None

        symbol_count = tags.get("symbol_count", 0)
        operation_start = {
            "system": system_name,
            "operation": operation,
            "start_time": start_time,
            "start_memory_mb": start_memory,
            "cpu_start": cpu_percent_start,
            "symbol_count": symbol_count,
        }

        try:
            yield operation_start
        finally:
            elapsed = time.perf_counter() - start_time

            end_memory = None
            memory_delta = None
            cpu_percent_end = None

            if psutil and start_memory is not None:
                try:
                    process = psutil.Process()
                    end_memory = process.memory_info().rss / 1024 / 1024
                    memory_delta = end_memory - start_memory
                    cpu_percent_end = process.cpu_percent()
                except Exception:
                    end_memory = None
                    memory_delta = None
                    cpu_percent_end = None

            # システム別のパフォーマンス指標を記録
            metrics_tags = {"system": system_name, "operation": operation, **tags}

            self.record_metric(
                f"system_{system_name}_duration", elapsed, "seconds", **metrics_tags
            )

            if memory_delta is not None:
                self.record_metric(
                    f"system_{system_name}_memory_delta",
                    memory_delta,
                    "MB",
                    **metrics_tags,
                )

            if symbol_count > 0:
                symbols_per_second = symbol_count / elapsed if elapsed > 0 else 0
                self.record_metric(
                    f"system_{system_name}_throughput",
                    symbols_per_second,
                    "symbols_per_sec",
                    **metrics_tags,
                )

            # CPU使用率が取得できた場合
            if cpu_percent_start is not None and cpu_percent_end is not None:
                self.record_metric(
                    f"system_{system_name}_cpu_usage",
                    cpu_percent_end,
                    "percent",
                    **metrics_tags,
                )

            # 要約情報をログ
            summary = {
                "system": system_name,
                "operation": operation,
                "duration_sec": round(elapsed, 3),
                "memory_delta_mb": round(memory_delta, 2) if memory_delta else None,
                "symbols_processed": symbol_count,
                "throughput_symbols_per_sec": (
                    round(symbols_per_second, 2) if symbol_count > 0 else None
                ),
            }

            self.logger.info(f"System benchmark: {json.dumps(summary)}")

            # ボトルネック分析用にフェーズ実行時間を記録
            # operationがフェーズ名に対応している場合
            phase_metadata = {
                "memory_delta_mb": memory_delta,
                "cpu_usage": cpu_percent_end,
                "symbol_count": symbol_count,
                "throughput": symbols_per_second if symbol_count > 0 else None,
            }

            self.bottleneck_analyzer.record_phase_timing(
                phase=operation,
                duration=elapsed,
                system_name=system_name,
                **phase_metadata,
            )

            # リアルタイムメトリクス記録
            throughput = symbols_per_second if symbol_count > 0 else None
            self.realtime_metrics.record_system_performance(
                system_name=system_name,
                operation=operation,
                duration=elapsed,
                throughput=throughput,
                memory_delta=memory_delta,
            )


class TradingSystemLogger:
    """Enhanced logging system for the trading application with error handling and UI integration."""

    def __init__(self, log_dir: Optional[Path] = None, ring_buffer_size: int = 1000):
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = MetricsCollector(self.log_dir)

        # 異常検知システム
        self.anomaly_detector = SystemAnomalyDetector(self.log_dir)
        self.performance_detector = PerformanceDegradationDetector(self.log_dir)
        self.bottleneck_analyzer = BottleneckAnalyzer(self.log_dir)
        self.progress_tracker = PredictiveProgressTracker(self.log_dir)
        self.realtime_metrics = RealTimeMetricsCollector(self.log_dir)

        # Ring buffer for UI Logger
        self.ring_buffer_size = ring_buffer_size
        self.ring_buffer: List[Dict[str, Any]] = []
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

    def _get_trace_id(
        self, trace_context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Get current trace ID from context or trace system."""
        if trace_context and "trace_id" in trace_context:
            return trace_context["trace_id"]

        if TRACE_CONTEXT_AVAILABLE:
            try:
                trace_info = get_current_trace_context()
                return getattr(trace_info, "trace_id", None) if trace_info else None
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
                                timestamp=datetime.now(timezone.utc).isoformat(),
                                system=extra.get("trading_system", "unknown"),
                                message=f"Error Code: {error_code_value}",
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
        """Log performance events with degradation detection."""

        # 基本ログ記録
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

        # パフォーマンス劣化検知
        system_name = details.get("system", "unknown") if details else "unknown"
        degradation_alert = self.performance_detector.record_execution(
            operation, duration, system_name
        )

        # 劣化検知時の追加ログ
        if degradation_alert:
            self.app_logger.warning(
                f"Performance degradation detected: {degradation_alert.message}",
                extra={
                    "alert_type": degradation_alert.alert_type,
                    "severity": degradation_alert.severity,
                    "operation": operation,
                    "slowdown_factor": degradation_alert.system_context.get(
                        "slowdown_factor"
                    ),
                },
            )

    def check_system_health(self, system_name: str = "unknown") -> Dict[str, Any]:
        """システム健康状態の包括的チェック。"""

        # リソース異常検知
        resource_alerts = self.anomaly_detector.check_system_health(system_name)

        # 最近のパフォーマンス劣化アラート
        performance_alerts = self.performance_detector.get_recent_degradations(hours=1)

        # ボトルネック分析
        bottleneck_analysis = self.bottleneck_analyzer.get_ui_data(system_name)

        # 健康状態サマリー
        health_summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system": system_name,
            "resource_alerts": len(resource_alerts),
            "performance_alerts": len(performance_alerts),
            "bottlenecks": len(bottleneck_analysis["bottlenecks"]),
            "total_errors": self.error_count,
            "error_rate": self.error_count
            / max(1, self.metrics.get_summary().get("total_events", 1)),
            "alerts": {
                "resource": [
                    {
                        "type": alert.alert_type,
                        "severity": alert.severity,
                        "message": alert.message,
                        "value": alert.current_value,
                        "threshold": alert.threshold,
                    }
                    for alert in resource_alerts
                ],
                "performance": [
                    {
                        "type": alert.alert_type,
                        "severity": alert.severity,
                        "message": alert.message,
                        "operation": alert.system_context.get("operation"),
                        "slowdown": alert.system_context.get("slowdown_factor"),
                    }
                    for alert in performance_alerts
                ],
            },
            "bottleneck_analysis": bottleneck_analysis,
        }

        # 健康状態に問題がある場合はログ記録
        if resource_alerts or performance_alerts or bottleneck_analysis["bottlenecks"]:
            self.app_logger.warning(
                f"System health issues detected for {system_name}", extra=health_summary
            )

        # リアルタイムメトリクスにアラート状態を更新
        all_alerts = []
        for alert in resource_alerts:
            all_alerts.append(
                {
                    "type": alert.alert_type,
                    "severity": alert.severity,
                    "message": alert.message,
                    "timestamp": alert.timestamp,
                }
            )
        for alert in performance_alerts:
            all_alerts.append(
                {
                    "type": alert.alert_type,
                    "severity": alert.severity,
                    "message": alert.message,
                    "timestamp": alert.timestamp,
                }
            )

        self.realtime_metrics.update_alerts(all_alerts)

        return health_summary


class BottleneckAnalyzer:
    """ボトルネック特定システム。

    フェーズ別実行時間分析でボトルネックを特定し、
    どの処理段階が最も時間を要しているかを視覚的に分析可能。
    """

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path("logs/bottleneck")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # フェーズ実行時間履歴
        self.phase_timings: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.max_history = 50  # 最新50回の実行を保持

        # ボトルネック分析結果
        self.bottleneck_reports: List[Dict[str, Any]] = []
        self._analysis_lock = threading.Lock()

        # ログ設定
        self.logger = logging.getLogger("bottleneck_analyzer")
        handler = logging.FileHandler(self.log_dir / "bottleneck_analysis.jsonl")
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def record_phase_timing(
        self, phase: str, duration: float, system_name: str = "unknown", **metadata
    ) -> None:
        """フェーズ実行時間を記録。"""

        timing_record = {
            "phase": phase,
            "duration": duration,
            "system": system_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata,
        }

        with self._analysis_lock:
            self.phase_timings[phase].append(timing_record)

            # 履歴サイズ制限
            if len(self.phase_timings[phase]) > self.max_history:
                self.phase_timings[phase] = self.phase_timings[phase][
                    -self.max_history :
                ]

    def analyze_bottlenecks(self, system_name: str = "all") -> Dict[str, Any]:
        """ボトルネック分析を実行。"""

        with self._analysis_lock:
            analysis = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "system": system_name,
                "phase_analysis": {},
                "bottlenecks": [],
                "recommendations": [],
            }

            # フェーズ別統計計算
            for phase, timings in self.phase_timings.items():
                # システム別フィルタリング
                if system_name != "all":
                    timings = [t for t in timings if t["system"] == system_name]

                if not timings:
                    continue

                durations = [t["duration"] for t in timings]

                phase_stats = {
                    "count": len(durations),
                    "avg_duration": sum(durations) / len(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "total_time": sum(durations),
                    "std_dev": self._calculate_std(
                        durations, sum(durations) / len(durations)
                    ),
                    "recent_trend": self._calculate_trend(
                        durations[-10:] if len(durations) >= 10 else durations
                    ),
                }

                analysis["phase_analysis"][phase] = phase_stats

            # ボトルネック特定
            if analysis["phase_analysis"]:
                bottlenecks = self._identify_bottlenecks(analysis["phase_analysis"])
                analysis["bottlenecks"] = bottlenecks
                analysis["recommendations"] = self._generate_recommendations(
                    bottlenecks
                )

            # 分析結果をログに記録
            self._log_analysis(analysis)

            # 結果履歴に追加
            self.bottleneck_reports.append(analysis)
            if len(self.bottleneck_reports) > 20:  # 最新20回の分析結果を保持
                self.bottleneck_reports = self.bottleneck_reports[-20:]

            return analysis

    def _calculate_std(self, values: List[float], mean: float) -> float:
        """標準偏差を計算。"""
        if len(values) <= 1:
            return 0.0

        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance**0.5

    def _calculate_trend(self, recent_values: List[float]) -> str:
        """最近の傾向を計算。"""
        if len(recent_values) < 3:
            return "insufficient_data"

        # 簡単な線形トレンド計算
        n = len(recent_values)
        x_avg = (n - 1) / 2
        y_avg = sum(recent_values) / n

        numerator = sum((i - x_avg) * (y - y_avg) for i, y in enumerate(recent_values))
        denominator = sum((i - x_avg) ** 2 for i in range(n))

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"

    def _identify_bottlenecks(
        self, phase_analysis: Dict[str, Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """ボトルネックを特定。"""

        # 総実行時間を計算
        total_time = sum(stats["total_time"] for stats in phase_analysis.values())

        bottlenecks = []

        for phase, stats in phase_analysis.items():
            time_percentage = (stats["total_time"] / total_time) * 100

            # ボトルネック判定基準
            is_bottleneck = (
                time_percentage > 30  # 全体時間の30%以上
                or stats["avg_duration"] > 5  # 平均5秒以上
                or stats["recent_trend"] == "increasing"  # 増加傾向
            )

            if is_bottleneck:
                severity = (
                    "high"
                    if time_percentage > 50
                    else "medium" if time_percentage > 30 else "low"
                )

                bottleneck = {
                    "phase": phase,
                    "severity": severity,
                    "time_percentage": time_percentage,
                    "avg_duration": stats["avg_duration"],
                    "total_time": stats["total_time"],
                    "trend": stats["recent_trend"],
                    "std_dev": stats["std_dev"],
                    "variability": (
                        "high"
                        if stats["std_dev"] > stats["avg_duration"] * 0.5
                        else "normal"
                    ),
                }

                bottlenecks.append(bottleneck)

        # 重要度順にソート
        bottlenecks.sort(key=lambda x: x["time_percentage"], reverse=True)

        return bottlenecks

    def _generate_recommendations(self, bottlenecks: List[Dict[str, Any]]) -> List[str]:
        """最適化推奨事項を生成。"""

        recommendations = []

        for bottleneck in bottlenecks:
            phase = bottleneck["phase"]
            severity = bottleneck["severity"]
            trend = bottleneck["trend"]
            variability = bottleneck["variability"]

            if severity == "high":
                recommendations.append(
                    f"⚠️ 高優先度: '{phase}'フェーズが全体時間の{bottleneck['time_percentage']:.1f}%を占有しています"
                )

            if trend == "increasing":
                recommendations.append(
                    f"📈 '{phase}'フェーズの実行時間が増加傾向です - 原因調査が必要"
                )

            if variability == "high":
                recommendations.append(
                    f"📊 '{phase}'フェーズの実行時間のばらつきが大きいです - 処理安定化を検討"
                )

            # フェーズ特有の推奨事項
            if "filter" in phase.lower():
                recommendations.append(
                    f"🔍 '{phase}': フィルター条件の最適化やインデックス追加を検討"
                )
            elif "signal" in phase.lower():
                recommendations.append(
                    f"📊 '{phase}': シグナル計算のキャッシュ化や並列処理を検討"
                )
            elif "allocation" in phase.lower():
                recommendations.append(
                    f"💰 '{phase}': 配分計算の最適化や事前計算を検討"
                )

        if not recommendations:
            recommendations.append("✅ 現在、明確なボトルネックは検出されていません")

        return recommendations

    def _log_analysis(self, analysis: Dict[str, Any]) -> None:
        """分析結果をログに記録。"""

        log_level = logging.WARNING if analysis["bottlenecks"] else logging.INFO

        summary = {
            "phase_count": len(analysis["phase_analysis"]),
            "bottleneck_count": len(analysis["bottlenecks"]),
            "high_severity_count": sum(
                1 for b in analysis["bottlenecks"] if b["severity"] == "high"
            ),
            "top_bottleneck": (
                analysis["bottlenecks"][0]["phase"] if analysis["bottlenecks"] else None
            ),
        }

        self.logger.log(
            log_level,
            f"Bottleneck analysis completed for {analysis['system']}: {summary['bottleneck_count']} bottlenecks found",
            extra={"analysis_summary": summary, "full_analysis": analysis},
        )

    def get_ui_data(self, system_name: str = "all") -> Dict[str, Any]:
        """UI表示用のデータを取得。"""

        analysis = self.analyze_bottlenecks(system_name)

        # UI用にフォーマット
        ui_data = {
            "timestamp": analysis["timestamp"],
            "system": system_name,
            "phases": [
                {
                    "name": phase,
                    "avg_duration": stats["avg_duration"],
                    "time_percentage": (
                        stats["total_time"]
                        / sum(
                            s["total_time"] for s in analysis["phase_analysis"].values()
                        )
                    )
                    * 100,
                    "trend": stats["recent_trend"],
                    "is_bottleneck": any(
                        b["phase"] == phase for b in analysis["bottlenecks"]
                    ),
                }
                for phase, stats in analysis["phase_analysis"].items()
            ],
            "bottlenecks": analysis["bottlenecks"],
            "recommendations": analysis["recommendations"],
        }

        # 時間割合で降順ソート
        ui_data["phases"].sort(key=lambda x: x["time_percentage"], reverse=True)

        return ui_data


class PredictiveProgressTracker:
    """予測進捗バー機能。

    過去の実行パターンと現在の処理速度から残り時間を予測表示。
    動的に更新される正確な進捗情報を提供。
    """

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path("logs/progress")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 履歴データ
        self.execution_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.max_history = 30  # 最新30回の実行を保持

        # 現在の実行状況
        self.current_executions: Dict[str, Dict[str, Any]] = {}
        self._progress_lock = threading.Lock()

        # ログ設定
        self.logger = logging.getLogger("progress_tracker")
        handler = logging.FileHandler(self.log_dir / "progress_predictions.jsonl")
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def start_tracking(
        self, operation: str, total_items: int, system_name: str = "unknown"
    ) -> str:
        """進捗追跡を開始。"""

        tracking_id = f"{operation}_{system_name}_{int(time.time())}"

        with self._progress_lock:
            self.current_executions[tracking_id] = {
                "operation": operation,
                "system": system_name,
                "total_items": total_items,
                "processed_items": 0,
                "start_time": time.perf_counter(),
                "last_update": time.perf_counter(),
                "phase_times": [],  # 各フェーズの処理時間
                "predicted_completion": None,
                "confidence_level": 0.0,
            }

        return tracking_id

    def update_progress(
        self, tracking_id: str, processed_items: int, phase: str = None
    ) -> Dict[str, Any]:
        """進捗を更新し、予測情報を返す。"""

        with self._progress_lock:
            if tracking_id not in self.current_executions:
                return {}

            execution = self.current_executions[tracking_id]
            current_time = time.perf_counter()

            # 基本情報更新
            execution["processed_items"] = processed_items
            execution["last_update"] = current_time

            # フェーズ情報記録
            if phase:
                execution["phase_times"].append(
                    {
                        "phase": phase,
                        "timestamp": current_time,
                        "items_processed": processed_items,
                    }
                )

            # 予測計算
            prediction = self._calculate_prediction(execution)
            execution.update(prediction)

            return self._format_progress_info(execution)

    def complete_tracking(self, tracking_id: str) -> None:
        """進捗追跡を完了し、履歴に記録。"""

        with self._progress_lock:
            if tracking_id not in self.current_executions:
                return

            execution = self.current_executions[tracking_id]
            completion_time = time.perf_counter()

            # 完了情報を履歴に記録
            history_record = {
                "operation": execution["operation"],
                "system": execution["system"],
                "total_items": execution["total_items"],
                "total_duration": completion_time - execution["start_time"],
                "phase_times": execution["phase_times"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "items_per_second": (
                    execution["total_items"]
                    / (completion_time - execution["start_time"])
                    if completion_time > execution["start_time"]
                    else 0
                ),
            }

            operation_key = f"{execution['operation']}_{execution['system']}"
            self.execution_history[operation_key].append(history_record)

            # 履歴サイズ制限
            if len(self.execution_history[operation_key]) > self.max_history:
                self.execution_history[operation_key] = self.execution_history[
                    operation_key
                ][-self.max_history :]

            # ログ記録
            self.logger.info(
                f"Completed tracking for {execution['operation']}",
                extra={"tracking_summary": history_record},
            )

            # 現在の実行から削除
            del self.current_executions[tracking_id]

    def _calculate_prediction(self, execution: Dict[str, Any]) -> Dict[str, Any]:
        """残り時間と完了予測を計算。"""

        current_time = time.perf_counter()
        elapsed = current_time - execution["start_time"]
        processed = execution["processed_items"]
        total = execution["total_items"]

        if processed == 0 or elapsed == 0:
            return {
                "predicted_completion": None,
                "estimated_remaining_seconds": None,
                "confidence_level": 0.0,
                "current_rate": 0.0,
            }

        # 現在の処理速度
        current_rate = processed / elapsed

        # 履歴ベースの予測
        operation_key = f"{execution['operation']}_{execution['system']}"
        historical_prediction = self._get_historical_prediction(
            operation_key, total, processed, elapsed
        )

        # 現在速度ベースの予測
        remaining_items = total - processed
        simple_remaining_seconds = (
            remaining_items / current_rate if current_rate > 0 else None
        )

        # 予測の統合
        if historical_prediction and simple_remaining_seconds:
            # 履歴データと現在速度を重み付け平均
            confidence = historical_prediction["confidence"]
            weight_historical = min(0.7, confidence)  # 最大70%の重み
            weight_current = 1.0 - weight_historical

            combined_remaining = (
                historical_prediction["remaining_seconds"] * weight_historical
                + simple_remaining_seconds * weight_current
            )

            final_confidence = (confidence + 0.3) / 2  # 現在速度での補正
        else:
            combined_remaining = simple_remaining_seconds
            final_confidence = 0.3 if simple_remaining_seconds else 0.0

        # 完了予定時刻
        predicted_completion = None
        if combined_remaining:
            predicted_completion = current_time + combined_remaining

        return {
            "predicted_completion": predicted_completion,
            "estimated_remaining_seconds": combined_remaining,
            "confidence_level": final_confidence,
            "current_rate": current_rate,
            "historical_info": historical_prediction,
        }

    def _get_historical_prediction(
        self, operation_key: str, total_items: int, processed_items: int, elapsed: float
    ) -> Optional[Dict[str, Any]]:
        """履歴データに基づく予測を取得。"""

        if operation_key not in self.execution_history:
            return None

        history = self.execution_history[operation_key]
        if len(history) < 3:  # 最低3回の履歴が必要
            return None

        # 類似サイズのタスクを検索
        similar_tasks = [
            h
            for h in history
            if abs(h["total_items"] - total_items) / max(total_items, 1)
            <= 0.3  # 30%以内の差
        ]

        if not similar_tasks:
            similar_tasks = history[-5:]  # 最新5回をフォールバック

        # 進捗率ベースの予測
        progress_ratio = processed_items / total_items if total_items > 0 else 0

        # 類似タスクの同じ進捗率での残り時間を推定
        remaining_estimates = []

        for task in similar_tasks:
            task_progress_ratio = progress_ratio
            expected_total_time = task["total_duration"]
            expected_elapsed = expected_total_time * task_progress_ratio
            expected_remaining = expected_total_time - expected_elapsed

            if expected_remaining > 0:
                # 現在の進捗速度を考慮した調整
                task_rate = task["items_per_second"]
                current_rate = processed_items / elapsed if elapsed > 0 else task_rate

                # 速度差による補正
                if task_rate > 0:
                    speed_factor = current_rate / task_rate
                    adjusted_remaining = expected_remaining / speed_factor
                    remaining_estimates.append(adjusted_remaining)

        if not remaining_estimates:
            return None

        # 統計計算
        avg_remaining = sum(remaining_estimates) / len(remaining_estimates)
        std_dev = self._calculate_std(remaining_estimates, avg_remaining)

        # 信頼度計算
        confidence = min(0.9, len(similar_tasks) / 10)  # サンプル数による信頼度
        if std_dev > 0:
            cv = std_dev / avg_remaining  # 変動係数
            confidence *= max(0.3, 1.0 - cv)  # 変動が大きいほど信頼度低下

        return {
            "remaining_seconds": avg_remaining,
            "confidence": confidence,
            "sample_count": len(remaining_estimates),
            "std_dev": std_dev,
        }

    def _calculate_std(self, values: List[float], mean: float) -> float:
        """標準偏差を計算。"""
        if len(values) <= 1:
            return 0.0

        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance**0.5

    def _format_progress_info(self, execution: Dict[str, Any]) -> Dict[str, Any]:
        """UI表示用の進捗情報をフォーマット。"""

        progress_percentage = (
            (execution["processed_items"] / execution["total_items"]) * 100
            if execution["total_items"] > 0
            else 0
        )

        # 残り時間の人間向け表示
        remaining_display = None
        if execution.get("estimated_remaining_seconds"):
            remaining_seconds = execution["estimated_remaining_seconds"]
            if remaining_seconds > 3600:
                remaining_display = f"{remaining_seconds/3600:.1f}時間"
            elif remaining_seconds > 60:
                remaining_display = f"{remaining_seconds/60:.1f}分"
            else:
                remaining_display = f"{remaining_seconds:.0f}秒"

        # 完了予定時刻
        completion_time_display = None
        if execution.get("predicted_completion"):
            completion_timestamp = execution["predicted_completion"]
            completion_dt = datetime.fromtimestamp(completion_timestamp)
            completion_time_display = completion_dt.strftime("%H:%M:%S")

        return {
            "tracking_id": next(
                k for k, v in self.current_executions.items() if v == execution
            ),
            "operation": execution["operation"],
            "system": execution["system"],
            "progress_percentage": round(progress_percentage, 1),
            "processed_items": execution["processed_items"],
            "total_items": execution["total_items"],
            "current_rate": round(execution.get("current_rate", 0), 2),
            "estimated_remaining_seconds": execution.get("estimated_remaining_seconds"),
            "remaining_display": remaining_display,
            "completion_time_display": completion_time_display,
            "confidence_level": round(execution.get("confidence_level", 0) * 100, 1),
            "phase_count": len(execution.get("phase_times", [])),
            "is_prediction_available": execution.get("estimated_remaining_seconds")
            is not None,
        }

    def get_all_active_progress(self) -> List[Dict[str, Any]]:
        """すべてのアクティブな進捗情報を取得。"""

        with self._progress_lock:
            return [
                self._format_progress_info(execution)
                for execution in self.current_executions.values()
            ]


class RealTimeMetricsCollector:
    """リアルタイムメトリクス収集・表示システム。

    CPU/メモリ/処理速度のリアルタイム監視とダッシュボード表示用
    データ収集機能。グラフ形式での可視化をサポート。
    """

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path("logs/realtime_metrics")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # リアルタイムデータ保存
        self.max_data_points = 100  # 最新100ポイントを保持

        # システムメトリクス履歴
        self.cpu_history: deque = deque(maxlen=self.max_data_points)
        self.memory_history: deque = deque(maxlen=self.max_data_points)
        self.throughput_history: deque = deque(maxlen=self.max_data_points)

        # システム別パフォーマンス履歴
        self.system_performance: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: {
                "duration": deque(maxlen=50),
                "throughput": deque(maxlen=50),
                "memory_delta": deque(maxlen=50),
                "timestamps": deque(maxlen=50),
            }
        )

        # アラート状態
        self.current_alerts: List[Dict[str, Any]] = []

        # エラー追跡用（ErrorLogger機能統合）
        self.error_count = 0
        self.errors_by_code: defaultdict = defaultdict(int)
        self.error_ring_buffer: deque = deque(maxlen=100)  # 最新100エラー保持

        # ロガー設定
        self.error_logger = logging.getLogger(f"{__name__}.errors")
        self.metrics = None  # メトリクス収集インスタンス（必要に応じて設定）

        # スレッドセーフ
        self._metrics_lock = threading.Lock()

        # 収集スレッド
        self.collection_active = False
        self.collection_thread: Optional[threading.Thread] = None
        self.collection_interval = 2.0  # 2秒間隔

        # ログ設定
        self.logger = logging.getLogger("realtime_metrics")
        handler = logging.FileHandler(self.log_dir / "realtime_metrics.jsonl")
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def start_collection(self) -> None:
        """リアルタイム収集を開始。"""

        if self.collection_active:
            return

        self.collection_active = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop, daemon=True
        )
        self.collection_thread.start()

        self.logger.info("Real-time metrics collection started")

    def stop_collection(self) -> None:
        """リアルタイム収集を停止。"""

        self.collection_active = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)

        self.logger.info("Real-time metrics collection stopped")

    def _collection_loop(self) -> None:
        """メトリクス収集ループ。"""

        while self.collection_active:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
                time.sleep(self.collection_interval)

    def _collect_system_metrics(self) -> None:
        """システムメトリクスを収集。"""

        if not psutil:
            return

        try:
            current_time = time.time()

            # システムリソース収集
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            with self._metrics_lock:
                # リアルタイムデータに追加
                self.cpu_history.append(
                    {"timestamp": current_time, "value": cpu_percent}
                )

                self.memory_history.append(
                    {"timestamp": current_time, "value": memory_percent}
                )

                # ディスクI/O情報（可能な場合）
                try:
                    disk_io = psutil.disk_io_counters()
                    network_io = psutil.net_io_counters()

                    # 追加のメトリクスをログに記録
                    extended_metrics = {
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory_percent,
                        "disk_read_mb": (
                            disk_io.read_bytes / 1024 / 1024 if disk_io else 0
                        ),
                        "disk_write_mb": (
                            disk_io.write_bytes / 1024 / 1024 if disk_io else 0
                        ),
                        "network_sent_mb": (
                            network_io.bytes_sent / 1024 / 1024 if network_io else 0
                        ),
                        "network_recv_mb": (
                            network_io.bytes_recv / 1024 / 1024 if network_io else 0
                        ),
                    }

                    self.logger.debug(
                        "System metrics collected", extra=extended_metrics
                    )

                except Exception:
                    pass  # ディスク/ネットワークI/O取得に失敗しても継続

        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")

    def record_system_performance(
        self,
        system_name: str,
        operation: str,
        duration: float,
        throughput: Optional[float] = None,
        memory_delta: Optional[float] = None,
    ) -> None:
        """システム別パフォーマンス情報を記録。"""

        current_time = time.time()

        with self._metrics_lock:
            system_data = self.system_performance[system_name]

            system_data["duration"].append(
                {"timestamp": current_time, "value": duration, "operation": operation}
            )

            system_data["timestamps"].append(current_time)

            if throughput is not None:
                system_data["throughput"].append(
                    {
                        "timestamp": current_time,
                        "value": throughput,
                        "operation": operation,
                    }
                )

                # 全体のスループット履歴にも追加
                self.throughput_history.append(
                    {
                        "timestamp": current_time,
                        "value": throughput,
                        "system": system_name,
                    }
                )

            if memory_delta is not None:
                system_data["memory_delta"].append(
                    {
                        "timestamp": current_time,
                        "value": memory_delta,
                        "operation": operation,
                    }
                )

    def update_alerts(self, alerts: List[Dict[str, Any]]) -> None:
        """現在のアラート状態を更新。"""

        with self._metrics_lock:
            self.current_alerts = alerts.copy()

    def get_dashboard_data(self, time_window_minutes: int = 10) -> Dict[str, Any]:
        """ダッシュボード表示用データを取得。"""

        current_time = time.time()
        time_cutoff = current_time - (time_window_minutes * 60)

        with self._metrics_lock:
            dashboard_data = {
                "timestamp": current_time,
                "time_window_minutes": time_window_minutes,
                "system_metrics": {
                    "cpu": self._filter_by_time(self.cpu_history, time_cutoff),
                    "memory": self._filter_by_time(self.memory_history, time_cutoff),
                    "throughput": self._filter_by_time(
                        self.throughput_history, time_cutoff
                    ),
                },
                "system_performance": {},
                "alerts": {
                    "count": len(self.current_alerts),
                    "by_severity": self._group_alerts_by_severity(),
                    "recent": self.current_alerts[-5:] if self.current_alerts else [],
                },
                "summary": self._calculate_summary_stats(time_cutoff),
            }

            # システム別パフォーマンス
            for system_name, system_data in self.system_performance.items():
                filtered_data = {}
                for metric_name, metric_data in system_data.items():
                    if metric_name == "timestamps":
                        continue
                    filtered_data[metric_name] = self._filter_by_time(
                        metric_data, time_cutoff
                    )

                if any(filtered_data.values()):  # データがある場合のみ含める
                    dashboard_data["system_performance"][system_name] = filtered_data

            return dashboard_data

    def _filter_by_time(
        self, data_history: deque, time_cutoff: float
    ) -> List[Dict[str, Any]]:
        """時間範囲でデータをフィルタリング。"""

        return [
            item for item in data_history if item.get("timestamp", 0) >= time_cutoff
        ]

    def _group_alerts_by_severity(self) -> Dict[str, int]:
        """アラートを重要度別にグループ化。"""

        severity_counts = {"critical": 0, "warning": 0, "info": 0}

        for alert in self.current_alerts:
            severity = alert.get("severity", "info")
            if severity in severity_counts:
                severity_counts[severity] += 1
            else:
                severity_counts["info"] += 1

        return severity_counts

    def _calculate_summary_stats(self, time_cutoff: float) -> Dict[str, Any]:
        """サマリー統計を計算。"""

        summary = {
            "avg_cpu": 0.0,
            "avg_memory": 0.0,
            "max_cpu": 0.0,
            "max_memory": 0.0,
            "total_throughput": 0.0,
            "active_systems": 0,
            "data_points": 0,
        }

        # CPU/メモリの統計
        recent_cpu = self._filter_by_time(self.cpu_history, time_cutoff)
        recent_memory = self._filter_by_time(self.memory_history, time_cutoff)
        recent_throughput = self._filter_by_time(self.throughput_history, time_cutoff)

        if recent_cpu:
            cpu_values = [item["value"] for item in recent_cpu]
            summary["avg_cpu"] = sum(cpu_values) / len(cpu_values)
            summary["max_cpu"] = max(cpu_values)

        if recent_memory:
            memory_values = [item["value"] for item in recent_memory]
            summary["avg_memory"] = sum(memory_values) / len(memory_values)
            summary["max_memory"] = max(memory_values)

        if recent_throughput:
            throughput_values = [item["value"] for item in recent_throughput]
            summary["total_throughput"] = sum(throughput_values)

        # アクティブシステム数
        summary["active_systems"] = len(
            [
                system
                for system, data in self.system_performance.items()
                if any(
                    self._filter_by_time(metric_data, time_cutoff)
                    for metric_name, metric_data in data.items()
                    if metric_name != "timestamps"
                )
            ]
        )

        summary["data_points"] = (
            len(recent_cpu) + len(recent_memory) + len(recent_throughput)
        )

        return summary

    def get_chart_data(
        self, metric_type: str = "cpu", time_window_minutes: int = 10
    ) -> Dict[str, Any]:
        """グラフ表示用の特定メトリクスデータを取得。"""

        current_time = time.time()
        time_cutoff = current_time - (time_window_minutes * 60)

        with self._metrics_lock:
            if metric_type == "cpu":
                data = self._filter_by_time(self.cpu_history, time_cutoff)
                unit = "%"
                title = "CPU使用率"
            elif metric_type == "memory":
                data = self._filter_by_time(self.memory_history, time_cutoff)
                unit = "%"
                title = "メモリ使用率"
            elif metric_type == "throughput":
                data = self._filter_by_time(self.throughput_history, time_cutoff)
                unit = "items/sec"
                title = "処理スループット"
            else:
                return {"error": f"Unknown metric type: {metric_type}"}

            # チャート形式のデータに変換
            chart_data = {
                "title": title,
                "unit": unit,
                "data_points": len(data),
                "time_range": {
                    "start": (
                        datetime.fromtimestamp(time_cutoff).isoformat()
                        if data
                        else None
                    ),
                    "end": datetime.fromtimestamp(current_time).isoformat(),
                },
                "series": [
                    {
                        "timestamp": datetime.fromtimestamp(
                            item["timestamp"]
                        ).isoformat(),
                        "value": item["value"],
                        "system": item.get("system"),  # スループットの場合のみ
                    }
                    for item in data
                ],
                "stats": {
                    "current": data[-1]["value"] if data else 0,
                    "average": (
                        sum(item["value"] for item in data) / len(data) if data else 0
                    ),
                    "min": min(item["value"] for item in data) if data else 0,
                    "max": max(item["value"] for item in data) if data else 0,
                },
            }

            return chart_data

    def _add_to_ring_buffer(self, error_data: Dict[str, Any]) -> None:
        """エラーデータをリングバッファに追加"""
        self.error_ring_buffer.append(error_data)

    def get_ring_buffer(self, last_n: int = 10) -> List[Dict[str, Any]]:
        """最新のエラーデータを取得"""
        return list(self.error_ring_buffer)[-last_n:]

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
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    system=error_context.get("system", "unknown"),
                    message=f"Exception: {str(error)}",
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
