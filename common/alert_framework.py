"""Alert Framework - リアルタイム監視ライブラリ。

実行時の各種メトリクスを監視し、閾値超過時にアラートを発火する
プラガブルな alert 設定フレームワーク。

使用例:
    from common.alert_framework import AlertManager, AlertCondition

    manager = AlertManager()
    manager.register_condition(
        AlertCondition(
            name="cache_freshness",
            metric_key="cache_age_seconds",
            operator=">",
            threshold=3600,
            action="log_warning"
        )
    )
    manager.check_metric("cache_age_seconds", 5400)  # Alert発火
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """アラートの重要度。"""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AlertAction(Enum):
    """アラートが発火した時の処理。"""

    LOG = "log"
    LOG_WARNING = "log_warning"
    LOG_ERROR = "log_error"
    NOTIFY = "notify"  # 通知（Slack など）
    STOP = "stop"  # 処理停止
    CUSTOM = "custom"  # カスタム callback


class ComparisonOperator(Enum):
    """比較演算子。"""

    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="


@dataclass
class AlertCondition:
    """アラート条件の定義。

    Attributes:
        name: 条件名（識別用）
        metric_key: 監視メトリクスキー
        operator: 比較演算子
        threshold: 閾値
        severity: 重要度（デフォルト WARNING）
        action: 発火時のアクション（デフォルト LOG_WARNING）
        description: 説明（optional）
    """

    name: str
    metric_key: str
    operator: ComparisonOperator
    threshold: float
    severity: AlertSeverity = AlertSeverity.WARNING
    action: AlertAction = AlertAction.LOG_WARNING
    description: str | None = None
    custom_callback: Callable[[dict[str, Any]], None] | None = None

    @staticmethod
    def from_config(
        name: str,
        metric_key: str,
        operator: ComparisonOperator | str,
        threshold: float | str | int,
        **kwargs,
    ) -> AlertCondition:
        """Configuration from flexible types.

        Args:
            name: Condition name
            metric_key: Metric key
            operator: Comparison operator (enum or string)
            threshold: Threshold value
            **kwargs: Additional fields (severity, action, etc.)

        Returns:
            AlertCondition instance
        """
        # Convert operator if needed
        op = operator
        if isinstance(op, str):
            op_map = {
                ">": ComparisonOperator.GREATER_THAN,
                ">=": ComparisonOperator.GREATER_EQUAL,
                "<": ComparisonOperator.LESS_THAN,
                "<=": ComparisonOperator.LESS_EQUAL,
                "==": ComparisonOperator.EQUAL,
                "!=": ComparisonOperator.NOT_EQUAL,
            }
            op = op_map.get(op, ComparisonOperator.GREATER_THAN)

        # Convert threshold to float
        threshold_float = float(threshold)

        return AlertCondition(
            name=name,
            metric_key=metric_key,
            operator=op,
            threshold=threshold_float,
            **kwargs,
        )

    def check(self, value: float) -> bool:
        """値が条件を満たすか判定。

        Args:
            value: チェック対象の値

        Returns:
            条件を満たす場合 True
        """
        try:
            value_float = float(value)
        except (ValueError, TypeError):
            return False

        op_fn_map = {
            ComparisonOperator.GREATER_THAN: lambda v, t: v > t,
            ComparisonOperator.GREATER_EQUAL: lambda v, t: v >= t,
            ComparisonOperator.LESS_THAN: lambda v, t: v < t,
            ComparisonOperator.LESS_EQUAL: lambda v, t: v <= t,
            ComparisonOperator.EQUAL: lambda v, t: abs(v - t) < 1e-9,
            ComparisonOperator.NOT_EQUAL: lambda v, t: abs(v - t) >= 1e-9,
        }

        op_fn = op_fn_map.get(self.operator)
        if op_fn is None:
            logger.warning(f"Unknown operator: {self.operator}")
            return False

        return op_fn(value_float, self.threshold)

    def __str__(self) -> str:
        return (
            f"AlertCondition(name={self.name}, "
            f"{self.metric_key} {self.operator.value} {self.threshold})"
        )


@dataclass
class AlertEvent:
    """発火したアラートイベント。"""

    condition: AlertCondition
    metric_value: float
    timestamp: datetime
    triggered: bool = True

    def __str__(self) -> str:
        return (
            f"[{self.timestamp.isoformat()}] {self.condition.name}: "
            f"{self.condition.metric_key}={self.metric_value} "
            f"(condition: {self.condition.operator.value} {self.condition.threshold})"
        )


class AlertManager:
    """アラート管理エンジン。

    複数の AlertCondition を登録し、メトリクスをチェック。
    条件が満たされたら適切なアクション（ログ、通知など）を実行。
    """

    def __init__(self):
        self.conditions: dict[str, AlertCondition] = {}
        self.events: list[AlertEvent] = []
        self.notifier_callback: Callable[[AlertEvent], None] | None = None

    def register_condition(self, condition: AlertCondition) -> None:
        """アラート条件を登録。

        Args:
            condition: AlertCondition インスタンス
        """
        self.conditions[condition.name] = condition
        logger.debug(f"Registered alert condition: {condition}")

    def register_notifier(self, callback: Callable[[AlertEvent], None]) -> None:
        """通知 callback を登録（Slack など）。

        Args:
            callback: AlertEvent を受け取る callback 関数
        """
        self.notifier_callback = callback

    def check_metric(self, metric_key: str, value: float) -> list[AlertEvent]:
        """メトリクス値をチェック。

        Args:
            metric_key: メトリクスキー
            value: 値

        Returns:
            発火したアラートイベントのリスト
        """
        triggered_events: list[AlertEvent] = []

        for cond in self.conditions.values():
            if cond.metric_key != metric_key:
                continue

            if cond.check(value):
                event = AlertEvent(
                    condition=cond,
                    metric_value=value,
                    timestamp=datetime.now(timezone.utc),
                    triggered=True,
                )
                triggered_events.append(event)
                self.events.append(event)

                # アクション実行
                self._execute_action(event, cond)

        return triggered_events

    def _execute_action(self, event: AlertEvent, condition: AlertCondition) -> None:
        """アラート発火時のアクション実行。

        Args:
            event: AlertEvent
            condition: AlertCondition
        """
        action = condition.action

        if action == AlertAction.LOG:
            logger.info(str(event))
        elif action == AlertAction.LOG_WARNING:
            logger.warning(str(event))
        elif action == AlertAction.LOG_ERROR:
            logger.error(str(event))
        elif action == AlertAction.NOTIFY:
            if self.notifier_callback:
                try:
                    self.notifier_callback(event)
                except Exception as e:
                    logger.exception(f"Notifier error: {e}")
            else:
                logger.warning("Notifier callback not registered")
        elif action == AlertAction.STOP:
            logger.critical(f"CRITICAL ALERT - STOPPING: {event}")
            raise RuntimeError(f"Critical alert triggered: {condition.name}")
        elif action == AlertAction.CUSTOM:
            if condition.custom_callback:
                try:
                    condition.custom_callback(
                        {
                            "metric_key": condition.metric_key,
                            "value": event.metric_value,
                            "condition_name": condition.name,
                        }
                    )
                except Exception as e:
                    logger.exception(f"Custom callback error: {e}")

    def get_events(self, severity: AlertSeverity | None = None) -> list[AlertEvent]:
        """記録されたイベントを取得。

        Args:
            severity: フィルター対象の重要度（None = すべて）

        Returns:
            AlertEvent のリスト
        """
        if severity is None:
            return self.events
        return [e for e in self.events if e.condition.severity == severity]

    def clear_events(self) -> None:
        """イベント履歴をクリア。"""
        self.events.clear()

    def get_stats(self) -> dict[str, Any]:
        """アラート統計情報を取得。

        Returns:
            統計情報（イベント数など）
        """
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len(
                [e for e in self.events if e.condition.severity == severity]
            )

        return {
            "total_events": len(self.events),
            "by_severity": severity_counts,
            "registered_conditions": len(self.conditions),
        }


# よく使う Condition プリセット
def make_freshness_alert(
    metric_key: str = "cache_age_seconds",
    max_age_seconds: int = 3600,
) -> AlertCondition:
    """キャッシュ freshness アラート。

    Args:
        metric_key: Metric キー（default: "cache_age_seconds"）
        max_age_seconds: 最大許容経過秒数

    Returns:
        AlertCondition
    """
    return AlertCondition.from_config(
        name="cache_freshness",
        metric_key=metric_key,
        operator=">",
        threshold=max_age_seconds,
        severity=AlertSeverity.WARNING,
        action=AlertAction.LOG_WARNING,
        description=f"Cache age exceeded {max_age_seconds}s",
    )


def make_latency_alert(
    metric_key: str = "operation_latency_ms",
    max_latency_ms: int = 1000,
) -> AlertCondition:
    """遅延アラート。

    Args:
        metric_key: Metric キー（default: "operation_latency_ms"）
        max_latency_ms: 最大許容遅延時間（ミリ秒）

    Returns:
        AlertCondition
    """
    return AlertCondition.from_config(
        name="operation_latency",
        metric_key=metric_key,
        operator=">",
        threshold=max_latency_ms,
        severity=AlertSeverity.WARNING,
        action=AlertAction.LOG_WARNING,
        description=f"Operation latency exceeded {max_latency_ms}ms",
    )


def make_memory_alert(
    metric_key: str = "memory_usage_mb",
    max_memory_mb: int = 1024,
) -> AlertCondition:
    """メモリ使用量アラート。

    Args:
        metric_key: Metric キー（default: "memory_usage_mb"）
        max_memory_mb: 最大許容メモリ（MB）

    Returns:
        AlertCondition
    """
    return AlertCondition.from_config(
        name="memory_usage",
        metric_key=metric_key,
        operator=">",
        threshold=max_memory_mb,
        severity=AlertSeverity.ERROR,
        action=AlertAction.LOG_ERROR,
        description=f"Memory usage exceeded {max_memory_mb}MB",
    )


__all__ = [
    "AlertSeverity",
    "AlertAction",
    "ComparisonOperator",
    "AlertCondition",
    "AlertEvent",
    "AlertManager",
    "make_freshness_alert",
    "make_latency_alert",
    "make_memory_alert",
]
