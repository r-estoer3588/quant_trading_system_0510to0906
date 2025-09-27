"""指標計算メトリクス収集モジュール

add_indicators の計算統計を収集・可視化する機能を提供。
計算済み/新規計算/スキップ/所要時間を記録し、CSV+ログで出力。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class IndicatorMetrics:
    """指標計算メトリクス"""

    symbol: str
    total_indicators: int = 0
    existing_count: int = 0  # 事前計算済み（スキップ）
    computed_count: int = 0  # 新規計算
    failed_count: int = 0  # 計算失敗
    computation_time: float = 0.0  # 総計算時間（秒）
    timestamp: str = ""

    # 詳細内訳
    existing_indicators: list[str] = field(default_factory=list)
    computed_indicators: list[str] = field(default_factory=list)
    failed_indicators: list[str] = field(default_factory=list)

    @property
    def skip_rate(self) -> float:
        """スキップ率（％）"""
        if self.total_indicators == 0:
            return 0.0
        return (self.existing_count / self.total_indicators) * 100

    @property
    def compute_rate(self) -> float:
        """新規計算率（％）"""
        if self.total_indicators == 0:
            return 0.0
        return (self.computed_count / self.total_indicators) * 100

    @property
    def success_rate(self) -> float:
        """成功率（％）"""
        if self.total_indicators == 0:
            return 100.0
        return ((self.existing_count + self.computed_count) / self.total_indicators) * 100


class IndicatorMetricsCollector:
    """指標メトリクス収集クラス"""

    def __init__(self, output_dir: Path | None = None):
        """
        Args:
            output_dir: メトリクス出力ディレクトリ（デフォルト: logs/indicator_metrics/）
        """
        self.output_dir = output_dir or Path("logs/indicator_metrics")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ログ設定
        self.logger = logging.getLogger("indicator_metrics")
        if not self.logger.handlers:
            handler = logging.FileHandler(self.output_dir / "indicator_metrics.log")
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # メトリクス収集リスト
        self.metrics_history: list[IndicatorMetrics] = []

    def wrap_add_indicators(self, add_indicators_func):
        """add_indicators関数をラップしてメトリクス収集"""

        def wrapper(df, symbol: str = "UNKNOWN", **kwargs):
            """メトリクス収集付きadd_indicators

            Args:
                df: 価格データDataFrame
                symbol: シンボル名（メトリクス識別用）
                **kwargs: add_indicators への追加引数

            Returns:
                指標付加後のDataFrame
            """
            if df is None or df.empty:
                self.logger.warning(f"{symbol}: Empty DataFrame")
                return df

            # 計算開始前の状態記録
            initial_columns = set(df.columns)
            start_time = time.time()

            # メトリクス初期化
            metrics = IndicatorMetrics(symbol=symbol, timestamp=pd.Timestamp.now().isoformat())

            # 期待される全指標リスト（add_indicatorsが生成する可能性のある列）
            expected_indicators = self._get_expected_indicators()

            # 計算前の既存指標をカウント
            for indicator in expected_indicators:
                if indicator in initial_columns:
                    metrics.existing_indicators.append(indicator)
                    metrics.existing_count += 1

            try:
                # 実際のadd_indicators実行
                result_df = add_indicators_func(df, **kwargs)

                if result_df is None:
                    self.logger.error(f"{symbol}: add_indicators returned None")
                    return df

                # 計算後の状態確認
                final_columns = set(result_df.columns)
                new_columns = final_columns - initial_columns

                # 新規計算された指標をカウント
                for indicator in expected_indicators:
                    if indicator in new_columns:
                        metrics.computed_indicators.append(indicator)
                        metrics.computed_count += 1
                    elif indicator not in initial_columns and indicator not in final_columns:
                        # 期待されていたが生成されなかった指標
                        metrics.failed_indicators.append(indicator)
                        metrics.failed_count += 1

                # 計算時間記録
                metrics.computation_time = time.time() - start_time
                metrics.total_indicators = len(expected_indicators)

                # メトリクス記録
                self._record_metrics(metrics)

                # ログ出力
                self.logger.info(
                    f"{symbol}: "
                    f"Total={metrics.total_indicators}, "
                    f"Existing={metrics.existing_count} ({metrics.skip_rate:.1f}%), "
                    f"Computed={metrics.computed_count} ({metrics.compute_rate:.1f}%), "
                    f"Failed={metrics.failed_count}, "
                    f"Time={metrics.computation_time:.3f}s"
                )

                return result_df

            except Exception as e:
                # エラー時も記録
                metrics.computation_time = time.time() - start_time
                metrics.total_indicators = len(expected_indicators)
                metrics.failed_count = metrics.total_indicators - metrics.existing_count
                metrics.failed_indicators = [
                    i for i in expected_indicators if i not in metrics.existing_indicators
                ]

                self._record_metrics(metrics)
                self.logger.error(f"{symbol}: add_indicators failed: {e}")
                raise

        return wrapper

    def _get_expected_indicators(self) -> list[str]:
        """add_indicatorsが生成する可能性のある全指標リスト"""
        indicators = []

        # ATR系
        for w in [10, 20, 40, 50]:
            indicators.append(f"atr{w}")

        # SMA系
        for w in [25, 50, 100, 150, 200]:
            indicators.append(f"sma{w}")

        # ROC系
        indicators.append("roc200")

        # RSI系
        for w in [3, 4]:
            indicators.append(f"rsi{w}")

        # ADX系
        for w in [7]:
            indicators.append(f"adx{w}")

        # 売買代金系
        for w in [20, 50]:
            indicators.append(f"dollarvolume{w}")

        # 平均出来高系
        for w in [50]:
            indicators.append(f"avgvolume{w}")

        # その他
        indicators.extend(
            [
                "atr_ratio",
                "atr_pct",
                "return_3d",
                "return_6d",
                "return_pct",
                "uptwodays",
                "twodayup",
                "drop3d",
                "hv50",
                "min_50",
                "max_70",
            ]
        )

        return indicators

    def _record_metrics(self, metrics: IndicatorMetrics):
        """メトリクスを履歴に記録"""
        self.metrics_history.append(metrics)

        # 定期的にCSV出力（100件ごと）
        if len(self.metrics_history) % 100 == 0:
            self.export_metrics()

    def export_metrics(self, filename: str | None = None):
        """メトリクスをCSV出力"""
        if not self.metrics_history:
            return

        filename = (
            filename or f"indicator_metrics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        filepath = self.output_dir / filename

        # DataFrameに変換
        data = []
        for m in self.metrics_history:
            data.append(
                {
                    "timestamp": m.timestamp,
                    "symbol": m.symbol,
                    "total_indicators": m.total_indicators,
                    "existing_count": m.existing_count,
                    "computed_count": m.computed_count,
                    "failed_count": m.failed_count,
                    "skip_rate": m.skip_rate,
                    "compute_rate": m.compute_rate,
                    "success_rate": m.success_rate,
                    "computation_time": m.computation_time,
                    "existing_indicators": "|".join(m.existing_indicators),
                    "computed_indicators": "|".join(m.computed_indicators),
                    "failed_indicators": "|".join(m.failed_indicators),
                }
            )

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)

        self.logger.info(f"Metrics exported to {filepath} ({len(data)} records)")

    def get_summary_stats(self) -> dict[str, Any]:
        """サマリー統計を取得"""
        if not self.metrics_history:
            return {}

        df = pd.DataFrame(
            [
                {
                    "skip_rate": m.skip_rate,
                    "compute_rate": m.compute_rate,
                    "success_rate": m.success_rate,
                    "computation_time": m.computation_time,
                    "total_indicators": m.total_indicators,
                }
                for m in self.metrics_history
            ]
        )

        return {
            "total_symbols": len(self.metrics_history),
            "avg_skip_rate": df["skip_rate"].mean(),
            "avg_compute_rate": df["compute_rate"].mean(),
            "avg_success_rate": df["success_rate"].mean(),
            "avg_computation_time": df["computation_time"].mean(),
            "total_computation_time": df["computation_time"].sum(),
            "max_computation_time": df["computation_time"].max(),
            "min_computation_time": df["computation_time"].min(),
        }


# グローバルインスタンス（シングルトン的利用）
_global_collector: IndicatorMetricsCollector | None = None


def get_metrics_collector() -> IndicatorMetricsCollector:
    """グローバルメトリクス収集インスタンス取得"""
    global _global_collector
    if _global_collector is None:
        _global_collector = IndicatorMetricsCollector()
    return _global_collector


def create_instrumented_add_indicators():
    """メトリクス収集機能付きadd_indicators作成

    Returns:
        メトリクス収集機能付きadd_indicators関数

    Usage:
        from common.indicator_metrics import create_instrumented_add_indicators
        add_indicators = create_instrumented_add_indicators()

        # 通常通り使用（symbolパラメータ追加）
        result_df = add_indicators(df, symbol="AAPL")
    """
    from indicators_common import add_indicators as original_add_indicators

    collector = get_metrics_collector()
    return collector.wrap_add_indicators(original_add_indicators)
