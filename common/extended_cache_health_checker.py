"""
拡張キャッシュヘルスチェッカー - 包括的品質監視システム

既存のCacheHealthCheckerを拡張し、詳細分析・レポート・サンプリング機能を追加
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import random

import numpy as np
import pandas as pd

from common.cache_health_checker import CacheHealthChecker
from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ExtendedHealthMetrics:
    """拡張健全性メトリクス"""

    symbol: str
    profile: str
    file_path: str

    # 基本統計
    total_rows: int
    total_columns: int
    file_size_mb: float

    # NaN詳細分析
    nan_rate_overall: float
    nan_columns: dict[str, float]  # カラム別NaN率
    columns_with_high_nan: list[str]  # 高NaN率カラム

    # カラム完全性
    expected_columns: list[str]
    missing_columns: list[str]
    unexpected_columns: list[str]

    # データ正規化検証
    price_anomalies: dict[str, int]
    volume_anomalies: int
    indicator_anomalies: dict[str, int]

    # 時系列品質
    date_gaps: int  # 欠損日数
    duplicate_dates: int  # 重複日数
    chronological_order: bool  # 日付順序

    # 既存チェック結果
    basic_health_results: dict[str, bool]

    # メタデータ
    analysis_timestamp: str

    def to_dict(self) -> dict:
        """辞書形式変換"""
        return asdict(self)


class ExtendedCacheHealthChecker:
    """拡張キャッシュ健全性チェッカー"""

    def __init__(
        self,
        nan_threshold: float = 0.5,
        sample_size: int | None = None,
        max_workers: int = 4,
    ):
        """
        Args:
            nan_threshold: 高NaN率閾値
            sample_size: サンプリングサイズ
            max_workers: 並列ワーカー数
        """
        self.settings = get_settings(create_dirs=True)
        self.nan_threshold = nan_threshold
        self.sample_size = sample_size
        self.max_workers = max_workers

        # 既存ヘルスチェッカーを利用
        self.basic_checker = CacheHealthChecker("[ExtendedChecker]")

        # 期待カラム定義
        self.expected_base_columns = {
            # OHLCV
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            # 基本指標
            "sma5",
            "sma10",
            "sma20",
            "sma25",
            "sma50",
            "sma100",
            "sma150",
            "sma200",
            "ema12",
            "ema20",
            "ema26",
            "ema50",
            "rsi3",
            "rsi4",
            "rsi14",
            "macd",
            "macd_signal",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "atr10",
            "atr14",
            "atr20",
            "atr40",
            "atr50",
            "obv",
            "stoch_k",
            "stoch_d",
            "williams_r",
            "cci",
            "roc20",
            "roc200",
            "mfi",
            "tsi",
            "ultimate_oscillator",
            "hv50",
            "dollarvolume20",
            "dollarvolume50",
            "avgvolume50",
            "return_pct",
            "return_3d",
            "return_6d",
            "drop3d",
            "atr_ratio",
            "atr_pct",
            "adx7",
        }

        # プロファイル別の期待カラム
        self.profile_expected_columns = {
            "base": self.expected_base_columns,
            "rolling": self.expected_base_columns,  # rollingも同様の内容を期待
            "full_backup": self.expected_base_columns,
        }

    def _get_sample_files(self, cache_dir: Path, profile: str) -> list[Path]:
        """ファイルサンプリング"""
        if not cache_dir.exists():
            logger.warning(f"Cache directory not found: {cache_dir}")
            return []

        # CSVファイルを探索
        csv_files = list(cache_dir.glob("*.csv"))
        if not csv_files:
            logger.warning(f"No CSV files found in {cache_dir}")
            return []

        logger.info(f"Found {len(csv_files)} CSV files in {cache_dir}")

        # サンプリング適用
        if self.sample_size is None or len(csv_files) <= self.sample_size:
            return csv_files

        # ランダムサンプリング
        sampled = random.sample(csv_files, self.sample_size)
        logger.info(f"Sampled {len(sampled)} files for analysis")
        return sampled

    def _analyze_time_series_quality(self, df: pd.DataFrame) -> tuple[int, int, bool]:
        """時系列品質分析"""
        date_gaps = 0
        duplicate_dates = 0
        chronological_order = True

        if "Date" not in df.columns or df.empty:
            return date_gaps, duplicate_dates, chronological_order

        try:
            # 日付列を変換
            dates = pd.to_datetime(df["Date"], errors="coerce").dropna()
            if len(dates) < 2:
                return date_gaps, duplicate_dates, chronological_order

            # 重複日付チェック
            duplicate_dates = len(dates) - len(dates.unique())

            # 時系列順序チェック
            chronological_order = dates.is_monotonic_increasing

            # 日付ギャップ分析（営業日ベース想定）
            unique_dates = dates.sort_values().unique()
            if len(unique_dates) > 1:
                date_range = pd.date_range(
                    start=unique_dates[0],
                    end=unique_dates[-1],
                    freq="B",  # 営業日
                )
                expected_dates = len(date_range)
                actual_dates = len(unique_dates)
                date_gaps = max(0, expected_dates - actual_dates)

        except Exception as e:
            logger.debug(f"Time series analysis failed: {e}")

        return date_gaps, duplicate_dates, chronological_order

    def _analyze_data_anomalies(
        self, df: pd.DataFrame
    ) -> tuple[dict[str, int], int, dict[str, int]]:
        """データ異常値分析"""
        price_anomalies = {}
        volume_anomalies = 0
        indicator_anomalies = {}

        # 価格異常値（OHLC）
        price_cols = ["Open", "High", "Low", "Close"]
        for col in price_cols:
            if col in df.columns:
                try:
                    prices = pd.to_numeric(df[col], errors="coerce").dropna()
                    if len(prices) > 0:
                        anomaly_count = (
                            (prices <= 0).sum()  # 非正値
                            + (prices > 1e6).sum()  # 異常高値
                            + (prices < 0.01).sum()  # 異常低値
                        )
                        price_anomalies[col] = int(anomaly_count)
                except Exception:
                    price_anomalies[col] = 0

        # ボリューム異常値
        if "Volume" in df.columns:
            try:
                volumes = pd.to_numeric(df["Volume"], errors="coerce").dropna()
                if len(volumes) > 0:
                    volume_anomalies = int(
                        (volumes < 0).sum() + (volumes > 1e12).sum()  # 負値  # 異常高値
                    )
            except Exception:
                volume_anomalies = 0

        # 指標異常値
        indicator_checks = {
            "rsi14": lambda x: ((x < 0) | (x > 100)).sum(),
            "rsi3": lambda x: ((x < 0) | (x > 100)).sum(),
            "rsi4": lambda x: ((x < 0) | (x > 100)).sum(),
            "stoch_k": lambda x: ((x < 0) | (x > 100)).sum(),
            "stoch_d": lambda x: ((x < 0) | (x > 100)).sum(),
            "williams_r": lambda x: ((x < -100) | (x > 0)).sum(),
        }

        for col, check_func in indicator_checks.items():
            if col in df.columns:
                try:
                    values = pd.to_numeric(df[col], errors="coerce").dropna()
                    if len(values) > 0:
                        indicator_anomalies[col] = int(check_func(values))
                except Exception:
                    indicator_anomalies[col] = 0

        return price_anomalies, volume_anomalies, indicator_anomalies

    def _analyze_single_file(
        self, file_path: Path, profile: str
    ) -> ExtendedHealthMetrics | None:
        """単一ファイル分析"""
        try:
            # ファイル読み込み
            df = pd.read_csv(file_path)
            symbol = file_path.stem

            # 既存ヘルスチェック実行
            basic_results = self.basic_checker.check_dataframe_health(
                df, symbol, profile
            )

            # 基本統計
            total_rows = len(df)
            total_columns = len(df.columns)
            file_size_mb = file_path.stat().st_size / (1024 * 1024)

            # NaN詳細分析
            nan_counts = df.isnull().sum()
            total_cells = total_rows * total_columns
            overall_nan_rate = (
                df.isnull().sum().sum() / total_cells if total_cells > 0 else 0
            )

            # カラム別NaN率
            nan_columns = {}
            high_nan_columns = []
            for col in df.columns:
                if total_rows > 0:
                    nan_rate = nan_counts[col] / total_rows
                    if nan_rate > 0:
                        nan_columns[col] = round(nan_rate, 4)
                    if nan_rate > self.nan_threshold:
                        high_nan_columns.append(col)

            # カラム完全性チェック
            actual_columns = set(df.columns)
            expected_columns = self.profile_expected_columns.get(profile, set())
            missing_columns = list(expected_columns - actual_columns)
            unexpected_columns = list(actual_columns - expected_columns)

            # 時系列品質分析
            date_gaps, duplicate_dates, chronological_order = (
                self._analyze_time_series_quality(df)
            )

            # データ異常値分析
            price_anomalies, volume_anomalies, indicator_anomalies = (
                self._analyze_data_anomalies(df)
            )

            return ExtendedHealthMetrics(
                symbol=symbol,
                profile=profile,
                file_path=str(file_path),
                total_rows=total_rows,
                total_columns=total_columns,
                file_size_mb=round(file_size_mb, 3),
                nan_rate_overall=round(overall_nan_rate, 4),
                nan_columns=nan_columns,
                columns_with_high_nan=high_nan_columns,
                expected_columns=list(expected_columns),
                missing_columns=missing_columns,
                unexpected_columns=unexpected_columns,
                price_anomalies=price_anomalies,
                volume_anomalies=volume_anomalies,
                indicator_anomalies=indicator_anomalies,
                date_gaps=date_gaps,
                duplicate_dates=duplicate_dates,
                chronological_order=chronological_order,
                basic_health_results=basic_results,
                analysis_timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {str(e)}")
            return None

    def analyze_profile(self, profile: str) -> list[ExtendedHealthMetrics]:
        """プロファイル分析（並列処理）"""
        cache_dir = self.settings.DATA_CACHE_DIR / profile
        sample_files = self._get_sample_files(cache_dir, profile)

        if not sample_files:
            logger.warning(f"No files to analyze in {profile}")
            return []

        logger.info(f"Analyzing {len(sample_files)} files in {profile} profile")

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(
                    self._analyze_single_file, file_path, profile
                ): file_path
                for file_path in sample_files
            }

            for future in as_completed(future_to_file):
                result = future.result()
                if result:
                    results.append(result)

        logger.info(f"Successfully analyzed {len(results)}/{len(sample_files)} files")
        return results

    def generate_comprehensive_report(
        self, all_results: dict[str, list[ExtendedHealthMetrics]]
    ) -> dict:
        """包括的レポート生成"""
        report = {
            "analysis_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_files_analyzed": sum(
                    len(results) for results in all_results.values()
                ),
                "sampling_enabled": self.sample_size is not None,
                "sample_size": self.sample_size,
                "nan_threshold": self.nan_threshold,
            },
            "profile_summaries": {},
            "overall_statistics": {},
            "recommendations": [],
        }

        # プロファイル別サマリー
        all_metrics = []
        for profile, metrics_list in all_results.items():
            if not metrics_list:
                report["profile_summaries"][profile] = {"status": "no_data"}
                continue

            all_metrics.extend(metrics_list)

            # 統計計算
            total_files = len(metrics_list)
            avg_nan_rate = np.mean([m.nan_rate_overall for m in metrics_list])
            files_with_high_nan = sum(
                1 for m in metrics_list if m.columns_with_high_nan
            )
            avg_file_size = np.mean([m.file_size_mb for m in metrics_list])
            total_missing_cols = sum(len(m.missing_columns) for m in metrics_list)
            files_with_date_issues = sum(
                1
                for m in metrics_list
                if m.date_gaps > 0 or m.duplicate_dates > 0 or not m.chronological_order
            )

            profile_summary = {
                "total_files": total_files,
                "average_nan_rate": round(avg_nan_rate, 4),
                "files_with_high_nan": files_with_high_nan,
                "high_nan_percentage": round(
                    files_with_high_nan / total_files * 100, 2
                ),
                "average_file_size_mb": round(avg_file_size, 3),
                "total_missing_columns": total_missing_cols,
                "files_with_date_issues": files_with_date_issues,
                "date_issues_percentage": round(
                    files_with_date_issues / total_files * 100, 2
                ),
            }

            report["profile_summaries"][profile] = profile_summary

        # 全体統計
        if all_metrics:
            report["overall_statistics"] = {
                "total_symbols_analyzed": len(set(m.symbol for m in all_metrics)),
                "average_rows_per_file": round(
                    np.mean([m.total_rows for m in all_metrics]), 1
                ),
                "average_columns_per_file": round(
                    np.mean([m.total_columns for m in all_metrics]), 1
                ),
                "overall_health_rate": round(
                    sum(
                        1
                        for m in all_metrics
                        if m.basic_health_results.get("overall_health", False)
                    )
                    / len(all_metrics)
                    * 100,
                    2,
                ),
            }

        # 推奨事項生成
        recommendations = []
        for profile, summary in report["profile_summaries"].items():
            if summary.get("status") == "no_data":
                continue

            if summary["high_nan_percentage"] > 10:
                recommendations.append(
                    f"{profile}: 高NaN率ファイルが{summary['high_nan_percentage']:.1f}% - 指標計算パラメータの見直し推奨"
                )

            if summary["total_missing_columns"] > 0:
                recommendations.append(
                    f"{profile}: {summary['total_missing_columns']}個の欠損カラム検出 - スキーマ統一が必要"
                )

            if summary["date_issues_percentage"] > 5:
                recommendations.append(
                    f"{profile}: {summary['date_issues_percentage']:.1f}%のファイルに日付問題 - 時系列データ品質の改善推奨"
                )

        report["recommendations"] = recommendations
        return report

    def export_results(
        self,
        all_results: dict[str, list[ExtendedHealthMetrics]],
        report: dict,
        output_dir: Path | None = None,
    ) -> tuple[Path, Path]:
        """結果エクスポート"""
        if output_dir is None:
            output_dir = self.settings.LOGS_DIR / "extended_cache_health"
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 詳細CSV出力
        csv_path = output_dir / f"extended_health_details_{timestamp}.csv"
        all_records = []
        for _profile, metrics_list in all_results.items():
            for metrics in metrics_list:
                all_records.append(metrics.to_dict())

        if all_records:
            df = pd.json_normalize(all_records)
            df.to_csv(csv_path, index=False)

        # レポートJSON出力
        json_path = output_dir / f"extended_health_report_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return csv_path, json_path


def main():
    """CLI実行"""
    import argparse

    parser = argparse.ArgumentParser(description="Extended Cache Health Checker")
    parser.add_argument(
        "--profiles",
        nargs="*",
        choices=["base", "rolling", "full_backup"],
        default=["base", "rolling"],
        help="分析対象プロファイル",
    )
    parser.add_argument(
        "--sample", type=int, default=10, help="サンプリングサイズ（デフォルト: 10）"
    )
    parser.add_argument(
        "--nan-threshold", type=float, default=0.5, help="高NaN率の閾値"
    )
    parser.add_argument("--workers", type=int, default=4, help="並列ワーカー数")

    args = parser.parse_args()

    # ロギング設定
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # 分析実行
    checker = ExtendedCacheHealthChecker(
        nan_threshold=args.nan_threshold,
        sample_size=args.sample,
        max_workers=args.workers,
    )

    print(f"拡張キャッシュヘルスチェック開始 - プロファイル: {args.profiles}")
    if args.sample:
        print(f"サンプリングサイズ: {args.sample}")

    all_results = {}
    for profile in args.profiles:
        print(f"\n{profile} プロファイル分析中...")
        all_results[profile] = checker.analyze_profile(profile)

    # レポート生成・出力
    report = checker.generate_comprehensive_report(all_results)
    csv_path, json_path = checker.export_results(all_results, report)

    # 結果表示
    print("\n=== 拡張キャッシュヘルス分析結果 ===")
    print(f"総分析ファイル数: {report['analysis_metadata']['total_files_analyzed']}")

    for profile, summary in report["profile_summaries"].items():
        if summary.get("status") == "no_data":
            print(f"\n{profile.upper()}: データなし")
            continue

        print(f"\n{profile.upper()}:")
        print(f"  ファイル数: {summary['total_files']}")
        print(f"  平均NaN率: {summary['average_nan_rate'] * 100:.2f}%")
        print(
            f"  高NaN率ファイル: {summary['files_with_high_nan']} ({summary['high_nan_percentage']:.1f}%)"
        )
        print(f"  平均サイズ: {summary['average_file_size_mb']:.3f}MB")
        print(f"  欠損カラム: {summary['total_missing_columns']}")
        print(
            f"  日付問題ファイル: {summary['files_with_date_issues']} ({summary['date_issues_percentage']:.1f}%)"
        )

    # 推奨事項
    if report["recommendations"]:
        print("\n=== 推奨事項 ===")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"{i}. {rec}")

    print("\n結果出力:")
    print(f"  詳細CSV: {csv_path}")
    print(f"  レポートJSON: {json_path}")


if __name__ == "__main__":
    main()
