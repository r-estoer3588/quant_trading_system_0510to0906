#!/usr/bin/env python3
"""
拡張キャッシュヘルスチェックテスト

使用方法:
  python scripts/test_extended_cache_health.py --sample 5 --profiles base rolling
"""

import logging
from pathlib import Path
import sys

# プロジェクトルートを認識
sys.path.append(str(Path(__file__).resolve().parent.parent))

from common.extended_cache_health_checker import ExtendedCacheHealthChecker


def test_extended_health_check():
    """拡張ヘルスチェックテスト実行"""

    # ロギング設定
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    print("=== 拡張キャッシュヘルスチェックテスト ===")

    # チェッカー初期化（サンプリング適用）
    checker = ExtendedCacheHealthChecker(
        nan_threshold=0.5, sample_size=5, max_workers=2  # 5ファイルサンプリング
    )

    # 対象プロファイル
    profiles = ["base", "rolling"]

    print(f"対象プロファイル: {profiles}")
    print(f"サンプリングサイズ: {checker.sample_size}")
    print(f"NaN閾値: {checker.nan_threshold}")

    # 分析実行
    all_results = {}
    for profile in profiles:
        print(f"\n[{profile.upper()}] 分析開始...")
        try:
            results = checker.analyze_profile(profile)
            all_results[profile] = results
            print(f"[{profile.upper()}] {len(results)}ファイル分析完了")
        except Exception as e:
            logger.error(f"[{profile.upper()}] 分析失敗: {e}")
            all_results[profile] = []

    if not any(all_results.values()):
        print("⚠️ 分析結果が空です - data_cacheディレクトリを確認してください")
        return

    # レポート生成
    print("\nレポート生成中...")
    try:
        report = checker.generate_comprehensive_report(all_results)
        csv_path, json_path = checker.export_results(all_results, report)

        print("\n=== 分析結果サマリー ===")
        metadata = report["analysis_metadata"]
        print(f"総分析ファイル: {metadata['total_files_analyzed']}")
        print(f"サンプリング: {'有効' if metadata['sampling_enabled'] else '無効'}")

        # プロファイル別結果
        for profile, summary in report["profile_summaries"].items():
            if summary.get("status") == "no_data":
                print(f"\n{profile.upper()}: データなし")
                continue

            print(f"\n{profile.upper()}:")
            print(f"  ファイル数: {summary['total_files']}")
            print(f"  平均NaN率: {summary['average_nan_rate']*100:.2f}%")
            print(
                f"  高NaN率ファイル: {summary['files_with_high_nan']} ({summary['high_nan_percentage']:.1f}%)"
            )
            print(f"  平均ファイルサイズ: {summary['average_file_size_mb']:.3f}MB")
            print(f"  欠損カラム: {summary['total_missing_columns']}")
            print(
                f"  日付問題ファイル: {summary['files_with_date_issues']} ({summary['date_issues_percentage']:.1f}%)"
            )

        # 推奨事項
        recommendations = report.get("recommendations", [])
        if recommendations:
            print("\n=== 推奨事項 ===")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        else:
            print("\n✅ 特に問題は検出されませんでした")

        # 全体統計
        overall = report.get("overall_statistics", {})
        if overall:
            print("\n=== 全体統計 ===")
            print(f"総銘柄数: {overall.get('total_symbols_analyzed', 'N/A')}")
            print(f"平均行数/ファイル: {overall.get('average_rows_per_file', 'N/A')}")
            print(f"平均カラム数/ファイル: {overall.get('average_columns_per_file', 'N/A')}")
            print(f"全体健全性率: {overall.get('overall_health_rate', 'N/A')}%")

        print("\n=== 出力ファイル ===")
        print(f"詳細CSV: {csv_path}")
        print(f"レポートJSON: {json_path}")

        # 結果詳細の一部表示
        print("\n=== サンプル詳細（最初の2件）===")
        sample_count = 0
        for profile, metrics_list in all_results.items():
            if sample_count >= 2:
                break
            for metrics in metrics_list[:1]:  # 各プロファイルから1件
                if sample_count >= 2:
                    break
                print(f"\n銘柄: {metrics.symbol} ({profile})")
                print(f"  行数: {metrics.total_rows}, カラム数: {metrics.total_columns}")
                print(f"  NaN率: {metrics.nan_rate_overall*100:.2f}%")
                print(f"  高NaNカラム: {len(metrics.columns_with_high_nan)}")
                print(f"  欠損カラム: {len(metrics.missing_columns)}")
                print(
                    f"  時系列問題: 日付欠損={metrics.date_gaps}, 重複={metrics.duplicate_dates}, 順序OK={metrics.chronological_order}"
                )
                print(f"  価格異常: {sum(metrics.price_anomalies.values())}")
                print(
                    f"  既存ヘルスチェック: {metrics.basic_health_results.get('overall_health', 'N/A')}"
                )
                sample_count += 1

        print("\n✅ 拡張キャッシュヘルスチェックテスト完了")

    except Exception as e:
        logger.error(f"レポート生成失敗: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extended Cache Health Test")
    parser.add_argument("--sample", type=int, default=5, help="サンプリングサイズ（デフォルト: 5）")
    parser.add_argument(
        "--profiles",
        nargs="*",
        choices=["base", "rolling", "full_backup"],
        default=["base", "rolling"],
        help="テスト対象プロファイル",
    )

    args = parser.parse_args()

    # テスト実行
    try:
        # グローバル設定を更新
        from common.extended_cache_health_checker import ExtendedCacheHealthChecker

        checker = ExtendedCacheHealthChecker(sample_size=args.sample)

        print(f"拡張ヘルスチェック開始 - サンプル: {args.sample}, プロファイル: {args.profiles}")

        all_results = {}
        for profile in args.profiles:
            all_results[profile] = checker.analyze_profile(profile)

        report = checker.generate_comprehensive_report(all_results)
        csv_path, json_path = checker.export_results(all_results, report)

        print(f"\n結果: 詳細CSV={csv_path}, JSON={json_path}")

    except KeyboardInterrupt:
        print("\n⚠️ ユーザー中断")
    except Exception as e:
        print(f"❌ エラー: {e}")
        sys.exit(1)
