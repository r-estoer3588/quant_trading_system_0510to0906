#!/usr/bin/env python3
"""指標スキップメトリクスのテスト・デモスクリプト

Usage:
    python scripts/test_indicator_metrics.py [--symbols AAPL,MSFT] [--samples 10]
"""

import argparse
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from common.cache_manager import CacheManager
from common.indicator_metrics import (
    create_instrumented_add_indicators,
    get_metrics_collector,
)
from config.settings import get_settings


def test_indicator_metrics(symbols: list[str], samples: int = 10):
    """指標メトリクス機能のテスト"""
    print("=== 指標スキップメトリクス テスト ===")

    # 設定とキャッシュマネージャー
    settings = get_settings(create_dirs=True)
    cache_mgr = CacheManager(settings)

    # メトリクス機能付きadd_indicators作成
    instrumented_add_indicators = create_instrumented_add_indicators()
    collector = get_metrics_collector()

    print(f"テスト対象: {len(symbols)}銘柄")
    print(f"サンプル数: {samples}件")

    # テスト実行
    processed_count = 0
    for symbol in symbols[:samples]:
        try:
            print(f"\n処理中: {symbol}")

            # キャッシュからデータ読み込み（rollingプロファイル使用）
            df = cache_mgr.read(symbol, "rolling")
            if df is None or df.empty:
                print(f"  ⚠️ データなし: {symbol}")
                continue

            print(f"  📊 データ行数: {len(df)}, 初期列数: {len(df.columns)}")

            # 指標計算（メトリクス収集付き）
            result = instrumented_add_indicators(df, symbol=symbol)

            if result is not None:
                print(f"  ✅ 完了: 最終列数 {len(result.columns)}")
                processed_count += 1
            else:
                print(f"  ❌ 失敗: {symbol}")

        except Exception as e:
            print(f"  💥 エラー: {symbol} - {e}")

    print(f"\n=== 処理完了: {processed_count}/{len(symbols[:samples])}銘柄 ===")

    # サマリー統計表示
    summary = collector.get_summary_stats()
    if summary:
        print("\n📈 サマリー統計:")
        print(f"  処理銘柄数: {summary['total_symbols']}")
        print(f"  平均スキップ率: {summary['avg_skip_rate']:.1f}%")
        print(f"  平均新規計算率: {summary['avg_compute_rate']:.1f}%")
        print(f"  平均成功率: {summary['avg_success_rate']:.1f}%")
        print(f"  平均計算時間: {summary['avg_computation_time']:.3f}秒")
        print(f"  総計算時間: {summary['total_computation_time']:.1f}秒")
        print(f"  最大計算時間: {summary['max_computation_time']:.3f}秒")

    # メトリクス CSV 出力
    collector.export_metrics("test_run_metrics.csv")
    print(f"\n📁 メトリクス出力: {collector.output_dir / 'test_run_metrics.csv'}")

    return processed_count


def get_sample_symbols(cache_mgr: CacheManager, n: int = 20) -> list[str]:
    """サンプル銘柄を取得"""
    rolling_dir = Path(cache_mgr.settings.cache.rolling_dir)
    if not rolling_dir.exists():
        return []

    csv_files = list(rolling_dir.glob("*.csv"))
    symbols = [f.stem for f in csv_files[:n]]
    return symbols


def main():
    parser = argparse.ArgumentParser(description="指標スキップメトリクス テスト")
    parser.add_argument(
        "--symbols", type=str, help="テスト対象銘柄 (カンマ区切り、例: AAPL,MSFT,GOOGL)"
    )
    parser.add_argument(
        "--samples", type=int, default=10, help="処理サンプル数 (デフォルト: 10)"
    )
    parser.add_argument(
        "--auto", action="store_true", help="自動で利用可能銘柄からサンプル選択"
    )

    args = parser.parse_args()

    # 銘柄リスト決定
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    elif args.auto:
        settings = get_settings()
        cache_mgr = CacheManager(settings)
        symbols = get_sample_symbols(cache_mgr, n=args.samples * 2)
        if not symbols:
            print("❌ 利用可能な銘柄が見つかりません")
            return 1
        print(f"🔍 自動選択: {len(symbols)}銘柄から {args.samples}件をサンプル")
    else:
        # デフォルト銘柄リスト
        symbols = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "TSLA",
            "NVDA",
            "META",
            "AMZN",
            "NFLX",
            "SPY",
            "QQQ",
        ]
        print("📌 デフォルト銘柄リストを使用")

    try:
        processed = test_indicator_metrics(symbols, args.samples)
        print(f"\n🏁 テスト完了: {processed}銘柄処理")
        return 0 if processed > 0 else 1
    except KeyboardInterrupt:
        print("\n⏹️ ユーザーによる中断")
        return 1
    except Exception as e:
        print(f"\n💥 予期しないエラー: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
