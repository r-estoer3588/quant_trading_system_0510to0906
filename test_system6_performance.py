#!/usr/bin/env python3
"""
System6パフォーマンステスト用スクリプト
最適化前後の処理時間を測定する
"""
import time
import sys

sys.path.append(".")

from strategies.system6_strategy import System6Strategy
from common.cache_manager import load_base_cache


def test_system6_performance():
    print("System6パフォーマンステスト開始...")

    # 小さなサンプルセット（10銘柄）で測定
    print("ベースキャッシュからサンプルデータ読み込み中...")
    sample_symbols = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
        "NVDA",
        "META",
        "NFLX",
        "INTC",
        "AMD",
    ]

    raw_data = {}
    for symbol in sample_symbols:
        try:
            df = load_base_cache(symbol, prefer_precomputed_indicators=True)
            if df is not None and not df.empty:
                raw_data[symbol] = df
        except Exception as e:
            print(f"⚠️  {symbol} の読み込みに失敗: {e}")

    print(f"📊 テスト対象: {len(raw_data)}銘柄")
    if not raw_data:
        print("❌ データが読み込めませんでした")
        return

    # System6戦略を初期化
    strategy = System6Strategy()

    # テスト1: reuse_indicators=True（最適化版）
    print("\n--- テスト1: 最適化版（reuse_indicators=True）---")
    start_time = time.time()

    try:
        prepared_data = strategy.prepare_data(
            raw_data,
            reuse_indicators=True,
            use_process_pool=False,  # シングルスレッドでテスト
        )

        candidates = strategy.generate_candidates(prepared_data, top_n=5)

        end_time = time.time()
        optimized_time = end_time - start_time

        print("✅ 最適化版実行完了!")
        print(f"⏱️  実行時間: {optimized_time:.2f}秒")
        print(f"📊 処理銘柄数: {len(prepared_data)}")
        print(f"🎯 候補数: {len(candidates) if candidates else 0}")

        if candidates and len(candidates) > 0:
            print("Top 3 candidates:")
            for i, item in enumerate(candidates[:3]):
                if isinstance(item, tuple) and len(item) >= 2:
                    symbol, _info = item
                    print(f"  {i+1}. {symbol}")
                else:
                    print(f"  {i+1}. {item}")

    except Exception as e:
        end_time = time.time()
        optimized_time = end_time - start_time
        print(f"❌ 最適化版でエラー発生 (経過時間: {optimized_time:.2f}秒)")
        print(f"エラー: {e}")
        return

    # テスト2: reuse_indicators=False（従来版）
    print("\n--- テスト2: 従来版（reuse_indicators=False）---")
    start_time = time.time()

    try:
        prepared_data_legacy = strategy.prepare_data(
            raw_data,
            reuse_indicators=False,
            use_process_pool=False,  # シングルスレッドでテスト
        )

        end_time = time.time()
        legacy_time = end_time - start_time

        print("✅ 従来版実行完了!")
        print(f"⏱️  実行時間: {legacy_time:.2f}秒")
        print(f"📊 処理銘柄数: {len(prepared_data_legacy)}")

    except Exception as e:
        end_time = time.time()
        legacy_time = end_time - start_time
        print(f"❌ 従来版でエラー発生 (経過時間: {legacy_time:.2f}秒)")
        print(f"エラー: {e}")
        legacy_time = None

    # 結果比較
    print("\n--- パフォーマンス比較 ---")
    print(f"最適化版: {optimized_time:.2f}秒")
    if legacy_time is not None:
        print(f"従来版:   {legacy_time:.2f}秒")
        if legacy_time > 0:
            speedup = legacy_time / optimized_time
            print(f"🚀 最適化効果: {speedup:.1f}x高速化")
            print(f"💾 時間短縮: {legacy_time - optimized_time:.2f}秒")


if __name__ == "__main__":
    test_system6_performance()
