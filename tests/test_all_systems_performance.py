#!/usr/bin/env python3
"""
全システム（System1-7）パフォーマンステスト用スクリプト
1000銘柄限定で各システムの処理時間を測定・比較する
"""
import random
import sys
import time
from typing import Any

sys.path.append(".")

from common.cache_manager import load_base_cache
from common.testing import set_test_determinism
from strategies.system1_strategy import System1Strategy
from strategies.system2_strategy import System2Strategy
from strategies.system3_strategy import System3Strategy
from strategies.system4_strategy import System4Strategy
from strategies.system5_strategy import System5Strategy
from strategies.system6_strategy import System6Strategy
from strategies.system7_strategy import System7Strategy


def load_sample_data(num_symbols: int = 1000) -> dict[str, Any]:
    """1000銘柄のサンプルデータを読み込み"""
    print(f"📦 {num_symbols}銘柄のサンプルデータ読み込み中...")

    # よく使われる銘柄リストを作成（SPYは必須）
    common_symbols = [
        "SPY",
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
        "CRM",
        "ORCL",
        "ADBE",
        "PYPL",
        "CSCO",
        "PEP",
        "KO",
        "DIS",
        "WMT",
        "BA",
        "JNJ",
        "PG",
        "V",
        "MA",
        "UNH",
        "HD",
        "MCD",
        "VZ",
        "T",
        "JPM",
        "BAC",
        "WFC",
        "C",
        "GS",
        "MS",
        "AXP",
        "COF",
        "SCHW",
        "BLK",
        "XOM",
        "CVX",
        "COP",
        "SLB",
        "EOG",
        "PSX",
        "VLO",
        "MPC",
        "KMI",
        "OKE",
    ]

    # 追加のランダム銘柄を生成（実際にはキャッシュから読み込み可能なものを使用）
    additional_symbols = []
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # 2-4文字のランダムシンボルを生成
    for _ in range(num_symbols - len(common_symbols)):
        length = random.choice([2, 3, 4])
        symbol = "".join(random.choices(alphabet, k=length))
        additional_symbols.append(symbol)

    all_symbols = common_symbols + additional_symbols[: num_symbols - len(common_symbols)]

    raw_data = {}
    loaded_count = 0

    for symbol in all_symbols:
        try:
            df = load_base_cache(symbol, prefer_precomputed_indicators=True)
            if df is not None and not df.empty and len(df) > 100:  # 十分なデータがある場合のみ
                raw_data[symbol] = df
                loaded_count += 1
                if loaded_count >= num_symbols:
                    break
        except Exception:
            continue  # エラーは無視して次へ

    print(f"✅ 実際に読み込めた銘柄数: {len(raw_data)}")
    return raw_data


def test_system_performance(
    system_class, system_name: str, raw_data: dict[str, Any]
) -> dict[str, float]:
    """個別システムの性能テスト"""
    print(f"\n{'='*50}")
    print(f"🔍 {system_name} 性能テスト開始")
    print(f"{'='*50}")

    strategy = system_class()
    results = {}

    # テスト1: シングルスレッド + インジケーター再利用ON
    print(f"\n--- {system_name}: 最適化版（single thread + reuse indicators）---")
    start_time = time.time()

    try:
        prepared_data = strategy.prepare_data(
            raw_data,
            reuse_indicators=True,
            use_process_pool=False,  # シングルスレッド
        )

        candidates = strategy.generate_candidates(prepared_data, top_n=10)

        end_time = time.time()
        optimized_time = end_time - start_time
        results["optimized_single"] = optimized_time

        print("✅ 最適化版（シングル）実行完了!")
        print(f"⏱️  実行時間: {optimized_time:.2f}秒")
        print(f"📊 処理銘柄数: {len(prepared_data)}")
        print(f"🎯 候補数: {len(candidates) if candidates else 0}")

    except Exception as e:
        end_time = time.time()
        optimized_time = end_time - start_time
        results["optimized_single"] = optimized_time
        print(f"❌ 最適化版（シングル）でエラー発生 (経過時間: {optimized_time:.2f}秒)")
        print(f"エラー: {e}")

    # テスト2: マルチプロセス + インジケーター再利用ON
    print(f"\n--- {system_name}: 並列処理版（multi process + reuse indicators）---")
    start_time = time.time()

    try:
        prepared_data_parallel = strategy.prepare_data(
            raw_data,
            reuse_indicators=True,
            use_process_pool=True,  # 並列処理
        )

        candidates_parallel = strategy.generate_candidates(prepared_data_parallel, top_n=10)

        end_time = time.time()
        parallel_time = end_time - start_time
        results["optimized_parallel"] = parallel_time

        print("✅ 並列処理版実行完了!")
        print(f"⏱️  実行時間: {parallel_time:.2f}秒")
        print(f"📊 処理銘柄数: {len(prepared_data_parallel)}")
        print(f"🎯 候補数: {len(candidates_parallel) if candidates_parallel else 0}")

    except Exception as e:
        end_time = time.time()
        parallel_time = end_time - start_time
        results["optimized_parallel"] = parallel_time
        print(f"❌ 並列処理版でエラー発生 (経過時間: {parallel_time:.2f}秒)")
        print(f"エラー: {e}")

    # パフォーマンス比較
    print(f"\n--- {system_name} パフォーマンス結果 ---")
    if "optimized_single" in results:
        print(f"シングルスレッド: {results['optimized_single']:.2f}秒")
    if "optimized_parallel" in results:
        print(f"並列処理:       {results['optimized_parallel']:.2f}秒")

        if (
            "optimized_single" in results
            and results["optimized_single"] > 0
            and results["optimized_parallel"] > 0
        ):
            speedup = results["optimized_single"] / results["optimized_parallel"]
            print(f"🚀 並列処理効果: {speedup:.1f}x高速化")
            print(
                f"💾 時間短縮: {results['optimized_single'] - results['optimized_parallel']:.2f}秒"
            )

    return results


def main():
    """メイン処理"""
    set_test_determinism()
    print("🚀 全システム性能テスト開始")
    print("=" * 60)

    # サンプルデータ読み込み
    raw_data = load_sample_data(num_symbols=1000)

    if not raw_data:
        print("❌ データが読み込めませんでした")
        return

    # 全システムのテストを実行
    systems = [
        (System1Strategy, "System1"),
        (System2Strategy, "System2"),
        (System3Strategy, "System3"),
        (System4Strategy, "System4"),
        (System5Strategy, "System5"),
        (System6Strategy, "System6"),
        (System7Strategy, "System7"),
    ]

    all_results = {}

    for system_class, system_name in systems:
        try:
            results = test_system_performance(system_class, system_name, raw_data)
            all_results[system_name] = results
        except Exception as e:
            print(f"❌ {system_name}のテストでエラー発生: {e}")
            all_results[system_name] = {"error": str(e)}

    # 全体サマリー
    print("\n" + "=" * 80)
    print("📊 全システム性能サマリー")
    print("=" * 80)

    print(f"{'System':<10} {'Single(秒)':<12} {'Parallel(秒)':<13} {'Speedup':<8} {'Status'}")
    print("-" * 60)

    for system_name in [
        "System1",
        "System2",
        "System3",
        "System4",
        "System5",
        "System6",
        "System7",
    ]:
        if system_name in all_results:
            results = all_results[system_name]

            if "error" in results:
                print(f"{system_name:<10} {'ERROR':<12} {'ERROR':<13} {'N/A':<8} ❌")
                continue

            single = results.get("optimized_single", 0)
            parallel = results.get("optimized_parallel", 0)

            if single > 0 and parallel > 0:
                speedup = single / parallel
                status = "✅"
            else:
                speedup = 0
                status = "⚠️"

            single_str = f"{single:.1f}" if single > 0 else "N/A"
            parallel_str = f"{parallel:.1f}" if parallel > 0 else "N/A"
            speedup_str = f"{speedup:.1f}x" if speedup > 0 else "N/A"

            print(
                f"{system_name:<10} {single_str:<12} {parallel_str:<13} {speedup_str:<8} {status}"
            )

    print("\n🎯 テスト完了！")


if __name__ == "__main__":
    main()
