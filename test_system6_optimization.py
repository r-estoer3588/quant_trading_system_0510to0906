#!/usr/bin/env python3
"""
System6最適化版の性能テスト
30分達成を目指した最適化の効果を測定
"""
import time
import sys

sys.path.append(".")

from strategies.system6_strategy import System6Strategy
from common.cache_manager import load_base_cache


def test_system6_optimization():
    """System6最適化版のテスト"""
    print("🚀 System6最適化版テスト開始")
    print("=" * 60)

    # テスト用データ読み込み（500銘柄）
    print("📦 テスト用データ読み込み中（500銘柄）...")

    test_symbols = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
        "NVDA",
        "META",
        "NFLX",
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
        "AA",
        "AAL",
        "AAOI",
    ]

    # より多くの銘柄でテスト
    additional_symbols = []
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i in range(450):  # 500銘柄まで拡張
        if i < len(alphabet):
            additional_symbols.append(alphabet[i])
        elif i < len(alphabet) * 2:
            additional_symbols.append(alphabet[i - len(alphabet)] + "A")
        else:
            additional_symbols.append(f"TEST{i}")

    all_test_symbols = test_symbols + additional_symbols[:450]

    raw_data = {}
    loaded_count = 0

    for symbol in all_test_symbols:
        try:
            df = load_base_cache(symbol, prefer_precomputed_indicators=True)
            if df is not None and not df.empty and len(df) > 100:
                raw_data[symbol] = df
                loaded_count += 1
                if loaded_count >= 500:  # 500銘柄に制限
                    break
        except Exception:
            continue

    print(f"✅ 実際に読み込めた銘柄数: {len(raw_data)}")

    if len(raw_data) < 50:
        print("❌ テスト用データが不足しています")
        return

    strategy = System6Strategy()

    # テスト1: 最適化版（enable_optimization=True）
    print("\n" + "=" * 60)
    print("🔥 最適化版テスト（enable_optimization=True）")
    print("=" * 60)

    start_time = time.time()

    try:
        prepared_data_opt = strategy.prepare_data(
            raw_data,
            reuse_indicators=True,
            use_process_pool=False,
            enable_optimization=True,  # 最適化有効
        )

        opt_time = time.time() - start_time

        print("\n✅ 最適化版完了!")
        print(f"⏱️  処理時間: {opt_time:.1f}秒 ({opt_time/60:.1f}分)")
        print(f"📊 処理銘柄数: {len(prepared_data_opt)}")

        # 候補生成もテスト
        candidates_start = time.time()
        candidates_opt = strategy.generate_candidates(prepared_data_opt, top_n=10)
        candidates_time = time.time() - candidates_start

        total_opt_time = opt_time + candidates_time

        print(f"🎯 候補生成時間: {candidates_time:.1f}秒")
        print(f"📈 候補数: {len(candidates_opt)}")
        print(f"🏁 総処理時間: {total_opt_time:.1f}秒 ({total_opt_time/60:.1f}分)")

    except Exception as e:
        opt_time = time.time() - start_time
        print(f"❌ 最適化版でエラー発生 (経過時間: {opt_time:.1f}秒)")
        print(f"エラー: {e}")
        return

    # テスト2: 従来版（enable_optimization=False）
    print("\n" + "=" * 60)
    print("🐌 従来版テスト（enable_optimization=False）")
    print("=" * 60)

    start_time = time.time()

    try:
        prepared_data_legacy = strategy.prepare_data(
            raw_data,
            reuse_indicators=True,
            use_process_pool=False,
            enable_optimization=False,  # 最適化無効
        )

        legacy_time = time.time() - start_time

        print("\n✅ 従来版完了!")
        print(f"⏱️  処理時間: {legacy_time:.1f}秒 ({legacy_time/60:.1f}分)")
        print(f"📊 処理銘柄数: {len(prepared_data_legacy)}")

        # 候補生成もテスト
        candidates_start = time.time()
        candidates_legacy = strategy.generate_candidates(prepared_data_legacy, top_n=10)
        candidates_legacy_time = time.time() - candidates_start

        total_legacy_time = legacy_time + candidates_legacy_time

        print(f"🎯 候補生成時間: {candidates_legacy_time:.1f}秒")
        print(f"📈 候補数: {len(candidates_legacy)}")
        print(f"🏁 総処理時間: {total_legacy_time:.1f}秒 ({total_legacy_time/60:.1f}分)")

    except Exception as e:
        legacy_time = time.time() - start_time
        print(f"❌ 従来版でエラー発生 (経過時間: {legacy_time:.1f}秒)")
        print(f"エラー: {e}")
        total_legacy_time = None

    # 結果比較
    print("\n" + "=" * 80)
    print("📊 パフォーマンス比較結果")
    print("=" * 80)

    print(f"🔥 最適化版: {total_opt_time:.1f}秒 ({total_opt_time/60:.1f}分)")

    if total_legacy_time is not None:
        print(f"🐌 従来版:   {total_legacy_time:.1f}秒 ({total_legacy_time/60:.1f}分)")

        if total_legacy_time > 0:
            speedup = total_legacy_time / total_opt_time
            time_saved = total_legacy_time - total_opt_time

            print("\n🚀 最適化効果:")
            print(f"   • 高速化倍率: {speedup:.1f}x")
            print(f"   • 時間短縮: {time_saved:.1f}秒 ({time_saved/60:.1f}分)")
            print(f"   • 短縮率: {(time_saved/total_legacy_time)*100:.1f}%")

            # 30分達成予測
            estimated_full_time = total_opt_time * (2351 / len(raw_data))  # 全銘柄での予測時間
            print(f"\n🎯 全銘柄(2351)での推定時間: {estimated_full_time/60:.1f}分")

            if estimated_full_time <= 30 * 60:  # 30分以内
                print("✅ 30分目標達成見込み！")
            else:
                needed_improvement = estimated_full_time / (30 * 60)
                print(f"⚠️ 30分達成にはさらに{needed_improvement:.1f}倍の高速化が必要")
        else:
            print("⚠️ 従来版の時間が0のため比較不可")
    else:
        print("⚠️ 従来版が失敗したため比較不可")

    print("\n🎉 テスト完了！")


if __name__ == "__main__":
    test_system6_optimization()
