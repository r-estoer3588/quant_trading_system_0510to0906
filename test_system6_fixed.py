#!/usr/bin/env python3
"""
System6固定版の性能テスト
既存インジケーター活用による真の高速化を検証
"""
import time
import sys

sys.path.append(".")

from strategies.system6_strategy import System6Strategy
from common.cache_manager import load_base_cache


def test_system6_fixed():
    """System6固定版のテスト"""
    print("🔧 System6固定版テスト開始")
    print("=" * 60)

    # テスト用データ読み込み（100銘柄）
    print("📦 テスト用データ読み込み中（100銘柄）...")

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
        "XOM",
        "CVX",
        "WFC",
        "BAC",
        "JPM",
        "GS",
        "MS",
        "IBM",
        "INTC",
        "AMD",
        "QCOM",
        "AMAT",
        "TXN",
        "MU",
        "AVGO",
        "COST",
        "SBUX",
        "NKE",
        "LULU",
        "ZM",
        "UBER",
        "LYFT",
        "ABNB",
        "SQ",
        "SHOP",
        "ROKU",
        "PINS",
        "SNAP",
        "TWTR",
        "SPOT",
        "DDOG",
        "SNOW",
        "ZS",
        "OKTA",
        "CRWD",
        "NET",
        "PLTR",
        "RBLX",
        "COIN",
        "HOOD",
        "SOFI",
        "UPST",
        "AFRM",
        "LC",
        "OPEN",
        "Z",
        "REYN",
        "CLOV",
        "WISH",
        "SPCE",
        "NKLA",
        "RIDE",
        "FSR",
        "LCID",
        "RIVN",
        "F",
        "GM",
        "TM",
        "HMC",
        "RACE",
        "TSRA",
        "ON",
        "MRVL",
        "LRCX",
        "KLAC",
        "ASML",
        "TSM",
        "UMC",
        "ASX",
        "GOLD",
        "NEM",
    ]

    # データ読み込み
    raw_dict = {}
    for symbol in test_symbols:
        try:
            df = load_base_cache(symbol)
            if df is not None and not df.empty:
                raw_dict[symbol] = df
        except Exception:
            continue
    valid_symbols = [s for s, df in raw_dict.items() if df is not None and not df.empty]
    print(f"✅ 実際に読み込めた銘柄数: {len(valid_symbols)}")

    if len(valid_symbols) < 10:
        print("❌ テスト用データが不足しています")
        return

    strategy = System6Strategy()

    # テスト1: 固定版（既存インジケーター活用）
    print("\n" + "=" * 60)
    print("🔧 固定版テスト（fixed_mode=True）")
    print("=" * 60)

    fixed_start = time.time()
    try:
        fixed_prepared = strategy.prepare_data(
            raw_dict,
            fixed_mode=True,
            ultra_mode=False,
            enable_optimization=False,
        )
        fixed_data_time = time.time() - fixed_start

        fixed_candidates_start = time.time()
        fixed_candidates_result = strategy.generate_candidates(
            fixed_prepared,
            fixed_mode=True,
            ultra_mode=False,
        )
        fixed_candidates, _ = fixed_candidates_result
        fixed_candidates_time = time.time() - fixed_candidates_start
        fixed_total_time = time.time() - fixed_start

        # 候補数を計算
        if isinstance(fixed_candidates, dict):
            fixed_candidate_count = sum(len(candidates) for candidates in fixed_candidates.values())
        else:
            fixed_candidate_count = 0

        print("✅ 固定版完了!")
        print(f"⏱️  データ準備時間: {fixed_data_time:.1f}秒 ({fixed_data_time/60:.1f}分)")
        print(f"📊 処理銘柄数: {len(fixed_prepared)}")
        print(f"🎯 候補生成時間: {fixed_candidates_time:.1f}秒")
        print(f"📈 候補数: {fixed_candidate_count}")
        print(f"🏁 総処理時間: {fixed_total_time:.1f}秒 ({fixed_total_time/60:.1f}分)")

    except Exception as e:
        print(f"❌ 固定版テストエラー: {e}")
        import traceback

        traceback.print_exc()
        return

    # テスト2: 従来版（再計算あり）
    print("\n" + "=" * 60)
    print("🐌 従来版テスト（fixed_mode=False）")
    print("=" * 60)

    original_start = time.time()
    try:
        original_prepared = strategy.prepare_data(
            raw_dict,
            fixed_mode=False,
            ultra_mode=False,
            enable_optimization=False,
        )
        original_data_time = time.time() - original_start

        original_candidates_start = time.time()
        original_candidates_result = strategy.generate_candidates(
            original_prepared,
            fixed_mode=False,
            ultra_mode=False,
        )
        original_candidates, _ = original_candidates_result
        original_candidates_time = time.time() - original_candidates_start
        original_total_time = time.time() - original_start

        # 候補数を計算
        if isinstance(original_candidates, dict):
            original_candidate_count = sum(
                len(candidates) for candidates in original_candidates.values()
            )
        else:
            original_candidate_count = 0

        print("✅ 従来版完了!")
        print(f"⏱️  データ準備時間: {original_data_time:.1f}秒 ({original_data_time/60:.1f}分)")
        print(f"📊 処理銘柄数: {len(original_prepared)}")
        print(f"🎯 候補生成時間: {original_candidates_time:.1f}秒")
        print(f"📈 候補数: {original_candidate_count}")
        print(f"🏁 総処理時間: {original_total_time:.1f}秒 ({original_total_time/60:.1f}分)")

    except Exception as e:
        print(f"❌ 従来版テストエラー: {e}")
        import traceback

        traceback.print_exc()
        return

    # 結果比較
    print("\n" + "=" * 60)
    print("📊 パフォーマンス比較結果")
    print("=" * 60)

    print(f"🔧 固定版: {fixed_total_time:.1f}秒 ({fixed_total_time/60:.1f}分)")
    print(f"🐌 従来版: {original_total_time:.1f}秒 ({original_total_time/60:.1f}分)")

    if original_total_time > 0:
        speedup = original_total_time / fixed_total_time
        time_saved = original_total_time - fixed_total_time
        reduction_pct = (time_saved / original_total_time) * 100

        print("🚀 固定版効果:")
        print(f"   • 高速化倍率: {speedup:.1f}x")
        print(f"   • 時間短縮: {time_saved:.1f}秒 ({time_saved/60:.1f}分)")
        print(f"   • 短縮率: {reduction_pct:.1f}%")

        # 全銘柄での推定時間
        if len(valid_symbols) > 0:
            projection_ratio = 2351 / len(valid_symbols)  # 全銘柄への拡大係数
            estimated_time_minutes = (fixed_total_time * projection_ratio) / 60
            print(f"\n🎯 全銘柄(2351)での推定時間: {estimated_time_minutes:.1f}分")
            if estimated_time_minutes <= 30:
                print("✅ 30分目標達成見込み！")
            else:
                print(f"⚠️  30分目標まであと {estimated_time_minutes - 30:.1f}分短縮が必要")

    print("\n🎉 テスト完了！")


if __name__ == "__main__":
    test_system6_fixed()
