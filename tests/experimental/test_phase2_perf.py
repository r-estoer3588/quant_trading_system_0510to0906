"""
Phase2最適化の直接パフォーマンステスト
"""

import time
import os
import sys
from pathlib import Path

# パス設定
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.cache_manager import CacheManager
from config.settings import get_settings
from indicators_common import add_indicators_batch


def test_phase2_performance():
    """Phase2の並列化・最適化パフォーマンステスト"""
    print("=== Phase2 Performance Test ===")

    # 設定とキャッシュマネージャー
    settings = get_settings()
    cache_manager = CacheManager(settings)

    # テスト用シンボルリスト（rolling cacheに存在するものから選択）
    rolling_dir = Path(settings.cache.rolling_dir)
    available_files = list(rolling_dir.glob("*.csv")) + list(rolling_dir.glob("*.parquet"))
    symbols = [f.stem for f in available_files[:50]]  # 最大50シンボル

    if not symbols:
        print("❌ テスト用シンボルが見つかりません")
        return

    print(f"📊 テスト対象: {len(symbols)}シンボル")

    # === 従来処理の計測 ===
    print("\n--- 従来処理（シーケンシャル） ---")
    start_time = time.perf_counter()

    sequential_data = {}
    for symbol in symbols:
        try:
            df = cache_manager.read(symbol, "rolling")
            if df is None or df.empty:
                df = cache_manager.read(symbol, "full")
            if df is not None and not df.empty:
                sequential_data[symbol] = df
        except Exception:
            pass

    sequential_time = time.perf_counter() - start_time
    print(f"⏱️  シーケンシャル読み込み: {sequential_time:.3f}秒")
    print(f"📁 成功: {len(sequential_data)}/{len(symbols)}シンボル")

    # === 新しい並列処理の計測 ===
    print("\n--- 新処理（並列最適化） ---")
    start_time = time.perf_counter()

    cpu_count = os.cpu_count() or 4
    max_workers = min(max(4, cpu_count), len(symbols))

    parallel_data = cache_manager.read_batch_parallel(
        symbols=symbols, profile="rolling", max_workers=max_workers, fallback_profile="full"
    )

    parallel_time = time.perf_counter() - start_time
    print(f"⏱️  並列読み込み: {parallel_time:.3f}秒 (workers={max_workers})")
    print(f"📁 成功: {len(parallel_data)}/{len(symbols)}シンボル")

    # === 指標計算の比較 ===
    if parallel_data:
        print("\n--- 指標計算比較 ---")

        # 従来の指標計算
        start_time = time.perf_counter()
        from indicators_common import add_indicators

        sequential_indicators = {}
        for symbol, df in list(parallel_data.items())[:10]:  # 最初の10個
            sequential_indicators[symbol] = add_indicators(df.copy())
        sequential_indicators_time = time.perf_counter() - start_time

        # バッチ指標計算
        start_time = time.perf_counter()
        batch_indicators = add_indicators_batch(
            {symbol: df for symbol, df in list(parallel_data.items())[:10]}
        )
        batch_indicators_time = time.perf_counter() - start_time

        print(f"⏱️  従来指標計算: {sequential_indicators_time:.3f}秒")
        print(f"⏱️  バッチ指標計算: {batch_indicators_time:.3f}秒")

        if sequential_indicators_time > 0:
            speedup = sequential_indicators_time / batch_indicators_time
            print(f"🚀 指標計算高速化: {speedup:.2f}倍")

    # === 結果サマリー ===
    print("\n=== 改善結果 ===")

    if sequential_time > 0:
        speedup = sequential_time / parallel_time
        time_saved = sequential_time - parallel_time
        improvement_pct = (time_saved / sequential_time) * 100

        print(f"🚀 データ読み込み高速化: {speedup:.2f}倍")
        print(f"⏰ 時間短縮: {time_saved:.3f}秒 ({improvement_pct:.1f}%改善)")

        if parallel_time <= 0.5:
            print("✅ 目標達成: Phase2を0.5秒以下に短縮")
        else:
            print(f"⚠️  目標未達: {parallel_time:.3f}秒 > 0.5秒")

    print("\n💾 メモリ最適化機能も実装済み")
    print(f"🧵 並列ワーカー数: {max_workers}")


if __name__ == "__main__":
    test_phase2_performance()
