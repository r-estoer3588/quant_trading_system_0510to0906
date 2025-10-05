"""
極限最適化: メモリマッピング + 超軽量データ構造
"""

from pathlib import Path
import sys
import time

# パス設定 - tests/experimental/ から2階層上のルートへ
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from config.settings import get_settings


def benchmark_io_methods():
    """各種I/O方法のベンチマーク"""
    print("=== I/O方法別ベンチマーク ===")

    settings = get_settings()
    rolling_dir = Path(settings.cache.rolling_dir)

    # テストファイルの選択
    test_files = list(rolling_dir.glob("*.csv"))[:10]  # 10ファイルで測定
    if not test_files:
        print("❌ テスト用ファイルが見つかりません")
        return

    print(f"📊 テスト対象: {len(test_files)}ファイル")

    methods = {}

    # === 方法1: 標準pandas読み込み ===
    start_time = time.perf_counter()
    for file_path in test_files:
        try:
            df = pd.read_csv(file_path)
        except Exception:
            pass
    methods["標準pandas"] = time.perf_counter() - start_time

    # === 方法2: dtype指定pandas ===
    start_time = time.perf_counter()
    dtype_map = {
        "Open": np.float32,
        "High": np.float32,
        "Low": np.float32,
        "Close": np.float32,
        "Volume": np.float32,
    }
    for file_path in test_files:
        try:
            df = pd.read_csv(file_path, dtype=dtype_map, low_memory=False)  # type: ignore
        except Exception:
            pass
    methods["dtype指定pandas"] = time.perf_counter() - start_time

    # === 方法3: 列選択pandas ===
    essential_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    start_time = time.perf_counter()
    for file_path in test_files:
        try:
            # まずヘッダーを読んで存在する列を確認
            sample = pd.read_csv(file_path, nrows=0)
            available_cols = [c for c in essential_cols if c in sample.columns]
            if available_cols:
                df = pd.read_csv(
                    file_path,
                    usecols=available_cols,
                    dtype=dtype_map,
                    low_memory=False,  # type: ignore
                )
        except Exception:
            pass
    methods["列選択pandas"] = time.perf_counter() - start_time

    # === 方法4: numpy直接読み込み ===
    start_time = time.perf_counter()
    for file_path in test_files:
        try:
            # numpyで直接数値データを読み込み（日付列は無視）
            # data = np.genfromtxt(  # Unused variable removed
            np.genfromtxt(
                file_path,
                delimiter=",",
                skip_header=1,
                usecols=(1, 2, 3, 4, 5),
                dtype=np.float32,
                invalid_raise=False,
            )
        except Exception:
            pass
    methods["numpy直接"] = time.perf_counter() - start_time

    # === 結果表示 ===
    print("\n--- I/O性能比較 ---")
    baseline_time = methods.get("標準pandas", 1.0)

    for method, elapsed in methods.items():
        speedup = baseline_time / elapsed if elapsed > 0 else float("inf")
        print(f"{method:15}: {elapsed:.3f}秒 (x{speedup:.2f})")

    # === 最速方法での50ファイル測定 ===
    print("\n--- 最速方法での50ファイル実測 ---")

    all_files = list(rolling_dir.glob("*.csv"))[:50]

    start_time = time.perf_counter()
    successful_reads = 0

    for file_path in all_files:
        try:
            # ヘッダーチェック + 最適読み込み
            with open(file_path) as f:
                first_line = f.readline().strip()

            # 基本的なOHLCVカラムがあるかチェック
            if any(col in first_line.lower() for col in ["open", "high", "low", "close"]):
                sample = pd.read_csv(file_path, nrows=0)
                available_cols = [c for c in essential_cols if c in sample.columns]

                if len(available_cols) >= 4:  # 最低4列（OHLC）
                    df = pd.read_csv(
                        file_path,
                        usecols=available_cols,
                        dtype={k: v for k, v in dtype_map.items() if k in available_cols},
                        low_memory=False,
                        engine="c",  # Cエンジン使用
                    )
                    if df is not None and len(df) > 0:
                        successful_reads += 1

        except Exception:
            continue

    fastest_time = time.perf_counter() - start_time

    print(f"⚡ 最適化読み込み: {fastest_time:.3f}秒")
    print(f"📁 成功読み込み: {successful_reads}/{len(all_files)}ファイル")

    if fastest_time <= 0.5:
        print("🎯 目標達成！0.5秒以下に成功")
    else:
        print(f"⚠️  目標未達: {fastest_time:.3f}秒 (目標0.5秒)")

        # 更なる最適化のアドバイス
        print("\n📋 追加最適化案:")
        print("  1. SSDストレージの使用")
        print("  2. Parquet/Featherフォーマット移行")
        print("  3. データのメモリキャッシュ化")
        print("  4. 非同期I/O (aiofiles)")
        print("  5. 列指向ストレージの採用")


if __name__ == "__main__":
    benchmark_io_methods()
