"""
より激進的なPhase2最適化
- メモリマップドファイル読み込み
- データ型の事前最適化
- 不要列の事前除外
"""

import time
import os
import sys
from pathlib import Path

# パス設定 - tests/experimental/ から2階層上のルートへ
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.cache_manager import CacheManager
from config.settings import get_settings
import pandas as pd


def test_ultra_optimized_phase2():
    """超最適化Phase2テスト"""
    print("=== Ultra-Optimized Phase2 Test ===")

    settings = get_settings()
    cache_manager = CacheManager(settings)

    # テスト用シンボル
    rolling_dir = Path(settings.cache.rolling_dir)
    available_files = list(rolling_dir.glob("*.csv")) + list(rolling_dir.glob("*.parquet"))
    symbols = [f.stem for f in available_files[:50]]

    if not symbols:
        print("❌ テスト用シンボルが見つかりません")
        return

    print(f"📊 テスト対象: {len(symbols)}シンボル")

    # === 超最適化読み込み ===
    print("\n--- 超最適化読み込み ---")
    start_time = time.perf_counter()

    ultra_data = {}

    # 必要最小限の列のみを定義
    essential_columns = [
        "date",
        "Date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    # 型最適化マッピング
    dtype_map = {
        "Open": "float32",
        "High": "float32",
        "Low": "float32",
        "Close": "float32",
        "Volume": "float32",
        "open": "float32",
        "high": "float32",
        "low": "float32",
        "close": "float32",
        "volume": "float32",
    }

    for symbol in symbols:
        try:
            # rolling → fullの順でファイルパスを探索
            for profile in ["rolling", "full"]:
                base_dir = rolling_dir if profile == "rolling" else Path(settings.cache.full_dir)

                # 各ファイル形式を試行
                for ext in [".parquet", ".feather", ".csv"]:
                    file_path = base_dir / f"{symbol}{ext}"
                    if file_path.exists():
                        try:
                            if ext == ".parquet":
                                # Parquetは列指定で高速読み込み
                                df = pd.read_parquet(
                                    file_path, columns=None
                                )  # 全列読み込み後に選別
                                available_cols = [c for c in essential_columns if c in df.columns]
                                if available_cols:
                                    df = df[available_cols]

                            elif ext == ".feather":
                                df = pd.read_feather(file_path)
                                available_cols = [c for c in essential_columns if c in df.columns]
                                if available_cols:
                                    df = df[available_cols]

                            else:  # CSV
                                # CSVは事前に列をチェック
                                try:
                                    sample = pd.read_csv(file_path, nrows=0)  # ヘッダーのみ
                                    available_cols = [
                                        c for c in essential_columns if c in sample.columns
                                    ]
                                    use_cols = available_cols if available_cols else None

                                    df = pd.read_csv(
                                        file_path,
                                        usecols=use_cols,
                                        dtype={
                                            k: v
                                            for k, v in dtype_map.items()
                                            if k in (available_cols or [])
                                        },
                                        parse_dates=(
                                            ["Date"] if "Date" in (available_cols or []) else None
                                        ),
                                        low_memory=False,  # 型推論を無効化して高速化
                                    )
                                except Exception:
                                    df = pd.read_csv(file_path, dtype=dtype_map)
                                    available_cols = [
                                        c for c in essential_columns if c in df.columns
                                    ]
                                    if available_cols:
                                        df = df[available_cols]

                            # 基本的なクリーニング
                            if df is not None and not df.empty:
                                # 列名正規化
                                df.columns = [c.lower() for c in df.columns]

                                # 重複削除
                                if df.columns.has_duplicates:
                                    df = df.loc[:, ~df.columns.duplicated(keep="first")]

                                ultra_data[symbol] = df
                                break  # 成功したらBreak

                        except Exception:
                            continue

                if symbol in ultra_data:
                    break  # 成功したらプロファイル探索終了

        except Exception:
            continue

    ultra_time = time.perf_counter() - start_time
    print(f"⏱️  超最適化読み込み: {ultra_time:.3f}秒")
    print(f"📁 成功: {len(ultra_data)}/{len(symbols)}シンボル")

    # === 従来の最適化処理 ===
    print("\n--- 従来最適化（比較用） ---")
    start_time = time.perf_counter()

    cpu_count = os.cpu_count() or 4
    max_workers = min(max(2, cpu_count // 2), min(8, len(symbols)))

    optimized_data = cache_manager.read_batch_parallel(
        symbols=symbols, profile="rolling", max_workers=max_workers, fallback_profile="full"
    )

    optimized_time = time.perf_counter() - start_time
    print(f"⏱️  従来最適化: {optimized_time:.3f}秒")
    print(f"📁 成功: {len(optimized_data)}/{len(symbols)}シンボル")

    # === 結果比較 ===
    print("\n=== 超最適化結果 ===")

    if optimized_time > 0:
        speedup = optimized_time / ultra_time
        time_saved = optimized_time - ultra_time
        improvement_pct = (time_saved / optimized_time) * 100

        print(f"🚀 超最適化高速化: {speedup:.2f}倍")
        print(f"⏰ 追加時間短縮: {time_saved:.3f}秒 ({improvement_pct:.1f}%追加改善)")

        if ultra_time <= 0.5:
            print("🎯 目標達成！Phase2を0.5秒以下に短縮成功")
        else:
            print(f"⚠️  あと少し: {ultra_time:.3f}秒 (目標0.5秒)")

    # メモリ使用量の推測
    if ultra_data:
        sample_df = next(iter(ultra_data.values()))
        estimated_memory_mb = (len(ultra_data) * sample_df.memory_usage(deep=True).sum()) / (
            1024 * 1024
        )
        print(f"💾 推定メモリ使用量: {estimated_memory_mb:.1f}MB")


if __name__ == "__main__":
    test_ultra_optimized_phase2()
