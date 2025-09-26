"""
System6 30分達成のための最適化版
インジケーター事前計算の完全活用 + バッチサイズ最適化
"""

import pandas as pd
import time
from typing import Dict, Optional, Callable, Any
from common.i18n import tr
from common.utils import is_today_run


def has_required_indicators_system6(df: pd.DataFrame) -> bool:
    """System6に必要な指標が事前計算済みかチェック"""
    required_cols = ["atr10", "dollarvolume50", "return_6d", "UpTwoDays", "filter", "setup"]
    return all(col in df.columns for col in required_cols)


def use_precomputed_indicators_system6(df: pd.DataFrame) -> pd.DataFrame:
    """事前計算済み指標をそのまま使用（再計算なし）"""
    # 必要な列のみを選択
    required_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "atr10",
        "dollarvolume50",
        "return_6d",
        "UpTwoDays",
        "filter",
        "setup",
    ]

    available_cols = [col for col in required_cols if col in df.columns]
    result_df = df[available_cols].copy()

    # 基本的なクリーニングのみ
    result_df = result_df.dropna(subset=["atr10", "dollarvolume50", "return_6d"])
    result_df = result_df.loc[~result_df.index.duplicated()].sort_index()
    result_df.index = pd.to_datetime(result_df.index).tz_localize(None)
    result_df.index.name = "Date"

    return result_df


def early_filter_system6(raw_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """System6用の早期フィルタリング - 明らかに条件を満たさない銘柄を除外"""
    filtered_dict = {}

    for symbol, df in raw_data_dict.items():
        try:
            # 基本的な条件チェック
            if len(df) < 50:
                continue

            # 最新の価格が5ドル以上かチェック
            latest_close = df["Close"].iloc[-1] if "Close" in df.columns else df["close"].iloc[-1]
            if latest_close < 5.0:
                continue

            # 最低限の出来高があるかチェック（概算）
            if "Volume" in df.columns:
                avg_volume = df["Volume"].tail(10).mean()
            elif "volume" in df.columns:
                avg_volume = df["volume"].tail(10).mean()
            else:
                continue

            if avg_volume * latest_close < 5_000_000:  # 5M dollar volume threshold
                continue

            filtered_dict[symbol] = df

        except Exception:
            continue  # エラーの場合はスキップ

    return filtered_dict


def prepare_data_optimized_system6(
    raw_data_dict: Dict[str, pd.DataFrame],
    *,
    progress_callback: Optional[Callable] = None,
    log_callback: Optional[Callable] = None,
    skip_callback: Optional[Callable] = None,
    reuse_indicators: bool = True,
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """
    System6用の30分達成最適化版データ準備

    最適化ポイント:
    1. 早期フィルタリングで処理対象を80%削減
    2. 事前計算済み指標の完全活用（再計算なし）
    3. 大きなバッチサイズで処理効率化
    """

    start_time = time.time()

    # Phase 1: 早期フィルタリング（1-2分で80%削減）
    if log_callback:
        log_callback("🔍 System6早期フィルタリング開始...")

    original_count = len(raw_data_dict)
    filtered_data = early_filter_system6(raw_data_dict)
    filtered_count = len(filtered_data)

    filter_time = time.time() - start_time
    if log_callback:
        reduction_pct = (1 - filtered_count / original_count) * 100 if original_count > 0 else 0
        log_callback(
            f"✅ 早期フィルタ完了: {original_count} → {filtered_count}銘柄 "
            f"({reduction_pct:.1f}%削減) | 経過: {filter_time:.1f}秒"
        )

    # Phase 2: インジケーター処理（事前計算済みを活用）
    result_dict = {}
    total = len(filtered_data)

    if total == 0:
        if log_callback:
            log_callback("⚠️ フィルタリング後の対象銘柄が0件")
        return result_dict

    # 大きなバッチサイズで効率化
    batch_size = min(200, max(50, total // 10))  # 適応的バッチサイズ

    phase2_start = time.time()
    processed = 0
    skipped = 0
    precomputed_used = 0
    recalculated = 0

    buffer = []

    for i, (symbol, df) in enumerate(filtered_data.items(), 1):
        try:
            if reuse_indicators and has_required_indicators_system6(df):
                # 事前計算済み指標をそのまま使用（高速）
                prepared_df = use_precomputed_indicators_system6(df)
                precomputed_used += 1
            else:
                # 再計算が必要（低速）
                from core.system6 import _compute_indicators_from_frame

                prepared_df = _compute_indicators_from_frame(df)
                recalculated += 1

            if not prepared_df.empty:
                result_dict[symbol] = prepared_df
                processed += 1
                buffer.append(symbol)
            else:
                skipped += 1
                if skip_callback:
                    skip_callback(symbol, "empty_after_processing")

        except Exception as e:
            skipped += 1
            if skip_callback:
                skip_callback(symbol, f"processing_error: {str(e)}")

        # バッチごとの進捗報告
        if (i % batch_size == 0 or i == total) and log_callback:
            elapsed = time.time() - phase2_start
            remain = (elapsed / i) * (total - i) if i > 0 else 0

            em, es = divmod(int(elapsed), 60)
            rm, rs = divmod(int(remain), 60)

            # サンプル銘柄表示
            sample = ", ".join(buffer[:10])
            if len(buffer) > 10:
                sample += f", ...(+{len(buffer) - 10} more)"

            msg = (
                f"📊 System6処理進捗: {i}/{total} | "
                f"経過: {em}m{es}s / 残り: ~{rm}m{rs}s\n"
                f"📈 事前計算活用: {precomputed_used}, 再計算: {recalculated}\n"
                f"銘柄: {sample}"
            )

            log_callback(msg)
            buffer.clear()

        if progress_callback:
            try:
                progress_callback(i, total)
            except Exception:
                pass

    # 最終サマリー
    total_time = time.time() - start_time
    tm, ts = divmod(int(total_time), 60)

    if log_callback:
        log_callback(
            f"🎯 System6最適化版完了!\n"
            f"⏱️ 総処理時間: {tm}m{ts}s\n"
            f"📊 処理結果: {processed}件成功, {skipped}件スキップ\n"
            f"🚀 事前計算活用: {precomputed_used}/{processed + skipped}件 "
            f"({precomputed_used/(processed + skipped)*100 if processed + skipped > 0 else 0:.1f}%)"
        )

    return result_dict
